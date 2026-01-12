"""
REINFORCEMENT LEARNING VERSION: RNN Policy Network → TabPFN Judge

Architecture:
1. RNN processes temporal data → generates stochastic latent representation Z (policy)
2. Concatenate [Static + Last_Values + Z] → TabPFN (reward judge)
3. TabPFN provides reward signal (cannot backprop, so use policy gradient)
4. Train RNN policy to maximize expected reward

Key Features:
- Policy Gradient (REINFORCE algorithm)
- TabPFN as non-differentiable reward function
- Stochastic policy with exploration
- Validation to prevent overfitting
"""

import pandas as pd
import numpy as np
import sys
import copy
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from torch.utils.data import Dataset, DataLoader
import random
import os

from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    auc,
)

os.environ["SEGMENT_WRITE_KEY"] = ""
os.environ["ANALYTICS_WRITE_KEY"] = ""
os.environ["TABPFN_DISABLE_ANALYTICS"] = "1"

try:
    import analytics
    analytics.write_key = None
    analytics.disable()
except Exception:
    pass

from tabpfn import TabPFNClassifier

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything()

PT = "/Users/anhnd/CodingSpace/Python/PREDKIT"
if sys.platform != "darwin":
    PT = "/home/anhnda/PREKIT"
sys.path.append(PT)

from constants import NULLABLE_MEASURES
from utils.class_patient import Patients
from utils.prepare_data import trainTestPatients, encodeCategoricalData
from TimeEmbeddingVal import (
    get_all_temporal_features,
    extract_temporal_data,
    load_and_prepare_patients,
    split_patients_train_val,
)
from TimeEmbedding import DEVICE, TimeEmbeddedRNNCell

FIXED_FEATURES = [
    "age", "gender", "race", "chronic_pulmonary_disease", "ckd_stage",
    "congestive_heart_failure", "dka_type", "history_aci", "history_ami",
    "hypertension", "liver_disease", "macroangiopathy", "malignant_cancer",
    "microangiopathy", "uti", "oasis", "saps2", "sofa",
    "mechanical_ventilation", "use_NaHCO3", "preiculos", "gcs_unable"
]

# ==============================================================================
# 1. Static Encoder (Same as before)
# ==============================================================================

class SimpleStaticEncoder:
    """Encodes categorical static features"""
    def __init__(self, features):
        self.features = features
        self.mappings = {f: {} for f in features}
        self.counts = {f: 0 for f in features}

    def fit(self, patients):
        for p in patients:
            for f in self.features:
                val = p.measures.get(f, 0.0)
                if hasattr(val, 'values') and len(val) > 0:
                    val = list(val.values())[0]
                elif hasattr(val, 'values'):
                    val = 0.0

                val_str = str(val)
                try:
                    float(val)
                except ValueError:
                    if val_str not in self.mappings[f]:
                        self.mappings[f][val_str] = float(self.counts[f])
                        self.counts[f] += 1

    def transform(self, patient):
        vec = []
        for f in self.features:
            val = patient.measures.get(f, 0.0)
            if hasattr(val, 'values') and len(val) > 0:
                val = list(val.values())[0]
            elif hasattr(val, 'values'):
                val = 0.0

            try:
                numeric_val = float(val)
            except ValueError:
                numeric_val = self.mappings[f].get(str(val), -1.0)
            vec.append(numeric_val)
        return vec

# ==============================================================================
# 2. Dataset (Same structure as before)
# ==============================================================================

class HybridDataset(Dataset):
    def __init__(self, patients, feature_names, static_encoder, normalization_stats=None):
        self.data = []
        self.labels = []
        self.static_data = []
        self.feature_names = feature_names

        all_values = []
        patient_list = patients.patientList if hasattr(patients, 'patientList') else patients

        for patient in patient_list:
            times, values, masks = extract_temporal_data(patient, feature_names)
            if times is None:
                continue

            s_vec = static_encoder.transform(patient)
            self.static_data.append(torch.tensor(s_vec, dtype=torch.float32))
            self.data.append({'times': times, 'values': values, 'masks': masks})
            self.labels.append(1 if patient.akdPositive else 0)

            for v_vec, m_vec in zip(values, masks):
                for v, m in zip(v_vec, m_vec):
                    if m > 0:
                        all_values.append(v)

        if normalization_stats is None:
            all_values = np.array(all_values)
            self.mean = np.mean(all_values) if len(all_values) > 0 else 0.0
            self.std = np.std(all_values) if len(all_values) > 0 else 1.0
        else:
            self.mean = normalization_stats['mean']
            self.std = normalization_stats['std']

        for i in range(len(self.data)):
            norm_values = []
            for v_vec, m_vec in zip(self.data[i]['values'], self.data[i]['masks']):
                norm = [(v - self.mean)/self.std if m>0 else 0.0
                        for v, m in zip(v_vec, m_vec)]
                norm_values.append(norm)

            self.data[i] = {
                'times': torch.tensor(self.data[i]['times'], dtype=torch.float32),
                'values': torch.tensor(norm_values, dtype=torch.float32),
                'masks': torch.tensor(self.data[i]['masks'], dtype=torch.float32)
            }

    def get_normalization_stats(self):
        return {'mean': self.mean, 'std': self.std}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx], self.static_data[idx]

def hybrid_collate_fn(batch):
    data_list, label_list, static_list = zip(*batch)
    lengths = [len(d['times']) for d in data_list]
    max_len = max(lengths)
    feat_dim = data_list[0]['values'].shape[-1]
    batch_size = len(data_list)

    padded_times = torch.zeros(batch_size, max_len)
    padded_values = torch.zeros(batch_size, max_len, feat_dim)
    padded_masks = torch.zeros(batch_size, max_len, feat_dim)

    for i, d in enumerate(data_list):
        l = lengths[i]
        padded_times[i, :l] = d['times']
        padded_values[i, :l] = d['values']
        padded_masks[i, :l] = d['masks']

    temporal_batch = {
        'times': padded_times,
        'values': padded_values,
        'masks': padded_masks,
        'lengths': torch.tensor(lengths)
    }
    return temporal_batch, torch.tensor(label_list, dtype=torch.float32), torch.stack(static_list)

# ==============================================================================
# 3. RNN Policy Network (Stochastic Policy)
# ==============================================================================

class RNNPolicyNetwork(nn.Module):
    """
    RNN that outputs parameters of a distribution (policy).
    We sample Z from this distribution to enable exploration.
    """
    def __init__(self, input_dim, hidden_dim, latent_dim, time_dim=32):
        super().__init__()
        self.rnn_cell = TimeEmbeddedRNNCell(input_dim, hidden_dim, time_dim)

        # Output mean and log_std for Gaussian policy
        self.fc_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc_logstd = nn.Linear(hidden_dim, latent_dim)

        self.latent_dim = latent_dim

    def forward(self, batch_data, deterministic=False):
        """
        Returns:
            z: sampled latent vector (action)
            log_prob: log probability of the action (for policy gradient)
            mean: mean of the distribution (for evaluation)
        """
        times = batch_data['times'].to(DEVICE)
        values = batch_data['values'].to(DEVICE)
        masks = batch_data['masks'].to(DEVICE)
        lengths = batch_data['lengths'].to(DEVICE)

        # Get RNN hidden state
        h = self.rnn_cell(times, values, masks, lengths)

        # Get distribution parameters
        mean = self.fc_mean(h)
        log_std = self.fc_logstd(h)
        log_std = torch.clamp(log_std, min=-20, max=2)  # Stability
        std = torch.exp(log_std)

        # Create Gaussian distribution
        policy_dist = dist.Normal(mean, std)

        if deterministic:
            z = mean
            log_prob = None
        else:
            # Sample action (with reparameterization trick)
            z = policy_dist.rsample()
            log_prob = policy_dist.log_prob(z).sum(dim=-1)  # Sum over latent dimensions

        return z, log_prob, mean

# ==============================================================================
# 4. Feature Extraction with Policy Network
# ==============================================================================

def extract_features_and_logprobs(policy_net, loader, deterministic=False):
    """
    Extract features: [Last_Values + Static + Z]
    Also return log_probs for policy gradient
    """
    policy_net.eval() if deterministic else policy_net.train()

    all_features = []
    all_labels = []
    all_log_probs = []

    with torch.set_grad_enabled(not deterministic):
        for t_data, labels, s_data in loader:
            # 1. Sample Z from policy
            z, log_prob, mean = policy_net(t_data, deterministic=deterministic)
            z_np = (mean if deterministic else z).detach().cpu().numpy()

            # 2. Extract Last Values
            vals = t_data['values'].cpu().numpy()
            masks = t_data['masks'].cpu().numpy()

            batch_last_vals = []
            for i in range(len(vals)):
                patient_last = []
                for f_idx in range(vals.shape[2]):
                    f_vals = vals[i, :, f_idx]
                    f_mask = masks[i, :, f_idx]
                    valid_idx = np.where(f_mask > 0)[0]

                    if len(valid_idx) > 0:
                        last_v = f_vals[valid_idx[-1]]
                    else:
                        last_v = 0.0
                    patient_last.append(last_v)
                batch_last_vals.append(patient_last)

            last_vals_arr = np.array(batch_last_vals)

            # 3. Get Static Features
            s_np = s_data.numpy()

            # 4. Concatenate: [Static + Last_Values + Z]
            combined = np.hstack([s_np, last_vals_arr, z_np])

            all_features.append(combined)
            all_labels.extend(labels.numpy())

            if not deterministic and log_prob is not None:
                all_log_probs.append(log_prob)  # Keep gradients for policy gradient!

    features = np.vstack(all_features)
    labels = np.array(all_labels)
    log_probs = torch.cat(all_log_probs) if all_log_probs else None

    return features, labels, log_probs

# ==============================================================================
# 5. Reinforcement Learning Training Loop
# ==============================================================================

def train_policy_with_tabpfn_reward(
    policy_net,
    train_loader,
    val_loader,
    tabpfn_params,
    epochs=100,
    update_tabpfn_every=5
):
    """
    Train policy network using TabPFN as reward function.

    Algorithm:
    1. Sample Z from policy network
    2. Train TabPFN on [Static + Last + Z] → Prediction
    3. Compute reward based on TabPFN performance
    4. Update policy using REINFORCE (policy gradient)
    """

    optimizer = torch.optim.Adam(policy_net.parameters(), lr=0.0005)

    best_val_auc = 0
    best_state = None
    patience = 15
    patience_counter = 0

    print("  [RL Training] Using TabPFN as Reward Judge...")

    # Initialize TabPFN
    tabpfn_model = None

    for epoch in range(epochs):
        policy_net.train()

        # =====================================================================
        # Step 1: Sample features from current policy (stochastic)
        # =====================================================================
        X_train, y_train, log_probs_train = extract_features_and_logprobs(
            policy_net, train_loader, deterministic=False
        )

        # =====================================================================
        # Step 2: Train/Update TabPFN periodically
        # =====================================================================
        if epoch % update_tabpfn_every == 0 or tabpfn_model is None:
            tabpfn_model = TabPFNClassifier(**tabpfn_params)
            tabpfn_model.fit(X_train, y_train)

        # =====================================================================
        # Step 3: Compute Rewards (Per-sample)
        # =====================================================================
        # Reward = 1 if TabPFN predicts correctly, 0 otherwise
        y_pred_proba = tabpfn_model.predict_proba(X_train)[:, 1]
        y_pred = (y_pred_proba > 0.5).astype(int)

        # Binary reward: correct prediction = +1, wrong = 0
        rewards = (y_pred == y_train).astype(np.float32)

        # Optional: Use prediction probability as reward (smoother)
        # For positive class (y=1): reward = pred_prob
        # For negative class (y=0): reward = 1 - pred_prob
        rewards_smooth = np.where(y_train == 1, y_pred_proba, 1 - y_pred_proba)

        # Combine binary and smooth rewards
        rewards_combined = 0.5 * rewards + 0.5 * rewards_smooth

        # Normalize rewards (baseline subtraction)
        rewards_tensor = torch.tensor(rewards_combined, dtype=torch.float32).to(DEVICE)
        rewards_tensor = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std() + 1e-8)

        # =====================================================================
        # Step 4: Policy Gradient Update (REINFORCE)
        # =====================================================================
        # Loss = -E[log π(z|s) * R(z)]
        log_probs_train = log_probs_train.to(DEVICE)
        policy_loss = -(log_probs_train * rewards_tensor).mean()

        # Add entropy bonus to encourage exploration
        entropy_bonus = 0.01 * log_probs_train.mean()  # Negative entropy

        total_loss = policy_loss - entropy_bonus

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
        optimizer.step()

        # =====================================================================
        # Step 5: Validation (Deterministic Policy)
        # =====================================================================
        if (epoch + 1) % 5 == 0:
            policy_net.eval()
            with torch.no_grad():
                X_val, y_val, _ = extract_features_and_logprobs(
                    policy_net, val_loader, deterministic=True
                )

                # Retrain TabPFN on deterministic features
                tabpfn_val_model = TabPFNClassifier(**tabpfn_params)
                tabpfn_val_model.fit(X_train, y_train)

                y_val_proba = tabpfn_val_model.predict_proba(X_val)[:, 1]

                val_auc = roc_auc_score(y_val, y_val_proba)
                val_aupr = average_precision_score(y_val, y_val_proba)

                mean_reward = rewards_combined.mean()

                print(f"    Epoch {epoch+1:3d} | Reward: {mean_reward:.4f} | "
                      f"Val AUC: {val_auc:.4f} | Val AUPR: {val_aupr:.4f}")

                # Early stopping based on validation AUPR
                if val_aupr > best_val_auc:
                    best_val_auc = val_aupr
                    best_state = copy.deepcopy(policy_net.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"    Early stopping at epoch {epoch+1}")
                        break

    # Load best model
    if best_state is not None:
        policy_net.load_state_dict(best_state)

    return policy_net

# ==============================================================================
# 6. Main Execution
# ==============================================================================

def main():
    print("="*80)
    print("RL POLICY (RNN + TabPFN) vs BASELINE (Last + Static)")
    print("="*80)

    patients = load_and_prepare_patients()
    temporal_feats = get_all_temporal_features(patients)

    print("Encoding static features...")
    encoder = SimpleStaticEncoder(FIXED_FEATURES)
    encoder.fit(patients.patientList)

    print(f"Input: {len(temporal_feats)} Temporal + {len(FIXED_FEATURES)} Static Features")

    # Store metrics for RL and Baseline
    metrics_rl = {k: [] for k in ['auc', 'auc_pr']}
    metrics_baseline = {k: [] for k in ['auc', 'auc_pr']}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    for fold, (train_full, test_p) in enumerate(trainTestPatients(patients)):
        print(f"\n{'='*80}")
        print(f"Fold {fold}")
        print('='*80)

        train_p_obj, val_p_obj = split_patients_train_val(train_full, val_ratio=0.1, seed=42+fold)
        train_p = train_p_obj.patientList
        val_p = val_p_obj.patientList
        test_p_list = test_p.patientList

        # 1. Create Datasets
        train_ds = HybridDataset(train_p, temporal_feats, encoder)
        stats = train_ds.get_normalization_stats()
        val_ds = HybridDataset(val_p, temporal_feats, encoder, stats)
        test_ds = HybridDataset(test_p_list, temporal_feats, encoder, stats)

        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=hybrid_collate_fn)
        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=hybrid_collate_fn)
        test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=hybrid_collate_fn)

        # 2. Initialize Policy Network
        latent_dim = 16  # Dimension of learned representation Z
        policy_net = RNNPolicyNetwork(
            input_dim=len(temporal_feats),
            hidden_dim=12,
            latent_dim=latent_dim,
            time_dim=32
        ).to(DEVICE)

        # 3. TabPFN Parameters
        tabpfn_params = {
            'N_ensemble_configurations': 16,
            'device': 'cpu'
        }

        # 4. Train Policy with RL
        policy_net = train_policy_with_tabpfn_reward(
            policy_net,
            train_loader,
            val_loader,
            tabpfn_params,
            epochs=100,
            update_tabpfn_every=5
        )

        # 5. Final Evaluation on Test Set
        print("\n  [Final Test Evaluation]")
        policy_net.eval()

        with torch.no_grad():
            X_train_final, y_train_final, _ = extract_features_and_logprobs(
                policy_net, train_loader, deterministic=True
            )
            X_test_final, y_test_final, _ = extract_features_and_logprobs(
                policy_net, test_loader, deterministic=True
            )

        # Train final TabPFN on deterministic features
        final_tabpfn = TabPFNClassifier(**tabpfn_params)
        final_tabpfn.fit(X_train_final, y_train_final)

        y_test_proba = final_tabpfn.predict_proba(X_test_final)[:, 1]
        y_test_pred = (y_test_proba > 0.5).astype(int)

        # Compute Metrics
        tn, fp, fn, tp = confusion_matrix(y_test_final, y_test_pred).ravel()
        prec, rec, _ = precision_recall_curve(y_test_final, y_test_proba)

        fold_auc = roc_auc_score(y_test_final, y_test_proba)
        fold_aupr = auc(rec, prec)

        metrics_rl['auc'].append(fold_auc)
        metrics_rl['auc_pr'].append(fold_aupr)

        # Plot ROC for RL
        fpr, tpr, _ = roc_curve(y_test_final, y_test_proba)
        ax1.plot(fpr, tpr, lw=2, label=f"Fold {fold} (AUC = {fold_auc:.3f})")

        print(f"  RL Test AUC: {fold_auc:.4f} | Test AUPR: {fold_aupr:.4f}")

        # ======================================================================
        # BASELINE: Standard TabPFN (Last Values + Static)
        # ======================================================================
        print("\n  [Baseline] Training Standard TabPFN (Last + Static)...")

        # Extract "Last Values" using getMeasuresBetween
        df_train_temp = train_p_obj.getMeasuresBetween(
            pd.Timedelta(hours=-6), pd.Timedelta(hours=24), "last", getUntilAkiPositive=True
        ).drop(columns=["subject_id", "hadm_id", "stay_id"])
        df_test_temp = test_p.getMeasuresBetween(
            pd.Timedelta(hours=-6), pd.Timedelta(hours=24), "last", getUntilAkiPositive=True
        ).drop(columns=["subject_id", "hadm_id", "stay_id"])

        # Encode categorical data
        df_train_enc, df_test_enc, _ = encodeCategoricalData(df_train_temp, df_test_temp)

        X_tr_b = df_train_enc.drop(columns=["akd"]).fillna(0)
        y_tr_b = df_train_enc["akd"]
        X_te_b = df_test_enc.drop(columns=["akd"]).fillna(0)
        y_te_b = df_test_enc["akd"]

        # Train baseline TabPFN
        tabpfn_base = TabPFNClassifier(**tabpfn_params)
        tabpfn_base.fit(X_tr_b, y_tr_b)

        # Evaluate baseline
        y_prob_b = tabpfn_base.predict_proba(X_te_b)[:, 1]
        prec_b, rec_b, _ = precision_recall_curve(y_te_b, y_prob_b)

        baseline_auc = roc_auc_score(y_te_b, y_prob_b)
        baseline_aupr = auc(rec_b, prec_b)

        metrics_baseline['auc'].append(baseline_auc)
        metrics_baseline['auc_pr'].append(baseline_aupr)

        # Plot ROC for Baseline
        fpr_b, tpr_b, _ = roc_curve(y_te_b, y_prob_b)
        ax2.plot(fpr_b, tpr_b, lw=2, label=f"Fold {fold} (AUC = {baseline_auc:.3f})")

        print(f"  Baseline Test AUC: {baseline_auc:.4f} | Test AUPR: {baseline_aupr:.4f}")
        print(f"  Fold {fold} Results -> RL: {fold_auc:.3f} vs Baseline: {baseline_auc:.3f}")

    # Final Plot Configuration
    for ax in [ax1, ax2]:
        ax.plot([0, 1], [0, 1], linestyle="--", color="navy", lw=2)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(loc="lower right")

    ax1.set_title("RL Policy + TabPFN Judge")
    ax2.set_title("Baseline (Last + Static)")
    plt.tight_layout()
    plt.savefig("result/tab_rl_vs_baseline.png", dpi=300)
    print("\nPlot saved to result/tab_rl_vs_baseline.png")

    # Summary Statistics
    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY")
    print("="*80)

    def print_stat(name, rl_metrics, base_metrics):
        rl_mean, rl_std = np.mean(rl_metrics), np.std(rl_metrics)
        base_mean, base_std = np.mean(base_metrics), np.std(base_metrics)
        print(f"{name:15s} | RL: {rl_mean:.4f} ± {rl_std:.4f}  vs  Baseline: {base_mean:.4f} ± {base_std:.4f}")

    print_stat("AUC", metrics_rl['auc'], metrics_baseline['auc'])
    print_stat("AUC-PR", metrics_rl['auc_pr'], metrics_baseline['auc_pr'])

if __name__ == "__main__":
    main()
