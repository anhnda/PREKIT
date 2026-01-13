"""
Tab_Xv6: Extended Handcrafted + RL-Optimized Learned Features

PROVEN RESULTS:
- Tab_Xv5 (Extended handcrafted only): +1.63% AUPR ✓
- Tab_RLv4 (Extended handcrafted + RL learned): +3.11% AUPR ✓

STRATEGY:
1. Use extended handcrafted features [Static + 7 stats per temporal feature]
   → Proven to work (+1.63% AUPR)

2. Add RL-optimized learned latent features
   → Use TabPFN DIRECTLY during training (no neural head mismatch)
   → Should add ~1.5% more AUPR on top

3. Simpler than Tab_RLv4:
   - Skip lengthy pretraining (50 epochs)
   - Direct RL training with extended handcrafted base
   - Faster convergence since handcrafted features already strong

Expected: +2.5% to +3.5% AUPR improvement over baseline
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

xseed = 42
seed_everything(xseed)

PT = "/Users/anhnd/CodingSpace/Python/PREDKIT"
if sys.platform != "darwin":
    PT = "/home/anhnda/PREKIT"
sys.path.append(PT)

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
# 1. Static Encoder (from Tab_RLv4)
# ==============================================================================

class SimpleStaticEncoder:
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
# 2. Dataset
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
# 3. RNN Policy Network (from Tab_RLv4, but simpler)
# ==============================================================================

class RNNPolicyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, time_dim=32):
        super().__init__()
        self.rnn_cell = TimeEmbeddedRNNCell(input_dim, hidden_dim, time_dim)
        self.fc_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc_logstd = nn.Linear(hidden_dim, latent_dim)
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

    def forward(self, batch_data, deterministic=False, temperature=1.0):
        times = batch_data['times'].to(DEVICE)
        values = batch_data['values'].to(DEVICE)
        masks = batch_data['masks'].to(DEVICE)
        lengths = batch_data['lengths'].to(DEVICE)

        h = self.rnn_cell(times, values, masks, lengths)
        mean = self.fc_mean(h)
        log_std = self.fc_logstd(h)
        log_std = torch.clamp(log_std, min=-20, max=2)
        std = torch.exp(log_std) * temperature

        policy_dist = dist.Normal(mean, std)

        if deterministic:
            z = mean
            log_prob = None
        else:
            z = policy_dist.rsample()
            log_prob = policy_dist.log_prob(z).sum(dim=-1)

        return z, log_prob, mean

# ==============================================================================
# 4. Extended Feature Extraction (SAME as Tab_RLv4)
# ==============================================================================

def extract_extended_features_and_logprobs(policy_net, loader, deterministic=False, temperature=1.0):
    """
    Extract: [Static + 7 stats per temporal feature + Learned Latent]
    """
    policy_net.eval() if deterministic else policy_net.train()

    all_features = []
    all_labels = []
    all_log_probs = []

    with torch.set_grad_enabled(not deterministic):
        for t_data, labels, s_data in loader:
            z, log_prob, mean = policy_net(t_data, deterministic=deterministic, temperature=temperature)
            z_np = (mean if deterministic else z).detach().cpu().numpy()

            vals = t_data['values'].cpu().numpy()
            masks = t_data['masks'].cpu().numpy()
            times = t_data['times'].cpu().numpy()

            batch_last_vals = []
            batch_mean_vals = []
            batch_std_vals = []
            batch_min_vals = []
            batch_max_vals = []
            batch_slope_vals = []
            batch_count_vals = []

            for i in range(len(vals)):
                patient_last = []
                patient_mean = []
                patient_std = []
                patient_min = []
                patient_max = []
                patient_slope = []
                patient_count = []

                for f_idx in range(vals.shape[2]):
                    f_vals = vals[i, :, f_idx]
                    f_mask = masks[i, :, f_idx]
                    f_times = times[i, :]
                    valid_idx = np.where(f_mask > 0)[0]

                    if len(valid_idx) > 0:
                        valid_vals = f_vals[valid_idx]
                        valid_times = f_times[valid_idx]

                        patient_last.append(valid_vals[-1])
                        patient_mean.append(np.mean(valid_vals))
                        patient_std.append(np.std(valid_vals) if len(valid_vals) > 1 else 0.0)
                        patient_min.append(np.min(valid_vals))
                        patient_max.append(np.max(valid_vals))
                        patient_count.append(len(valid_vals))

                        if len(valid_vals) >= 2:
                            slope = (valid_vals[-1] - valid_vals[0]) / (valid_times[-1] - valid_times[0] + 1e-6)
                        else:
                            slope = 0.0
                        patient_slope.append(slope)
                    else:
                        patient_last.append(0.0)
                        patient_mean.append(0.0)
                        patient_std.append(0.0)
                        patient_min.append(0.0)
                        patient_max.append(0.0)
                        patient_slope.append(0.0)
                        patient_count.append(0.0)

                batch_last_vals.append(patient_last)
                batch_mean_vals.append(patient_mean)
                batch_std_vals.append(patient_std)
                batch_min_vals.append(patient_min)
                batch_max_vals.append(patient_max)
                batch_slope_vals.append(patient_slope)
                batch_count_vals.append(patient_count)

            last_vals_arr = np.array(batch_last_vals)
            mean_vals_arr = np.array(batch_mean_vals)
            std_vals_arr = np.array(batch_std_vals)
            min_vals_arr = np.array(batch_min_vals)
            max_vals_arr = np.array(batch_max_vals)
            slope_vals_arr = np.array(batch_slope_vals)
            count_vals_arr = np.array(batch_count_vals)
            s_np = s_data.numpy()

            # Extended features + learned latent
            combined = np.hstack([
                s_np, last_vals_arr, mean_vals_arr, std_vals_arr,
                min_vals_arr, max_vals_arr, slope_vals_arr, count_vals_arr, z_np
            ])

            all_features.append(combined)
            all_labels.extend(labels.numpy())

            if not deterministic and log_prob is not None:
                all_log_probs.append(log_prob)

    features = np.vstack(all_features)
    labels = np.array(all_labels)
    log_probs = torch.cat(all_log_probs) if all_log_probs else None

    return features, labels, log_probs

# ==============================================================================
# 5. Simplified RL Training (No Pretraining)
# ==============================================================================

def train_policy_rl(
    policy_net,
    train_loader,
    val_loader,
    tabpfn_params,
    epochs=80,
    update_tabpfn_every=5
):
    """
    Simplified RL training:
    - No pretraining (extended handcrafted features already strong)
    - Direct RL optimization with TabPFN rewards
    """
    print("  [RL Training] Optimizing learned features for TabPFN directly...")

    optimizer = torch.optim.Adam(policy_net.parameters(), lr=0.0001)

    best_val_aupr = 0
    best_state = None
    patience = 20
    patience_counter = 0

    tabpfn_model = None

    for epoch in range(epochs):
        # Temperature annealing
        temperature = max(0.3, 0.8 - epoch / (epochs * 0.7))

        policy_net.train()

        # Extract features with stochastic sampling
        X_train, y_train, log_probs_train = extract_extended_features_and_logprobs(
            policy_net, train_loader, deterministic=False, temperature=temperature
        )

        # Update TabPFN periodically
        if epoch % update_tabpfn_every == 0 or tabpfn_model is None:
            tabpfn_model = TabPFNClassifier(**tabpfn_params)
            tabpfn_model.fit(X_train, y_train)

        # Compute rewards using validation set
        X_val_stoch, y_val, _ = extract_extended_features_and_logprobs(
            policy_net, val_loader, deterministic=False, temperature=temperature
        )

        y_val_proba = tabpfn_model.predict_proba(X_val_stoch)[:, 1]
        val_aupr = average_precision_score(y_val, y_val_proba)

        # Training rewards
        y_train_proba = tabpfn_model.predict_proba(X_train)[:, 1]
        rewards_smooth = np.where(y_train == 1, y_train_proba, 1 - y_train_proba)
        rewards_combined = rewards_smooth + 0.5 * val_aupr

        rewards_tensor = torch.tensor(rewards_combined, dtype=torch.float32).to(DEVICE)
        rewards_tensor = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std() + 1e-8)

        # Policy gradient
        log_probs_train = log_probs_train.to(DEVICE)
        policy_loss = -(log_probs_train * rewards_tensor).mean()
        entropy_bonus = 0.001 * log_probs_train.mean()
        total_loss = policy_loss - entropy_bonus

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=0.5)
        optimizer.step()

        # Validation every 5 epochs
        if (epoch + 1) % 5 == 0:
            policy_net.eval()
            with torch.no_grad():
                X_val_det, y_val_det, _ = extract_extended_features_and_logprobs(
                    policy_net, val_loader, deterministic=True
                )

                X_train_det, y_train_det, _ = extract_extended_features_and_logprobs(
                    policy_net, train_loader, deterministic=True
                )

                tabpfn_val_model = TabPFNClassifier(**tabpfn_params)
                tabpfn_val_model.fit(X_train_det, y_train_det)

                y_val_proba_det = tabpfn_val_model.predict_proba(X_val_det)[:, 1]
                val_auc = roc_auc_score(y_val_det, y_val_proba_det)
                val_aupr_det = average_precision_score(y_val_det, y_val_proba_det)

                print(f"    Epoch {epoch+1:3d} | Temp: {temperature:.3f} | "
                      f"Val AUC: {val_auc:.4f} | Val AUPR: {val_aupr_det:.4f}")

                if val_aupr_det > best_val_aupr:
                    best_val_aupr = val_aupr_det
                    best_state = copy.deepcopy(policy_net.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"    Early stopping at epoch {epoch+1}")
                        break

    if best_state is not None:
        policy_net.load_state_dict(best_state)

    print(f"  [RL Training] Completed. Best Val AUPR: {best_val_aupr:.4f}")
    return policy_net

# ==============================================================================
# 6. Main
# ==============================================================================

def main():
    print("="*80)
    print("Tab_Xv6: Extended Handcrafted + RL-Optimized Learned Features")
    print("="*80)
    print("\nProven Strategy:")
    print("  1. Extended handcrafted (7 stats/feature) → +1.63% AUPR ✓")
    print("  2. RL-optimized learned latent on top → +1.5% more AUPR (expected)")
    print("  3. No pretraining needed (handcrafted base already strong)")
    print("="*80)

    patients = load_and_prepare_patients()
    temporal_feats = get_all_temporal_features(patients)

    encoder = SimpleStaticEncoder(FIXED_FEATURES)
    encoder.fit(patients.patientList)

    print(f"\nStatic features: {len(FIXED_FEATURES)}")
    print(f"Temporal features: {len(temporal_feats)}")
    print(f"Extended handcrafted: {len(temporal_feats)} x 7 = {len(temporal_feats)*7}")
    print(f"Learned latent: 32")
    print(f"Total: {len(FIXED_FEATURES) + len(temporal_feats)*7 + 32}")

    metrics_xv6 = {k: [] for k in ['auc', 'auc_pr']}
    metrics_baseline = {k: [] for k in ['auc', 'auc_pr']}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    for fold, (train_full, test_p) in enumerate(trainTestPatients(patients, seed=xseed)):
        print(f"\n{'='*80}")
        print(f"Fold {fold}")
        print('='*80)

        train_p_obj, val_p_obj = split_patients_train_val(train_full, val_ratio=0.1, seed=42+fold)
        train_p = train_p_obj.patientList
        val_p = val_p_obj.patientList
        test_p_list = test_p.patientList

        train_ds = HybridDataset(train_p, temporal_feats, encoder)
        stats = train_ds.get_normalization_stats()
        val_ds = HybridDataset(val_p, temporal_feats, encoder, stats)
        test_ds = HybridDataset(test_p_list, temporal_feats, encoder, stats)

        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=hybrid_collate_fn)
        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=hybrid_collate_fn)
        test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=hybrid_collate_fn)

        # Policy network
        latent_dim = 32
        policy_net = RNNPolicyNetwork(
            input_dim=len(temporal_feats),
            hidden_dim=24,
            latent_dim=latent_dim,
            time_dim=32
        ).to(DEVICE)

        tabpfn_params = {'device': 'cuda' if torch.cuda.is_available() else 'cpu'}

        # RL training (no pretraining)
        policy_net = train_policy_rl(
            policy_net,
            train_loader,
            val_loader,
            tabpfn_params,
            epochs=80,
            update_tabpfn_every=5
        )

        # Test evaluation
        print("\n  [Test Evaluation]")
        policy_net.eval()

        with torch.no_grad():
            X_train_final, y_train_final, _ = extract_extended_features_and_logprobs(
                policy_net, train_loader, deterministic=True
            )
            X_test_final, y_test_final, _ = extract_extended_features_and_logprobs(
                policy_net, test_loader, deterministic=True
            )

        print(f"  Test features shape: {X_test_final.shape}")

        final_tabpfn = TabPFNClassifier(**tabpfn_params)
        final_tabpfn.fit(X_train_final, y_train_final)

        y_test_proba = final_tabpfn.predict_proba(X_test_final)[:, 1]
        prec, rec, _ = precision_recall_curve(y_test_final, y_test_proba)

        fold_auc = roc_auc_score(y_test_final, y_test_proba)
        fold_aupr = auc(rec, prec)

        metrics_xv6['auc'].append(fold_auc)
        metrics_xv6['auc_pr'].append(fold_aupr)

        fpr, tpr, _ = roc_curve(y_test_final, y_test_proba)
        ax1.plot(fpr, tpr, lw=2, label=f"Fold {fold} (AUC = {fold_auc:.3f})")

        print(f"  Tab_Xv6 AUC: {fold_auc:.4f} | AUPR: {fold_aupr:.4f}")

        # Baseline
        df_train_temp = train_p_obj.getMeasuresBetween(
            pd.Timedelta(hours=-6), pd.Timedelta(hours=24), "last", getUntilAkiPositive=True
        ).drop(columns=["subject_id", "hadm_id", "stay_id"])
        df_test_temp = test_p.getMeasuresBetween(
            pd.Timedelta(hours=-6), pd.Timedelta(hours=24), "last", getUntilAkiPositive=True
        ).drop(columns=["subject_id", "hadm_id", "stay_id"])

        df_train_enc, df_test_enc, _ = encodeCategoricalData(df_train_temp, df_test_temp)

        X_tr_b = df_train_enc.drop(columns=["akd"]).fillna(0)
        y_tr_b = df_train_enc["akd"]
        X_te_b = df_test_enc.drop(columns=["akd"]).fillna(0)
        y_te_b = df_test_enc["akd"]

        tabpfn_base = TabPFNClassifier(**tabpfn_params)
        tabpfn_base.fit(X_tr_b, y_tr_b)

        y_prob_b = tabpfn_base.predict_proba(X_te_b)[:, 1]
        prec_b, rec_b, _ = precision_recall_curve(y_te_b, y_prob_b)

        baseline_auc = roc_auc_score(y_te_b, y_prob_b)
        baseline_aupr = auc(rec_b, prec_b)

        metrics_baseline['auc'].append(baseline_auc)
        metrics_baseline['auc_pr'].append(baseline_aupr)

        fpr_b, tpr_b, _ = roc_curve(y_te_b, y_prob_b)
        ax2.plot(fpr_b, tpr_b, lw=2, label=f"Fold {fold} (AUC = {baseline_auc:.3f})")

        print(f"  Baseline AUC: {baseline_auc:.4f} | AUPR: {baseline_aupr:.4f}")

        improvement = ((fold_aupr - baseline_aupr) / baseline_aupr) * 100
        symbol = "✓" if fold_aupr > baseline_aupr else "✗"
        print(f"  Improvement: {improvement:+.2f}% {symbol}")

    # Plot
    for ax in [ax1, ax2]:
        ax.plot([0, 1], [0, 1], linestyle="--", color="navy", lw=2)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(loc="lower right")

    ax1.set_title("Tab_Xv6: Extended Features + RL Learned")
    ax2.set_title("Baseline (Last Only)")
    plt.tight_layout()
    plt.savefig("result/tab_xv6_vs_baseline.png", dpi=300)
    print("\nPlot saved to result/tab_xv6_vs_baseline.png")

    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)

    def print_stat(name, xv6_metrics, base_metrics):
        xv6_mean, xv6_std = np.mean(xv6_metrics), np.std(xv6_metrics)
        base_mean, base_std = np.mean(base_metrics), np.std(base_metrics)
        improvement = ((xv6_mean - base_mean) / base_mean) * 100
        symbol = "✓" if xv6_mean > base_mean else "✗"
        print(f"{name} | Tab_Xv6: {xv6_mean:.4f}±{xv6_std:.4f} vs Baseline: {base_mean:.4f}±{base_std:.4f} ({improvement:+.2f}%) {symbol}")

    print_stat("AUC   ", metrics_xv6['auc'], metrics_baseline['auc'])
    print_stat("AUC-PR", metrics_xv6['auc_pr'], metrics_baseline['auc_pr'])

    print("\n" + "="*80)
    print("Tab_Xv6 = Best of Both Worlds:")
    print("✓ Extended handcrafted features (proven: +1.63% AUPR)")
    print("✓ RL-optimized learned features (optimized for TabPFN)")
    print("✓ Simpler than Tab_RLv4 (no pretraining)")
    print("="*80)

if __name__ == "__main__":
    main()
