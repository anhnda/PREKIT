"""
REINFORCEMENT LEARNING V6: Two-Component Z Architecture

Problem in V5B:
- Single Z (64-96 dims) must compress handcrafted (105 dims) AND learn novel patterns
- Information bottleneck causes performance degradation

Solution in V6:
1. Z_preserved (100 dims): Explicitly preserves handcrafted features
   - Strong reconstruction loss (weight = 2.0)
   - Ensures handcrafted information is maintained

2. Z_novel (28 dims): Learns additional patterns beyond handcrafted
   - Optimized via RL rewards
   - Captures patterns not in simple statistics

3. Separate training objectives:
   - Z_preserved: Reconstruction-focused (pretraining)
   - Z_novel: Reward-focused (RL training)

Architecture:
  RNN(temporal + handcrafted + static) → [Z_preserved, Z_novel]
                                           ↓           ↓
  Reconstruct(Z_preserved) → handcrafted (strong loss)
  TabPFN([Static + Last + Z_preserved + Z_novel]) → prediction
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

xseed = 42
seed_everything(xseed)

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
# 1. Dataset
# ==============================================================================

class HybridDatasetV6(Dataset):
    def __init__(self, patients, feature_names, static_features_df, normalization_stats=None):
        """
        Args:
            patients: Patients object or list of patients
            feature_names: List of temporal feature names
            static_features_df: DataFrame with properly encoded static features
            normalization_stats: Pre-computed mean/std for temporal features
        """
        self.data = []
        self.labels = []
        self.static_data = []
        self.handcrafted_data = []
        self.last_values_data = []
        self.feature_names = feature_names

        all_values = []
        patient_list = patients.patientList if hasattr(patients, 'patientList') else patients

        for idx, patient in enumerate(patient_list):
            times, values, masks = extract_temporal_data(patient, feature_names)
            if times is None:
                continue

            # Use properly encoded static features from dataframe
            s_vec = static_features_df.iloc[idx].values
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

            handcrafted = self._compute_handcrafted_features(
                norm_values, self.data[i]['masks'], self.data[i]['times']
            )
            # Replace NaN/Inf with 0
            handcrafted = np.nan_to_num(handcrafted, nan=0.0, posinf=0.0, neginf=0.0)
            self.handcrafted_data.append(torch.tensor(handcrafted, dtype=torch.float32))

            last_vals = self._extract_last_values(norm_values, self.data[i]['masks'])
            self.last_values_data.append(torch.tensor(last_vals, dtype=torch.float32))

            self.data[i] = {
                'times': torch.tensor(self.data[i]['times'], dtype=torch.float32),
                'values': torch.tensor(norm_values, dtype=torch.float32),
                'masks': torch.tensor(self.data[i]['masks'], dtype=torch.float32)
            }

    def _compute_handcrafted_features(self, norm_values, masks, times):
        handcrafted = []
        for f_idx in range(len(norm_values[0])):
            f_vals = np.array([norm_values[t][f_idx] for t in range(len(norm_values))])
            f_mask = np.array([masks[t][f_idx] for t in range(len(masks))])
            f_times = np.array(times)
            valid_idx = np.where(f_mask > 0)[0]

            if len(valid_idx) > 0:
                valid_vals = f_vals[valid_idx]
                valid_times = f_times[valid_idx]

                last = valid_vals[-1]
                mean_val = np.mean(valid_vals)
                std_val = np.std(valid_vals) if len(valid_vals) > 1 else 0.0
                min_val = np.min(valid_vals)
                max_val = np.max(valid_vals)
                count = len(valid_vals)
                slope = (valid_vals[-1] - valid_vals[0]) / (valid_times[-1] - valid_times[0] + 1e-6) if len(valid_vals) >= 2 else 0.0
                handcrafted.extend([last, mean_val, std_val, min_val, max_val, slope, count])
            else:
                handcrafted.extend([0.0] * 7)
        return handcrafted

    def _extract_last_values(self, norm_values, masks):
        last_vals = []
        for f_idx in range(len(norm_values[0])):
            f_vals = np.array([norm_values[t][f_idx] for t in range(len(norm_values))])
            f_mask = np.array([masks[t][f_idx] for t in range(len(masks))])
            valid_idx = np.where(f_mask > 0)[0]
            if len(valid_idx) > 0:
                last_vals.append(f_vals[valid_idx[-1]])
            else:
                last_vals.append(0.0)
        return last_vals

    def get_normalization_stats(self):
        return {'mean': self.mean, 'std': self.std}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (self.data[idx], self.labels[idx], self.static_data[idx],
                self.handcrafted_data[idx], self.last_values_data[idx])

def hybrid_collate_fn_v6(batch):
    data_list, label_list, static_list, handcrafted_list, last_list = zip(*batch)
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
    return (temporal_batch, torch.tensor(label_list, dtype=torch.float32),
            torch.stack(static_list), torch.stack(handcrafted_list), torch.stack(last_list))

# ==============================================================================
# 2. RNN Policy Network with Two-Component Z
# ==============================================================================

class RNNPolicyNetworkTwoComponent(nn.Module):
    """
    V6: Two-component Z architecture
    - Z_preserved: Large dimension to preserve handcrafted features
    - Z_novel: Smaller dimension to learn new patterns
    """
    def __init__(self, input_dim, hidden_dim, z_preserved_dim, z_novel_dim,
                 auxiliary_dim, handcrafted_dim, time_dim=32):
        super().__init__()
        self.rnn_cell = TimeEmbeddedRNNCell(input_dim, hidden_dim, time_dim)

        combined_dim = hidden_dim + auxiliary_dim

        # Z_preserved: For handcrafted feature compression
        self.fc_preserved_mean = nn.Linear(combined_dim, z_preserved_dim)
        self.fc_preserved_logstd = nn.Linear(combined_dim, z_preserved_dim)

        # Z_novel: For learning new patterns
        self.fc_novel_mean = nn.Linear(combined_dim, z_novel_dim)
        self.fc_novel_logstd = nn.Linear(combined_dim, z_novel_dim)

        # Reconstruction head: Z_preserved → handcrafted features
        self.reconstruction_head = nn.Sequential(
            nn.Linear(z_preserved_dim, z_preserved_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(z_preserved_dim, handcrafted_dim)
        )

        self.z_preserved_dim = z_preserved_dim
        self.z_novel_dim = z_novel_dim
        self.hidden_dim = hidden_dim
        self.handcrafted_dim = handcrafted_dim

    def forward(self, batch_data, auxiliary_features, deterministic=False, temperature=1.0):
        times = batch_data['times'].to(DEVICE)
        values = batch_data['values'].to(DEVICE)
        masks = batch_data['masks'].to(DEVICE)
        lengths = batch_data['lengths'].to(DEVICE)

        h = self.rnn_cell(times, values, masks, lengths)
        auxiliary = auxiliary_features.to(DEVICE)
        combined = torch.cat([h, auxiliary], dim=1)

        # Generate Z_preserved
        preserved_mean = self.fc_preserved_mean(combined)
        preserved_log_std = self.fc_preserved_logstd(combined)
        preserved_log_std = torch.clamp(preserved_log_std, min=-20, max=2)
        preserved_std = torch.exp(preserved_log_std) * temperature

        # Generate Z_novel
        novel_mean = self.fc_novel_mean(combined)
        novel_log_std = self.fc_novel_logstd(combined)
        novel_log_std = torch.clamp(novel_log_std, min=-20, max=2)
        novel_std = torch.exp(novel_log_std) * temperature

        if deterministic:
            z_preserved = preserved_mean
            z_novel = novel_mean
            log_prob = None
        else:
            preserved_dist = dist.Normal(preserved_mean, preserved_std)
            novel_dist = dist.Normal(novel_mean, novel_std)

            z_preserved = preserved_dist.rsample()
            z_novel = novel_dist.rsample()

            # Combined log probability
            log_prob_preserved = preserved_dist.log_prob(z_preserved).sum(dim=-1)
            log_prob_novel = novel_dist.log_prob(z_novel).sum(dim=-1)
            log_prob = log_prob_preserved + log_prob_novel

        # Concatenate both components
        z_combined = torch.cat([z_preserved, z_novel], dim=1)

        return z_combined, z_preserved, z_novel, log_prob, preserved_mean, novel_mean

    def reconstruct_handcrafted(self, z_preserved):
        """Reconstruct handcrafted features from Z_preserved only"""
        return self.reconstruction_head(z_preserved)

class SupervisedHead(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        return torch.sigmoid(self.fc3(x))

# ==============================================================================
# 3. Enhanced Pretraining with Strong Reconstruction Focus
# ==============================================================================

def pretrain_rnn_two_component(policy_net, train_loader, val_loader, static_dim, epochs=50):
    """
    Pretraining with emphasis on Z_preserved reconstruction
    - Z_preserved: Strong reconstruction loss (weight = 2.0)
    - Z_novel: Classification loss
    """
    print("  [Two-Component Pretraining] Strong reconstruction for Z_preserved...")

    supervised_head = SupervisedHead(
        policy_net.z_preserved_dim + policy_net.z_novel_dim + static_dim
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        list(policy_net.parameters()) + list(supervised_head.parameters()),
        lr=0.001
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    criterion_cls = nn.BCELoss()
    criterion_recon = nn.MSELoss()

    best_auc = 0
    best_state = None
    patience = 12
    counter = 0

    for epoch in range(epochs):
        policy_net.train()
        supervised_head.train()

        for t_data, labels, s_data, handcrafted, last_vals in train_loader:
            labels = labels.to(DEVICE)
            s_data = s_data.to(DEVICE)
            handcrafted = handcrafted.to(DEVICE)

            auxiliary = torch.cat([handcrafted, s_data], dim=1)

            z_combined, z_preserved, z_novel, _, _, _ = policy_net(
                t_data, auxiliary, deterministic=True
            )

            # Classification loss (using both components)
            combined = torch.cat([z_combined, s_data], dim=1)
            preds = supervised_head(combined).squeeze(-1)
            cls_loss = criterion_cls(preds, labels)

            # Reconstruction loss: Z_preserved should strongly preserve handcrafted
            recon_handcrafted = policy_net.reconstruct_handcrafted(z_preserved)
            recon_loss = criterion_recon(recon_handcrafted, handcrafted)

            # Combined loss (STRONG reconstruction weight = 2.0)
            total_loss = cls_loss + 2.0 * recon_loss

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(policy_net.parameters()) + list(supervised_head.parameters()),
                max_norm=1.0
            )
            optimizer.step()

        if (epoch + 1) % 3 == 0:
            policy_net.eval()
            supervised_head.eval()
            all_preds, all_labels = [], []
            total_recon_error = 0
            num_batches = 0

            with torch.no_grad():
                for t_data, labels, s_data, handcrafted, last_vals in val_loader:
                    s_data = s_data.to(DEVICE)
                    handcrafted = handcrafted.to(DEVICE)
                    auxiliary = torch.cat([handcrafted, s_data], dim=1)

                    z_combined, z_preserved, z_novel, _, _, _ = policy_net(
                        t_data, auxiliary, deterministic=True
                    )

                    combined = torch.cat([z_combined, s_data], dim=1)
                    preds = supervised_head(combined).squeeze(-1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.numpy())

                    # Track reconstruction error
                    recon = policy_net.reconstruct_handcrafted(z_preserved)
                    recon_error = torch.mean((recon - handcrafted) ** 2).item()
                    total_recon_error += recon_error
                    num_batches += 1

            val_aupr = average_precision_score(all_labels, all_preds)
            avg_recon_error = total_recon_error / num_batches
            scheduler.step(val_aupr)

            print(f"    Pretrain Epoch {epoch+1} | Val AUPR: {val_aupr:.4f} | Recon Error: {avg_recon_error:.6f}")

            if val_aupr > best_auc:
                best_auc = val_aupr
                best_state = copy.deepcopy(policy_net.state_dict())
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    break

    if best_state is not None:
        policy_net.load_state_dict(best_state)

    print(f"  [Two-Component Pretraining] Completed. Best Val AUPR: {best_auc:.4f}")
    return policy_net, best_auc

# ==============================================================================
# 4. Feature Extraction
# ==============================================================================

def extract_two_component_features(policy_net, loader, deterministic=False, temperature=1.0):
    """Extract [Static + Last + Z_preserved + Z_novel]"""
    policy_net.eval() if deterministic else policy_net.train()

    all_features = []
    all_labels = []
    all_log_probs = []

    with torch.set_grad_enabled(not deterministic):
        for t_data, labels, s_data, handcrafted, last_vals in loader:
            s_data = s_data.to(DEVICE)
            handcrafted = handcrafted.to(DEVICE)

            auxiliary = torch.cat([handcrafted, s_data], dim=1)

            z_combined, z_preserved, z_novel, log_prob, preserved_mean, novel_mean = policy_net(
                t_data, auxiliary, deterministic=deterministic, temperature=temperature
            )

            if deterministic:
                z_combined_np = torch.cat([preserved_mean, novel_mean], dim=1).detach().cpu().numpy()
            else:
                z_combined_np = z_combined.detach().cpu().numpy()

            s_np = s_data.cpu().numpy()
            last_np = last_vals.numpy()

            # [Static + Last + Z_preserved + Z_novel]
            combined = np.hstack([s_np, last_np, z_combined_np])

            all_features.append(combined)
            all_labels.extend(labels.numpy())

            if not deterministic and log_prob is not None:
                all_log_probs.append(log_prob)

    features = np.vstack(all_features)
    labels = np.array(all_labels)
    log_probs = torch.cat(all_log_probs) if all_log_probs else None

    return features, labels, log_probs

# ==============================================================================
# 5. RL Training
# ==============================================================================

def train_policy_two_component_rl(
    policy_net,
    train_loader,
    val_loader,
    tabpfn_params,
    pretrain_aupr,
    epochs=120,
    update_tabpfn_every=5
):
    if pretrain_aupr < 0.75:
        lr = 0.0002
        print(f"  [Two-Component RL] Using higher LR (0.0002)")
    else:
        lr = 0.0001
        print(f"  [Two-Component RL] Using standard LR (0.0001)")

    optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr)

    best_val_auc = 0
    best_state = None
    patience = 30
    patience_counter = 0

    print(f"  [Two-Component RL] Training for up to {epochs} epochs...")

    tabpfn_model = None
    val_history = []

    for epoch in range(epochs):
        temperature = max(0.25, 0.7 - epoch / (epochs * 0.6))

        policy_net.train()

        X_train, y_train, log_probs_train = extract_two_component_features(
            policy_net, train_loader, deterministic=False, temperature=temperature
        )

        if epoch % update_tabpfn_every == 0 or tabpfn_model is None:
            tabpfn_model = TabPFNClassifier(**tabpfn_params)
            tabpfn_model.fit(X_train, y_train)

        X_val_stoch, y_val, _ = extract_two_component_features(
            policy_net, val_loader, deterministic=False, temperature=temperature
        )

        y_val_proba = tabpfn_model.predict_proba(X_val_stoch)[:, 1]
        val_aupr = average_precision_score(y_val, y_val_proba)
        val_history.append(val_aupr)

        y_train_proba = tabpfn_model.predict_proba(X_train)[:, 1]
        rewards_smooth = np.where(y_train == 1, y_train_proba, 1 - y_train_proba)

        aupr_weight = 0.3 + 0.3 * min(1.0, epoch / 50)
        rewards_combined = rewards_smooth + aupr_weight * val_aupr

        rewards_tensor = torch.tensor(rewards_combined, dtype=torch.float32).to(DEVICE)
        rewards_tensor = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std() + 1e-8)

        log_probs_train = log_probs_train.to(DEVICE)
        policy_loss = -(log_probs_train * rewards_tensor).mean()

        entropy_bonus = 0.001 * log_probs_train.mean()
        total_loss = policy_loss - entropy_bonus

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=0.5)
        optimizer.step()

        if (epoch + 1) % 5 == 0:
            policy_net.eval()
            with torch.no_grad():
                X_val_det, y_val_det, _ = extract_two_component_features(
                    policy_net, val_loader, deterministic=True
                )

                X_train_det, y_train_det, _ = extract_two_component_features(
                    policy_net, train_loader, deterministic=True
                )

                tabpfn_val_model = TabPFNClassifier(**tabpfn_params)
                tabpfn_val_model.fit(X_train_det, y_train_det)

                y_val_proba_det = tabpfn_val_model.predict_proba(X_val_det)[:, 1]

                val_auc = roc_auc_score(y_val_det, y_val_proba_det)
                val_aupr_det = average_precision_score(y_val_det, y_val_proba_det)

                recent_avg = np.mean(val_history[-10:]) if len(val_history) >= 10 else val_aupr_det

                print(f"    Epoch {epoch+1:3d} | Temp: {temperature:.3f} | AUPR wt: {aupr_weight:.2f} | "
                      f"Val AUC: {val_auc:.4f} | Val AUPR: {val_aupr_det:.4f} | Avg: {recent_avg:.4f}")

                if val_aupr_det > best_val_auc:
                    best_val_auc = val_aupr_det
                    best_state = copy.deepcopy(policy_net.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"    Early stopping at epoch {epoch+1}")
                        break

    if best_state is not None:
        policy_net.load_state_dict(best_state)

    return policy_net

# ==============================================================================
# 6. Main Execution
# ==============================================================================

def main():
    print("="*80)
    print("RL V6: Two-Component Z Architecture")
    print("Z_preserved (100 dims) + Z_novel (28 dims)")
    print("="*80)

    patients = load_and_prepare_patients()
    temporal_feats = get_all_temporal_features(patients)

    print("Extracting and encoding static features...")
    df_static_all = patients.getMeasuresBetween(
        pd.Timedelta(hours=-6), pd.Timedelta(hours=24), "last", getUntilAkiPositive=True
    )
    static_cols = [col for col in FIXED_FEATURES if col in df_static_all.columns]
    df_static_all = df_static_all[static_cols]

    print(f"Input: {len(temporal_feats)} Temporal + {len(static_cols)} Static Features")

    num_handcrafted = len(temporal_feats) * 7
    print(f"Handcrafted dimension: {num_handcrafted} (7 stats per temporal feature)")
    print(f"Z_preserved dimension: 100 (for handcrafted compression)")
    print(f"Z_novel dimension: 28 (for novel pattern learning)")

    metrics_rl = {k: [] for k in ['auc', 'auc_pr']}
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

        # Extract and encode static features properly for this fold
        df_static_train_raw = train_p_obj.getMeasuresBetween(
            pd.Timedelta(hours=-6), pd.Timedelta(hours=24), "last", getUntilAkiPositive=True
        )
        df_static_val_raw = val_p_obj.getMeasuresBetween(
            pd.Timedelta(hours=-6), pd.Timedelta(hours=24), "last", getUntilAkiPositive=True
        )
        df_static_test_raw = test_p.getMeasuresBetween(
            pd.Timedelta(hours=-6), pd.Timedelta(hours=24), "last", getUntilAkiPositive=True
        )

        # Keep only static features for encoding
        df_static_train = df_static_train_raw[static_cols].reset_index(drop=True)
        df_static_val = df_static_val_raw[static_cols].reset_index(drop=True)
        df_static_test = df_static_test_raw[static_cols].reset_index(drop=True)

        # Encode static features using the same method as baseline
        df_static_train_enc, df_static_test_enc, df_static_val_enc = encodeCategoricalData(
            df_static_train, df_static_test, df_static_val
        )

        # Fill NaN values with 0 (important!)
        df_static_train_enc = df_static_train_enc.fillna(0)
        df_static_val_enc = df_static_val_enc.fillna(0)
        df_static_test_enc = df_static_test_enc.fillna(0)

        # Now create datasets with properly encoded static features
        train_ds = HybridDatasetV6(train_p, temporal_feats, df_static_train_enc)
        stats = train_ds.get_normalization_stats()
        val_ds = HybridDatasetV6(val_p, temporal_feats, df_static_val_enc, stats)
        test_ds = HybridDatasetV6(test_p_list, temporal_feats, df_static_test_enc, stats)

        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=hybrid_collate_fn_v6)
        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=hybrid_collate_fn_v6)
        test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=hybrid_collate_fn_v6)

        # Get actual encoded static feature dimension
        static_dim = df_static_train_enc.shape[1]
        auxiliary_dim_actual = num_handcrafted + static_dim

        print(f"  Static features encoded dimension: {static_dim} (from {len(static_cols)} features)")
        print(f"  Auxiliary dimension: {auxiliary_dim_actual}")

        # Two-component architecture
        z_preserved_dim = 100  # Large enough for handcrafted (105)
        z_novel_dim = 28       # For novel patterns

        policy_net = RNNPolicyNetworkTwoComponent(
            input_dim=len(temporal_feats),
            hidden_dim=24,
            z_preserved_dim=z_preserved_dim,
            z_novel_dim=z_novel_dim,
            auxiliary_dim=auxiliary_dim_actual,
            handcrafted_dim=num_handcrafted,
            time_dim=32
        ).to(DEVICE)

        tabpfn_params = {
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }

        # STEP 1: Pretraining with strong reconstruction
        policy_net, pretrain_aupr = pretrain_rnn_two_component(
            policy_net, train_loader, val_loader, static_dim, epochs=50
        )

        # STEP 2: RL Fine-tuning
        policy_net = train_policy_two_component_rl(
            policy_net,
            train_loader,
            val_loader,
            tabpfn_params,
            pretrain_aupr,
            epochs=120,
            update_tabpfn_every=5
        )

        # RL Evaluation
        print("\n  [RL Test Evaluation]")
        policy_net.eval()

        with torch.no_grad():
            X_train_final, y_train_final, _ = extract_two_component_features(
                policy_net, train_loader, deterministic=True
            )
            X_test_final, y_test_final, _ = extract_two_component_features(
                policy_net, test_loader, deterministic=True
            )

        final_tabpfn = TabPFNClassifier(**tabpfn_params)
        final_tabpfn.fit(X_train_final, y_train_final)

        y_test_proba = final_tabpfn.predict_proba(X_test_final)[:, 1]
        y_test_pred = (y_test_proba > 0.5).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_test_final, y_test_pred).ravel()
        prec, rec, _ = precision_recall_curve(y_test_final, y_test_proba)

        fold_auc = roc_auc_score(y_test_final, y_test_proba)
        fold_aupr = auc(rec, prec)

        metrics_rl['auc'].append(fold_auc)
        metrics_rl['auc_pr'].append(fold_aupr)

        fpr, tpr, _ = roc_curve(y_test_final, y_test_proba)
        ax1.plot(fpr, tpr, lw=2, label=f"Fold {fold} (AUC = {fold_auc:.3f})")

        print(f"  RL Test AUC: {fold_auc:.4f} | Test AUPR: {fold_aupr:.4f}")

        # BASELINE (Same as V3)
        print("\n  [Baseline] Simple [Static + Last]...")

        df_train_temp = train_p_obj.getMeasuresBetween(
            pd.Timedelta(hours=-6), pd.Timedelta(hours=24), "last", getUntilAkiPositive=True
        ).drop(columns=["subject_id", "hadm_id", "stay_id"])
        df_test_temp = test_p.getMeasuresBetween(
            pd.Timedelta(hours=-6), pd.Timedelta(hours=24), "last", getUntilAkiPositive=True
        ).drop(columns=["subject_id", "hadm_id", "stay_id"])

        df_train_enc, df_test_enc, _ = encodeCategoricalData(df_train_temp, df_test_temp)

        X_train_baseline = df_train_enc.drop(columns=["akd"]).fillna(0)
        y_train_baseline = df_train_enc["akd"]
        X_test_baseline = df_test_enc.drop(columns=["akd"]).fillna(0)
        y_test_baseline = df_test_enc["akd"]

        tabpfn_base = TabPFNClassifier(**tabpfn_params)
        tabpfn_base.fit(X_train_baseline, y_train_baseline)

        y_prob_b = tabpfn_base.predict_proba(X_test_baseline)[:, 1]
        prec_b, rec_b, _ = precision_recall_curve(y_test_baseline, y_prob_b)

        baseline_auc = roc_auc_score(y_test_baseline, y_prob_b)
        baseline_aupr = auc(rec_b, prec_b)

        metrics_baseline['auc'].append(baseline_auc)
        metrics_baseline['auc_pr'].append(baseline_aupr)

        fpr_b, tpr_b, _ = roc_curve(y_test_baseline, y_prob_b)
        ax2.plot(fpr_b, tpr_b, lw=2, label=f"Fold {fold} (AUC = {baseline_auc:.3f})")

        print(f"  Baseline Test AUC: {baseline_auc:.4f} | Test AUPR: {baseline_aupr:.4f}")
        print(f"  Fold {fold} Results -> RL: {fold_auc:.3f} vs Baseline: {baseline_auc:.3f}")

    # Final Plot
    for ax in [ax1, ax2]:
        ax.plot([0, 1], [0, 1], linestyle="--", color="navy", lw=2)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(loc="lower right")

    ax1.set_title("RL V6 [Static+Last+Z_preserved+Z_novel]")
    ax2.set_title("Baseline [Static+Last]")
    plt.tight_layout()
    plt.savefig("result/tab_rlv6_two_component.png", dpi=300)
    print("\nPlot saved to result/tab_rlv6_two_component.png")

    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY")
    print("="*80)

    def print_stat(name, rl_metrics, base_metrics):
        rl_mean, rl_std = np.mean(rl_metrics), np.std(rl_metrics)
        base_mean, base_std = np.mean(base_metrics), np.std(base_metrics)
        improvement = ((rl_mean - base_mean) / base_mean) * 100
        symbol = "✓" if rl_mean > base_mean else "✗"
        print(f"{name:15s} | RL: {rl_mean:.4f} ± {rl_std:.4f}  vs  Baseline: {base_mean:.4f} ± {base_std:.4f}  ({improvement:+.2f}%) {symbol}")

    print_stat("AUC", metrics_rl['auc'], metrics_baseline['auc'])
    print_stat("AUC-PR", metrics_rl['auc_pr'], metrics_baseline['auc_pr'])

    print("\n" + "="*80)
    print("V6 ARCHITECTURE:")
    print("✓ Z_preserved (100 dims) - Preserves handcrafted features")
    print("✓ Z_novel (28 dims) - Learns novel patterns")
    print("✓ Strong reconstruction loss (weight = 2.0)")
    print("✓ Separate optimization objectives")
    print("="*80)

if __name__ == "__main__":
    main()
