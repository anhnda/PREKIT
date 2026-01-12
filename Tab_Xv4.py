"""
Tab_Xv4: Per-Feature Learned Statistics (Correct Architecture!)

Problem with Tab_Xv3: Dimension mismatch!
- Learned stats: Global statistics from Bi-GRU (batch, 7*hidden*2)
- True stats: Per-feature statistics (batch, num_features*7)
- Stat MSE ~13 and not decreasing → model not learning!

Solution: Learn statistics PER TEMPORAL FEATURE (like Tab_RLv4)
- For each temporal feature: learn [last, mean, std, min, max, slope, count]
- Direct 1-to-1 correspondence with handcrafted features
- Proper supervision that actually teaches the model!

Architecture:
  For each temporal feature independently:
    Time Series → 1D CNN + Attention → 7 Statistics

  TabPFN Input: [Baseline + Learned Stats for all features]
"""

import pandas as pd
import numpy as np
import sys
import copy
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================================================================
# 1. Per-Feature Statistical Encoder
# ==============================================================================

class PerFeatureStatisticalEncoder(nn.Module):
    """
    For EACH temporal feature, learn 7 statistics independently.
    This matches the granularity of handcrafted features!
    """
    def __init__(self, hidden_dim=32):
        super().__init__()
        self.hidden_dim = hidden_dim

        # 1D CNN to process temporal sequence (per feature)
        self.conv1 = nn.Conv1d(1, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)

        # Attention for aggregation
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # Statistical heads (output 7 statistics)
        # Use Tanh to bound outputs to [-1, 1] range
        self.stat_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),  # 3 = last + max + attention
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 7),  # 7 statistics
            nn.Tanh()  # CRITICAL: Bound outputs to [-1, 1]
        )

    def forward(self, values, mask):
        """
        Process ONE temporal feature's time series
        Args:
            values: (batch, seq_len) - one feature's values over time
            mask: (batch, seq_len) - validity mask
        Returns:
            stats: (batch, 7) - [last, mean, std, min, max, slope, count]
        """
        batch_size, seq_len = values.shape

        # Add channel dimension: (batch, 1, seq_len)
        x = values.unsqueeze(1)

        # Apply convolutions
        x = F.relu(self.conv1(x))  # (batch, hidden, seq_len)
        x = F.relu(self.conv2(x))  # (batch, hidden, seq_len)

        # Transpose for attention: (batch, seq_len, hidden)
        x = x.transpose(1, 2)

        # Mask invalid positions
        mask_expanded = mask.unsqueeze(-1)  # (batch, seq_len, 1)
        x = x * mask_expanded

        # Compute statistics
        # 1. Last value
        lengths = mask.sum(dim=1, keepdim=True).long()  # (batch, 1)
        lengths = torch.clamp(lengths - 1, min=0)
        last_idx = lengths.unsqueeze(-1).expand(-1, -1, self.hidden_dim)  # (batch, 1, hidden)
        last_state = x.gather(1, last_idx).squeeze(1)  # (batch, hidden)

        # 2. Max pooling
        masked_for_max = x.clone()
        masked_for_max[mask_expanded.expand_as(x) == 0] = -1e9
        max_pool = torch.max(masked_for_max, dim=1)[0]  # (batch, hidden)

        # 3. Attention-weighted aggregation
        attn_weights = self.attention(x)  # (batch, seq_len, 1)
        attn_weights = attn_weights.masked_fill(mask.unsqueeze(-1) == 0, -1e9)
        attn_weights = F.softmax(attn_weights, dim=1)
        context = torch.sum(x * attn_weights, dim=1)  # (batch, hidden)

        # Combine all views
        combined = torch.cat([last_state, max_pool, context], dim=1)  # (batch, hidden*3)

        # Predict 7 statistics
        stats = self.stat_head(combined)  # (batch, 7)

        return stats


class MultiFeatureStatisticalEncoder(nn.Module):
    """
    Applies PerFeatureStatisticalEncoder to ALL temporal features
    """
    def __init__(self, num_features, hidden_dim=32):
        super().__init__()
        self.num_features = num_features

        # One encoder per feature (weight sharing)
        self.feature_encoder = PerFeatureStatisticalEncoder(hidden_dim)

    def forward(self, batch_data):
        """
        Args:
            batch_data: dict with 'values' (batch, seq_len, num_features)
                                   'masks' (batch, seq_len, num_features)
        Returns:
            all_stats: (batch, num_features * 7)
        """
        values = batch_data['values'].to(DEVICE)  # (batch, seq_len, num_features)
        masks = batch_data['masks'].to(DEVICE)

        batch_size, seq_len, num_features = values.shape

        # Process each feature independently
        all_stats = []
        for f_idx in range(num_features):
            feat_values = values[:, :, f_idx]  # (batch, seq_len)
            feat_mask = masks[:, :, f_idx]  # (batch, seq_len)

            stats = self.feature_encoder(feat_values, feat_mask)  # (batch, 7)
            all_stats.append(stats)

        # Stack: (batch, num_features, 7) → flatten to (batch, num_features*7)
        all_stats = torch.stack(all_stats, dim=1)  # (batch, num_features, 7)
        all_stats = all_stats.view(batch_size, -1)  # (batch, num_features*7)

        return all_stats


# ==============================================================================
# 2. Dataset
# ==============================================================================

class StatisticalDatasetV4(Dataset):
    def __init__(self, patients, feature_names, baseline_features_df, normalization_stats=None):
        self.data = []
        self.labels = []
        self.baseline_features = []
        self.true_stats = []
        self.feature_names = feature_names

        all_values = []
        patient_list = patients.patientList if hasattr(patients, 'patientList') else patients

        for idx, patient in enumerate(patient_list):
            times, values, masks = extract_temporal_data(patient, feature_names)
            if times is None:
                continue

            baseline_vec = baseline_features_df.iloc[idx].values
            self.baseline_features.append(torch.tensor(baseline_vec, dtype=torch.float32))

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

            # Compute ground truth PER FEATURE
            true_handcrafted = self._compute_handcrafted_features(
                norm_values, self.data[i]['masks'], self.data[i]['times']
            )
            true_handcrafted = np.nan_to_num(true_handcrafted, nan=0.0, posinf=0.0, neginf=0.0)
            self.true_stats.append(torch.tensor(true_handcrafted, dtype=torch.float32))

            self.data[i] = {
                'times': torch.tensor(self.data[i]['times'], dtype=torch.float32),
                'values': torch.tensor(norm_values, dtype=torch.float32),
                'masks': torch.tensor(self.data[i]['masks'], dtype=torch.float32)
            }

    def _compute_handcrafted_features(self, norm_values, masks, times):
        """Returns: (num_features * 7,) matching learned stats exactly!"""
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
                slope = (valid_vals[-1] - valid_vals[0]) / (valid_times[-1] - valid_times[0] + 1e-6) if len(valid_vals) >= 2 else 0.0
                count = len(valid_idx)

                # Normalize to [-1, 1] range using tanh(x/3) to match model output
                # Values are already normalized (z-score), so typically in [-3, 3]
                last = np.tanh(last / 3.0)
                mean_val = np.tanh(mean_val / 3.0)
                std_val = np.tanh(std_val / 2.0)  # Std is typically 0-2
                min_val = np.tanh(min_val / 3.0)
                max_val = np.tanh(max_val / 3.0)
                slope = np.tanh(slope / 0.5)  # Slope is typically small
                count = np.tanh(count / 10.0)  # Normalize count (typically 1-20)

                # Order: [last, mean, std, min, max, slope, count]
                handcrafted.extend([last, mean_val, std_val, min_val, max_val, slope, count])
            else:
                handcrafted.extend([0.0] * 7)

        return handcrafted

    def get_normalization_stats(self):
        return {'mean': self.mean, 'std': self.std}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (self.data[idx], self.labels[idx], self.baseline_features[idx],
                self.true_stats[idx])


def statistical_collate_fn(batch):
    data_list, label_list, baseline_list, true_stats_list = zip(*batch)
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
            torch.stack(baseline_list), torch.stack(true_stats_list))


# ==============================================================================
# 3. Supervised Head
# ==============================================================================

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
# 4. Training
# ==============================================================================

def train_encoder(encoder, train_loader, val_loader, baseline_dim, num_features, epochs=50):
    print("  [Training] Per-feature statistical encoder with supervision...")

    learned_stats_dim = num_features * 7  # Exactly matches handcrafted!

    supervised_head = SupervisedHead(baseline_dim + learned_stats_dim).to(DEVICE)

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(supervised_head.parameters()),
        lr=0.001
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )

    criterion_cls = nn.BCELoss()
    criterion_stat = nn.MSELoss()

    best_auc = 0
    best_state = None
    patience = 12
    counter = 0

    for epoch in range(epochs):
        encoder.train()
        supervised_head.train()

        for t_data, labels, baseline_feats, true_stats in train_loader:
            labels = labels.to(DEVICE)
            baseline_feats = baseline_feats.to(DEVICE)
            true_stats = true_stats.to(DEVICE)

            # Extract learned statistics (per feature!)
            learned_stats = encoder(t_data)  # (batch, num_features*7)

            # Classification loss
            combined = torch.cat([baseline_feats, learned_stats], dim=1)
            preds = supervised_head(combined).squeeze(-1)
            cls_loss = criterion_cls(preds, labels)

            # Statistical matching loss (now dimensions match!)
            stat_loss = criterion_stat(learned_stats, true_stats)

            # Safety check: skip batch if stat_loss is too large
            if stat_loss.item() > 100.0:
                print(f"    WARNING: Skipping batch with stat_loss={stat_loss.item():.2f}")
                continue

            # Combined loss (reduced stat weight to 0.3 for stability)
            total_loss = cls_loss + 0.3 * stat_loss

            # Check for NaN/Inf
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print(f"    WARNING: NaN/Inf loss detected, skipping batch")
                continue

            optimizer.zero_grad()
            total_loss.backward()

            # Aggressive gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(supervised_head.parameters()),
                max_norm=0.5  # More aggressive clipping
            )
            optimizer.step()

        if (epoch + 1) % 3 == 0:
            encoder.eval()
            supervised_head.eval()
            all_preds, all_labels = [], []
            total_stat_error = 0
            num_batches = 0

            with torch.no_grad():
                for t_data, labels, baseline_feats, true_stats in val_loader:
                    baseline_feats = baseline_feats.to(DEVICE)
                    true_stats = true_stats.to(DEVICE)

                    learned_stats = encoder(t_data)
                    combined = torch.cat([baseline_feats, learned_stats], dim=1)
                    preds = supervised_head(combined).squeeze(-1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.numpy())

                    stat_error = torch.mean((learned_stats - true_stats) ** 2).item()
                    total_stat_error += stat_error
                    num_batches += 1

            val_aupr = average_precision_score(all_labels, all_preds)
            avg_stat_error = total_stat_error / num_batches
            scheduler.step(val_aupr)

            print(f"    Epoch {epoch+1} | Val AUPR: {val_aupr:.4f} | Stat MSE: {avg_stat_error:.6f}")

            if val_aupr > best_auc:
                best_auc = val_aupr
                best_state = copy.deepcopy(encoder.state_dict())
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    break

    if best_state is not None:
        encoder.load_state_dict(best_state)

    print(f"  [Training] Completed. Best Val AUPR: {best_auc:.4f}")
    return encoder, best_auc


def extract_features(encoder, loader):
    encoder.eval()
    all_features = []
    all_labels = []

    with torch.no_grad():
        for t_data, labels, baseline_feats, _ in loader:
            learned_stats = encoder(t_data)
            learned_stats_np = learned_stats.cpu().numpy()
            baseline_np = baseline_feats.numpy()

            combined = np.hstack([baseline_np, learned_stats_np])
            all_features.append(combined)
            all_labels.extend(labels.numpy())

    return np.vstack(all_features), np.array(all_labels)


# ==============================================================================
# 5. Main
# ==============================================================================

def main():
    print("="*80)
    print("Tab_Xv4: Per-Feature Learned Statistics (Correct Architecture!)")
    print("="*80)

    patients = load_and_prepare_patients()
    temporal_feats = get_all_temporal_features(patients)

    print(f"Temporal features: {len(temporal_feats)}")
    print(f"Per feature: 7 learned statistics [last, mean, std, min, max, slope, count]")
    print(f"Total learned: {len(temporal_feats)} * 7 = {len(temporal_feats)*7} statistics")

    metrics_rl = {k: [] for k in ['auc', 'auc_pr']}
    metrics_baseline = {k: [] for k in ['auc', 'auc_pr']}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    for fold, (train_full, test_p) in enumerate(trainTestPatients(patients, seed=xseed)):
        print(f"\n{'='*80}")
        print(f"Fold {fold}")
        print('='*80)

        train_p_obj, val_p_obj = split_patients_train_val(train_full, val_ratio=0.1, seed=42+fold)

        # Baseline features
        df_train_base = train_p_obj.getMeasuresBetween(
            pd.Timedelta(hours=-6), pd.Timedelta(hours=24), "last", getUntilAkiPositive=True
        ).drop(columns=["subject_id", "hadm_id", "stay_id"])
        df_val_base = val_p_obj.getMeasuresBetween(
            pd.Timedelta(hours=-6), pd.Timedelta(hours=24), "last", getUntilAkiPositive=True
        ).drop(columns=["subject_id", "hadm_id", "stay_id"])
        df_test_base = test_p.getMeasuresBetween(
            pd.Timedelta(hours=-6), pd.Timedelta(hours=24), "last", getUntilAkiPositive=True
        ).drop(columns=["subject_id", "hadm_id", "stay_id"])

        df_train_enc, df_test_enc, df_val_enc = encodeCategoricalData(
            df_train_base, df_test_base, df_val_base
        )

        X_train_baseline = df_train_enc.drop(columns=["akd"]).fillna(0)
        X_val_baseline = df_val_enc.drop(columns=["akd"]).fillna(0)
        X_test_baseline = df_test_enc.drop(columns=["akd"]).fillna(0)

        baseline_dim = X_train_baseline.shape[1]

        # Create datasets
        train_p = train_p_obj.patientList
        val_p = val_p_obj.patientList
        test_p_list = test_p.patientList

        train_ds = StatisticalDatasetV4(train_p, temporal_feats, X_train_baseline)
        stats = train_ds.get_normalization_stats()
        val_ds = StatisticalDatasetV4(val_p, temporal_feats, X_val_baseline, stats)
        test_ds = StatisticalDatasetV4(test_p_list, temporal_feats, X_test_baseline, stats)

        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=statistical_collate_fn)
        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=statistical_collate_fn)
        test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=statistical_collate_fn)

        # Create encoder
        encoder = MultiFeatureStatisticalEncoder(
            num_features=len(temporal_feats),
            hidden_dim=32
        ).to(DEVICE)

        tabpfn_params = {'device': 'cuda' if torch.cuda.is_available() else 'cpu'}

        # Train
        encoder, pretrain_aupr = train_encoder(
            encoder, train_loader, val_loader, baseline_dim, len(temporal_feats), epochs=50
        )

        # Evaluate
        print("\n  [Test Evaluation]")
        X_train_final, y_train_final = extract_features(encoder, train_loader)
        X_test_final, y_test_final = extract_features(encoder, test_loader)

        final_tabpfn = TabPFNClassifier(**tabpfn_params)
        final_tabpfn.fit(X_train_final, y_train_final)

        y_test_proba = final_tabpfn.predict_proba(X_test_final)[:, 1]
        prec, rec, _ = precision_recall_curve(y_test_final, y_test_proba)

        fold_auc = roc_auc_score(y_test_final, y_test_proba)
        fold_aupr = auc(rec, prec)

        metrics_rl['auc'].append(fold_auc)
        metrics_rl['auc_pr'].append(fold_aupr)

        fpr, tpr, _ = roc_curve(y_test_final, y_test_proba)
        ax1.plot(fpr, tpr, lw=2, label=f"Fold {fold} (AUC = {fold_auc:.3f})")

        print(f"  Tab_Xv4 AUC: {fold_auc:.4f} | AUPR: {fold_aupr:.4f}")

        # Baseline
        y_train_baseline = df_train_enc["akd"]
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

    ax1.set_title("Tab_Xv4: Baseline + Per-Feature Learned Stats")
    ax2.set_title("Baseline")
    plt.tight_layout()
    plt.savefig("result/tab_xv4_vs_baseline.png", dpi=300)

    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)

    def print_stat(name, rl_metrics, base_metrics):
        rl_mean, rl_std = np.mean(rl_metrics), np.std(rl_metrics)
        base_mean, base_std = np.mean(base_metrics), np.std(base_metrics)
        improvement = ((rl_mean - base_mean) / base_mean) * 100
        symbol = "✓" if rl_mean > base_mean else "✗"
        print(f"{name} | Tab_Xv4: {rl_mean:.4f}±{rl_std:.4f} vs Baseline: {base_mean:.4f}±{base_std:.4f} ({improvement:+.2f}%) {symbol}")

    print_stat("AUC   ", metrics_rl['auc'], metrics_baseline['auc'])
    print_stat("AUC-PR", metrics_rl['auc_pr'], metrics_baseline['auc_pr'])

    print("\n" + "="*80)
    print("Tab_Xv4 KEY FIX:")
    print("✓ Processes EACH temporal feature independently")
    print("✓ Learns 7 statistics PER FEATURE (not global)")
    print("✓ Dimensions match: learned (N*7) = true (N*7)")
    print("✓ Proper supervision that actually teaches the model!")
    print("✓ Stat MSE should now decrease significantly!")
    print("="*80)

if __name__ == "__main__":
    main()
