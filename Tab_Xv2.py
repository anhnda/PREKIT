"""
Tab_Xv2: Baseline Features + Learned Temporal Context

Key insight: Fair comparison requires same feature sets!

Baseline: [All Static + All Temporal Last Values]
Tab_Xv2: [All Static + All Temporal Last Values] + [Learned Z]

This isolates the contribution of learned temporal context without confounding
by different feature selection.
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
from XEmbedding import DeepStatisticalTemporalEncoder

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================================================================
# 1. Dataset with Baseline Features
# ==============================================================================

class HybridDatasetV2(Dataset):
    """
    Dataset that stores:
    1. Temporal data for encoder
    2. Baseline features (from getMeasuresBetween)
    """
    def __init__(self, patients, feature_names, baseline_features_df, normalization_stats=None):
        self.data = []
        self.labels = []
        self.baseline_features = []
        self.feature_names = feature_names

        all_values = []
        patient_list = patients.patientList if hasattr(patients, 'patientList') else patients

        for idx, patient in enumerate(patient_list):
            times, values, masks = extract_temporal_data(patient, feature_names)
            if times is None:
                continue

            # Store baseline features (same as baseline uses)
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
        return self.data[idx], self.labels[idx], self.baseline_features[idx]

def hybrid_collate_fn(batch):
    data_list, label_list, baseline_list = zip(*batch)
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
    return temporal_batch, torch.tensor(label_list, dtype=torch.float32), torch.stack(baseline_list)

# ==============================================================================
# 2. Supervised Head
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
# 3. Pretraining
# ==============================================================================

def pretrain_encoder(policy_net, train_loader, val_loader, baseline_dim, epochs=50):
    print("  [Pretraining] Learning temporal context on top of baseline features...")

    supervised_head = SupervisedHead(
        policy_net.latent_dim + baseline_dim
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        list(policy_net.parameters()) + list(supervised_head.parameters()),
        lr=0.001
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    criterion = nn.BCELoss()

    best_auc = 0
    best_state = None
    patience = 12
    counter = 0

    for epoch in range(epochs):
        policy_net.train()
        supervised_head.train()

        for t_data, labels, baseline_feats in train_loader:
            labels = labels.to(DEVICE)
            baseline_feats = baseline_feats.to(DEVICE)

            z, _, _ = policy_net(t_data, deterministic=True)
            combined = torch.cat([baseline_feats, z], dim=1)

            preds = supervised_head(combined).squeeze(-1)
            loss = criterion(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(policy_net.parameters()) + list(supervised_head.parameters()),
                max_norm=1.0
            )
            optimizer.step()

        if (epoch + 1) % 3 == 0:
            policy_net.eval()
            supervised_head.eval()
            all_preds, all_labels = [], []

            with torch.no_grad():
                for t_data, labels, baseline_feats in val_loader:
                    baseline_feats = baseline_feats.to(DEVICE)
                    z, _, _ = policy_net(t_data, deterministic=True)
                    combined = torch.cat([baseline_feats, z], dim=1)
                    preds = supervised_head(combined).squeeze(-1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.numpy())

            val_aupr = average_precision_score(all_labels, all_preds)
            scheduler.step(val_aupr)

            print(f"    Pretrain Epoch {epoch+1} | Val AUPR: {val_aupr:.4f}")

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

    print(f"  [Pretraining] Completed. Best Val AUPR: {best_auc:.4f}")
    return policy_net, best_auc

# ==============================================================================
# 4. Feature Extraction: [Baseline Features + Learned Z]
# ==============================================================================

def extract_features_with_context(policy_net, loader, deterministic=False, temperature=1.0):
    """
    Extract: [Baseline Features + Learned Z]
    Pure augmentation of baseline with learned temporal context
    """
    policy_net.eval() if deterministic else policy_net.train()

    all_features = []
    all_labels = []
    all_log_probs = []

    with torch.set_grad_enabled(not deterministic):
        for t_data, labels, baseline_feats in loader:
            # Get learned temporal context
            z, log_prob, mean = policy_net(t_data, deterministic=deterministic, temperature=temperature)
            z_np = (mean if deterministic else z).detach().cpu().numpy()

            # Use baseline features as-is
            baseline_np = baseline_feats.numpy()

            # Concatenate: [Baseline + Z]
            combined = np.hstack([baseline_np, z_np])

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

def train_policy_rl(
    policy_net,
    train_loader,
    val_loader,
    tabpfn_params,
    pretrain_aupr,
    epochs=60,
    update_tabpfn_every=3
):
    """
    Conservative RL training to avoid degradation
    """

    # Very conservative learning rate
    if pretrain_aupr < 0.75:
        lr = 0.00005
        print(f"  [RL] Using higher LR (0.00005) due to lower pretrain AUPR")
    else:
        lr = 0.00003
        print(f"  [RL] Using very conservative LR (0.00003)")

    optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr)

    best_val_auc = 0
    best_state = None
    patience = 12  # Strict early stopping
    patience_counter = 0

    print(f"  [RL] Training for up to {epochs} epochs...")

    tabpfn_model = None
    val_history = []

    for epoch in range(epochs):
        # Very slow temperature decay
        temperature = max(0.5, 1.0 - epoch / (epochs * 1.5))

        policy_net.train()

        X_train, y_train, log_probs_train = extract_features_with_context(
            policy_net, train_loader, deterministic=False, temperature=temperature
        )

        if epoch % update_tabpfn_every == 0 or tabpfn_model is None:
            tabpfn_model = TabPFNClassifier(**tabpfn_params)
            tabpfn_model.fit(X_train, y_train)

        X_val_stoch, y_val, _ = extract_features_with_context(
            policy_net, val_loader, deterministic=False, temperature=temperature
        )

        y_val_proba = tabpfn_model.predict_proba(X_val_stoch)[:, 1]
        val_aupr = average_precision_score(y_val, y_val_proba)
        val_history.append(val_aupr)

        # Compute rewards
        y_train_proba = tabpfn_model.predict_proba(X_train)[:, 1]
        rewards_smooth = np.where(y_train == 1, y_train_proba, 1 - y_train_proba)

        # Very conservative AUPR weight
        aupr_weight = 0.1 + 0.2 * min(1.0, epoch / 30)  # 0.1 → 0.3

        rewards_combined = rewards_smooth + aupr_weight * val_aupr

        rewards_tensor = torch.tensor(rewards_combined, dtype=torch.float32).to(DEVICE)
        rewards_tensor = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std() + 1e-8)

        log_probs_train = log_probs_train.to(DEVICE)
        policy_loss = -(log_probs_train * rewards_tensor).mean()

        # Strong entropy bonus
        entropy_bonus = 0.01 * log_probs_train.mean()
        total_loss = policy_loss - entropy_bonus

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=0.5)
        optimizer.step()

        # Frequent validation
        if (epoch + 1) % 3 == 0 or epoch == 0:
            policy_net.eval()
            with torch.no_grad():
                X_val_det, y_val_det, _ = extract_features_with_context(
                    policy_net, val_loader, deterministic=True
                )

                X_train_det, y_train_det, _ = extract_features_with_context(
                    policy_net, train_loader, deterministic=True
                )

                tabpfn_val_model = TabPFNClassifier(**tabpfn_params)
                tabpfn_val_model.fit(X_train_det, y_train_det)

                y_val_proba_det = tabpfn_val_model.predict_proba(X_val_det)[:, 1]

                val_auc = roc_auc_score(y_val_det, y_val_proba_det)
                val_aupr_det = average_precision_score(y_val_det, y_val_proba_det)

                degradation_warning = ""
                if best_val_auc > 0 and (best_val_auc - val_aupr_det) > 0.02:
                    degradation_warning = " ⚠ DEGRADING"

                print(f"    Epoch {epoch+1:3d} | Temp: {temperature:.3f} | "
                      f"Val AUC: {val_auc:.4f} | Val AUPR: {val_aupr_det:.4f} | "
                      f"Best: {best_val_auc:.4f}{degradation_warning}")

                if val_aupr_det > best_val_auc:
                    best_val_auc = val_aupr_det
                    best_state = copy.deepcopy(policy_net.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience or (best_val_auc - val_aupr_det) > 0.05:
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
    print("Tab_Xv2: Baseline Features + Learned Temporal Context")
    print("="*80)

    patients = load_and_prepare_patients()
    temporal_feats = get_all_temporal_features(patients)

    print(f"Temporal features for encoder: {len(temporal_feats)}")
    print(f"Feature set: [Baseline (All Static + All Temporal Last)] + [Learned Z]")

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

        # Extract baseline features (same as baseline uses)
        print("  Extracting baseline features...")
        df_train_base = train_p_obj.getMeasuresBetween(
            pd.Timedelta(hours=-6), pd.Timedelta(hours=24), "last", getUntilAkiPositive=True
        ).drop(columns=["subject_id", "hadm_id", "stay_id"])
        df_val_base = val_p_obj.getMeasuresBetween(
            pd.Timedelta(hours=-6), pd.Timedelta(hours=24), "last", getUntilAkiPositive=True
        ).drop(columns=["subject_id", "hadm_id", "stay_id"])
        df_test_base = test_p.getMeasuresBetween(
            pd.Timedelta(hours=-6), pd.Timedelta(hours=24), "last", getUntilAkiPositive=True
        ).drop(columns=["subject_id", "hadm_id", "stay_id"])

        # Encode baseline features
        df_train_enc, df_test_enc, df_val_enc = encodeCategoricalData(
            df_train_base, df_test_base, df_val_base
        )

        X_train_baseline = df_train_enc.drop(columns=["akd"]).fillna(0)
        X_val_baseline = df_val_enc.drop(columns=["akd"]).fillna(0)
        X_test_baseline = df_test_enc.drop(columns=["akd"]).fillna(0)

        baseline_dim = X_train_baseline.shape[1]
        print(f"  Baseline feature dimension: {baseline_dim}")

        # Create datasets with baseline features
        train_ds = HybridDatasetV2(train_p, temporal_feats, X_train_baseline)
        stats = train_ds.get_normalization_stats()
        val_ds = HybridDatasetV2(val_p, temporal_feats, X_val_baseline, stats)
        test_ds = HybridDatasetV2(test_p_list, temporal_feats, X_test_baseline, stats)

        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=hybrid_collate_fn)
        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=hybrid_collate_fn)
        test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=hybrid_collate_fn)

        # Create encoder
        latent_dim = 32
        policy_net = DeepStatisticalTemporalEncoder(
            input_dim=len(temporal_feats),
            hidden_dim=32,
            latent_dim=latent_dim,
            time_dim=32
        ).to(DEVICE)

        tabpfn_params = {
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }

        # STEP 1: Pretraining
        policy_net, pretrain_aupr = pretrain_encoder(
            policy_net, train_loader, val_loader, baseline_dim, epochs=50
        )

        # STEP 2: RL Fine-tuning
        print(f"\n  Starting RL from pretrain AUPR: {pretrain_aupr:.4f}")
        policy_net = train_policy_rl(
            policy_net,
            train_loader,
            val_loader,
            tabpfn_params,
            pretrain_aupr,
            epochs=60,
            update_tabpfn_every=3
        )

        # Final Evaluation
        print("\n  [Final Test Evaluation]")
        policy_net.eval()

        with torch.no_grad():
            X_train_final, y_train_final, _ = extract_features_with_context(
                policy_net, train_loader, deterministic=True
            )
            X_test_final, y_test_final, _ = extract_features_with_context(
                policy_net, test_loader, deterministic=True
            )

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

        print(f"  Tab_Xv2 Test AUC: {fold_auc:.4f} | Test AUPR: {fold_aupr:.4f}")

        # BASELINE (same features without learned Z)
        print("\n  [Baseline] Same features, no learned context...")

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

        print(f"  Baseline Test AUC: {baseline_auc:.4f} | Test AUPR: {baseline_aupr:.4f}")
        print(f"  Fold {fold} -> Tab_Xv2: {fold_auc:.3f} vs Baseline: {baseline_auc:.3f}")

        improvement = ((fold_aupr - baseline_aupr) / baseline_aupr) * 100
        print(f"  AUPR Improvement: {improvement:+.2f}%")

    # Final Plot
    for ax in [ax1, ax2]:
        ax.plot([0, 1], [0, 1], linestyle="--", color="navy", lw=2)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(loc="lower right")

    ax1.set_title("Tab_Xv2: Baseline + Learned Context")
    ax2.set_title("Baseline Only")
    plt.tight_layout()
    plt.savefig("result/tab_xv2_vs_baseline.png", dpi=300)
    print("\nPlot saved to result/tab_xv2_vs_baseline.png")

    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY")
    print("="*80)

    def print_stat(name, rl_metrics, base_metrics):
        rl_mean, rl_std = np.mean(rl_metrics), np.std(rl_metrics)
        base_mean, base_std = np.mean(base_metrics), np.std(base_metrics)
        improvement = ((rl_mean - base_mean) / base_mean) * 100
        symbol = "✓" if rl_mean > base_mean else "✗"
        print(f"{name:15s} | Tab_Xv2: {rl_mean:.4f} ± {rl_std:.4f}  vs  Baseline: {base_mean:.4f} ± {base_std:.4f}  ({improvement:+.2f}%) {symbol}")

    print_stat("AUC", metrics_rl['auc'], metrics_baseline['auc'])
    print_stat("AUC-PR", metrics_rl['auc_pr'], metrics_baseline['auc_pr'])

    print("\n" + "="*80)
    print("Tab_Xv2 KEY DIFFERENCE:")
    print("✓ Uses EXACT SAME features as baseline")
    print("✓ Only adds learned temporal context (Z)")
    print("✓ Fair comparison: isolated contribution of learned representations")
    print("✓ If Tab_Xv2 > Baseline, then learned context helps!")
    print("✓ If Tab_Xv2 ≈ Baseline, then Z is redundant with handcrafted features")
    print("="*80)

if __name__ == "__main__":
    main()
