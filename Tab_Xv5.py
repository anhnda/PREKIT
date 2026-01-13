"""
Tab_Xv5: AUGMENT (Don't Replace) - Learnable + Handcrafted Features

KEY INSIGHT from Tab_RLv4's success:
- Tab_RLv4 uses BOTH handcrafted features AND learned latent → +3.11% AUPR ✓
- Tab_Xv4 tries to REPLACE handcrafted with learned → -1.41% AUPR ✗

STRATEGY:
1. Keep ALL baseline handcrafted features (don't replace them!)
2. Add complementary learned features that capture what handcrafted can't
3. Use powerful temporal encoder (Bi-GRU + Multi-head Attention)
4. Optimize DIRECTLY for classification task (not reconstruction)
5. Learn unconstrained latent representation (no Tanh bounds)

Architecture:
  Temporal Data → Bi-GRU + Multi-head Attention → Learned Latent (32-dim)

  TabPFN Input: [Baseline Handcrafted Features + Learned Latent]
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^
                 Proven features (keep!)           Learned complement

Why this should work:
- Handcrafted features: Provide strong statistical priors (last, mean, std, etc.)
- Learned features: Capture complex patterns handcrafted can't (interactions, non-linear trends)
- No reconstruction loss: Model learns what's useful for prediction, not matching stats
- Complementary: Two feature types work together, not competing
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
# 1. Enhanced Temporal Encoder (More Powerful than Tab_Xv4)
# ==============================================================================

class TemporalFeatureEncoder(nn.Module):
    """
    Powerful temporal encoder using Bi-GRU + Multi-head Attention
    Goal: Learn complementary features that handcrafted features can't capture
    """
    def __init__(self, num_features, hidden_dim=64, latent_dim=32, num_heads=4):
        super().__init__()
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # Bi-directional GRU for temporal modeling
        self.gru = nn.GRU(
            num_features,
            hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )

        # Multi-head attention for feature interactions
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,  # *2 for bidirectional
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )

        # Learnable query for attention pooling
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim * 2))

        # Feature extraction head (NO constraints, learn freely)
        self.latent_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, latent_dim)
            # NO Tanh! Let model learn unconstrained representation
        )

    def forward(self, batch_data):
        """
        Args:
            batch_data: dict with 'values' (batch, seq_len, num_features)
                                   'masks' (batch, seq_len, num_features)
        Returns:
            latent: (batch, latent_dim) - Unconstrained learned features
        """
        values = batch_data['values'].to(DEVICE)
        masks = batch_data['masks'].to(DEVICE)
        lengths = batch_data['lengths'].to(DEVICE)

        batch_size, seq_len, _ = values.shape

        # Apply mask to values
        masked_values = values * masks

        # Pack sequence for efficient GRU processing
        packed = nn.utils.rnn.pack_padded_sequence(
            masked_values,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        # Bi-GRU encoding
        gru_out, _ = self.gru(packed)
        gru_out, _ = nn.utils.rnn.pad_packed_sequence(gru_out, batch_first=True)
        # gru_out: (batch, seq_len, hidden*2)

        # Multi-head attention with learnable query
        # This allows the model to focus on relevant time points
        query = self.query.expand(batch_size, -1, -1)  # (batch, 1, hidden*2)

        # Create attention mask from sequence mask
        # Aggregate masks across features (any feature valid = timestep valid)
        attn_mask = (masks.sum(dim=-1) == 0)  # (batch, seq_len), True where all features masked

        # Attention: query attends to gru_out
        attn_out, _ = self.attention(
            query,
            gru_out,
            gru_out,
            key_padding_mask=attn_mask
        )  # (batch, 1, hidden*2)

        # Squeeze and project to latent space
        context = attn_out.squeeze(1)  # (batch, hidden*2)
        latent = self.latent_head(context)  # (batch, latent_dim)

        return latent


# ==============================================================================
# 2. Dataset (Similar to Tab_Xv4 but no true_stats needed)
# ==============================================================================

class AugmentedDataset(Dataset):
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


def augmented_collate_fn(batch):
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
    return (temporal_batch, torch.tensor(label_list, dtype=torch.float32),
            torch.stack(baseline_list))


# ==============================================================================
# 3. Task-Driven Supervised Training (NO Reconstruction Loss!)
# ==============================================================================

class ClassificationHead(nn.Module):
    """Simple classification head for end-to-end training"""
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


def train_encoder_task_driven(encoder, train_loader, val_loader, baseline_dim, latent_dim, epochs=80):
    """
    KEY DIFFERENCE from Tab_Xv4:
    - NO stat_loss (reconstruction)
    - ONLY classification loss
    - Model learns what's useful for prediction, not what matches handcrafted features
    """
    print("  [Training] Task-driven learning (classification only, no reconstruction)...")

    classifier = ClassificationHead(baseline_dim + latent_dim).to(DEVICE)

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(classifier.parameters()),
        lr=0.001,
        weight_decay=1e-5  # Light regularization
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=7
    )

    criterion = nn.BCELoss()

    best_aupr = 0
    best_state = None
    patience = 15
    counter = 0

    for epoch in range(epochs):
        encoder.train()
        classifier.train()

        total_loss = 0
        num_batches = 0

        for t_data, labels, baseline_feats in train_loader:
            labels = labels.to(DEVICE)
            baseline_feats = baseline_feats.to(DEVICE)

            # Extract learned features
            learned_latent = encoder(t_data)  # (batch, latent_dim)

            # Augment baseline with learned features
            combined = torch.cat([baseline_feats, learned_latent], dim=1)

            # Classification loss ONLY
            preds = classifier(combined).squeeze(-1)
            loss = criterion(preds, labels)

            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(classifier.parameters()),
                max_norm=1.0
            )

            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches

        # Validation
        if (epoch + 1) % 2 == 0:
            encoder.eval()
            classifier.eval()
            all_preds, all_labels = [], []

            with torch.no_grad():
                for t_data, labels, baseline_feats in val_loader:
                    baseline_feats = baseline_feats.to(DEVICE)

                    learned_latent = encoder(t_data)
                    combined = torch.cat([baseline_feats, learned_latent], dim=1)
                    preds = classifier(combined).squeeze(-1)

                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.numpy())

            val_auc = roc_auc_score(all_labels, all_preds)
            val_aupr = average_precision_score(all_labels, all_preds)
            scheduler.step(val_aupr)

            print(f"    Epoch {epoch+1:3d} | Loss: {avg_loss:.4f} | Val AUC: {val_auc:.4f} | Val AUPR: {val_aupr:.4f}")

            if val_aupr > best_aupr:
                best_aupr = val_aupr
                best_state = copy.deepcopy(encoder.state_dict())
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    print(f"    Early stopping at epoch {epoch+1}")
                    break

    if best_state is not None:
        encoder.load_state_dict(best_state)

    print(f"  [Training] Completed. Best Val AUPR: {best_aupr:.4f}")
    return encoder, best_aupr


def extract_augmented_features(encoder, loader):
    """Extract baseline + learned features"""
    encoder.eval()
    all_features = []
    all_labels = []

    with torch.no_grad():
        for t_data, labels, baseline_feats in loader:
            learned_latent = encoder(t_data)
            learned_np = learned_latent.cpu().numpy()
            baseline_np = baseline_feats.numpy()

            # AUGMENT: Concatenate baseline + learned
            combined = np.hstack([baseline_np, learned_np])
            all_features.append(combined)
            all_labels.extend(labels.numpy())

    return np.vstack(all_features), np.array(all_labels)


# ==============================================================================
# 4. Main
# ==============================================================================

def main():
    print("="*80)
    print("Tab_Xv5: AUGMENT Strategy - Handcrafted + Learned Features")
    print("="*80)
    print("\nKey Insight: Don't replace handcrafted features, AUGMENT them!")
    print("  - Keep ALL baseline handcrafted features (proven to work)")
    print("  - Add complementary learned features (capture complex patterns)")
    print("  - No reconstruction loss (task-driven learning)")
    print("="*80)

    patients = load_and_prepare_patients()
    temporal_feats = get_all_temporal_features(patients)

    print(f"\nTemporal features: {len(temporal_feats)}")
    print(f"Learned latent dim: 32 (unconstrained)")

    metrics_xv5 = {k: [] for k in ['auc', 'auc_pr']}
    metrics_baseline = {k: [] for k in ['auc', 'auc_pr']}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    for fold, (train_full, test_p) in enumerate(trainTestPatients(patients, seed=xseed)):
        print(f"\n{'='*80}")
        print(f"Fold {fold}")
        print('='*80)

        train_p_obj, val_p_obj = split_patients_train_val(train_full, val_ratio=0.1, seed=42+fold)

        # Get baseline handcrafted features
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
        print(f"  Baseline features: {baseline_dim}")

        # Create datasets
        train_p = train_p_obj.patientList
        val_p = val_p_obj.patientList
        test_p_list = test_p.patientList

        train_ds = AugmentedDataset(train_p, temporal_feats, X_train_baseline)
        stats = train_ds.get_normalization_stats()
        val_ds = AugmentedDataset(val_p, temporal_feats, X_val_baseline, stats)
        test_ds = AugmentedDataset(test_p_list, temporal_feats, X_test_baseline, stats)

        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=augmented_collate_fn)
        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=augmented_collate_fn)
        test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=augmented_collate_fn)

        # Create encoder (more powerful than Tab_Xv4)
        latent_dim = 32
        encoder = TemporalFeatureEncoder(
            num_features=len(temporal_feats),
            hidden_dim=64,
            latent_dim=latent_dim,
            num_heads=4
        ).to(DEVICE)

        print(f"  Learned latent dim: {latent_dim}")
        print(f"  Total features: {baseline_dim} (baseline) + {latent_dim} (learned) = {baseline_dim + latent_dim}")

        tabpfn_params = {'device': 'cuda' if torch.cuda.is_available() else 'cpu'}

        # Train encoder (task-driven, no reconstruction)
        encoder, pretrain_aupr = train_encoder_task_driven(
            encoder, train_loader, val_loader, baseline_dim, latent_dim, epochs=80
        )

        # Evaluate
        print("\n  [Test Evaluation]")
        X_train_final, y_train_final = extract_augmented_features(encoder, train_loader)
        X_test_final, y_test_final = extract_augmented_features(encoder, test_loader)

        print(f"  Train features shape: {X_train_final.shape}")
        print(f"  Test features shape: {X_test_final.shape}")

        final_tabpfn = TabPFNClassifier(**tabpfn_params)
        final_tabpfn.fit(X_train_final, y_train_final)

        y_test_proba = final_tabpfn.predict_proba(X_test_final)[:, 1]
        prec, rec, _ = precision_recall_curve(y_test_final, y_test_proba)

        fold_auc = roc_auc_score(y_test_final, y_test_proba)
        fold_aupr = auc(rec, prec)

        metrics_xv5['auc'].append(fold_auc)
        metrics_xv5['auc_pr'].append(fold_aupr)

        fpr, tpr, _ = roc_curve(y_test_final, y_test_proba)
        ax1.plot(fpr, tpr, lw=2, label=f"Fold {fold} (AUC = {fold_auc:.3f})")

        print(f"  Tab_Xv5 AUC: {fold_auc:.4f} | AUPR: {fold_aupr:.4f}")

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

    ax1.set_title("Tab_Xv5: Baseline + Learned Features (AUGMENT)")
    ax2.set_title("Baseline (Handcrafted Only)")
    plt.tight_layout()
    plt.savefig("result/tab_xv5_vs_baseline.png", dpi=300)
    print("\nPlot saved to result/tab_xv5_vs_baseline.png")

    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)

    def print_stat(name, xv5_metrics, base_metrics):
        xv5_mean, xv5_std = np.mean(xv5_metrics), np.std(xv5_metrics)
        base_mean, base_std = np.mean(base_metrics), np.std(base_metrics)
        improvement = ((xv5_mean - base_mean) / base_mean) * 100
        symbol = "✓" if xv5_mean > base_mean else "✗"
        print(f"{name} | Tab_Xv5: {xv5_mean:.4f}±{xv5_std:.4f} vs Baseline: {base_mean:.4f}±{base_std:.4f} ({improvement:+.2f}%) {symbol}")

    print_stat("AUC   ", metrics_xv5['auc'], metrics_baseline['auc'])
    print_stat("AUC-PR", metrics_xv5['auc_pr'], metrics_baseline['auc_pr'])

    print("\n" + "="*80)
    print("Tab_Xv5 KEY IMPROVEMENTS:")
    print("✓ AUGMENTS handcrafted features (don't replace them!)")
    print("✓ More powerful encoder: Bi-GRU + Multi-head Attention")
    print("✓ Task-driven learning: Classification loss ONLY (no reconstruction)")
    print("✓ Unconstrained latent space (no Tanh bounds)")
    print("✓ Learns complementary features that handcrafted can't capture")
    print("="*80)

if __name__ == "__main__":
    main()
