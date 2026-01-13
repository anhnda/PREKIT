"""
Tab_Xv6: Rich Handcrafted Features + Learned Latent

CRITICAL INSIGHT from comparing Tab_RLv4 vs Baseline:
- Baseline uses: getMeasuresBetween(..., "last") → ONLY last value per feature
- Tab_RLv4 extracts: [last, mean, std, min, max, slope, count] per feature → 7x richer!

The SUCCESS of Tab_RLv4 comes from TWO things:
1. Rich handcrafted features (7 stats per temporal feature) instead of just "last"
2. Learned latent features on top

This version:
1. Extracts SAME extended handcrafted features as Tab_RLv4 (7 stats per temporal feature)
2. Adds learned latent features (32-dim) using Bi-GRU + Attention
3. Simple supervised training (no RL complexity)

Expected improvement:
- Extended handcrafted alone should boost performance
- Learned latent provides additional complementary patterns
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
# Static Features (like Tab_RLv4)
# ==============================================================================

FIXED_FEATURES = [
    "age", "gender", "race", "chronic_pulmonary_disease", "ckd_stage",
    "congestive_heart_failure", "dka_type", "history_aci", "history_ami",
    "hypertension", "liver_disease", "macroangiopathy", "malignant_cancer",
    "microangiopathy", "uti", "oasis", "saps2", "sofa",
    "mechanical_ventilation", "use_NaHCO3", "preiculos", "gcs_unable"
]

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
# Temporal Encoder
# ==============================================================================

class TemporalFeatureEncoder(nn.Module):
    """
    Bi-GRU + Multi-head Attention encoder for learning complementary features
    """
    def __init__(self, num_features, hidden_dim=64, latent_dim=32, num_heads=4):
        super().__init__()
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # Bi-directional GRU
        self.gru = nn.GRU(
            num_features,
            hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )

        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )

        # Learnable query
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim * 2))

        # Projection to latent space
        self.latent_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, latent_dim)
        )

    def forward(self, batch_data):
        values = batch_data['values'].to(DEVICE)
        masks = batch_data['masks'].to(DEVICE)
        lengths = batch_data['lengths'].to(DEVICE)

        batch_size = values.shape[0]
        masked_values = values * masks

        # Pack and process through GRU
        packed = nn.utils.rnn.pack_padded_sequence(
            masked_values,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )
        gru_out, _ = self.gru(packed)
        gru_out, _ = nn.utils.rnn.pad_packed_sequence(gru_out, batch_first=True)

        # Attention pooling
        query = self.query.expand(batch_size, -1, -1)
        attn_mask = (masks.sum(dim=-1) == 0)
        attn_out, _ = self.attention(query, gru_out, gru_out, key_padding_mask=attn_mask)

        context = attn_out.squeeze(1)
        latent = self.latent_head(context)

        return latent

# ==============================================================================
# Dataset with EXTENDED Handcrafted Feature Extraction
# ==============================================================================

class ExtendedDataset(Dataset):
    """
    Like Tab_RLv4: Extract 7 statistics per temporal feature
    """
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


def extended_collate_fn(batch):
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
# Extended Feature Extraction (SAME as Tab_RLv4!)
# ==============================================================================

def extract_extended_handcrafted_features(batch_data, static_data):
    """
    Extract 7 statistics per temporal feature:
    [last, mean, std, min, max, slope, count] for each feature

    This is THE KEY that makes Tab_RLv4 work!
    """
    vals = batch_data['values'].cpu().numpy()
    masks = batch_data['masks'].cpu().numpy()
    times = batch_data['times'].cpu().numpy()
    s_np = static_data.numpy()

    batch_features = []

    for i in range(len(vals)):
        patient_stats = []

        for f_idx in range(vals.shape[2]):
            f_vals = vals[i, :, f_idx]
            f_mask = masks[i, :, f_idx]
            f_times = times[i, :]
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

                if len(valid_vals) >= 2:
                    slope = (valid_vals[-1] - valid_vals[0]) / (valid_times[-1] - valid_times[0] + 1e-6)
                else:
                    slope = 0.0

                patient_stats.extend([last, mean_val, std_val, min_val, max_val, slope, count])
            else:
                patient_stats.extend([0.0] * 7)

        batch_features.append(patient_stats)

    extended_handcrafted = np.array(batch_features)

    # Combine: [Static + Extended Handcrafted Features]
    combined = np.hstack([s_np, extended_handcrafted])

    return combined


def extract_all_features(encoder, loader):
    """
    Extract: [Static + Extended Handcrafted (7 per temporal feat) + Learned Latent]
    """
    encoder.eval()
    all_features = []
    all_labels = []

    with torch.no_grad():
        for t_data, labels, s_data in loader:
            # Extended handcrafted features
            handcrafted_features = extract_extended_handcrafted_features(t_data, s_data)

            # Learned latent features
            learned_latent = encoder(t_data).cpu().numpy()

            # Combine ALL features
            combined = np.hstack([handcrafted_features, learned_latent])

            all_features.append(combined)
            all_labels.extend(labels.numpy())

    return np.vstack(all_features), np.array(all_labels)


# ==============================================================================
# Training
# ==============================================================================

class ClassificationHead(nn.Module):
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


def train_encoder(encoder, train_loader, val_loader, num_static_feats, num_temporal_feats, latent_dim, epochs=80):
    print("  [Training] Task-driven encoder with extended features...")

    # Feature dimensions:
    # Static: num_static_feats
    # Extended handcrafted: num_temporal_feats * 7
    # Learned latent: latent_dim
    total_dim = num_static_feats + num_temporal_feats * 7 + latent_dim

    classifier = ClassificationHead(total_dim).to(DEVICE)

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(classifier.parameters()),
        lr=0.001,
        weight_decay=1e-5
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

        for t_data, labels, s_data in train_loader:
            labels = labels.to(DEVICE)

            # Extract extended handcrafted features
            handcrafted_feats = extract_extended_handcrafted_features(t_data, s_data)
            handcrafted_feats = torch.tensor(handcrafted_feats, dtype=torch.float32).to(DEVICE)

            # Extract learned latent
            learned_latent = encoder(t_data)

            # Combine all features
            combined = torch.cat([handcrafted_feats, learned_latent], dim=1)

            # Classification loss
            preds = classifier(combined).squeeze(-1)
            loss = criterion(preds, labels)

            optimizer.zero_grad()
            loss.backward()
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
                for t_data, labels, s_data in val_loader:
                    handcrafted_feats = extract_extended_handcrafted_features(t_data, s_data)
                    handcrafted_feats = torch.tensor(handcrafted_feats, dtype=torch.float32).to(DEVICE)
                    learned_latent = encoder(t_data)
                    combined = torch.cat([handcrafted_feats, learned_latent], dim=1)
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
    return encoder


# ==============================================================================
# Main
# ==============================================================================

def main():
    print("="*80)
    print("Tab_Xv6: Extended Handcrafted Features + Learned Latent")
    print("="*80)
    print("\nKEY INSIGHT:")
    print("  Tab_RLv4 extracts 7 stats per temporal feature (not just 'last')!")
    print("  We do the SAME + add learned latent on top")
    print("="*80)

    patients = load_and_prepare_patients()
    temporal_feats = get_all_temporal_features(patients)

    # Static encoder
    static_encoder = SimpleStaticEncoder(FIXED_FEATURES)
    static_encoder.fit(patients.patientList)

    print(f"\nStatic features: {len(FIXED_FEATURES)}")
    print(f"Temporal features: {len(temporal_feats)}")
    print(f"Extended handcrafted: {len(temporal_feats)} x 7 = {len(temporal_feats)*7}")
    print(f"Learned latent: 32")
    print(f"Total: {len(FIXED_FEATURES)} + {len(temporal_feats)*7} + 32 = {len(FIXED_FEATURES) + len(temporal_feats)*7 + 32}")

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

        # Create datasets
        train_ds = ExtendedDataset(train_p, temporal_feats, static_encoder)
        stats = train_ds.get_normalization_stats()
        val_ds = ExtendedDataset(val_p, temporal_feats, static_encoder, stats)
        test_ds = ExtendedDataset(test_p_list, temporal_feats, static_encoder, stats)

        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=extended_collate_fn)
        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=extended_collate_fn)
        test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=extended_collate_fn)

        # Create encoder
        latent_dim = 32
        encoder = TemporalFeatureEncoder(
            num_features=len(temporal_feats),
            hidden_dim=64,
            latent_dim=latent_dim,
            num_heads=4
        ).to(DEVICE)

        tabpfn_params = {'device': 'cuda' if torch.cuda.is_available() else 'cpu'}

        # Train
        encoder = train_encoder(
            encoder, train_loader, val_loader,
            num_static_feats=len(FIXED_FEATURES),
            num_temporal_feats=len(temporal_feats),
            latent_dim=latent_dim,
            epochs=80
        )

        # Evaluate
        print("\n  [Test Evaluation]")
        X_train_final, y_train_final = extract_all_features(encoder, train_loader)
        X_test_final, y_test_final = extract_all_features(encoder, test_loader)

        print(f"  Train shape: {X_train_final.shape}")
        print(f"  Test shape: {X_test_final.shape}")

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

    ax1.set_title("Tab_Xv6: Extended Features + Learned Latent")
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
    print("Tab_Xv6 = Tab_RLv4 Strategy WITHOUT RL:")
    print("✓ Extended handcrafted: 7 stats per temporal feature")
    print("✓ Learned latent: 32-dim from Bi-GRU + Attention")
    print("✓ Simple supervised training (no RL complexity)")
    print("="*80)

if __name__ == "__main__":
    main()
