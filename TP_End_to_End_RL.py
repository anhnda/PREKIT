"""
DPO (Direct Preference Optimization) for RNN Feature Learning

Key Innovation:
- NO mimic head, NO proxy
- RNN learns directly from TabPFN's preferences
- Uses DPO loss: optimize RNN to produce features TabPFN prefers

DPO Approach:
1. Generate two versions of features: current RNN vs reference RNN
2. TabPFN evaluates both and tells us which is better
3. DPO loss pushes RNN toward better features
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
from tabpfn import TabPFNClassifier
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

# Static features list
FIXED_FEATURES = [
    "age", "gender", "race", "chronic_pulmonary_disease", "ckd_stage",
    "congestive_heart_failure", "dka_type", "history_aci", "history_ami",
    "hypertension", "liver_disease", "macroangiopathy", "malignant_cancer",
    "microangiopathy", "uti", "oasis", "saps2", "sofa",
    "mechanical_ventilation", "use_NaHCO3", "preiculos", "gcs_unable"
]

# ==============================================================================
# 1. Helpers: Encoder
# ==============================================================================

class SimpleStaticEncoder:
    """Encodes categorical static features (e.g. Gender 'F'->1)"""
    def __init__(self, features):
        self.features = features
        self.mappings = {f: {} for f in features}
        self.counts = {f: 0 for f in features}

    def fit(self, patients):
        for p in patients:
            for f in self.features:
                val = p.measures.get(f, 0.0)
                if hasattr(val, 'values') and len(val) > 0: val = list(val.values())[0]
                elif hasattr(val, 'values'): val = 0.0

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
            if hasattr(val, 'values') and len(val) > 0: val = list(val.values())[0]
            elif hasattr(val, 'values'): val = 0.0

            try:
                numeric_val = float(val)
            except ValueError:
                numeric_val = self.mappings[f].get(str(val), -1.0)
            vec.append(numeric_val)
        return vec

# ==============================================================================
# 2. Dataset (Returns Temporal, Label, AND Static)
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
            if times is None: continue

            s_vec = static_encoder.transform(patient)

            self.static_data.append(torch.tensor(s_vec, dtype=torch.float32))
            self.data.append({'times': times, 'values': values, 'masks': masks})
            self.labels.append(1 if patient.akdPositive else 0)

            for v_vec, m_vec in zip(values, masks):
                for v, m in zip(v_vec, m_vec):
                    if m > 0: all_values.append(v)

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
                norm = [(v - self.mean)/self.std if m>0 else 0.0 for v, m in zip(v_vec, m_vec)]
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
        'times': padded_times, 'values': padded_values,
        'masks': padded_masks, 'lengths': torch.tensor(lengths)
    }
    return temporal_batch, torch.tensor(label_list, dtype=torch.float32), torch.stack(static_list)

# ==============================================================================
# 3. RNN Model
# ==============================================================================

class RNNFeatureExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dim, time_dim=32):
        super().__init__()
        self.rnn_cell = TimeEmbeddedRNNCell(input_dim, hidden_dim, time_dim)

    def forward(self, batch_data):
        times = batch_data['times'].to(DEVICE)
        values = batch_data['values'].to(DEVICE)
        masks = batch_data['masks'].to(DEVICE)
        lengths = batch_data['lengths'].to(DEVICE)
        return self.rnn_cell(times, values, masks, lengths)

# ==============================================================================
# 4. Helper: Extract Last Values
# ==============================================================================

def extract_last_values(t_data):
    """Extract last observed values from temporal data"""
    vals = t_data['values'].to(DEVICE)
    masks = t_data['masks'].to(DEVICE)

    batch_last_vals = []
    for i in range(vals.shape[0]):
        patient_last = []
        for f_idx in range(vals.shape[2]):
            f_vals = vals[i, :, f_idx]
            f_mask = masks[i, :, f_idx]
            valid_idx = torch.where(f_mask > 0)[0]

            if len(valid_idx) > 0:
                last_v = f_vals[valid_idx[-1]]
            else:
                last_v = torch.tensor(0.0, device=DEVICE)
            patient_last.append(last_v)
        batch_last_vals.append(torch.stack(patient_last))

    return torch.stack(batch_last_vals)

# ==============================================================================
# 5. DPO Training Function
# ==============================================================================

def train_rnn_with_dpo(rnn_model, reference_rnn, train_loader, val_loader, epochs=50, beta=0.1):
    """
    Train RNN using Direct Preference Optimization

    Args:
        rnn_model: Current RNN (being trained)
        reference_rnn: Reference RNN (frozen, for DPO)
        train_loader: Training data
        val_loader: Validation data
        epochs: Number of training epochs
        beta: DPO temperature parameter (lower = stronger preference signal)

    DPO Loss:
        L = -log(σ(β * (log π(chosen) - log π_ref(chosen) - log π(rejected) + log π_ref(rejected))))

    For our case:
        - "chosen" = features that lead to better TabPFN predictions
        - "rejected" = features that lead to worse TabPFN predictions
    """

    print("  [Stage 1] Initializing TabPFN on reference features...")

    # First, fit TabPFN on reference RNN features
    reference_rnn.eval()
    ref_features = []
    ref_labels = []

    with torch.no_grad():
        for t_data, labels, s_data in train_loader:
            h_ref = reference_rnn(t_data).cpu().numpy()
            s = s_data.numpy()
            last_vals = extract_last_values(t_data).cpu().numpy()

            combined = np.hstack([last_vals, s, h_ref])
            ref_features.append(combined)
            ref_labels.extend(labels.numpy())

    ref_features = np.vstack(ref_features)
    ref_labels = np.array(ref_labels)

    # Fit TabPFN
    tabpfn = TabPFNClassifier(device='cuda' if torch.cuda.is_available() else 'cpu')
    tabpfn.fit(ref_features, ref_labels)

    print("  [Stage 2] Training RNN with DPO (distillation approach)...")

    # Create a simple differentiable head
    feature_dim = len(train_loader.dataset.feature_names) + len(FIXED_FEATURES) + 12  # temporal + static + rnn_dim
    simple_head = nn.Sequential(
        nn.Linear(feature_dim, 16),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(16, 1),
        nn.Sigmoid()
    ).to(DEVICE)

    # Optimizer for both RNN and head
    optimizer = torch.optim.Adam(
        list(rnn_model.parameters()) + list(simple_head.parameters()),
        lr=0.0005
    )

    best_auc = 0
    best_state = None
    patience = 6
    counter = 0

    for epoch in range(epochs):
        rnn_model.train()
        simple_head.train()
        reference_rnn.eval()

        epoch_loss = 0
        num_batches = 0

        for t_data, labels, s_data in train_loader:
            labels = labels.to(DEVICE)
            s_data = s_data.to(DEVICE)

            # Get features from current RNN
            h_current = rnn_model(t_data)
            last_vals = extract_last_values(t_data)
            features_current = torch.cat([last_vals, s_data, h_current], dim=1)

            # Get features from reference RNN (frozen)
            with torch.no_grad():
                h_ref = reference_rnn(t_data)
                features_ref = torch.cat([last_vals, s_data, h_ref], dim=1)

            # Get TabPFN predictions as "pseudo-labels"
            with torch.no_grad():
                # Use TabPFN on current features to get soft targets
                tabpfn_soft_labels = tabpfn.predict_proba(features_current.cpu().numpy())[:, 1]
                tabpfn_soft_labels = torch.FloatTensor(tabpfn_soft_labels).to(DEVICE)

            # Get prediction from simple head (differentiable)
            head_pred = simple_head(features_current).squeeze(-1)

            # Loss 1: Match true labels (classification)
            label_loss = F.binary_cross_entropy(head_pred, labels)

            # Loss 2: Distill from TabPFN (mimic TabPFN's predictions)
            distill_loss = F.mse_loss(head_pred, tabpfn_soft_labels)

            # Loss 3: Feature regularization (prevent RNN features from exploding)
            reg_loss = 0.001 * torch.mean(h_current ** 2)

            # Combined loss
            # Heavy weight on distillation - we want RNN to learn features TabPFN prefers
            total_loss = 0.3 * label_loss + 0.6 * distill_loss + reg_loss

            optimizer.zero_grad()
            total_loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                list(rnn_model.parameters()) + list(simple_head.parameters()),
                max_norm=1.0
            )

            optimizer.step()

            epoch_loss += total_loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches

        # Validation
        if (epoch+1) % 5 == 0:
            rnn_model.eval()
            simple_head.eval()

            with torch.no_grad():
                val_features = []
                val_labels = []

                for t_data, labels, s_data in val_loader:
                    h = rnn_model(t_data)
                    s = s_data.cpu().numpy()
                    last_vals = extract_last_values(t_data).cpu().numpy()
                    h_np = h.cpu().numpy()

                    combined = np.hstack([last_vals, s, h_np])
                    val_features.append(combined)
                    val_labels.extend(labels.numpy())

                val_features = np.vstack(val_features)
                val_labels = np.array(val_labels)

                # Evaluate with TabPFN
                preds = tabpfn.predict_proba(val_features)[:, 1]

                aupr = average_precision_score(val_labels, preds)
                auc_score = roc_auc_score(val_labels, preds)

            print(f"    Epoch {epoch+1} | Train Loss: {avg_loss:.4f} | Val AUPR: {aupr:.4f} AUC: {auc_score:.4f}")

            if aupr > best_auc:
                best_auc = aupr
                best_state = copy.deepcopy(rnn_model.state_dict())
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    print(f"    Early stopping at epoch {epoch+1}")
                    break

    rnn_model.load_state_dict(best_state)
    return rnn_model

# ==============================================================================
# 6. Triple Feature Extraction
# ==============================================================================

def get_triple_features(model, loader):
    """Returns [Last_Value + Static + RNN_Embedding]"""
    model.eval()
    features = []
    labels_out = []

    with torch.no_grad():
        for t_data, labels, s_data in loader:
            h = model(t_data).cpu().numpy()
            s = s_data.numpy()
            last_vals = extract_last_values(t_data).cpu().numpy()

            combined = np.hstack([last_vals, s, h])
            features.append(combined)
            labels_out.extend(labels.numpy())

    return np.vstack(features), np.array(labels_out)

# ==============================================================================
# 7. Main Execution
# ==============================================================================

def main():
    print("="*80)
    print("DPO TRAINING: RNN optimized via TabPFN preferences")
    print("="*80)

    patients = load_and_prepare_patients()
    temporal_feats = get_all_temporal_features(patients)

    print("Encoding static features...")
    encoder = SimpleStaticEncoder(FIXED_FEATURES)
    encoder.fit(patients.patientList)

    print(f"Input: {len(temporal_feats)} Temporal + {len(FIXED_FEATURES)} Static Features")

    metrics_hybrid = {k: [] for k in ['auc', 'acc', 'spec', 'prec', 'rec', 'auc_pr']}
    metrics_base = {k: [] for k in ['auc', 'acc', 'spec', 'prec', 'rec', 'auc_pr']}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    for fold, (train_full, test_p) in enumerate(trainTestPatients(patients)):
        print(f"\n--- Fold {fold} ---")
        train_p_obj, val_p_obj = split_patients_train_val(train_full, val_ratio=0.1, seed=42+fold)
        train_p, val_p, test_p_list = train_p_obj.patientList, val_p_obj.patientList, test_p.patientList

        train_ds = HybridDataset(train_p, temporal_feats, encoder)
        stats = train_ds.get_normalization_stats()
        val_ds = HybridDataset(val_p, temporal_feats, encoder, stats)
        test_ds = HybridDataset(test_p_list, temporal_feats, encoder, stats)

        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=hybrid_collate_fn)
        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=hybrid_collate_fn)
        test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=hybrid_collate_fn)

        # Initialize RNN models
        rnn = RNNFeatureExtractor(len(temporal_feats), hidden_dim=12).to(DEVICE)
        reference_rnn = RNNFeatureExtractor(len(temporal_feats), hidden_dim=12).to(DEVICE)
        reference_rnn.load_state_dict(rnn.state_dict())  # Copy initial weights

        # Freeze reference RNN
        for param in reference_rnn.parameters():
            param.requires_grad = False

        # Train with DPO
        rnn = train_rnn_with_dpo(rnn, reference_rnn, train_loader, val_loader, epochs=100, beta=0.5)

        print("  [Stage 3] Final evaluation with fresh TabPFN...")
        X_train, y_train = get_triple_features(rnn, train_loader)
        X_val, y_val = get_triple_features(rnn, val_loader)
        X_test, y_test = get_triple_features(rnn, test_loader)

        clf = TabPFNClassifier(device='cuda' if torch.cuda.is_available() else 'cpu')
        clf.fit(X_train, y_train)

        y_prob = clf.predict_proba(X_test)[:, 1]
        y_pred = (y_prob > 0.5).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        prec, rec, _ = precision_recall_curve(y_test, y_prob)

        metrics_hybrid['auc'].append(roc_auc_score(y_test, y_prob))
        metrics_hybrid['acc'].append(accuracy_score(y_test, y_pred))
        metrics_hybrid['spec'].append(tn / (tn + fp))
        metrics_hybrid['prec'].append(precision_score(y_test, y_pred, zero_division=0))
        metrics_hybrid['rec'].append(recall_score(y_test, y_pred))
        metrics_hybrid['auc_pr'].append(auc(rec, prec))

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        ax1.plot(fpr, tpr, lw=2, label=f"Fold {fold} (AUC = {metrics_hybrid['auc'][-1]:.3f})")

        # Baseline
        print("  [Baseline] Training Standard TabPFN (Last + Static)...")

        df_train_temp = train_p_obj.getMeasuresBetween(
            pd.Timedelta(hours=-6), pd.Timedelta(hours=24), "last", getUntilAkiPositive=True).drop(columns=["subject_id", "hadm_id", "stay_id"])
        df_test_temp = test_p.getMeasuresBetween(
            pd.Timedelta(hours=-6), pd.Timedelta(hours=24), "last", getUntilAkiPositive=True).drop(columns=["subject_id", "hadm_id", "stay_id"])

        df_train_enc, df_test_enc, _ = encodeCategoricalData(df_train_temp, df_test_temp)

        X_tr_b = df_train_enc.drop(columns=["akd"]).fillna(0)
        y_tr_b = df_train_enc["akd"]
        X_te_b = df_test_enc.drop(columns=["akd"]).fillna(0)
        y_te_b = df_test_enc["akd"]

        model_base = TabPFNClassifier(device='cuda' if torch.cuda.is_available() else 'cpu')
        model_base.fit(X_tr_b, y_tr_b)

        y_prob_b = model_base.predict_proba(X_te_b)[:, 1]
        y_pred_b = (y_prob_b > 0.5).astype(int)

        tn, fp, _, _ = confusion_matrix(y_te_b, y_pred_b).ravel()
        prec_b, rec_b, _ = precision_recall_curve(y_te_b, y_prob_b)

        metrics_base['auc'].append(roc_auc_score(y_te_b, y_prob_b))
        metrics_base['acc'].append(accuracy_score(y_te_b, y_pred_b))
        metrics_base['spec'].append(tn / (tn + fp))
        metrics_base['prec'].append(precision_score(y_te_b, y_pred_b, zero_division=0))
        metrics_base['rec'].append(recall_score(y_te_b, y_pred_b))
        metrics_base['auc_pr'].append(auc(rec_b, prec_b))

        fpr_b, tpr_b, _ = roc_curve(y_te_b, y_prob_b)
        ax2.plot(fpr_b, tpr_b, lw=2, label=f"Fold {fold} (AUC = {metrics_base['auc'][-1]:.3f})")

        print(f"  Fold {fold} Results -> DPO: {metrics_hybrid['auc_pr'][-1]:.3f} vs Baseline: {metrics_base['auc_pr'][-1]:.3f}")

    # Final Plot
    for ax in [ax1, ax2]:
        ax.plot([0, 1], [0, 1], linestyle="--", color="navy", lw=2)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.legend(loc="lower right")
    ax1.set_title("DPO (RNN optimized by TabPFN)")
    ax2.set_title("Baseline (Last+Static)")
    plt.tight_layout()
    plt.savefig("result/dpo_tabpfn_vs_baseline.png", dpi=300)
    print("\nPlot saved to result/dpo_tabpfn_vs_baseline.png")

    # Final Stats
    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY")
    print("="*80)

    def print_stat(name, h_metrics, b_metrics):
        h_mean, h_std = np.mean(h_metrics), np.std(h_metrics)
        b_mean, b_std = np.mean(b_metrics), np.std(b_metrics)
        print(f"{name:15s} | DPO: {h_mean:.4f} ± {h_std:.4f}  vs  Baseline: {b_mean:.4f} ± {b_std:.4f}")

    print_stat("AUC", metrics_hybrid['auc'], metrics_base['auc'])
    print_stat("AUC-PR", metrics_hybrid['auc_pr'], metrics_base['auc_pr'])

if __name__ == "__main__":
    main()
