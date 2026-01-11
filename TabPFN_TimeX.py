"""
TRIPLE HYBRID MODEL: [Learned Trend + Last Value + Static Context] -> TabPFN
vs
STRONG BASELINE: [Last Value + Static Context] -> TabPFN

Features:
- Categorical Encoding (for Gender/Race)
- Gated RNN Pre-training
- Triple Feature Concatenation
- Full Metrics & Plots
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
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
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
    "mechanical_ventilation", "use_NaHCO3", "preiculos", "gcs_unable", "egfr"
]

# ==============================================================================
# 1. Helpers: Encoder & Gated Head
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

class GatedDecisionHead(nn.Module):
    """Mimics Tree logic for RNN pre-training"""
    def __init__(self, input_dim, hidden_dim=32, dropout=0.3):
        super(GatedDecisionHead, self).__init__()
        self.gate = nn.Sequential(nn.Linear(input_dim, input_dim), nn.Sigmoid())
        self.fc1 = nn.Linear(input_dim, hidden_dim * 2) 
        self.fc2 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.dropout = nn.Dropout(dropout)
        self.final = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        mask = self.gate(x)
        x = x * mask 
        out = self.fc1(x)
        out = F.glu(out, dim=-1)
        out = self.dropout(out)
        residual = out
        out = self.fc2(out)
        out = F.glu(out, dim=-1)
        out = out + residual
        return torch.sigmoid(self.final(out))

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
        # Support both list of patients or Patients object
        patient_list = patients.patientList if hasattr(patients, 'patientList') else patients
        
        for patient in patient_list:
            times, values, masks = extract_temporal_data(patient, feature_names)
            if times is None: continue
            
            # Encode Static Features
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
# 3. RNN Model & Pre-training
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

def train_rnn_extractor(model, train_loader, val_loader, criterion, optimizer, epochs=50):
    rnn_dim = model.rnn_cell.hidden_dim
    static_dim = len(FIXED_FEATURES)
    
    # Pre-train using [RNN + Static] -> Gated Head
    temp_head = GatedDecisionHead(input_dim=rnn_dim + static_dim).to(DEVICE)
    full_optimizer = torch.optim.Adam(list(model.parameters()) + list(temp_head.parameters()), lr=0.001)
    
    best_auc = 0
    best_state = None
    patience = 6
    counter = 0
    
    print("  [Stage 1] Pre-training RNN with Gated Head...")
    for epoch in range(epochs):
        model.train()
        temp_head.train()
        
        for t_data, labels, s_data in train_loader: 
            labels = labels.to(DEVICE)
            s_data = s_data.to(DEVICE)
            h = model(t_data)
            combined = torch.cat([h, s_data], dim=1)
            preds = temp_head(combined).squeeze(-1)
            loss = criterion(preds, labels)
            full_optimizer.zero_grad()
            loss.backward()
            full_optimizer.step()
            
        if (epoch+1) % 5 == 0:
            model.eval()
            temp_head.eval()
            all_preds, all_lbls = [], []
            with torch.no_grad():
                for t_data, labels, s_data in val_loader:
                    s_data = s_data.to(DEVICE)
                    h = model(t_data)
                    combined = torch.cat([h, s_data], dim=1)
                    preds = temp_head(combined).squeeze(-1)
                    all_preds.extend(preds.cpu().numpy())
                    all_lbls.extend(labels.cpu().numpy())
            
            auc_val = roc_auc_score(all_lbls, all_preds)
            print(f"    Epoch {epoch+1} Val AUC: {auc_val:.4f}")
            
            if auc_val > best_auc:
                best_auc = auc_val
                best_state = copy.deepcopy(model.state_dict())
                counter = 0
            else:
                counter += 1
                if counter >= patience: break
                
    model.load_state_dict(best_state)
    return model

# ==============================================================================
# 4. Triple Feature Extraction (The "Secret Sauce")
# ==============================================================================

def get_triple_features(model, loader):
    """Returns [RNN_Embedding (128) + Last_Value (25) + Static (23)]"""
    model.eval()
    features = []
    labels_out = []
    
    with torch.no_grad():
        for t_data, labels, s_data in loader:
            # 1. Get RNN Embedding (128 dims)
            h = model(t_data).cpu().numpy()
            
            # 2. Get Static Features (23 dims)
            s = s_data.numpy()
            
            # 3. Extract "Last Values" manually (25 dims)
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
                        last_v = 0.0 # Mean imputation
                    patient_last.append(last_v)
                batch_last_vals.append(patient_last)
            
            last_vals_arr = np.array(batch_last_vals)
            
            # 4. TRIPLE CONCATENATION
            combined = np.hstack([h, last_vals_arr, s])
            
            features.append(combined)
            labels_out.extend(labels.numpy())
            
    return np.vstack(features), np.array(labels_out)

# ==============================================================================
# 5. Main Execution
# ==============================================================================

def main():
    print("="*80)
    print("TRIPLE HYBRID (Learned + Last + Static) vs BASELINE (Last + Static)")
    print("="*80)
    
    patients = load_and_prepare_patients()
    temporal_feats = get_all_temporal_features(patients)
    
    print("Encoding static features...")
    encoder = SimpleStaticEncoder(FIXED_FEATURES)
    encoder.fit(patients.patientList)
    
    print(f"Input: {len(temporal_feats)} Temporal + {len(FIXED_FEATURES)} Static Features")
    
    # Store metrics for Hybrid
    metrics_hybrid = {k: [] for k in ['auc', 'acc', 'spec', 'prec', 'rec', 'auc_pr']}
    # Store metrics for Baseline
    metrics_base = {k: [] for k in ['auc', 'acc', 'spec', 'prec', 'rec', 'auc_pr']}
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    for fold, (train_full, test_p) in enumerate(trainTestPatients(patients)):
        print(f"\n--- Fold {fold} ---")
        train_p_obj, val_p_obj = split_patients_train_val(train_full, val_ratio=0.1, seed=42+fold)
        train_p, val_p, test_p_list = train_p_obj.patientList, val_p_obj.patientList, test_p.patientList
        
        # 1. Hybrid Data Setup
        train_ds = HybridDataset(train_p, temporal_feats, encoder)
        stats = train_ds.get_normalization_stats()
        val_ds = HybridDataset(val_p, temporal_feats, encoder, stats)
        test_ds = HybridDataset(test_p_list, temporal_feats, encoder, stats)
        
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=hybrid_collate_fn)
        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=hybrid_collate_fn)
        test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=hybrid_collate_fn)
        
        # 2. Stage 1: RNN Training
        rnn = RNNFeatureExtractor(len(temporal_feats), hidden_dim=48).to(DEVICE)
        opt = torch.optim.Adam(rnn.parameters(), lr=0.001)
        rnn = train_rnn_extractor(rnn, train_loader, val_loader, nn.BCELoss(), opt)
        
        # 3. Stage 2: Triple Feature Fusion
        print("  [Stage 2] Extracting Triple Features...")
        X_train, y_train = get_triple_features(rnn, train_loader)
        X_val, y_val = get_triple_features(rnn, val_loader)
        X_test, y_test = get_triple_features(rnn, test_loader)
        
        # 4. Stage 3: TabPFN Training
        ratio = np.sum(y_train==0) / (np.sum(y_train==1) + 1e-6)
        
        clf = TabPFNClassifier(device='cuda' if torch.cuda.is_available() else 'cpu')

        clf.fit(X_train, y_train)
        
        # 5. Hybrid Evaluation
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
        
        # ======================================================================
        # BASELINE: Standard TabPFN (Last Values + Static)
        # ======================================================================
        print("  [Baseline] Training Standard TabPFN (Last + Static)...")
        
        # Extract "Last Values"
        df_train_temp = train_p_obj.getMeasuresBetween(
            pd.Timedelta(hours=-6), pd.Timedelta(hours=24), "last", getUntilAkiPositive=True).drop(columns=["subject_id", "hadm_id", "stay_id"])
        df_val_temp = val_p_obj.getMeasuresBetween(
            pd.Timedelta(hours=-6), pd.Timedelta(hours=24), "last", getUntilAkiPositive=True).drop(columns=["subject_id", "hadm_id", "stay_id"])
        df_test_temp = test_p.getMeasuresBetween(
            pd.Timedelta(hours=-6), pd.Timedelta(hours=24), "last", getUntilAkiPositive=True).drop(columns=["subject_id", "hadm_id", "stay_id"])
            
        # Encode
        df_train_enc, df_val_enc, _ = encodeCategoricalData(df_train_temp, df_val_temp)
        df_train_enc, df_test_enc, _ = encodeCategoricalData(df_train_temp, df_test_temp)
        
        X_tr_b = df_train_enc.drop(columns=["akd"]).fillna(0)
        y_tr_b = df_train_enc["akd"]
        X_te_b = df_test_enc.drop(columns=["akd"]).fillna(0)
        y_te_b = df_test_enc["akd"]
        
        model_base  = TabPFNClassifier(device='cuda' if torch.cuda.is_available() else 'cpu')

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
        
        print(f"  Fold {fold} Results -> Hybrid: {metrics_hybrid['auc'][-1]:.3f} vs Baseline: {metrics_base['auc'][-1]:.3f}")

    # Final Plot Config
    
    for ax in [ax1, ax2]:
        ax.plot([0, 1], [0, 1], linestyle="--", color="navy", lw=2)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.legend(loc="lower right")
    ax1.set_title("Triple Hybrid (RNN+Last+Static)")
    ax2.set_title("Baseline (Last+Static)")
    plt.tight_layout()
    plt.savefig("result/triple_hybrid_vs_baseline.png", dpi=300)
    print("\nPlot saved to result/triple_hybrid_vs_baseline.png")

    # Final Stats
    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY")
    print("="*80)
    
    def print_stat(name, h_metrics, b_metrics):
        h_mean, h_std = np.mean(h_metrics), np.std(h_metrics)
        b_mean, b_std = np.mean(b_metrics), np.std(b_metrics)
        print(f"{name:15s} | Hybrid: {h_mean:.4f} ± {h_std:.4f}  vs  Baseline: {b_mean:.4f} ± {b_std:.4f}")
        
    print_stat("AUC", metrics_hybrid['auc'], metrics_base['auc'])
    print_stat("AUC-PR", metrics_hybrid['auc_pr'], metrics_base['auc_pr'])
    print_stat("Accuracy", metrics_hybrid['acc'], metrics_base['acc'])
    print_stat("Specificity", metrics_hybrid['spec'], metrics_base['spec'])
    print_stat("Precision", metrics_hybrid['prec'], metrics_base['prec'])
    print_stat("Recall", metrics_hybrid['rec'], metrics_base['rec'])

if __name__ == "__main__":
    main()

