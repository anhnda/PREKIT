"""
Tab_Xv5: Hybrid Explicit-Interaction Architecture

Philosophy:
1. Don't force a Neural Network to learn basic math (Mean, Max, Slope). 
   Calculate these explicitly on CPU (100% accuracy, 0 params).
2. Use Neural Network ONLY for what stats miss: 
   - Cross-feature interactions (Transformer)
   - Complex temporal motifs (CNN)
   
Input to TabPFN: [Static_Baseline] + [Explicit_Stats] + [Learned_Interaction_Z]
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
    confusion_matrix
)

# Disable TabPFN telemetry
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

# --- Seeding ---
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

# --- Path Setup ---
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
# 1. Feature Interaction Encoder (The "Soft" Path)
# ==============================================================================
# ==============================================================================
# 1. Feature Interaction Encoder (The "Soft" Path)
# ==============================================================================

class FeatureInteractionEncoder(nn.Module):
    def __init__(self, num_features, hidden_dim=64, nhead=4, latent_dim=16):
        super().__init__()
        self.num_features = num_features
        self.hidden_dim = hidden_dim

        # A. Temporal Extractor (CNN per feature)
        self.conv1 = nn.Conv1d(1, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm1d(hidden_dim)

        # B. Feature Identity Embedding
        self.feature_id_embedding = nn.Parameter(torch.randn(1, num_features, hidden_dim))

        # C. Cross-Feature Interaction (Transformer)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=nhead, 
            dim_feedforward=hidden_dim*2,
            batch_first=True,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # D. Patient Aggregation
        self.attention_pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # E. Final Projection
        self.fc_out = nn.Linear(hidden_dim, latent_dim)

    def forward(self, batch_data):
        values = batch_data['values'].to(DEVICE) 
        masks = batch_data['masks'].to(DEVICE)   
        
        # FIX: Rename F -> n_feats so we don't overwrite torch.nn.functional (F)
        B, L, n_feats = values.size()
        H = self.hidden_dim

        # --- Step 1: Per-Feature Temporal Extraction (CNN) ---
        # Reshape to [B*n_feats, 1, L]
        x = values.permute(0, 2, 1).reshape(B * n_feats, 1, L)
        m = masks.permute(0, 2, 1).reshape(B * n_feats, 1, L)
        
        # Now F.relu works because F refers to the module, not the integer size
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x)) 
        x = self.bn(x)
        
        # Mask out invalid time steps
        x = x * m 
        
        # Max Pooling over time
        x_summ = F.max_pool1d(x, kernel_size=L).squeeze(-1)
        
        # Reshape back to [B, n_feats, H]
        x_summ = x_summ.view(B, n_feats, H)

        # --- Step 2: Cross-Feature Interaction (Transformer) ---
        x_summ = x_summ + self.feature_id_embedding
        x_interact = self.transformer(x_summ) 

        # --- Step 3: Aggregation to Patient Vector ---
        attn_weights = self.attention_pool(x_interact)
        attn_weights = F.softmax(attn_weights, dim=1) 
        
        z_patient = torch.sum(x_interact * attn_weights, dim=1) 
        z = self.fc_out(z_patient)
        
        return z

# ==============================================================================
# 2. Dataset (The "Hard" Path + Data Prep)
# ==============================================================================

class HybridDatasetV5(Dataset):
    def __init__(self, patients, feature_names, baseline_features_df, normalization_stats=None):
        self.data = []
        self.labels = []
        self.baseline_features = []
        self.explicit_stats = [] 
        self.feature_names = feature_names

        all_values = []
        patient_list = patients.patientList if hasattr(patients, 'patientList') else patients

        # 1. First Pass: Collect raw data for normalization
        raw_cache = []
        for idx, patient in enumerate(patient_list):
            times, values, masks = extract_temporal_data(patient, feature_names)
            if times is None: continue
            
            # Store raw for second pass
            raw_cache.append({
                'times': times, 'values': values, 'masks': masks,
                'idx': idx, 'label': 1 if patient.akdPositive else 0
            })
            
            # Collect values for stats
            for v_vec, m_vec in zip(values, masks):
                for v, m in zip(v_vec, m_vec):
                    if m > 0: all_values.append(v)

        # 2. Calculate Global Normalization Stats
        if normalization_stats is None:
            all_values = np.array(all_values)
            self.mean = np.mean(all_values) if len(all_values) > 0 else 0.0
            self.std = np.std(all_values) if len(all_values) > 0 else 1.0
        else:
            self.mean = normalization_stats['mean']
            self.std = normalization_stats['std']

        # 3. Second Pass: Normalize & Compute Explicit Stats
        for item in raw_cache:
            # Normalize Values (Z-score)
            norm_values = []
            for v_vec, m_vec in zip(item['values'], item['masks']):
                norm = [(v - self.mean)/self.std if m>0 else 0.0 for v, m in zip(v_vec, m_vec)]
                norm_values.append(norm)

            # --- EXPLICIT STATS COMPUTATION (CPU) ---
            # 7 stats per feature: Last, Mean, Std, Min, Max, Slope, Count
            handcrafted = self._compute_handcrafted_features(norm_values, item['masks'], item['times'])
            self.explicit_stats.append(torch.tensor(handcrafted, dtype=torch.float32))

            # Store Temporal Data for NN
            self.data.append({
                'times': torch.tensor(item['times'], dtype=torch.float32),
                'values': torch.tensor(norm_values, dtype=torch.float32),
                'masks': torch.tensor(item['masks'], dtype=torch.float32)
            })
            
            # Store Baseline Static
            baseline_vec = baseline_features_df.iloc[item['idx']].values
            self.baseline_features.append(torch.tensor(baseline_vec, dtype=torch.float32))
            
            self.labels.append(item['label'])

    def _compute_handcrafted_features(self, norm_values, masks, times):
        stats_list = []
        num_feats = len(norm_values[0])
        
        for f_idx in range(num_feats):
            # Extract valid series for this feature
            f_vals = np.array([row[f_idx] for row in norm_values])
            f_mask = np.array([row[f_idx] for row in masks])
            f_times = np.array(times)
            
            valid_idx = np.where(f_mask > 0)[0]
            
            if len(valid_idx) > 0:
                vals = f_vals[valid_idx]
                ts = f_times[valid_idx]
                
                s_last = vals[-1]
                s_mean = np.mean(vals)
                s_std = np.std(vals) if len(vals) > 1 else 0.0
                s_min = np.min(vals)
                s_max = np.max(vals)
                s_count = len(vals)
                
                if len(vals) >= 2:
                    s_slope = (vals[-1] - vals[0]) / (ts[-1] - ts[0] + 1e-6)
                else:
                    s_slope = 0.0
                
                # Transformations for stability
                # Count: log scale
                s_count = np.log1p(s_count) 
                # Slope: tanh to clip extreme outliers
                s_slope = np.tanh(s_slope)
                
                stats_list.extend([s_last, s_mean, s_std, s_min, s_max, s_slope, s_count])
            else:
                # Feature completely missing for this patient
                stats_list.extend([0.0] * 7)
                
        return stats_list

    def get_normalization_stats(self): return {'mean': self.mean, 'std': self.std}
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        return (self.data[idx], self.labels[idx], self.baseline_features[idx], self.explicit_stats[idx])


def hybrid_collate_fn(batch):
    data_list, label_list, base_list, stat_list = zip(*batch)
    lengths = [len(d['times']) for d in data_list]
    max_len = max(lengths)
    feat_dim = data_list[0]['values'].shape[-1]
    
    # Pad Temporal Data
    padded_values = torch.zeros(len(batch), max_len, feat_dim)
    padded_masks = torch.zeros(len(batch), max_len, feat_dim)
    
    for i, d in enumerate(data_list):
        l = lengths[i]
        padded_values[i, :l] = d['values']
        padded_masks[i, :l] = d['masks']
        
    temporal_batch = {
        'values': padded_values, 
        'masks': padded_masks, 
        'lengths': torch.tensor(lengths)
    }
    
    return (temporal_batch, 
            torch.tensor(label_list, dtype=torch.float32), 
            torch.stack(base_list), 
            torch.stack(stat_list))

# ==============================================================================
# 3. Training Utilities
# ==============================================================================

class AuxiliaryHead(nn.Module):
    """Simple MLP to supervise the Encoder."""
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    def forward(self, x): return self.net(x).squeeze(-1)

def train_hybrid_encoder(encoder, train_loader, val_loader, epochs=30):
    print("  [Training] Hybrid Interaction Encoder...")
    
    # We train using [Learned Z + Explicit Stats] to predict the label.
    # We deliberately EXCLUDE baseline here to force Z to learn temporal patterns.
    
    # Calculate input dim: Z_dim (16) + Explicit_dim (Num_Feats * 7)
    dummy_stats = next(iter(train_loader))[3]
    input_dim = 16 + dummy_stats.shape[1]
    
    aux_head = AuxiliaryHead(input_dim).to(DEVICE)
    
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(aux_head.parameters()), 
        lr=0.001
    )
    # Binary Cross Entropy
    criterion = nn.BCELoss()
    
    for epoch in range(epochs):
        encoder.train()
        aux_head.train()
        total_loss = 0
        
        for t_data, labels, base, explicit_stats in train_loader:
            labels = labels.to(DEVICE)
            explicit_stats = explicit_stats.to(DEVICE)
            
            # Forward Pass
            z = encoder(t_data)
            
            # Combine Z with Explicit Stats
            combined = torch.cat([z, explicit_stats], dim=1)
            
            preds = aux_head(combined)
            loss = criterion(preds, labels)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            
        # Validation (Simple print)
        if (epoch+1) % 5 == 0:
            avg_loss = total_loss / len(train_loader)
            print(f"    Epoch {epoch+1:02d} | Train Loss: {avg_loss:.4f}")

    return encoder

# ==============================================================================
# 4. Main Execution
# ==============================================================================

def main():
    print("="*80)
    print("Tab_Xv5: Hybrid Explicit-Interaction Architecture")
    print("="*80)

    # 1. Load Data
    patients = load_and_prepare_patients()
    temporal_feats = get_all_temporal_features(patients)
    print(f"Temporal Features: {len(temporal_feats)}")
    
    # 2. Main Loop
    metrics_hybrid = {'auc': [], 'aupr': []}
    metrics_base = {'auc': [], 'aupr': []}
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    for fold, (train_full, test_p) in enumerate(trainTestPatients(patients, seed=xseed)):
        print(f"\n{'='*60}\nFold {fold}\n{'='*60}")
        
        # A. Split Train/Val
        train_p_obj, val_p_obj = split_patients_train_val(train_full, val_ratio=0.1, seed=42+fold)
        
        # B. Prepare Static Baseline (Demographics + Last Values of Fixed Features)
        # Note: We use -6h to 24h window for "Last" baseline check
        def get_baseline_df(p_obj):
            return p_obj.getMeasuresBetween(
                pd.Timedelta(hours=-6), pd.Timedelta(hours=24), "last", getUntilAkiPositive=True
            ).drop(columns=["subject_id", "hadm_id", "stay_id"])

        df_train = get_baseline_df(train_p_obj)
        df_val = get_baseline_df(val_p_obj)
        df_test = get_baseline_df(test_p)
        
        # Encode Categorical Data
        df_train_enc, df_test_enc, df_val_enc = encodeCategoricalData(df_train, df_test, df_val)
        
        X_tr_base = df_train_enc.drop(columns=["akd"]).fillna(0)
        X_val_base = df_val_enc.drop(columns=["akd"]).fillna(0)
        X_te_base = df_test_enc.drop(columns=["akd"]).fillna(0)
        
        # C. Create Hybrid Datasets
        # This will compute the Explicit Stats (Hard Path)
        print("  Creating Hybrid Datasets (Computing Explicit Stats)...")
        train_ds = HybridDatasetV5(train_p_obj.patientList, temporal_feats, X_tr_base)
        stats = train_ds.get_normalization_stats()
        val_ds = HybridDatasetV5(val_p_obj.patientList, temporal_feats, X_val_base, stats)
        test_ds = HybridDatasetV5(test_p.patientList, temporal_feats, X_te_base, stats)
        
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=hybrid_collate_fn)
        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=hybrid_collate_fn)
        test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=hybrid_collate_fn)
        
        # D. Train Interaction Encoder (Soft Path)
        # 16 latent dimensions is enough for "extra info"
        encoder = FeatureInteractionEncoder(
            num_features=len(temporal_feats),
            hidden_dim=64,
            latent_dim=16 
        ).to(DEVICE)
        
        encoder = train_hybrid_encoder(encoder, train_loader, val_loader, epochs=30)
        
        # E. Extract Final Combined Features
        # Vector = [Static (Baseline)] + [Explicit Stats (7*F)] + [Learned Z (16)]
        print("  Extracting final vectors for TabPFN...")
        encoder.eval()
        
        def get_vectors(loader):
            feats, labs = [], []
            with torch.no_grad():
                for t_data, labels, base, explicit_stats in loader:
                    z = encoder(t_data).cpu()
                    
                    # CONCATENATE EVERYTHING
                    combined = torch.cat([base, explicit_stats, z], dim=1)
                    feats.append(combined.numpy())
                    labs.extend(labels.numpy())
            return np.vstack(feats), np.array(labs)
            
        X_train_final, y_train_final = get_vectors(train_loader)
        X_test_final, y_test_final = get_vectors(test_loader)
        
        print(f"  Feature Vector Shape: {X_train_final.shape}")
        
        # F. TabPFN Training
        print("  Training TabPFN...")
        # Use ensemble_configurations=4 for speed/performance balance
        classifier = TabPFNClassifier(device='cuda' if torch.cuda.is_available() else 'cpu')
        classifier.fit(X_train_final, y_train_final)
        
        # G. Evaluation
        y_prob = classifier.predict_proba(X_test_final)[:, 1]
        
        fold_auc = roc_auc_score(y_test_final, y_prob)
        prec, rec, _ = precision_recall_curve(y_test_final, y_prob)
        fold_aupr = auc(rec, prec)
        
        metrics_hybrid['auc'].append(fold_auc)
        metrics_hybrid['aupr'].append(fold_aupr)
        
        print(f"  [Hybrid] AUC: {fold_auc:.4f} | AUPR: {fold_aupr:.4f}")
        
        # Plot Hybrid
        fpr, tpr, _ = roc_curve(y_test_final, y_prob)
        ax1.plot(fpr, tpr, lw=2, label=f"Fold {fold} (AUC={fold_auc:.3f})")
        
        # H. Baseline Comparison (Static Only)
        base_clf = TabPFNClassifier(device='cuda' if torch.cuda.is_available() else 'cpu')
        base_clf.fit(X_tr_base, y_train_final)
        y_prob_base = base_clf.predict_proba(X_te_base)[:, 1]
        
        base_auc = roc_auc_score(y_test_final, y_prob_base)
        base_prec, base_rec, _ = precision_recall_curve(y_test_final, y_prob_base)
        base_aupr = auc(base_rec, base_prec)
        
        metrics_base['auc'].append(base_auc)
        metrics_base['aupr'].append(base_aupr)
        
        print(f"  [Baseln] AUC: {base_auc:.4f} | AUPR: {base_aupr:.4f}")
        
        fpr_b, tpr_b, _ = roc_curve(y_test_final, y_prob_base)
        ax2.plot(fpr_b, tpr_b, lw=2, label=f"Fold {fold} (AUC={base_auc:.3f})")
        
    # --- Final Plots & Stats ---
    for ax, title in zip([ax1, ax2], ["Tab_Xv5 (Hybrid)", "Baseline (Static Only)"]):
        ax.plot([0, 1], [0, 1], linestyle="--", color="navy", lw=2)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.legend(loc="lower right")
        ax.set_title(title)
        
    plt.tight_layout()
    plt.savefig("result/tab_xv5_hybrid.png", dpi=300)
    print("\nResults saved to result/tab_xv5_hybrid.png")
    
    print("\n" + "="*80)
    print("FINAL SUMMARY (Tab_Xv5)")
    print("="*80)
    
    def print_res(name, m_hyb, m_base):
        h_mean, h_std = np.mean(m_hyb), np.std(m_hyb)
        b_mean, b_std = np.mean(m_base), np.std(m_base)
        imp = ((h_mean - b_mean)/b_mean)*100
        print(f"{name:6s} | Hybrid: {h_mean:.4f}±{h_std:.4f} vs Base: {b_mean:.4f}±{b_std:.4f} ({imp:+.2f}%)")
        
    print_res("AUC", metrics_hybrid['auc'], metrics_base['auc'])
    print_res("AUPR", metrics_hybrid['aupr'], metrics_base['aupr'])

if __name__ == "__main__":
    main()