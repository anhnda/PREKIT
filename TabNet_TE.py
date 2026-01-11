"""
OPTIMIZED HYBRID MODEL: Time-Embedded RNN + Gated Head Pre-training + XGBoost

- Stage 1: RNN is trained using a 'Gated Decision Head' (mimics XGBoost logic).
- Stage 2: We extract the RNN embeddings (h) and concatenate with Static Features (s).
- Stage 3: XGBoost predicts using the combined [h, s] vector.
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
from xgboost import XGBClassifier

from sklearn.metrics import roc_auc_score

PT = "/Users/anhnd/CodingSpace/Python/PREDKIT"
if sys.platform != "darwin":
    PT = "/home/anhnda/PREKIT"
sys.path.append(PT)

from constants import NULLABLE_MEASURES
from utils.class_patient import Patients
from utils.prepare_data import trainTestPatients
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
# 1. The Gated Decision Head (The "Smart" Head)
# ==============================================================================

class GatedDecisionHead(nn.Module):
    """
    Mimics XGBoost/Tree logic using Gated Linear Units (GLU).
    Used to train the RNN so it produces 'tree-friendly' features.
    """
    def __init__(self, input_dim, hidden_dim=64, dropout=0.3):
        super(GatedDecisionHead, self).__init__()
        
        # Feature Selection Gate (Soft Mask)
        self.gate = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Sigmoid() 
        )
        
        # Gated Processing Layers
        self.fc1 = nn.Linear(input_dim, hidden_dim * 2) 
        self.fc2 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.dropout = nn.Dropout(dropout)
        self.final = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # 1. Select Features (Masking)
        mask = self.gate(x)
        x = x * mask 
        
        # 2. Process (GLU = Signal * Sigmoid(Gate))
        out = self.fc1(x)
        out = F.glu(out, dim=-1)
        out = self.dropout(out)
        
        residual = out
        out = self.fc2(out)
        out = F.glu(out, dim=-1)
        out = out + residual # Skip connection
        
        return torch.sigmoid(self.final(out))

# ==============================================================================
# 2. Dataset & Collate (Returns Static Features)
# ==============================================================================

class HybridDataset(Dataset):
    def __init__(self, patients, feature_names, normalization_stats=None):
        self.data = []
        self.labels = []
        self.static_data = [] 
        self.feature_names = feature_names

        all_values = []
        for patient in patients.patientList:
            times, values, masks = extract_temporal_data(patient, feature_names)
            if times is None: continue
            
            # Static Data
            s_vec = []
            for feat in FIXED_FEATURES:
                val = patient.measures.get(feat, 0.0)
                if hasattr(val, 'values'): val = list(val.values())[0] if len(val)>0 else 0.0
                s_vec.append(float(val))
            
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
# 3. RNN Feature Extractor (The Backbone)
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
    """
    Pre-trains RNN using the GatedDecisionHead.
    This ensures RNN learns features suitable for tree-based logic.
    """
    # USE THE SMART HEAD HERE
    rnn_dim = model.rnn_cell.hidden_dim
    static_dim = len(FIXED_FEATURES)
    
    # We train the head on [RNN_h + Static_s] to perfectly mimic the final XGBoost task
    temp_head = GatedDecisionHead(input_dim=rnn_dim + static_dim).to(DEVICE)
    
    # We need to optimize both the RNN and the Head
    full_optimizer = torch.optim.Adam(
        list(model.parameters()) + list(temp_head.parameters()), 
        lr=0.001
    )
    
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
            
            # 1. Get RNN features
            h = model(t_data)
            
            # 2. Combine with Static (just like XGBoost will do later)
            combined = torch.cat([h, s_data], dim=1)
            
            # 3. Predict with Gated Head
            preds = temp_head(combined).squeeze(-1)
            loss = criterion(preds, labels)
            
            full_optimizer.zero_grad()
            loss.backward()
            full_optimizer.step()
            
        # Validation
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
                if counter >= patience: 
                    print("    Early stopping.")
                    break
                
    model.load_state_dict(best_state)
    print(f"  RNN Pre-trained. Best Val AUC: {best_auc:.4f}")
    return model

# ==============================================================================
# 4. Feature Extraction & XGBoost
# ==============================================================================

def get_hybrid_features(model, loader):
    """Extracts RNN Embeddings AND Static Features."""
    model.eval()
    features = []
    labels_out = []
    
    with torch.no_grad():
        for t_data, labels, s_data in loader:
            h = model(t_data).cpu().numpy()
            s = s_data.numpy()
            combined = np.hstack([h, s])
            features.append(combined)
            labels_out.extend(labels.numpy())
            
    return np.vstack(features), np.array(labels_out)

def main():
    print("="*80)
    print("OPTIMIZED HYBRID: RNN + GATED PRE-TRAIN + XGBOOST")
    print("="*80)
    
    patients = load_and_prepare_patients()
    temporal_feats = get_all_temporal_features(patients)
    print(f"Input: {len(temporal_feats)} Temporal + {len(FIXED_FEATURES)} Static Features")
    
    auc_scores = []
    
    for fold, (train_full, test_p) in enumerate(trainTestPatients(patients)):
        print(f"\n--- Fold {fold} ---")
        train_p, val_p = split_patients_train_val(train_full, val_ratio=0.1, seed=42+fold)
        
        # 1. Data Loaders
        train_ds = HybridDataset(train_p, temporal_feats)
        stats = train_ds.get_normalization_stats()
        val_ds = HybridDataset(val_p, temporal_feats, stats)
        test_ds = HybridDataset(test_p, temporal_feats, stats)
        
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=hybrid_collate_fn)
        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=hybrid_collate_fn)
        test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=hybrid_collate_fn)
        
        # 2. Stage 1: Train RNN with Gated Head
        rnn = RNNFeatureExtractor(len(temporal_feats), hidden_dim=128).to(DEVICE)
        opt = torch.optim.Adam(rnn.parameters(), lr=0.001)
        # Pass full_optimizer logic is inside train_rnn_extractor now
        rnn = train_rnn_extractor(rnn, train_loader, val_loader, nn.BCELoss(), opt)
        
        # 3. Feature Fusion
        print("  [Stage 2] Extracting Hybrid Features...")
        X_train, y_train = get_hybrid_features(rnn, train_loader)
        X_val, y_val = get_hybrid_features(rnn, val_loader)
        X_test, y_test = get_hybrid_features(rnn, test_loader)
        
        # 4. Stage 3: Train XGBoost
        print(f"  [Stage 3] Training XGBoost on {X_train.shape[1]} features...")
        
        ratio = np.sum(y_train==0) / np.sum(y_train==1)
        
        clf = XGBClassifier(
            n_estimators=1000,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=ratio,
            early_stopping_rounds=20,
            eval_metric='auc',
            random_state=42
        )
        
        clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        
        y_prob = clf.predict_proba(X_test)[:, 1]
        score = roc_auc_score(y_test, y_prob)
        auc_scores.append(score)
        
        print(f"  Fold {fold} Hybrid AUC: {score:.4f}")
        
    print(f"\nAverage AUC: {np.mean(auc_scores):.4f} ± {np.std(auc_scores):.4f}")

if __name__ == "__main__":
    main()