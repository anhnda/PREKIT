"""
PURE TEMPORAL HYBRID: Time-Embedded RNN + XGBoost
(NO STATIC FEATURES / DEMOGRAPHICS)
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

# ==============================================================================
# 1. Gated Decision Head (Optimized for Temporal Embeddings Only)
# ==============================================================================

class GatedDecisionHead(nn.Module):
    """
    Learns to classify the RNN embeddings using tree-like logic (GLU).
    """
    def __init__(self, input_dim, hidden_dim=64, dropout=0.3):
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
# 2. Pure Temporal Dataset
# ==============================================================================

class TemporalDataset(Dataset):
    def __init__(self, patients, feature_names, normalization_stats=None):
        self.data = []
        self.labels = []
        
        # Support both list of patients or Patients object
        patient_list = patients.patientList if hasattr(patients, 'patientList') else patients
        all_values = []

        for patient in patient_list:
            times, values, masks = extract_temporal_data(patient, feature_names)
            if times is None: continue
            
            self.data.append({'times': times, 'values': values, 'masks': masks})
            self.labels.append(1 if patient.akdPositive else 0)
            
            for v_vec, m_vec in zip(values, masks):
                for v, m in zip(v_vec, m_vec):
                    if m > 0: all_values.append(v)

        # Normalization
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
        return self.data[idx], self.labels[idx]

def temporal_collate_fn(batch):
    data_list, label_list = zip(*batch)
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
    return temporal_batch, torch.tensor(label_list, dtype=torch.float32)

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
    # Pure Temporal Dimensions
    rnn_dim = model.rnn_cell.hidden_dim
    
    # Train using the Gated Head on just RNN output
    temp_head = GatedDecisionHead(input_dim=rnn_dim).to(DEVICE)
    
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
        
        for t_data, labels in train_loader: 
            labels = labels.to(DEVICE)
            h = model(t_data)
            preds = temp_head(h).squeeze(-1)
            loss = criterion(preds, labels)
            
            full_optimizer.zero_grad()
            loss.backward()
            full_optimizer.step()
            
        if (epoch+1) % 5 == 0:
            model.eval()
            temp_head.eval()
            all_preds, all_lbls = [], []
            with torch.no_grad():
                for t_data, labels in val_loader:
                    h = model(t_data)
                    preds = temp_head(h).squeeze(-1)
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
    print(f"  RNN Pre-trained. Best Val AUC: {best_auc:.4f}")
    return model

def get_rnn_features(model, loader):
    model.eval()
    features = []
    labels_out = []
    with torch.no_grad():
        for t_data, labels in loader:
            h = model(t_data).cpu().numpy()
            features.append(h)
            labels_out.extend(labels.numpy())
    return np.vstack(features), np.array(labels_out)

# ==============================================================================
# 4. Main
# ==============================================================================

def main():
    print("="*80)
    print("PURE TEMPORAL HYBRID: TIME-EMBEDDED RNN + XGBOOST")
    print("="*80)
    
    patients = load_and_prepare_patients()
    temporal_feats = get_all_temporal_features(patients)
    print(f"Input: {len(temporal_feats)} Temporal Features ONLY")
    
    metrics = {k: [] for k in ['auc', 'acc', 'spec', 'prec', 'rec', 'auc_pr']}
    
    fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))
    
    for fold, (train_full, test_p) in enumerate(trainTestPatients(patients)):
        print(f"\n--- Fold {fold} ---")
        train_p_obj, val_p_obj = split_patients_train_val(train_full, val_ratio=0.1, seed=42+fold)
        train_p, val_p, test_p_list = train_p_obj.patientList, val_p_obj.patientList, test_p.patientList
        
        # 1. Dataset
        train_ds = TemporalDataset(train_p, temporal_feats)
        stats = train_ds.get_normalization_stats()
        val_ds = TemporalDataset(val_p, temporal_feats, stats)
        test_ds = TemporalDataset(test_p_list, temporal_feats, stats)
        
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=temporal_collate_fn)
        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=temporal_collate_fn)
        test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=temporal_collate_fn)
        
        # 2. Stage 1: RNN Training
        rnn = RNNFeatureExtractor(len(temporal_feats), hidden_dim=128).to(DEVICE)
        opt = torch.optim.Adam(rnn.parameters(), lr=0.001)
        rnn = train_rnn_extractor(rnn, train_loader, val_loader, nn.BCELoss(), opt)
        
        # 3. Stage 2: Extract Embeddings (128 dims)
        X_train, y_train = get_rnn_features(rnn, train_loader)
        X_val, y_val = get_rnn_features(rnn, val_loader)
        X_test, y_test = get_rnn_features(rnn, test_loader)
        
        # 4. Stage 3: XGBoost Training
        ratio = np.sum(y_train==0) / (np.sum(y_train==1) + 1e-6)
        
        clf = XGBClassifier(
            n_estimators=1000, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, scale_pos_weight=ratio, 
            early_stopping_rounds=30, eval_metric='auc', random_state=42
        )
        clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        
        # 5. Evaluation
        y_prob = clf.predict_proba(X_test)[:, 1]
        y_pred = (y_prob > 0.5).astype(int)
        
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        
        metrics['auc'].append(roc_auc_score(y_test, y_prob))
        metrics['acc'].append(accuracy_score(y_test, y_pred))
        metrics['spec'].append(tn / (tn + fp))
        metrics['prec'].append(precision_score(y_test, y_pred, zero_division=0))
        metrics['rec'].append(recall_score(y_test, y_pred))
        metrics['auc_pr'].append(auc(rec, prec))
        
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        ax1.plot(fpr, tpr, lw=2, label=f"Fold {fold} (AUC = {metrics['auc'][-1]:.3f})")
        
        print(f"  Fold {fold} AUC: {metrics['auc'][-1]:.3f}")

    # Final Plot
    ax1.plot([0, 1], [0, 1], linestyle="--", color="navy", lw=2)
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title("Pure Temporal Hybrid (RNN + XGB)")
    plt.tight_layout()
    plt.savefig("result/pure_temporal_hybrid.png", dpi=300)

    # Final Stats
    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY (TEMPORAL ONLY)")
    print("="*80)
    def print_stat(name, m):
        print(f"{name:15s} | {np.mean(m):.4f} ± {np.std(m):.4f}")
        
    print_stat("AUC", metrics['auc'])
    print_stat("AUC-PR", metrics['auc_pr'])
    print_stat("Accuracy", metrics['acc'])
    print_stat("Specificity", metrics['spec'])

if __name__ == "__main__":
    main()