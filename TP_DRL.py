"""
RL-Based Feature Learning (REINFORCE)
-------------------------------------
Agent:   RNN (Gaussian Policy)
Action:  Generates embedding vector 'z' via sampling
Reward:  TabPFN confidence in the TRUE class
Goal:    Maximize TabPFN's ability to classify patients correctly
"""

import pandas as pd
import numpy as np
import sys
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from torch.utils.data import Dataset, DataLoader
from tabpfn import TabPFNClassifier
import random
import os
from matplotlib import pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    auc,
    average_precision_score
)

# ==============================================================================
# 0. Setup & Constants
# ==============================================================================

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

# Paths (Adjust as needed)
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
# 1. Helpers: Encoder & Dataset
# ==============================================================================

class SimpleStaticEncoder:
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

def extract_last_values(t_data):
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
# 2. RL Agent: Gaussian RNN Policy
# ==============================================================================

class GaussianRNNPolicy(nn.Module):
    def __init__(self, input_dim, hidden_dim, z_dim=12):
        super().__init__()
        self.rnn_cell = TimeEmbeddedRNNCell(input_dim, hidden_dim)
        
        # Policy Heads: Mean and Log-Standard Deviation
        self.mu_head = nn.Linear(hidden_dim, z_dim)
        self.log_sigma_head = nn.Linear(hidden_dim, z_dim)
        
        # Initialize slightly to avoid initial explosion
        nn.init.constant_(self.log_sigma_head.weight, 0)
        nn.init.constant_(self.log_sigma_head.bias, -1.0) # Start with small noise

    def forward(self, batch_data):
        h = self.rnn_cell(
            batch_data['times'].to(DEVICE),
            batch_data['values'].to(DEVICE),
            batch_data['masks'].to(DEVICE),
            batch_data['lengths'].to(DEVICE)
        )
        
        mu = self.mu_head(h)
        log_sigma = self.log_sigma_head(h)
        
        # Clamp for numerical stability: Avoid too small or too large std dev
        sigma = torch.exp(torch.clamp(log_sigma, min=-5, max=0.5))
        
        return mu, sigma
    
    def get_deterministic_embedding(self, batch_data):
        """Use this for testing (returns just the mean)"""
        mu, _ = self.forward(batch_data)
        return mu

# ==============================================================================
# 3. RL Training Function (REINFORCE)
# ==============================================================================

def train_rnn_rl(rnn_policy, train_loader, tabpfn, epochs=20, alpha_baseline=0.1):
    """
    Trains the RNN using REINFORCE with TabPFN as the Reward Function.
    """
    optimizer = torch.optim.Adam(rnn_policy.parameters(), lr=0.0005)
    
    # Baseline for variance reduction (Moving Average of rewards)
    baseline = 0.5 
    
    print("\nStarting RL Training Loop...")
    
    for epoch in range(epochs):
        rnn_policy.train()
        epoch_rewards = []
        epoch_loss = 0
        
        # 1. Periodically Re-Fit TabPFN? 
        # For true RL, the environment (TabPFN + Support Set) should ideally be fixed 
        # or updated carefully. Here we assume TabPFN is already fit on the training data
        # with the *current* best guess of Z (done in main loop).
        
        for batch_idx, (t_data, labels, s_data) in enumerate(train_loader):
            optimizer.zero_grad()
            
            # --- A. Agent Acts ---
            mu, sigma = rnn_policy(t_data)
            
            # Create distribution and sample Action 'z'
            dist_normal = dist.Normal(mu, sigma)
            z_action = dist_normal.sample()
            
            # Log Probability of the action
            log_prob = dist_normal.log_prob(z_action).sum(dim=1)
            
            # --- B. Environment (TabPFN) Rewards ---
            with torch.no_grad():
                last_vals = extract_last_values(t_data).cpu().numpy()
                static_vals = s_data.numpy()
                z_vals = z_action.cpu().numpy()
                
                # Combine features: [Last, Static, Z]
                X_batch = np.hstack([last_vals, static_vals, z_vals])
                y_true = labels.numpy().astype(int)
                
                # Get TabPFN predictions
                # Note: TabPFN batch inference can be slow.
                probs = tabpfn.predict_proba(X_batch)
                
                # Reward = Confidence in the TRUE class
                # This incentivizes z to make the patient "look like" their true class
                rewards = []
                for i in range(len(y_true)):
                    r = probs[i, y_true[i]]
                    rewards.append(r)
                
                rewards = torch.tensor(rewards, dtype=torch.float32).to(DEVICE)
            
            # --- C. Policy Update ---
            batch_avg_reward = rewards.mean().item()
            epoch_rewards.append(batch_avg_reward)
            
            # Update baseline
            baseline = (1 - alpha_baseline) * baseline + alpha_baseline * batch_avg_reward
            
            # Advantage
            advantage = rewards - baseline
            
            # Loss = - E[ log(p(a|s)) * A ]
            loss = -(log_prob * advantage).mean()
            
            # Optional: Entropy Regularization to prevent collapse (keep exploring)
            entropy = dist_normal.entropy().mean()
            loss = loss - 0.01 * entropy
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(rnn_policy.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()

        avg_r = np.mean(epoch_rewards)
        print(f"  Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/len(train_loader):.4f} | Avg Reward (Conf): {avg_r:.4f} | Baseline: {baseline:.4f}")

    return rnn_policy

# ==============================================================================
# 4. Helper: Get Features for TabPFN
# ==============================================================================

def get_triple_features(model, loader, deterministic=True):
    model.eval()
    features = []
    labels_out = []
    
    with torch.no_grad():
        for t_data, labels, s_data in loader:
            if deterministic:
                z = model.get_deterministic_embedding(t_data).cpu().numpy()
            else:
                mu, sigma = model(t_data)
                z = dist.Normal(mu, sigma).sample().cpu().numpy()
                
            s = s_data.numpy()
            last_vals = extract_last_values(t_data).cpu().numpy()
            
            combined = np.hstack([last_vals, s, z])
            features.append(combined)
            labels_out.extend(labels.numpy())
            
    return np.vstack(features), np.array(labels_out)

# ==============================================================================
# 5. Main Execution
# ==============================================================================

def main():
    print("="*80)
    print("DEEP RL TRAINING: TabPFN as Reward Function")
    print("="*80)

    patients = load_and_prepare_patients()
    temporal_feats = get_all_temporal_features(patients)
    encoder = SimpleStaticEncoder(FIXED_FEATURES)
    encoder.fit(patients.patientList)

    print(f"Features: {len(temporal_feats)} Temporal, {len(FIXED_FEATURES)} Static")

    metrics_rl = {'auc': [], 'auc_pr': []}
    metrics_base = {'auc': [], 'auc_pr': []}

    for fold, (train_full, test_p) in enumerate(trainTestPatients(patients)):
        print(f"\n--- Fold {fold} ---")
        train_p_obj, val_p_obj = split_patients_train_val(train_full, val_ratio=0.1, seed=42+fold)
        
        train_ds = HybridDataset(train_p_obj.patientList, temporal_feats, encoder)
        stats = train_ds.get_normalization_stats()
        # For RL, we can merge Train and Val to maximize support set, 
        # or keep val to monitor overfitting. Let's merge for the Reward Function support.
        # But we need a separate val set to check if RL is working.
        val_ds = HybridDataset(val_p_obj.patientList, temporal_feats, encoder, stats)
        test_ds = HybridDataset(test_p.patientList, temporal_feats, encoder, stats)

        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=hybrid_collate_fn)
        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=hybrid_collate_fn)
        test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=hybrid_collate_fn)

        # 1. Initialize RL Policy
        rnn_policy = GaussianRNNPolicy(len(temporal_feats), hidden_dim=16, z_dim=12).to(DEVICE)
        
        # 2. Initial "Cold" Fit of TabPFN
        # We need to initialize TabPFN with *something* so it can start giving rewards.
        # We use the initial random Z from the RNN.
        print("  [Setup] Initializing TabPFN Environment...")
        X_train_init, y_train_init = get_triple_features(rnn_policy, train_loader, deterministic=False)
        
        # The Environment (TabPFN)
        # N_ensemble_configurations can be lowered to speed up RL loop
        tabpfn_env = TabPFNClassifier(device='cuda' if torch.cuda.is_available() else 'cpu') 
        tabpfn_env.fit(X_train_init, y_train_init)

        # 3. RL Training
        # Optional: Train in cycles. Train RNN -> Update TabPFN Support Set -> Train RNN
        # This helps TabPFN adapt to the evolving Z features.
        cycles = 3
        epochs_per_cycle = 10
        
        for c in range(cycles):
            print(f"  [Cycle {c+1}/{cycles}] RL Optimization...")
            rnn_policy = train_rnn_rl(rnn_policy, train_loader, tabpfn_env, epochs=epochs_per_cycle)
            
            # Update the Environment (TabPFN) with the new, better features
            # This allows TabPFN to learn the new manifold created by the RNN
            print(f"  [Cycle {c+1}/{cycles}] Updating TabPFN Support Set...")
            X_train_new, y_train_new = get_triple_features(rnn_policy, train_loader, deterministic=True)
            tabpfn_env.fit(X_train_new, y_train_new)

        # 4. Final Evaluation
        print("  [Final] Evaluating on Test Set...")
        
        # Train a fresh, full-power TabPFN on the final features
        final_tabpfn = TabPFNClassifier(device='cuda' if torch.cuda.is_available() else 'cpu')
        X_train_final, y_train_final = get_triple_features(rnn_policy, train_loader, deterministic=True)
        X_test_final, y_test_final = get_triple_features(rnn_policy, test_loader, deterministic=True)
        
        final_tabpfn.fit(X_train_final, y_train_final)
        y_prob = final_tabpfn.predict_proba(X_test_final)[:, 1]
        
        auc_score = roc_auc_score(y_test_final, y_prob)
        precision, recall, _ = precision_recall_curve(y_test_final, y_prob)
        pr_auc = auc(recall, precision)
        
        metrics_rl['auc'].append(auc_score)
        metrics_rl['auc_pr'].append(pr_auc)
        print(f"  Fold {fold} RL Result | AUC: {auc_score:.4f} | PR-AUC: {pr_auc:.4f}")

        # 5. Baseline (Last + Static only)
        # We simulate this by zeroing out Z or just dropping it
        print("  [Baseline] Evaluating Last+Static...")
        X_tr_b = X_train_final[:, :-12] # Remove last 12 cols (Z)
        X_te_b = X_test_final[:, :-12]
        
        base_pfn = TabPFNClassifier(device='cuda' if torch.cuda.is_available() else 'cpu')
        base_pfn.fit(X_tr_b, y_train_final)
        y_prob_b = base_pfn.predict_proba(X_te_b)[:, 1]
        
        b_auc = roc_auc_score(y_test_final, y_prob_b)
        bp, br, _ = precision_recall_curve(y_test_final, y_prob_b)
        b_pr = auc(br, bp)
        
        metrics_base['auc'].append(b_auc)
        metrics_base['auc_pr'].append(b_pr)
        print(f"  Fold {fold} Baseline  | AUC: {b_auc:.4f} | PR-AUC: {b_pr:.4f}")

    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print(f"RL Agent -> AUC: {np.mean(metrics_rl['auc']):.4f} | PR-AUC: {np.mean(metrics_rl['auc_pr']):.4f}")
    print(f"Baseline -> AUC: {np.mean(metrics_base['auc']):.4f} | PR-AUC: {np.mean(metrics_base['auc_pr']):.4f}")
    print("="*80)

if __name__ == "__main__":
    main()