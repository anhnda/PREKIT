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
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

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
        
        # Robust list handling
        p_list = patients.patientList if hasattr(patients, 'patientList') else patients
        if not isinstance(p_list, list): p_list = []

        for patient in p_list:
            times, values, masks = extract_temporal_data(patient, feature_names)
            # CRITICAL: Skip if no temporal data
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
            if len(all_values) > 0:
                self.mean = np.mean(all_values)
                self.std = np.std(all_values) + 1e-6
            else:
                self.mean = 0.0; self.std = 1.0
        else:
            self.mean = normalization_stats['mean']
            self.std = normalization_stats['std']

        # Create Normalized Tensors
        for i in range(len(self.data)):
            raw_vals = self.data[i]['values']
            masks = self.data[i]['masks']
            norm_values = []
            for v_vec, m_vec in zip(raw_vals, masks):
                norm = [(v - self.mean)/self.std if m>0 else 0.0 for v, m in zip(v_vec, m_vec)]
                norm_values.append(norm)
            
            self.data[i] = {
                'times': torch.tensor(self.data[i]['times'], dtype=torch.float32),
                'norm_values': torch.tensor(norm_values, dtype=torch.float32), 
                'raw_values': torch.tensor(raw_vals, dtype=torch.float32),     
                'masks': torch.tensor(masks, dtype=torch.float32)
            }

    def get_normalization_stats(self): return {'mean': self.mean, 'std': self.std}
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx], self.labels[idx], self.static_data[idx]

def hybrid_collate_fn(batch):
    data_list, label_list, static_list = zip(*batch)
    lengths = [len(d['times']) for d in data_list]
    max_len = max(lengths)
    feat_dim = data_list[0]['norm_values'].shape[-1]
    
    padded_times = torch.zeros(len(batch), max_len)
    padded_norm = torch.zeros(len(batch), max_len, feat_dim)
    padded_raw = torch.zeros(len(batch), max_len, feat_dim)
    padded_masks = torch.zeros(len(batch), max_len, feat_dim)

    for i, d in enumerate(data_list):
        l = lengths[i]
        padded_times[i, :l] = d['times']
        padded_norm[i, :l] = d['norm_values']
        padded_raw[i, :l] = d['raw_values']
        padded_masks[i, :l] = d['masks']
        
    return {
        'times': padded_times, 'norm_values': padded_norm, 
        'raw_values': padded_raw, 'masks': padded_masks, 
        'lengths': torch.tensor(lengths)
    }, torch.tensor(label_list, dtype=torch.float32), torch.stack(static_list)

def extract_last_raw_values(t_data):
    vals = t_data['raw_values'].to(DEVICE)
    masks = t_data['masks'].to(DEVICE)
    batch_last_vals = []
    for i in range(vals.shape[0]):
        patient_last = []
        for f_idx in range(vals.shape[2]):
            valid_idx = torch.where(masks[i, :, f_idx] > 0)[0]
            if len(valid_idx) > 0:
                last_v = vals[i, :, f_idx][valid_idx[-1]]
            else:
                last_v = torch.tensor(0.0, device=DEVICE)
            patient_last.append(last_v)
        batch_last_vals.append(torch.stack(patient_last))
    return torch.stack(batch_last_vals)

def get_features_for_fitting(model, loader, deterministic=True):
    model.eval()
    last_list, static_list, z_list, label_list = [], [], [], []
    with torch.no_grad():
        for t_data, labels, s_data in loader:
            if deterministic:
                z = model.get_deterministic_embedding(t_data).cpu().numpy()
            else:
                mu, sigma = model(t_data)
                z = dist.Normal(mu, sigma).sample().cpu().numpy()
            
            last_list.append(extract_last_raw_values(t_data).cpu().numpy())
            static_list.append(s_data.numpy())
            z_list.append(z)
            label_list.extend(labels.numpy())
    
    if len(last_list) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])
        
    return np.vstack(last_list), np.vstack(static_list), np.vstack(z_list), np.array(label_list)

# ==============================================================================
# 2. RL Agent
# ==============================================================================

class GaussianRNNPolicy(nn.Module):
    def __init__(self, input_dim, hidden_dim, z_dim=12):
        super().__init__()
        self.rnn_cell = TimeEmbeddedRNNCell(input_dim, hidden_dim)
        self.fc_shared = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(0.1)
        self.mu_head = nn.Linear(hidden_dim, z_dim)
        self.log_sigma_head = nn.Linear(hidden_dim, z_dim)
        nn.init.constant_(self.log_sigma_head.weight, 0)
        nn.init.constant_(self.log_sigma_head.bias, -1.0) 

    def forward(self, batch_data):
        h = self.rnn_cell(
            batch_data['times'].to(DEVICE), batch_data['norm_values'].to(DEVICE), 
            batch_data['masks'].to(DEVICE), batch_data['lengths'].to(DEVICE)
        )
        h = self.dropout(F.relu(self.fc_shared(h)))
        mu = self.mu_head(h)
        log_sigma = self.log_sigma_head(h)
        sigma = torch.exp(torch.clamp(log_sigma, min=-3, max=0))
        return mu, sigma
    
    def get_deterministic_embedding(self, batch_data):
        mu, _ = self.forward(batch_data)
        return mu

# ==============================================================================
# 3. RL Training (Robust Iterative Judge)
# ==============================================================================

def train_rnn_rl(rnn_policy, train_loader, base_tabpfn, epochs=15):
    """
    Updates the Judge every 3 epochs so the agent cannot 'cheat' a frozen model.
    """
    optimizer = torch.optim.Adam(rnn_policy.parameters(), lr=0.0001)
    
    print(f"  [RL] Starting Fine-Tuning (Iterative Judge)...")
    print(f"  {'Epoch':>5} | {'Base Prob':>10} | {'Agent Prob':>10} | {'Avg Delta':>10} | {'Reward':>8}")
    
    # Initial Environment Fit
    env_tabpfn = TabPFNClassifier(device='cuda' if torch.cuda.is_available() else 'cpu')
    last_tr, stat_tr, z_tr, y_tr = get_features_for_fitting(rnn_policy, train_loader)
    
    # Augment data for Judge: [Full] + [Masked]
    X_full = np.hstack([last_tr, stat_tr, z_tr])
    X_masked = np.hstack([np.zeros_like(last_tr), stat_tr, z_tr])
    env_tabpfn.fit(np.vstack([X_full, X_masked]), np.concatenate([y_tr, y_tr]))

    for epoch in range(epochs):
        rnn_policy.train()
        track_rewards, track_deltas = [], []
        track_base, track_agent = [], []
        
        # --- RE-FIT JUDGE EVERY 3 EPOCHS ---
        if epoch > 0 and epoch % 3 == 0:
            # Refresh Z features with current policy
            last_tr, stat_tr, z_tr, y_tr = get_features_for_fitting(rnn_policy, train_loader)
            X_full = np.hstack([last_tr, stat_tr, z_tr])
            X_masked = np.hstack([np.zeros_like(last_tr), stat_tr, z_tr])
            env_tabpfn.fit(np.vstack([X_full, X_masked]), np.concatenate([y_tr, y_tr]))

        for t_data, labels, s_data in train_loader:
            optimizer.zero_grad()
            mu, sigma = rnn_policy(t_data)
            dist_normal = dist.Normal(mu, sigma)
            z_action = dist_normal.sample()
            log_prob = dist_normal.log_prob(z_action).sum(dim=1)
            
            with torch.no_grad():
                last_vals = extract_last_raw_values(t_data).cpu().numpy()
                static_vals = s_data.numpy()
                z_vals = z_action.cpu().numpy()
                y_true = labels.numpy().astype(int)

                p_base = base_tabpfn.predict_proba(np.hstack([last_vals, static_vals]))[:, 1]
                p_env = env_tabpfn.predict_proba(np.hstack([last_vals, static_vals, z_vals]))[:, 1]
                
                rewards, deltas = [], []
                for i in range(len(y_true)):
                    target = 1.0 if y_true[i] == 1 else 0.0
                    imp = (p_env[i] - p_base[i]) if target == 1 else (p_base[i] - p_env[i])
                    # Clip reward to be safe
                    r = np.clip(imp * 5.0, -2.0, 2.0)
                    rewards.append(r)
                    deltas.append(imp)
                    
                rewards = torch.tensor(rewards, dtype=torch.float32).to(DEVICE)
                track_base.extend(p_base); track_agent.extend(p_env)
                track_deltas.extend(deltas); track_rewards.extend(rewards.cpu().numpy())

            loss = -(log_prob * rewards).mean() - 0.05 * dist_normal.entropy().mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(rnn_policy.parameters(), max_norm=0.5)
            optimizer.step()

        if (epoch+1) % 3 == 0:
            print(f"  {epoch+1:05d} | {np.mean(track_base):10.4f} | {np.mean(track_agent):10.4f} | {np.mean(track_deltas):10.4f} | {np.mean(track_rewards):8.4f}")

    return rnn_policy

# ==============================================================================
# 5. Main Execution
# ==============================================================================

def main():
    print("="*80)
    print("STABLE DEEP RL V4: Robust Iterative Judge + Strict Baseline")
    print("="*80)

    patients = load_and_prepare_patients()
    temporal_feats = get_all_temporal_features(patients)
    
    if len(temporal_feats) == 0:
        print("ERROR: No temporal features found.")
        return

    encoder = SimpleStaticEncoder(FIXED_FEATURES)
    encoder.fit(patients.patientList)

    metrics_rl = {'auc': [], 'auc_pr': []}
    metrics_base = {'auc': [], 'auc_pr': []}

    for fold, (train_full, test_p) in enumerate(trainTestPatients(patients)):
        print(f"\n--- Fold {fold} ---")
        train_p_obj, val_p_obj = split_patients_train_val(train_full, val_ratio=0.1, seed=42+fold)
        
        # 1. Load Data with Empty Check
        train_ds = HybridDataset(train_p_obj.patientList, temporal_feats, encoder)
        if len(train_ds) == 0:
            print("  Warning: Train set empty for this fold. Skipping.")
            continue
            
        stats = train_ds.get_normalization_stats()
        test_ds = HybridDataset(test_p.patientList, temporal_feats, encoder, stats)
        
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=hybrid_collate_fn)
        test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=hybrid_collate_fn)

        rnn_policy = GaussianRNNPolicy(len(temporal_feats), hidden_dim=64, z_dim=3).to(DEVICE)

        # --------------------------------------------------------
        # STAGE 1: SUPERVISED WARM-UP
        # --------------------------------------------------------
        print("  [Stage 1] Supervised Warm-Up (15 Epochs)...")
        warmup_head = nn.Linear(12, 1).to(DEVICE)
        warmup_opt = torch.optim.Adam(list(rnn_policy.parameters()) + list(warmup_head.parameters()), lr=0.001)
        bce = nn.BCELoss()
        
        rnn_policy.train()
        for epoch in range(15):
            for t_data, labels, s_data in train_loader:
                warmup_opt.zero_grad()
                mu, _ = rnn_policy(t_data) 
                pred = torch.sigmoid(warmup_head(mu)).squeeze()
                loss = bce(pred, labels.to(DEVICE))
                loss.backward()
                warmup_opt.step()

        # --------------------------------------------------------
        # STAGE 2: RL WITH ITERATIVE JUDGE
        # --------------------------------------------------------
        last_tr, stat_tr, z_tr, y_tr = get_features_for_fitting(rnn_policy, train_loader)
        if len(y_tr) == 0:
            print("  Error: No features for Stage 2.")
            continue

        # Fixed Baseline (The Target)
        base_tabpfn = TabPFNClassifier(device='cuda' if torch.cuda.is_available() else 'cpu')
        base_tabpfn.fit(np.hstack([last_tr, stat_tr]), y_tr)
        
        # Start Training (Judge fits inside the function)
        rnn_policy = train_rnn_rl(rnn_policy, train_loader, base_tabpfn, epochs=15)

        # --------------------------------------------------------
        # EVALUATION: RL AGENT
        # --------------------------------------------------------
        print("  [Final] Evaluating RL Agent...")
        last_tr_new, stat_tr_new, z_tr_new, y_tr_new = get_features_for_fitting(rnn_policy, train_loader)
        last_te, stat_te, z_te, y_te = get_features_for_fitting(rnn_policy, test_loader)
        
        final_rl_clf = TabPFNClassifier(device='cuda' if torch.cuda.is_available() else 'cpu')
        final_rl_clf.fit(np.hstack([last_tr_new, stat_tr_new, z_tr_new]), y_tr_new)
        probs_rl = final_rl_clf.predict_proba(np.hstack([last_te, stat_te, z_te]))[:, 1]
        
        auc_rl = roc_auc_score(y_te, probs_rl)
        rp, rr, _ = precision_recall_curve(y_te, probs_rl)
        pr_rl = auc(rr, rp)
        metrics_rl['auc'].append(auc_rl); metrics_rl['auc_pr'].append(pr_rl)

        # --------------------------------------------------------
        # EVALUATION: STRICT PANDAS BASELINE
        # --------------------------------------------------------
        print("  [Final] Evaluating Pandas Baseline...")
        
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

        base_pfn_hard = TabPFNClassifier(device='cuda' if torch.cuda.is_available() else 'cpu')
        base_pfn_hard.fit(X_tr_b, y_tr_b)
        probs_base = base_pfn_hard.predict_proba(X_te_b)[:, 1]
        
        auc_base = roc_auc_score(y_te_b, probs_base)
        bp, br, _ = precision_recall_curve(y_te_b, probs_base)
        pr_base = auc(br, bp)
        metrics_base['auc'].append(auc_base); metrics_base['auc_pr'].append(pr_base)

        print(f"  Fold {fold} Result:")
        print(f"    Pandas Baseline: AUC {auc_base:.4f} | PR-AUC {pr_base:.4f}")
        print(f"    RL Agent       : AUC {auc_rl:.4f} | PR-AUC {pr_rl:.4f}  <-- {'IMPROVED' if pr_rl > pr_base else 'Same/Worse'}")

    print("\n" + "="*80)
    print("FINAL SUMMARY (Mean ± Std)")
    def print_stat(name, h_metrics, b_metrics):
        h_mean, h_std = np.mean(h_metrics), np.std(h_metrics)
        b_mean, b_std = np.mean(b_metrics), np.std(b_metrics)
        print(f"{name:15s} | RL Agent: {h_mean:.4f} ± {h_std:.4f}  vs  Pandas Baseline: {b_mean:.4f} ± {b_std:.4f}")

    print_stat("AUC", metrics_rl['auc'], metrics_base['auc'])
    print_stat("AUC-PR", metrics_rl['auc_pr'], metrics_base['auc_pr'])

if __name__ == "__main__":
    main()