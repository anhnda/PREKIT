"""
Tab_Xv1: Deep Statistical Temporal Encoder with Raw Last Values

Goal: Learn temporal representations that capture statistical patterns (mean, max, min, slope)
      using Bi-GRU with attention, while keeping raw last values as anchors.

Key approach:
1. DeepStatisticalTemporalEncoder: Bi-GRU with attention that learns statistical patterns
   - Last state (like "last value")
   - Max pooling (like "max")
   - Attention-weighted average (learned "mean")
   - Time embeddings capture temporal dynamics (like "slope")
2. Feature extraction: [Static + Last_Raw_Values + Learned_Z]
   - Static: demographic and clinical features
   - Last_Raw_Values: hard anchor to recent observations
   - Learned_Z: contextual representation from temporal encoder
3. RL training: Policy gradient to optimize for downstream performance
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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FIXED_FEATURES = [
    "age", "gender", "race", "chronic_pulmonary_disease", "ckd_stage",
    "congestive_heart_failure", "dka_type", "history_aci", "history_ami",
    "hypertension", "liver_disease", "macroangiopathy", "malignant_cancer",
    "microangiopathy", "uti", "oasis", "saps2", "sofa",
    "mechanical_ventilation", "use_NaHCO3", "preiculos", "gcs_unable"
]

# ==============================================================================
# 1. Static Encoder
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
# 2. Dataset
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
        'times': padded_times,
        'values': padded_values,
        'masks': padded_masks,
        'lengths': torch.tensor(lengths)
    }
    return temporal_batch, torch.tensor(label_list, dtype=torch.float32), torch.stack(static_list)

# ==============================================================================
# 3. Deep Statistical Temporal Encoder (from XEmbedding.py)
# ==============================================================================

class ContinuousTimeEmbedding(nn.Module):
    """
    Learns a continuous representation of time intervals.
    Replaces fixed sin/cos encoding with a learnable MLP approach.
    """
    def __init__(self, time_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.LayerNorm(time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )

    def forward(self, x):
        return self.fc(x)

class DeepStatisticalTemporalEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, time_dim=32):
        super().__init__()
        self.hidden_dim = hidden_dim

        # 1. Feature Projection (Value + Mask)
        self.input_proj = nn.Linear(input_dim * 2, hidden_dim)

        # 2. Time Embedding
        self.time_embed = ContinuousTimeEmbedding(time_dim)

        # 3. Bi-Directional GRU
        rnn_input_dim = hidden_dim + time_dim
        self.rnn = nn.GRU(
            input_size=rnn_input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1
        )

        # 4. Attention Mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

        # 5. Output Projection
        concat_dim = (hidden_dim * 2) * 3  # (Last + Max + Avg) * bidirectional

        self.fc_mean = nn.Linear(concat_dim, latent_dim)
        self.fc_logstd = nn.Linear(concat_dim, latent_dim)
        self.latent_dim = latent_dim

    def forward(self, batch_data, deterministic=False, temperature=1.0):
        # Unpack data
        times = batch_data['times'].to(DEVICE)
        values = batch_data['values'].to(DEVICE)
        masks = batch_data['masks'].to(DEVICE)
        lengths = batch_data['lengths'].to(DEVICE)

        B, L, D = values.size()

        # 1. Calculate Delta Time (Time Gaps)
        prev_times = torch.cat([torch.zeros(B, 1).to(DEVICE), times[:, :-1]], dim=1)
        delta_t = (times - prev_times).unsqueeze(-1)
        t_embed = self.time_embed(delta_t)

        # 2. Embed Values & Masks
        inputs = torch.cat([values, masks], dim=-1)
        x_embed = self.input_proj(inputs)

        # 3. Combine Time + Features
        rnn_input = torch.cat([x_embed, t_embed], dim=-1)

        # 4. Pack sequence for RNN
        packed_input = nn.utils.rnn.pack_padded_sequence(
            rnn_input, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        # Run Bi-GRU
        packed_output, _ = self.rnn(packed_input)

        # Unpack
        rnn_outputs, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True, total_length=L
        )

        # Create mask for valid timesteps
        mask_seq = torch.arange(L, device=DEVICE).expand(B, L) < lengths.unsqueeze(1)
        mask_seq = mask_seq.unsqueeze(-1).float()

        # Apply mask to outputs
        rnn_outputs = rnn_outputs * mask_seq

        # 5. Statistical Abstractions
        # Last State
        idx = (lengths - 1).view(-1, 1).expand(B, rnn_outputs.size(2)).unsqueeze(1)
        last_state = rnn_outputs.gather(1, idx).squeeze(1)

        # Max Pooling
        masked_for_max = rnn_outputs.clone()
        masked_for_max[mask_seq.expand_as(rnn_outputs) == 0] = -1e9
        max_pool = torch.max(masked_for_max, dim=1)[0]

        # Attention-weighted average
        attn_weights = self.attention(rnn_outputs)
        attn_weights = attn_weights.masked_fill(mask_seq == 0, -1e9)
        attn_weights = F.softmax(attn_weights, dim=1)
        context_vec = torch.sum(rnn_outputs * attn_weights, dim=1)

        # Concatenate all statistical views
        combined = torch.cat([last_state, max_pool, context_vec], dim=1)

        # Latent Projection
        mean = self.fc_mean(combined)
        log_std = self.fc_logstd(combined)
        log_std = torch.clamp(log_std, min=-20, max=2)
        std = torch.exp(log_std) * temperature

        policy_dist = dist.Normal(mean, std)

        if deterministic:
            z = mean
            log_prob = None
        else:
            z = policy_dist.rsample()
            log_prob = policy_dist.log_prob(z).sum(dim=-1)

        return z, log_prob, mean


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
# 4. Enhanced Pretraining
# ==============================================================================

def pretrain_rnn_enhanced(policy_net, train_loader, val_loader, epochs=50):
    print("  [Enhanced Pretraining] Supervised learning with deep encoder...")

    supervised_head = SupervisedHead(
        policy_net.latent_dim + len(FIXED_FEATURES)
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

        for t_data, labels, s_data in train_loader:
            labels = labels.to(DEVICE)
            s_data = s_data.to(DEVICE)

            z, _, _ = policy_net(t_data, deterministic=True)
            combined = torch.cat([z, s_data], dim=1)

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
                for t_data, labels, s_data in val_loader:
                    s_data = s_data.to(DEVICE)
                    z, _, _ = policy_net(t_data, deterministic=True)
                    combined = torch.cat([z, s_data], dim=1)
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

    print(f"  [Enhanced Pretraining] Completed. Best Val AUPR: {best_auc:.4f}")
    return policy_net, best_auc

# ==============================================================================
# 5. Feature Extraction: [Static + Last_Raw_Values + Learned_Z]
# ==============================================================================

def extract_extended_features_and_logprobs(policy_net, loader, deterministic=False, temperature=1.0):
    """
    V5: Extract features: [Static + Last_Raw_Values + Learned_Z]
    Retains 'Last' as a hard anchor, but replaces Mean/Max/Slope/etc with Learned Z.
    """
    policy_net.eval() if deterministic else policy_net.train()

    all_features = []
    all_labels = []
    all_log_probs = []

    with torch.set_grad_enabled(not deterministic):
        for t_data, labels, s_data in loader:
            # 1. Get Learned Representations (Z)
            z, log_prob, mean = policy_net(t_data, deterministic=deterministic, temperature=temperature)
            z_np = (mean if deterministic else z).detach().cpu().numpy()

            # 2. Extract RAW Last Values (The "skip connection")
            vals = t_data['values'].cpu().numpy()
            masks = t_data['masks'].cpu().numpy()

            # Efficiently extract the last valid value for each feature per patient
            batch_last_vals = []

            for i in range(len(vals)):
                patient_last = []
                for f_idx in range(vals.shape[2]):
                    # Get mask for this specific feature/patient
                    f_mask = masks[i, :, f_idx]

                    # Find indices where data exists
                    valid_idx = np.where(f_mask > 0)[0]

                    if len(valid_idx) > 0:
                        # Take the value at the last valid index
                        last_val = vals[i, valid_idx[-1], f_idx]
                        patient_last.append(last_val)
                    else:
                        # Impute 0.0 if feature is completely missing
                        patient_last.append(0.0)

                batch_last_vals.append(patient_last)

            last_vals_arr = np.array(batch_last_vals)
            s_np = s_data.numpy()

            # 3. Concatenate: [Static (Raw) + Last (Raw) + Z (Learned Context)]
            # This combination is very powerful for TabPFN
            combined = np.hstack([s_np, last_vals_arr, z_np])

            all_features.append(combined)
            all_labels.extend(labels.numpy())

            if not deterministic and log_prob is not None:
                all_log_probs.append(log_prob)

    features = np.vstack(all_features)
    labels = np.array(all_labels)
    log_probs = torch.cat(all_log_probs) if all_log_probs else None

    return features, labels, log_probs

# ==============================================================================
# 6. Adaptive RL Training
# ==============================================================================

def train_policy_adaptive_rl(
    policy_net,
    train_loader,
    val_loader,
    tabpfn_params,
    pretrain_aupr,
    epochs=80,  # Reduced from 120
    update_tabpfn_every=3  # More frequent updates
):
    """
    Adaptive RL with careful training to avoid degradation
    Key fixes:
    - Slower temperature decay
    - Stricter early stopping
    - Lower learning rate
    - More frequent TabPFN updates
    """

    # More conservative learning rate
    if pretrain_aupr < 0.75:
        lr = 0.0001  # Reduced from 0.0002
        print(f"  [Adaptive RL] Using higher LR (0.0001) due to lower pretrain AUPR")
    else:
        lr = 0.00005  # Reduced from 0.0001
        print(f"  [Adaptive RL] Using conservative LR (0.00005)")

    optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr)

    best_val_auc = 0
    best_state = None
    patience = 15  # Reduced from 30 - stop earlier if degrading
    patience_counter = 0

    print(f"  [Adaptive RL] Training for up to {epochs} epochs with strict early stopping...")

    tabpfn_model = None
    val_history = []

    for epoch in range(epochs):
        # Slower temperature decay - keep exploration longer
        temperature = max(0.4, 1.0 - epoch / (epochs * 1.2))  # Slower decay, higher minimum

        policy_net.train()

        X_train, y_train, log_probs_train = extract_extended_features_and_logprobs(
            policy_net, train_loader, deterministic=False, temperature=temperature
        )

        if epoch % update_tabpfn_every == 0 or tabpfn_model is None:
            tabpfn_model = TabPFNClassifier(**tabpfn_params)
            tabpfn_model.fit(X_train, y_train)

        # Validation-based rewards
        X_val_stoch, y_val, _ = extract_extended_features_and_logprobs(
            policy_net, val_loader, deterministic=False, temperature=temperature
        )

        y_val_proba = tabpfn_model.predict_proba(X_val_stoch)[:, 1]
        val_aupr = average_precision_score(y_val, y_val_proba)
        val_history.append(val_aupr)

        # Compute rewards - more conservative approach
        y_train_proba = tabpfn_model.predict_proba(X_train)[:, 1]
        rewards_smooth = np.where(y_train == 1, y_train_proba, 1 - y_train_proba)

        # More conservative AUPR weight - don't increase as aggressively
        aupr_weight = 0.2 + 0.2 * min(1.0, epoch / 40)  # 0.2 → 0.4 (was 0.3 → 0.6)

        rewards_combined = rewards_smooth + aupr_weight * val_aupr

        rewards_tensor = torch.tensor(rewards_combined, dtype=torch.float32).to(DEVICE)
        rewards_tensor = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std() + 1e-8)

        log_probs_train = log_probs_train.to(DEVICE)
        policy_loss = -(log_probs_train * rewards_tensor).mean()

        # Stronger entropy bonus to encourage exploration
        entropy_bonus = 0.005 * log_probs_train.mean()  # Increased from 0.001
        total_loss = policy_loss - entropy_bonus

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=0.5)
        optimizer.step()

        # More frequent validation to catch degradation early
        if (epoch + 1) % 3 == 0 or epoch == 0:
            policy_net.eval()
            with torch.no_grad():
                X_val_det, y_val_det, _ = extract_extended_features_and_logprobs(
                    policy_net, val_loader, deterministic=True
                )

                X_train_det, y_train_det, _ = extract_extended_features_and_logprobs(
                    policy_net, train_loader, deterministic=True
                )

                tabpfn_val_model = TabPFNClassifier(**tabpfn_params)
                tabpfn_val_model.fit(X_train_det, y_train_det)

                y_val_proba_det = tabpfn_val_model.predict_proba(X_val_det)[:, 1]

                val_auc = roc_auc_score(y_val_det, y_val_proba_det)
                val_aupr_det = average_precision_score(y_val_det, y_val_proba_det)

                # Moving average for stability
                recent_avg = np.mean(val_history[-10:]) if len(val_history) >= 10 else val_aupr_det

                # Check for significant degradation
                degradation_warning = ""
                if best_val_auc > 0 and (best_val_auc - val_aupr_det) > 0.02:
                    degradation_warning = " ⚠ DEGRADING"

                print(f"    Epoch {epoch+1:3d} | Temp: {temperature:.3f} | LR: {lr:.6f} | "
                      f"Val AUC: {val_auc:.4f} | Val AUPR: {val_aupr_det:.4f} | "
                      f"Best: {best_val_auc:.4f}{degradation_warning}")

                # Early stopping - stricter criteria
                if val_aupr_det > best_val_auc:
                    best_val_auc = val_aupr_det
                    best_state = copy.deepcopy(policy_net.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1
                    # Also stop if we've degraded significantly
                    if patience_counter >= patience or (best_val_auc - val_aupr_det) > 0.05:
                        print(f"    Early stopping at epoch {epoch+1} (patience: {patience_counter}/{patience})")
                        break

    if best_state is not None:
        policy_net.load_state_dict(best_state)

    return policy_net

# ==============================================================================
# 7. Main Execution
# ==============================================================================

def main():
    print("="*80)
    print("Tab_Xv1: Deep Statistical Temporal Encoder + Raw Last Values")
    print("="*80)

    patients = load_and_prepare_patients()
    temporal_feats = get_all_temporal_features(patients)

    print("Encoding static features...")
    encoder = SimpleStaticEncoder(FIXED_FEATURES)
    encoder.fit(patients.patientList)

    print(f"Input: {len(temporal_feats)} Temporal + {len(FIXED_FEATURES)} Static Features")
    print(f"Feature extraction: [Static + Last_Raw + Learned_Z]")

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

        train_ds = HybridDataset(train_p, temporal_feats, encoder)
        stats = train_ds.get_normalization_stats()
        val_ds = HybridDataset(val_p, temporal_feats, encoder, stats)
        test_ds = HybridDataset(test_p_list, temporal_feats, encoder, stats)

        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=hybrid_collate_fn)
        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=hybrid_collate_fn)
        test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=hybrid_collate_fn)

        # Use DeepStatisticalTemporalEncoder
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
        policy_net, pretrain_aupr = pretrain_rnn_enhanced(policy_net, train_loader, val_loader, epochs=50)

        # STEP 2: RL Fine-tuning (with conservative settings)
        print(f"\n  Starting RL fine-tuning from pretrain AUPR: {pretrain_aupr:.4f}")
        policy_net = train_policy_adaptive_rl(
            policy_net,
            train_loader,
            val_loader,
            tabpfn_params,
            pretrain_aupr,
            epochs=80,  # Reduced
            update_tabpfn_every=3  # More frequent
        )

        # Final Evaluation (After RL)
        print("\n  [Final Test Evaluation - After RL]")
        policy_net.eval()

        with torch.no_grad():
            X_train_final, y_train_final, _ = extract_extended_features_and_logprobs(
                policy_net, train_loader, deterministic=True
            )
            X_test_final, y_test_final, _ = extract_extended_features_and_logprobs(
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

        print(f"  Test AUC: {fold_auc:.4f} | Test AUPR: {fold_aupr:.4f}")

        # BASELINE
        print("\n  [Baseline] Standard TabPFN (Last + Static)...")

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

        print(f"  Baseline Test AUC: {baseline_auc:.4f} | Test AUPR: {baseline_aupr:.4f}")
        print(f"  Fold {fold} Results -> Tab_Xv1: {fold_auc:.3f} vs Baseline: {baseline_auc:.3f}")

    # Final Plot
    for ax in [ax1, ax2]:
        ax.plot([0, 1], [0, 1], linestyle="--", color="navy", lw=2)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(loc="lower right")

    ax1.set_title("Tab_Xv1: Deep Encoder [Static+Last+Z]")
    ax2.set_title("Baseline [Static+Last]")
    plt.tight_layout()
    plt.savefig("result/tab_xv1_vs_baseline.png", dpi=300)
    print("\nPlot saved to result/tab_xv1_vs_baseline.png")

    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY")
    print("="*80)

    def print_stat(name, rl_metrics, base_metrics):
        rl_mean, rl_std = np.mean(rl_metrics), np.std(rl_metrics)
        base_mean, base_std = np.mean(base_metrics), np.std(base_metrics)
        improvement = ((rl_mean - base_mean) / base_mean) * 100
        symbol = "✓" if rl_mean > base_mean else "✗"
        print(f"{name:15s} | Tab_Xv1: {rl_mean:.4f} ± {rl_std:.4f}  vs  Baseline: {base_mean:.4f} ± {base_std:.4f}  ({improvement:+.2f}%) {symbol}")

    print_stat("AUC", metrics_rl['auc'], metrics_baseline['auc'])
    print_stat("AUC-PR", metrics_rl['auc_pr'], metrics_baseline['auc_pr'])

    print("\n" + "="*80)
    print("Tab_Xv1 INNOVATION:")
    print("✓ Bi-GRU learns temporal statistical patterns (mean, max, trend)")
    print("✓ Attention mechanism focuses on critical events")
    print("✓ Raw last values as hard anchor (skip connection)")
    print("✓ Learned context Z captures complex temporal dependencies")
    print("\nTRAINING IMPROVEMENTS vs RLv4:")
    print("✓ More conservative learning rate (0.00005 vs 0.0001)")
    print("✓ Slower temperature decay (0.4 min vs 0.25)")
    print("✓ Stricter early stopping (15 patience vs 30)")
    print("✓ More frequent validation (every 3 epochs vs 5)")
    print("✓ Stronger entropy bonus (0.005 vs 0.001)")
    print("✓ Conservative reward weighting (0.2-0.4 vs 0.3-0.6)")
    print("="*80)

if __name__ == "__main__":
    main()
