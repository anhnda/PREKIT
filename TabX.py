"""
TabX: Learnable Statistical Feature Aggregator

Problem:
- Handcrafted features (mean, std, min, max, slope, count) significantly improve performance
- But they are manually engineered - can we learn them instead?

Solution:
Multi-Head Statistical Aggregator Network that learns to compute statistics:
1. Central Tendency Head: Learns mean-like aggregation (uniform attention)
2. Variability Head: Learns std-like aggregation (variance estimation)
3. Extrema Heads: Learn min/max detection (attention to extremes)
4. Trend Head: Learns slope-like patterns (temporal weighting)
5. Recency Head: Learns last-value importance (recent attention)
6. Density Head: Learns count-like features (observation density)

Architecture:
  Irregular Time Series → Multi-Head Statistical Aggregator → Learned Stats
                       ↓
  Learned Stats + Static → TabPFN → Prediction

Key Innovation:
- Auxiliary loss during pretraining: Match learned stats to actual handcrafted stats
- This provides supervision for what to learn
- But allows network to discover better aggregations
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
import math

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
# 1. Multi-Head Statistical Aggregator
# ==============================================================================

class StatisticalHead(nn.Module):
    """
    Single statistical aggregation head using attention mechanism
    Learns to compute a specific statistic (mean, max, min, etc.)
    """
    def __init__(self, input_dim, hidden_dim, output_dim, head_type="general"):
        super().__init__()
        self.head_type = head_type

        # Query network: what statistic to compute
        self.query_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Key network: which values are relevant
        self.key_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Value network: what to aggregate
        self.value_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

        # Temporal encoding for trend detection
        self.time_encoder = nn.Linear(1, hidden_dim)

        self.hidden_dim = hidden_dim
        self.scale = math.sqrt(hidden_dim)

    def forward(self, values, times, masks):
        """
        Args:
            values: (batch, seq_len, feat_dim)
            times: (batch, seq_len)
            masks: (batch, seq_len, feat_dim)
        Returns:
            aggregated: (batch, feat_dim, output_dim)
        """
        batch_size, seq_len, feat_dim = values.shape

        # Process each feature independently
        outputs = []

        for f in range(feat_dim):
            feat_values = values[:, :, f:f+1]  # (batch, seq_len, 1)
            feat_masks = masks[:, :, f:f+1]    # (batch, seq_len, 1)

            # Mask out invalid values
            feat_values_masked = feat_values * feat_masks

            # Encode temporal information
            time_enc = self.time_encoder(times.unsqueeze(-1))  # (batch, seq_len, hidden)

            # Concatenate value and time
            combined_input = torch.cat([feat_values_masked, time_enc], dim=-1)  # (batch, seq_len, 1+hidden)

            # Compute query (global - what to look for)
            # Use masked mean as global context
            valid_counts = feat_masks.sum(dim=1, keepdim=True).clamp(min=1)
            global_context = feat_values_masked.sum(dim=1, keepdim=True) / valid_counts

            # Handle potential NaN in global context
            global_context = torch.nan_to_num(global_context, nan=0.0)

            query = self.query_net(torch.cat([global_context, torch.zeros_like(time_enc[:, :1, :])], dim=-1))  # (batch, 1, hidden)

            # Compute keys and values
            keys = self.key_net(combined_input)    # (batch, seq_len, hidden)
            values_proj = self.value_net(combined_input)  # (batch, seq_len, output_dim)

            # Attention scores
            attention = torch.matmul(query, keys.transpose(-2, -1)) / self.scale  # (batch, 1, seq_len)

            # Mask out invalid positions
            mask_attention = (feat_masks.squeeze(-1) == 0)  # (batch, seq_len)

            # Check if all positions are masked (no valid observations for this feature)
            all_masked = mask_attention.all(dim=1)  # (batch,)

            attention = attention.masked_fill(mask_attention.unsqueeze(1), float('-inf'))

            # Softmax attention (will produce NaN if all values are -inf)
            attention_weights = F.softmax(attention, dim=-1)  # (batch, 1, seq_len)

            # Replace NaN with uniform weights for all-masked cases
            attention_weights = torch.where(
                all_masked.view(-1, 1, 1).expand_as(attention_weights),
                torch.zeros_like(attention_weights),  # Use zeros for all-masked
                attention_weights
            )

            # Aggregate
            aggregated = torch.matmul(attention_weights, values_proj).squeeze(1)  # (batch, output_dim)

            outputs.append(aggregated)

        # Stack all features
        output = torch.stack(outputs, dim=1)  # (batch, feat_dim, output_dim)

        return output


class MultiHeadStatisticalAggregator(nn.Module):
    """
    Multiple specialized heads for learning different statistics
    """
    def __init__(self, input_dim, hidden_dim=32, num_features=15):
        super().__init__()

        self.num_features = num_features

        # Different heads for different statistics
        # Each head outputs 1 value per feature
        self.central_head = StatisticalHead(1 + hidden_dim, hidden_dim, 1, "central")  # mean-like
        self.variability_head = StatisticalHead(1 + hidden_dim, hidden_dim, 1, "variability")  # std-like
        self.max_head = StatisticalHead(1 + hidden_dim, hidden_dim, 1, "max")  # max-like
        self.min_head = StatisticalHead(1 + hidden_dim, hidden_dim, 1, "min")  # min-like
        self.trend_head = StatisticalHead(1 + hidden_dim, hidden_dim, 1, "trend")  # slope-like
        self.recency_head = StatisticalHead(1 + hidden_dim, hidden_dim, 1, "recency")  # last-like

        # Density estimator (count-like)
        self.density_net = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_features)
        )

    def forward(self, values, times, masks):
        """
        Args:
            values: (batch, seq_len, feat_dim) - normalized values
            times: (batch, seq_len) - time stamps
            masks: (batch, seq_len, feat_dim) - validity masks
        Returns:
            learned_stats: (batch, feat_dim * 7) - 7 statistics per feature
        """
        # Compute learned statistics through different heads
        central = self.central_head(values, times, masks).squeeze(-1)  # (batch, feat_dim)
        variability = self.variability_head(values, times, masks).squeeze(-1)
        max_vals = self.max_head(values, times, masks).squeeze(-1)
        min_vals = self.min_head(values, times, masks).squeeze(-1)
        trend = self.trend_head(values, times, masks).squeeze(-1)
        recency = self.recency_head(values, times, masks).squeeze(-1)

        # Density (count-like): based on mask density
        density_input = masks.float().mean(dim=1)  # (batch, feat_dim) - fraction of valid obs
        density = self.density_net(density_input)  # (batch, feat_dim)

        # Concatenate all statistics
        learned_stats = torch.cat([
            recency, central, variability, min_vals, max_vals, trend, density
        ], dim=1)  # (batch, feat_dim * 7)

        # Safety: Replace any NaN or Inf values with 0
        learned_stats = torch.nan_to_num(learned_stats, nan=0.0, posinf=0.0, neginf=0.0)

        return learned_stats


# ==============================================================================
# 2. Dataset with Ground Truth Statistics
# ==============================================================================

class StatisticalDataset(Dataset):
    def __init__(self, patients, feature_names, static_features_df, normalization_stats=None):
        self.data = []
        self.labels = []
        self.static_data = []
        self.true_stats = []  # Ground truth handcrafted features
        self.last_values_data = []
        self.feature_names = feature_names

        all_values = []
        patient_list = patients.patientList if hasattr(patients, 'patientList') else patients

        for idx, patient in enumerate(patient_list):
            times, values, masks = extract_temporal_data(patient, feature_names)
            if times is None:
                continue

            s_vec = static_features_df.iloc[idx].values
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

            # Compute ground truth handcrafted features
            true_handcrafted = self._compute_handcrafted_features(
                norm_values, self.data[i]['masks'], self.data[i]['times']
            )
            true_handcrafted = np.nan_to_num(true_handcrafted, nan=0.0, posinf=0.0, neginf=0.0)
            self.true_stats.append(torch.tensor(true_handcrafted, dtype=torch.float32))

            last_vals = self._extract_last_values(norm_values, self.data[i]['masks'])
            self.last_values_data.append(torch.tensor(last_vals, dtype=torch.float32))

            self.data[i] = {
                'times': torch.tensor(self.data[i]['times'], dtype=torch.float32),
                'values': torch.tensor(norm_values, dtype=torch.float32),
                'masks': torch.tensor(self.data[i]['masks'], dtype=torch.float32)
            }

    def _compute_handcrafted_features(self, norm_values, masks, times):
        handcrafted = []
        for f_idx in range(len(norm_values[0])):
            f_vals = np.array([norm_values[t][f_idx] for t in range(len(norm_values))])
            f_mask = np.array([masks[t][f_idx] for t in range(len(masks))])
            f_times = np.array(times)
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
                slope = (valid_vals[-1] - valid_vals[0]) / (valid_times[-1] - valid_times[0] + 1e-6) if len(valid_vals) >= 2 else 0.0
                handcrafted.extend([last, mean_val, std_val, min_val, max_val, slope, count])
            else:
                handcrafted.extend([0.0] * 7)
        return handcrafted

    def _extract_last_values(self, norm_values, masks):
        last_vals = []
        for f_idx in range(len(norm_values[0])):
            f_vals = np.array([norm_values[t][f_idx] for t in range(len(norm_values))])
            f_mask = np.array([masks[t][f_idx] for t in range(len(masks))])
            valid_idx = np.where(f_mask > 0)[0]
            if len(valid_idx) > 0:
                last_vals.append(f_vals[valid_idx[-1]])
            else:
                last_vals.append(0.0)
        return last_vals

    def get_normalization_stats(self):
        return {'mean': self.mean, 'std': self.std}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (self.data[idx], self.labels[idx], self.static_data[idx],
                self.true_stats[idx], self.last_values_data[idx])


def statistical_collate_fn(batch):
    data_list, label_list, static_list, true_stats_list, last_list = zip(*batch)
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
            torch.stack(static_list), torch.stack(true_stats_list), torch.stack(last_list))


# ==============================================================================
# 3. Policy Network with Learned Statistics
# ==============================================================================

class PolicyNetworkWithLearnedStats(nn.Module):
    """
    Policy network using learned statistical features
    """
    def __init__(self, num_features, hidden_dim, latent_dim):
        super().__init__()

        # Multi-head statistical aggregator
        self.stat_aggregator = MultiHeadStatisticalAggregator(
            input_dim=1,  # per feature
            hidden_dim=hidden_dim,
            num_features=num_features
        )

        # Number of learned statistics per feature
        self.num_stats_per_feature = 7  # recency, central, var, min, max, trend, density
        self.learned_stats_dim = num_features * self.num_stats_per_feature

        # Policy head: learned_stats → latent Z (for exploration/exploitation)
        self.fc_mean = nn.Linear(self.learned_stats_dim, latent_dim)
        self.fc_logstd = nn.Linear(self.learned_stats_dim, latent_dim)

        self.latent_dim = latent_dim
        self.num_features = num_features

    def forward(self, batch_data, deterministic=False, temperature=1.0):
        times = batch_data['times'].to(DEVICE)
        values = batch_data['values'].to(DEVICE)
        masks = batch_data['masks'].to(DEVICE)

        # Learn statistics through multi-head aggregator
        learned_stats = self.stat_aggregator(values, times, masks)  # (batch, feat_dim * 7)

        # Generate policy distribution over latent Z
        mean = self.fc_mean(learned_stats)
        log_std = self.fc_logstd(learned_stats)
        log_std = torch.clamp(log_std, min=-20, max=2)
        std = torch.exp(log_std) * temperature

        policy_dist = dist.Normal(mean, std)

        if deterministic:
            z = mean
            log_prob = None
        else:
            z = policy_dist.rsample()
            log_prob = policy_dist.log_prob(z).sum(dim=-1)

        return learned_stats, z, log_prob, mean


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
# 4. Pretraining with Statistical Matching Loss
# ==============================================================================

def pretrain_with_stat_matching(policy_net, train_loader, val_loader, static_dim, epochs=50):
    """
    Pretraining with two objectives:
    1. Classification loss (primary)
    2. Statistical matching loss (auxiliary) - match learned stats to true handcrafted
    """
    print("  [Statistical Matching Pretraining]")

    supervised_head = SupervisedHead(
        policy_net.learned_stats_dim + policy_net.latent_dim + static_dim
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        list(policy_net.parameters()) + list(supervised_head.parameters()),
        lr=0.001
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )

    criterion_cls = nn.BCELoss()
    criterion_stat = nn.MSELoss()

    best_auc = 0
    best_state = None
    patience = 12
    counter = 0

    for epoch in range(epochs):
        policy_net.train()
        supervised_head.train()

        for t_data, labels, s_data, true_stats, last_vals in train_loader:
            labels = labels.to(DEVICE)
            s_data = s_data.to(DEVICE)
            true_stats = true_stats.to(DEVICE)

            # Forward pass
            learned_stats, z, _, _ = policy_net(t_data, deterministic=True)

            # Classification loss
            combined = torch.cat([learned_stats, z, s_data], dim=1)
            preds = supervised_head(combined).squeeze(-1)
            cls_loss = criterion_cls(preds, labels)

            # Statistical matching loss: learned stats should match true handcrafted
            stat_loss = criterion_stat(learned_stats, true_stats)

            # Combined loss (stat matching weight = 0.5)
            total_loss = cls_loss + 0.5 * stat_loss

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(policy_net.parameters()) + list(supervised_head.parameters()),
                max_norm=1.0
            )
            optimizer.step()

        if (epoch + 1) % 3 == 0:
            policy_net.eval()
            supervised_head.eval()
            all_preds, all_labels = [], []
            total_stat_error = 0
            num_batches = 0

            with torch.no_grad():
                for t_data, labels, s_data, true_stats, last_vals in val_loader:
                    s_data = s_data.to(DEVICE)
                    true_stats = true_stats.to(DEVICE)

                    learned_stats, z, _, _ = policy_net(t_data, deterministic=True)

                    combined = torch.cat([learned_stats, z, s_data], dim=1)
                    preds = supervised_head(combined).squeeze(-1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.numpy())

                    # Track stat matching error
                    stat_error = torch.mean((learned_stats - true_stats) ** 2).item()
                    total_stat_error += stat_error
                    num_batches += 1

            val_aupr = average_precision_score(all_labels, all_preds)
            avg_stat_error = total_stat_error / num_batches
            scheduler.step(val_aupr)

            print(f"    Pretrain Epoch {epoch+1} | Val AUPR: {val_aupr:.4f} | Stat MSE: {avg_stat_error:.6f}")

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

    print(f"  [Statistical Matching Pretraining] Completed. Best Val AUPR: {best_auc:.4f}")
    return policy_net, best_auc


# ==============================================================================
# 5. Feature Extraction
# ==============================================================================

def extract_learned_features(policy_net, loader, deterministic=False, temperature=1.0):
    """Extract [Static + Last + Learned_Stats + Z]"""
    policy_net.eval() if deterministic else policy_net.train()

    all_features = []
    all_labels = []
    all_log_probs = []

    with torch.set_grad_enabled(not deterministic):
        for t_data, labels, s_data, true_stats, last_vals in loader:
            s_data = s_data.to(DEVICE)

            learned_stats, z, log_prob, mean = policy_net(
                t_data, deterministic=deterministic, temperature=temperature
            )

            stats_np = learned_stats.detach().cpu().numpy()
            z_np = (mean if deterministic else z).detach().cpu().numpy()
            s_np = s_data.cpu().numpy()
            last_np = last_vals.numpy()

            # [Static + Last + Learned_Stats + Z]
            combined = np.hstack([s_np, last_np, stats_np, z_np])

            all_features.append(combined)
            all_labels.extend(labels.numpy())

            if not deterministic and log_prob is not None:
                all_log_probs.append(log_prob)

    features = np.vstack(all_features)
    labels = np.array(all_labels)
    log_probs = torch.cat(all_log_probs) if all_log_probs else None

    return features, labels, log_probs


# ==============================================================================
# 6. RL Training
# ==============================================================================

def train_policy_rl(policy_net, train_loader, val_loader, tabpfn_params,
                    pretrain_aupr, epochs=120, update_tabpfn_every=5):
    if pretrain_aupr < 0.75:
        lr = 0.0002
        print(f"  [RL Training] Using higher LR (0.0002)")
    else:
        lr = 0.0001
        print(f"  [RL Training] Using standard LR (0.0001)")

    optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr)

    best_val_auc = 0
    best_state = None
    patience = 30
    patience_counter = 0

    print(f"  [RL Training] Training for up to {epochs} epochs...")

    tabpfn_model = None
    val_history = []

    for epoch in range(epochs):
        temperature = max(0.25, 0.7 - epoch / (epochs * 0.6))

        policy_net.train()

        X_train, y_train, log_probs_train = extract_learned_features(
            policy_net, train_loader, deterministic=False, temperature=temperature
        )

        if epoch % update_tabpfn_every == 0 or tabpfn_model is None:
            tabpfn_model = TabPFNClassifier(**tabpfn_params)
            tabpfn_model.fit(X_train, y_train)

        X_val_stoch, y_val, _ = extract_learned_features(
            policy_net, val_loader, deterministic=False, temperature=temperature
        )

        y_val_proba = tabpfn_model.predict_proba(X_val_stoch)[:, 1]
        val_aupr = average_precision_score(y_val, y_val_proba)
        val_history.append(val_aupr)

        y_train_proba = tabpfn_model.predict_proba(X_train)[:, 1]
        rewards_smooth = np.where(y_train == 1, y_train_proba, 1 - y_train_proba)

        aupr_weight = 0.3 + 0.3 * min(1.0, epoch / 50)
        rewards_combined = rewards_smooth + aupr_weight * val_aupr

        rewards_tensor = torch.tensor(rewards_combined, dtype=torch.float32).to(DEVICE)
        rewards_tensor = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std() + 1e-8)

        log_probs_train = log_probs_train.to(DEVICE)
        policy_loss = -(log_probs_train * rewards_tensor).mean()

        entropy_bonus = 0.001 * log_probs_train.mean()
        total_loss = policy_loss - entropy_bonus

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=0.5)
        optimizer.step()

        if (epoch + 1) % 5 == 0:
            policy_net.eval()
            with torch.no_grad():
                X_val_det, y_val_det, _ = extract_learned_features(
                    policy_net, val_loader, deterministic=True
                )

                X_train_det, y_train_det, _ = extract_learned_features(
                    policy_net, train_loader, deterministic=True
                )

                tabpfn_val_model = TabPFNClassifier(**tabpfn_params)
                tabpfn_val_model.fit(X_train_det, y_train_det)

                y_val_proba_det = tabpfn_val_model.predict_proba(X_val_det)[:, 1]

                val_auc = roc_auc_score(y_val_det, y_val_proba_det)
                val_aupr_det = average_precision_score(y_val_det, y_val_proba_det)

                recent_avg = np.mean(val_history[-10:]) if len(val_history) >= 10 else val_aupr_det

                print(f"    Epoch {epoch+1:3d} | Temp: {temperature:.3f} | "
                      f"Val AUC: {val_auc:.4f} | Val AUPR: {val_aupr_det:.4f} | Avg: {recent_avg:.4f}")

                if val_aupr_det > best_val_auc:
                    best_val_auc = val_aupr_det
                    best_state = copy.deepcopy(policy_net.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"    Early stopping at epoch {epoch+1}")
                        break

    if best_state is not None:
        policy_net.load_state_dict(best_state)

    return policy_net


# ==============================================================================
# 7. Main Execution
# ==============================================================================

def main():
    print("="*80)
    print("TabX: Learnable Statistical Feature Aggregator")
    print("Multi-Head Architecture Learning Mean, Std, Min, Max, Slope, etc.")
    print("="*80)

    patients = load_and_prepare_patients()
    temporal_feats = get_all_temporal_features(patients)

    print("Extracting and encoding static features...")
    df_static_all = patients.getMeasuresBetween(
        pd.Timedelta(hours=-6), pd.Timedelta(hours=24), "last", getUntilAkiPositive=True
    )
    static_cols = [col for col in FIXED_FEATURES if col in df_static_all.columns]
    df_static_all = df_static_all[static_cols]

    print(f"Input: {len(temporal_feats)} Temporal + {len(static_cols)} Static Features")
    print(f"Learning 7 statistics per temporal feature: last, mean, std, min, max, slope, count")

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

        # Extract and encode static features
        df_static_train_raw = train_p_obj.getMeasuresBetween(
            pd.Timedelta(hours=-6), pd.Timedelta(hours=24), "last", getUntilAkiPositive=True
        )
        df_static_val_raw = val_p_obj.getMeasuresBetween(
            pd.Timedelta(hours=-6), pd.Timedelta(hours=24), "last", getUntilAkiPositive=True
        )
        df_static_test_raw = test_p.getMeasuresBetween(
            pd.Timedelta(hours=-6), pd.Timedelta(hours=24), "last", getUntilAkiPositive=True
        )

        df_static_train = df_static_train_raw[static_cols].reset_index(drop=True)
        df_static_val = df_static_val_raw[static_cols].reset_index(drop=True)
        df_static_test = df_static_test_raw[static_cols].reset_index(drop=True)

        df_static_train_enc, df_static_test_enc, df_static_val_enc = encodeCategoricalData(
            df_static_train, df_static_test, df_static_val
        )

        df_static_train_enc = df_static_train_enc.fillna(0)
        df_static_val_enc = df_static_val_enc.fillna(0)
        df_static_test_enc = df_static_test_enc.fillna(0)

        # Create datasets
        train_ds = StatisticalDataset(train_p, temporal_feats, df_static_train_enc)
        stats = train_ds.get_normalization_stats()
        val_ds = StatisticalDataset(val_p, temporal_feats, df_static_val_enc, stats)
        test_ds = StatisticalDataset(test_p_list, temporal_feats, df_static_test_enc, stats)

        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=statistical_collate_fn)
        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=statistical_collate_fn)
        test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=statistical_collate_fn)

        static_dim = df_static_train_enc.shape[1]

        print(f"  Static features encoded dimension: {static_dim}")

        # Create policy network with learned statistics
        latent_dim = 32
        policy_net = PolicyNetworkWithLearnedStats(
            num_features=len(temporal_feats),
            hidden_dim=32,
            latent_dim=latent_dim
        ).to(DEVICE)

        tabpfn_params = {
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }

        # STEP 1: Pretraining with statistical matching
        policy_net, pretrain_aupr = pretrain_with_stat_matching(
            policy_net, train_loader, val_loader, static_dim, epochs=50
        )

        # STEP 2: RL Fine-tuning
        policy_net = train_policy_rl(
            policy_net, train_loader, val_loader, tabpfn_params,
            pretrain_aupr, epochs=120, update_tabpfn_every=5
        )

        # RL Evaluation
        print("\n  [RL Test Evaluation]")
        policy_net.eval()

        with torch.no_grad():
            X_train_final, y_train_final, _ = extract_learned_features(
                policy_net, train_loader, deterministic=True
            )
            X_test_final, y_test_final, _ = extract_learned_features(
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

        print(f"  RL Test AUC: {fold_auc:.4f} | Test AUPR: {fold_aupr:.4f}")

        # BASELINE
        print("\n  [Baseline] Simple [Static + Last]...")

        df_train_temp = train_p_obj.getMeasuresBetween(
            pd.Timedelta(hours=-6), pd.Timedelta(hours=24), "last", getUntilAkiPositive=True
        ).drop(columns=["subject_id", "hadm_id", "stay_id"])
        df_test_temp = test_p.getMeasuresBetween(
            pd.Timedelta(hours=-6), pd.Timedelta(hours=24), "last", getUntilAkiPositive=True
        ).drop(columns=["subject_id", "hadm_id", "stay_id"])

        df_train_enc, df_test_enc, _ = encodeCategoricalData(df_train_temp, df_test_temp)

        X_train_baseline = df_train_enc.drop(columns=["akd"]).fillna(0)
        y_train_baseline = df_train_enc["akd"]
        X_test_baseline = df_test_enc.drop(columns=["akd"]).fillna(0)
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

        print(f"  Baseline Test AUC: {baseline_auc:.4f} | Test AUPR: {baseline_aupr:.4f}")
        print(f"  Fold {fold} Results -> RL: {fold_auc:.3f} vs Baseline: {baseline_auc:.3f}")

    # Final Plot
    for ax in [ax1, ax2]:
        ax.plot([0, 1], [0, 1], linestyle="--", color="navy", lw=2)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(loc="lower right")

    ax1.set_title("TabX [Static+Last+LearnedStats+Z]")
    ax2.set_title("Baseline [Static+Last]")
    plt.tight_layout()
    plt.savefig("result/tabx_learned_statistics.png", dpi=300)
    print("\nPlot saved to result/tabx_learned_statistics.png")

    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY")
    print("="*80)

    def print_stat(name, rl_metrics, base_metrics):
        rl_mean, rl_std = np.mean(rl_metrics), np.std(rl_metrics)
        base_mean, base_std = np.mean(base_metrics), np.std(base_metrics)
        improvement = ((rl_mean - base_mean) / base_mean) * 100
        symbol = "✓" if rl_mean > base_mean else "✗"
        print(f"{name:15s} | RL: {rl_mean:.4f} ± {rl_std:.4f}  vs  Baseline: {base_mean:.4f} ± {base_std:.4f}  ({improvement:+.2f}%) {symbol}")

    print_stat("AUC", metrics_rl['auc'], metrics_baseline['auc'])
    print_stat("AUC-PR", metrics_rl['auc_pr'], metrics_baseline['auc_pr'])

    print("\n" + "="*80)
    print("TabX INNOVATION:")
    print("✓ Multi-head attention learns to compute statistics")
    print("✓ Auxiliary loss matches learned stats to handcrafted")
    print("✓ Can discover better aggregations than manual features")
    print("✓ End-to-end learnable, no feature engineering")
    print("="*80)

if __name__ == "__main__":
    main()
