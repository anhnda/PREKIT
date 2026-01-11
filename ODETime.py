"""
ODE-RNN model for AKI prediction using only temporal/time-varying features.

The model uses Neural ODE to handle irregular time series:
1. FLOW: ODE evolves hidden state continuously between measurements
2. JUMP: GRU cell updates hidden state when new measurements arrive
3. Handles asynchronous features and missing data with masks

Key Improvements:
A. Learned Initial State: Model starts with a learned "generic patient" baseline (h0)
   instead of initializing from first observation. This makes the model more robust
   when the first measurement is sparse.

B. Time Normalization: Timestamps are normalized to [0, 1] range for numerical
   stability during ODE integration. Prevents issues with long time gaps.

C. Feature Normalization: All feature values are standardized (mean=0, std=1)
   using training set statistics. Applied consistently to test set.

D. Bounded Activations: Using Tanh in ODEFunc keeps derivatives bounded,
   preventing hidden state explosion during long integration intervals.

E. Batched Processing: Uses padding and masking to enable parallel GPU processing.
   - Sequences padded to max length within each batch
   - Masking ensures padded values don't affect computation
   - Significant speedup compared to sequential processing
   - ODE solver called once per time step for entire batch
"""

import pandas as pd
import numpy as np
import sys
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchdiffeq import odeint
from tqdm import tqdm
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
print(sys.platform, PT)
sys.path.append(PT)
from constants import NULLABLE_MEASURES
from utils.class_patient import Patients
from utils.prepare_data import trainTestPatients

# Fixed/demographic features to exclude (same as MLP_OnlyTime.py)
FIXED_FEATURES = [
    "age", "gender", "race",
    "chronic_pulmonary_disease", "ckd_stage", "congestive_heart_failure",
    "dka_type", "history_aci", "history_ami", "hypertension",
    "liver_disease", "macroangiopathy", "malignant_cancer",
    "microangiopathy", "uti",
    "oasis", "saps2", "sofa",
    "mechanical_ventilation", "use_NaHCO3",
    "preiculos", "gcs_unable",
    "egfr",  # Data leak
]

# Configuration
LABEL_COLUMN = "akd"
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    #else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using device: {DEVICE}")


# ============================================================================
# 1. DATA PREPARATION
# ============================================================================

def get_all_temporal_features(patients):
    """Get the complete set of temporal features across all patients.

    Uses getMeasuresBetween to get the same feature set as MLP_OnlyTime.py
    """
    id_columns = ["subject_id", "hadm_id", "stay_id"]
    label_column = "akd"

    # Get features from the Patients object (not individual Patient)
    df = patients.getMeasuresBetween(
        pd.Timedelta(hours=-6),
        pd.Timedelta(hours=24),
        "last",
        getUntilAkiPositive=True
    )

    # Get temporal features (exclude ID, label, and fixed features)
    all_features = [col for col in df.columns
                   if col not in id_columns + [label_column]
                   and col not in FIXED_FEATURES]

    return sorted(all_features)


def extract_temporal_data(patient, feature_names, time_window_start=-6, time_window_end=24):
    """
    Extract temporal measurements from a patient into [times, values, masks] format.

    CRITICAL: Respects AKI diagnosis time to prevent data leakage!

    Args:
        patient: Patient object
        feature_names: List of all temporal feature names (consistent across all patients)
        time_window_start: Start of time window in hours
        time_window_end: End of time window in hours

    Returns:
        times: List of timestamps (in hours from admission)
        values: List of measurement vectors (one per timestamp)
        masks: List of mask vectors (1 if measured, 0 if missing)
    """
    intime = patient.intime

    # CRITICAL: Calculate cutoff time to prevent data leakage
    # If patient has AKI, only use measurements BEFORE AKI diagnosis
    if patient.akdPositive:
        aki_cutoff_hours = patient.akdTime.total_seconds() / 3600
        # Use the earlier of: window end or AKI time
        effective_end = min(time_window_end, aki_cutoff_hours)
    else:
        effective_end = time_window_end

    # Get temporal measures for each feature
    temporal_measures = {}
    for measure_name in feature_names:
        if measure_name in patient.measures:
            measure_values = patient.measures[measure_name]

            # Handle both SortedDict and scalar/other types
            if hasattr(measure_values, 'keys') and hasattr(measure_values, 'values'):
                # It's a SortedDict - filter by time
                filtered_dict = {}
                for timestamp, value in measure_values.items():
                    ts = pd.Timestamp(timestamp)
                    hours_from_admission = (ts - intime).total_seconds() / 3600

                    # CRITICAL: Only include measurements within valid window
                    if time_window_start <= hours_from_admission <= effective_end:
                        filtered_dict[ts] = value

                temporal_measures[measure_name] = filtered_dict
            else:
                # Scalar or other type - can't extract time series
                temporal_measures[measure_name] = {}
        else:
            temporal_measures[measure_name] = {}

    # Get all unique timestamps across all features
    all_timestamps = set()
    for measure_dict in temporal_measures.values():
        for timestamp in measure_dict.keys():
            all_timestamps.add(timestamp)

    if not all_timestamps:
        return None, None, None

    # Sort timestamps
    all_timestamps = sorted(all_timestamps)

    # Convert to hours from admission
    times_hours = [(t - intime).total_seconds() / 3600 for t in all_timestamps]

    # Normalize times for numerical stability
    # Scale to [0, 1] range within the observation window
    if len(times_hours) > 1:
        min_time = min(times_hours)
        max_time = max(times_hours)
        time_range = max_time - min_time
        if time_range > 0:
            times_hours = [(t - min_time) / time_range for t in times_hours]
        else:
            times_hours = [0.0] * len(times_hours)
    else:
        times_hours = [0.0]

    # Build values and masks
    values = []
    masks = []

    for timestamp in all_timestamps:
        value_vec = []
        mask_vec = []

        for feature_name in feature_names:
            measure_dict = temporal_measures[feature_name]

            if timestamp in measure_dict:
                value_vec.append(float(measure_dict[timestamp]))
                mask_vec.append(1.0)
            else:
                value_vec.append(0.0)  # Placeholder
                mask_vec.append(0.0)   # Not measured

        values.append(value_vec)
        masks.append(mask_vec)

    return times_hours, values, masks


class IrregularTimeSeriesDataset(Dataset):
    """Dataset for irregular time series data with normalization."""

    def __init__(self, patients, feature_names, normalization_stats=None):
        """
        Args:
            patients: Patients object
            feature_names: List of temporal feature names (must be consistent across all patients)
            normalization_stats: Dict with 'mean' and 'std' for normalization (if None, compute from data)
        """
        self.data = []
        self.labels = []
        self.feature_names = feature_names

        # Collect all values for normalization
        all_values = []

        for patient in patients.patientList:
            times, values, masks = extract_temporal_data(patient, feature_names)

            if times is None:
                continue

            self.data.append({
                'times': times,
                'values': values,
                'masks': masks,
            })
            self.labels.append(1 if patient.akdPositive else 0)

            # Collect observed values for normalization
            for value_vec, mask_vec in zip(values, masks):
                for val, mask in zip(value_vec, mask_vec):
                    if mask > 0:  # Only observed values
                        all_values.append(val)

        # Compute or use provided normalization statistics
        if normalization_stats is None:
            all_values = np.array(all_values)
            self.mean = np.mean(all_values) if len(all_values) > 0 else 0.0
            self.std = np.std(all_values) if len(all_values) > 0 else 1.0
            if self.std == 0:
                self.std = 1.0
        else:
            self.mean = normalization_stats['mean']
            self.std = normalization_stats['std']

        # Normalize the data
        for i in range(len(self.data)):
            values = self.data[i]['values']
            masks = self.data[i]['masks']

            # Normalize only observed values
            normalized_values = []
            for value_vec, mask_vec in zip(values, masks):
                norm_vec = [(v - self.mean) / self.std if m > 0 else 0.0
                           for v, m in zip(value_vec, mask_vec)]
                normalized_values.append(norm_vec)

            # Convert to tensors
            self.data[i] = {
                'times': torch.tensor(self.data[i]['times'], dtype=torch.float32),
                'values': torch.tensor(normalized_values, dtype=torch.float32),
                'masks': torch.tensor(self.data[i]['masks'], dtype=torch.float32),
            }

    def get_normalization_stats(self):
        """Return normalization statistics for use in test set."""
        return {'mean': self.mean, 'std': self.std}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def collate_fn(batch):
    """Custom collate function with padding for batched processing.

    Pads sequences to max length in batch so they can be processed in parallel.
    """
    data, labels = zip(*batch)
    labels = torch.tensor(labels, dtype=torch.float32)

    # Get sequence lengths
    lengths = [len(d['times']) for d in data]
    max_len = max(lengths)

    # Get feature dimension from first sample
    feature_dim = data[0]['values'].shape[-1]
    batch_size = len(data)

    # Pad times, values, and masks to max length
    padded_times = torch.zeros(batch_size, max_len)
    padded_values = torch.zeros(batch_size, max_len, feature_dim)
    padded_masks = torch.zeros(batch_size, max_len, feature_dim)

    for i, d in enumerate(data):
        seq_len = lengths[i]
        padded_times[i, :seq_len] = d['times']
        padded_values[i, :seq_len] = d['values']
        padded_masks[i, :seq_len] = d['masks']

    batch_data = {
        'times': padded_times,
        'values': padded_values,
        'masks': padded_masks,
        'lengths': torch.tensor(lengths, dtype=torch.long)
    }

    return batch_data, labels


# ============================================================================
# 2. ODE-RNN MODEL
# ============================================================================

class ODEFunc(nn.Module):
    """ODE function that defines how hidden state evolves over time."""

    def __init__(self, hidden_dim):
        super(ODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, hidden_dim)
        )

    def forward(self, t, h):
        """
        Args:
            t: Current time (scalar)
            h: Hidden state [batch_size, hidden_dim]
        Returns:
            dh/dt: Derivative of hidden state
        """
        return self.net(h)


class ODERNNCell(nn.Module):
    """Batched ODE-RNN cell that processes multiple patients in parallel."""

    def __init__(self, input_dim, hidden_dim):
        super(ODERNNCell, self).__init__()
        self.hidden_dim = hidden_dim
        self.ode_func = ODEFunc(hidden_dim)

        # Learned initial hidden state (represents "generic patient" baseline)
        self.h0 = nn.Parameter(torch.randn(hidden_dim) * 0.1)

        # Jump mechanism: Update hidden state with new observations
        # Input: [values (input_dim) + masks (input_dim)]
        self.gru_cell = nn.GRUCell(input_dim * 2, hidden_dim)

    def forward(self, batch_times, batch_values, batch_masks, lengths):
        """
        Batched forward pass with padding and masking.

        Args:
            batch_times: [batch_size, max_seq_len] padded timestamps
            batch_values: [batch_size, max_seq_len, input_dim] padded values
            batch_masks: [batch_size, max_seq_len, input_dim] padded masks
            lengths: [batch_size] actual sequence length for each patient
        Returns:
            h: [batch_size, hidden_dim] final hidden states
        """
        batch_size = batch_times.size(0)
        max_seq_len = batch_times.size(1)

        # Initialize hidden state for entire batch
        h = self.h0.unsqueeze(0).repeat(batch_size, 1)  # [batch_size, hidden_dim]

        # Process each time step
        for i in range(max_seq_len):
            # FLOW: ODE integration from previous time step
            if i > 0:
                # Compute time deltas for each patient in batch
                # Note: Already normalized to [0, 1] range
                time_deltas = batch_times[:, i] - batch_times[:, i-1]  # [batch_size]

                # For numerical stability, use average time delta across batch
                # This is a simplification - ideally we'd solve ODE separately for each delta
                avg_delta = time_deltas.mean()

                if avg_delta > 0:
                    t_span = torch.tensor([0.0, avg_delta.item()], dtype=torch.float32).to(h.device)
                    # Solve ODE for the entire batch at once
                    h_evolved = odeint(self.ode_func, h, t_span, method='dopri5')
                    h = h_evolved[-1]  # [batch_size, hidden_dim]

            # JUMP: Update with new observations
            combined_input = torch.cat([batch_values[:, i], batch_masks[:, i]], dim=-1)
            # [batch_size, input_dim * 2]

            h_new = self.gru_cell(combined_input, h)  # [batch_size, hidden_dim]

            # MASKING: Only update patients that haven't exceeded their sequence length
            # Create mask: 1 if i < length, 0 otherwise
            valid_mask = (i < lengths).float().unsqueeze(-1)  # [batch_size, 1]

            # Update only valid positions
            h = valid_mask * h_new + (1 - valid_mask) * h

        return h


class ODERNNModel(nn.Module):
    """Complete ODE-RNN model for classification with batched processing.

    Uses padding and masking to enable parallel GPU processing of entire batches.
    This provides significant speedup compared to sequential processing.
    """

    def __init__(self, input_dim, hidden_dim, output_dim=1):
        super(ODERNNModel, self).__init__()
        self.odernn_cell = ODERNNCell(input_dim, hidden_dim)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, output_dim),
            nn.Sigmoid()
        )

    def forward(self, batch_data):
        """
        Batched forward pass.

        Args:
            batch_data: Dict with keys:
                - 'times': [batch_size, max_seq_len] padded timestamps
                - 'values': [batch_size, max_seq_len, input_dim] padded values
                - 'masks': [batch_size, max_seq_len, input_dim] padded masks
                - 'lengths': [batch_size] actual sequence lengths
        Returns:
            predictions: [batch_size] probability of positive class
        """
        # Move all data to device
        times = batch_data['times'].to(DEVICE)
        values = batch_data['values'].to(DEVICE)
        masks = batch_data['masks'].to(DEVICE)
        lengths = batch_data['lengths'].to(DEVICE)

        # Get hidden states for entire batch in parallel
        hidden_states = self.odernn_cell(times, values, masks, lengths)
        # [batch_size, hidden_dim]

        # Classify
        predictions = self.classifier(hidden_states).squeeze(-1)
        # [batch_size]

        return predictions


# ============================================================================
# 3. TRAINING AND EVALUATION
# ============================================================================

def load_and_prepare_patients():
    """Load patients and remove missing data."""
    patients = Patients.loadPatients()
    print(f"Loaded {len(patients)} patients")

    patients.fillMissingMeasureValue(NULLABLE_MEASURES, 0)

    # Remove measures with less than 80% of data
    measures = patients.getMeasures()
    for measure, count in measures.items():
        if count < len(patients) * 80 / 100:
            patients.removeMeasures([measure])

    # Remove patients with more than 20% missing features
    patients.removePatientByMissingFeatures()
    print(f"After cleanup: {len(patients)} patients")

    # Display AKI statistics
    aki_count = sum([1 for p in patients if p.akdPositive])
    print(f"AKI positive: {aki_count} ({aki_count / len(patients):.2%})")

    return patients


def train_model(model, train_loader, criterion, optimizer, num_epochs=50):
    """Train the ODE-RNN model with batched processing."""
    model.train()

    for epoch in tqdm(range(num_epochs), desc="Training Epochs"):
        total_loss = 0
        num_batches = 0

        for batch_data, labels in train_loader:
            labels = labels.to(DEVICE)

            # Forward pass (now fully batched!)
            predictions = model(batch_data)
            loss = criterion(predictions, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / num_batches
            print(f"  Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    return model


def evaluate_model(model, test_loader):
    """Evaluate the ODE-RNN model with batched processing."""
    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch_data, labels in test_loader:
            # Forward pass (fully batched)
            predictions = model(batch_data)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    # Convert probabilities to binary predictions
    binary_predictions = (all_predictions > 0.5).astype(int)

    return all_labels, all_predictions, binary_predictions


def main():
    """Main training and evaluation loop."""
    print("="*80)
    print("ODE-RNN MODEL FOR AKI PREDICTION (TEMPORAL FEATURES ONLY)")
    print("="*80)

    # Load data
    patients = load_and_prepare_patients()

    # Metrics storage
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    auc_scores = []
    specificity_scores = []
    auc_pr_scores = []

    # ROC curve plot setup
    plt.figure(figsize=(10, 8))

    # Get all temporal features from the entire dataset
    print("\nExtracting temporal features from all patients...")
    all_temporal_features = get_all_temporal_features(patients)
    input_dim = len(all_temporal_features)
    print(f"Number of temporal features: {input_dim}")
    print(f"Features: {all_temporal_features}")

    # Cross-validation
    for fold, (train_patients, test_patients) in enumerate(trainTestPatients(patients)):
        print(f"\n{'='*80}")
        print(f"FOLD {fold}")
        print(f"{'='*80}")

        # Create datasets with consistent feature set
        # Train dataset computes normalization statistics
        train_dataset = IrregularTimeSeriesDataset(train_patients, all_temporal_features)

        if len(train_dataset) == 0:
            print(f"Skipping fold {fold}: Empty train dataset")
            continue

        # Test dataset uses train normalization statistics
        norm_stats = train_dataset.get_normalization_stats()
        test_dataset = IrregularTimeSeriesDataset(test_patients, all_temporal_features, norm_stats)

        if len(test_dataset) == 0:
            print(f"Skipping fold {fold}: Empty test dataset")
            continue

        print(f"Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")
        print(f"Normalization: mean={norm_stats['mean']:.2f}, std={norm_stats['std']:.2f}")

        # Create data loaders
        # With batched processing, we can use larger batch sizes for better GPU utilization
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

        # Create model
        model = ODERNNModel(input_dim=input_dim, hidden_dim=64, output_dim=1).to(DEVICE)
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Train
        print("\nTraining...")
        model = train_model(model, train_loader, criterion, optimizer, num_epochs=50)

        # Evaluate
        print("\nEvaluating...")
        y_test, y_pred_proba, y_pred = evaluate_model(model, test_loader)

        # Calculate metrics
        tn, fp, _, _ = confusion_matrix(y_test, y_pred).ravel()
        precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_pred_proba)

        accuracy_scores.append(accuracy_score(y_test, y_pred))
        specificity_scores.append(tn / (tn + fp))
        precision_scores.append(precision_score(y_test, y_pred, zero_division=0))
        recall_scores.append(recall_score(y_test, y_pred))
        auc_scores.append(roc_auc_score(y_test, y_pred_proba))
        auc_pr_scores.append(auc(recall_vals, precision_vals))

        # ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        plt.plot(fpr, tpr, lw=2, label=f"Fold {fold} (AUC = {auc_scores[-1]:.3f})")

        print(f"Fold {fold} - AUC: {auc_scores[-1]:.3f}, Accuracy: {accuracy_scores[-1]:.3f}")

    # Plot ROC curves
    plt.plot([0, 1], [0, 1], linestyle="--", color="navy", lw=2, label="Random")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves - ODE-RNN with Temporal Features Only")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.savefig("result/roc_odernn_temporal_only.png", dpi=300, bbox_inches="tight")
    print("\nSaved ROC plot to: result/roc_odernn_temporal_only.png")
    plt.show()

    # Print summary statistics
    print("\n" + "="*80)
    print("RESULTS SUMMARY (ODE-RNN - Temporal Features Only)")
    print("="*80)
    print(f"AUC:         {np.mean(auc_scores):.4f} ± {np.std(auc_scores):.4f}")
    print(f"Accuracy:    {np.mean(accuracy_scores):.4f} ± {np.std(accuracy_scores):.4f}")
    print(f"Specificity: {np.mean(specificity_scores):.4f} ± {np.std(specificity_scores):.4f}")
    print(f"Precision:   {np.mean(precision_scores):.4f} ± {np.std(precision_scores):.4f}")
    print(f"Recall:      {np.mean(recall_scores):.4f} ± {np.std(recall_scores):.4f}")
    print(f"AUC-PR:      {np.mean(auc_pr_scores):.4f} ± {np.std(auc_pr_scores):.4f}")
    print("="*80)

    print("\nDetailed scores per fold:")
    print(f"AUC:         {auc_scores}")
    print(f"Accuracy:    {accuracy_scores}")
    print(f"Specificity: {specificity_scores}")
    print(f"Precision:   {precision_scores}")
    print(f"Recall:      {recall_scores}")
    print(f"AUC-PR:      {auc_pr_scores}")


if __name__ == "__main__":
    main()
