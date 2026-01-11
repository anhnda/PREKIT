"""
MLP with Time-Aware Features for AKI Prediction.

Instead of just using "last" aggregation, extract temporal statistics:
- last_value: Most recent measurement
- first_value: Initial measurement
- mean_value: Average over window
- std_value: Variation
- trend: Linear slope
- count: Number of observations

This captures temporal patterns while keeping MLP's simplicity.
"""

import pandas as pd
import numpy as np
import sys
from matplotlib import pyplot as plt
from sklearn.neural_network import MLPClassifier
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
from scipy import stats

PT = "/Users/anhnd/CodingSpace/Python/PREDKIT"
if sys.platform != "darwin":
    PT = "/home/anhnda/PREKIT"
sys.path.append(PT)

from constants import NULLABLE_MEASURES
from utils.class_patient import Patients
from utils.prepare_data import encodeCategoricalData, trainTestPatients

# Fixed features (same as MLP_OnlyTime.py)
FIXED_FEATURES = [
    "age", "gender", "race",
    "chronic_pulmonary_disease", "ckd_stage", "congestive_heart_failure",
    "dka_type", "history_aci", "history_ami", "hypertension",
    "liver_disease", "macroangiopathy", "malignant_cancer",
    "microangiopathy", "uti",
    "oasis", "saps2", "sofa",
    "mechanical_ventilation", "use_NaHCO3",
    "preiculos", "gcs_unable",
    "egfr",
]

ID_COLUMNS = ["subject_id", "hadm_id", "stay_id"]
LABEL_COLUMN = "akd"


def extract_temporal_features(patients, time_window_start=-6, time_window_end=24):
    """
    Extract temporal statistics for each patient.

    For each feature, compute:
    - last: Most recent value (like original MLP)
    - first: Initial value in window
    - mean: Average value
    - std: Standard deviation
    - trend: Linear trend (slope)
    - count: Number of observations
    """
    data = []

    for patient in patients.patientList:
        intime = patient.intime
        patient_data = {'subject_id': patient.subject_id}

        # Calculate AKI cutoff if positive
        if patient.akdPositive:
            aki_cutoff_hours = patient.akdTime.total_seconds() / 3600
            effective_end = min(time_window_end, aki_cutoff_hours)
        else:
            effective_end = time_window_end

        # For each temporal feature
        for measure_name, measure_values in patient.measures.items():
            if measure_name in FIXED_FEATURES:
                continue

            # Check if it's a temporal feature (SortedDict)
            if not (hasattr(measure_values, 'keys') and hasattr(measure_values, 'values')):
                continue

            # Extract measurements within time window
            measurements = []
            timestamps = []

            for timestamp, value in measure_values.items():
                ts = pd.Timestamp(timestamp)
                hours_from_admission = (ts - intime).total_seconds() / 3600

                if time_window_start <= hours_from_admission <= effective_end:
                    measurements.append(float(value))
                    timestamps.append(hours_from_admission)

            # Compute temporal statistics
            if len(measurements) == 0:
                # No measurements - all NaN
                patient_data[f'{measure_name}_last'] = np.nan
                patient_data[f'{measure_name}_first'] = np.nan
                patient_data[f'{measure_name}_mean'] = np.nan
                patient_data[f'{measure_name}_std'] = np.nan
                patient_data[f'{measure_name}_trend'] = np.nan
                patient_data[f'{measure_name}_count'] = 0
            elif len(measurements) == 1:
                # Single measurement
                patient_data[f'{measure_name}_last'] = measurements[0]
                patient_data[f'{measure_name}_first'] = measurements[0]
                patient_data[f'{measure_name}_mean'] = measurements[0]
                patient_data[f'{measure_name}_std'] = 0
                patient_data[f'{measure_name}_trend'] = 0
                patient_data[f'{measure_name}_count'] = 1
            else:
                # Multiple measurements
                patient_data[f'{measure_name}_last'] = measurements[-1]
                patient_data[f'{measure_name}_first'] = measurements[0]
                patient_data[f'{measure_name}_mean'] = np.mean(measurements)
                patient_data[f'{measure_name}_std'] = np.std(measurements)

                # Linear trend (slope)
                slope, _, _, _, _ = stats.linregress(timestamps, measurements)
                patient_data[f'{measure_name}_trend'] = slope

                patient_data[f'{measure_name}_count'] = len(measurements)

        # Add label
        patient_data[LABEL_COLUMN] = 1 if patient.akdPositive else 0

        data.append(patient_data)

    return pd.DataFrame(data)


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

    aki_count = sum([1 for p in patients if p.akdPositive])
    print(f"AKI positive: {aki_count} ({aki_count / len(patients):.2%})")

    return patients


def create_model():
    """Create MLP classifier."""
    return MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42)


def evaluate_model():
    """Train and evaluate MLP model with temporal features."""
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

    # Cross-validation
    for fold, (train_patients, test_patients) in enumerate(trainTestPatients(patients)):
        print(f"\n--- Fold {fold} ---")

        # Extract temporal features
        print("Extracting temporal features...")
        df_train = extract_temporal_features(train_patients)
        df_test = extract_temporal_features(test_patients)

        # Remove ID columns
        df_train = df_train.drop(columns=['subject_id'])
        df_test = df_test.drop(columns=['subject_id'])

        # Prepare X and y
        X_train = df_train.drop(columns=[LABEL_COLUMN])
        y_train = df_train[LABEL_COLUMN]

        X_test = df_test.drop(columns=[LABEL_COLUMN])
        y_test = df_test[LABEL_COLUMN]

        # Fill missing values with 0
        X_train = X_train.fillna(0)
        X_test = X_test.fillna(0)

        print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
        print(f"Total features: {X_train.shape[1]} (was 25 base features)")

        # Train model
        model = create_model()
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

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
    plt.title("ROC Curves - MLP with Temporal Features")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.savefig("result/roc_mlp_timeaware.png", dpi=300, bbox_inches="tight")
    print("\nSaved ROC plot to: result/roc_mlp_timeaware.png")
    plt.show()

    # Print summary statistics
    print("\n" + "="*80)
    print("RESULTS SUMMARY (MLP with Temporal Features)")
    print("="*80)
    print(f"AUC:         {np.mean(auc_scores):.4f} ± {np.std(auc_scores):.4f}")
    print(f"Accuracy:    {np.mean(accuracy_scores):.4f} ± {np.std(accuracy_scores):.4f}")
    print(f"Specificity: {np.mean(specificity_scores):.4f} ± {np.std(specificity_scores):.4f}")
    print(f"Precision:   {np.mean(precision_scores):.4f} ± {np.std(precision_scores):.4f}")
    print(f"Recall:      {np.mean(recall_scores):.4f} ± {np.std(recall_scores):.4f}")
    print(f"AUC-PR:      {np.mean(auc_pr_scores):.4f} ± {np.std(auc_pr_scores):.4f}")
    print("="*80)

    print("\nComparison:")
    print("  MLP (last only):     AUC 0.770")
    print("  ODE-RNN:             AUC 0.746")
    print(f"  MLP (time-aware):    AUC {np.mean(auc_scores):.3f}")


if __name__ == "__main__":
    evaluate_model()
