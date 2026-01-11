"""
MLP model for AKI prediction using only temporal/time-varying features.
Excludes fixed demographic features like age, gender, medical history, etc.
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

sys.path.append("/Users/anhnd/CodingSpace/Python/PREDKIT")
from constants import NULLABLE_MEASURES
from utils.class_patient import Patients
from utils.prepare_data import encodeCategoricalData, trainTestPatients

# Fixed/demographic features to exclude (not time-varying)
# These are features WITHOUT timestamps - they're constant for each patient
FIXED_FEATURES = [
    # Demographics
    "age",
    "gender",
    "race",
    # Medical history (fixed at admission)
    "chronic_pulmonary_disease",
    "ckd_stage",
    "congestive_heart_failure",
    "dka_type",
    "history_aci",
    "history_ami",
    "hypertension",
    "liver_disease",
    "macroangiopathy",
    "malignant_cancer",
    "microangiopathy",
    "uti",
    # Severity scores (calculated at admission, not time-varying)
    "oasis",
    "saps2",
    "sofa",
    # Interventions that may not have timestamps
    "mechanical_ventilation",
    "use_NaHCO3",
    # Length of stay (not a temporal measurement)
    "preiculos",
    # GCS unable flag (not a measurement)
    "gcs_unable",
    # Data leak features (directly related to AKI outcome)
    "egfr",  # Estimated GFR - calculated from creatinine, reflects kidney function
]

# Configuration
HOW = "last"  # Aggregation method for temporal measures
ID_COLUMNS = ["subject_id", "hadm_id", "stay_id"]
LABEL_COLUMN = "akd"


def load_and_prepare_patients():
    """Load patients and remove missing data."""
    patients = Patients.loadPatients()
    print(f"Loaded {len(patients)} patients")

    patients.fillMissingMeasureValue(NULLABLE_MEASURES, 0)

    # Remove measures with less than 80% of data
    measures = patients.getMeasures()
    removed_measures = []
    for measure, count in measures.items():
        if count < len(patients) * 80 / 100:
            patients.removeMeasures([measure])
            removed_measures.append(measure)
            print(f"Removed measure: {measure} (count: {count})")

    # Remove patients with more than 20% missing features
    patients.removePatientByMissingFeatures()
    print(f"After removing patients with >20% missing: {len(patients)} patients")

    # Display AKI statistics
    aki_count = sum([1 for p in patients if p.akdPositive])
    print(f"AKI positive: {aki_count} ({aki_count / len(patients):.2%})")

    return patients


def create_model():
    """Create MLP classifier."""
    return MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)


def filter_temporal_features(df, show_details=True):
    """Remove fixed demographic features, keeping only temporal measurements.

    This should be called BEFORE categorical encoding to prevent creating
    encoded columns like 'gender_F', 'race_ASIAN', etc.
    """
    columns_to_drop = [col for col in FIXED_FEATURES if col in df.columns]
    if columns_to_drop and show_details:
        print(f"Dropping {len(columns_to_drop)} fixed features: {columns_to_drop}")

    df = df.drop(columns=columns_to_drop, errors='ignore')

    remaining_features = [col for col in df.columns if col not in ID_COLUMNS + [LABEL_COLUMN]]
    if show_details:
        print(f"Using {len(remaining_features)} temporal features: {sorted(remaining_features)}")

    return df


def evaluate_model():
    """Train and evaluate MLP model using only temporal features."""
    # Load and prepare data
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

        # Get temporal data
        df_train = train_patients.getMeasuresBetween(
            pd.Timedelta(hours=-6),
            pd.Timedelta(hours=24),
            HOW,
            getUntilAkiPositive=True
        )
        df_train = df_train.drop(columns=ID_COLUMNS)

        df_test = test_patients.getMeasuresBetween(
            pd.Timedelta(hours=-6),
            pd.Timedelta(hours=24),
            HOW,
            getUntilAkiPositive=True
        )
        df_test = df_test.drop(columns=ID_COLUMNS)

        # Filter to only temporal features BEFORE encoding
        # This prevents creating encoded columns like 'gender_F', 'race_ASIAN', etc.
        df_train = filter_temporal_features(df_train, show_details=(fold == 0))
        df_test = filter_temporal_features(df_test, show_details=False)

        # Encode categorical data (should be minimal after filtering)
        df_train, df_test, _ = encodeCategoricalData(df_train, df_test)

        # Prepare X and y
        X_train = df_train.drop(columns=[LABEL_COLUMN])
        y_train = df_train[LABEL_COLUMN]

        X_test = df_test.drop(columns=[LABEL_COLUMN])
        y_test = df_test[LABEL_COLUMN]

        # Fill missing values
        X_train = X_train.fillna(0)
        X_test = X_test.fillna(0)

        print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")

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
        precision_scores.append(precision_score(y_test, y_pred))
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
    plt.title("ROC Curves - MLP with Temporal Features Only")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.savefig("result/roc_mlp_temporal_only.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Print summary statistics
    print("\n" + "="*60)
    print("RESULTS SUMMARY (Temporal Features Only)")
    print("="*60)
    print(f"AUC:         {np.mean(auc_scores):.4f} ± {np.std(auc_scores):.4f}")
    print(f"Accuracy:    {np.mean(accuracy_scores):.4f} ± {np.std(accuracy_scores):.4f}")
    print(f"Specificity: {np.mean(specificity_scores):.4f} ± {np.std(specificity_scores):.4f}")
    print(f"Precision:   {np.mean(precision_scores):.4f} ± {np.std(precision_scores):.4f}")
    print(f"Recall:      {np.mean(recall_scores):.4f} ± {np.std(recall_scores):.4f}")
    print(f"AUC-PR:      {np.mean(auc_pr_scores):.4f} ± {np.std(auc_pr_scores):.4f}")
    print("="*60)

    print("\nDetailed scores per fold:")
    print(f"AUC:         {auc_scores}")
    print(f"Accuracy:    {accuracy_scores}")
    print(f"Specificity: {specificity_scores}")
    print(f"Precision:   {precision_scores}")
    print(f"Recall:      {recall_scores}")
    print(f"AUC-PR:      {auc_pr_scores}")


if __name__ == "__main__":
    evaluate_model()
