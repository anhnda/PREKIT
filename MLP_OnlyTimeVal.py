"""
MLP with Validation Split and Early Stopping for AKI Prediction.

Key improvements over MLP_OnlyTime.py:
1. Split training data into train (90%) / validation (10%)
2. Use warm_start to train incrementally
3. Evaluate on validation periodically
4. Save best model based on validation AUC
5. Prevents overfitting with early stopping
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
import copy

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

# Configuration
HOW = "last"
ID_COLUMNS = ["subject_id", "hadm_id", "stay_id"]
LABEL_COLUMN = "akd"


def split_patients_train_val(patients, val_ratio=0.1, seed=42):
    """Split patients into train and validation sets."""
    np.random.seed(seed)

    n_patients = len(patients.patientList)
    indices = np.arange(n_patients)
    np.random.shuffle(indices)

    n_val = int(n_patients * val_ratio)
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]

    from utils.class_patient import Patients as PatientsClass

    train_patients = PatientsClass(patients=[])
    val_patients = PatientsClass(patients=[])

    for idx in train_indices:
        train_patients.patientList.append(patients.patientList[idx])

    for idx in val_indices:
        val_patients.patientList.append(patients.patientList[idx])

    return train_patients, val_patients


def load_and_prepare_patients():
    """Load patients and remove missing data."""
    patients = Patients.loadPatients()
    print(f"Loaded {len(patients)} patients")

    patients.fillMissingMeasureValue(NULLABLE_MEASURES, 0)

    measures = patients.getMeasures()
    for measure, count in measures.items():
        if count < len(patients) * 80 / 100:
            patients.removeMeasures([measure])

    patients.removePatientByMissingFeatures()
    print(f"After cleanup: {len(patients)} patients")

    aki_count = sum([1 for p in patients if p.akdPositive])
    print(f"AKI positive: {aki_count} ({aki_count / len(patients):.2%})")

    return patients


def filter_temporal_features(df, show_details=True):
    """Remove fixed demographic features, keeping only temporal measurements."""
    columns_to_drop = [col for col in FIXED_FEATURES if col in df.columns]
    if columns_to_drop and show_details:
        print(f"Dropping {len(columns_to_drop)} fixed features: {columns_to_drop}")

    df = df.drop(columns=columns_to_drop, errors='ignore')

    remaining_features = [col for col in df.columns if col not in ID_COLUMNS + [LABEL_COLUMN]]
    if show_details:
        print(f"Using {len(remaining_features)} temporal features: {sorted(remaining_features)}")

    return df


def train_mlp_with_validation(X_train, y_train, X_val, y_val,
                               max_iter=500, eval_every=50, patience=5):
    """
    Train MLP with validation monitoring.

    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        max_iter: Maximum iterations
        eval_every: Evaluate every N iterations
        patience: Stop if no improvement for N evaluations

    Returns:
        best_model: Model with best validation AUC
        history: Training history
    """
    model = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        max_iter=eval_every,  # Train for eval_every iterations at a time
        warm_start=True,       # Continue training from where we left off
        random_state=42
    )

    best_val_auc = 0.0
    best_model = None
    epochs_without_improvement = 0
    current_iter = 0

    history = {
        'iterations': [],
        'val_auc': [],
        'val_accuracy': [],
    }

    print(f"Training with validation (evaluate every {eval_every} iterations, patience={patience})")

    while current_iter < max_iter:
        # Train for eval_every iterations
        model.fit(X_train, y_train)
        current_iter += eval_every

        # Evaluate on validation
        val_pred_proba = model.predict_proba(X_val)[:, 1]
        val_pred = model.predict(X_val)

        val_auc = roc_auc_score(y_val, val_pred_proba)
        val_accuracy = accuracy_score(y_val, val_pred)

        history['iterations'].append(current_iter)
        history['val_auc'].append(val_auc)
        history['val_accuracy'].append(val_accuracy)

        print(f"  Iteration {current_iter}/{max_iter} - "
              f"Val AUC: {val_auc:.4f}, Val Acc: {val_accuracy:.4f}")

        # Check for improvement
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model = copy.deepcopy(model)
            epochs_without_improvement = 0
            print(f"    ✓ New best validation AUC: {best_val_auc:.4f}")
        else:
            epochs_without_improvement += 1
            print(f"    No improvement ({epochs_without_improvement}/{patience})")

            # Early stopping
            if epochs_without_improvement >= patience:
                print(f"\n  Early stopping at iteration {current_iter}")
                break

    if best_model is None:
        best_model = model

    print(f"\nBest validation AUC: {best_val_auc:.4f}")

    return best_model, history


def evaluate_model():
    """Train and evaluate MLP model with validation."""
    # Load data
    patients = load_and_prepare_patients()

    # Metrics storage
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    auc_scores = []
    specificity_scores = []
    auc_pr_scores = []

    # ROC curve plot
    plt.figure(figsize=(10, 8))

    # Cross-validation
    for fold, (train_patients_full, test_patients) in enumerate(trainTestPatients(patients)):
        print(f"\n{'='*80}")
        print(f"FOLD {fold}")
        print(f"{'='*80}")

        # Split train into train/val (90%/10%)
        train_patients, val_patients = split_patients_train_val(
            train_patients_full, val_ratio=0.1, seed=42+fold
        )

        print(f"Original train size: {len(train_patients_full.patientList)}")
        print(f"Split - Train: {len(train_patients.patientList)}, "
              f"Val: {len(val_patients.patientList)}, "
              f"Test: {len(test_patients.patientList)}")

        # Get temporal data for train
        df_train = train_patients.getMeasuresBetween(
            pd.Timedelta(hours=-6),
            pd.Timedelta(hours=24),
            HOW,
            getUntilAkiPositive=True
        )
        df_train = df_train.drop(columns=ID_COLUMNS)

        # Get temporal data for validation
        df_val = val_patients.getMeasuresBetween(
            pd.Timedelta(hours=-6),
            pd.Timedelta(hours=24),
            HOW,
            getUntilAkiPositive=True
        )
        df_val = df_val.drop(columns=ID_COLUMNS)

        # Get temporal data for test
        df_test = test_patients.getMeasuresBetween(
            pd.Timedelta(hours=-6),
            pd.Timedelta(hours=24),
            HOW,
            getUntilAkiPositive=True
        )
        df_test = df_test.drop(columns=ID_COLUMNS)

        # Filter temporal features
        df_train = filter_temporal_features(df_train, show_details=(fold == 0))
        df_val = filter_temporal_features(df_val, show_details=False)
        df_test = filter_temporal_features(df_test, show_details=False)

        # Encode categorical data
        df_train, df_val, _ = encodeCategoricalData(df_train, df_val)
        df_train, df_test, _ = encodeCategoricalData(df_train, df_test)

        # Ensure val and test have same columns as train
        for df in [df_val, df_test]:
            missing_cols = set(df_train.columns) - set(df.columns)
            for col in missing_cols:
                if col != LABEL_COLUMN:
                    df[col] = 0
            # Reorder
            df = df[[col for col in df_train.columns if col in df.columns]]

        # Prepare data
        X_train = df_train.drop(columns=[LABEL_COLUMN])
        y_train = df_train[LABEL_COLUMN]

        X_val = df_val.drop(columns=[LABEL_COLUMN])
        y_val = df_val[LABEL_COLUMN]

        X_test = df_test.drop(columns=[LABEL_COLUMN])
        y_test = df_test[LABEL_COLUMN]

        # Fill missing values
        X_train = X_train.fillna(0)
        X_val = X_val.fillna(0)
        X_test = X_test.fillna(0)

        print(f"\nDataset shapes:")
        print(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

        # Train with validation
        print("\nTraining with validation monitoring...")
        model, history = train_mlp_with_validation(
            X_train, y_train, X_val, y_val,
            max_iter=500,
            eval_every=50,
            patience=5
        )

        # Evaluate on test set
        print("\nEvaluating on test set...")
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

        print(f"\nFold {fold} FINAL - AUC: {auc_scores[-1]:.3f}, Accuracy: {accuracy_scores[-1]:.3f}")

        # Plot validation history
        if len(history['val_auc']) > 0:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))

            ax.plot(history['iterations'], history['val_auc'], marker='o', label='Val AUC')
            ax.axhline(y=max(history['val_auc']), color='r', linestyle='--',
                      label=f'Best: {max(history["val_auc"]):.3f}')
            ax.set_xlabel('Iterations')
            ax.set_ylabel('AUC')
            ax.set_title(f'Fold {fold} - Validation AUC')
            ax.legend()
            ax.grid(alpha=0.3)

            plt.tight_layout()
            plt.savefig(f"result/mlp_fold_{fold}_validation.png", dpi=150)
            plt.close()

    # Plot ROC curves
    plt.plot([0, 1], [0, 1], linestyle="--", color="navy", lw=2, label="Random")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves - MLP with Validation (Temporal Features Only)")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.savefig("result/roc_mlp_temporal_val.png", dpi=300, bbox_inches="tight")
    print("\nSaved ROC plot to: result/roc_mlp_temporal_val.png")
    plt.show()

    # Print summary
    print("\n" + "="*80)
    print("RESULTS SUMMARY (MLP with Validation - Temporal Features Only)")
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

    print("\n" + "="*80)
    print("COMPARISON")
    print("="*80)
    print("  MLP (last, no val):         AUC 0.770 ± 0.021")
    print(f"  MLP (last, with val):       AUC {np.mean(auc_scores):.3f} ± {np.std(auc_scores):.3f}")


if __name__ == "__main__":
    evaluate_model()
