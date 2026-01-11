"""
Time-Embedded RNN with Validation Split and Early Stopping.

Key improvements over TimeEmbedding.py:
1. Split training data into train (90%) / validation (10%)
2. Evaluate on validation every 5 epochs
3. Save best model based on validation AUC
4. Prevents overfitting by stopping when validation performance stops improving
"""

import pandas as pd
import numpy as np
import sys
import copy
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
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

# Import components from TimeEmbedding.py
from TimeEmbedding import (
    FIXED_FEATURES,
    LABEL_COLUMN,
    DEVICE,
    get_all_temporal_features,
    extract_temporal_data,
    IrregularTimeSeriesDataset,
    collate_fn,
    TimeEmbeddedRNNModel,
    load_and_prepare_patients,
    evaluate_model,
)


def split_patients_train_val(patients, val_ratio=0.1, seed=42):
    """
    Split patients into train and validation sets.

    Args:
        patients: Patients object
        val_ratio: Fraction for validation (default 10%)
        seed: Random seed for reproducibility
    """
    np.random.seed(seed)

    # Get indices
    n_patients = len(patients.patientList)
    indices = np.arange(n_patients)
    np.random.shuffle(indices)

    # Split
    n_val = int(n_patients * val_ratio)
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]

    # Create new Patients objects
    from utils.class_patient import Patients as PatientsClass

    train_patients = PatientsClass(patients=[])
    val_patients = PatientsClass(patients=[])

    for idx in train_indices:
        train_patients.patientList.append(patients.patientList[idx])

    for idx in val_indices:
        val_patients.patientList.append(patients.patientList[idx])

    return train_patients, val_patients


def train_model_with_validation(model, train_loader, val_loader, criterion, optimizer,
                                 num_epochs=100, eval_every=5, patience=10):
    """
    Train with validation monitoring and early stopping.

    Args:
        model: The neural network model
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss function
        optimizer: Optimizer
        num_epochs: Maximum number of epochs
        eval_every: Evaluate on validation every N epochs
        patience: Stop if no improvement for N evaluations

    Returns:
        best_model: Model with best validation AUC
        history: Training history
    """
    best_val_auc = 0.0
    best_model_state = None
    epochs_without_improvement = 0

    history = {
        'train_loss': [],
        'val_auc': [],
        'val_accuracy': [],
    }

    print(f"Training with validation (evaluate every {eval_every} epochs, patience={patience})")

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_loss = 0
        num_batches = 0

        for batch_data, labels in train_loader:
            labels = labels.to(DEVICE)

            # Forward pass
            predictions = model(batch_data)
            loss = criterion(predictions, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_train_loss = total_loss / num_batches
        history['train_loss'].append(avg_train_loss)

        # Validation phase (every eval_every epochs)
        if (epoch + 1) % eval_every == 0:
            model.eval()
            val_predictions = []
            val_labels = []

            with torch.no_grad():
                for batch_data, labels in val_loader:
                    predictions = model(batch_data)
                    val_predictions.extend(predictions.cpu().numpy())
                    val_labels.extend(labels.cpu().numpy())

            val_predictions = np.array(val_predictions)
            val_labels = np.array(val_labels)
            val_binary_preds = (val_predictions > 0.5).astype(int)

            val_auc = roc_auc_score(val_labels, val_predictions)
            val_accuracy = accuracy_score(val_labels, val_binary_preds)

            history['val_auc'].append(val_auc)
            history['val_accuracy'].append(val_accuracy)

            print(f"  Epoch [{epoch+1}/{num_epochs}] - "
                  f"Train Loss: {avg_train_loss:.4f}, "
                  f"Val AUC: {val_auc:.4f}, "
                  f"Val Acc: {val_accuracy:.4f}")

            # Check for improvement
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_model_state = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
                print(f"    ✓ New best validation AUC: {best_val_auc:.4f}")
            else:
                epochs_without_improvement += 1
                print(f"    No improvement ({epochs_without_improvement}/{patience})")

                # Early stopping
                if epochs_without_improvement >= patience:
                    print(f"\n  Early stopping at epoch {epoch+1}")
                    break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\nLoaded best model with validation AUC: {best_val_auc:.4f}")

    return model, history


def main():
    """Main training with validation."""
    print("="*80)
    print("TIME-EMBEDDED RNN WITH VALIDATION SPLIT")
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

    # ROC curve plot
    plt.figure(figsize=(10, 8))

    # Get all temporal features
    print("\nExtracting temporal features...")
    all_temporal_features = get_all_temporal_features(patients)
    input_dim = len(all_temporal_features)
    print(f"Number of temporal features: {input_dim}")

    # Cross-validation
    for fold, (train_patients, test_patients) in enumerate(trainTestPatients(patients)):
        print(f"\n{'='*80}")
        print(f"FOLD {fold}")
        print(f"{'='*80}")

        # Split training data into train/val (90%/10%)
        train_patients_split, val_patients_split = split_patients_train_val(
            train_patients, val_ratio=0.1, seed=42+fold
        )

        print(f"Original train size: {len(train_patients.patientList)}")
        print(f"Split - Train: {len(train_patients_split.patientList)}, "
              f"Val: {len(val_patients_split.patientList)}, "
              f"Test: {len(test_patients.patientList)}")

        # Create datasets
        train_dataset = IrregularTimeSeriesDataset(train_patients_split, all_temporal_features)

        if len(train_dataset) == 0:
            print(f"Skipping fold {fold}: Empty train dataset")
            continue

        norm_stats = train_dataset.get_normalization_stats()
        val_dataset = IrregularTimeSeriesDataset(val_patients_split, all_temporal_features, norm_stats)
        test_dataset = IrregularTimeSeriesDataset(test_patients, all_temporal_features, norm_stats)

        print(f"Dataset sizes - Train: {len(train_dataset)}, "
              f"Val: {len(val_dataset)}, Test: {len(test_dataset)}")

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

        # Create model
        model = TimeEmbeddedRNNModel(
            input_dim=input_dim,
            hidden_dim=128,
            time_dim=32,
            output_dim=1
        ).to(DEVICE)

        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Train with validation
        print("\nTraining with validation monitoring...")
        model, history = train_model_with_validation(
            model, train_loader, val_loader, criterion, optimizer,
            num_epochs=100,
            eval_every=5,
            patience=10
        )

        # Evaluate on test set
        print("\nEvaluating on test set...")
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

        print(f"\nFold {fold} FINAL - AUC: {auc_scores[-1]:.3f}, Accuracy: {accuracy_scores[-1]:.3f}")

        # Plot validation history for this fold
        if len(history['val_auc']) > 0:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

            # Loss
            ax1.plot(history['train_loss'], label='Train Loss')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.set_title(f'Fold {fold} - Training Loss')
            ax1.legend()
            ax1.grid(alpha=0.3)

            # Validation AUC
            eval_epochs = np.arange(5, len(history['val_auc'])*5+1, 5)
            ax2.plot(eval_epochs, history['val_auc'], marker='o', label='Val AUC')
            ax2.axhline(y=max(history['val_auc']), color='r', linestyle='--',
                       label=f'Best: {max(history["val_auc"]):.3f}')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('AUC')
            ax2.set_title(f'Fold {fold} - Validation AUC')
            ax2.legend()
            ax2.grid(alpha=0.3)

            plt.tight_layout()
            plt.savefig(f"result/fold_{fold}_training_history.png", dpi=150)
            plt.close()

    # Plot ROC curves
    plt.plot([0, 1], [0, 1], linestyle="--", color="navy", lw=2, label="Random")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves - Time-Embedded RNN (with Validation)")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.savefig("result/roc_time_embedded_rnn_val.png", dpi=300, bbox_inches="tight")
    print("\nSaved ROC plot to: result/roc_time_embedded_rnn_val.png")
    plt.show()

    # Print summary
    print("\n" + "="*80)
    print("RESULTS SUMMARY (Time-Embedded RNN with Validation)")
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
    print("  MLP (last only):            AUC 0.770 ± 0.021")
    print("  ODE-RNN:                    AUC 0.746")
    print("  MLP (time-aware):           AUC 0.771 ± 0.012")
    print(f"  Time-Embedded RNN (val):    AUC {np.mean(auc_scores):.3f} ± {np.std(auc_scores):.3f}")


if __name__ == "__main__":
    main()
