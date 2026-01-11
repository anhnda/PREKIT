"""
Hybrid: Time-Embedded RNN + TabNet

Stage 1: Time-Embedded RNN learns temporal embeddings from irregular time series
Stage 2: TabNet classifier uses those embeddings as features (Differentiable Tree Mimic)
"""

import pandas as pd
import numpy as np
import sys
import copy
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Replaces XGBoost
from pytorch_tabnet.tab_model import TabNetClassifier

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
from utils.prepare_data import trainTestPatients, encodeCategoricalData

# Import from TimeEmbeddingVal
from TimeEmbeddingVal import (
    get_all_temporal_features,
    IrregularTimeSeriesDataset,
    collate_fn,
    load_and_prepare_patients,
    split_patients_train_val,
)

# Import RNN components from TimeEmbedding
from TimeEmbedding import (
    DEVICE,
    TimeEmbeddedRNNCell,
)


class TimeEmbeddingExtractor(nn.Module):
    """Extract temporal embeddings using Time-Embedded RNN."""

    def __init__(self, input_dim, hidden_dim, time_dim=32):
        super(TimeEmbeddingExtractor, self).__init__()
        self.rnn_cell = TimeEmbeddedRNNCell(input_dim, hidden_dim, time_dim)

    def forward(self, batch_data):
        times = batch_data['times'].to(DEVICE)
        values = batch_data['values'].to(DEVICE)
        masks = batch_data['masks'].to(DEVICE)
        lengths = batch_data['lengths'].to(DEVICE)

        hidden_states = self.rnn_cell(times, values, masks, lengths)
        return hidden_states


def train_embedding_extractor(model, train_loader, val_loader, criterion, optimizer,
                               num_epochs=100, eval_every=5, patience=10):
    """Train Time-Embedded RNN for embedding extraction."""

    best_val_auc = 0.0
    best_model_state = None
    epochs_without_improvement = 0

    # Temporary linear classifier for Stage 1 training
    temp_classifier = nn.Sequential(
        nn.Linear(model.rnn_cell.hidden_dim, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
        nn.Sigmoid()
    ).to(DEVICE)

    for epoch in range(num_epochs):
        model.train()
        temp_classifier.train()
        total_loss = 0
        num_batches = 0

        for batch_data, labels in train_loader:
            labels = labels.to(DEVICE).float()

            # Get embeddings
            embeddings = model(batch_data)
            predictions = temp_classifier(embeddings).squeeze(-1)
            loss = criterion(predictions, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches

        # Validation
        if (epoch + 1) % eval_every == 0:
            model.eval()
            temp_classifier.eval()
            val_predictions = []
            val_labels = []

            with torch.no_grad():
                for batch_data, labels in val_loader:
                    embeddings = model(batch_data)
                    predictions = temp_classifier(embeddings).squeeze(-1)
                    val_predictions.extend(predictions.cpu().numpy())
                    val_labels.extend(labels.cpu().numpy())

            val_predictions = np.array(val_predictions)
            val_labels = np.array(val_labels)
            val_auc = roc_auc_score(val_labels, val_predictions)

            print(f"  Epoch [{epoch+1}/{num_epochs}] - Loss: {avg_loss:.4f}, Val AUC: {val_auc:.4f}")

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_model_state = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
                print(f"    ✓ New best: {best_val_auc:.4f}")
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model


def extract_embeddings_and_static(model, data_loader, patients_list):
    """
    Extract embeddings from RNN AND concatenate with static features.
    Use this to feed TabNet both temporal summary and static context.
    """
    model.eval()
    embeddings_list = []
    labels_list = []
    
    # We need to access static features. Since the dataset is providing batches,
    # we need to ensure we can get static features aligned with the batch.
    # Note: This assumes the IrregularTimeSeriesDataset has been updated to return static data
    # OR we extract it manually here if dataset doesn't support it yet.
    
    # Check if dataset returns static data (tuple length 3)
    sample_batch = next(iter(data_loader))
    has_static = len(sample_batch) >= 3
    
    with torch.no_grad():
        for batch in data_loader:
            if has_static:
                batch_data, labels, static_data = batch
            else:
                batch_data, labels = batch
                static_data = None # Handle fallback if needed
            
            # 1. Temporal Embeddings
            h = model(batch_data).cpu().numpy()
            
            # 2. Static Features (if available in batch)
            if static_data is not None:
                s = static_data.cpu().numpy()
                combined = np.hstack([h, s])
            else:
                combined = h
                
            embeddings_list.append(combined)
            labels_list.extend(labels.cpu().numpy())

    embeddings = np.vstack(embeddings_list)
    labels = np.array(labels_list)

    return embeddings, labels


def main():
    print("="*80)
    print("HYBRID: TIME-EMBEDDED RNN + TABNET")
    print("="*80)

    patients = load_and_prepare_patients()

    # Metrics
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    auc_scores = []
    specificity_scores = []
    auc_pr_scores = []

    fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))

    all_temporal_features = get_all_temporal_features(patients)
    input_dim = len(all_temporal_features)
    print(f"Temporal features: {input_dim}")

    for fold, (train_patients_full, test_patients) in enumerate(trainTestPatients(patients)):
        print(f"\n{'='*80}")
        print(f"FOLD {fold}")
        print(f"{'='*80}")

        # Split train into train/val
        train_patients, val_patients = split_patients_train_val(
            train_patients_full, val_ratio=0.1, seed=42+fold
        )

        print(f"Train: {len(train_patients.patientList)}, "
              f"Val: {len(val_patients.patientList)}, "
              f"Test: {len(test_patients.patientList)}")

        # Create datasets
        # NOTE: Ensure IrregularTimeSeriesDataset returns static features!
        train_dataset = IrregularTimeSeriesDataset(train_patients, all_temporal_features)
        if len(train_dataset) == 0: continue

        norm_stats = train_dataset.get_normalization_stats()
        val_dataset = IrregularTimeSeriesDataset(val_patients, all_temporal_features, norm_stats)
        test_dataset = IrregularTimeSeriesDataset(test_patients, all_temporal_features, norm_stats)

        # Data loaders
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

        # STAGE 1: Train Time-Embedded RNN
        print("\n[STAGE 1] Training Time-Embedded RNN...")
        embedding_model = TimeEmbeddingExtractor(
            input_dim=input_dim,
            hidden_dim=128,
            time_dim=32
        ).to(DEVICE)

        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(embedding_model.parameters(), lr=0.0005)

        embedding_model = train_embedding_extractor(
            embedding_model, train_loader, val_loader, criterion, optimizer,
            num_epochs=50, eval_every=5, patience=4
        )

        # Extract embeddings (Hybrid Features: Temporal + Static)
        print("\n[STAGE 2] Extracting features for TabNet...")
        X_train, y_train = extract_embeddings_and_static(embedding_model, train_loader, train_patients.patientList)
        X_val, y_val = extract_embeddings_and_static(embedding_model, val_loader, val_patients.patientList)
        X_test, y_test = extract_embeddings_and_static(embedding_model, test_loader, test_patients.patientList)

        print(f"Feature shapes - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

        # STAGE 2: Train TabNet on embeddings
        print("\n[STAGE 3] Training TabNet Classifier...")
        
        # TabNet Configuration
        clf = TabNetClassifier(
            n_d=64, n_a=64,            # Width of decision/attention layers
            n_steps=5,                 # Number of steps in the architecture
            gamma=1.5,                 # Coefficient for feature re-usage
            n_independent=2,           # Independent GLU layers
            n_shared=2,                # Shared GLU layers
            lambda_sparse=1e-4,        # Sparsity regularization
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=2e-2),
            scheduler_params={"step_size":10, "gamma":0.9},
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            mask_type='entmax',        # 'sparsemax' or 'entmax'
            device_name=DEVICE.type,
            verbose=1
        )

        clf.fit(
            X_train=X_train, y_train=y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            eval_name=['train', 'valid'],
            eval_metric=['auc'],
            max_epochs=100,
            patience=15,
            batch_size=256,            # TabNet likes larger batches
            virtual_batch_size=128,
            num_workers=0,
            drop_last=False
        )

        # Evaluate TabNet
        # TabNet predict_proba returns [prob_class_0, prob_class_1]
        y_pred_proba = clf.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba > 0.5).astype(int)

        tn, fp, _, _ = confusion_matrix(y_test, y_pred).ravel()
        precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_pred_proba)

        accuracy_scores.append(accuracy_score(y_test, y_pred))
        specificity_scores.append(tn / (tn + fp))
        precision_scores.append(precision_score(y_test, y_pred, zero_division=0))
        recall_scores.append(recall_score(y_test, y_pred))
        auc_scores.append(roc_auc_score(y_test, y_pred_proba))
        auc_pr_scores.append(auc(recall_vals, precision_vals))

        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        ax1.plot(fpr, tpr, lw=2, label=f"Fold {fold} (AUC = {auc_scores[-1]:.3f})")

        print(f"\n[Hybrid TabNet] Fold {fold} - AUC: {auc_scores[-1]:.3f}, Accuracy: {accuracy_scores[-1]:.3f}")

    # Plot
    ax1.plot([0, 1], [0, 1], linestyle="--", color="navy", lw=2, label="Random")
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title("Time-Embedded RNN + TabNet")
    ax1.legend(loc="lower right")
    ax1.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("result/roc_hybrid_tabnet.png", dpi=300, bbox_inches="tight")
    print("\nSaved plot to: result/roc_hybrid_tabnet.png")
    plt.show()

    # Print summaries
    print("\n" + "="*80)
    print("RESULTS SUMMARY - TIME-EMBEDDED RNN + TABNET")
    print("="*80)
    print(f"AUC:         {np.mean(auc_scores):.4f} ± {np.std(auc_scores):.4f}")
    print(f"Accuracy:    {np.mean(accuracy_scores):.4f} ± {np.std(accuracy_scores):.4f}")
    print(f"Specificity: {np.mean(specificity_scores):.4f} ± {np.std(specificity_scores):.4f}")
    print(f"Precision:   {np.mean(precision_scores):.4f} ± {np.std(precision_scores):.4f}")
    print(f"Recall:      {np.mean(recall_scores):.4f} ± {np.std(recall_scores):.4f}")
    print(f"AUC-PR:      {np.mean(auc_pr_scores):.4f} ± {np.std(auc_pr_scores):.4f}")
    print("="*80)

if __name__ == "__main__":
    main()