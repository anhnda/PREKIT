"""
Hybrid: Time-Embedded RNN + XGBoost

Stage 1: Time-Embedded RNN learns temporal embeddings from irregular time series
Stage 2: XGBoost classifier uses those embeddings as features
"""

import pandas as pd
import numpy as np
import sys
import copy
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from xgboost import XGBClassifier
from pytorch_tabnet.tab_network import TabNet
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
    """Train Time-Embedded RNN for embedding extraction with TabNet head."""

    best_val_auc = 0.0
    best_model_state = None
    epochs_without_improvement = 0

    # TabNet classifier head for training
    input_dim = model.rnn_cell.hidden_dim
    temp_classifier = TabNet(
        input_dim=input_dim,
        output_dim=1,
        n_d=32,  # Width of the decision prediction layer
        n_a=32,  # Width of the attention embedding
        n_steps=3,  # Number of sequential decision steps
        gamma=1.3,  # Coefficient for feature reusage
        n_independent=2,  # Number of independent GLU layers
        n_shared=2,  # Number of shared GLU layers
        epsilon=1e-15,
        momentum=0.02,
        mask_type='sparsemax'
    ).to(DEVICE)

    # Create separate optimizer for TabNet
    tabnet_optimizer = torch.optim.Adam(temp_classifier.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        model.train()
        temp_classifier.train()
        total_loss = 0
        num_batches = 0

        for batch_data, labels in train_loader:
            labels = labels.to(DEVICE).float()

            # Get embeddings
            embeddings = model(batch_data)

            # Forward through TabNet
            predictions, M_loss = temp_classifier(embeddings)
            predictions = torch.sigmoid(predictions.squeeze(-1))

            # Loss = classification loss + sparsity loss
            classification_loss = criterion(predictions, labels)
            loss = classification_loss - 1e-3 * M_loss  # Negative because we want to maximize sparsity

            optimizer.zero_grad()
            tabnet_optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tabnet_optimizer.step()

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
                    predictions, _ = temp_classifier(embeddings)
                    predictions = torch.sigmoid(predictions.squeeze(-1))
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


def extract_embeddings(model, data_loader):
    """Extract embeddings from trained model."""
    model.eval()
    embeddings_list = []
    labels_list = []

    with torch.no_grad():
        for batch_data, labels in data_loader:
            embeddings = model(batch_data)
            embeddings_list.append(embeddings.cpu().numpy())
            labels_list.extend(labels.cpu().numpy())

    embeddings = np.vstack(embeddings_list)
    labels = np.array(labels_list)

    return embeddings, labels


def main():
    print("="*80)
    print("COMPARISON: XGBOOST ON LAST FEATURES VS TIME-EMBEDDED RNN + XGBOOST")
    print("="*80)

    patients = load_and_prepare_patients()

    # Metrics for TimeEmbedding + XGBoost
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    auc_scores = []
    specificity_scores = []
    auc_pr_scores = []

    # Metrics for XGBoost on Last Features (baseline)
    accuracy_scores_last = []
    precision_scores_last = []
    recall_scores_last = []
    auc_scores_last = []
    specificity_scores_last = []
    auc_pr_scores_last = []

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

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
        train_dataset = IrregularTimeSeriesDataset(train_patients, all_temporal_features)
        if len(train_dataset) == 0:
            continue

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
            num_epochs=100, eval_every=5, patience=4
        )

        # Extract embeddings
        print("\n[STAGE 2] Extracting embeddings...")
        X_train, y_train = extract_embeddings(embedding_model, train_loader)
        X_val, y_val = extract_embeddings(embedding_model, val_loader)
        X_test, y_test = extract_embeddings(embedding_model, test_loader)

        print(f"Embedding shapes - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

        # STAGE 2: Train XGBoost on embeddings
        print("\n[STAGE 3] Training XGBoost on embeddings...")
        xgb_model = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            eval_metric='auc',
            use_label_encoder=False,
            verbosity=0
        )

        xgb_model.fit(X_train, y_train)

        # Evaluate TimeEmbedding + XGBoost
        y_pred = xgb_model.predict(X_test)
        y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]

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

        print(f"\n[TimeEmbedding + XGBoost] Fold {fold} - AUC: {auc_scores[-1]:.3f}, Accuracy: {accuracy_scores[-1]:.3f}")

        # ========================================================================
        # BASELINE: XGBoost on Last Features (same data split!)
        # ========================================================================
        print(f"\n[Baseline] Training XGBoost on Last Features...")

        # Get last features from same patients
        FIXED_FEATURES = [
            "age", "gender", "race", "chronic_pulmonary_disease", "ckd_stage",
            "congestive_heart_failure", "dka_type", "history_aci", "history_ami",
            "hypertension", "liver_disease", "macroangiopathy", "malignant_cancer",
            "microangiopathy", "uti", "oasis", "saps2", "sofa",
            "mechanical_ventilation", "use_NaHCO3", "preiculos", "gcs_unable", "egfr"
        ]
        ID_COLUMNS = ["subject_id", "hadm_id", "stay_id"]
        LABEL_COLUMN = "akd"

        df_train_last = train_patients.getMeasuresBetween(
            pd.Timedelta(hours=-6), pd.Timedelta(hours=24), "last", getUntilAkiPositive=True
        ).drop(columns=ID_COLUMNS)

        df_val_last = val_patients.getMeasuresBetween(
            pd.Timedelta(hours=-6), pd.Timedelta(hours=24), "last", getUntilAkiPositive=True
        ).drop(columns=ID_COLUMNS)

        df_test_last = test_patients.getMeasuresBetween(
            pd.Timedelta(hours=-6), pd.Timedelta(hours=24), "last", getUntilAkiPositive=True
        ).drop(columns=ID_COLUMNS)

        # Filter temporal features
        for df in [df_train_last, df_val_last, df_test_last]:
            for col in FIXED_FEATURES:
                if col in df.columns:
                    df.drop(columns=[col], inplace=True)

        # Encode
        df_train_last, df_val_last, _ = encodeCategoricalData(df_train_last, df_val_last)
        df_train_last, df_test_last, _ = encodeCategoricalData(df_train_last, df_test_last)

        # Prepare data
        X_train_last = df_train_last.drop(columns=[LABEL_COLUMN]).fillna(0)
        y_train_last = df_train_last[LABEL_COLUMN]
        X_test_last = df_test_last.drop(columns=[LABEL_COLUMN]).fillna(0)
        y_test_last = df_test_last[LABEL_COLUMN]

        # Train XGBoost
        xgb_baseline = XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0, random_state=42,
            eval_metric='auc', use_label_encoder=False, verbosity=0
        )
        xgb_baseline.fit(X_train_last, y_train_last)

        # Evaluate baseline
        y_pred_last = xgb_baseline.predict(X_test_last)
        y_pred_proba_last = xgb_baseline.predict_proba(X_test_last)[:, 1]

        tn_last, fp_last, _, _ = confusion_matrix(y_test_last, y_pred_last).ravel()
        precision_vals_last, recall_vals_last, _ = precision_recall_curve(y_test_last, y_pred_proba_last)

        accuracy_scores_last.append(accuracy_score(y_test_last, y_pred_last))
        specificity_scores_last.append(tn_last / (tn_last + fp_last))
        precision_scores_last.append(precision_score(y_test_last, y_pred_last, zero_division=0))
        recall_scores_last.append(recall_score(y_test_last, y_pred_last))
        auc_scores_last.append(roc_auc_score(y_test_last, y_pred_proba_last))
        auc_pr_scores_last.append(auc(recall_vals_last, precision_vals_last))

        fpr_last, tpr_last, _ = roc_curve(y_test_last, y_pred_proba_last)
        ax2.plot(fpr_last, tpr_last, lw=2, label=f"Fold {fold} (AUC = {auc_scores_last[-1]:.3f})")

        print(f"[Baseline Last Features] Fold {fold} - AUC: {auc_scores_last[-1]:.3f}, Accuracy: {accuracy_scores_last[-1]:.3f}")

    # Configure both subplots
    for ax in [ax1, ax2]:
        ax.plot([0, 1], [0, 1], linestyle="--", color="navy", lw=2, label="Random")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(loc="lower right")
        ax.grid(alpha=0.3)

    ax1.set_title("Time-Embedded RNN + XGBoost")
    ax2.set_title("XGBoost on Last Features (Baseline)")

    plt.tight_layout()
    plt.savefig("result/roc_comparison_timeembed_vs_baseline.png", dpi=300, bbox_inches="tight")
    print("\nSaved comparison plot to: result/roc_comparison_timeembed_vs_baseline.png")
    plt.show()

    # Print summaries
    print("\n" + "="*80)
    print("RESULTS SUMMARY - TIME-EMBEDDED RNN + XGBOOST")
    print("="*80)
    print(f"AUC:         {np.mean(auc_scores):.4f} ± {np.std(auc_scores):.4f}")
    print(f"Accuracy:    {np.mean(accuracy_scores):.4f} ± {np.std(accuracy_scores):.4f}")
    print(f"Specificity: {np.mean(specificity_scores):.4f} ± {np.std(specificity_scores):.4f}")
    print(f"Precision:   {np.mean(precision_scores):.4f} ± {np.std(precision_scores):.4f}")
    print(f"Recall:      {np.mean(recall_scores):.4f} ± {np.std(recall_scores):.4f}")
    print(f"AUC-PR:      {np.mean(auc_pr_scores):.4f} ± {np.std(auc_pr_scores):.4f}")
    print("="*80)

    print("\n" + "="*80)
    print("RESULTS SUMMARY - XGBOOST ON LAST FEATURES (BASELINE, SAME DATA SPLIT)")
    print("="*80)
    print(f"AUC:         {np.mean(auc_scores_last):.4f} ± {np.std(auc_scores_last):.4f}")
    print(f"Accuracy:    {np.mean(accuracy_scores_last):.4f} ± {np.std(accuracy_scores_last):.4f}")
    print(f"Specificity: {np.mean(specificity_scores_last):.4f} ± {np.std(specificity_scores_last):.4f}")
    print(f"Precision:   {np.mean(precision_scores_last):.4f} ± {np.std(precision_scores_last):.4f}")
    print(f"Recall:      {np.mean(recall_scores_last):.4f} ± {np.std(recall_scores_last):.4f}")
    print(f"AUC-PR:      {np.mean(auc_pr_scores_last):.4f} ± {np.std(auc_pr_scores_last):.4f}")
    print("="*80)

    print("\n" + "="*80)
    print("FINAL COMPARISON (ON SAME DATA SPLITS)")
    print("="*80)
    print(f"TimeEmbed RNN + XGBoost:    AUC {np.mean(auc_scores):.4f} ± {np.std(auc_scores):.4f}")
    print(f"XGBoost (last features):    AUC {np.mean(auc_scores_last):.4f} ± {np.std(auc_scores_last):.4f}")
    print(f"\nDifference: {np.mean(auc_scores) - np.mean(auc_scores_last):.4f}")

    if np.mean(auc_scores) > np.mean(auc_scores_last):
        print("✓ TimeEmbed RNN + XGBoost performs better")
    else:
        print("✗ Simple XGBoost on last features performs better")


if __name__ == "__main__":
    main()
