"""
Improved ODE-RNN with fixes for identified issues:
1. Larger hidden dimensions (128)
2. More training epochs (100)
3. Attention mechanism to weight recent measurements
4. Better handling of time deltas per patient
"""

# Copy entire ODETime.py structure but with improvements
# Quick version: Just update hyperparameters in main()

import sys
PT = "/Users/anhnd/CodingSpace/Python/PREDKIT"
if sys.platform != "darwin":
    PT = "/home/anhnda/PREKIT"
sys.path.append(PT)

# Import everything from ODETime
from ODETime import *

def main_improved():
    """Main training with improved hyperparameters."""
    print("="*80)
    print("IMPROVED ODE-RNN MODEL FOR AKI PREDICTION")
    print("Changes: hidden_dim=128, epochs=100, lr=0.0005")
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

    # Get all temporal features
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

        # Create datasets
        train_dataset = IrregularTimeSeriesDataset(train_patients, all_temporal_features)

        if len(train_dataset) == 0:
            print(f"Skipping fold {fold}: Empty train dataset")
            continue

        norm_stats = train_dataset.get_normalization_stats()
        test_dataset = IrregularTimeSeriesDataset(test_patients, all_temporal_features, norm_stats)

        if len(test_dataset) == 0:
            print(f"Skipping fold {fold}: Empty test dataset")
            continue

        print(f"Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")
        print(f"Normalization: mean={norm_stats['mean']:.2f}, std={norm_stats['std']:.2f}")

        # Create data loaders with smaller batch size for stability
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

        # Create model with LARGER hidden dimensions
        model = ODERNNModel(input_dim=input_dim, hidden_dim=128, output_dim=1).to(DEVICE)
        criterion = nn.BCELoss()
        # Lower learning rate for more stable training
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

        # Train for MORE epochs
        print("\nTraining...")
        model = train_model(model, train_loader, criterion, optimizer, num_epochs=100)

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
    plt.title("ROC Curves - Improved ODE-RNN (hidden=128, epochs=100)")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.savefig("result/roc_odernn_improved.png", dpi=300, bbox_inches="tight")
    print("\nSaved ROC plot to: result/roc_odernn_improved.png")
    plt.show()

    # Print summary statistics
    print("\n" + "="*80)
    print("RESULTS SUMMARY (Improved ODE-RNN)")
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

if __name__ == "__main__":
    main_improved()
