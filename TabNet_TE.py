"""
True End-to-End Differentiable Hybrid: Time-Embedded RNN + TabNet

- No more stages.
- One single training loop.
- Gradient flows from the TabNet head all the way back to the RNN inputs.
"""

import pandas as pd
import numpy as np
import sys
import copy
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# We use the internal TabNet module, not the Classifier wrapper
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
from utils.prepare_data import trainTestPatients

# Import existing helpers
from TimeEmbeddingVal import (
    get_all_temporal_features,
    IrregularTimeSeriesDataset,
    collate_fn,
    load_and_prepare_patients,
    split_patients_train_val,
)
from TimeEmbedding import DEVICE, TimeEmbeddedRNNCell

# ==============================================================================
# 1. The Unified Model (RNN + TabNet)
# ==============================================================================

class EndToEndRNNTabNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, static_dim, time_dim=32):
        super(EndToEndRNNTabNet, self).__init__()
        
        # Part 1: Temporal Encoder (RNN)
        self.rnn = TimeEmbeddedRNNCell(input_dim, hidden_dim, time_dim)
        
        # Part 2: Differentiable "Tree-like" Head (TabNet)
        # Input size is RNN hidden state + Static Features
        combined_dim = hidden_dim + static_dim
        
        self.tabnet = TabNet(
            input_dim=combined_dim,
            output_dim=1,
            n_d=16,            # Dimension of the decision prediction layer
            n_a=16,            # Dimension of the attention embedding
            n_steps=3,         # Number of decision steps (like tree depth)
            gamma=1.3,         # Coefficient for feature reusage
            n_independent=2,   # Layers independent across steps
            n_shared=2,        # Layers shared across steps
            epsilon=1e-15,
            momentum=0.02,
            mask_type='sparsemax' # Sparse selection (like a tree split)
        )

    def forward(self, batch_data):
        # 1. Run RNN on temporal data
        times = batch_data['times'].to(DEVICE)
        values = batch_data['values'].to(DEVICE)
        masks = batch_data['masks'].to(DEVICE)
        lengths = batch_data['lengths'].to(DEVICE)
        
        # h shape: [batch_size, hidden_dim]
        h = self.rnn(times, values, masks, lengths)
        
        # 2. Get Static Data
        # s shape: [batch_size, static_dim]
        s = batch_data['static'].to(DEVICE)
        
        # 3. Concatenate (End-to-End connection)
        # x shape: [batch_size, hidden_dim + static_dim]
        x = torch.cat([h, s], dim=1)
        
        # 4. Run TabNet Head
        # Returns logits and the sparsity loss (M_loss)
        logits, M_loss = self.tabnet(x)
        
        return logits, M_loss

# ==============================================================================
# 2. End-to-End Training Loop
# ==============================================================================

def train_end_to_end(model, train_loader, val_loader, criterion, optimizer, num_epochs=50, patience=10):
    best_val_auc = 0.0
    best_model_state = None
    epochs_no_improve = 0
    
    print(f"Starting End-to-End training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch_data, labels in train_loader:
            labels = labels.to(DEVICE).float()
            
            # Forward Pass (Runs RNN + TabNet together)
            logits, M_loss = model(batch_data)
            probs = torch.sigmoid(logits).squeeze(-1)
            
            # Loss = Binary Cross Entropy + TabNet Sparsity Penalty
            # The penalty forces the model to select features cleanly (like a tree)
            cls_loss = criterion(probs, labels)
            loss = cls_loss + (1e-3 * M_loss) 
            
            # Backward Pass (Updates BOTH RNN and TabNet weights)
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient Clipping (Important for RNN stability)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        
        # --- Validation ---
        if (epoch + 1) % 5 == 0:
            model.eval()
            val_probs = []
            val_true = []
            
            with torch.no_grad():
                for batch_data, labels in val_loader:
                    logits, _ = model(batch_data)
                    probs = torch.sigmoid(logits).squeeze(-1)
                    val_probs.extend(probs.cpu().numpy())
                    val_true.extend(labels.cpu().numpy())
            
            val_auc = roc_auc_score(val_true, val_probs)
            print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss:.4f} | Val AUC: {val_auc:.4f}")
            
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_model_state = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
                print(f"  ✓ New Best AUC: {best_val_auc:.4f}")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
                    
    if best_model_state:
        model.load_state_dict(best_model_state)
    return model

# ==============================================================================
# 3. Main Execution
# ==============================================================================

def main():
    print("="*80)
    print("END-TO-END DIFFERENTIABLE HYBRID (RNN + TABNET)")
    print("="*80)
    
    patients = load_and_prepare_patients()
    
    # Feature Setup
    temporal_feats = get_all_temporal_features(patients)
    # Define static features manually or extract from helper
    static_feats = [
        "age", "gender", "race", "chronic_pulmonary_disease", "ckd_stage",
        "congestive_heart_failure", "dka_type", "history_aci", "history_ami", 
        "hypertension", "liver_disease", "macroangiopathy", "malignant_cancer", 
        "microangiopathy", "uti", "oasis", "saps2", "sofa", 
        "mechanical_ventilation", "use_NaHCO3", "preiculos", "gcs_unable", "egfr"
    ]
    
    input_dim = len(temporal_feats)
    static_dim = len(static_feats)
    print(f"Features: {input_dim} temporal, {static_dim} static")

    auc_scores = []

    for fold, (train_full, test_p) in enumerate(trainTestPatients(patients)):
        print(f"\n--- Fold {fold} ---")
        train_p, val_p = split_patients_train_val(train_full, val_ratio=0.1, seed=42+fold)
        
        # Ensure your IrregularTimeSeriesDataset handles the 'static' logic we fixed!
        train_ds = IrregularTimeSeriesDataset(train_p, temporal_feats)
        norm_stats = train_ds.get_normalization_stats()
        val_ds = IrregularTimeSeriesDataset(val_p, temporal_feats, norm_stats)
        test_ds = IrregularTimeSeriesDataset(test_p, temporal_feats, norm_stats)
        
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=collate_fn)
        test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=collate_fn)
        
        # Initialize Unified Model
        model = EndToEndRNNTabNet(
            input_dim=input_dim,
            hidden_dim=128,    # RNN output size
            static_dim=static_dim
        ).to(DEVICE)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        criterion = nn.BCELoss()
        
        # Train
        model = train_end_to_end(model, train_loader, val_loader, criterion, optimizer)
        
        # Evaluate
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for batch_data, labels in test_loader:
                logits, _ = model(batch_data)
                probs = torch.sigmoid(logits).squeeze(-1)
                y_true.extend(labels.numpy())
                y_pred.extend(probs.cpu().numpy())
        
        score = roc_auc_score(y_true, y_pred)
        auc_scores.append(score)
        print(f"Fold {fold} Test AUC: {score:.4f}")

    print(f"\nAverage AUC: {np.mean(auc_scores):.4f} ± {np.std(auc_scores):.4f}")

if __name__ == "__main__":
    main()