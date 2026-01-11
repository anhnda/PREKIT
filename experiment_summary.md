# RNN Feature Learning for TabPFN - Experiment Summary

## The Core Problem
- **RNN learns valuable temporal patterns** → Helps XGBoost significantly (+1.8% AUC)
- **Same RNN features hurt TabPFN** → Worse than baseline (-1.7% AUC-PR)
- **Root cause**: Training objective/head doesn't produce features TabPFN can use

## KEY FINDING: [RNN + Static] Training is Better!

**Critical Result**:
- Training with [RNN + Static]: AUC-PR 0.7955 (gap: -0.0169) ✅
- Training with [RNN only]: AUC-PR 0.7895 (gap: -0.0229) ❌
- **Conclusion**: ALL future experiments MUST use [RNN + Static] during training

## Completed Experiments

### 1. TP_StaticLearnLast_RNN.py - Attention-Based Head
- **Training**: [RNN + Static] ✅
- **Head**: Multi-head attention + layer norm (mimics transformer architecture)
- **Result**: AUC-PR 0.7955 vs Baseline 0.8124 (-0.0169)
- **Issue**: Complex head might overfit, but better than time-only training

### 2. TP_TimeLearnThenCat.py - Time-Only Training
- **Training**: [RNN only] ❌
- **Head**: Attention-based
- **Result**: AUC-PR 0.7895 vs Baseline 0.8124 (-0.0229)
- **Conclusion**: Worse than static training - DON'T USE

### 3. TP_SimpleHead_RNN.py - Simple MLP Head
- **Training**: [RNN + Static] ✅ (CORRECT)
- **Head**: BatchNorm + MLP + heavy dropout + LR scheduler
- **Changes**: Simpler architecture, stronger regularization
- **Status**: 🔄 To test

## Proposed Future Experiments

### 3. Contrastive Learning Objective
Instead of binary classification, train RNN to:
- **Pull similar patients together** (same outcome)
- **Push different patients apart** (different outcome)
- Produces more discriminative features

### 4. Feature Quality Constraints
Add losses to ensure RNN features are:
- **Well-calibrated**: Match distribution of raw features
- **Complementary**: Low correlation with "last values"
- **Informative**: High mutual information with label

### 5. Two-Stage with Feature Selection
- Train RNN normally
- **Select only top-k most important RNN features** (k=3-5)
- Reduces dimensionality curse for TabPFN

### 6. Ensemble/Distillation Approach
- Train TabPFN on raw features (baseline)
- Train RNN to **match TabPFN's predictions** on time series
- RNN learns what TabPFN needs from temporal data

### 7. Different RNN Architectures
- **GRU instead of custom TimeEmbeddedRNN**
- **Bi-directional RNN** - see future and past
- **Temporal Attention** directly in RNN

### 8. Training with TabPFN in the loop
- Use TabPFN predictions during RNN training
- Ensure RNN features improve TabPFN performance

## Key Insights

1. **XGBoost vs TabPFN difference**:
   - XGBoost: Poor at temporal patterns → RNN helps
   - TabPFN: Already good at feature interactions → RNN must complement, not duplicate

2. **The temporal information IS valuable** - we just need the right extraction method

3. **Challenge**: TabPFN expects well-behaved, interpretable features. RNN features might be:
   - Different scale/distribution
   - Correlated with existing features
   - Too high-dimensional

## Next Steps
1. Test `TP_SimpleHead_RNN.py`
2. Implement contrastive learning version
3. Try feature selection approach (simplest to implement)
