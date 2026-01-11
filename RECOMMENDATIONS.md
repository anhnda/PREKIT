# ODE-RNN vs MLP Performance Analysis

## Current Results

| Model | AUC | Accuracy | Key Approach |
|-------|-----|----------|--------------|
| **MLP (last aggregation)** | **0.795** | **0.744** | Uses most recent measurement before AKI |
| **ODE-RNN (fixed)** | 0.746 | 0.715 | Models full temporal evolution |

## Why MLP Outperforms ODE-RNN

### 1. **Recency Bias is Crucial**
- **MLP**: Uses "last" value → most recent measurement before AKI diagnosis
- **Recent measurements are most predictive** of imminent AKI
- **ODE-RNN**: Weights all time points → dilutes signal with earlier (less predictive) measurements

### 2. **Data Sparsity**
- **87% missing data** (only 3.2/25 features per timestamp)
- ODE-RNN must handle extreme missingness with masks
- Sparse data makes temporal modeling harder

### 3. **Information Loss in Batching**
- ODE-RNN uses **average time delta across batch** → loses patient-specific temporal patterns
- Time normalization to [0,1] per patient → loses absolute timing information

### 4. **Model Complexity vs Data**
- ODE-RNN is more complex → needs more data/better hyperparameters
- Current: hidden_dim=64, epochs=50 might be insufficient
- MLP is simpler → less prone to overfitting with limited data

## Recommendations

### Option 1: **Stick with MLP** (Recommended)
```python
# Use MLP_OnlyTime.py as is
# AUC 0.795 is strong performance
# Simple, interpretable, works well
```

**Pros:**
- ✓ Best current performance
- ✓ Simple and interpretable
- ✓ Fast training
- ✓ Proven to work

**Cons:**
- ✗ Doesn't leverage full temporal information
- ✗ Loses information about trends and dynamics

### Option 2: **Improve ODE-RNN Hyperparameters**
```python
# Try these in ODETime.py:
hidden_dim = 128  # or 256
epochs = 100      # or 200 with early stopping
lr = 0.0005       # lower learning rate
batch_size = 16   # smaller for stability
```

**Expected improvement:** AUC 0.746 → 0.76-0.78 (still likely below MLP)

### Option 3: **Simpler Temporal Model - GRU/LSTM**
Create a basic recurrent model without ODE complexity:

```python
class SimpleGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.gru = nn.GRU(input_dim * 2, hidden_dim, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, values, masks, lengths):
        # Concatenate values and masks
        x = torch.cat([values, masks], dim=-1)

        # Pack padded sequence
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )

        # Run GRU
        _, hidden = self.gru(packed)

        # Classify
        return torch.sigmoid(self.classifier(hidden[-1]))
```

**Expected performance:** AUC 0.75-0.78

**Pros:**
- ✓ Simpler than ODE-RNN
- ✓ Still models temporal dynamics
- ✓ Faster training
- ✓ Less prone to instability

### Option 4: **MLP with Time-Aware Features** (Best compromise)
Extract temporal statistics from time series and feed to MLP:

```python
# For each feature, compute:
features = [
    'last_value',      # Current value (like original MLP)
    'first_value',     # Initial value
    'mean_value',      # Average over window
    'std_value',       # Variation
    'trend',           # Linear trend (slope)
    'time_since_last', # Recency indicator
]

# Result: 25 original features → 150 temporal features
# Feed to MLP classifier
```

**Expected performance:** AUC 0.80-0.82

**Pros:**
- ✓✓ Likely best performance
- ✓ Captures temporal patterns without complexity
- ✓ Simple MLP architecture (proven to work)
- ✓ Interpretable features

**Cons:**
- ✗ Feature engineering required
- ✗ Doesn't model interactions between time points

## Summary

**For Research/Production:**
- **Use MLP with time-aware features** (Option 4)
- If that's too complex, **stick with current MLP** (Option 1)

**For Learning/Experimentation:**
- Try **SimpleGRU** (Option 3) to understand temporal modeling
- ODE-RNN is interesting but might be overkill for this problem

## Why ODE-RNN Might Not Be Ideal Here

1. **Medical context**: AKI prediction is about **current state** more than **trajectory**
2. **MLP "last" = most recent labs** → these are what doctors use clinically
3. **Temporal evolution might add noise** rather than signal in sparse medical data
4. **ODE complexity** assumes smooth continuous dynamics, but medical measurements are discrete/sparse

## Next Steps

**Quick Win:**
```bash
# Run MLP with different aggregations
python MLP_OnlyTime.py --how="mean"    # Try mean instead of last
python MLP_OnlyTime.py --how="max"     # Try max
```

**Best Investment:**
Create `MLP_TimeAware.py` with temporal feature engineering (Option 4)
