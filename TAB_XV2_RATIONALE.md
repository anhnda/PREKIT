# Tab_Xv2: Fair Comparison with Baseline

## Problem with Tab_Xv1

**Tab_Xv1 vs Baseline used DIFFERENT feature sets:**

```
Baseline Features:
├─ getMeasuresBetween(..., "last")
├─ Returns ALL features (static + temporal)
└─ ~70+ features total

Tab_Xv1 Features:
├─ SimpleStaticEncoder(FIXED_FEATURES)  # 22 static features
├─ get_all_temporal_features() last values  # 15 temporal features
├─ Learned Z  # 32 dimensions
└─ ~69 features total (BUT DIFFERENT ONES!)
```

**Result:** Comparing apples to oranges! Can't tell if learned Z helps or if difference is due to feature selection.

## Solution: Tab_Xv2

**Use IDENTICAL baseline features, only add learned context:**

```
Baseline:
└─ [All Static + All Temporal Last Values]

Tab_Xv2:
└─ [All Static + All Temporal Last Values] + [Learned Z]
```

**Result:** Pure comparison! Any improvement is directly attributable to learned temporal context.

## Key Changes in Tab_Xv2

### 1. Dataset (Lines 89-149)
```python
class HybridDatasetV2(Dataset):
    def __init__(self, patients, feature_names, baseline_features_df, ...):
        # Store baseline features (same as baseline uses)
        baseline_vec = baseline_features_df.iloc[idx].values
        self.baseline_features.append(...)
```
Takes `baseline_features_df` from `getMeasuresBetween()` - the EXACT features baseline uses.

### 2. Feature Extraction (Lines 294-323)
```python
def extract_features_with_context(policy_net, loader, ...):
    # Get learned temporal context
    z_np = ...

    # Use baseline features as-is
    baseline_np = baseline_feats.numpy()

    # Concatenate: [Baseline + Z]
    combined = np.hstack([baseline_np, z_np])
```
Simple concatenation: Baseline features + Learned Z. No other changes.

### 3. Main Loop (Lines 493-534)
```python
# Extract baseline features (same as baseline uses)
df_train_base = train_p_obj.getMeasuresBetween(
    pd.Timedelta(hours=-6), pd.Timedelta(hours=24), "last", ...
)

# Encode (same encoding as baseline)
df_train_enc, df_test_enc, df_val_enc = encodeCategoricalData(...)

X_train_baseline = df_train_enc.drop(columns=["akd"]).fillna(0)

# Create datasets WITH these baseline features
train_ds = HybridDatasetV2(train_p, temporal_feats, X_train_baseline)
```
Uses exact same feature extraction pipeline as baseline.

## Expected Outcomes

### Scenario 1: Tab_Xv2 > Baseline
**Interpretation:** Learned temporal context provides information beyond handcrafted last values!
- The Bi-GRU encoder captures temporal patterns that simple "last value" misses
- The 7 learned statistics (mean, std, min, max, slope, etc.) add value
- **Conclusion:** Deep learning > handcrafted features ✓

### Scenario 2: Tab_Xv2 ≈ Baseline
**Interpretation:** Learned context is redundant with baseline features
- The "last value" already contains most temporal information
- Learned statistics don't add much beyond what's already there
- **Conclusion:** For this task, handcrafted features sufficient

### Scenario 3: Tab_Xv2 < Baseline
**Interpretation:** Learned context is noisy or RL training degraded performance
- Need to check if RL training is stable
- May need better architecture or training procedure
- **Conclusion:** Current approach needs improvement

## Why This Matters

### Scientific Rigor
- **Tab_Xv1**: Confounded comparison (different features + learned context)
- **Tab_Xv2**: Controlled comparison (same features ± learned context)

### Actionable Insights
With Tab_Xv2, we can definitively answer:
1. **Does learned temporal context help?** → Compare Tab_Xv2 vs Baseline
2. **How much does it help?** → Measure exact AUPR improvement
3. **Is it worth the complexity?** → Cost-benefit analysis

### Next Steps Based on Results
- **If Tab_Xv2 >> Baseline**: Use Tab_Xv2, learned context works!
- **If Tab_Xv2 ≈ Baseline**: Use Baseline, simpler is better
- **If Tab_Xv2 < Baseline**: Debug training, improve architecture

## Conservative RL Settings

To avoid the degradation seen in Tab_Xv1, Tab_Xv2 uses:

```python
# Very conservative learning rate
lr = 0.00003  # Was 0.00005 in Xv1

# Very slow temperature decay
temperature = max(0.5, 1.0 - epoch / (epochs * 1.5))  # Slower than Xv1

# Stricter early stopping
patience = 12  # Was 15 in Xv1

# Conservative reward weight
aupr_weight = 0.1 + 0.2 * min(1.0, epoch / 30)  # 0.1 → 0.3 (was 0.2 → 0.4)

# Stronger entropy
entropy_bonus = 0.01  # Was 0.005 in Xv1
```

These settings prioritize **stability over exploration**. Better to maintain pretrain performance than risk degradation.

## Comparison Table

| Aspect | Baseline | Tab_Xv1 | Tab_Xv2 |
|--------|----------|---------|---------|
| Static Features | All (getMeasuresBetween) | Selected (FIXED_FEATURES) | All (getMeasuresBetween) ✓ |
| Temporal Features | All last values | Selected 15 last values | All last values ✓ |
| Learned Context | None | Z (32 dim) | Z (32 dim) ✓ |
| Fair Comparison? | N/A | ✗ Different features | ✓ Same features |
| Isolates Contribution? | N/A | ✗ Confounded | ✓ Clean comparison |

## Expected Performance

Based on Tab_Xv1 results:

**Fold 0:**
- Baseline: 0.8192
- Tab_Xv1: 0.8302 (+1.3%)
- **Tab_Xv2 prediction: ~0.82-0.83** (similar to Tab_Xv1 since features are comparable)

**Fold 1:**
- Baseline: 0.8514
- Tab_Xv1: 0.8308 (-2.4%)
- **Tab_Xv2 prediction: ~0.85-0.86** (should match or beat baseline with same features)

**Fold 2:**
- Baseline: 0.8283
- Tab_Xv1: 0.7789 (-6.0%)
- **Tab_Xv2 prediction: ~0.83-0.84** (using baseline features should prevent this drop)

**Key insight:** Tab_Xv2 should have **lower variance** across folds because it uses the same robust feature set as baseline.

---

**Bottom line:** Tab_Xv2 is a scientifically rigorous test of whether deep learning can improve on handcrafted features, with a controlled experiment design.
