# Tab_Xv4: The Critical Fix

## 🔴 What Went Wrong in Tab_Xv3

Looking at the training logs:
```
Epoch 3  | Val AUPR: 0.6224 | Stat MSE: 13.160132
Epoch 48 | Val AUPR: 0.7950 | Stat MSE: 12.964899
```

**Problem:** Stat MSE stays at ~13 and barely decreases! This means the model is NOT learning to match handcrafted features at all.

### Root Cause: Dimension Mismatch

**Tab_Xv3 Architecture:**
```python
# Process ALL features together in Bi-GRU
values: (batch, seq_len, 15 features) → Bi-GRU → hidden states

# Extract global statistics from RNN outputs
learned_stats: (batch, 7 * hidden*2) = (batch, 7*64) = (batch, 448)

# But handcrafted features are PER FEATURE!
true_stats: (batch, num_features * 7) = (batch, 15*7) = (batch, 105)

# Dimension mismatch: 448 ≠ 105
# Solution in Xv3: padding/truncation... doesn't work!
```

**The fundamental problem:**
- Bi-GRU processes all features jointly → learns global temporal patterns
- Handcrafted stats are per-feature → independent statistics for each feature
- **Apples to oranges comparison!**

### Example to Illustrate

**Handcrafted (Tab_RLv4):**
```python
For Feature 0 (e.g., heart_rate):
  last=120, mean=115, std=5, min=110, max=125, slope=2, count=10

For Feature 1 (e.g., blood_pressure):
  last=140, mean=135, std=8, min=125, max=145, slope=-1, count=12

# Total: 15 features * 7 stats = 105 values
```

**Tab_Xv3 (WRONG):**
```python
# All features → Bi-GRU → Global hidden states
# Extract 7 global statistics from hidden states
# Result: 448 values that don't correspond to any specific feature!

# When we try to match:
learned[0:105] vs true[0:105]  # Arbitrary truncation!
# Model has no idea what it's supposed to learn
```

## ✅ The Fix: Tab_Xv4

### Key Insight
**Process EACH temporal feature INDEPENDENTLY**, just like handcrafted features!

```python
For Feature 0:
  time_series_0 → 1D CNN + Attention → [last, mean, std, min, max, slope, count]

For Feature 1:
  time_series_1 → 1D CNN + Attention → [last, mean, std, min, max, slope, count]

...

For Feature N:
  time_series_N → 1D CNN + Attention → [last, mean, std, min, max, slope, count]

# Total output: N features * 7 stats
# Matches handcrafted EXACTLY!
```

### Architecture Comparison

**Tab_Xv3 (Global Processing):**
```
[Feature0, Feature1, ..., Feature14]
           ↓
       Bi-GRU (process all together)
           ↓
   Global hidden states
           ↓
   7 global statistics (448-dim)
           ↓
   ??? How to match per-feature stats? ???
```

**Tab_Xv4 (Per-Feature Processing):**
```
Feature0 → 1D CNN → Attn → [7 stats for Feature0]
Feature1 → 1D CNN → Attn → [7 stats for Feature1]
...
Feature14 → 1D CNN → Attn → [7 stats for Feature14]
           ↓
Concatenate: [7*15 = 105 statistics]
           ↓
Perfect 1-to-1 correspondence with handcrafted!
```

## 🏗️ Tab_Xv4 Architecture Details

### PerFeatureStatisticalEncoder (Lines 119-167)

```python
class PerFeatureStatisticalEncoder(nn.Module):
    def forward(self, values, mask):
        # values: (batch, seq_len) - ONE feature's time series

        # 1D CNN to extract patterns
        x = Conv1d(values)  # (batch, hidden, seq_len)

        # Compute statistics:
        last_state = x[last_valid_idx]
        max_pool = max(x)
        context = attention_weighted_avg(x)

        # Predict 7 statistics
        stats = MLP([last, max, context])  # (batch, 7)
        return stats
```

### MultiFeatureStatisticalEncoder (Lines 170-198)

```python
class MultiFeatureStatisticalEncoder(nn.Module):
    def forward(self, batch_data):
        # Process each feature independently
        all_stats = []
        for f_idx in range(num_features):
            feat_values = values[:, :, f_idx]  # One feature
            stats = self.feature_encoder(feat_values)  # (batch, 7)
            all_stats.append(stats)

        # Stack: (batch, num_features, 7) → (batch, num_features*7)
        return flatten(all_stats)
```

## 📊 Expected Results

### Stat MSE Should Decrease!

**Tab_Xv3 (WRONG):**
```
Stat MSE: 13.16 → 12.96 (barely changes)
Reason: Model doesn't know what to learn
```

**Tab_Xv4 (CORRECT):**
```
Stat MSE: ?? → < 1.0 (should decrease significantly!)
Reason: Clear 1-to-1 correspondence, proper supervision
```

### Performance Should Improve

**Expected outcome:**

| Version | Stat MSE | AUPR | Reason |
|---------|----------|------|--------|
| Tab_Xv3 | ~13 (bad) | < Baseline | Model not learning |
| Tab_Xv4 | < 1 (good) | ≥ Baseline | Proper supervision |

If Tab_Xv4 still struggles, then:
- Check if 1D CNN is powerful enough (may need more layers)
- Increase supervision weight (0.5 → 0.7)
- Or just use handcrafted features (Tab_RLv4)

## 🎯 Why This Should Work

1. **Architecture matches problem structure**
   - Handcrafted: per-feature stats
   - Tab_Xv4: per-feature encoder
   - ✓ Perfect alignment!

2. **Dimensions match exactly**
   - Learned: (batch, N*7)
   - True: (batch, N*7)
   - ✓ No padding/truncation hacks!

3. **Clear supervision signal**
   - Each of the 7 outputs has clear meaning
   - Model knows exactly what it should predict
   - ✓ Strong learning signal!

4. **Simpler architecture**
   - No complex Bi-GRU over all features
   - Simple 1D CNN per feature
   - ✓ Easier to train and debug!

## 🔬 Validation Checklist

When running Tab_Xv4, check:

1. ✓ **Stat MSE decreases?**
   - Should go from ~5-10 → < 1.0
   - If stays high, architecture is still wrong

2. ✓ **Val AUPR improves?**
   - Should reach pretrain AUPR > 0.75
   - If low, need better encoder

3. ✓ **Test AUPR ≥ Baseline?**
   - Should match or beat baseline
   - If not, learned stats still not good enough

4. ✓ **Per-feature stats make sense?**
   - Print learned vs true for one sample
   - Manually verify they're correlated

## 💡 Key Lesson

**Match your architecture to the problem structure!**

- Problem: Compute statistics PER temporal feature
- Wrong: Process all features together (Tab_Xv3)
- Right: Process each feature independently (Tab_Xv4)

Just like how CNNs use convolutions for spatial structure, we use per-feature processing for feature-independent statistics.

---

**Bottom Line:** Tab_Xv4 fixes the fundamental architectural mismatch. Now the model has a chance to actually learn what handcrafted features compute!
