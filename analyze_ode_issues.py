"""
Analyze potential issues with ODE-RNN performance compared to MLP.
"""

import pandas as pd
import numpy as np
import sys

sys.path.append("/Users/anhnd/CodingSpace/Python/PREDKIT")
from constants import NULLABLE_MEASURES
from utils.class_patient import Patients
from utils.prepare_data import trainTestPatients
from ODETime import get_all_temporal_features, extract_temporal_data

# Load patients
print("Loading patients...")
patients = Patients.loadPatients()
patients.fillMissingMeasureValue(NULLABLE_MEASURES, 0)

measures = patients.getMeasures()
for measure, count in measures.items():
    if count < len(patients) * 80 / 100:
        patients.removeMeasures([measure])

patients.removePatientByMissingFeatures()
print(f"Total patients: {len(patients)}\n")

# Get features
all_features = get_all_temporal_features(patients)
print(f"Total features: {len(all_features)}")

# Get first fold
train_patients, test_patients = next(trainTestPatients(patients))

print("\n" + "="*80)
print("ANALYZING TEMPORAL DATA CHARACTERISTICS")
print("="*80)

# Analyze sequence lengths
sequence_lengths = []
measurement_counts = []
time_spans = []
missing_ratios = []

for patient in train_patients.patientList[:100]:
    times, values, masks = extract_temporal_data(patient, all_features)

    if times is not None:
        sequence_lengths.append(len(times))

        # Count measurements per timestamp
        for mask_vec in masks:
            measurement_counts.append(sum(mask_vec))

        # Missing data ratio
        all_masks = np.array(masks)
        total_possible = all_masks.size
        total_observed = all_masks.sum()
        missing_ratios.append(1 - total_observed / total_possible)

print(f"\nSequence length statistics (first 100 patients):")
print(f"  Mean: {np.mean(sequence_lengths):.1f}")
print(f"  Median: {np.median(sequence_lengths):.1f}")
print(f"  Min: {np.min(sequence_lengths)}")
print(f"  Max: {np.max(sequence_lengths)}")
print(f"  Std: {np.std(sequence_lengths):.1f}")

print(f"\nMeasurements per timestamp:")
print(f"  Mean: {np.mean(measurement_counts):.1f}")
print(f"  Median: {np.median(measurement_counts):.1f}")
print(f"  (Out of {len(all_features)} possible features)")

print(f"\nMissing data ratio per patient:")
print(f"  Mean: {np.mean(missing_ratios):.2%}")
print(f"  Median: {np.median(missing_ratios):.2%}")

# Analyze temporal patterns
print("\n" + "="*80)
print("TEMPORAL PATTERNS")
print("="*80)

# Check how MLP aggregates vs how ODE uses time series
# For MLP: Uses "last" value
# For ODE: Uses all time points with evolution

# Sample a patient
sample_patient = train_patients.patientList[0]
times, values, masks = extract_temporal_data(sample_patient, all_features)

if times is not None:
    print(f"\nExample patient with {len(times)} time points:")

    # Check a feature that varies over time (e.g., hr)
    hr_idx = all_features.index('hr') if 'hr' in all_features else 0

    hr_measurements = []
    for i, (value_vec, mask_vec) in enumerate(zip(values, masks)):
        if mask_vec[hr_idx] > 0:
            hr_measurements.append((i, value_vec[hr_idx]))

    if hr_measurements:
        print(f"\n  HR measurements: {len(hr_measurements)} observations")
        print(f"  First 5 observations: {hr_measurements[:5]}")

        # Check if there's significant variation
        hr_values = [v for _, v in hr_measurements]
        if len(hr_values) > 1:
            print(f"  HR range: {min(hr_values):.1f} - {max(hr_values):.1f}")
            print(f"  HR std: {np.std(hr_values):.1f}")

print("\n" + "="*80)
print("POTENTIAL ISSUES AND RECOMMENDATIONS")
print("="*80)

issues = []
recommendations = []

# Issue 1: Sparse sequences
avg_seq_len = np.mean(sequence_lengths)
if avg_seq_len < 10:
    issues.append("Very short sequences (avg < 10 time points)")
    recommendations.append("ODE-RNN needs sufficient temporal points to learn dynamics")

# Issue 2: High missing data
avg_missing = np.mean(missing_ratios)
if avg_missing > 0.8:
    issues.append(f"Very high missing data ({avg_missing:.1%})")
    recommendations.append("Consider using only features with >50% observations")

# Issue 3: Time normalization
issues.append("Time normalization to [0,1] per patient loses absolute timing information")
recommendations.append("Consider keeping absolute hours or using relative time deltas")

# Issue 4: Average time delta in ODE
issues.append("Using average time delta across batch loses patient-specific temporal information")
recommendations.append("Process patients individually or use per-patient time deltas")

# Issue 5: Model capacity
issues.append("Hidden dim=64 might be insufficient for 25 features")
recommendations.append("Try hidden_dim=128 or 256")

# Issue 6: Training epochs
issues.append("Only 50 epochs might not be enough for ODE-RNN convergence")
recommendations.append("Try 100-200 epochs with early stopping")

# Issue 7: MLP uses "last" aggregation
issues.append("MLP uses 'last' value which might be closest to AKI event")
recommendations.append("ODE-RNN might be weighting earlier observations too much")

print("\nIdentified Issues:")
for i, issue in enumerate(issues, 1):
    print(f"{i}. {issue}")

print("\nRecommendations:")
for i, rec in enumerate(recommendations, 1):
    print(f"{i}. {rec}")

print("\n" + "="*80)
print("KEY INSIGHT")
print("="*80)
print("""
MLP with 'last' aggregation (AUC 0.795) works well because:
- It uses the most recent measurement before AKI
- Recent measurements are most predictive of imminent AKI
- Simple aggregation avoids overfitting to temporal noise

ODE-RNN (AUC 0.746) might be struggling because:
- It tries to model full temporal evolution
- May overfit to early measurements that are less predictive
- More complex model needs more data/better hyperparameters
- Time normalization and batching approximations lose information

SUGGESTION: Try simpler temporal models first:
1. MLP with time-aware features (mean, std, trend of each feature)
2. LSTM/GRU without ODE (simpler recurrent model)
3. Attention-based model (weight recent measurements more)
""")
