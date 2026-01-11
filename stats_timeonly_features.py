"""
Analyze time gaps between last timestamp of each feature and the overall last timestamp.

For each patient:
- total_last = the latest timestamp across ALL temporal features
- last_i = the latest timestamp for feature i
- time_gap_i = total_last - last_i

This helps understand how "stale" each feature's measurements are.
"""

import pandas as pd
import numpy as np
import sys
from matplotlib import pyplot as plt
import seaborn as sns
from collections import defaultdict

sys.path.append("/Users/anhnd/CodingSpace/Python/PREDKIT")
from constants import NULLABLE_MEASURES
from utils.class_patient import Patients

# Fixed/demographic features to exclude (same as MLP_OnlyTime.py)
FIXED_FEATURES = [
    # Demographics
    "age",
    "gender",
    "race",
    # Medical history (fixed at admission)
    "chronic_pulmonary_disease",
    "ckd_stage",
    "congestive_heart_failure",
    "dka_type",
    "history_aci",
    "history_ami",
    "hypertension",
    "liver_disease",
    "macroangiopathy",
    "malignant_cancer",
    "microangiopathy",
    "uti",
    # Severity scores (calculated at admission, not time-varying)
    "oasis",
    "saps2",
    "sofa",
    # Interventions that may not have timestamps
    "mechanical_ventilation",
    "use_NaHCO3",
    # Length of stay (not a temporal measurement)
    "preiculos",
    # GCS unable flag (not a measurement)
    "gcs_unable",
    # Data leak features (directly related to AKI outcome)
    "egfr",  # Estimated GFR - calculated from creatinine, reflects kidney function
]


def load_patients():
    """Load and prepare patients."""
    patients = Patients.loadPatients()
    print(f"Loaded {len(patients)} patients")

    patients.fillMissingMeasureValue(NULLABLE_MEASURES, 0)

    # Remove measures with less than 80% of data
    measures = patients.getMeasures()
    for measure, count in measures.items():
        if count < len(patients) * 80 / 100:
            patients.removeMeasures([measure])

    # Remove patients with more than 20% missing features
    patients.removePatientByMissingFeatures()
    print(f"After cleanup: {len(patients)} patients")

    return patients


def get_temporal_features(patient):
    """Get only temporal features (those with timestamps) for a patient.

    Returns only measures that are SortedDict (contain timestamps as keys).
    """
    all_measures = patient.measures

    # Filter out fixed features AND only keep SortedDict (temporal) measures
    temporal_measures = {}
    for measure_name, measure_values in all_measures.items():
        # Skip fixed features
        if measure_name in FIXED_FEATURES:
            continue

        # Only include if it's a dict-like object (SortedDict) with timestamps
        if hasattr(measure_values, 'keys') and hasattr(measure_values, 'values'):
            temporal_measures[measure_name] = measure_values

    return temporal_measures


def calculate_time_gaps(patients):
    """
    Calculate time gaps between each feature's last timestamp and the overall last timestamp.

    Returns:
        dict: {feature_name: [list of time gaps in hours]}
    """
    feature_time_gaps = defaultdict(list)

    for patient_idx, patient in enumerate(patients.patientList):
        if patient_idx % 100 == 0:
            print(f"Processing patient {patient_idx}/{len(patients)}")

        temporal_measures = get_temporal_features(patient)

        if not temporal_measures:
            continue

        # Find the overall last timestamp across all features
        all_timestamps = []
        feature_last_times = {}

        for measure_name, measure_dict in temporal_measures.items():
            if not measure_dict:
                continue

            # measure_dict is a SortedDict where keys are timestamps
            # Get all timestamps (keys) for this feature
            timestamps = list(measure_dict.keys())

            if timestamps:
                # Convert to pandas Timestamp if needed
                timestamps = [pd.Timestamp(t) if not isinstance(t, pd.Timestamp) else t
                             for t in timestamps]

                all_timestamps.extend(timestamps)
                feature_last_times[measure_name] = max(timestamps)

        if not all_timestamps:
            continue

        total_last = max(all_timestamps)

        # Calculate time gap for each feature
        for measure_name, last_time in feature_last_times.items():
            time_gap = total_last - last_time
            time_gap_hours = time_gap.total_seconds() / 3600  # Convert to hours
            feature_time_gaps[measure_name].append(time_gap_hours)

    return dict(feature_time_gaps)


def plot_time_gap_distributions(feature_time_gaps):
    """Visualize the distribution of time gaps for each feature."""

    # Prepare data for plotting
    features = sorted(feature_time_gaps.keys())

    # Calculate statistics
    stats_data = []
    for feature in features:
        gaps = feature_time_gaps[feature]
        stats_data.append({
            'Feature': feature,
            'Mean (hrs)': np.mean(gaps),
            'Median (hrs)': np.median(gaps),
            'Std (hrs)': np.std(gaps),
            'Min (hrs)': np.min(gaps),
            'Max (hrs)': np.max(gaps),
            'Count': len(gaps),
            'Pct_Zero': (np.array(gaps) == 0).sum() / len(gaps) * 100,
        })

    stats_df = pd.DataFrame(stats_data)
    stats_df = stats_df.sort_values('Mean (hrs)', ascending=False)

    print("\n" + "="*80)
    print("TIME GAP STATISTICS (hours)")
    print("="*80)
    print(stats_df.to_string(index=False))
    print("="*80)

    # Save to CSV
    stats_df.to_csv("result/time_gap_statistics.csv", index=False)
    print("\nSaved statistics to: result/time_gap_statistics.csv")

    # Plot 1: Box plot of time gaps
    fig, ax = plt.subplots(figsize=(14, 10))

    # Prepare data for box plot
    data_for_plot = []
    labels_for_plot = []
    for feature in stats_df['Feature']:
        gaps = feature_time_gaps[feature]
        # Filter out very large outliers for better visualization
        gaps_filtered = [g for g in gaps if g <= 48]  # Only show gaps <= 48 hours
        if gaps_filtered:
            data_for_plot.append(gaps_filtered)
            labels_for_plot.append(feature)

    bp = ax.boxplot(data_for_plot, labels=labels_for_plot, vert=False, patch_artist=True)

    # Color the boxes
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')

    ax.set_xlabel('Time Gap (hours)', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)
    ax.set_title('Distribution of Time Gaps: Last Feature Measurement to Overall Last Measurement\n(Filtered to ≤48 hours for visualization)', fontsize=14)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig("result/time_gap_boxplot.png", dpi=300, bbox_inches="tight")
    print("Saved box plot to: result/time_gap_boxplot.png")
    plt.show()

    # Plot 2: Histogram of mean time gaps per feature
    fig, ax = plt.subplots(figsize=(12, 8))

    colors = plt.cm.viridis(np.linspace(0, 1, len(stats_df)))
    bars = ax.barh(stats_df['Feature'], stats_df['Mean (hrs)'], color=colors)

    ax.set_xlabel('Mean Time Gap (hours)', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)
    ax.set_title('Mean Time Gap by Feature: Last Feature Measurement to Overall Last Measurement', fontsize=14)
    ax.grid(axis='x', alpha=0.3)

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, stats_df['Mean (hrs)'])):
        ax.text(val + 0.1, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig("result/time_gap_mean_barplot.png", dpi=300, bbox_inches="tight")
    print("Saved bar plot to: result/time_gap_mean_barplot.png")
    plt.show()

    # Plot 3: Heatmap showing percentage of patients with zero time gap
    fig, ax = plt.subplots(figsize=(10, 8))

    pct_zero_sorted = stats_df.sort_values('Pct_Zero', ascending=False)
    colors_heat = plt.cm.RdYlGn(pct_zero_sorted['Pct_Zero'] / 100)

    bars = ax.barh(pct_zero_sorted['Feature'], pct_zero_sorted['Pct_Zero'], color=colors_heat)

    ax.set_xlabel('Percentage of Patients (%)', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)
    ax.set_title('Percentage of Patients Where Feature Has the Latest Timestamp (Zero Time Gap)', fontsize=14)
    ax.grid(axis='x', alpha=0.3)

    # Add value labels
    for bar, val in zip(bars, pct_zero_sorted['Pct_Zero']):
        ax.text(val + 0.5, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}%', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig("result/time_gap_zero_percentage.png", dpi=300, bbox_inches="tight")
    print("Saved zero percentage plot to: result/time_gap_zero_percentage.png")
    plt.show()

    # Plot 4: Distribution of all time gaps (histogram)
    fig, ax = plt.subplots(figsize=(12, 6))

    all_gaps = []
    for gaps in feature_time_gaps.values():
        all_gaps.extend(gaps)

    # Filter for visualization
    all_gaps_filtered = [g for g in all_gaps if g <= 48]

    ax.hist(all_gaps_filtered, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Time Gap (hours)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Overall Distribution of Time Gaps Across All Features and Patients\n(Filtered to ≤48 hours)', fontsize=14)
    ax.grid(axis='y', alpha=0.3)

    # Add statistics text
    mean_gap = np.mean(all_gaps)
    median_gap = np.median(all_gaps)
    zero_pct = (np.array(all_gaps) == 0).sum() / len(all_gaps) * 100

    stats_text = f'Mean: {mean_gap:.2f}h\nMedian: {median_gap:.2f}h\nZero gaps: {zero_pct:.1f}%'
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig("result/time_gap_overall_histogram.png", dpi=300, bbox_inches="tight")
    print("Saved overall histogram to: result/time_gap_overall_histogram.png")
    plt.show()


def main():
    """Main analysis function."""
    print("="*80)
    print("TIME GAP ANALYSIS FOR TEMPORAL FEATURES")
    print("="*80)

    # Load data
    patients = load_patients()

    # Calculate time gaps
    print("\nCalculating time gaps...")
    feature_time_gaps = calculate_time_gaps(patients)

    print(f"\nAnalyzed {len(feature_time_gaps)} temporal features")

    # Visualize
    print("\nGenerating visualizations...")
    plot_time_gap_distributions(feature_time_gaps)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
