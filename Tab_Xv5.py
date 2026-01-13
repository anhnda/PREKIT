"""
Tab_Xv5: Extended Handcrafted Features (Pure Statistical Baseline)

CRITICAL INSIGHT from Tab_RLv4 analysis:
Problem with supervised learning approach:
- Training uses: Encoder → Neural Head (differentiable) ← Wrong objective!
- Testing uses: Features → TabPFN (non-differentiable) ← Real objective!
- Features optimized for neural head ≠ Features TabPFN needs

RL solves this by using TabPFN DIRECTLY during training.

THIS VERSION:
Skip learned features entirely. Just test if EXTENDED HANDCRAFTED features help.

Compare:
- Baseline: getMeasuresBetween(..., "last") → Only LAST value per temporal feature
- Tab_Xv5: Extract 7 stats per temporal feature [last, mean, std, min, max, slope, count]

This isolates the contribution of richer handcrafted features.
If this works, we can add RL-optimized learned features on top (like Tab_RLv4).
"""

import pandas as pd
import numpy as np
import sys
import copy
from matplotlib import pyplot as plt
import torch
import random
import os

from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    auc,
)

os.environ["SEGMENT_WRITE_KEY"] = ""
os.environ["ANALYTICS_WRITE_KEY"] = ""
os.environ["TABPFN_DISABLE_ANALYTICS"] = "1"

try:
    import analytics
    analytics.write_key = None
    analytics.disable()
except Exception:
    pass

from tabpfn import TabPFNClassifier

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

xseed = 42
seed_everything(xseed)

PT = "/Users/anhnd/CodingSpace/Python/PREDKIT"
if sys.platform != "darwin":
    PT = "/home/anhnda/PREKIT"
sys.path.append(PT)

from utils.prepare_data import trainTestPatients, encodeCategoricalData
from TimeEmbeddingVal import (
    get_all_temporal_features,
    extract_temporal_data,
    load_and_prepare_patients,
    split_patients_train_val,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================================================================
# Static Features Encoder
# ==============================================================================

FIXED_FEATURES = [
    "age", "gender", "race", "chronic_pulmonary_disease", "ckd_stage",
    "congestive_heart_failure", "dka_type", "history_aci", "history_ami",
    "hypertension", "liver_disease", "macroangiopathy", "malignant_cancer",
    "microangiopathy", "uti", "oasis", "saps2", "sofa",
    "mechanical_ventilation", "use_NaHCO3", "preiculos", "gcs_unable"
]

class SimpleStaticEncoder:
    def __init__(self, features):
        self.features = features
        self.mappings = {f: {} for f in features}
        self.counts = {f: 0 for f in features}

    def fit(self, patients):
        for p in patients:
            for f in self.features:
                val = p.measures.get(f, 0.0)
                if hasattr(val, 'values') and len(val) > 0:
                    val = list(val.values())[0]
                elif hasattr(val, 'values'):
                    val = 0.0

                val_str = str(val)
                try:
                    float(val)
                except ValueError:
                    if val_str not in self.mappings[f]:
                        self.mappings[f][val_str] = float(self.counts[f])
                        self.counts[f] += 1

    def transform(self, patient):
        vec = []
        for f in self.features:
            val = patient.measures.get(f, 0.0)
            if hasattr(val, 'values') and len(val) > 0:
                val = list(val.values())[0]
            elif hasattr(val, 'values'):
                val = 0.0

            try:
                numeric_val = float(val)
            except ValueError:
                numeric_val = self.mappings[f].get(str(val), -1.0)
            vec.append(numeric_val)
        return vec

# ==============================================================================
# Extended Feature Extraction (Pure Handcrafted, No Learning)
# ==============================================================================

def extract_extended_features_from_patients(patients, temporal_feats, static_encoder):
    """
    Extract extended handcrafted features:
    [Static Features + 7 Statistics per Temporal Feature]

    7 statistics: [last, mean, std, min, max, slope, count]

    No learning, just pure statistical extraction like Tab_RLv4 does.
    """
    all_features = []
    all_labels = []

    # Collect all values for normalization
    all_values = []
    for patient in patients:
        times, values, masks = extract_temporal_data(patient, temporal_feats)
        if times is None:
            continue
        for v_vec, m_vec in zip(values, masks):
            for v, m in zip(v_vec, m_vec):
                if m > 0:
                    all_values.append(v)

    all_values = np.array(all_values)
    mean_val = np.mean(all_values) if len(all_values) > 0 else 0.0
    std_val = np.std(all_values) if len(all_values) > 0 else 1.0

    # Extract features for each patient
    for patient in patients:
        times, values, masks = extract_temporal_data(patient, temporal_feats)
        if times is None:
            continue

        # Static features
        static_vec = static_encoder.transform(patient)

        # Normalize temporal values
        norm_values = []
        for v_vec, m_vec in zip(values, masks):
            norm = [(v - mean_val) / std_val if m > 0 else 0.0
                    for v, m in zip(v_vec, m_vec)]
            norm_values.append(norm)

        # Extract 7 statistics per temporal feature
        temporal_stats = []
        for f_idx in range(len(temporal_feats)):
            f_vals = np.array([norm_values[t][f_idx] for t in range(len(norm_values))])
            f_mask = np.array([masks[t][f_idx] for t in range(len(masks))])
            f_times = np.array(times)
            valid_idx = np.where(f_mask > 0)[0]

            if len(valid_idx) > 0:
                valid_vals = f_vals[valid_idx]
                valid_times = f_times[valid_idx]

                last = valid_vals[-1]
                mean = np.mean(valid_vals)
                std = np.std(valid_vals) if len(valid_vals) > 1 else 0.0
                min_val = np.min(valid_vals)
                max_val = np.max(valid_vals)
                count = len(valid_vals)

                if len(valid_vals) >= 2:
                    slope = (valid_vals[-1] - valid_vals[0]) / (valid_times[-1] - valid_times[0] + 1e-6)
                else:
                    slope = 0.0

                temporal_stats.extend([last, mean, std, min_val, max_val, slope, count])
            else:
                temporal_stats.extend([0.0] * 7)

        # Combine: [Static + Extended Temporal Stats]
        combined = static_vec + temporal_stats
        all_features.append(combined)
        all_labels.append(1 if patient.akdPositive else 0)

    return np.array(all_features), np.array(all_labels)

# ==============================================================================
# Main
# ==============================================================================

def main():
    print("="*80)
    print("Tab_Xv5: Pure Extended Handcrafted Features (No Learning)")
    print("="*80)
    print("\nStrategy:")
    print("  1. Extract RICH handcrafted features (7 stats per temporal feature)")
    print("  2. Feed directly to TabPFN (no learned features)")
    print("  3. Test if extended statistics alone beat baseline")
    print("\nWhy no learning?")
    print("  - Supervised learning optimizes for neural head, not TabPFN")
    print("  - RL is needed to optimize for TabPFN directly")
    print("  - First test if extended handcrafted features help alone")
    print("="*80)

    patients = load_and_prepare_patients()
    temporal_feats = get_all_temporal_features(patients)

    # Static encoder
    static_encoder = SimpleStaticEncoder(FIXED_FEATURES)
    static_encoder.fit(patients.patientList)

    print(f"\nStatic features: {len(FIXED_FEATURES)}")
    print(f"Temporal features: {len(temporal_feats)}")
    print(f"Extended stats: {len(temporal_feats)} x 7 = {len(temporal_feats) * 7}")
    print(f"Total features: {len(FIXED_FEATURES) + len(temporal_feats) * 7}")

    metrics_xv5 = {k: [] for k in ['auc', 'auc_pr']}
    metrics_baseline = {k: [] for k in ['auc', 'auc_pr']}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    for fold, (train_full, test_p) in enumerate(trainTestPatients(patients, seed=xseed)):
        print(f"\n{'='*80}")
        print(f"Fold {fold}")
        print('='*80)

        train_p_obj, val_p_obj = split_patients_train_val(train_full, val_ratio=0.1, seed=42+fold)

        # Extract extended features
        print("  Extracting extended handcrafted features...")
        X_train, y_train = extract_extended_features_from_patients(
            train_p_obj.patientList, temporal_feats, static_encoder
        )
        X_test, y_test = extract_extended_features_from_patients(
            test_p.patientList, temporal_feats, static_encoder
        )

        print(f"  Train shape: {X_train.shape}")
        print(f"  Test shape: {X_test.shape}")

        # Train TabPFN
        tabpfn_params = {'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
        tabpfn = TabPFNClassifier(**tabpfn_params)
        tabpfn.fit(X_train, y_train)

        # Evaluate
        y_test_proba = tabpfn.predict_proba(X_test)[:, 1]
        prec, rec, _ = precision_recall_curve(y_test, y_test_proba)

        fold_auc = roc_auc_score(y_test, y_test_proba)
        fold_aupr = auc(rec, prec)

        metrics_xv5['auc'].append(fold_auc)
        metrics_xv5['auc_pr'].append(fold_aupr)

        fpr, tpr, _ = roc_curve(y_test, y_test_proba)
        ax1.plot(fpr, tpr, lw=2, label=f"Fold {fold} (AUC = {fold_auc:.3f})")

        print(f"  Tab_Xv5 AUC: {fold_auc:.4f} | AUPR: {fold_aupr:.4f}")

        # Baseline
        df_train_temp = train_p_obj.getMeasuresBetween(
            pd.Timedelta(hours=-6), pd.Timedelta(hours=24), "last", getUntilAkiPositive=True
        ).drop(columns=["subject_id", "hadm_id", "stay_id"])
        df_test_temp = test_p.getMeasuresBetween(
            pd.Timedelta(hours=-6), pd.Timedelta(hours=24), "last", getUntilAkiPositive=True
        ).drop(columns=["subject_id", "hadm_id", "stay_id"])

        df_train_enc, df_test_enc, _ = encodeCategoricalData(df_train_temp, df_test_temp)

        X_tr_b = df_train_enc.drop(columns=["akd"]).fillna(0)
        y_tr_b = df_train_enc["akd"]
        X_te_b = df_test_enc.drop(columns=["akd"]).fillna(0)
        y_te_b = df_test_enc["akd"]

        tabpfn_base = TabPFNClassifier(**tabpfn_params)
        tabpfn_base.fit(X_tr_b, y_tr_b)

        y_prob_b = tabpfn_base.predict_proba(X_te_b)[:, 1]
        prec_b, rec_b, _ = precision_recall_curve(y_te_b, y_prob_b)

        baseline_auc = roc_auc_score(y_te_b, y_prob_b)
        baseline_aupr = auc(rec_b, prec_b)

        metrics_baseline['auc'].append(baseline_auc)
        metrics_baseline['auc_pr'].append(baseline_aupr)

        fpr_b, tpr_b, _ = roc_curve(y_te_b, y_prob_b)
        ax2.plot(fpr_b, tpr_b, lw=2, label=f"Fold {fold} (AUC = {baseline_auc:.3f})")

        print(f"  Baseline AUC: {baseline_auc:.4f} | AUPR: {baseline_aupr:.4f}")

        improvement = ((fold_aupr - baseline_aupr) / baseline_aupr) * 100
        symbol = "✓" if fold_aupr > baseline_aupr else "✗"
        print(f"  Improvement: {improvement:+.2f}% {symbol}")

    # Plot
    for ax in [ax1, ax2]:
        ax.plot([0, 1], [0, 1], linestyle="--", color="navy", lw=2)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(loc="lower right")

    ax1.set_title("Tab_Xv5: Extended Handcrafted Features")
    ax2.set_title("Baseline (Last Only)")
    plt.tight_layout()
    plt.savefig("result/tab_xv5_vs_baseline.png", dpi=300)
    print("\nPlot saved to result/tab_xv5_vs_baseline.png")

    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)

    def print_stat(name, xv5_metrics, base_metrics):
        xv5_mean, xv5_std = np.mean(xv5_metrics), np.std(xv5_metrics)
        base_mean, base_std = np.mean(base_metrics), np.std(base_metrics)
        improvement = ((xv5_mean - base_mean) / base_mean) * 100
        symbol = "✓" if xv5_mean > base_mean else "✗"
        print(f"{name} | Tab_Xv5: {xv5_mean:.4f}±{xv5_std:.4f} vs Baseline: {base_mean:.4f}±{base_std:.4f} ({improvement:+.2f}%) {symbol}")

    print_stat("AUC   ", metrics_xv5['auc'], metrics_baseline['auc'])
    print_stat("AUC-PR", metrics_xv5['auc_pr'], metrics_baseline['auc_pr'])

    print("\n" + "="*80)
    print("ANALYSIS:")
    print("  If this WORKS: Extended handcrafted features are valuable!")
    print("                  → Add RL-optimized learned features on top (like Tab_RLv4)")
    print("  If this FAILS: Need RL from the start to optimize for TabPFN")
    print("="*80)

if __name__ == "__main__":
    main()
