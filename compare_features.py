"""
Compare features used by MLP_OnlyTime.py vs ODETime.py
"""

import pandas as pd
import sys

sys.path.append("/Users/anhnd/CodingSpace/Python/PREDKIT")
from constants import NULLABLE_MEASURES
from utils.class_patient import Patients
from utils.prepare_data import encodeCategoricalData, trainTestPatients

# Same fixed features from both scripts
FIXED_FEATURES = [
    "age", "gender", "race",
    "chronic_pulmonary_disease", "ckd_stage", "congestive_heart_failure",
    "dka_type", "history_aci", "history_ami", "hypertension",
    "liver_disease", "macroangiopathy", "malignant_cancer",
    "microangiopathy", "uti",
    "oasis", "saps2", "sofa",
    "mechanical_ventilation", "use_NaHCO3",
    "preiculos", "gcs_unable",
    "egfr",
]

ID_COLUMNS = ["subject_id", "hadm_id", "stay_id"]
LABEL_COLUMN = "akd"

# Load patients
print("Loading patients...")
patients = Patients.loadPatients()
patients.fillMissingMeasureValue(NULLABLE_MEASURES, 0)

# Remove sparse features
measures = patients.getMeasures()
for measure, count in measures.items():
    if count < len(patients) * 80 / 100:
        patients.removeMeasures([measure])

patients.removePatientByMissingFeatures()
print(f"Total patients: {len(patients)}\n")

# Get one fold
train_patients, test_patients = next(trainTestPatients(patients))

# ============================================================================
# MLP METHOD: Using getMeasuresBetween
# ============================================================================
print("="*80)
print("MLP_OnlyTime.py METHOD")
print("="*80)

df_train_mlp = train_patients.getMeasuresBetween(
    pd.Timedelta(hours=-6),
    pd.Timedelta(hours=24),
    "last",
    getUntilAkiPositive=True
)
df_train_mlp = df_train_mlp.drop(columns=ID_COLUMNS)

# Filter temporal features
columns_to_drop = [col for col in FIXED_FEATURES if col in df_train_mlp.columns]
df_train_mlp = df_train_mlp.drop(columns=columns_to_drop, errors='ignore')

# Before encoding
features_before_encoding = [col for col in df_train_mlp.columns if col != LABEL_COLUMN]
print(f"\nFeatures BEFORE encoding: {len(features_before_encoding)}")
print(sorted(features_before_encoding))

# After encoding
df_train_mlp_encoded, _, _ = encodeCategoricalData(df_train_mlp, df_train_mlp.copy())
X_train_mlp = df_train_mlp_encoded.drop(columns=[LABEL_COLUMN])

print(f"\nFeatures AFTER encoding: {len(X_train_mlp.columns)}")
print(sorted(X_train_mlp.columns.tolist()))

print(f"\nData shape: {X_train_mlp.shape}")
print(f"Sample patient data (first 5 features):")
print(X_train_mlp.iloc[0, :5])

# ============================================================================
# ODE METHOD: Using raw SortedDict extraction
# ============================================================================
print("\n" + "="*80)
print("ODETime.py METHOD")
print("="*80)

# Get all temporal features
all_features = set()
for patient in train_patients.patientList:
    for measure_name, measure_values in patient.measures.items():
        if measure_name in FIXED_FEATURES:
            continue
        if hasattr(measure_values, 'keys') and hasattr(measure_values, 'values'):
            all_features.add(measure_name)

ode_features = sorted(all_features)
print(f"\nTemporal features: {len(ode_features)}")
print(ode_features)

# Check a sample patient
patient = train_patients.patientList[0]
print(f"\nSample patient temporal data:")
for feat in ode_features[:5]:
    if feat in patient.measures:
        measure_dict = patient.measures[feat]
        if hasattr(measure_dict, 'keys'):
            print(f"  {feat}: {len(measure_dict)} measurements")
            if len(measure_dict) > 0:
                timestamps = list(measure_dict.keys())[:3]
                print(f"    First 3 timestamps: {timestamps}")

# ============================================================================
# COMPARISON
# ============================================================================
print("\n" + "="*80)
print("COMPARISON")
print("="*80)

mlp_set = set(features_before_encoding)
ode_set = set(ode_features)

print(f"\nMLP features (before encoding): {len(mlp_set)}")
print(f"ODE features: {len(ode_set)}")

in_mlp_not_ode = mlp_set - ode_set
in_ode_not_mlp = ode_set - mlp_set

if in_mlp_not_ode:
    print(f"\nIn MLP but NOT in ODE ({len(in_mlp_not_ode)}): {sorted(in_mlp_not_ode)}")
else:
    print("\n✓ All MLP features are in ODE")

if in_ode_not_mlp:
    print(f"\nIn ODE but NOT in MLP ({len(in_ode_not_mlp)}): {sorted(in_ode_not_mlp)}")
else:
    print("\n✓ All ODE features are in MLP")

if mlp_set == ode_set:
    print("\n✓✓✓ BOTH USE THE SAME FEATURES ✓✓✓")
else:
    print("\n✗✗✗ DIFFERENT FEATURES USED ✗✗✗")

# ============================================================================
# CHECK getUntilAkiPositive ISSUE
# ============================================================================
print("\n" + "="*80)
print("CRITICAL: getUntilAkiPositive CHECK")
print("="*80)

# Check a patient with AKI
aki_patient = None
for p in train_patients.patientList[:20]:
    if p.akdPositive:
        aki_patient = p
        break

if aki_patient:
    print(f"\nFound AKI patient:")
    print(f"  AKI Time: {aki_patient.akdTime}")
    print(f"  Intime: {aki_patient.intime}")

    # Check if raw extraction respects AKI time
    if 'hr' in aki_patient.measures and hasattr(aki_patient.measures['hr'], 'keys'):
        hr_dict = aki_patient.measures['hr']
        timestamps = sorted(hr_dict.keys())

        aki_absolute_time = aki_patient.intime + aki_patient.akdTime

        timestamps_after_aki = [t for t in timestamps if pd.Timestamp(t) > aki_absolute_time]

        print(f"\n  HR measurements: {len(timestamps)}")
        print(f"  HR measurements AFTER AKI: {len(timestamps_after_aki)}")

        if timestamps_after_aki:
            print(f"\n  ✗✗✗ ODETime.py IS USING DATA AFTER AKI DIAGNOSIS! ✗✗✗")
            print(f"  This is DATA LEAKAGE!")
            print(f"  Example timestamps after AKI:")
            for ts in timestamps_after_aki[:3]:
                print(f"    {ts} (value: {hr_dict[ts]})")
        else:
            print(f"\n  ✓ No data after AKI (might be outside time window)")
