"""
Explore the data format to understand how timestamps and measures are stored.
"""

import pandas as pd
import sys

sys.path.append("/Users/anhnd/CodingSpace/Python/PREDKIT")
from constants import NULLABLE_MEASURES
from utils.class_patient import Patients


def explore_patient_data():
    """Explore the structure of patient data."""

    # Load patients
    print("Loading patients...")
    patients = Patients.loadPatients()
    print(f"Loaded {len(patients)} patients\n")

    # Look at the first patient
    patient = patients.patientList[0]

    print("="*80)
    print("PATIENT OBJECT STRUCTURE")
    print("="*80)
    print(f"Patient type: {type(patient)}")
    print(f"Patient attributes: {dir(patient)}\n")

    # Check measures
    print("="*80)
    print("MEASURES STRUCTURE")
    print("="*80)
    print(f"Type of measures: {type(patient.measures)}")
    print(f"Number of measures: {len(patient.measures)}")
    print(f"Measure keys (first 10): {list(patient.measures.keys())[:10]}\n")

    # Examine a few different types of measures
    sample_measures = list(patient.measures.keys())[:5]

    for measure_name in sample_measures:
        measure_value = patient.measures[measure_name]
        print(f"\n--- Measure: {measure_name} ---")
        print(f"Type: {type(measure_value)}")

        if isinstance(measure_value, list):
            print(f"Length: {len(measure_value)}")
            if len(measure_value) > 0:
                first_item = measure_value[0]
                print(f"First item type: {type(first_item)}")
                print(f"First item attributes: {dir(first_item)}")
                print(f"First item value: {first_item}")

                # Try to get time
                if hasattr(first_item, 'time'):
                    time_attr = first_item.time
                    print(f"Time attribute type: {type(time_attr)}")
                    if callable(time_attr):
                        print(f"Time value: {time_attr()}")
                    else:
                        print(f"Time value: {time_attr}")

                if hasattr(first_item, 'value'):
                    value_attr = first_item.value
                    print(f"Value attribute type: {type(value_attr)}")
                    if callable(value_attr):
                        print(f"Value: {value_attr()}")
                    else:
                        print(f"Value: {value_attr}")

        elif isinstance(measure_value, (int, float)):
            print(f"Value: {measure_value}")
        else:
            print(f"Value: {measure_value}")
            print(f"Attributes: {dir(measure_value)}")

    # Check if there's a method to get temporal data
    print("\n" + "="*80)
    print("METHODS TO GET TEMPORAL DATA")
    print("="*80)

    if hasattr(patient, 'getMeasuresBetween'):
        print("patient.getMeasuresBetween() exists")

        # Try to get measures
        try:
            df = patient.getMeasuresBetween(
                pd.Timedelta(hours=-6),
                pd.Timedelta(hours=24),
                "last"
            )
            print(f"\ngetMeasuresBetween() returns DataFrame:")
            print(f"Shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
            print(f"\nFirst row:")
            print(df.iloc[0])
        except Exception as e:
            print(f"Error calling getMeasuresBetween(): {e}")

    # Check the raw measure data structure
    print("\n" + "="*80)
    print("DETAILED EXAMINATION OF TEMPORAL MEASURES")
    print("="*80)

    # Look for measures that should have timestamps
    temporal_measure_candidates = ['hr', 'sbp', 'dbp', 'sodium', 'potassium', 'wbc']

    for measure_name in temporal_measure_candidates:
        if measure_name in patient.measures:
            measure_value = patient.measures[measure_name]
            print(f"\n--- {measure_name} ---")
            print(f"Type: {type(measure_value)}")

            if isinstance(measure_value, list) and len(measure_value) > 0:
                print(f"Number of measurements: {len(measure_value)}")

                # Check first few items
                for i, item in enumerate(measure_value[:3]):
                    print(f"\n  Measurement {i}:")
                    print(f"    Type: {type(item)}")
                    print(f"    String representation: {item}")

                    # Try to access time
                    if hasattr(item, 'time'):
                        try:
                            time_val = item.time() if callable(item.time) else item.time
                            print(f"    Time: {time_val} (type: {type(time_val)})")
                        except Exception as e:
                            print(f"    Error getting time: {e}")

                    # Try to access value
                    if hasattr(item, 'value'):
                        try:
                            val = item.value() if callable(item.value) else item.value
                            print(f"    Value: {val}")
                        except Exception as e:
                            print(f"    Error getting value: {e}")

                    # Check all attributes
                    attrs = [attr for attr in dir(item) if not attr.startswith('_')]
                    print(f"    Available attributes: {attrs}")

    # Check if there's charttime or other timestamp info
    print("\n" + "="*80)
    print("CHECKING FOR TIMESTAMP INFORMATION")
    print("="*80)

    # Look at patient attributes that might have time info
    time_related_attrs = [attr for attr in dir(patient) if 'time' in attr.lower() or 'date' in attr.lower() or 'chart' in attr.lower()]
    print(f"Time-related patient attributes: {time_related_attrs}")

    for attr in time_related_attrs:
        try:
            val = getattr(patient, attr)
            if not callable(val):
                print(f"{attr}: {val} (type: {type(val)})")
        except:
            pass


if __name__ == "__main__":
    explore_patient_data()
