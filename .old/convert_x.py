import json
import torch
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Union
import numpy as np
import os

def parse_timestamp(timestamp_str: str) -> datetime:
    """Parse timestamp string to datetime object."""
    return datetime.fromisoformat(timestamp_str.replace('T', ' '))


def get_time_offset_hours(start_time: datetime, current_time: datetime) -> float:
    """Get time offset in hours from start time."""
    delta = current_time - start_time
    return delta.total_seconds() / 3600.0


def analyze_json_structure(data_list: List[Dict]) -> Dict[str, Any]:
    """
    Analyze the JSON structure to determine profile configuration.

    Args:
        data_list: List of patient dictionaries

    Returns:
        Profile configuration dictionary
    """

    profile_config = {}
    profile_feature_indices = {}
    temporal_features = set()

    # Analyze all samples to get complete feature set
    all_profile_features = set()
    all_temporal_features = set()

    # First pass: separate profile vs temporal features
    for sample in data_list:
        measures = sample['measures']

        for key, value in measures.items():
            if isinstance(value, dict) and value:
                # This is temporal data - check if it has timestamp-like keys
                first_key = list(value.keys())[0]
                if isinstance(first_key, str) and ('T' in first_key or ':' in first_key):
                    all_temporal_features.add(key)
                else:
                    # Dictionary but not timestamp format - treat as profile
                    all_profile_features.add(key)
            elif not isinstance(value, dict):
                # This is profile data (scalar value)
                all_profile_features.add(key)

    print(f"Found profile features: {sorted(all_profile_features)}")
    print(f"Found temporal features: {sorted(all_temporal_features)}")

    # Determine profile feature types and ranges
    profile_idx = 0

    for feature in sorted(all_profile_features):
        values = []
        for sample in data_list:
            if feature in sample['measures']:
                value = sample['measures'][feature]
                # Only add if it's not a dictionary (temporal data)
                if not isinstance(value, dict):
                    values.append(value)

        if not values:
            continue

        # Special handling for known continuous features that might be integers
        known_continuous = {
            'age', 'egfr', 'oasis', 'saps2', 'sofa', 'preiculos', 'weight',
            'hematocrit', 'mch', 'mchc', 'mcv', 'rbc', 'rdw', 'chloride',
            'potassium', 'sodium'
        }

        # Special handling for known categorical features
        known_categorical = {
            'gender', 'race', 'liver_disease', 'chronic_pulmonary_disease',
            'congestive_heart_failure', 'hypertension', 'malignant_cancer',
            'microangiopathy', 'ckd_stage', 'dka_type', 'history_aci',
            'history_ami', 'macroangiopathy', 'mechanical_ventilation',
            'use_NaHCO3', 'uti'
        }

        # Determine if categorical or continuous
        if feature in known_continuous:
            is_categorical = False
        elif feature in known_categorical:
            is_categorical = True
        else:
            # For unknown features, use heuristics
            # Check if all values are strings or explicit booleans
            is_categorical = all(
                isinstance(v, (bool, str)) or
                (isinstance(v, (int, float)) and v in [0, 1] and len(set(values)) <= 10)
                for v in values if v is not None
            )

        if is_categorical:
            # Handle different categorical types
            if feature == 'gender':
                categories = ['M', 'F', 'OTHER']
            elif feature in ['chronic_pulmonary_disease', 'congestive_heart_failure',
                             'hypertension', 'malignant_cancer', 'microangiopathy']:
                categories = [False, True]  # Boolean features
            elif feature == 'race':
                unique_vals = list(set(str(v) for v in values if v is not None and not isinstance(v, dict)))
                categories = sorted(unique_vals)
            elif feature == 'liver_disease':
                categories = ['NONE', 'MILD', 'MODERATE', 'SEVERE']
            elif feature in ['ckd_stage', 'dka_type', 'history_aci', 'history_ami',
                             'macroangiopathy', 'mechanical_ventilation', 'use_NaHCO3', 'uti']:
                unique_vals = sorted(set(v for v in values if v is not None and not isinstance(v, dict)))
                categories = [int(v) if isinstance(v, (int, float)) else v for v in unique_vals]
            else:
                # Generic categorical - filter out dict values
                unique_vals = sorted(set(v for v in values if v is not None and not isinstance(v, dict)))
                categories = unique_vals

            profile_config[feature] = {
                'type': 'categorical',
                'categories': categories
            }
        else:
            # Continuous feature
            numeric_values = [v for v in values if
                              v is not None and isinstance(v, (int, float)) and not isinstance(v, dict)]
            if numeric_values:
                min_val = float(min(numeric_values))
                max_val = float(max(numeric_values))

                # Add some padding to avoid edge cases
                range_padding = (max_val - min_val) * 0.1 if max_val > min_val else 1.0
                min_val = max(0, min_val - range_padding)  # Don't go below 0 for medical values
                max_val = max_val + range_padding

                profile_config[feature] = {
                    'type': 'continuous',
                    'min_val': min_val,
                    'max_val': max_val
                }

        profile_feature_indices[feature] = profile_idx
        profile_idx += 1

    return profile_config, profile_feature_indices, sorted(all_temporal_features)


def convert_json_to_transformer_format(
        sample: Dict[str, Any],
        profile_config: Dict[str, Any],
        profile_feature_indices: Dict[str, int],
        temporal_features: List[str]
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], bool]:
    """
    Convert a single JSON sample to transformer format.

    Args:
        sample: Single patient dictionary
        profile_config: Profile feature configuration
        profile_feature_indices: Mapping from feature names to indices
        temporal_features: List of temporal feature names

    Returns:
        Tuple of (profile_tensor, time_series_dict, label)
    """

    measures = sample['measures']

    # Parse timing information
    intime = parse_timestamp(sample['intime'])
    akd_time_seconds = sample['akdTime']
    end_time = intime + timedelta(seconds=akd_time_seconds)

    # Extract label
    label = sample['akdPositive']

    # Create profile tensor
    profile_size = len(profile_feature_indices)
    profile_tensor = torch.zeros(profile_size, dtype=torch.float32)

    for feature_name, config in profile_config.items():
        if feature_name in measures and feature_name in profile_feature_indices:
            idx = profile_feature_indices[feature_name]
            value = measures[feature_name]

            if value is not None:
                if config['type'] == 'categorical':
                    # Find category index
                    try:
                        if isinstance(value, bool):
                            cat_idx = int(value)
                        elif isinstance(value, str):
                            cat_idx = config['categories'].index(value)
                        else:
                            cat_idx = config['categories'].index(value)
                        profile_tensor[idx] = float(cat_idx)
                    except (ValueError, IndexError):
                        # Handle unknown categories
                        profile_tensor[idx] = 0.0
                else:
                    # Continuous value
                    profile_tensor[idx] = float(value)

    # Extract temporal data
    all_values = []
    all_feature_ids = []
    all_timestamps = []

    # Create feature ID mapping
    feature_id_map = {name: i for i, name in enumerate(temporal_features)}

    for feature_name in temporal_features:
        if feature_name in measures and isinstance(measures[feature_name], dict):
            time_series = measures[feature_name]

            for timestamp_str, value in time_series.items():
                timestamp = parse_timestamp(timestamp_str)

                # Only include data up to intime + akdTime
                if timestamp <= end_time:
                    time_offset_hours = get_time_offset_hours(intime, timestamp)

                    if time_offset_hours >= 0:  # Only include data after admission
                        all_values.append(float(value))
                        all_feature_ids.append(feature_id_map[feature_name])
                        all_timestamps.append(time_offset_hours)

    # Sort by timestamp
    if all_values:
        sorted_indices = np.argsort(all_timestamps)
        all_values = [all_values[i] for i in sorted_indices]
        all_feature_ids = [all_feature_ids[i] for i in sorted_indices]
        all_timestamps = [all_timestamps[i] for i in sorted_indices]
    else:
        # Handle edge case of no temporal data
        all_values = [0.0]
        all_feature_ids = [0]
        all_timestamps = [0.0]

    # Create time series dictionary
    time_series_dict = {
        'values': torch.tensor(all_values, dtype=torch.float32),
        'feature_ids': torch.tensor(all_feature_ids, dtype=torch.long),
        'timestamps': torch.tensor(all_timestamps, dtype=torch.float32)
    }

    return profile_tensor, time_series_dict, label


def load_and_convert_dataset(json_file_path: str) -> Dict[str, Any]:
    """
    Load JSON dataset and convert to transformer format.

    Args:
        json_file_path: Path to JSON file containing list of JSON instances

    Returns:
        Dictionary containing converted dataset
    """

    # Load JSON data from file
    print(f"Loading data from: {json_file_path}")

    try:
        with open(json_file_path, 'r') as f:
            # Try to load as a single JSON array first
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                # If that fails, try reading line by line (JSONL format)
                f.seek(0)  # Reset file pointer
                data = []
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:  # Skip empty lines
                        try:
                            data.append(json.loads(line))
                        except json.JSONDecodeError as e:
                            print(f"Warning: Skipping malformed JSON on line {line_num}: {e}")
                            continue

    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {json_file_path}")
    except Exception as e:
        raise Exception(f"Error reading file {json_file_path}: {e}")

    # Ensure we have a list
    if not isinstance(data, list):
        if isinstance(data, dict):
            data = [data]  # Convert single object to list
        else:
            raise ValueError("JSON file should contain a list of objects or a single object")

    print(f"Loaded {len(data)} samples")

    # Filter out invalid samples
    valid_data = []
    for i, sample in enumerate(data):
        if not isinstance(sample, dict):
            print(f"Warning: Skipping non-dictionary sample at index {i}")
            continue

        # Check required fields
        required_fields = ['subject_id', 'intime', 'akdPositive', 'akdTime', 'measures']
        missing_fields = [field for field in required_fields if field not in sample]

        if missing_fields:
            print(f"Warning: Skipping sample {i} due to missing fields: {missing_fields}")
            continue

        # Check if measures is a dictionary
        if not isinstance(sample['measures'], dict):
            print(f"Warning: Skipping sample {i} - 'measures' is not a dictionary")
            continue

        valid_data.append(sample)

    print(f"Valid samples after filtering: {len(valid_data)}")

    if len(valid_data) == 0:
        raise ValueError("No valid samples found in the dataset")

    # Analyze structure and create configuration
    profile_config, profile_feature_indices, temporal_features = analyze_json_structure(valid_data)

    print(f"\nProfile configuration:")
    for feature, config in profile_config.items():
        if config['type'] == 'categorical':
            print(f"  {feature}: categorical, {len(config['categories'])} categories")
        else:
            print(f"  {feature}: continuous, range [{config['min_val']:.2f}, {config['max_val']:.2f}]")

    print(f"\nTemporal features: {temporal_features}")

    # Convert all samples
    profiles_list = []
    ts_data_list = []
    labels_list = []

    for i, sample in enumerate(valid_data):
        try:
            profile, ts_dict, label = convert_json_to_transformer_format(
                sample, profile_config, profile_feature_indices, temporal_features
            )

            profiles_list.append(profile)
            ts_data_list.append(ts_dict)
            labels_list.append(label)

        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            continue

    if len(profiles_list) == 0:
        raise ValueError("No samples could be successfully converted")

    # Stack profiles and convert labels
    profiles_tensor = torch.stack(profiles_list)
    labels_tensor = torch.tensor(labels_list, dtype=torch.long)

    print(f"\nDataset summary:")
    print(f"Total converted samples: {len(profiles_list)}")
    print(f"Profile tensor shape: {profiles_tensor.shape}")
    print(f"Label distribution: {torch.bincount(labels_tensor)}")

    # Split into train/validation

    # Create indices for splitting
    #indices = torch.randperm(n_samples)
    #train_indices = indices[:n_train]
    #val_indices = indices[n_train:]


    return {
        'train_profiles': profiles_tensor,
        'train_ts_data': ts_data_list,
        'y_train': labels_tensor,
        'profile_config': profile_config,
        'profile_feature_indices': profile_feature_indices,
        'temporal_features': temporal_features,
        'device': 'cpu'
    }


def demonstrate_conversion_from_file():


    # Convert the data
    converted_data = load_and_convert_dataset("train_fold_0_part_0.json")

    print(f"\nConversion Results:")
    print(f"Training samples: {len(converted_data['train_ts_data'])}")
    #print(f"Validation samples: {len(converted_data['val_ts_data'])}")
    print(f"Profile tensor shape: {converted_data['train_profiles'].shape}")

    # Show profile configuration
    print(f"\nProfile Configuration:")
    for feature, config in converted_data['profile_config'].items():
        if config['type'] == 'categorical':
            print(f"  {feature}: categorical - {config['categories']}")
        else:
            print(f"  {feature}: continuous - [{config['min_val']:.2f}, {config['max_val']:.2f}]")

    # Show first sample details
    if len(converted_data['train_ts_data']) > 0:
        first_sample = converted_data['train_ts_data'][0]
        first_profile = converted_data['train_profiles'][0]

        print(f"\nFirst sample details:")
        print(f"Profile values: {first_profile}")
        print(f"Time series values: {first_sample['values']}")
        print(f"Feature IDs: {first_sample['feature_ids']}")
        print(f"Timestamps (hours): {first_sample['timestamps']}")
        print(f"Number of observations: {len(first_sample['values'])}")

    return converted_data



def create_training_pipeline_from_file( device: str = 'cpu'):
    """
    Create a complete training pipeline from JSON file.

    Args:
        json_file_path: Path to JSON file containing list of patient records
        device: Device to use ('cpu' or 'cuda')

    Returns:
        Trained model and results
    """

    # print(f"Loading dataset from file: {json_file_path}")

    # Convert data
    data_train = load_and_convert_dataset("train_fold_0_part_0.json")
    data_train['device'] = device

    data_val = load_and_convert_dataset("val_fold_0_part_0.json")
    # Move data to device
    data_train['train_profiles'] = data_train['train_profiles'].to(device)
    data_train['y_train'] = data_train['y_train'].to(device)
    data_val['val_profiles'] = data_val['val_profiles'].to(device)
    data_val['y_val'] = data_val['y_val'].to(device)

    # Import required classes (assuming they're available)
    try:
        from transformer_tabpfn import IrregularTimeSeriesWithProfilesTransformer, DifferentiableTabPFN, \
            IrregularTimeSeriesWithProfilesTrainer
    except ImportError:
        print("Warning: Could not import transformer classes. Make sure they are available in your environment.")
        return None, None, None

    # Create models with appropriate sizing
    num_temporal_features = len(data_train['temporal_features'])
    num_profile_features = len(data_train['profile_config'])

    transformer = IrregularTimeSeriesWithProfilesTransformer(
        num_ts_features=num_temporal_features,
        profile_config=data_train['profile_config'],
        ts_d_model=min(128, max(32, num_temporal_features * 8)),  # Scale with features
        nhead=8,
        num_layers=4,
        ts_output_dim=64,
        profile_output_dim=32,
        final_output_dim=64,
        max_sequence_length=500,  # Increase for medical data
        max_time=168.0,  # 7 days max
        dropout=0.15,
        pooling='attention'
    )

    tabpfn = DifferentiableTabPFN(
        N_ensemble_configurations=12,
        device=device,
        temperature=0.9
    )

    # Create trainer
    trainer = IrregularTimeSeriesWithProfilesTrainer(transformer, tabpfn, device)

    print(f"\nModel Configuration:")
    print(f"Temporal features: {num_temporal_features}")
    print(f"Profile features: {num_profile_features}")
    print(f"Model parameters: {sum(p.numel() for p in transformer.parameters()):,}")
    print(f"Max sequence length: 500")
    print(f"Max time window: 168 hours (7 days)")

    # Determine batch size based on dataset size
    n_samples = len(data_train['train_ts_data'])
    batch_size = min(16, max(4, n_samples // 10))

    print(f"\nTraining Configuration:")
    print(f"Batch size: {batch_size}")
    print(f"Training samples: {len(data_train['train_ts_data'])}")
    print(f"Validation samples: {len(data_val['val_ts_data'])}")

    # Train
    history = trainer.train(
        train_ts_data=data_train['train_ts_data'],
        train_profiles=data_train['train_profiles'],
        profile_feature_indices=data_train['profile_feature_indices'],
        y_train=data_train['y_train'],
        val_ts_data=data_val['val_ts_data'],
        val_profiles=data_val['val_profiles'],
        y_val=data_val['y_val'],
        epochs=50,
        batch_size=batch_size,
        lr=0.001,
        verbose=True
    )

    return trainer, history, data_train


def demonstrate_conversion():


    print("=== Demonstrating JSON Conversion ===\n")

    # Convert the data
    converted_data = load_and_convert_dataset("train_fold_0_part_0.json")

    print(f"\nConversion Results:")
    print(f"Training samples: {len(converted_data['train_ts_data'])}")
    print(f"Validation samples: {len(converted_data['val_ts_data'])}")
    print(f"Profile tensor shape: {converted_data['train_profiles'].shape}")
    print(f"Labels: {converted_data['y_train']}")

    # Show first sample details
    if len(converted_data['train_ts_data']) > 0:
        first_sample = converted_data['train_ts_data'][0]
        first_profile = converted_data['train_profiles'][0]

        print(f"\nFirst sample details:")
        print(f"Profile values: {first_profile}")
        print(f"Time series values: {first_sample['values']}")
        print(f"Feature IDs: {first_sample['feature_ids']}")
        print(f"Timestamps (hours): {first_sample['timestamps']}")
        print(f"Number of observations: {len(first_sample['values'])}")

    return converted_data





if __name__ == "__main__":
    print("=== JSON File to Transformer Format Converter ===\n")

    # Demonstrate conversion from file
    # converted_data = demonstrate_conversion_from_file()
    create_training_pipeline_from_file(device="cuda")
