import json
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any
import argparse
import os
import sys


class KibanaFileProcessor:
    """Process JSON files with medical data for Kibana import"""

    def __init__(self):
        self.time_series_fields = [
            'ag', 'bg', 'bicarbonate', 'bun', 'calcium', 'dbp', 'gcs', 'gcs_unable',
            'hb', 'hr', 'phosphate', 'plt', 'rr', 'sbp', 'scr', 'wbc', 'weight'
        ]

        self.static_fields = [
            'age', 'chloride', 'chronic_pulmonary_disease', 'ckd_stage',
            'congestive_heart_failure', 'dka_type', 'egfr', 'gender', 'hematocrit',
            'history_aci', 'history_ami', 'hypertension', 'liver_disease',
            'macroangiopathy', 'malignant_cancer', 'mch', 'mchc', 'mcv',
            'mechanical_ventilation', 'microangiopathy', 'oasis', 'potassium',
            'preiculos', 'race', 'rbc', 'rdw', 'saps2', 'sodium', 'sofa',
            'use_NaHCO3', 'uti'
        ]

        self.vital_signs = ['hr', 'sbp', 'dbp', 'rr', 'gcs', 'gcs_unable']
        self.lab_results = ['ag', 'bg', 'bicarbonate', 'bun', 'calcium', 'hb',
                            'phosphate', 'plt', 'scr', 'wbc']

    def process_file(self, input_file: str, output_file: str, index_name: str = "medical_data", max_records: int = 100):
        """
        Process input JSON file and create Kibana-ready NDJSON output

        Args:
            input_file: Path to input JSON file containing list of medical records
            output_file: Path to output NDJSON file for Elasticsearch bulk import
            index_name: Elasticsearch index name
            max_records: Maximum number of records to process (default: 100)
        """
        print(f"Processing file: {input_file}")

        # Read input JSON file
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"Error: File {input_file} not found")
            return False
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in {input_file}: {e}")
            return False

        # Ensure data is a list
        if not isinstance(data, list):
            print("Error: Input JSON must contain a list of objects")
            return False

        # Limit to max_records
        data = data[:max_records]
        print(f"Found {len(data)} records to process (limited to first {max_records})")

        # Transform all records
        all_documents = []
        processed_count = 0
        error_count = 0

        for i, record in enumerate(data):
            try:
                # Transform each record to time-series format
                time_series_docs = self.transform_to_time_series(record)
                all_documents.extend(time_series_docs)
                processed_count += 1

                if (i + 1) % 10 == 0:
                    print(f"Processed {i + 1} records...")

            except Exception as e:
                print(f"Error processing record {i}: {e}")
                error_count += 1
                continue

        print(f"Transformation complete:")
        print(f"  - Records processed: {processed_count}")
        print(f"  - Records with errors: {error_count}")
        print(f"  - Total documents created: {len(all_documents)}")

        # Write NDJSON output for Elasticsearch bulk import
        self.save_for_elasticsearch(all_documents, output_file, index_name)

        # Generate mapping file
        mapping_file = output_file.replace('.ndjson', '_mapping.json')
        self.save_index_mapping(mapping_file, index_name)

        print(f"Files created:")
        print(f"  - Data file: {output_file}")
        print(f"  - Mapping file: {mapping_file}")

        return True

    def transform_to_time_series(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Transform single record to time-series format"""
        results = []

        # Extract base document with static fields
        base_doc = self._extract_base_document(data)

        # Process time-series measurements
        measures = data.get('measures', {})

        for field_name in self.time_series_fields:
            field_data = measures.get(field_name)

            if field_data and isinstance(field_data, dict):
                # Handle time-series data (dict with timestamp keys)
                for timestamp, value in field_data.items():
                    doc = {
                        **base_doc,
                        'timestamp': timestamp,
                        '@timestamp': self._parse_timestamp(timestamp),
                        'measurement_type': field_name,
                        'measurement_category': self._get_measurement_category(field_name),
                        'value': value,
                        'numeric_value': value if isinstance(value, (int, float)) else None,
                        'unit': self._get_measurement_unit(field_name)
                    }
                    results.append(doc)

            elif field_data is not None:
                # Handle single-value measurements
                doc = {
                    **base_doc,
                    'timestamp': data.get('intime'),
                    '@timestamp': self._parse_timestamp(data.get('intime')),
                    'measurement_type': field_name,
                    'measurement_category': self._get_measurement_category(field_name),
                    'value': field_data,
                    'numeric_value': field_data if isinstance(field_data, (int, float)) else None,
                    'unit': self._get_measurement_unit(field_name)
                }
                results.append(doc)

        return results

    def _extract_base_document(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract static fields for base document"""
        base_doc = {}
        measures = data.get('measures', {})

        # Add top-level fields
        for field in ['subject_id', 'hadm_id', 'stay_id', 'intime', 'akdPositive', 'akdTime']:
            if field in data:
                base_doc[field] = data[field]

        # Add static measure fields
        for field in self.static_fields:
            if field in measures:
                base_doc[field] = measures[field]

        # Add admission timestamp
        if 'intime' in data:
            base_doc['admission_timestamp'] = self._parse_timestamp(data['intime'])

        return base_doc

    def _parse_timestamp(self, timestamp: str) -> str:
        """Parse and format timestamp to standard ISO format for Elasticsearch"""
        if not timestamp:
            return None

        try:
            # Handle different timestamp formats
            if isinstance(timestamp, str):
                # Handle timestamps that start with year 2140 (convert to reasonable year)
                if timestamp.startswith('2140'):
                    # Convert 2140 to 2024 for realistic dates
                    timestamp = timestamp.replace('2140', '2024', 1)

                # Remove any timezone info and parse
                timestamp_clean = timestamp.replace('Z', '').replace('+00:00', '')
                dt = datetime.fromisoformat(timestamp_clean)

                # Return in standard ISO format with timezone
                return dt.strftime('%Y-%m-%dT%H:%M:%S.000Z')
            return timestamp
        except (ValueError, AttributeError) as e:
            print(f"Warning: Could not parse timestamp {timestamp}: {e}")
            return timestamp

    def _get_measurement_category(self, field_name: str) -> str:
        """Categorize measurement types"""
        if field_name in self.vital_signs:
            return 'vital_signs'
        elif field_name in self.lab_results:
            return 'lab_results'
        else:
            return 'other'

    def _get_measurement_unit(self, field_name: str) -> str:
        """Get measurement units for better visualization"""
        units = {
            'hr': 'bpm', 'sbp': 'mmHg', 'dbp': 'mmHg', 'rr': 'breaths/min',
            'ag': 'mEq/L', 'bg': 'mg/dL', 'bicarbonate': 'mEq/L', 'bun': 'mg/dL',
            'calcium': 'mg/dL', 'hb': 'g/dL', 'phosphate': 'mg/dL', 'plt': 'K/uL',
            'scr': 'mg/dL', 'wbc': 'K/uL', 'weight': 'kg', 'gcs': 'points'
        }
        return units.get(field_name, '')

    def save_for_elasticsearch(self, data: List[Dict], output_file: str, index_name: str):
        """Save data in NDJSON format for Elasticsearch bulk indexing"""
        with open(output_file, 'w', encoding='utf-8') as f:
            for doc in data:
                # Create index action
                index_action = {"index": {"_index": index_name}}
                f.write(json.dumps(index_action) + '\n')
                f.write(json.dumps(doc, default=str) + '\n')

    def save_index_mapping(self, mapping_file: str, index_name: str):
        """Generate and save Elasticsearch index mapping"""
        mapping = {
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "index": {
                    "mapping": {
                        "total_fields": {
                            "limit": 2000
                        }
                    }
                }
            },
            "mappings": {
                "properties": {
                    "@timestamp": {"type": "date"},
                    "timestamp": {"type": "date"},
                    "admission_timestamp": {"type": "date"},
                    "subject_id": {"type": "long"},
                    "hadm_id": {"type": "long"},
                    "stay_id": {"type": "long"},
                    "measurement_type": {"type": "keyword"},
                    "measurement_category": {"type": "keyword"},
                    "value": {"type": "double"},
                    "numeric_value": {"type": "double"},
                    "unit": {"type": "keyword"},
                    "age": {"type": "integer"},
                    "gender": {"type": "keyword"},
                    "race": {"type": "keyword"},
                    "akdPositive": {"type": "boolean"},
                    "akdTime": {"type": "double"},
                    "hypertension": {"type": "boolean"},
                    "congestive_heart_failure": {"type": "boolean"},
                    "chronic_pulmonary_disease": {"type": "boolean"},
                    "malignant_cancer": {"type": "boolean"},
                    "liver_disease": {"type": "keyword"},
                    "ckd_stage": {"type": "integer"},
                    "dka_type": {"type": "integer"},
                    "egfr": {"type": "double"},
                    "oasis": {"type": "double"},
                    "saps2": {"type": "integer"},
                    "sofa": {"type": "integer"},
                    "chloride": {"type": "double"},
                    "hematocrit": {"type": "double"},
                    "history_aci": {"type": "integer"},
                    "history_ami": {"type": "integer"},
                    "macroangiopathy": {"type": "integer"},
                    "mch": {"type": "double"},
                    "mchc": {"type": "double"},
                    "mcv": {"type": "double"},
                    "mechanical_ventilation": {"type": "integer"},
                    "microangiopathy": {"type": "integer"},
                    "potassium": {"type": "double"},
                    "preiculos": {"type": "double"},
                    "rbc": {"type": "double"},
                    "rdw": {"type": "double"},
                    "sodium": {"type": "double"},
                    "use_NaHCO3": {"type": "integer"},
                    "uti": {"type": "integer"}
                }
            }
        }

        with open(mapping_file, 'w', encoding='utf-8') as f:
            json.dump(mapping, f, indent=2)

    def create_import_script(self, output_file: str, index_name: str):
        """Create a shell script for easy Elasticsearch import"""
        script_file = output_file.replace('.ndjson', '_import.sh')
        mapping_file = output_file.replace('.ndjson', '_mapping.json')

        script_content = f"""#!/bin/bash

# Elasticsearch import script for {index_name}
# Make sure Elasticsearch is running on localhost:9200

echo "Creating index with mapping..."
curl -X PUT "localhost:9200/{index_name}" -H 'Content-Type: application/json' -d @{mapping_file}

echo "\\nImporting data..."
curl -X POST "localhost:9200/{index_name}/_bulk" -H 'Content-Type: application/x-ndjson' --data-binary @{output_file}

echo "\\nIndex stats:"
curl -X GET "localhost:9200/{index_name}/_stats/docs"

echo "\\nDone! You can now create visualizations in Kibana."
"""

        with open(script_file, 'w') as f:
            f.write(script_content)

        # Make script executable on Unix systems
        try:
            os.chmod(script_file, 0o755)
        except:
            pass

        print(f"  - Import script: {script_file}")


def main():
    parser = argparse.ArgumentParser(description='Convert medical JSON data for Kibana import')
    parser.add_argument('-f', '--input_file', default='train_fold_0_part_0.json')
    parser.add_argument('-o', '--output', help='Output NDJSON file (default: input_file_kibana.ndjson)')
    parser.add_argument('-i', '--index', default='medical_data',
                        help='Elasticsearch index name (default: medical_data)')
    parser.add_argument('--create-script', action='store_true', help='Create import script for Elasticsearch')

    args = parser.parse_args()

    # Generate output filename if not provided
    if not args.output:
        base_name = os.path.splitext(args.input_file)[0]
        args.output = f"{base_name}_kibana.ndjson"

    # Process the file
    processor = KibanaFileProcessor()
    success = processor.process_file(args.input_file, args.output, args.index)

    if success and args.create_script:
        processor.create_import_script(args.output, args.index)

    if success:
        print("\\nTo import into Elasticsearch:")
        print(
            f"1. Create index: curl -X PUT 'localhost:9200/{args.index}' -H 'Content-Type: application/json' -d @{args.output.replace('.ndjson', '_mapping.json')}'")
        print(
            f"2. Import data: curl -X POST 'localhost:9200/{args.index}/_bulk' -H 'Content-Type: application/x-ndjson' --data-binary @{args.output}")
    else:
        print("Processing failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()