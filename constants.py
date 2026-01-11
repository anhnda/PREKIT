from pathlib import Path
from pandasql import PandaSQL
import sys
from secret import MIMIC_PATH_STR, POSTGRESQL_CONNECTION_STRING


MIMIC_PATH = Path(MIMIC_PATH_STR)
PT = "/Users/anhnd/CodingSpace/Python/PREDKIT"
if sys.platform != "darwin":  
    PT = "/home/anhnda/PREKIT"
# temporary path
TEMP_PATH = Path( "%s/tmp"  %PT)
TEMP_PATH.mkdir(parents=True, exist_ok=True)

# result path 
RESULT_PATH = Path("result")
RESULT_PATH.mkdir(parents=True, exist_ok=True)

# measures whose null represent false value
NULLABLE_MEASURES = [
    "dka_type",
    "macroangiopathy",
    "microangiopathy",
    "mechanical_ventilation",
    "use_NaHCO3",
    "history_aci",
    "history_ami",
    "congestive_heart_failure",
    "liver_disease",
    "ckd_stage",
    "malignant_cancer",
    "hypertension",
    "uti",
    "chronic_pulmonary_disease",
]

# categorical values
CATEGORICAL_MEASURES = [
    "dka_type",
    "gender",
    "race",
    "liver_disease",
    "ckd_stage",
]


TARGET_PATIENT_FILE = "target_patients.csv"

queryPostgresDf = None # PandaSQL(POSTGRESQL_CONNECTION_STRING)

## Archived notebooks
ARCHIVED_NOTEBOOKS_PATH = Path("archived_notebooks_limited-new")
ARCHIVED_NOTEBOOKS_PATH.mkdir(parents=True, exist_ok=True)