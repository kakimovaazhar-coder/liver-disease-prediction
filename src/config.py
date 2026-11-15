"""Project-wide paths and constants.

All other modules import paths from here, so the project can be run from any
working directory (console, VS Code, PyCharm or Google Colab).
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RAW_CSV = RAW_DIR / "Liver_disease_data.csv"
CLEAN_PARQUET = PROCESSED_DIR / "liver_clean.parquet"
DB_PATH = DATA_DIR / "liver.db"

RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
TABLE_DIR = RESULTS_DIR / "tables"
MODEL_DIR = RESULTS_DIR / "models"
BENCH_DIR = RESULTS_DIR / "benchmark_tmp"   # large temporary files, git-ignored
SQL_DIR = ROOT / "sql"
DOCS_DIR = ROOT / "docs"

# Official source (Smart Data Ukimet / data.gov.kz)
DATASET_PAGE = ("https://ashyq.data.gov.kz/dataset/"
                "magda-ds-8ec6e917-d808-494a-b84d-177dd857d7c4/details")
DATASET_URL = "http://s3.sdu.gov.kz/magda-datasets/Liver_disease_data.csv"

TARGET = "Diagnosis"
NUMERIC_FEATURES = ["Age", "BMI", "AlcoholConsumption", "PhysicalActivity",
                    "LiverFunctionTest"]
BINARY_FEATURES = ["Gender", "Smoking", "Diabetes", "Hypertension"]
ORDINAL_FEATURES = ["GeneticRisk"]           # 0 = low, 1 = medium, 2 = high
ALL_FEATURES = NUMERIC_FEATURES + BINARY_FEATURES + ORDINAL_FEATURES

# Plausible value ranges used by the data-quality audit
VALID_RANGES = {
    "Age": (18, 100),
    "BMI": (10, 60),
    "AlcoholConsumption": (0, 30),     # units per week (dataset documentation)
    "PhysicalActivity": (0, 15),       # hours per week
    "LiverFunctionTest": (0, 150),     # composite score
    "Gender": (0, 1), "Smoking": (0, 1), "Diabetes": (0, 1),
    "Hypertension": (0, 1), "GeneticRisk": (0, 2), "Diagnosis": (0, 1),
}

LABELS = {
    "Gender": {0: "Male", 1: "Female"},
    "GeneticRisk": {0: "Low", 1: "Medium", 2: "High"},
    "Diagnosis": {0: "No liver disease", 1: "Liver disease"},
}

MIN_GROUP_SIZE = 10        # groups with fewer patients are suppressed in outputs
RANDOM_STATE = 42
TEST_SIZE = 0.2
CHUNK_SIZE = 500


def ensure_dirs() -> None:
    """Create all output folders if they do not exist yet."""
    for folder in (RAW_DIR, PROCESSED_DIR, FIG_DIR, TABLE_DIR, MODEL_DIR,
                   BENCH_DIR, DOCS_DIR):
        folder.mkdir(parents=True, exist_ok=True)
