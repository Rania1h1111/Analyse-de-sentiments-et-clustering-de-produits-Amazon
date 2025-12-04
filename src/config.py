from pathlib import Path

# Racine du projet = dossier parent de src
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

RAW_AMAZON_FILE = RAW_DATA_DIR / "amazon_reviews.csv"
PROCESSED_AMAZON_FILE_1000  = PROCESSED_DATA_DIR / "amazon_reviews_clean.csv"
PROCESSED_AMAZON_FILE_8000 = PROCESSED_DATA_DIR / "amazon_reviews_8000.csv"


MODELS_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
