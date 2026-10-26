"""
Configuration module for SmartDeal AI.
Defines system paths, constants, deal thresholds, and default parameters.
"""

from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
DATABASE_DIR = BASE_DIR / "database"

# Ensure runtime directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
DATABASE_DIR.mkdir(parents=True, exist_ok=True)

# File Paths
DB_PATH = DATABASE_DIR / "smartdeal.db"
SCHEMA_PATH = DATABASE_DIR / "schema.sql"
SAMPLE_PRODUCTS_PATH = DATA_DIR / "sample_products.csv"
SAMPLE_HISTORY_PATH = DATA_DIR / "sample_price_history.csv"
CLEANED_PRODUCTS_PATH = DATA_DIR / "cleaned_products.csv"
MODEL_PIPELINE_PATH = MODELS_DIR / "best_model_pipeline.joblib"
MODEL_METRICS_PATH = MODELS_DIR / "model_metrics.json"

# Random Seed for Reproducibility
RANDOM_SEED = 42

# Currency & Formatting
CURRENCY_SYMBOL = "₹"
LOCALE = "en_IN"

# Deal Scoring Thresholds
DEAL_TIERS = {
    "Great Deal": {"min_score": 80, "badge_color": "#10B981", "text_color": "#ffffff"},
    "Good Deal": {"min_score": 65, "badge_color": "#3B82F6", "text_color": "#ffffff"},
    "Fair Price": {"min_score": 45, "badge_color": "#F59E0B", "text_color": "#000000"},
    "Above Reference": {"min_score": 0, "badge_color": "#EF4444", "text_color": "#ffffff"},
}

# Supported Categories & Standard Brands for Demonstration Data
PRODUCT_CATEGORIES = [
    "Laptops",
    "Smartphones",
    "Audio & Headphones",
    "Smartwatches & Wearables",
    "Televisions",
    "Tablets",
    "Cameras",
    "Home Appliances",
]

POPULAR_BRANDS = {
    "Laptops": ["Apple", "Dell", "HP", "ASUS", "Lenovo", "Acer"],
    "Smartphones": ["Apple", "Samsung", "OnePlus", "Xiaomi", "Google", "Realme"],
    "Audio & Headphones": ["Sony", "Bose", "Sennheiser", "boAt", "JBL", "Noise"],
    "Smartwatches & Wearables": ["Apple", "Samsung", "Garmin", "Fitbit", "Noise", "Amazfit"],
    "Televisions": ["Samsung", "LG", "Sony", "TCL", "Mi", "Vu"],
    "Tablets": ["Apple", "Samsung", "Lenovo", "Xiaomi", "OnePlus"],
    "Cameras": ["Canon", "Sony", "Nikon", "Fujifilm", "Panasonic"],
    "Home Appliances": ["Dyson", "LG", "Samsung", "Philips", "Bosch", "IFB"],
}

RETAILERS = [
    "Amazon India",
    "Flipkart",
    "Croma",
    "Reliance Digital",
    "Tata CLiQ",
]

# ML Pipeline Settings
NUMERICAL_FEATURES = ["original_price", "rating", "review_count"]
CATEGORICAL_FEATURES = ["category", "brand"]
TARGET_VARIABLE = "fair_reference_price"

# Test split fraction
TEST_SIZE = 0.20
