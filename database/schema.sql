-- SQLite Database Schema for SmartDeal AI

CREATE TABLE IF NOT EXISTS products (
    product_id TEXT PRIMARY KEY,
    product_name TEXT NOT NULL,
    category TEXT NOT NULL,
    brand TEXT NOT NULL,
    original_price REAL NOT NULL CHECK(original_price >= 0),
    current_price REAL NOT NULL CHECK(current_price >= 0),
    discount_percentage REAL DEFAULT 0.0 CHECK(discount_percentage >= 0 AND discount_percentage <= 100),
    rating REAL DEFAULT 4.0 CHECK(rating >= 0 AND rating <= 5),
    review_count INTEGER DEFAULT 0 CHECK(review_count >= 0),
    retailer TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    fair_reference_price REAL DEFAULT NULL,
    is_synthetic INTEGER DEFAULT 1
);

CREATE TABLE IF NOT EXISTS price_history (
    history_id INTEGER PRIMARY KEY AUTOINCREMENT,
    product_id TEXT NOT NULL,
    observed_date TEXT NOT NULL,
    price REAL NOT NULL CHECK(price >= 0),
    retailer TEXT,
    in_stock INTEGER DEFAULT 1,
    is_synthetic INTEGER DEFAULT 1,
    FOREIGN KEY(product_id) REFERENCES products(product_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS predictions (
    prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    category TEXT NOT NULL,
    brand TEXT NOT NULL,
    original_price REAL NOT NULL,
    rating REAL,
    review_count INTEGER,
    predicted_price REAL NOT NULL,
    model_name TEXT NOT NULL,
    deal_score REAL,
    deal_rating TEXT
);

CREATE TABLE IF NOT EXISTS watchlist (
    watchlist_id INTEGER PRIMARY KEY AUTOINCREMENT,
    product_id TEXT NOT NULL UNIQUE,
    target_price REAL NOT NULL,
    added_date TEXT NOT NULL,
    notes TEXT,
    FOREIGN KEY(product_id) REFERENCES products(product_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_products_category ON products(category);
CREATE INDEX IF NOT EXISTS idx_products_brand ON products(brand);
CREATE INDEX IF NOT EXISTS idx_price_history_product_id ON price_history(product_id);
CREATE INDEX IF NOT EXISTS idx_price_history_date ON price_history(observed_date);
