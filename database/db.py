"""
Database manager for SmartDeal AI.
Provides SQLite connection management, initialization, parameterized query execution,
and transaction safety for products, price history, predictions, and watchlist items.
"""

import sqlite3
import logging
from contextlib import contextmanager
from typing import Optional, List, Dict, Any, Tuple
import pandas as pd

from config import DB_PATH, SCHEMA_PATH

logger = logging.getLogger(__name__)


def get_db_connection() -> sqlite3.Connection:
    """Create and return a SQLite connection configured with Row factory."""
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


@contextmanager
def db_session():
    """Context manager for SQLite database transactions."""
    conn = get_db_connection()
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.error(f"Database transaction error: {e}", exc_info=True)
        raise
    finally:
        conn.close()


def init_db(force_recreate: bool = False) -> bool:
    """
    Initialize SQLite database schema from schema.sql.
    If force_recreate is True, drops existing tables and recreates them.
    """
    try:
        if not SCHEMA_PATH.exists():
            raise FileNotFoundError(f"Schema file not found at {SCHEMA_PATH}")

        with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
            schema_sql = f.read()

        with db_session() as conn:
            cursor = conn.cursor()
            if force_recreate:
                logger.info("Dropping existing tables...")
                cursor.executescript("""
                    DROP TABLE IF EXISTS watchlist;
                    DROP TABLE IF EXISTS predictions;
                    DROP TABLE IF EXISTS price_history;
                    DROP TABLE IF EXISTS products;
                """)
            cursor.executescript(schema_sql)
            logger.info("Database initialized successfully.")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}", exc_info=True)
        raise


def execute_query(query: str, params: Optional[Tuple[Any, ...]] = None) -> int:
    """Execute an INSERT, UPDATE, or DELETE query with parameterized inputs."""
    with db_session() as conn:
        cursor = conn.cursor()
        cursor.execute(query, params or ())
        return cursor.rowcount


def fetch_all(query: str, params: Optional[Tuple[Any, ...]] = None) -> List[Dict[str, Any]]:
    """Execute a SELECT query and return rows as a list of dicts."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query, params or ())
        rows = cursor.fetchall()
        return [dict(row) for row in rows]


def fetch_one(query: str, params: Optional[Tuple[Any, ...]] = None) -> Optional[Dict[str, Any]]:
    """Execute a SELECT query and return a single record as dict, or None."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query, params or ())
        row = cursor.fetchone()
        return dict(row) if row else None


def fetch_df(query: str, params: Optional[Tuple[Any, ...]] = None) -> pd.DataFrame:
    """Execute a query and return results as a Pandas DataFrame."""
    with get_db_connection() as conn:
        return pd.read_sql_query(query, conn, params=params or ())


def insert_or_replace_products(df: pd.DataFrame) -> int:
    """Insert or replace products in bulk from a DataFrame."""
    required_cols = [
        "product_id", "product_name", "category", "brand",
        "original_price", "current_price", "discount_percentage",
        "rating", "review_count", "retailer", "timestamp"
    ]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column in products DataFrame: {col}")

    # Ensure optional columns have safe defaults
    working_df = df.copy()
    if "fair_reference_price" not in working_df.columns:
        working_df["fair_reference_price"] = None
    if "is_synthetic" not in working_df.columns:
        working_df["is_synthetic"] = 1

    records = working_df[[
        "product_id", "product_name", "category", "brand",
        "original_price", "current_price", "discount_percentage",
        "rating", "review_count", "retailer", "timestamp",
        "fair_reference_price", "is_synthetic"
    ]].values.tolist()

    query = """
    INSERT OR REPLACE INTO products (
        product_id, product_name, category, brand,
        original_price, current_price, discount_percentage,
        rating, review_count, retailer, timestamp,
        fair_reference_price, is_synthetic
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    with db_session() as conn:
        cursor = conn.cursor()
        cursor.executemany(query, records)
        return cursor.rowcount


def insert_price_history(df: pd.DataFrame) -> int:
    """Insert historical price observations in bulk from a DataFrame."""
    required_cols = ["product_id", "observed_date", "price"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column in history DataFrame: {col}")

    working_df = df.copy()
    if "retailer" not in working_df.columns:
        working_df["retailer"] = "Unknown"
    if "in_stock" not in working_df.columns:
        working_df["in_stock"] = 1
    if "is_synthetic" not in working_df.columns:
        working_df["is_synthetic"] = 1

    records = working_df[[
        "product_id", "observed_date", "price", "retailer", "in_stock", "is_synthetic"
    ]].values.tolist()

    query = """
    INSERT INTO price_history (
        product_id, observed_date, price, retailer, in_stock, is_synthetic
    ) VALUES (?, ?, ?, ?, ?, ?)
    """
    with db_session() as conn:
        cursor = conn.cursor()
        cursor.executemany(query, records)
        return cursor.rowcount


def log_prediction(category: str, brand: str, original_price: float, rating: float,
                   review_count: int, predicted_price: float, model_name: str,
                   deal_score: Optional[float] = None, deal_rating: Optional[str] = None) -> int:
    """Save an inference record into the predictions log table."""
    from datetime import datetime
    now_iso = datetime.now().isoformat()
    query = """
    INSERT INTO predictions (
        created_at, category, brand, original_price, rating, review_count,
        predicted_price, model_name, deal_score, deal_rating
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    return execute_query(query, (
        now_iso, category, brand, original_price, rating, review_count,
        predicted_price, model_name, deal_score, deal_rating
    ))
