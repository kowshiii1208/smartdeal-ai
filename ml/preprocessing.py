"""
Preprocessing pipeline for SmartDeal AI.
Builds scikit-learn ColumnTransformer for categorical and numerical features
with robust imputation, scaling, and one-hot encoding.
"""

from typing import Tuple, List, Optional
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from config import NUMERICAL_FEATURES, CATEGORICAL_FEATURES, TARGET_VARIABLE


def build_preprocessor(
    num_features: Optional[List[str]] = None,
    cat_features: Optional[List[str]] = None,
) -> ColumnTransformer:
    """
    Construct a scikit-learn ColumnTransformer for tabular product features.
    
    Numerical pipeline:
      - Median imputation for missing values.
      - Standard scaling (zero mean, unit variance).
      
    Categorical pipeline:
      - Constant 'Unknown' imputation for missing values.
      - OneHotEncoder with handle_unknown='ignore' to gracefully handle unseen brands/categories.
    """
    num_cols = num_features or NUMERICAL_FEATURES
    cat_cols = cat_features or CATEGORICAL_FEATURES

    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, num_cols),
            ("cat", categorical_pipeline, cat_cols),
        ],
        remainder="drop",
    )

    return preprocessor


def prepare_features_and_target(
    df: pd.DataFrame,
    target_col: str = TARGET_VARIABLE,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Extract feature matrix X and target vector y while strictly preventing target leakage.
    
    Target Leakage Prevention:
    - We NEVER include 'current_price' or 'discount_percentage' when predicting fair_reference_price,
      because current_price is the variable we want to evaluate against the fair reference price!
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

    # Drop any row with missing target
    valid_data = df.dropna(subset=[target_col]).copy()
    
    # Feature columns
    feature_cols = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    missing_features = [col for col in feature_cols if col not in valid_data.columns]
    if missing_features:
        raise ValueError(f"Missing required feature columns: {missing_features}")

    X = valid_data[feature_cols].copy()
    y = valid_data[target_col].copy()

    return X, y
