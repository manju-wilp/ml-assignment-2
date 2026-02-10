"""
Load and process data
"""

import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

# Configuration
RANDOM_STATE = 42
TEST_SIZE = 0.20
CLASS_IMBALANCE_THRESHOLD = 0.20

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "dataset"
MODEL_DIR = BASE_DIR / "model" / "saved_models"
TRAIN_FILE = DATA_DIR / "train.csv"
TEST_FILE = DATA_DIR / "test.csv"

# Ensure directories exist
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def load_and_preprocess_data():
    """
    Load dataset and perform initial preprocessing.
    Returns: X_train, X_test, y_train, y_test, preprocessor, feature_schema
    """
    print("Loading data...")
    df_train = pd.read_csv(TRAIN_FILE)
    df_test = pd.read_csv(TEST_FILE)
    
    # Combine for consistent preprocessing
    df = pd.concat([df_train, df_test], axis=0, ignore_index=True)
    print(f"Total dataset shape: {df.shape}")
    
    # Drop non-predictive columns
    cols_to_drop = ['id', 'Unnamed: 0']
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    
    # Separate features and target
    target_col = 'satisfaction'
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Encode target: 'satisfied' = 1 (positive class), 'neutral or dissatisfied' = 0
    y = (y == 'satisfied').astype(int)
    
    print(f"Features shape: {X.shape}")
    print(f"Target distribution:\n{pd.Series(y).value_counts(normalize=True)}")
    
    # Check class imbalance
    pos_ratio = y.sum() / len(y)
    imbalance_ratio = abs(pos_ratio - 0.5)
    use_class_weight = imbalance_ratio > CLASS_IMBALANCE_THRESHOLD
    print(f"Class imbalance ratio: {imbalance_ratio:.4f}")
    print(f"Using class_weight='balanced': {use_class_weight}")
    
    # Identify numeric and categorical columns
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    
    print(f"Numeric features ({len(numeric_features)}): {numeric_features[:5]}...")
    print(f"Categorical features ({len(categorical_features)}): {categorical_features}")
    
    # Store schema for validation
    feature_schema = {
        'columns': X.columns.tolist(),
        'numeric_features': numeric_features,
        'categorical_features': categorical_features,
        'dtypes': X.dtypes.astype(str).to_dict()
    }
    
    # Save schema
    with open(MODEL_DIR / 'feature_schema.json', 'w') as f:
        json.dump(feature_schema, f, indent=2)
    
    # Generate sample test CSV (10 rows, no target)
    sample_df = X.sample(n=10, random_state=RANDOM_STATE)
    sample_df.to_csv(BASE_DIR / 'sample_test.csv', index=False)
    print(f"Sample test CSV saved with {len(sample_df)} rows")
    
    # Stratified train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test, feature_schema, use_class_weight


def create_preprocessor(numeric_features, categorical_features, scale=True):
    """
    Create preprocessing pipeline.
    
    Args:
        numeric_features: List of numeric column names
        categorical_features: List of categorical column names
        scale: Whether to apply StandardScaler (for LR/KNN)
    
    Returns:
        ColumnTransformer
    """
    numeric_steps = [('imputer', SimpleImputer(strategy='median'))]
    if scale:
        numeric_steps.append(('scaler', StandardScaler()))
    
    numeric_transformer = Pipeline(steps=numeric_steps)
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    transformers = [('num', numeric_transformer, numeric_features)]
    if categorical_features:
        transformers.append(('cat', categorical_transformer, categorical_features))
    
    preprocessor = ColumnTransformer(transformers=transformers)
    
    return preprocessor

def main():
    """Main training pipeline."""
    print("="*60)
    print("AIRLINE PASSENGER SATISFACTION - MODEL TRAINING")
    print("="*60)
    
    # Load and preprocess
    X_train, X_test, y_train, y_test, feature_schema, use_class_weight = load_and_preprocess_data()
    
    # Train all models
    results_df = train_and_evaluate_models(X_train, X_test, y_train, y_test, feature_schema, use_class_weight)
    
    # Display results
    print("\n" + "="*60)
    print("FINAL RESULTS - ALL MODELS")
    print("="*60)
    print(results_df.to_string(index=False))
    
    # Save results
    results_path = MODEL_DIR / "metrics_comparison.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to {results_path}")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"All models saved in: {MODEL_DIR}")
    print(f"Sample test CSV: {BASE_DIR / 'sample_test.csv'}")
    print(f"Feature schema: {MODEL_DIR / 'feature_schema.json'}")


if __name__ == "__main__":
    main()


