"""
train_models.py
---------------
Trains 6 classification models on the Breast Cancer Wisconsin (Diagnostic) dataset,
evaluates each using 6 metrics, saves trained models and artifacts for deployment.
"""

import os
import warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Metrics
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    classification_report,
    confusion_matrix,
)

warnings.filterwarnings("ignore")

# ── Paths ───────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)


# ── 1. Load Dataset ────────────────────────────────────────────────────────────
print("=" * 60)
print("Loading Breast Cancer Wisconsin (Diagnostic) Dataset")
print("=" * 60)

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="target")

print(f"Dataset shape    : {X.shape}")
print(f"Features         : {X.shape[1]}")
print(f"Instances        : {X.shape[0]}")
print(f"Target classes   : {dict(zip(data.target_names, np.bincount(y)))}")
print(f"Missing values   : {X.isnull().sum().sum()}")
print()

# ── 2. Train / Test Split ──────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train size: {X_train.shape[0]}  |  Test size: {X_test.shape[0]}")

# ── 3. Feature Scaling ─────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

# Save test data (for Streamlit upload demo)
test_df = pd.DataFrame(X_test_scaled, columns=data.feature_names)
test_df["target"] = y_test.values
test_df.to_csv(os.path.join(DATA_DIR, "test_data.csv"), index=False)

# Also save feature names and target names
joblib.dump(list(data.feature_names), os.path.join(MODEL_DIR, "feature_names.pkl"))
joblib.dump(list(data.target_names), os.path.join(MODEL_DIR, "target_names.pkl"))

print("Scaler saved.  Test data saved.\n")


# ── 4. Define Models ───────────────────────────────────────────────────────────
models = {
    "Logistic Regression": LogisticRegression(max_iter=10000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "kNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "Random Forest (Ensemble)": RandomForestClassifier(
        n_estimators=100, random_state=42
    ),
    "XGBoost (Ensemble)": XGBClassifier(
        n_estimators=100,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
    ),
}

model_filenames = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "kNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest (Ensemble)": "random_forest.pkl",
    "XGBoost (Ensemble)": "xgboost_model.pkl",
}


# ── 5. Train, Evaluate, Save ───────────────────────────────────────────────────
def evaluate_model(model, X_tr, y_tr, X_te, y_te):
    """Train a model and return metrics dict."""
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    y_prob = (
        model.predict_proba(X_te)[:, 1]
        if hasattr(model, "predict_proba")
        else model.decision_function(X_te)
    )

    metrics = {
        "Accuracy": round(accuracy_score(y_te, y_pred), 4),
        "AUC": round(roc_auc_score(y_te, y_prob), 4),
        "Precision": round(precision_score(y_te, y_pred), 4),
        "Recall": round(recall_score(y_te, y_pred), 4),
        "F1": round(f1_score(y_te, y_pred), 4),
        "MCC": round(matthews_corrcoef(y_te, y_pred), 4),
    }
    return metrics, y_pred


results = []
print("=" * 60)
print("Training & Evaluating Models")
print("=" * 60)

for name, model in models.items():
    print(f"\n▸ {name}")
    metrics, y_pred = evaluate_model(
        model, X_train_scaled, y_train, X_test_scaled, y_test
    )

    # Save trained model
    fname = model_filenames[name]
    joblib.dump(model, os.path.join(MODEL_DIR, fname))
    print(f"  Saved  → model/{fname}")

    # Print metrics
    for k, v in metrics.items():
        print(f"  {k:12s}: {v}")

    # Print classification report
    print(f"\n  Classification Report:")
    report = classification_report(y_test, y_pred, target_names=data.target_names)
    for line in report.split("\n"):
        print(f"  {line}")

    metrics["Model"] = name
    results.append(metrics)

# ── 6. Save Metrics Summary ────────────────────────────────────────────────────
metrics_df = pd.DataFrame(results)
cols = ["Model", "Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
metrics_df = metrics_df[cols]
metrics_df.to_csv(os.path.join(MODEL_DIR, "metrics.csv"), index=False)

print("\n" + "=" * 60)
print("COMPARISON TABLE")
print("=" * 60)
print(metrics_df.to_string(index=False))
print(f"\nMetrics saved → model/metrics.csv")
print("All models trained and saved successfully!")
