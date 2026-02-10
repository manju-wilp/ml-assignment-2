"""
app.py – Streamlit Web Application
Breast Cancer Wisconsin (Diagnostic) Classification
Implements 6 ML models with evaluation metrics, confusion matrix, and classification report.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
    roc_curve,
)

# ── Page Config ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ML Assignment 2 – Breast Cancer Classification",
    page_icon="🔬",
    layout="wide",
)

# ── Paths ───────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
DATA_DIR = os.path.join(BASE_DIR, "data")

# ── Model Registry ──────────────────────────────────────────────────────────────
MODEL_MAP = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "kNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest (Ensemble)": "random_forest.pkl",
    "XGBoost (Ensemble)": "xgboost_model.pkl",
}


# ── Helper Functions ────────────────────────────────────────────────────────────
@st.cache_resource
def load_model(model_name):
    """Load a trained model from disk."""
    path = os.path.join(MODEL_DIR, MODEL_MAP[model_name])
    return joblib.load(path)


@st.cache_resource
def load_scaler():
    """Load the fitted StandardScaler."""
    return joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))


@st.cache_data
def load_metrics():
    """Load the pre-computed metrics comparison table."""
    return pd.read_csv(os.path.join(MODEL_DIR, "metrics.csv"))


@st.cache_data
def load_feature_names():
    """Load feature names."""
    return joblib.load(os.path.join(MODEL_DIR, "feature_names.pkl"))


@st.cache_data
def load_target_names():
    """Load target names."""
    return joblib.load(os.path.join(MODEL_DIR, "target_names.pkl"))


@st.cache_data
def load_default_test_data():
    """Load the bundled test data (with target for evaluation)."""
    return pd.read_csv(os.path.join(DATA_DIR, "test_data.csv"))


@st.cache_data
def load_sample_data():
    """Load sample data without target (for download)."""
    return pd.read_csv(os.path.join(BASE_DIR, "sample_data.csv"))


def compute_metrics(y_true, y_pred, y_prob):
    """Compute all 6 evaluation metrics."""
    return {
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "AUC": round(roc_auc_score(y_true, y_prob), 4),
        "Precision": round(precision_score(y_true, y_pred), 4),
        "Recall": round(recall_score(y_true, y_pred), 4),
        "F1": round(f1_score(y_true, y_pred), 4),
        "MCC": round(matthews_corrcoef(y_true, y_pred), 4),
    }


def plot_confusion_matrix(y_true, y_pred, target_names):
    """Plot confusion matrix as a heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=target_names,
        yticklabels=target_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    return fig


def plot_roc_curve(y_true, y_prob):
    """Plot ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_val = roc_auc_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {auc_val:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    plt.tight_layout()
    return fig


# ── Sidebar ─────────────────────────────────────────────────────────────────────
st.sidebar.title("⚙️ Configuration")

model_choice = st.sidebar.selectbox(
    "Select a Classification Model",
    list(MODEL_MAP.keys()),
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📄 Upload Test Data (CSV)")
st.sidebar.markdown(
    "Upload a CSV file with 30 feature columns (scaled) for prediction. "
    "Optionally include a `target` column (0 = malignant, 1 = benign) for evaluation."
)

# Download sample CSV button (without target)
sample_data = load_sample_data()
csv_data = sample_data.to_csv(index=False).encode('utf-8')
st.sidebar.download_button(
    label="⬇️ Download Sample CSV",
    data=csv_data,
    file_name="sample_data.csv",
    mime="text/csv",
    help="Download a sample CSV file (5 rows, no target column) for making predictions"
)

uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

use_default = st.sidebar.checkbox("Use bundled test data", value=True)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Dataset:** Breast Cancer Wisconsin (Diagnostic)\n\n"
    "**Features:** 30 | **Instances:** 569\n\n"
    "**Task:** Binary Classification (Malignant / Benign)"
)


# ── Main Content ────────────────────────────────────────────────────────────────
st.title("🔬 Breast Cancer Classification Dashboard")
st.markdown(
    "Comparing **6 Machine Learning models** on the Breast Cancer Wisconsin "
    "(Diagnostic) dataset. Select a model from the sidebar to see detailed results."
)

# Load resources
feature_names = load_feature_names()
target_names = load_target_names()

# ── Data Loading ────────────────────────────────────────────────────────────────
data_loaded = False
evaluation_mode = False  # Whether we have ground truth labels

if uploaded_file is not None:
    try:
        test_df = pd.read_csv(uploaded_file)
        has_target = "target" in test_df.columns
        
        if has_target:
            evaluation_mode = True
            st.success(f"✅ Uploaded data with target: {test_df.shape[0]} rows (Evaluation Mode)")
        else:
            st.success(f"✅ Uploaded data: {test_df.shape[0]} rows (Prediction Mode)")
        
        data_loaded = True
    except Exception as e:
        st.error(f"Error reading file: {e}")
elif use_default:
    test_df = load_default_test_data()
    evaluation_mode = True
    data_loaded = True
    st.info(f"📦 Using bundled test data: {test_df.shape[0]} rows (Evaluation Mode)")
else:
    st.warning("⚠️ Please upload a CSV file or check 'Use bundled test data'.")

if data_loaded:
    # Separate features and target (if present)
    if evaluation_mode:
        y_true = test_df["target"].values
        feature_cols = [c for c in test_df.columns if c != "target"]
    else:
        y_true = None
        feature_cols = list(test_df.columns)
    
    X_test = test_df[feature_cols].values

    # Note: uploaded data should already be scaled (StandardScaler applied during preprocessing)
    # If you're uploading raw data, uncomment below to apply scaling:
    # if uploaded_file is not None:
    #     scaler = load_scaler()
    #     try:
    #         X_test = scaler.transform(X_test)
    #     except Exception:
    #         st.warning("⚠️ Could not apply scaler. Ensure uploaded data has the correct features.")

    # ── Tab Layout ──────────────────────────────────────────────────────────────
    if evaluation_mode:
        tab_labels = ["📊 Model Evaluation", "📈 All Models Comparison", "📋 Dataset Info"]
    else:
        tab_labels = ["🔮 Make Predictions", "📈 All Models Comparison", "📋 Dataset Info"]
    
    tabs = st.tabs(tab_labels)

    # ── Tab 1: Model Evaluation or Predictions ──────────────────────────────────
    with tabs[0]:
        st.header(f"Model: {model_choice}")

        model = load_model(model_choice)
        y_pred = model.predict(X_test)
        y_prob = (
            model.predict_proba(X_test)[:, 1]
            if hasattr(model, "predict_proba")
            else model.decision_function(X_test)
        )

        if evaluation_mode:
            # Metrics
            metrics = compute_metrics(y_true, y_pred, y_prob)

            col1, col2, col3, col4, col5, col6 = st.columns(6)
            col1.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
            col2.metric("AUC", f"{metrics['AUC']:.4f}")
            col3.metric("Precision", f"{metrics['Precision']:.4f}")
            col4.metric("Recall", f"{metrics['Recall']:.4f}")
            col5.metric("F1 Score", f"{metrics['F1']:.4f}")
            col6.metric("MCC", f"{metrics['MCC']:.4f}")

            st.markdown("---")

            # Confusion Matrix and ROC side by side
            chart_col1, chart_col2 = st.columns(2)
            with chart_col1:
                st.subheader("Confusion Matrix")
                fig_cm = plot_confusion_matrix(y_true, y_pred, target_names)
                st.pyplot(fig_cm)

            with chart_col2:
                st.subheader("ROC Curve")
                fig_roc = plot_roc_curve(y_true, y_prob)
                st.pyplot(fig_roc)

            # Classification Report
            st.subheader("Classification Report")
            report_dict = classification_report(
                y_true, y_pred, target_names=target_names, output_dict=True
            )
            report_df = pd.DataFrame(report_dict).transpose()
            st.dataframe(report_df.style.format("{:.4f}"), width="stretch")
        else:
            # Prediction Mode - show predictions
            st.subheader("🔮 Predictions")
            
            predictions_df = pd.DataFrame({
                "Row": range(1, len(y_pred) + 1),
                "Prediction (0=Malignant, 1=Benign)": y_pred,
                "Predicted Class": [target_names[p] for p in y_pred],
                "Confidence (Benign)": [f"{p:.4f}" for p in y_prob]
            })
            
            st.dataframe(predictions_df, width="stretch", hide_index=True)
            
            # Summary statistics
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Predictions", len(y_pred))
                st.metric("Predicted Malignant", int((y_pred == 0).sum()))
            with col2:
                st.metric("Predicted Benign", int((y_pred == 1).sum()))
                st.metric("Avg Confidence", f"{np.mean(y_prob):.4f}")
            
            # Download predictions
            st.markdown("---")
            st.subheader("📥 Download Predictions")
            csv_predictions = predictions_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Predictions as CSV",
                data=csv_predictions,
                file_name=f"predictions_{model_choice.replace(' ', '_')}.csv",
                mime="text/csv"
            )

    # ── Tab 2: All Models Comparison ────────────────────────────────────────────
    with tabs[1]:
        st.header("📈 All Models — Comparison Table")
        metrics_df = load_metrics()
        st.dataframe(
            metrics_df.style.highlight_max(
                axis=0,
                subset=["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"],
                color="#90EE90",
            ).format(
                {
                    "Accuracy": "{:.4f}",
                    "AUC": "{:.4f}",
                    "Precision": "{:.4f}",
                    "Recall": "{:.4f}",
                    "F1": "{:.4f}",
                    "MCC": "{:.4f}",
                }
            ),
            width="stretch",
        )

        st.markdown("---")

        # Bar chart comparison
        st.subheader("Metric Comparison (Bar Chart)")
        metric_to_plot = st.selectbox(
            "Select metric to visualize",
            ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"],
        )
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = sns.color_palette("viridis", len(metrics_df))
        bars = ax.bar(metrics_df["Model"], metrics_df[metric_to_plot], color=colors)
        ax.set_ylabel(metric_to_plot)
        ax.set_title(f"{metric_to_plot} by Model")
        ax.set_ylim(0.8, 1.0)
        plt.xticks(rotation=30, ha="right")
        for bar, val in zip(bars, metrics_df[metric_to_plot]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.002,
                f"{val:.4f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        plt.tight_layout()
        st.pyplot(fig)

    # ── Tab 3: Dataset Info ─────────────────────────────────────────────────────
    with tabs[2]:
        st.header("📋 Dataset Information")

        st.markdown(
            """
        **Dataset:** Breast Cancer Wisconsin (Diagnostic)

        **Source:** UCI Machine Learning Repository / sklearn

        **Description:** Features are computed from a digitized image of a fine needle
        aspirate (FNA) of a breast mass. They describe characteristics of the cell nuclei
        present in the image.

        - **Instances:** 569
        - **Features:** 30 numeric features (mean, standard error, and "worst" of 10 real-valued features)
        - **Target:** Binary — Malignant (0) or Benign (1)
        - **Class Distribution:** 212 Malignant, 357 Benign
        """
        )

        st.subheader("Feature List")
        feature_df = pd.DataFrame(
            {"#": range(1, len(feature_names) + 1), "Feature Name": feature_names}
        )
        st.dataframe(feature_df, width="stretch", hide_index=True)

        st.subheader("Data Preview")
        st.dataframe(test_df.head(10), width="stretch")

        if evaluation_mode:
            st.subheader("Class Distribution (Test Data)")
            class_counts = pd.Series(y_true).value_counts().sort_index()
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.bar(
                [target_names[i] for i in class_counts.index],
                class_counts.values,
                color=["#ff6b6b", "#51cf66"],
            )
            ax.set_ylabel("Count")
            ax.set_title("Test Data Class Distribution")
            plt.tight_layout()
            st.pyplot(fig)


# ── Footer ──────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "ML Assignment 2 | Breast Cancer Classification | BITS Pilani MTech"
    "</div>",
    unsafe_allow_html=True,
)
