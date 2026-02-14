# Breast Cancer Classification - ML Assignment 2

**Name**: Manjunath S

**BITS ID**: 2025AA05935

**Date**: 14-02-2026

## 1. Problem Statement

This assignment tackles the **binary classification** of breast tumors as **Malignant** or **Benign** using the Breast Cancer Wisconsin (Diagnostic) dataset. The goal is to build, evaluate, and compare 6 machine learning classification models and deploy them via an interactive Streamlit web application.

Accurate classification of breast tumors is critical for early detection and treatment of breast cancer, which is one of the most common cancers worldwide.

---

## 2. Dataset Description

| Property                     | Details                                               |
| ---------------------------- | ----------------------------------------------------- |
| **Name**               | Breast Cancer Wisconsin (Diagnostic)                  |
| **Source**             | UCI Machine Learning Repository /`sklearn.datasets` |
| **Instances**          | 569                                                   |
| **Features**           | 30 numeric features                                   |
| **Target**             | Binary - Malignant (0) or Benign (1)                  |
| **Class Distribution** | 212 Malignant, 357 Benign                             |
| **Missing Values**     | None                                                  |

### Feature Details

Features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. They describe characteristics of the cell nuclei present in the image. For each of 10 real-valued features, the **mean**, **standard error**, and **worst** (largest) values are recorded, giving 30 features in total:

- Radius, Texture, Perimeter, Area, Smoothness
- Compactness, Concavity, Concave points, Symmetry, Fractal dimension

---

## 3. Models Used

### Comparison Table

| ML Model Name            | Accuracy | AUC    | Precision | Recall | F1     | MCC    |
| ------------------------ | -------- | ------ | --------- | ------ | ------ | ------ |
| Logistic Regression      | 0.9825   | 0.9954 | 0.9861    | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree            | 0.9123   | 0.9157 | 0.9559    | 0.9028 | 0.9286 | 0.8174 |
| kNN                      | 0.9561   | 0.9788 | 0.9589    | 0.9722 | 0.9655 | 0.9054 |
| Naive Bayes              | 0.9298   | 0.9868 | 0.9444    | 0.9444 | 0.9444 | 0.8492 |
| Random Forest (Ensemble) | 0.9561   | 0.9939 | 0.9589    | 0.9722 | 0.9655 | 0.9054 |
| XGBoost (Ensemble)       | 0.9561   | 0.9901 | 0.9467    | 0.9861 | 0.9660 | 0.9058 |

---

## 4. Observations on Model Performance

| ML Model Name                      | Observation about model performance                                                                                                                                                                                                                                                                                                                            |
| ---------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Logistic Regression**      | Achieved the **highest accuracy (98.25%)** and best overall performance across all metrics. <br />The linear decision boundary works exceptionally well on this dataset because the features (after scaling) <br />are largely linearly separable. Its high AUC (0.9954) confirms excellent discrimination between malignant and benign classes.          |
| **Decision Tree**            | Showed the **lowest performance** among all models (accuracy 91.23%, MCC 0.8174). <br />Decision trees are prone to overfitting and create axis-aligned splits that may not capture the smooth class boundaries in this dataset. <br />The relatively low recall (0.9028) means it misses more malignant cases, which is concerning in a medical context. |
| **kNN**                      | Performed well with 95.61% accuracy. KNN benefits from feature scaling (StandardScaler) and works effectively in <br />the 30-dimensional feature space. Its non-parametric nature allows it to capture complex decision boundaries without assumptions <br />about data distribution.                                                                         |
| **Naive Bayes**              | Achieved moderate accuracy (92.98%) but a notably **high AUC (0.9868)**, indicating strong probabilistic ranking ability even though <br />hard predictions are less accurate. The conditional independence assumption is partially violated in this dataset <br />(many features are correlated), limiting its classification accuracy.                  |
| **Random Forest (Ensemble)** | Matched kNN in accuracy (95.61%) with a **very high AUC (0.9939)**, second only to Logistic Regression. <br />The ensemble of decision trees effectively reduces variance and overfitting compared to a single Decision Tree, <br />dramatically improving performance (from 91.23% to 95.61%).                                                           |
| **XGBoost (Ensemble)**       | Also achieved 95.61% accuracy with the **highest recall (0.9861)** among non-LR models, meaning it correctly identifies the <br />most malignant cases. XGBoost's gradient boosting mechanism and built-in regularization make it a strong and robust classifier. <br />Its AUC of 0.9901 confirms excellent ranking capability.                          |

### Key Insights

1. **Logistic Regression dominates** on this dataset - the features are well-behaved after scaling and largely linearly separable, making a simple linear model highly effective.
2. **Ensemble methods** (Random Forest, XGBoost) significantly outperform the single Decision Tree, demonstrating the power of ensembling to reduce overfitting.
3. **All models achieve AUC > 0.91**, showing the dataset has strong discriminative features.
4. **Feature scaling** is critical - KNN and Logistic Regression benefit greatly from StandardScaler normalization.

---

