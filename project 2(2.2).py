
# ============================================================
# PROJECT: DIABETES DISEASE PREDICTION USING MACHINE LEARNING
# IDE: Spyder
# Dataset: diabetes.csv
# Algorithms: Logistic Regression, Decision Tree, KNN,
#             Random Forest
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    RocCurveDisplay
)

sns.set_theme(style="whitegrid")

# ============================================================
# 1. LOAD DATASET
# ============================================================

DATA_PATH = r"C:\Users\pbhav\Downloads\ml project\diabetes(1).csv"

df = pd.read_csv(DATA_PATH)

print("\n========== DATASET PREVIEW ==========")
print(df.head(10))

print("\n========== DATASET INFORMATION ==========")
print("Number of rows and columns:", df.shape)
print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nDataset information:")
df.info()

print("\nMissing values:")
print(df.isnull().sum())

print("\nStatistical summary:")
print(df.describe())

print("\nDuplicate rows:", df.duplicated().sum())

# ============================================================
# 2. SHOW DATASET VISUALIZATIONS
# ============================================================

# Outcome distribution
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="Outcome")
plt.title("Diabetes Outcome Distribution")
plt.xlabel("Outcome (0 = Negative, 1 = Positive)")
plt.ylabel("Number of People")
plt.tight_layout()
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 7))
sns.heatmap(
    df.corr(numeric_only=True),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
plt.title("Correlation Between Dataset Features")
plt.tight_layout()
plt.show()

# Feature distributions
df.hist(figsize=(12, 9), bins=20)
plt.suptitle("Distribution of Diabetes Dataset Features")
plt.tight_layout()
plt.show()

# ============================================================
# 3. PREPROCESS DATA
# ============================================================

X = df.drop("Outcome", axis=1).copy()
y = df["Outcome"].copy()

# In this dataset, zero can indicate missing measurements.
# Zero pregnancies is valid, so keep it unchanged.
zero_columns = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

X[zero_columns] = X[zero_columns].replace(0, np.nan)

# Split before fitting any preprocessing steps.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining records:", X_train.shape[0])
print("Testing records:", X_test.shape[0])

# ============================================================
# 4. CREATE MACHINE LEARNING MODELS
# ============================================================

models = {
    "Logistic Regression": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42
        ))
    ]),

    "Decision Tree": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", DecisionTreeClassifier(
            class_weight="balanced",
            random_state=42
        ))
    ]),

    "KNN": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(n_neighbors=5))
    ]),

    "Random Forest": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42
        ))
    ])
}

# ============================================================
# 5. TRAIN MODELS AND EVALUATE PERFORMANCE
# ============================================================

results = []
predictions = {}
probabilities = {}

for name, model in models.items():

    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    predictions[name] = y_pred
    probabilities[name] = y_prob

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test, y_pred, zero_division=0
    )
    recall = recall_score(
        y_test, y_pred, zero_division=0
    )
    f1 = f1_score(
        y_test, y_pred, zero_division=0
    )
    auc = roc_auc_score(y_test, y_prob)

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "ROC-AUC": auc
    })

    print(f"\n========== {name} ==========")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {auc:.4f}")

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        y_pred,
        target_names=["Negative class", "Positive class"],
        zero_division=0
    ))

# ============================================================
# 6. COMPARE ALL MODELS
# ============================================================

results_df = pd.DataFrame(results)

print("\n========== MODEL COMPARISON ==========")
print(results_df.round(4).to_string(index=False))

# Save comparison results
results_df.to_csv(
    "diabetes_model_comparison.csv",
    index=False
)

print("\nComparison saved as diabetes_model_comparison.csv")

# Compare evaluation metrics
results_df.set_index("Model")[
    ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
].plot(kind="bar", figsize=(12, 6))

plt.title("Performance Comparison of Machine Learning Models")
plt.ylabel("Score")
plt.ylim(0, 1.05)
plt.xticks(rotation=20, ha="right")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

# ============================================================
# 7. CONFUSION MATRICES
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(11, 9))

for ax, (name, y_pred) in zip(axes.flatten(), predictions.items()):

    cm = confusion_matrix(y_test, y_pred)

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"],
        ax=ax
    )

    ax.set_title(name)
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("Actual Class")

plt.suptitle("Confusion Matrices of All Models")
plt.tight_layout()
plt.show()

# ============================================================
# 8. ROC CURVES
# ============================================================

fig, ax = plt.subplots(figsize=(8, 6))

for name, y_prob in probabilities.items():
    RocCurveDisplay.from_predictions(
        y_test,
        y_prob,
        name=name,
        ax=ax
    )

ax.plot([0, 1], [0, 1], "k--", label="Random baseline")
ax.set_title("ROC Curve Comparison")
plt.tight_layout()
plt.show()

# ============================================================
# 9. RANDOM FOREST FEATURE IMPORTANCE
# ============================================================

rf_pipeline = models["Random Forest"]
rf_model = rf_pipeline.named_steps["model"]

importance = pd.Series(
    rf_model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\n========== FEATURE IMPORTANCE ==========")
print(importance)

plt.figure(figsize=(9, 5))
importance.plot(kind="bar")
plt.title("Random Forest Feature Importance")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# ============================================================
# 10. PREDICT A NEW EXAMPLE
# ============================================================

new_data = pd.DataFrame([{
    "Pregnancies": 2,
    "Glucose": 120,
    "BloodPressure": 70,
    "SkinThickness": 25,
    "Insulin": 80,
    "BMI": 28.5,
    "DiabetesPedigreeFunction": 0.4,
    "Age": 30
}])

# Use the trained Random Forest model
prediction = rf_pipeline.predict(new_data)[0]
probability = rf_pipeline.predict_proba(new_data)[0][1]

print("\n========== NEW EXAMPLE PREDICTION ==========")

if prediction == 1:
    print("Predicted class: Diabetes-positive")
else:
    print("Predicted class: Diabetes-negative")

print(
    "Estimated model probability for positive class:",
    round(probability * 100, 2), "%"
)

print("\nProject completed successfully!")
