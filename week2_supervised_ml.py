
# Week 2 Task: Supervised Machine Learning Models

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# 1. Load the dataset
df = pd.read_csv("student-mat.csv", sep=";")

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)
print(df.head())

# 2. Separate features and targets
# G3 is the final grade, so it must not be included in the input features.
X = df.drop(columns=["G3"])
y_regression = df["G3"]

# Classification target: 1 = Pass, 0 = Fail
y_classification = (df["G3"] >= 10).astype(int)

# 3. Identify numerical and categorical columns
numeric_columns = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_columns = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

# 4. Prepare preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", StandardScaler(), numeric_columns),
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns
        )
    ]
)

# --------------------------------------------------
# PART A: REGRESSION MODELS
# --------------------------------------------------

print("\n===== REGRESSION MODELS =====")

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X, y_regression, test_size=0.2, random_state=42
)

regression_models = {
    "Linear Regression": LinearRegression(),
    "Random Forest Regressor": RandomForestRegressor(
        n_estimators=200, random_state=42
    )
}

regression_results = []

for name, model in regression_models.items():
    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train_reg, y_train_reg)
    predictions = pipeline.predict(X_test_reg)

    mae = mean_absolute_error(y_test_reg, predictions)
    rmse = np.sqrt(mean_squared_error(y_test_reg, predictions))

    regression_results.append({
        "Model": name,
        "MAE": round(mae, 3),
        "RMSE": round(rmse, 3)
    })

    print(f"\n{name}")
    print("MAE:", round(mae, 3))
    print("RMSE:", round(rmse, 3))

regression_df = pd.DataFrame(regression_results)
regression_df.to_csv("week2_regression_comparison.csv", index=False)

print("\nRegression comparison:")
print(regression_df.to_string(index=False))

# --------------------------------------------------
# PART B: CLASSIFICATION MODELS
# --------------------------------------------------

print("\n===== CLASSIFICATION MODELS =====")

X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(
    X,
    y_classification,
    test_size=0.2,
    random_state=42,
    stratify=y_classification
)

classification_models = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=5, random_state=42
    )
}

classification_results = []

for name, model in classification_models.items():
    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train_cls, y_train_cls)
    predictions = pipeline.predict(X_test_cls)

    accuracy = accuracy_score(y_test_cls, predictions)
    precision = precision_score(
        y_test_cls, predictions, zero_division=0
    )
    recall = recall_score(
        y_test_cls, predictions, zero_division=0
    )
    f1 = f1_score(y_test_cls, predictions, zero_division=0)

    classification_results.append({
        "Model": name,
        "Accuracy": round(accuracy, 3),
        "Precision": round(precision, 3),
        "Recall": round(recall, 3),
        "F1-Score": round(f1, 3)
    })

    print(f"\n{name}")
    print("Accuracy:", round(accuracy, 3))
    print("Precision:", round(precision, 3))
    print("Recall:", round(recall, 3))
    print("F1-Score:", round(f1, 3))

classification_df = pd.DataFrame(classification_results)
classification_df.to_csv(
    "week2_classification_comparison.csv", index=False
)

print("\nClassification comparison:")
print(classification_df.to_string(index=False))

# --------------------------------------------------
# PART C: VISUALIZE MODEL COMPARISONS
# --------------------------------------------------

regression_df.set_index("Model")[["MAE", "RMSE"]].plot(
    kind="bar", figsize=(9, 5)
)
plt.title("Regression Model Comparison")
plt.ylabel("Error (lower is better)")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("week2_regression_comparison.png")
plt.show()

classification_df.set_index("Model")[
    ["Accuracy", "Precision", "Recall", "F1-Score"]
].plot(kind="bar", figsize=(10, 5))

plt.title("Classification Model Comparison")
plt.ylabel("Score (higher is better)")
plt.ylim(0, 1)
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("week2_classification_comparison.png")
plt.show()

print("\nWeek 2 task completed!")
print("Comparison CSV files and charts have been saved.")
