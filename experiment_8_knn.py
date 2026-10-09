import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)


# ============================================================
# 1. Load dataset
# ============================================================

df = pd.read_csv("weather_forecast_data.csv")

X = df[
    [
        "Temperature",
        "Humidity",
        "Wind_Speed",
        "Cloud_Cover",
        "Pressure"
    ]
]

y = (df["Rain"] == "rain").astype(int)


# ============================================================
# 2. 5-fold Stratified Cross Validation
# ============================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# ============================================================
# 3. KNN Pipeline
# ============================================================

model = Pipeline([
    (
        "scaler",
        StandardScaler()
    ),
    (
        "knn",
        KNeighborsClassifier(
            n_neighbors=5,
            weights="uniform",
            metric="minkowski"
        )
    )
])


# ============================================================
# 4. Evaluation
# ============================================================

results = {
    "ACC": [],
    "PRE": [],
    "REC": [],
    "F1": [],
    "AUROC": [],
    "AUPRC": []
}


for train_idx, test_idx in cv.split(X, y):

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    # Train
    model.fit(X_train, y_train)

    # Prediction
    y_pred = model.predict(X_test)

    # Probability
    y_prob = model.predict_proba(X_test)[:, 1]

    # Metrics
    results["ACC"].append(
        accuracy_score(
            y_test,
            y_pred
        )
    )

    results["PRE"].append(
        precision_score(
            y_test,
            y_pred,
            zero_division=0
        )
    )

    results["REC"].append(
        recall_score(
            y_test,
            y_pred,
            zero_division=0
        )
    )

    results["F1"].append(
        f1_score(
            y_test,
            y_pred,
            zero_division=0
        )
    )

    results["AUROC"].append(
        roc_auc_score(
            y_test,
            y_prob
        )
    )

    results["AUPRC"].append(
        average_precision_score(
            y_test,
            y_prob
        )
    )


# ============================================================
# 5. Print Results
# ============================================================

print("\n" + "=" * 70)
print("EXPERIMENT 8: KNN")
print("=" * 70)

print(
    f"{'Metric':<10}"
    f"{'Mean':>12}"
    f"{'Std':>12}"
)

print("-" * 70)

for metric, values in results.items():

    mean = np.mean(values) * 100
    std = np.std(values) * 100

    print(
        f"{metric:<10}"
        f"{mean:>10.2f}%"
        f"{std:>10.2f}%"
    )

print("=" * 70)