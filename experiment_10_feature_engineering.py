import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

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


# ============================================================
# 2. Original features
# ============================================================

features = [
    "Temperature",
    "Humidity",
    "Wind_Speed",
    "Cloud_Cover",
    "Pressure"
]

X = df[features].copy()

y = (df["Rain"] == "rain").astype(int)


# ============================================================
# 3. Feature Engineering
# ============================================================

# Humidity × Cloud Cover
X["Humidity_Cloud"] = (
    X["Humidity"] * X["Cloud_Cover"]
)

# Temperature × Humidity
X["Temperature_Humidity"] = (
    X["Temperature"] * X["Humidity"]
)

# Temperature × Cloud Cover
X["Temperature_Cloud"] = (
    X["Temperature"] * X["Cloud_Cover"]
)

# Humidity × Wind Speed
X["Humidity_Wind"] = (
    X["Humidity"] * X["Wind_Speed"]
)

# Cloud Cover × Wind Speed
X["Cloud_Wind"] = (
    X["Cloud_Cover"] * X["Wind_Speed"]
)


# ============================================================
# 4. Display features
# ============================================================

print("\n" + "=" * 70)
print("EXPERIMENT 10: FEATURE ENGINEERING")
print("=" * 70)

print("\nOriginal features:")
print(features)

print("\nNew engineered features:")

engineered_features = [
    "Humidity_Cloud",
    "Temperature_Humidity",
    "Temperature_Cloud",
    "Humidity_Wind",
    "Cloud_Wind"
]

for feature in engineered_features:
    print(" -", feature)

print("\nTotal features:", X.shape[1])


# ============================================================
# 5. 5-fold Stratified Cross Validation
# ============================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# ============================================================
# 6. Random Forest
# ============================================================

model = RandomForestClassifier(
    n_estimators=100,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1
)


# ============================================================
# 7. Evaluation
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
    model.fit(
        X_train,
        y_train
    )

    # Prediction
    y_pred = model.predict(
        X_test
    )

    # Probability
    y_prob = model.predict_proba(
        X_test
    )[:, 1]

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
# 8. Print results
# ============================================================

print("\n" + "=" * 70)
print("FEATURE-ENGINEERED RANDOM FOREST RESULTS")
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