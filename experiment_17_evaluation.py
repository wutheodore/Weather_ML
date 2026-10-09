import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score
)


# ============================================================
# 1. Load dataset
# ============================================================

df = pd.read_csv("weather_forecast_data.csv")


# ============================================================
# 2. Features and target
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
# 3. Display dataset information
# ============================================================

print("\n" + "=" * 70)
print("EXPERIMENT 17: ROC vs PRECISION-RECALL + CONFUSION MATRIX")
print("=" * 70)

print("\nDataset shape:")
print(X.shape)

print("\nTarget distribution:")
print(
    pd.Series(
        y,
        name="Rain"
    ).value_counts()
)

print("\nTarget percentage:")
print(
    pd.Series(
        y,
        name="Rain"
    ).value_counts(
        normalize=True
    ).mul(100)
)


# ============================================================
# 4. Cross-validation
# ============================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# ============================================================
# 5. Store out-of-fold predictions
# ============================================================

y_true_all = []
y_prob_all = []
y_pred_all = []


# ============================================================
# 6. Train Random Forest using 5-fold CV
# ============================================================

for train_idx, test_idx in cv.split(X, y):

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]


    model = RandomForestClassifier(
        n_estimators=100,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1
    )


    model.fit(
        X_train,
        y_train
    )


    # Probability of Rain
    y_prob = model.predict_proba(
        X_test
    )[:, 1]


    # Default threshold = 0.50
    y_pred = (
        y_prob >= 0.50
    ).astype(int)


    y_true_all.extend(
        y_test
    )

    y_prob_all.extend(
        y_prob
    )

    y_pred_all.extend(
        y_pred
    )


# Convert to numpy arrays

y_true_all = np.array(
    y_true_all
)

y_prob_all = np.array(
    y_prob_all
)

y_pred_all = np.array(
    y_pred_all
)


# ============================================================
# 7. Confusion Matrix
# ============================================================

cm = confusion_matrix(
    y_true_all,
    y_pred_all
)


print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print(cm)


print("\nClassification Report:")
print(
    classification_report(
        y_true_all,
        y_pred_all,
        target_names=[
            "No Rain",
            "Rain"
        ],
        digits=4
    )
)


# ============================================================
# 8. Extract confusion matrix values
# ============================================================

tn, fp, fn, tp = cm.ravel()

print("\nConfusion Matrix Interpretation:")
print(
    f"True Negatives  (TN): {tn}"
)

print(
    f"False Positives (FP): {fp}"
)

print(
    f"False Negatives (FN): {fn}"
)

print(
    f"True Positives  (TP): {tp}"
)


# ============================================================
# 9. ROC Curve
# ============================================================

fpr, tpr, roc_thresholds = roc_curve(
    y_true_all,
    y_prob_all
)

auroc = roc_auc_score(
    y_true_all,
    y_prob_all
)


# ============================================================
# 10. Precision-Recall Curve
# ============================================================

precision, recall, pr_thresholds = precision_recall_curve(
    y_true_all,
    y_prob_all
)

auprc = average_precision_score(
    y_true_all,
    y_prob_all
)


# ============================================================
# 11. Print metrics
# ============================================================

print("\n" + "=" * 70)
print("ROC vs PRECISION-RECALL")
print("=" * 70)

print(
    f"AUROC : {auroc * 100:.2f}%"
)

print(
    f"AUPRC : {auprc * 100:.2f}%"
)


# ============================================================
# 12. ROC Curve Plot
# ============================================================

plt.figure(
    figsize=(8, 6)
)

plt.plot(
    fpr,
    tpr,
    label=f"Random Forest (AUROC = {auroc:.4f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "ROC Curve - Random Forest"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    "experiment_17_roc_curve.png",
    dpi=300
)

plt.show()


# ============================================================
# 13. Precision-Recall Curve Plot
# ============================================================

plt.figure(
    figsize=(8, 6)
)

plt.plot(
    recall,
    precision,
    label=f"Random Forest (AUPRC = {auprc:.4f})"
)

# Positive-class prevalence
baseline = np.mean(
    y_true_all
)

plt.axhline(
    baseline,
    linestyle="--",
    label=f"Baseline = {baseline:.4f}"
)

plt.xlabel(
    "Recall"
)

plt.ylabel(
    "Precision"
)

plt.title(
    "Precision-Recall Curve - Random Forest"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    "experiment_17_pr_curve.png",
    dpi=300
)

plt.show()


# ============================================================
# 14. Confusion Matrix Plot
# ============================================================

plt.figure(
    figsize=(7, 6)
)

plt.imshow(
    cm,
    interpolation="nearest"
)

plt.title(
    "Confusion Matrix - Random Forest"
)

plt.colorbar()

plt.xticks(
    [0, 1],
    ["No Rain", "Rain"]
)

plt.yticks(
    [0, 1],
    ["No Rain", "Rain"]
)

plt.xlabel(
    "Predicted Label"
)

plt.ylabel(
    "True Label"
)


# Display numbers inside matrix

for i in range(2):

    for j in range(2):

        plt.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center"
        )


plt.tight_layout()

plt.savefig(
    "experiment_17_confusion_matrix.png",
    dpi=300
)

plt.show()


# ============================================================
# 15. Final summary
# ============================================================

print("\n" + "=" * 70)
print("EXPERIMENT 17 COMPLETE")
print("=" * 70)

print(
    "\nGenerated files:"
)

print(
    " - experiment_17_roc_curve.png"
)

print(
    " - experiment_17_pr_curve.png"
)

print(
    " - experiment_17_confusion_matrix.png"
)

print("=" * 70)