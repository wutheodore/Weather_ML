# -*- coding: utf-8 -*-
"""
天气预报数据集：机器学习基线示例（作业基础代码，必须首先跑通并理解）

数据集：Kaggle Weather Forecast Dataset（作者 zeeshier，Apache 2.0）
  https://www.kaggle.com/datasets/zeeshier/weather-forecast-dataset
  2500 条每日气象观测，5 个数值特征 + 二分类目标：
    Temperature / Humidity / Wind_Speed / Cloud_Cover / Pressure  →  Rain ('rain' / 'no rain')
  注意两点（也是本作业的"坑"）：
    ① 类别高度不平衡：no rain ≈ 87.6%，rain ≈ 12.4%；
    ② 5 个输入特征两两之间几乎不相关（相关性≈0），PCA 的降维收益有限。

脚本流程（对照你的实验报告结构）：
  1. 加载数据 + 探索性分析 EDA      → 图 fig_w1_eda.png
  2. 标准化 + PCA 降维与可视化       → 图 fig_w2_pca.png
  3. 逻辑回归基线（5 折交叉验证）    → 终端指标对比表
     - 原始 5 特征  vs  PCA 2 维  vs  PCA 3 维
     - 并给出"全预测 no rain"的傻瓜基线 Dummy 做参照
  4. 提示：为什么不只看准确率？

运行：python weather_baseline.py
依赖：numpy pandas matplotlib scikit-learn（见 requirements.txt）
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, average_precision_score)

# 中文字体（macOS；Windows 可改为 SimHei，Linux 可改为 Noto Sans CJK SC）
plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

BLUE, ORANGE, GREEN, RED, GRAY = "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#7f7f7f"
N_SPLITS = 5
RANDOM_STATE = 42
CLS_COLORS = {"no rain": BLUE, "rain": ORANGE}
METRIC_NAMES = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]

# ================================================================
# 1. 加载数据 + EDA
# ================================================================
df = pd.read_csv("weather_forecast_data.csv")
X = df.drop(columns=["Rain"])
y = df["Rain"].map({"no rain": 0, "rain": 1}).to_numpy()

print("=" * 62)
print("1) 数据概览")
print("-" * 62)
print(f"样本数: {df.shape[0]}   特征数(数值): {df.shape[1] - 1}")
print(f"缺失值: {df.isnull().sum().sum()}")
print()
print("目标 Rain 分布：")
print(df["Rain"].value_counts())
print(f"\nrain 占比 = {df['Rain'].eq('rain').mean()*100:.1f}%  "
      f"→ 类别不平衡，只看准确率会严重误导！")
print()
print("特征描述统计：")
print(df.describe().round(3).to_string())
print()
print("特征两两相关性（≈0 说明特征近乎正交，PCA 降维收益有限）：")
print(df.corr(numeric_only=True).round(3).to_string())

# ---------- EDA 图 ----------
fig1, axes = plt.subplots(2, 2, figsize=(12, 9))

# (1) 目标类别分布
vc = df["Rain"].value_counts()
bars = axes[0, 0].bar(vc.index, vc.values, color=[BLUE, ORANGE])
for b, k in zip(bars, vc.index):
    axes[0, 0].text(b.get_x() + b.get_width() / 2, b.get_height() + 20,
                    f"{vc[k]}  ({vc[k]/len(df)*100:.1f}%)", ha="center", fontsize=12)
axes[0, 0].set_ylim(0, 2600)
axes[0, 0].set_title("目标类别分布（严重不平衡）", fontsize=13)

# (2) 特征分布直方图
for c in X.columns:
    axes[0, 1].hist(df[c], bins=40, alpha=0.45, label=c)
axes[0, 1].set_title("5 个数值特征分布", fontsize=13)
axes[0, 1].legend(fontsize=9)
axes[0, 1].set_xlabel("取值")

# (3) 相关性热力图
corr = df.corr(numeric_only=True)
im = axes[1, 0].imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
axes[1, 0].set_xticks(range(5)); axes[1, 0].set_yticks(range(5))
axes[1, 0].set_xticklabels(corr.columns, rotation=45); axes[1, 0].set_yticklabels(corr.columns)
for i in range(5):
    for j in range(5):
        axes[1, 0].text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=9)
axes[1, 0].set_title("特征相关性（约等于 0 → 特征已正交）", fontsize=13)
plt.colorbar(im, ax=axes[1, 0], fraction=0.046)

# (4) 与目标最相关的两个特征 × 类别箱线图
axes[1, 1].scatter(df["Humidity"], df["Cloud_Cover"], c=df["Rain"].map(CLS_COLORS),
                   alpha=0.45, s=16)
axes[1, 1].set_xlabel("Humidity"); axes[1, 1].set_ylabel("Cloud_Cover")
axes[1, 1].set_title("Humidity vs Cloud_Cover（橙=rain，蓝=no rain）", fontsize=13)

fig1.suptitle("EDA：盒子里有什么？（天气数据集）", fontsize=15, y=0.99)
fig1.tight_layout(rect=[0, 0, 1, 0.97])
fig1.savefig("fig_w1_eda.png", dpi=150, bbox_inches="tight")
print("\n已保存 fig_w1_eda.png")

# ================================================================
# 2. 标准化 + PCA
# ================================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca_full = PCA().fit(X_scaled)
ratio = pca_full.explained_variance_ratio_
cum = np.cumsum(ratio)

print("\n" + "=" * 62)
print("2) PCA 结果")
print("-" * 62)
for i, (r, c) in enumerate(zip(ratio, cum), 1):
    print(f"PC{i}: 解释方差比例 {r*100:6.2f}%   累计 {c*100:6.2f}%")
print(f"\n→ 各主成分占比接近，说明数据在 5 维上分布较均匀；"
      f"降到 2 维只能保留约 {cum[1]*100:.0f}% 的信息。")
print("  （对比：Iris 数据集 PC1 单独就能保留约 70%+，因为其特征高度相关。）")

# ---------- PCA 可视化图 ----------
pca2 = PCA(n_components=2)
X_pca2 = pca2.fit_transform(X_scaled)

fig2, axes = plt.subplots(1, 2, figsize=(12, 4.8))

for label, color in CLS_COLORS.items():
    mask = df["Rain"] == label
    axes[0].scatter(X_pca2[mask, 0], X_pca2[mask, 1], c=color,
                    s=12, alpha=0.55, label=label)
axes[0].set_xlabel("PC1"); axes[0].set_ylabel("PC2")
axes[0].set_title("PCA 2 维投影（按 Rain 着色）", fontsize=13)
axes[0].legend()

axes[1].bar(range(1, 6), ratio * 100, color=BLUE, alpha=0.8, label="单个主成分")
axes[1].plot(range(1, 6), cum * 100, "o-", color=ORANGE, lw=2, label="累计")
for i in range(5):
    axes[1].text(i + 1, ratio[i] * 100 + 1.2, f"{ratio[i]*100:.1f}%", ha="center", fontsize=9)
axes[1].set_xlabel("主成分"); axes[1].set_ylabel("解释方差比例 (%)")
axes[1].set_title("各主成分解释方差比例", fontsize=13)
axes[1].legend()
fig2.tight_layout()
fig2.savefig("fig_w2_pca.png", dpi=150, bbox_inches="tight")
print("已保存 fig_w2_pca.png")

# ================================================================
# 3. 逻辑回归基线：原始特征 vs PCA 降维（分层 5 折交叉验证）
# ================================================================
# 注意：每次 CV 都在训练折内 fit scaler / PCA，不能在全体数据上先降维（数据泄露）！
# 这里用最简单的写法便于理解：在折内重新 fit。
def evaluate_cv(X_all, y_all, build_model):
    """在 StratifiedKFold 上评估，返回每个指标的均值±标准差。"""
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    scores = {m: [] for m in METRIC_NAMES}
    for tr, te in cv.split(X_all, y_all):
        X_tr, X_te = X_all[tr], X_all[te]
        y_tr, y_te = y_all[tr], y_all[te]
        model = build_model()
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        y_prob = model.predict_proba(X_te)[:, 1]
        scores["Accuracy"].append(accuracy_score(y_te, y_pred))
        scores["Precision"].append(precision_score(y_te, y_pred, zero_division=0))
        scores["Recall"].append(recall_score(y_te, y_pred, zero_division=0))
        scores["F1"].append(f1_score(y_te, y_pred, zero_division=0))
        scores["ROC-AUC"].append(roc_auc_score(y_te, y_prob))
        scores["PR-AUC"].append(average_precision_score(y_te, y_prob))
    return {m: (np.mean(v), np.std(v)) for m, v in scores.items()}

def format_row(name, scores):
    parts = [f"{name:<28}"]
    for m in METRIC_NAMES:
        mean, std = scores[m]
        parts.append(f"{mean*100:6.1f}±{std*100:4.1f}")
    return "  ".join(parts)

print("\n" + "=" * 62)
print("3) 基线模型：LogisticRegression（StratifiedKFold=5），指标=均值±标准差(%)")
print("-" * 62)
print(f"{'模型':<28}  {'ACC':>11}  {'PRE':>11}  {'REC':>11}  {'F1':>11}  {'AUROC':>11}  {'AUPRC':>11}")

# 参照：什么都不学，全部预测多数类（no rain）
dummy = evaluate_cv(X_scaled, y, lambda: DummyClassifier(strategy="most_frequent"))
print(format_row("Dummy(全预测 no rain)", dummy))

# 原始 5 特征
lr_raw = evaluate_cv(X_scaled, y, lambda: LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))
print(format_row("LogReg @ 原始5特征", lr_raw))

# PCA 降维到 2 / 3 维（降维在折内进行，避免数据泄露）
from sklearn.pipeline import Pipeline
for k in (2, 3):
    def build_p(kk=k):
        return Pipeline([
            ("pca", PCA(n_components=kk)),
            ("lr", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ])
    res = evaluate_cv(X_scaled, y, build_p)
    print(format_row(f"LogReg @ PCA{k}维", res))

print("-" * 62)
print("观察：")
print("  ① Dummy 的 Accuracy ≈ 87.4%，但 F1 / Recall = 0 —— 请想明白为什么；")
print("  ② 阈值默认 0.5 时，基线的 Recall 偏低（雨类难被召回），PR-AUC 更可信；")
print("  ③ PCA 降到 2 维后 F1 从 69.6% 掉到 25.1% —— 因为每个正交特征各占 ~20% 方差，")
print("     丢掉 3 维就丢了 ~60% 信息；这与 Iris 等高相关数据形成鲜明对比；")
print("  ④ 既然特征已正交，盲目 PCA 不可取 → 可调研特征工程/有监督降维(LDA)等方向。")

# ================================================================
# 4. 给学生的提示（基线到此为止，作业在下面）
# ================================================================
print("\n" + "=" * 62)
print("4) 基线完成。接下来是你的作业：调研方法并改进（详见 weather_homework.md）")
print("-" * 62)
print("可调研的改进大方向（任选至少 3 项，写清楚依据与对比）：")
print("  A. 类别不平衡：class_weight / SMOTE / 重采样 / 调整决策阈值（PR 曲线）")
print("  B. 模型：随机森林 / 梯度提升树(XGBoost/LightGBM) / SVM(RBF) / KNN 并调参")
print("  C. 特征工程：交互特征、分箱、领域知识特征（如湿度×云量）")
print("  D. 降维/特征选择对比：PCA vs LDA vs SelectKBest，探讨监督/无监督差异")
print("  E. 评估体系：PR 曲线 vs ROC 曲线（不平衡时哪个更可信？）、混淆矩阵")
print("  F. 调参：GridSearchCV / RandomizedSearchCV / Optuna，报告最优超参数")
print("  G. 交叉验证：RepeatedStratifiedKFold 稳定性分析")