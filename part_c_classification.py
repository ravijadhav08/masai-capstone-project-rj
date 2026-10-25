"""Part C - Delay Classifier and Alert Threshold.

Reads data/modelling_table.csv, builds a stratified 840/180/180 split,
compares a blind baseline with three classifiers, measures which columns
carry signal, tunes the forest on the 7 best columns, picks the alert
threshold from the operations rule and evaluates the final configuration
exactly once on the test rows.
"""
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.shipment_prep import FEATURES, NOISE_COLUMNS

os.makedirs("outputs", exist_ok=True)

model_df = pd.read_csv("data/modelling_table.csv")
X = model_df[FEATURES]
y = model_df["is_delayed"]

# ---- task 1: stratified two-step split ----
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

TRAIN_DELAYED = int(y_train.sum())
VAL_DELAYED = int(y_val.sum())
TEST_DELAYED = int(y_test.sum())
assert (TRAIN_DELAYED, VAL_DELAYED, TEST_DELAYED) == (195, 42, 41), (
    TRAIN_DELAYED, VAL_DELAYED, TEST_DELAYED
)


def safe_div(num, den):
    return num / den if den else 0.0


def clf_row(name, y_true, y_pred, precision_forced=None):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    if precision_forced is not None:
        precision = precision_forced
    else:
        precision = round(float(safe_div(tp, tp + fp)), 4)
    return {
        "model": name,
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": precision,
        "recall": round(float(safe_div(tp, tp + fn)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
    }


# ---- task 2: blind baseline + three classifiers (validation, 0.5 rule) ----
model_rows = [clf_row("blind_all_on_time", y_val, np.zeros(len(y_val), dtype=int),
                      precision_forced=0)]

log_reg = Pipeline([("scaler", StandardScaler()),
                    ("model", LogisticRegression())]).fit(X_train, y_train)
model_rows.append(clf_row("logistic_regression", y_val, log_reg.predict(X_val)))

best_d, best_f1 = None, -1.0
for d in [2, 3, 4, 5, 6, 8, 10]:
    tree = DecisionTreeClassifier(criterion="gini", max_depth=d, random_state=42).fit(X_train, y_train)
    f1 = f1_score(y_val, tree.predict(X_val))
    if f1 > best_f1:
        best_d, best_f1 = d, f1
model_rows.append(clf_row(f"decision_tree_depth_{best_d}", y_val,
                          DecisionTreeClassifier(criterion="gini", max_depth=best_d,
                                                 random_state=42).fit(X_train, y_train).predict(X_val)))

rf_default = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
model_rows.append(clf_row("random_forest_default", y_val, rf_default.predict(X_val)))
pd.DataFrame(model_rows, dtype=object).to_csv("outputs/part_c_models.csv", index=False)

# ---- task 3: which columns carry signal ----
forest17 = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
rf_importance = pd.Series(forest17.feature_importances_, index=FEATURES)

selector_all = SelectKBest(f_classif, k="all").fit(X_train, y_train)
f_scores = pd.Series(selector_all.scores_, index=FEATURES)
f_rank = f_scores.rank(ascending=False, method="min").astype(int)

feature_scores = pd.DataFrame({
    "feature": FEATURES,
    "f_score": [round(float(f_scores[f]), 4) for f in FEATURES],
    "f_rank": [int(f_rank[f]) for f in FEATURES],
    "rf_importance": [round(float(rf_importance[f]), 4) for f in FEATURES],
    "is_noise": [f in NOISE_COLUMNS for f in FEATURES],
})
feature_scores.to_csv("outputs/part_c_feature_scores.csv", index=False)

selectk_rows = []
for k in [5, 7, 10, 12, 17]:
    pipe = Pipeline([
        ("selector", SelectKBest(f_classif, k=k)),
        ("model", RandomForestClassifier(n_estimators=100, random_state=42)),
    ]).fit(X_train, y_train)
    selectk_rows.append({"k": k, "val_f1": round(float(f1_score(y_val, pipe.predict(X_val))), 4)})
pd.DataFrame(selectk_rows).to_csv("outputs/part_c_selectk.csv", index=False)

# ---- task 4: tuning on the 7 selected columns (training rows only) ----
selector7 = SelectKBest(f_classif, k=7).fit(X_train, y_train)
X_train7 = selector7.transform(X_train)
X_val7 = selector7.transform(X_val)

param_grid = {
    "n_estimators": [50, 100, 150],
    "max_depth": [4, 6, 8],
    "min_samples_leaf": [1, 5, 10],
}
rf = RandomForestClassifier(random_state=42)

grid = GridSearchCV(rf, param_grid, cv=5).fit(X_train7, y_train)
grid_val_f1 = f1_score(y_val, grid.best_estimator_.predict(X_val7))

rand = RandomizedSearchCV(rf, param_grid, n_iter=9, cv=5, random_state=42).fit(X_train7, y_train)
rand_val_f1 = f1_score(y_val, rand.best_estimator_.predict(X_val7))

default7 = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train7, y_train)
default_val_f1 = f1_score(y_val, default7.predict(X_val7))

tuning_rows = [
    {
        "search": "grid_search", "candidates": 27, "cv_folds": 5, "total_fits": 135,
        "best_params": str(grid.best_params_),
        "best_cv_accuracy": round(float(grid.best_score_), 4),
        "val_f1": round(float(grid_val_f1), 4),
    },
    {
        "search": "randomized_search", "candidates": 9, "cv_folds": 5, "total_fits": 45,
        "best_params": str(rand.best_params_),
        "best_cv_accuracy": round(float(rand.best_score_), 4),
        "val_f1": round(float(rand_val_f1), 4),
    },
    {
        "search": "default_forest_same_7_features", "candidates": 1, "cv_folds": 0,
        "total_fits": 1, "best_params": "defaults", "best_cv_accuracy": "",
        "val_f1": round(float(default_val_f1), 4),
    },
]
tuning_df = pd.DataFrame(tuning_rows)
tuning_df.to_csv("outputs/part_c_tuning.csv", index=False)

# ---- task 5: choose the final configuration ----
tie_order = {"default_forest_same_7_features": 0, "grid_search": 1, "randomized_search": 2}
winner = tuning_df.sort_values(
    ["val_f1", "search"],
    ascending=[False, True],
    key=lambda s: s.map(tie_order) if s.name == "search" else s,
).iloc[0]
winner_name = winner["search"]
if winner_name == "default_forest_same_7_features":
    chosen_params = {}
elif winner_name == "grid_search":
    chosen_params = dict(grid.best_params_)
else:
    chosen_params = dict(rand.best_params_)

final_model = Pipeline([
    ("selector", SelectKBest(f_classif, k=7)),
    ("model", RandomForestClassifier(random_state=42, **chosen_params)),
]).fit(X_train, y_train)

# ---- task 6: alert threshold from validation probabilities ----
val_proba = final_model.predict_proba(X_val)[:, 1]
threshold_rows = []
for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    y_pred = (val_proba >= t).astype(int)
    threshold_rows.append({"threshold": t, **{
        k: v for k, v in clf_row("", y_val, y_pred).items() if k != "model"
    }})
thresholds_df = pd.DataFrame(threshold_rows)
thresholds_df.to_csv("outputs/part_c_thresholds.csv", index=False)

eligible = thresholds_df[thresholds_df["recall"] >= 0.80]
chosen_threshold = float(eligible["threshold"].max()) if len(eligible) else 0.1

# ---- task 7: exactly one test evaluation ----
test_proba = final_model.predict_proba(X_test)[:, 1]
test_pred = (test_proba >= chosen_threshold).astype(int)
final_test_row = {"threshold": chosen_threshold, **{
    k: v for k, v in clf_row("", y_test, test_pred).items() if k != "model"
}}
pd.DataFrame([final_test_row]).to_csv("outputs/part_c_final_test.csv", index=False)

n_estimators = chosen_params.get("n_estimators", 100)
max_depth = chosen_params.get("max_depth", None)
min_samples_leaf = chosen_params.get("min_samples_leaf", 1)
final_config = [{
    "selector_k": 7,
    "n_estimators": n_estimators,
    "max_depth": max_depth if max_depth is not None else "unlimited",
    "min_samples_leaf": min_samples_leaf,
    "threshold": chosen_threshold,
    "chosen_from": winner_name,
}]
pd.DataFrame(final_config, dtype=object).to_csv("outputs/part_c_final_config.csv", index=False)

print("Part C done: tree depth =", best_d, "| winner =", winner_name,
      "| params =", chosen_params, "| threshold =", chosen_threshold)
