"""Part B - Delivery-Hours Regression.

Reads data/modelling_table.csv, splits 840/180/180, demonstrates leakage-safe
scaling, compares a baseline, a depth-tuned decision tree, random forest and
gradient boosting, then scores the single best model once on the test rows.
"""
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from src.shipment_prep import FEATURES

os.makedirs("outputs", exist_ok=True)

model_df = pd.read_csv("data/modelling_table.csv")
X = model_df[FEATURES]
y = model_df["delivery_hours"]

# ---- task 1: two-step split ----
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# ---- task 2: scaling without leakage ----
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
train_scaled_max_abs_mean = float(np.abs(X_train_scaled.mean(axis=0)).max())
test_scaled_max_abs_mean = float(np.abs(X_test_scaled.mean(axis=0)).max())

checks = [
    ("train_rows", len(X_train)),
    ("val_rows", len(X_val)),
    ("test_rows", len(X_test)),
    ("train_scaled_max_abs_mean", round(train_scaled_max_abs_mean, 4)),
    ("test_scaled_max_abs_mean", round(test_scaled_max_abs_mean, 4)),
]
pd.DataFrame(checks, columns=["metric", "value"], dtype=object).to_csv(
    "outputs/part_b_checks.csv", index=False
)


def reg_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_true, y_pred)
    return mae, mse, rmse, r2


def metrics_row(model_name, split, y_true, y_pred):
    mae, mse, rmse, r2 = reg_metrics(y_true, y_pred)
    return {
        "model": model_name, "split": split,
        "MAE": round(mae, 4), "MSE": round(mse, 4),
        "RMSE": round(rmse, 4), "R2": round(r2, 4),
    }


# ---- task 3: mean baseline on validation ----
baseline_pred = np.full(len(y_val), y_train.mean())
metric_rows = [metrics_row("mean_baseline", "val", y_val, baseline_pred)]

# ---- task 4: tree-depth diagnostic ----
depth_results = []
best_depth, best_val_rmse = None, np.inf
for d in [1, 2, 3, 4, 6, 8, 12]:
    tree = DecisionTreeRegressor(max_depth=d, random_state=42).fit(X_train, y_train)
    tr_rmse = float(np.sqrt(mean_squared_error(y_train, tree.predict(X_train))))
    va_rmse = float(np.sqrt(mean_squared_error(y_val, tree.predict(X_val))))
    depth_results.append({"max_depth": d, "train_rmse": round(tr_rmse, 4), "val_rmse": round(va_rmse, 4)})
    if va_rmse < best_val_rmse:
        best_depth, best_val_rmse = d, va_rmse
unlimited = DecisionTreeRegressor(random_state=42).fit(X_train, y_train)
unl_tr = float(np.sqrt(mean_squared_error(y_train, unlimited.predict(X_train))))
unl_va = float(np.sqrt(mean_squared_error(y_val, unlimited.predict(X_val))))
depth_results.append({"max_depth": "unlimited", "train_rmse": round(unl_tr, 4), "val_rmse": round(unl_va, 4)})
if unl_va < best_val_rmse:
    best_depth, best_val_rmse = None, unl_va
pd.DataFrame(depth_results).to_csv("outputs/part_b_tree_depth.csv", index=False)

# ---- task 5: four candidate models (validation) ----
candidates = {
    "linear_regression": Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]),
    "decision_tree": DecisionTreeRegressor(max_depth=best_depth, random_state=42),
    "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "gradient_boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
}
fitted = {}
for name, model in candidates.items():
    model.fit(X_train, y_train)
    fitted[name] = model
    metric_rows.append(metrics_row(name, "val", y_val, model.predict(X_val)))

# ---- task 6: learning-rate experiment ----
lr_rows = []
for lr in [0.01, 0.1, 1.0]:
    gb = GradientBoostingRegressor(n_estimators=100, learning_rate=lr, random_state=42).fit(X_train, y_train)
    _, _, rmse, r2 = reg_metrics(y_val, gb.predict(X_val))
    lr_rows.append({"learning_rate": lr, "val_rmse": round(rmse, 4), "val_r2": round(r2, 4)})
pd.DataFrame(lr_rows).to_csv("outputs/part_b_gb_learning_rate.csv", index=False)

# ---- task 7: cross-validated comparison + single test scoring ----
val_rmse = {r["model"]: r["RMSE"] for r in metric_rows if r["split"] == "val" and r["model"] != "mean_baseline"}
ranked = sorted(val_rmse, key=val_rmse.get)
top_two = ranked[:2]

X_trainval = pd.concat([X_train, X_val])
y_trainval = pd.concat([y_train, y_val])
cv = KFold(n_splits=5, shuffle=True, random_state=42)
cv_rows = []
for name in top_two:
    scores = cross_val_score(fitted[name], X_trainval, y_trainval, cv=cv)
    cv_rows.append({
        "model": name,
        "mean_r2": round(float(scores.mean()), 4),
        "std_r2": round(float(scores.std()), 4),
    })
pd.DataFrame(cv_rows).to_csv("outputs/part_b_cv.csv", index=False)

final_name = ranked[0]
metric_rows.append(metrics_row(final_name, "test", y_test, fitted[final_name].predict(X_test)))
pd.DataFrame(metric_rows).to_csv("outputs/part_b_metrics.csv", index=False)

print("Part B done: best depth =", best_depth, "| final model =", final_name)
