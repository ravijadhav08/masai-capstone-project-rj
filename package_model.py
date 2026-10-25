"""Package the final delay classifier for reuse.

Reads outputs/part_c_final_config.csv, rebuilds Part C's stratified split,
fits the final Pipeline on the training rows, saves models/delay_model.pkl
plus the three companion CSV files in models/, then reloads the pickle and
verifies it produces identical test probabilities.
"""
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.shipment_prep import FEATURES, clean_shipments

os.makedirs("models", exist_ok=True)

config = pd.read_csv("outputs/part_c_final_config.csv", dtype=str).iloc[0]
selector_k = int(config["selector_k"])
n_estimators = int(config["n_estimators"])
max_depth = None if config["max_depth"] == "unlimited" else int(config["max_depth"])
min_samples_leaf = int(config["min_samples_leaf"])
threshold = float(config["threshold"])

# ---- rebuild Part C's stratified split with the same settings ----
model_df = pd.read_csv("data/modelling_table.csv")
X = model_df[FEATURES]
y = model_df["is_delayed"]
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

final_pipeline = Pipeline([
    ("selector", SelectKBest(f_classif, k=selector_k)),
    ("model", RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
    )),
]).fit(X_train, y_train)

joblib.dump(final_pipeline, "models/delay_model.pkl")

# ---- companion files ----
pd.DataFrame({"threshold": [threshold]}).to_csv("models/delay_threshold.csv", index=False)
pd.DataFrame({"feature": FEATURES}).to_csv("models/feature_order.csv", index=False)

raw = pd.read_csv("data/raw_shipments.csv")
_, medians = clean_shipments(raw)
median_rows = [(c, round(medians[c], 4)) for c in ["distance_km", "parcel_weight_kg", "declared_value_inr"]]
pd.DataFrame(median_rows, columns=["column", "median"], dtype=object).to_csv(
    "models/impute_values.csv", index=False
)

# ---- reload check: identical probabilities from disk ----
reloaded = joblib.load("models/delay_model.pkl")
proba_memory = final_pipeline.predict_proba(X_test)[:, 1]
proba_reloaded = reloaded.predict_proba(X_test)[:, 1]
max_diff = float(np.max(np.abs(proba_memory - proba_reloaded)))

check_rows = [
    ("test_rows_scored", len(X_test)),
    ("reload_max_abs_probability_diff", max_diff),
    ("threshold", threshold),
]
pd.DataFrame(check_rows, columns=["metric", "value"], dtype=object).to_csv(
    "outputs/part_d_reload_check.csv", index=False
)

print("Part D packaging done: test rows =", len(X_test), "| max proba diff =", max_diff)
