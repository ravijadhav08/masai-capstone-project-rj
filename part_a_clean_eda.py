"""Part A - Data Foundation and Delay Evidence.

Reads data/raw_shipments.csv and data/hubs.csv, cleans the export with
src.shipment_prep.clean_shipments / build_features, writes
data/modelling_table.csv and the Part A evidence files in outputs/.
"""
import os

import numpy as np
import pandas as pd

from src.shipment_prep import FEATURES, build_features, clean_shipments

os.makedirs("outputs", exist_ok=True)

raw = pd.read_csv("data/raw_shipments.csv")
hubs = pd.read_csv("data/hubs.csv")

# ---- "before" state: after removing duplicates, before any filling ----
raw_rows = len(raw)
deduped = raw.drop_duplicates().reset_index(drop=True)
rows_after_dedup = len(deduped)

missing_distance_km_before = int(deduped["distance_km"].isna().sum())
missing_parcel_weight_before = int(deduped["parcel_weight_kg"].isna().sum())
sentinel_weight_rows = int((deduped["parcel_weight_kg"] > 100).sum())
missing_declared_value_before = int(deduped["declared_value"].isna().sum())
missing_payment_mode_before = int(deduped["payment_mode"].isna().sum())
service_type_spellings_before = int(deduped["service_type"].nunique())

# ---- cleaning and features ----
clean_df, medians = clean_shipments(raw)
service_type_spellings_after = int(clean_df["service_type"].nunique())

model_df = build_features(clean_df, hubs)
rows_after_hub_merge = len(model_df)

model_df["is_delayed"] = (model_df["delivery_hours"] > model_df["promised_hours"]).astype(int)
model_df.to_csv("data/modelling_table.csv", index=False)

# ---- checks file ----
nulls_in_features_and_target = int(
    model_df[FEATURES + ["delivery_hours", "is_delayed"]].isna().sum().sum()
)
delayed_count = int(model_df["is_delayed"].sum())
delay_rate = delayed_count / len(model_df)

declared_value_mean_inr = float(model_df["declared_value_inr"].mean())
declared_value_median_inr = float(model_df["declared_value_inr"].median())
log_declared_value_mean = float(model_df["log_declared_value"].mean())
log_declared_value_median = float(model_df["log_declared_value"].median())

hours_q1 = float(model_df["delivery_hours"].quantile(0.25))
hours_q3 = float(model_df["delivery_hours"].quantile(0.75))
iqr = hours_q3 - hours_q1
hours_lower_bound = hours_q1 - 1.5 * iqr
hours_upper_bound = hours_q3 + 1.5 * iqr
hours_outlier_count = int(
    ((model_df["delivery_hours"] < hours_lower_bound)
     | (model_df["delivery_hours"] > hours_upper_bound)).sum()
)

checks = [
    ("raw_rows", raw_rows),
    ("duplicate_rows_removed", raw_rows - rows_after_dedup),
    ("rows_after_dedup", rows_after_dedup),
    ("missing_distance_km_before", missing_distance_km_before),
    ("missing_parcel_weight_before", missing_parcel_weight_before),
    ("sentinel_weight_rows", sentinel_weight_rows),
    ("missing_declared_value_before", missing_declared_value_before),
    ("missing_payment_mode_before", missing_payment_mode_before),
    ("service_type_spellings_before", service_type_spellings_before),
    ("service_type_spellings_after", service_type_spellings_after),
    ("median_distance_km", round(medians["distance_km"], 4)),
    ("median_parcel_weight_kg", round(medians["parcel_weight_kg"], 4)),
    ("median_declared_value_inr", round(medians["declared_value_inr"], 4)),
    ("rows_after_hub_merge", rows_after_hub_merge),
    ("nulls_in_features_and_target", nulls_in_features_and_target),
    ("delayed_count", delayed_count),
    ("delay_rate", round(delay_rate, 4)),
    ("declared_value_mean_inr", round(declared_value_mean_inr, 4)),
    ("declared_value_median_inr", round(declared_value_median_inr, 4)),
    ("log_declared_value_mean", round(log_declared_value_mean, 4)),
    ("log_declared_value_median", round(log_declared_value_median, 4)),
    ("hours_q1", round(hours_q1, 4)),
    ("hours_q3", round(hours_q3, 4)),
    ("hours_lower_bound", round(hours_lower_bound, 4)),
    ("hours_upper_bound", round(hours_upper_bound, 4)),
    ("hours_outlier_count", hours_outlier_count),
]
checks_df = pd.DataFrame(checks, columns=["metric", "value"], dtype=object)
checks_df.to_csv("outputs/part_a_checks.csv", index=False)

# ---- delay evidence by group ----
band_labels = ["0-200", "201-400", "401-600", "601+"]
model_df["distance_band"] = pd.cut(
    model_df["distance_km"], bins=[0, 200, 400, 600, 2000], labels=band_labels, right=True
)


def group_rows(dimension, keys):
    rows = []
    for key in keys:
        if dimension == "distance_band":
            sub = model_df[model_df["distance_band"] == key]
        else:
            sub = model_df[model_df[dimension] == key]
        shipments = len(sub)
        delayed = int(sub["is_delayed"].sum())
        rows.append({
            "dimension": dimension,
            "group": str(key),
            "shipments": shipments,
            "delayed": delayed,
            "delay_rate": round(delayed / shipments, 4),
            "mean_hours": round(float(sub["delivery_hours"].mean()), 4),
        })
    return rows


delay_by_group = (
    group_rows("service_type", sorted(model_df["service_type"].unique()))
    + group_rows("hub_tier", sorted(model_df["hub_tier"].unique()))
    + group_rows("distance_band", band_labels)
)
pd.DataFrame(delay_by_group).to_csv("outputs/part_a_delay_by_group.csv", index=False)

# ---- correlations ----
corr = model_df.corr(numeric_only=True).round(4)
corr.index.name = "column"
corr.reset_index().to_csv("outputs/part_a_correlations.csv", index=False)

print("Part A done:", len(model_df), "rows;", delayed_count, "delayed")
