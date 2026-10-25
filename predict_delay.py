"""Score the 48 new parcels and turn them into an action list.

Loads data/new_shipments.csv, the packaged model in models/ and the hub
segments, reuses src.shipment_prep.clean_shipments / build_features so the
preprocessing is identical to training, then applies the Part C alert
threshold and the Part D segment actions.
"""
import os

import joblib
import numpy as np
import pandas as pd

from src.shipment_prep import build_features, clean_shipments

new_shipments = pd.read_csv("data/new_shipments.csv")
hubs = pd.read_csv("data/hubs.csv")
threshold = float(pd.read_csv("models/delay_threshold.csv")["threshold"].iloc[0])
feature_order = pd.read_csv("models/feature_order.csv")["feature"].tolist()
impute_values = pd.read_csv("models/impute_values.csv")
medians = dict(zip(impute_values["column"], impute_values["median"].astype(float)))
hub_segments = pd.read_csv("outputs/hub_segments.csv")
model = joblib.load("models/delay_model.pkl")

# identical preprocessing to training, with the stored medians
clean_new, _ = clean_shipments(new_shipments, medians)
featured = build_features(clean_new, hubs)
X_new = featured[feature_order]

proba = model.predict_proba(X_new)[:, 1]

actions = pd.DataFrame({
    "shipment_id": featured["shipment_id"],
    "hub_id": featured["hub_id"],
})
actions["delay_probability"] = np.round(proba, 4)
actions["flagged"] = (actions["delay_probability"] >= threshold).astype(int)
actions = actions.merge(hub_segments[["hub_id", "segment"]], on="hub_id", how="left")


def decide(row):
    if row["flagged"] == 1 and row["segment"] == "Escalate":
        return "Escalate to hub manager"
    if row["flagged"] == 1:
        return "Proactive customer call"
    return "No action"


actions["action"] = actions.apply(decide, axis=1)
actions = actions[["shipment_id", "hub_id", "segment", "delay_probability", "flagged", "action"]]
actions.to_csv("outputs/new_shipment_actions.csv", index=False)

n_flagged = int(actions["flagged"].sum())
n_escalate = int((actions["action"] == "Escalate to hub manager").sum())
print(f"predict_delay done: {len(actions)} parcels, {n_flagged} flagged, {n_escalate} escalations")
