"""Part D - Hub Segments.

Profiles the 24 hubs from data/modelling_table.csv, segments them with
K-Means on four scaled profile columns, measures cluster separation for
K = 2..5, names the three business segments by delay rate and lists the
hubs whose operating segment contradicts their tier label.
"""
import os
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

os.makedirs("outputs", exist_ok=True)

model_df = pd.read_csv("data/modelling_table.csv")
hubs = pd.read_csv("data/hubs.csv")

PROFILE_COLS = ["mean_delivery_hours", "delay_rate", "mean_distance_km", "mean_n_stops"]

# ---- task 1: hub profile (groupby returns hubs in ascending hub_id order) ----
profile = (
    model_df.groupby("hub_id")
    .agg(
        shipments=("delivery_hours", "size"),
        mean_delivery_hours=("delivery_hours", "mean"),
        delay_rate=("is_delayed", "mean"),
        mean_distance_km=("distance_km", "mean"),
        mean_n_stops=("n_stops", "mean"),
    )
    .reset_index()
)
profile_out = profile.copy()
for c in PROFILE_COLS:
    profile_out[c] = profile_out[c].round(4)
profile_out.to_csv("outputs/hub_profile.csv", index=False)

# ---- tasks 2 + 3: K-Means on the scaled profile matrix, separation evidence ----
scaled = StandardScaler().fit_transform(profile[PROFILE_COLS])
pairs = list(combinations(range(len(profile)), 2))
pair_dists = np.array([np.linalg.norm(scaled[i] - scaled[j]) for i, j in pairs])


def separation(labels):
    same = [d for (i, j), d in zip(pairs, pair_dists) if labels[i] == labels[j]]
    diff = [d for (i, j), d in zip(pairs, pair_dists) if labels[i] != labels[j]]
    sizes = sorted(np.bincount(labels), reverse=True)
    return np.mean(diff) / np.mean(same), "-".join(str(s) for s in sizes)


separation_rows = []
kmeans_fit = {}
for k in [2, 3, 4, 5]:
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(scaled)
    kmeans_fit[k] = km
    ratio, sizes = separation(km.labels_)
    separation_rows.append({"K": k, "across_over_within": round(float(ratio), 4), "cluster_sizes": sizes})
pd.DataFrame(separation_rows).to_csv("outputs/part_d_separation.csv", index=False)

# ---- task 4: segment names by cluster mean delay rate ----
km3 = kmeans_fit[3]
cluster_delay = pd.Series(profile["delay_rate"].groupby(km3.labels_).mean())
ordered_clusters = cluster_delay.sort_values().index.tolist()  # lowest delay first
SEGMENT_NAMES = ["Standard", "Watch", "Escalate"]
cluster_to_segment = {c: SEGMENT_NAMES[i] for i, c in enumerate(ordered_clusters)}

segments = profile[["hub_id", "delay_rate"]].copy()
segments["segment"] = [cluster_to_segment[c] for c in km3.labels_]
segments = segments.merge(hubs[["hub_id", "hub_city", "hub_tier"]], on="hub_id", how="left")
segments[["hub_id", "segment", "hub_city", "hub_tier"]].to_csv(
    "outputs/hub_segments.csv", index=False
)

seg_profile = (
    segments.merge(profile, on="hub_id")
    .groupby("segment")
    .agg(
        hubs=("hub_id", "size"),
        mean_delivery_hours=("mean_delivery_hours", "mean"),
        delay_rate=("delay_rate_y", "mean"),
        mean_distance_km=("mean_distance_km", "mean"),
        mean_n_stops=("mean_n_stops", "mean"),
    )
    .reindex(SEGMENT_NAMES)
    .reset_index()
)
for c in PROFILE_COLS:
    seg_profile[c] = seg_profile[c].round(4)
seg_profile.to_csv("outputs/segment_profile.csv", index=False)

TIER_TO_SEGMENT = {1: "Standard", 2: "Watch", 3: "Escalate"}
mismatch = segments[segments["segment"] != segments["hub_tier"].map(TIER_TO_SEGMENT)]
mismatch[["hub_id", "segment", "hub_city", "hub_tier"]].to_csv(
    "outputs/tier_vs_segment_mismatch.csv", index=False
)

print("Part D done: segments", dict(seg_profile["hubs"]), "| mismatches:", len(mismatch))
