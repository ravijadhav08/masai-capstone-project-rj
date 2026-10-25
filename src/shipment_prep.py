"""Shared constants and cleaning/feature functions for the Delay Intelligence pipeline.

Used by part_a_clean_eda.py (historical parcels) and predict_delay.py (new parcels)
so that training and scoring always apply the identical preprocessing.
"""
import numpy as np
import pandas as pd

PROMISED_HOURS = {"Express": 36, "Standard": 60, "Economy": 96}
FEATURES = [
    "promised_hours", "distance_km", "parcel_weight_kg", "log_declared_value", "n_stops",
    "rain_flag", "sale_event_flag", "is_cod", "is_weekend", "dispatch_hour",
    "hub_tier", "hub_daily_capacity",
    "label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no",
]
NOISE_COLUMNS = ["label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no"]

FILL_COLUMNS = ["distance_km", "parcel_weight_kg", "declared_value_inr"]


def clean_shipments(raw, medians=None):
    """Clean a raw shipments export.

    Steps, in order:
    1. drop exact duplicate rows;
    2. service_type: strip whitespace and title-case (Express / Standard / Economy);
    3. payment_mode: missing -> "Unknown";
    4. declared_value: strip rupee sign and thousands commas, convert to a number
       stored in declared_value_inr, drop the text column;
    5. dispatch_time: convert to datetime;
    6. parcel_weight_kg above 100 kg is a scanner-fault sentinel -> missing;
    7. fill missing distance_km, parcel_weight_kg and declared_value_inr with the
       column median (computed here if medians is None, else taken from medians).

    Returns (clean_df, medians); medians has exactly the keys distance_km,
    parcel_weight_kg, declared_value_inr with float values.
    """
    df = raw.copy()

    # 1. exact duplicates
    df = df.drop_duplicates().reset_index(drop=True)

    # 2. service_type spelling
    df["service_type"] = df["service_type"].str.strip().str.title()

    # 3. payment_mode placeholder
    df["payment_mode"] = df["payment_mode"].fillna("Unknown")

    # 4. declared_value text -> number
    inr_text = (df["declared_value"].str.replace("\u20b9", "", regex=False)
                .str.replace(",", "", regex=False))
    df["declared_value_inr"] = pd.to_numeric(inr_text, errors="coerce")
    df = df.drop(columns=["declared_value"])

    # 5. dispatch_time -> datetime
    df["dispatch_time"] = pd.to_datetime(df["dispatch_time"])

    # 6. sentinel weights -> missing (network accepts parcels up to 40 kg)
    df.loc[df["parcel_weight_kg"] > 100, "parcel_weight_kg"] = np.nan

    # 7. median fill
    if medians is None:
        medians = {c: float(df[c].median()) for c in FILL_COLUMNS}
    else:
        medians = {c: float(medians[c]) for c in FILL_COLUMNS}
    for c in FILL_COLUMNS:
        df[c] = df[c].fillna(medians[c])

    return df, medians


def build_features(clean, hubs):
    """Attach hub columns (left merge on hub_id) and build model features.

    Adds hub_city, region, hub_tier, hub_daily_capacity plus promised_hours,
    is_cod, is_weekend, dispatch_hour and log_declared_value. Never touches
    delivery_hours, so it also works on parcels that have not been delivered.
    """
    df = clean.merge(hubs, on="hub_id", how="left")
    df["promised_hours"] = df["service_type"].map(PROMISED_HOURS).astype(int)
    df["is_cod"] = (df["payment_mode"] == "COD").astype(int)
    df["is_weekend"] = (df["dispatch_time"].dt.dayofweek >= 5).astype(int)
    df["dispatch_hour"] = df["dispatch_time"].dt.hour.astype(int)
    df["log_declared_value"] = np.log(df["declared_value_inr"])
    return df
