"""
Step 1 — Data Merging & Preprocessing
"""

import glob
import os
import re
import pandas as pd
import numpy as np

# Config
DATA_DIR   = os.path.dirname(os.path.abspath(__file__))
OUT_DIR    = os.path.join(DATA_DIR, "data")
os.makedirs(OUT_DIR, exist_ok=True)

# Sleep stage → ordinal score (higher = deeper / more restorative)
STAGE_MAP = {"REM": 1, "Core": 2, "Deep": 3}

# Load all nightly files
pattern = os.path.join(DATA_DIR, "data", "sleep_logs", "sleep_2026-*.csv")
files = sorted([
    f for f in glob.glob(pattern)
    if "_empty" not in f and "副本" not in f and "_filled" not in f
])

if not files:
    raise FileNotFoundError(f"No sleep CSV files found matching: {pattern}")

print(f"Found {len(files)} night(s):")

frames = []
for path in files:
    fname = os.path.basename(path)

    # Extract night date from filename (the date you went to bed)
    m = re.search(r"sleep_(\d{4}-\d{2}-\d{2})", fname)
    if not m:
        print(f"  Skipping {fname} — cannot parse date from filename")
        continue
    night_date = m.group(1)

    df = pd.read_csv(path)

    # Parse timestamp; drop rows where it fails (safety for future files)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    bad = df["timestamp"].isna().sum()
    if bad:
        print(f"  WARNING {fname}: {bad} row(s) with unparseable timestamp — dropped")
    df = df.dropna(subset=["timestamp"]).copy()

    if df.empty:
        print(f"  Skipping {fname} — no valid rows after parsing")
        continue

    # Add metadata columns
    df.insert(0, "night_date", night_date)
    df["elapsed_min"] = (
        (df["timestamp"] - df["timestamp"].iloc[0])
        .dt.total_seconds() / 60
    ).round(1)

    # Numeric encoding of sleep stage
    df["stage_numeric"] = df["sleep_stage"].map(STAGE_MAP)
    unmapped = df["stage_numeric"].isna().sum()
    if unmapped:
        print(f"  WARNING {fname}: {unmapped} unknown sleep_stage value(s)")

    print(f"  {fname}: {len(df)} rows | "
          f"{df['timestamp'].iloc[0].strftime('%H:%M')} – "
          f"{df['timestamp'].iloc[-1].strftime('%H:%M')} | "
          f"sleep_time={df['sleep time'].iloc[0]}h")

    frames.append(df)

# Merge
all_df = pd.concat(frames, ignore_index=True)
all_df = all_df.sort_values(["night_date", "timestamp"]).reset_index(drop=True)

print(f"\nMerged dataset: {len(all_df)} rows across {all_df['night_date'].nunique()} nights")

# Save raw merged file
raw_path = os.path.join(OUT_DIR, "all_nights_raw.csv")
all_df.to_csv(raw_path, index=False)
print(f"Saved: {raw_path}")

# Per-night summary
env_cols = [
    "outdoor_temperature", "outdoor_humidity", "precipitation",
    "wind_speed", "inside_temperature", "inside_humidity"
]

def pct_stage(series, stage):
    return round((series == stage).sum() / len(series) * 100, 1)

summaries = []
for night, grp in all_df.groupby("night_date"):
    row = {"night_date": night}

    # Total sleep duration (from 'sleep time' column — consistent per night)
    row["sleep_duration_h"] = grp["sleep time"].iloc[0]

    # Mean sleep score and stage proportions
    row["mean_sleep_score"]  = round(grp["sleep_score"].mean(), 3)
    row["pct_deep"]          = pct_stage(grp["sleep_stage"], "Deep")
    row["pct_core"]          = pct_stage(grp["sleep_stage"], "Core")
    row["pct_rem"]           = pct_stage(grp["sleep_stage"], "REM")
    row["mean_stage_numeric"] = round(grp["stage_numeric"].mean(), 3)

    # Environmental means and ranges
    for col in env_cols:
        vals = grp[col].dropna()
        row[f"{col}_mean"] = round(vals.mean(), 3)
        row[f"{col}_std"]  = round(vals.std(), 3)
        row[f"{col}_min"]  = round(vals.min(), 3)
        row[f"{col}_max"]  = round(vals.max(), 3)

    summaries.append(row)

summary_df = pd.DataFrame(summaries)
summary_df["night_date"] = pd.to_datetime(summary_df["night_date"])
summary_df = summary_df.sort_values("night_date").reset_index(drop=True)

# Print summary table
print("\n── Nightly Summary ──────────────────────────────────────────────────────")
display_cols = [
    "night_date", "sleep_duration_h", "mean_sleep_score",
    "pct_deep", "pct_core", "pct_rem",
    "inside_temperature_mean", "inside_humidity_mean",
    "outdoor_temperature_mean"
]
print(summary_df[display_cols].to_string(index=False))

summary_path = os.path.join(OUT_DIR, "nightly_summary.csv")
summary_df.to_csv(summary_path, index=False)
print(f"\nSaved: {summary_path}")
print("\nStep 1 complete.")
