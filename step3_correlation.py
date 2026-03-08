"""
Step 3 — Time Series Correlation Analysis
Project: Nocturnal Microclimate vs Sleep Architecture
------------------------------------------------------
Three levels of analysis:
  A. Nightly summary correlation matrix (Pearson + Spearman, n=9)
  B. Within-night lag cross-correlation  (env variable → sleep score, pooled)
  C. Per-night Spearman correlation table (inside_temp vs stage_numeric)

Outputs (saved to plots/):
  fig4_correlation_heatmap.png
  fig5_cross_correlation_lags.png
  fig6_pernight_correlation.png

Prints a stats summary to console.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from scipy.signal import correlate, correlation_lags

warnings.filterwarnings("ignore")

# ── Load ──────────────────────────────────────────────────────────────────────
BASE  = os.path.dirname(os.path.abspath(__file__))
PLOTS = os.path.join(BASE, "plots")
os.makedirs(PLOTS, exist_ok=True)

raw  = pd.read_csv(os.path.join(BASE, "data", "all_nights_raw.csv"))
summ = pd.read_csv(os.path.join(BASE, "data", "nightly_summary.csv"))
raw["timestamp"]   = pd.to_datetime(raw["timestamp"])
summ["night_date"] = pd.to_datetime(summ["night_date"])

STAGE_MAP   = {"REM": 1, "Core": 2, "Deep": 3}
raw["stage_numeric"] = raw["sleep_stage"].map(STAGE_MAP)

# ══════════════════════════════════════════════════════════════════════════════
# A. Nightly-level correlation matrix
# ══════════════════════════════════════════════════════════════════════════════
env_cols = [
    "inside_temperature_mean", "inside_humidity_mean",
    "outdoor_temperature_mean", "outdoor_humidity_mean",
    "wind_speed_mean",
]
sleep_cols = [
    "sleep_duration_h", "mean_sleep_score",
    "pct_deep", "pct_core", "pct_rem",
]
all_cols = env_cols + sleep_cols

labels = {
    "inside_temperature_mean":  "Indoor Temp",
    "inside_humidity_mean":     "Indoor Humid.",
    "outdoor_temperature_mean": "Outdoor Temp",
    "outdoor_humidity_mean":    "Outdoor Humid.",
    "wind_speed_mean":          "Wind Speed",
    "sleep_duration_h":         "Sleep Duration",
    "mean_sleep_score":         "Sleep Score",
    "pct_deep":                 "Deep %",
    "pct_core":                 "Core %",
    "pct_rem":                  "REM %",
}

data_mat = summ[all_cols].copy()
n = len(data_mat)

# Compute Pearson and Spearman matrices with p-values
def corr_matrix(df, method="pearson"):
    cols = df.columns.tolist()
    r_mat = pd.DataFrame(np.eye(len(cols)), index=cols, columns=cols)
    p_mat = pd.DataFrame(np.zeros((len(cols), len(cols))), index=cols, columns=cols)
    for i, c1 in enumerate(cols):
        for j, c2 in enumerate(cols):
            if i == j:
                continue
            if method == "pearson":
                r, p = stats.pearsonr(df[c1], df[c2])
            else:
                r, p = stats.spearmanr(df[c1], df[c2])
            r_mat.loc[c1, c2] = r
            p_mat.loc[c1, c2] = p
    return r_mat, p_mat

r_pear, p_pear = corr_matrix(data_mat, "pearson")
r_spear, p_spear = corr_matrix(data_mat, "spearman")

# Print key results
print("=" * 60)
print("A. NIGHTLY CORRELATION  (n=9 nights)")
print("   Focus: env variables → sleep quality metrics")
print("=" * 60)
print(f"\n{'Variable':<24} {'vs Sleep Score':>16}  {'vs Deep%':>10}  {'vs Duration':>12}")
print("-" * 65)
for col in env_cols:
    r_sc, p_sc = stats.spearmanr(summ[col], summ["mean_sleep_score"])
    r_dp, p_dp = stats.spearmanr(summ[col], summ["pct_deep"])
    r_du, p_du = stats.spearmanr(summ[col], summ["sleep_duration_h"])
    sig_sc = "*" if p_sc < 0.05 else ("~" if p_sc < 0.10 else " ")
    sig_dp = "*" if p_dp < 0.05 else ("~" if p_dp < 0.10 else " ")
    sig_du = "*" if p_du < 0.05 else ("~" if p_du < 0.10 else " ")
    print(f"  {labels[col]:<22} r={r_sc:+.3f}{sig_sc} p={p_sc:.3f}  "
          f"r={r_dp:+.3f}{sig_dp} p={p_dp:.3f}  "
          f"r={r_du:+.3f}{sig_du} p={p_du:.3f}")
print("\n  (* p<0.05, ~ p<0.10)")

# ── Fig 4: dual heatmap (Pearson | Spearman) ─────────────────────────────────
tick_labels = [labels[c] for c in all_cols]

fig4, (ax_p, ax_s) = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
fig4.suptitle("Fig 4 · Nightly Correlation Matrix  (n = 9 nights)",
              fontsize=13, fontweight="bold")

for ax, r_mat, p_mat, title in [
    (ax_p, r_pear,  p_pear,  "Pearson r"),
    (ax_s, r_spear, p_spear, "Spearman ρ"),
]:
    im = ax.imshow(r_mat.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(all_cols)))
    ax.set_yticks(range(len(all_cols)))
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(tick_labels, fontsize=8)
    ax.set_title(title, fontsize=11)

    for i in range(len(all_cols)):
        for j in range(len(all_cols)):
            r_val = r_mat.values[i, j]
            p_val = p_mat.values[i, j]
            sig   = "**" if p_val < 0.01 else ("*" if p_val < 0.05 else
                    ("~" if p_val < 0.10 else ""))
            color = "white" if abs(r_val) > 0.5 else "black"
            ax.text(j, i, f"{r_val:.2f}{sig}", ha="center", va="center",
                    fontsize=6.5, color=color)

    # Vertical divider between env and sleep cols
    ax.axhline(len(env_cols) - 0.5, color="white", lw=2)
    ax.axvline(len(env_cols) - 0.5, color="white", lw=2)
    fig4.colorbar(im, ax=ax, shrink=0.8)

fig4.savefig(os.path.join(PLOTS, "fig4_correlation_heatmap.png"),
             dpi=150, bbox_inches="tight")
print("\nSaved: fig4_correlation_heatmap.png")

# ══════════════════════════════════════════════════════════════════════════════
# B. Within-night lag cross-correlation (pooled across all nights)
#    Q: does env variable at time t predict sleep score t+lag minutes later?
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("B. LAG CROSS-CORRELATION  (within-night, pooled, 15-min steps)")
print("=" * 60)

lag_vars = [
    ("inside_temperature", "Indoor Temp"),
    ("inside_humidity",    "Indoor Humidity"),
    ("outdoor_temperature","Outdoor Temp"),
    ("wind_speed",         "Wind Speed"),
]

max_lag = 8   # ±8 steps = ±2 hours

lag_results = {}   # var -> (lags, correlations)

for raw_col, _ in lag_vars:
    all_lags_r = []
    for night, grp in raw.groupby("night_date"):
        x = grp[raw_col].values.astype(float)
        y = grp["stage_numeric"].values.astype(float)
        if len(x) < max_lag * 2 + 3:
            continue
        # z-score within night
        x = (x - x.mean()) / (x.std() + 1e-9)
        y = (y - y.mean()) / (y.std() + 1e-9)
        # manual lag loop (pearson r at each lag)
        night_lags = []
        for lag in range(-max_lag, max_lag + 1):
            if lag >= 0:
                xv, yv = x[:len(x)-lag] if lag > 0 else x, \
                         y[lag:] if lag > 0 else y
            else:
                xv, yv = x[-lag:], y[:len(y)+lag]
            if len(xv) < 3:
                night_lags.append(np.nan)
                continue
            r, _ = stats.pearsonr(xv, yv)
            night_lags.append(r)
        all_lags_r.append(night_lags)

    arr = np.array(all_lags_r)
    mean_r = np.nanmean(arr, axis=0)
    lag_results[raw_col] = mean_r

lags_x = np.arange(-max_lag, max_lag + 1) * 15  # convert to minutes

print(f"\n  {'Variable':<20} {'Best lag (min)':>15}  {'Peak r':>8}")
print("  " + "-" * 46)
for raw_col, label in lag_vars:
    rs   = lag_results[raw_col]
    best = np.argmax(np.abs(rs))
    print(f"  {label:<20} {lags_x[best]:>+14} min  {rs[best]:>+.3f}")

# ── Fig 5: cross-correlation plots ───────────────────────────────────────────
fig5, axes5 = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
fig5.suptitle(
    "Fig 5 · Lag Cross-Correlation: Environmental Variables → Sleep Stage\n"
    "(positive lag = env leads sleep; pooled mean across 9 nights)",
    fontsize=12, fontweight="bold")

colors5 = ["#EF5350", "#42A5F5", "#FF7043", "#26A69A"]

for ax, (raw_col, label), color in zip(axes5.flatten(), lag_vars, colors5):
    rs = lag_results[raw_col]
    ax.bar(lags_x, rs, width=12, color=color, alpha=0.7, edgecolor="white")
    ax.axhline(0, color="black", lw=0.8)
    ax.axvline(0, color="gray", lw=1, ls="--", alpha=0.6)
    ax.set_xlabel("Lag (minutes)", fontsize=9)
    ax.set_ylabel("Mean Pearson r", fontsize=9)
    ax.set_title(label, fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    # Annotate peak
    best = int(np.argmax(np.abs(rs)))
    ax.annotate(f"peak r={rs[best]:+.2f}\n@ {lags_x[best]:+d} min",
                xy=(lags_x[best], rs[best]),
                xytext=(lags_x[best] + 20, rs[best] * 0.7),
                fontsize=7.5, color=color,
                arrowprops=dict(arrowstyle="->", color=color, lw=1))

fig5.savefig(os.path.join(PLOTS, "fig5_cross_correlation_lags.png"),
             dpi=150, bbox_inches="tight")
print("\nSaved: fig5_cross_correlation_lags.png")

# ══════════════════════════════════════════════════════════════════════════════
# C. Per-night Spearman table + heatmap
#    Shows how inside_temp & humidity correlate with sleep depth each night
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("C. PER-NIGHT SPEARMAN  (inside_temp & humidity → stage_numeric)")
print("=" * 60)

pernight_rows = []
for night, grp in raw.groupby("night_date"):
    row = {"night": pd.to_datetime(night).strftime("%b ") + str(pd.to_datetime(night).day)}
    for col in ["inside_temperature", "inside_humidity",
                "outdoor_temperature", "wind_speed"]:
        r, p = stats.spearmanr(grp[col], grp["stage_numeric"])
        row[f"{col}_r"] = round(r, 3)
        row[f"{col}_p"] = round(p, 3)
    pernight_rows.append(row)

pn_df = pd.DataFrame(pernight_rows)
print("\n  Inside Temp → Stage    Inside Humidity → Stage")
print(f"  {'Night':<10}", end="")
for col in ["inside_temperature", "inside_humidity", "outdoor_temperature", "wind_speed"]:
    print(f"  {col[:12]:>12}", end="")
print()
print("  " + "-" * 65)
for _, r in pn_df.iterrows():
    print(f"  {r['night']:<10}", end="")
    for col in ["inside_temperature", "inside_humidity", "outdoor_temperature", "wind_speed"]:
        sig = "*" if r[f"{col}_p"] < 0.05 else ("~" if r[f"{col}_p"] < 0.10 else " ")
        print(f"  {r[f'{col}_r']:>+8.3f}{sig}   ", end="")
    print()

# ── Fig 6: per-night heatmap ──────────────────────────────────────────────────
r_cols = ["inside_temperature_r", "inside_humidity_r",
          "outdoor_temperature_r", "wind_speed_r"]
r_labels = ["Indoor\nTemp", "Indoor\nHumid.", "Outdoor\nTemp", "Wind\nSpeed"]
p_cols   = [c.replace("_r", "_p") for c in r_cols]

r_vals = pn_df[r_cols].values
p_vals = pn_df[p_cols].values
nights_lbl = pn_df["night"].tolist()

fig6, ax6 = plt.subplots(figsize=(8, 5), constrained_layout=True)
fig6.suptitle("Fig 6 · Per-Night Spearman ρ: Env Variables → Sleep Stage Depth",
              fontsize=12, fontweight="bold")

im6 = ax6.imshow(r_vals, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
ax6.set_xticks(range(len(r_cols)))
ax6.set_xticklabels(r_labels, fontsize=9)
ax6.set_yticks(range(len(nights_lbl)))
ax6.set_yticklabels(nights_lbl, fontsize=9)

for i in range(len(nights_lbl)):
    for j in range(len(r_cols)):
        sig   = "**" if p_vals[i, j] < 0.01 else ("*" if p_vals[i, j] < 0.05 else
                ("~" if p_vals[i, j] < 0.10 else ""))
        color = "white" if abs(r_vals[i, j]) > 0.5 else "black"
        ax6.text(j, i, f"{r_vals[i,j]:+.2f}{sig}", ha="center", va="center",
                 fontsize=9, color=color)

fig6.colorbar(im6, ax=ax6, label="Spearman ρ", shrink=0.85)

fig6.savefig(os.path.join(PLOTS, "fig6_pernight_correlation.png"),
             dpi=150, bbox_inches="tight")
print("\nSaved: fig6_pernight_correlation.png")

plt.show()
print("\nStep 3 complete.")
