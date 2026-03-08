"""
Step 2 — EDA Visualisation
Project: Nocturnal Microclimate vs Sleep Architecture
------------------------------------------------------
Outputs (saved to plots/):
  fig1_sleep_stage_timelines.png   — per-night sleep stage + indoor temp
  fig2_cross_night_summary.png     — nightly trend overview
  fig3_env_vs_sleep_quality.png    — scatter: env variables vs sleep score
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Load data ─────────────────────────────────────────────────────────────────
BASE  = os.path.dirname(os.path.abspath(__file__))
PLOTS = os.path.join(BASE, "plots")
os.makedirs(PLOTS, exist_ok=True)

raw  = pd.read_csv(os.path.join(BASE, "data", "all_nights_raw.csv"))
summ = pd.read_csv(os.path.join(BASE, "data", "nightly_summary.csv"))

raw["timestamp"]   = pd.to_datetime(raw["timestamp"])
summ["night_date"] = pd.to_datetime(summ["night_date"])

nights = raw["night_date"].unique()

def short_date(d):
    dt = pd.to_datetime(d)
    return f"{dt.month}/{dt.day}"

def long_date(d):
    dt = pd.to_datetime(d)
    return dt.strftime("%b ") + str(dt.day)

STAGE_COLOR = {"Deep": "#1565C0", "Core": "#42A5F5", "REM": "#CE93D8"}
STAGE_ORDER = ["Deep", "Core", "REM"]
xlabs = [short_date(d) for d in summ["night_date"]]

# ══════════════════════════════════════════════════════════════════════════════
# Fig 1 — Per-night sleep stage timeline + indoor temp & humidity
# ══════════════════════════════════════════════════════════════════════════════
n_nights = len(nights)
fig1, axes = plt.subplots(n_nights, 1, figsize=(13, n_nights * 1.9),
                           sharex=False, constrained_layout=True)
fig1.suptitle("Fig 1 · Nightly Sleep Architecture with Indoor Microclimate",
              fontsize=13, fontweight="bold")

for ax, night in zip(axes, nights):
    grp = raw[raw["night_date"] == night].copy()
    t   = grp["elapsed_min"].values

    # Sleep stage background bands
    for _, row in grp.iterrows():
        ax.axvspan(row["elapsed_min"], row["elapsed_min"] + 15,
                   color=STAGE_COLOR.get(row["sleep_stage"], "#ccc"),
                   alpha=0.35, linewidth=0)

    # Indoor temperature (left y-axis)
    ax.plot(t, grp["inside_temperature"], color="#E53935", lw=1.8,
            label="Indoor Temp (°C)")
    ax.set_ylabel("Temp (°C)", fontsize=8, color="#E53935")
    ax.tick_params(axis="y", labelcolor="#E53935", labelsize=7)

    # Indoor humidity (right y-axis)
    ax2 = ax.twinx()
    ax2.plot(t, grp["inside_humidity"], color="#1E88E5", lw=1.4,
             linestyle="--", label="Indoor Humidity (%)")
    ax2.set_ylabel("Humidity (%)", fontsize=8, color="#1E88E5")
    ax2.tick_params(axis="y", labelcolor="#1E88E5", labelsize=7)

    label = long_date(night)
    dur   = grp["sleep time"].iloc[0]
    score = grp["sleep_score"].mean()
    ax.set_title(f"{label}  |  {dur}h sleep  |  avg score {score:.2f}",
                 fontsize=8.5, loc="left", pad=2)
    ax.set_xlim(0, t[-1] + 15)
    ax.tick_params(axis="x", labelsize=7)
    ax.grid(axis="x", alpha=0.2)

axes[-1].set_xlabel("Minutes from sleep onset", fontsize=9)

legend_patches = [mpatches.Patch(color=STAGE_COLOR[s], alpha=0.5, label=s)
                  for s in STAGE_ORDER]
legend_patches += [
    plt.Line2D([0], [0], color="#E53935", lw=1.8, label="Indoor Temp"),
    plt.Line2D([0], [0], color="#1E88E5", lw=1.4, ls="--", label="Indoor Humidity"),
]
fig1.legend(handles=legend_patches, loc="lower center",
            ncol=5, fontsize=8.5, bbox_to_anchor=(0.5, -0.01))

fig1.savefig(os.path.join(PLOTS, "fig1_sleep_stage_timelines.png"),
             dpi=150, bbox_inches="tight")
print("Saved: fig1_sleep_stage_timelines.png")

# ══════════════════════════════════════════════════════════════════════════════
# Fig 2 — Cross-night summary: trends + sleep stage stacked bar
# ══════════════════════════════════════════════════════════════════════════════
fig2, axes2 = plt.subplots(3, 2, figsize=(13, 9), constrained_layout=True)
fig2.suptitle("Fig 2 · Cross-Night Summary Trends", fontsize=13, fontweight="bold")

xi = range(len(summ))

def trend_ax(ax, col, ylabel, color, title):
    ax.plot(xi, summ[col], color=color, lw=2, marker="o", ms=6)
    ax.fill_between(xi, summ[col], alpha=0.12, color=color)
    ax.set_xticks(xi)
    ax.set_xticklabels(xlabs, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.3)

trend_ax(axes2[0, 0], "sleep_duration_h",        "Hours", "#5C6BC0", "Sleep Duration (h)")
trend_ax(axes2[0, 1], "mean_sleep_score",         "Score", "#26A69A", "Mean Sleep Score")
trend_ax(axes2[1, 0], "inside_temperature_mean",  "°C",    "#EF5350", "Indoor Temp Mean (°C)")
trend_ax(axes2[1, 1], "inside_humidity_mean",     "%",     "#42A5F5", "Indoor Humidity Mean (%)")
trend_ax(axes2[2, 0], "outdoor_temperature_mean", "°C",    "#FF7043", "Outdoor Temp Mean (°C)")

# Stacked bar — sleep stage proportions
ax_bar = axes2[2, 1]
bar_w  = 0.6
ax_bar.bar(xi, summ["pct_deep"], bar_w, label="Deep",
           color=STAGE_COLOR["Deep"], alpha=0.85)
ax_bar.bar(xi, summ["pct_core"], bar_w, bottom=summ["pct_deep"],
           label="Core", color=STAGE_COLOR["Core"], alpha=0.85)
ax_bar.bar(xi, summ["pct_rem"],  bar_w,
           bottom=summ["pct_deep"] + summ["pct_core"],
           label="REM", color=STAGE_COLOR["REM"], alpha=0.85)
ax_bar.set_xticks(xi)
ax_bar.set_xticklabels(xlabs, fontsize=8)
ax_bar.set_ylabel("%", fontsize=9)
ax_bar.set_title("Sleep Stage Proportions (%)", fontsize=10)
ax_bar.legend(fontsize=8)
ax_bar.grid(axis="y", alpha=0.3)
ax_bar.set_ylim(0, 105)

fig2.savefig(os.path.join(PLOTS, "fig2_cross_night_summary.png"),
             dpi=150, bbox_inches="tight")
print("Saved: fig2_cross_night_summary.png")

# ══════════════════════════════════════════════════════════════════════════════
# Fig 3 — Environment vs Sleep Quality scatter plots
# ══════════════════════════════════════════════════════════════════════════════
env_vars = [
    ("inside_temperature_mean",  "Indoor Temp Mean (°C)"),
    ("inside_humidity_mean",     "Indoor Humidity Mean (%)"),
    ("outdoor_temperature_mean", "Outdoor Temp Mean (°C)"),
    ("outdoor_humidity_mean",    "Outdoor Humidity Mean (%)"),
    ("wind_speed_mean",          "Wind Speed Mean (m/s)"),
    ("inside_temperature_std",   "Indoor Temp Variability (std)"),
]

fig3, axes3 = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
fig3.suptitle("Fig 3 · Environmental Variables vs Sleep Quality",
              fontsize=13, fontweight="bold")

for ax, (col, xlabel) in zip(axes3.flatten(), env_vars):
    x_vals = summ[col].values
    y_vals = summ["mean_sleep_score"].values

    ax.scatter(x_vals, y_vals, color="#5C6BC0", s=80, zorder=3, alpha=0.85)

    for xi_v, yi_v, lbl in zip(x_vals, y_vals, xlabs):
        ax.annotate(lbl, (xi_v, yi_v), textcoords="offset points",
                    xytext=(4, 4), fontsize=7, color="#555")

    if len(x_vals) > 2:
        m, b = np.polyfit(x_vals, y_vals, 1)
        corr = np.corrcoef(x_vals, y_vals)[0, 1]
        x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
        ax.plot(x_line, m * x_line + b, color="#EF5350", lw=1.5,
                label=f"r = {corr:.2f}")
        ax.legend(fontsize=8)

    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel("Mean Sleep Score", fontsize=9)
    ax.grid(True, alpha=0.3)

fig3.savefig(os.path.join(PLOTS, "fig3_env_vs_sleep_quality.png"),
             dpi=150, bbox_inches="tight")
print("Saved: fig3_env_vs_sleep_quality.png")

plt.show()
print("\nStep 2 complete — all figures saved to plots/")
