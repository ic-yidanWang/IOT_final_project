"""
Step 2 — EDA Visualisation
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import spearmanr
from statsmodels.nonparametric.smoothers_lowess import lowess

BASE  = os.path.dirname(os.path.abspath(__file__))
PLOTS = os.path.join(BASE, "plots")
os.makedirs(PLOTS, exist_ok=True)

summ = pd.read_csv(os.path.join(BASE, "data", "nightly_summary.csv"))
summ["night_date"] = pd.to_datetime(summ["night_date"])
xlabs = [f"{d.month}/{d.day}" for d in summ["night_date"]]

ENV_VARS = [
    ("inside_temperature_mean",  "Indoor Temp (°C)",      "#EF5350"),
    ("inside_humidity_mean",     "Indoor Humidity (% RH)", "#FF7043"),
    ("outdoor_temperature_mean", "Outdoor Temp (°C)",      "#42A5F5"),
    ("outdoor_humidity_mean",    "Outdoor Humidity (% RH)","#5C6BC0"),
    ("wind_speed_mean",          "Wind Speed (m/s)",       "#29B6F6"),
]

SLEEP_TARGETS = [
    ("mean_sleep_score", "Sleep Score",  "fig3a_env_vs_score.png"),
    ("pct_deep",         "Deep Sleep %", "fig3b_env_vs_deep.png"),
    ("pct_rem",          "REM Sleep %",  "fig3c_env_vs_rem.png"),
]


def fit_and_plot(ax, x_vals, y_vals, labels, env_color):
    """Scatter + linear fit only."""

    # scatter with night labels
    ax.scatter(x_vals, y_vals, color=env_color, s=70, zorder=5,
               edgecolors="white", linewidths=0.5)
    for xi, yi, lbl in zip(x_vals, y_vals, labels):
        ax.annotate(lbl, (xi, yi), textcoords="offset points",
                    xytext=(4, 4), fontsize=6.5, color="#444")

    x_line = np.linspace(x_vals.min(), x_vals.max(), 200)

    # Linear fit
    coef1 = np.polyfit(x_vals, y_vals, 1)
    ax.plot(x_line, np.polyval(coef1, x_line),
            color="#333333", lw=1.4, ls="--", alpha=0.7, label="Linear")

    # Spearman r annotation
    rho, p = spearmanr(x_vals, y_vals)
    sig = "**" if p < 0.01 else ("*" if p < 0.05 else ("~" if p < 0.10 else "ns"))
    ax.annotate(f"ρ={rho:+.2f}  p={p:.3f} {sig}",
                xy=(0.04, 0.93), xycoords="axes fraction",
                fontsize=8, color="#222",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))

    ax.grid(True, alpha=0.2)


# Generate one figure per sleep outcome
for sleep_col, sleep_lbl, fname in SLEEP_TARGETS:

    fig, axes = plt.subplots(1, len(ENV_VARS),
                             figsize=(16, 4.5), constrained_layout=True)
    fig.suptitle(
        f"Environmental Variables vs {sleep_lbl}  (n = {len(summ)} nights)\n"
        "Fit: dashed = linear regression",
        fontsize=11, fontweight="bold"
    )

    y_vals = summ[sleep_col].values

    for ax, (col, xlabel, color) in zip(axes, ENV_VARS):
        x_vals = summ[col].values
        fit_and_plot(ax, x_vals, y_vals, xlabs, color)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel(sleep_lbl if ax is axes[0] else "", fontsize=9)
        ax.set_title(xlabel.split(" (")[0], fontsize=9.5, fontweight="bold")

    axes[0].legend(fontsize=7.5, loc="lower right")

    fig.savefig(os.path.join(PLOTS, fname), dpi=150, bbox_inches="tight")
    print(f"Saved: plots/{fname}")

plt.show()
print("\nStep 2 complete.")
