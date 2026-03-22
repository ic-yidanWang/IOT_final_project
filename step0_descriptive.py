"""
Step 0 — Descriptive Statistics & Normality Testing
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats

BASE  = os.path.dirname(os.path.abspath(__file__))
PLOTS = os.path.join(BASE, "plots")
os.makedirs(PLOTS, exist_ok=True)

summ = pd.read_csv(os.path.join(BASE, "data", "nightly_summary.csv"))
summ["night_date"] = pd.to_datetime(summ["night_date"])
N = len(summ)

ENV_VARS = {
    "inside_temperature_mean":  ("Indoor Temp",     "°C",    "#EF5350"),
    "inside_humidity_mean":     ("Indoor Humidity",  "% RH",  "#FF7043"),
    "outdoor_temperature_mean": ("Outdoor Temp",    "°C",    "#42A5F5"),
    "outdoor_humidity_mean":    ("Outdoor Humidity", "% RH",  "#5C6BC0"),
    "wind_speed_mean":          ("Wind Speed",       "m/s",   "#29B6F6"),
}

SLEEP_VARS = {
    "sleep_duration_h":  ("Sleep Duration",  "hours",  "#66BB6A"),
    "mean_sleep_score":  ("Sleep Score",     "score",  "#26A69A"),
    "pct_deep":          ("Deep Sleep",      "%",      "#1565C0"),
    "pct_core":          ("Core Sleep",      "%",      "#42A5F5"),
    "pct_rem":           ("REM Sleep",       "%",      "#CE93D8"),
}

# A. Descriptive statistics (console)
ALL_VARS = {**ENV_VARS, **SLEEP_VARS}

print("=" * 80)
print(f"A. DESCRIPTIVE STATISTICS  (n = {N} nights, nightly means)")
print("=" * 80)
print(f"  {'Variable':<22} {'n':>3} {'Mean':>8} {'SD':>7} {'Min':>8} {'Max':>8} {'Skew':>7} {'Kurt':>7}")
print("  " + "-" * 74)

for col, (lbl, unit, _) in ALL_VARS.items():
    s = summ[col].dropna()
    print(f"  {lbl + ' (' + unit + ')':<28} {len(s):>3} "
          f"{s.mean():>8.3f} {s.std(ddof=1):>7.3f} "
          f"{s.min():>8.3f} {s.max():>8.3f} "
          f"{stats.skew(s):>7.3f} {stats.kurtosis(s):>7.3f}")

print("\n  Kurtosis = excess kurtosis (normal = 0)")

# B. Shapiro-Wilk normality test (console)
print("\n" + "=" * 80)
print(f"B. SHAPIRO-WILK NORMALITY TEST  (n = {N}, alpha = 0.05)")
print("=" * 80)
print(f"  H0: data are normally distributed  |  Reject if p < 0.05\n")
print(f"  {'Variable':<28} {'W':>8}  {'p':>8}  {'Normal?':>8}  Conclusion")
print("  " + "-" * 68)

sw_rows = []
any_non_normal = False
for col, (lbl, unit, _) in ALL_VARS.items():
    s = summ[col].dropna().values
    W, p = stats.shapiro(s)
    normal = p >= 0.05
    if not normal:
        any_non_normal = True
    verdict   = "Yes" if normal else "No *"
    conclusion = "Pearson OK" if normal else "Use Spearman"
    print(f"  {lbl:<28} {W:>8.4f}  {p:>8.4f}  {verdict:>8}  {conclusion}")
    sw_rows.append({"col": col, "label": lbl, "unit": unit,
                    "W": W, "p": p, "normal": normal})

sw_df = pd.DataFrame(sw_rows)

if any_non_normal:
    print("\n  CONCLUSION: At least one variable violates normality (p < 0.05).")
    print("  => Spearman rank correlation used throughout.")
else:
    print("\n  CONCLUSION: All variables appear normally distributed.")
    print("  => Spearman used as robust default regardless.")

# Fig 0a — Environmental variable distributions (box + jittered strip)
fig0a, axes = plt.subplots(1, len(ENV_VARS),
                            figsize=(13, 5), constrained_layout=True)
fig0a.suptitle(f"Fig 0a · Environmental Variable Distributions  (n = {N} nights)",
               fontsize=12, fontweight="bold")

for ax, (col, (lbl, unit, color)) in zip(axes, ENV_VARS.items()):
    vals = summ[col].dropna().values
    bp = ax.boxplot(vals, patch_artist=True, widths=0.45,
                    medianprops=dict(color="black", lw=2),
                    boxprops=dict(facecolor=color, alpha=0.5),
                    whiskerprops=dict(lw=1.2),
                    capprops=dict(lw=1.5),
                    flierprops=dict(marker="o", color=color, ms=5))

    # Jittered strip
    jitter = np.random.default_rng(42).uniform(-0.12, 0.12, size=len(vals))
    ax.scatter(1 + jitter, vals, color=color, alpha=0.85, s=35, zorder=4)

    # Mean marker
    ax.scatter(1, vals.mean(), marker="D", color="black", s=40, zorder=5,
               label=f"mean={vals.mean():.2f}")

    ax.set_xticks([])
    ax.set_title(f"{lbl}", fontsize=10, fontweight="bold")
    ax.set_ylabel(unit, fontsize=9)
    ax.legend(fontsize=7.5, loc="upper right")
    ax.grid(axis="y", alpha=0.25)

fig0a.savefig(os.path.join(PLOTS, "fig0a_env_boxplots.png"), dpi=150, bbox_inches="tight")
print("Saved: plots/fig0a_env_boxplots.png")

# Fig 0b — Sleep variable distributions (box + jittered strip)

fig0b, axes = plt.subplots(1, len(SLEEP_VARS),
                            figsize=(14, 5), constrained_layout=True)
fig0b.suptitle(f"Fig 0b · Sleep Variable Distributions  (n = {N} nights)",
               fontsize=12, fontweight="bold")

for ax, (col, (lbl, unit, color)) in zip(axes, SLEEP_VARS.items()):
    vals = summ[col].dropna().values
    ax.boxplot(vals, patch_artist=True, widths=0.45,
               medianprops=dict(color="black", lw=2),
               boxprops=dict(facecolor=color, alpha=0.5),
               whiskerprops=dict(lw=1.2),
               capprops=dict(lw=1.5),
               flierprops=dict(marker="o", color=color, ms=5))

    jitter = np.random.default_rng(42).uniform(-0.12, 0.12, size=len(vals))
    ax.scatter(1 + jitter, vals, color=color, alpha=0.85, s=35, zorder=4)
    ax.scatter(1, vals.mean(), marker="D", color="black", s=40, zorder=5,
               label=f"mean={vals.mean():.2f}")

    ax.set_xticks([])
    ax.set_title(f"{lbl}", fontsize=10, fontweight="bold")
    ax.set_ylabel(unit, fontsize=9)
    ax.legend(fontsize=7.5, loc="upper right")
    ax.grid(axis="y", alpha=0.25)

fig0b.savefig(os.path.join(PLOTS, "fig0b_sleep_boxplots.png"), dpi=150, bbox_inches="tight")
print("Saved: plots/fig0b_sleep_boxplots.png")

# Fig 0c — Shapiro-Wilk normality test
fig0c, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
fig0c.suptitle(f"Fig 0c · Shapiro-Wilk Normality Test  (n = {N}, α = 0.05)",
               fontsize=12, fontweight="bold")

for ax, group, title in zip(
    axes,
    [
        sw_df[sw_df["col"].isin(ENV_VARS.keys())],
        sw_df[sw_df["col"].isin(SLEEP_VARS.keys())],
    ],
    ["Environmental Variables", "Sleep Variables"]
):
    bar_colors = ["#4CAF50" if r else "#EF5350" for r in group["normal"]]
    bars = ax.barh(group["label"], group["W"], color=bar_colors,
                   alpha=0.82, edgecolor="white", height=0.55)
    ax.axvline(0.90, color="#FFA726", lw=1.8, ls="--", alpha=0.85,
               label="W = 0.90 (guideline)")
    ax.axvline(0.95, color="#9E9E9E", lw=1.2, ls=":", alpha=0.6,
               label="W = 0.95")

    for bar, row in zip(bars, group.itertuples()):
        sig  = " *" if not row.normal else ""
        col_txt = "#1B5E20" if row.normal else "#B71C1C"
        ax.text(bar.get_width() + 0.003,
                bar.get_y() + bar.get_height() / 2,
                f"p={row.p:.3f}{sig}",
                va="center", fontsize=9, color=col_txt,
                fontweight="bold" if not row.normal else "normal")

    ax.set_xlim(0.70, 1.06)
    ax.set_xlabel("Shapiro-Wilk W", fontsize=10)
    ax.set_title(f"{title}\n(green = normal, red = non-normal*)",
                 fontsize=10)
    ax.legend(fontsize=8.5, loc="lower right")
    ax.grid(axis="x", alpha=0.25)

fig0c.savefig(os.path.join(PLOTS, "fig0c_normality.png"), dpi=150, bbox_inches="tight")
print("Saved: plots/fig0c_normality.png")

plt.show()
print("\nStep 0 complete.")
