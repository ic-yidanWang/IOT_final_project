"""
Power Analysis — Nocturnal Microclimate vs Sleep Architecture
=============================================================
Three outputs:
  1. A priori  : minimum n for r = 0.3 / 0.5 / 0.7  (a=0.05, power=0.80, two-tailed)
  2. Post-hoc  : actual power for current study at those effect sizes
  3. Figure    : power–vs–n sensitivity curves  →  plots/power_analysis.png

Method: Fisher z-transformation (standard approach for Pearson/Spearman r)
  z_r  = arctanh(r)
  SE   = 1 / sqrt(n - 3)
  NCP  = z_r * sqrt(n - 3)
  Power (two-tailed) = Φ(NCP - z_{a/2}) + Φ(-NCP - z_{a/2})
                     ≈ Φ(NCP - z_{a/2})   when NCP >> 0
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import norm

# ── Config ─────────────────────────────────────────────────────────────────────
BASE   = os.path.dirname(os.path.abspath(__file__))
PLOTS  = os.path.join(BASE, "plots")
os.makedirs(PLOTS, exist_ok=True)

# Load nightly summary to get actual n
summ    = pd.read_csv(os.path.join(BASE, "data", "nightly_summary.csv"))
N_STUDY = len(summ)          # current study nights (auto from nightly_summary.csv)
N_ORIG  = 9                  # reference baseline for post-hoc comparison column

ALPHA       = 0.05           # significance level (two-tailed)
TARGET_PWR  = 0.80           # desired power
EFFECTS     = [0.3, 0.5, 0.7]       # small / medium / large (Cohen 1988)
EFFECT_LBLS = ["r = 0.30 (small)", "r = 0.50 (medium)", "r = 0.70 (large)"]
COLORS      = ["#EF5350", "#FF9800", "#4CAF50"]   # red / orange / green

z_alpha2 = norm.ppf(1 - ALPHA / 2)   # 1.959 …
z_beta   = norm.ppf(TARGET_PWR)       # 0.842 …


# ── Helpers ────────────────────────────────────────────────────────────────────
def fisher_z(r):
    """Fisher z-transform of r."""
    return np.arctanh(r)


def power_from_n(r, n, alpha=0.05):
    """Two-tailed power via Fisher z for given r and sample size n."""
    if n <= 3:
        return 0.0
    zr  = fisher_z(r)
    ncp = zr * np.sqrt(n - 3)
    za2 = norm.ppf(1 - alpha / 2)
    # Power = P(reject H0) = Φ(NCP − z_{a/2}) + Φ(−NCP − z_{a/2})
    pwr = norm.cdf(ncp - za2) + norm.cdf(-ncp - za2)
    return pwr


def min_n(r, alpha=0.05, power=0.80):
    """Minimum integer n required to achieve 'power' for effect r."""
    za2 = norm.ppf(1 - alpha / 2)
    zb  = norm.ppf(power)
    n_exact = ((za2 + zb) / fisher_z(r)) ** 2 + 3
    return int(np.ceil(n_exact))


# ══════════════════════════════════════════════════════════════════════════════
# 1.  A priori power analysis
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print(f"A PRIORI POWER ANALYSIS  (a={ALPHA}, power={TARGET_PWR})")
print(f"Method: Fisher z-transformation (two-tailed)")
print("=" * 60)
print(f"\n{'Effect size r':<16} {'Fisher z_r':<14} {'Min n required'}")
print("-" * 45)
apriori = {}
for r, lbl in zip(EFFECTS, EFFECT_LBLS):
    zr = fisher_z(r)
    n_min = min_n(r, ALPHA, TARGET_PWR)
    apriori[r] = n_min
    print(f"  {lbl:<30}  z={zr:.4f}   n >= {n_min}")

print(f"\n  Critical z (a/2) = {z_alpha2:.4f}")
print(f"  z_b  (power=0.80) = {z_beta:.4f}")
print(f"  Formula: n = ceil(((z_{{a/2}} + z_b) / z_r)^2 + 3)")


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Post-hoc power analysis
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print(f"POST-HOC POWER ANALYSIS  (n = {N_STUDY} nights, a = {ALPHA})")
print("=" * 60)
print(f"\n{'Effect size r':<16} {'Power @ n={:2d}'.format(N_ORIG):<18} {'Power @ n={:2d}'.format(N_STUDY):<18} {'Adequate?'}")
print("-" * 60)
posthoc = {}
for r, lbl in zip(EFFECTS, EFFECT_LBLS):
    pwr9  = power_from_n(r, N_ORIG)
    pwr11 = power_from_n(r, N_STUDY)
    adequate = "YES" if pwr11 >= TARGET_PWR else "NO  ← underpowered"
    posthoc[r] = {"pwr9": pwr9, "pwr11": pwr11}
    print(f"  {lbl:<30}  {pwr9:.3f}             {pwr11:.3f}             {adequate}")

print(f"\n  Interpretation:")
for r, lbl in zip(EFFECTS, EFFECT_LBLS):
    n_req = apriori[r]
    pwr   = posthoc[r]["pwr11"]
    delta = n_req - N_STUDY
    if delta <= 0:
        msg = f"sufficient (n_req={n_req}, surplus={-delta})"
    else:
        msg = f"underpowered — needs {delta} more night(s) (n_req={n_req})"
    print(f"  {lbl}: {msg}  |  1-b = {pwr:.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Sensitivity figure
# ══════════════════════════════════════════════════════════════════════════════
n_range = np.arange(4, 81, 1)

fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)

for r, lbl, col in zip(EFFECTS, EFFECT_LBLS, COLORS):
    pwr_curve = [power_from_n(r, n) for n in n_range]
    ax.plot(n_range, pwr_curve, color=col, lw=2.5, label=lbl)

    # Mark the minimum n
    n_min = apriori[r]
    pwr_min = power_from_n(r, n_min)
    ax.scatter([n_min], [pwr_min], color=col, s=80, zorder=5)
    ax.annotate(f"n={n_min}", (n_min, pwr_min),
                textcoords="offset points", xytext=(5, -12),
                fontsize=9, color=col, fontweight="bold")

# Reference lines
ax.axvline(N_STUDY, color="#5C6BC0", lw=2, ls="--", alpha=0.85,
           label=f"Current study (n={N_STUDY})")
if N_ORIG != N_STUDY:
    ax.axvline(N_ORIG, color="#9E9E9E", lw=1.5, ls=":", alpha=0.7,
               label=f"Original study (n={N_ORIG})")
ax.axhline(TARGET_PWR, color="#263238", lw=1.5, ls="-.", alpha=0.65,
           label=f"Target power = {TARGET_PWR}")

# Shade underpowered region
ax.fill_between(n_range, 0, TARGET_PWR, alpha=0.05, color="red")
ax.text(5, TARGET_PWR / 2, "Underpowered\nzone", fontsize=9,
        color="#c62828", alpha=0.7, va="center")

# Annotations for current study power
for r, lbl, col in zip(EFFECTS, EFFECT_LBLS, COLORS):
    p_cur = power_from_n(r, N_STUDY)
    ax.scatter([N_STUDY], [p_cur], color=col, marker="D", s=55, zorder=6)
    ax.annotate(f"{p_cur:.2f}", (N_STUDY, p_cur),
                textcoords="offset points", xytext=(7, 3),
                fontsize=8.5, color=col)

ax.set_xlabel("Sample size  n  (number of nights)", fontsize=11)
ax.set_ylabel("Statistical Power  (1 − b)", fontsize=11)
ax.set_title(
    "Power–Sample Size Sensitivity Curve\n"
    f"Spearman correlation  |  a = {ALPHA} (two-tailed)  |  Fisher z method",
    fontsize=12, fontweight="bold"
)
ax.set_xlim(4, 80)
ax.set_ylim(0, 1.05)
ax.set_yticks(np.arange(0, 1.1, 0.1))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1f}"))
ax.grid(True, alpha=0.25)
ax.legend(fontsize=9.5, loc="lower right", framealpha=0.9)

fig.savefig(os.path.join(PLOTS, "power_analysis.png"), dpi=150, bbox_inches="tight")
print(f"\nSaved: plots/power_analysis.png")

plt.show()
print("\nPower analysis complete.")
