"""
Step 4 — Sleep Architecture & Cycle Structure Analysis
Project: Nocturnal Microclimate vs Sleep Architecture
------------------------------------------------------
Pure sleep-stage analysis (Apple Watch data), no environmental variables.

Three analyses:
  A. Stage transition matrix (Markov) — how stages flow into each other
  B. Sleep cycle identification — auto-detect NREM+REM cycles per night
  C. Sleep efficiency dashboard — stage durations, cycle counts, onset speed

Outputs (saved to plots/):
  fig7_hypnograms.png            — per-night hypnogram (sleep stage over time)
  fig8_transition_matrix.png     — stage transition heatmap
  fig9_sleep_efficiency.png      — efficiency metrics per night
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from collections import defaultdict

# ── Load ──────────────────────────────────────────────────────────────────────
BASE  = os.path.dirname(os.path.abspath(__file__))
PLOTS = os.path.join(BASE, "plots")
os.makedirs(PLOTS, exist_ok=True)

raw = pd.read_csv(os.path.join(BASE, "data", "all_nights_raw.csv"))
raw["timestamp"] = pd.to_datetime(raw["timestamp"])

STAGE_ORDER  = ["Deep", "Core", "REM"]
STAGE_Y      = {"Deep": 3, "Core": 2, "REM": 1}   # y-position in hypnogram
STAGE_COLOR  = {"Deep": "#1565C0", "Core": "#42A5F5", "REM": "#CE93D8"}
STAGE_NUM    = {"REM": 1, "Core": 2, "Deep": 3}

nights = raw["night_date"].unique()

def short_date(d):
    dt = pd.to_datetime(d)
    return f"{dt.month}/{dt.day}"

def long_date(d):
    dt = pd.to_datetime(d)
    return dt.strftime("%b ") + str(dt.day)

# ══════════════════════════════════════════════════════════════════════════════
# A. Stage Transition Matrix (pooled across all nights)
# ══════════════════════════════════════════════════════════════════════════════
trans_count = defaultdict(lambda: defaultdict(int))   # from → to

for night, grp in raw.groupby("night_date"):
    stages = grp["sleep_stage"].tolist()
    for i in range(len(stages) - 1):
        trans_count[stages[i]][stages[i + 1]] += 1

# Build probability matrix
trans_df = pd.DataFrame(0.0, index=STAGE_ORDER, columns=STAGE_ORDER)
for frm in STAGE_ORDER:
    total = sum(trans_count[frm].values())
    for to in STAGE_ORDER:
        trans_df.loc[frm, to] = trans_count[frm][to] / total if total > 0 else 0

print("=" * 55)
print("A. STAGE TRANSITION MATRIX (probability, pooled)")
print("=" * 55)
print(trans_df.round(3).to_string())

# ── Fig 8: transition heatmap ─────────────────────────────────────────────────
fig8, ax8 = plt.subplots(figsize=(6, 5), constrained_layout=True)
fig8.suptitle("Fig 8 · Sleep Stage Transition Matrix\n(pooled across 9 nights)",
              fontsize=12, fontweight="bold")

im = ax8.imshow(trans_df.values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
ax8.set_xticks(range(3)); ax8.set_xticklabels(["→ Deep", "→ Core", "→ REM"], fontsize=11)
ax8.set_yticks(range(3)); ax8.set_yticklabels(["Deep →", "Core →", "REM →"],  fontsize=11)
ax8.set_title("From \\ To", fontsize=10)

for i in range(3):
    for j in range(3):
        v = trans_df.values[i, j]
        col = "white" if v > 0.5 else "black"
        # raw count
        cnt = trans_count[STAGE_ORDER[i]][STAGE_ORDER[j]]
        ax8.text(j, i, f"{v:.2f}\n(n={cnt})", ha="center", va="center",
                 fontsize=10, color=col, fontweight="bold" if i == j else "normal")

fig8.colorbar(im, ax=ax8, label="Transition Probability", shrink=0.85)

fig8.savefig(os.path.join(PLOTS, "fig8_transition_matrix.png"),
             dpi=150, bbox_inches="tight")
print("\nSaved: fig8_transition_matrix.png")

# ══════════════════════════════════════════════════════════════════════════════
# B. Sleep Cycle Identification
#    A "cycle" = one REM episode preceded by at least one NREM block.
#    We detect REM episodes and mark cycle boundaries.
# ══════════════════════════════════════════════════════════════════════════════
def identify_cycles(grp):
    """Return list of cycle dicts: {cycle_no, nrem_start, rem_start, rem_end, duration_min}"""
    stages = grp["sleep_stage"].tolist()
    elapsed = grp["elapsed_min"].tolist()

    cycles = []
    cycle_no = 0
    i = 0
    nrem_start = None

    while i < len(stages):
        # Find start of NREM block
        if stages[i] in ("Core", "Deep") and nrem_start is None:
            nrem_start = elapsed[i]

        # Find REM block after NREM
        if stages[i] == "REM" and nrem_start is not None:
            rem_start = elapsed[i]
            j = i
            while j < len(stages) and stages[j] == "REM":
                j += 1
            rem_end = elapsed[j - 1] + 15
            cycle_no += 1
            cycles.append({
                "cycle_no":    cycle_no,
                "nrem_start":  nrem_start,
                "rem_start":   rem_start,
                "rem_end":     rem_end,
                "duration_min": rem_end - nrem_start,
            })
            nrem_start = None   # reset for next cycle
            i = j
            continue
        i += 1

    return cycles

print("\n" + "=" * 55)
print("B. SLEEP CYCLE IDENTIFICATION")
print("=" * 55)

all_cycle_rows = []
for night in nights:
    grp    = raw[raw["night_date"] == night].copy()
    cycles = identify_cycles(grp)
    label  = long_date(night)
    dur    = grp["sleep time"].iloc[0]
    print(f"\n  {label}  ({dur}h sleep)  →  {len(cycles)} cycle(s) detected")
    for c in cycles:
        print(f"    Cycle {c['cycle_no']}: NREM onset {c['nrem_start']:.0f} min "
              f"→ REM {c['rem_start']:.0f}-{c['rem_end']:.0f} min "
              f"| duration {c['duration_min']:.0f} min")
        all_cycle_rows.append({"night": label, **c})

cycles_df = pd.DataFrame(all_cycle_rows)

# ══════════════════════════════════════════════════════════════════════════════
# Fig 7 — Hypnograms (one per night, stacked, with cycle markers)
# ══════════════════════════════════════════════════════════════════════════════
n_nights = len(nights)
fig7, axes7 = plt.subplots(n_nights, 1, figsize=(13, n_nights * 1.65),
                            sharex=False, constrained_layout=True)
fig7.suptitle("Fig 7 · Nightly Hypnograms with Sleep Cycle Boundaries",
              fontsize=13, fontweight="bold")

for ax, night in zip(axes7, nights):
    grp    = raw[raw["night_date"] == night].copy()
    t      = grp["elapsed_min"].values
    stages = grp["sleep_stage"].values
    cycles = identify_cycles(grp)
    label  = long_date(night)
    dur    = grp["sleep time"].iloc[0]

    # Stage step line
    y_vals = [STAGE_Y[s] for s in stages]
    ax.step(t, y_vals, where="post", color="#37474F", lw=1.5, zorder=3)

    # Filled area coloured by stage
    for k in range(len(t) - 1):
        ax.fill_between([t[k], t[k + 1]], 0, STAGE_Y[stages[k]],
                        color=STAGE_COLOR[stages[k]], alpha=0.45, step="post")
    # last segment
    ax.fill_between([t[-1], t[-1] + 15], 0, STAGE_Y[stages[-1]],
                    color=STAGE_COLOR[stages[-1]], alpha=0.45)

    # Cycle boundary lines
    for c in cycles:
        ax.axvline(c["nrem_start"], color="#546E7A", lw=1.2, ls=":", alpha=0.7)
        ax.axvline(c["rem_end"],    color="#546E7A", lw=1.2, ls=":", alpha=0.7)
        mid = (c["nrem_start"] + c["rem_end"]) / 2
        ax.text(mid, 3.55, f"C{c['cycle_no']}",
                ha="center", va="bottom", fontsize=7, color="#546E7A", fontweight="bold")

    ax.set_yticks([1, 2, 3])
    ax.set_yticklabels(["REM", "Core", "Deep"], fontsize=7.5)
    ax.set_ylim(0.5, 3.8)
    ax.set_xlim(0, t[-1] + 20)
    ax.tick_params(axis="x", labelsize=7)
    ax.grid(axis="x", alpha=0.2)

    n_cyc = len(cycles)
    ax.set_title(f"{label}  |  {dur}h  |  {n_cyc} cycle{'s' if n_cyc != 1 else ''}",
                 fontsize=8.5, loc="left", pad=2)

axes7[-1].set_xlabel("Minutes from sleep onset", fontsize=9)

legend_p = [mpatches.Patch(color=STAGE_COLOR[s], alpha=0.6, label=s) for s in STAGE_ORDER]
fig7.legend(handles=legend_p, loc="lower center", ncol=3,
            fontsize=9, bbox_to_anchor=(0.5, -0.01))

fig7.savefig(os.path.join(PLOTS, "fig7_hypnograms.png"),
             dpi=150, bbox_inches="tight")
print("\nSaved: fig7_hypnograms.png")

# ══════════════════════════════════════════════════════════════════════════════
# C. Sleep Efficiency Dashboard
# ══════════════════════════════════════════════════════════════════════════════
eff_rows = []
for night in nights:
    grp    = raw[raw["night_date"] == night].copy()
    cycles = identify_cycles(grp)
    stages = grp["sleep_stage"].values
    n_rows = len(stages)
    label  = short_date(night)

    # Time in each stage (minutes, 15 min per row)
    deep_min = (stages == "Deep").sum() * 15
    core_min = (stages == "Core").sum() * 15
    rem_min  = (stages == "REM").sum()  * 15
    total_min = n_rows * 15

    # Sleep onset latency = elapsed_min of first row
    onset_lat = grp["elapsed_min"].iloc[0]

    # First REM latency
    rem_rows = grp[grp["sleep_stage"] == "REM"]
    first_rem = rem_rows["elapsed_min"].iloc[0] if len(rem_rows) > 0 else np.nan

    # First Deep latency
    deep_rows = grp[grp["sleep_stage"] == "Deep"]
    first_deep = deep_rows["elapsed_min"].iloc[0] if len(deep_rows) > 0 else np.nan

    # Cycle count & avg cycle length
    n_cycles    = len(cycles)
    avg_cycle   = np.mean([c["duration_min"] for c in cycles]) if cycles else np.nan

    eff_rows.append({
        "night":      label,
        "total_min":  total_min,
        "deep_min":   deep_min,
        "core_min":   core_min,
        "rem_min":    rem_min,
        "first_rem":  first_rem,
        "first_deep": first_deep,
        "n_cycles":   n_cycles,
        "avg_cycle":  avg_cycle,
        "sleep_dur_h": grp["sleep time"].iloc[0],
    })

eff_df = pd.DataFrame(eff_rows)

print("\n" + "=" * 55)
print("C. SLEEP EFFICIENCY METRICS")
print("=" * 55)
print(eff_df[["night","sleep_dur_h","deep_min","rem_min","first_rem",
              "first_deep","n_cycles","avg_cycle"]].to_string(index=False))

# ── Fig 9: efficiency dashboard ───────────────────────────────────────────────
fig9 = plt.figure(figsize=(14, 9), constrained_layout=True)
fig9.suptitle("Fig 9 · Sleep Efficiency Dashboard (per night)",
              fontsize=13, fontweight="bold")
gs9  = GridSpec(2, 3, figure=fig9)

x9    = range(len(eff_df))
xlabs9 = eff_df["night"].tolist()

def eff_ax(ax, col, ylabel, color, title, annotate=True):
    vals = eff_df[col].values
    bars = ax.bar(x9, vals, color=color, alpha=0.8, edgecolor="white", width=0.6)
    if annotate:
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + max(vals) * 0.02,
                    f"{v:.0f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x9); ax.set_xticklabels(xlabs9, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=9); ax.set_title(title, fontsize=10)
    ax.grid(axis="y", alpha=0.3); ax.set_ylim(0, max(vals) * 1.2)

# Stacked bar: stage composition
ax_stack = fig9.add_subplot(gs9[0, :2])
ax_stack.bar(x9, eff_df["deep_min"], 0.6, label="Deep",
             color=STAGE_COLOR["Deep"], alpha=0.85)
ax_stack.bar(x9, eff_df["core_min"], 0.6, bottom=eff_df["deep_min"],
             label="Core", color=STAGE_COLOR["Core"], alpha=0.85)
ax_stack.bar(x9, eff_df["rem_min"],  0.6,
             bottom=eff_df["deep_min"] + eff_df["core_min"],
             label="REM", color=STAGE_COLOR["REM"], alpha=0.85)
ax_stack.set_xticks(x9); ax_stack.set_xticklabels(xlabs9, fontsize=8)
ax_stack.set_ylabel("Minutes", fontsize=9)
ax_stack.set_title("Stage Duration Composition (minutes)", fontsize=10)
ax_stack.legend(fontsize=9); ax_stack.grid(axis="y", alpha=0.3)

# Cycle count
ax_cyc = fig9.add_subplot(gs9[0, 2])
eff_ax(ax_cyc, "n_cycles", "Count", "#5C6BC0", "Sleep Cycles per Night")

# First REM latency
ax_rem = fig9.add_subplot(gs9[1, 0])
eff_ax(ax_rem, "first_rem", "Minutes", "#CE93D8", "First REM Latency (min)")

# First Deep latency
ax_deep = fig9.add_subplot(gs9[1, 1])
eff_ax(ax_deep, "first_deep", "Minutes", "#1565C0", "First Deep Latency (min)")

# Avg cycle length
ax_avg = fig9.add_subplot(gs9[1, 2])
eff_df_clean = eff_df.dropna(subset=["avg_cycle"])
vals_avg = eff_df["avg_cycle"].fillna(0).values
bars_avg = ax_avg.bar(x9, vals_avg, 0.6, color="#26A69A", alpha=0.8, edgecolor="white")
for b, v in zip(bars_avg, vals_avg):
    if v > 0:
        ax_avg.text(b.get_x() + b.get_width() / 2, v + 2,
                    f"{v:.0f}", ha="center", va="bottom", fontsize=8)
ax_avg.set_xticks(x9); ax_avg.set_xticklabels(xlabs9, fontsize=8)
ax_avg.set_ylabel("Minutes", fontsize=9)
ax_avg.set_title("Avg Cycle Duration (min)", fontsize=10)
ax_avg.grid(axis="y", alpha=0.3)

fig9.savefig(os.path.join(PLOTS, "fig9_sleep_efficiency.png"),
             dpi=150, bbox_inches="tight")
print("\nSaved: fig9_sleep_efficiency.png")

plt.show()
print("\nStep 4 complete.")
