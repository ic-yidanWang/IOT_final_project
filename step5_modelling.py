"""
Step 5 — Statistical Modelling
"""

import os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error, r2_score

warnings.filterwarnings("ignore")

BASE  = os.path.dirname(os.path.abspath(__file__))
PLOTS = os.path.join(BASE, "plots")
os.makedirs(PLOTS, exist_ok=True)

raw  = pd.read_csv(os.path.join(BASE, "data", "all_nights_raw.csv"))
summ = pd.read_csv(os.path.join(BASE, "data", "nightly_summary.csv"))
raw["timestamp"]   = pd.to_datetime(raw["timestamp"])
summ["night_date"] = pd.to_datetime(summ["night_date"])

STAGE_MAP = {"REM": 0, "Core": 1, "Deep": 2}
raw["stage_numeric"] = raw["sleep_stage"].map(STAGE_MAP)
xlabs = [f"{pd.to_datetime(d).month}/{pd.to_datetime(d).day}"
         for d in summ["night_date"]]


# Level A — Nightly OLS regression
print("=" * 60)
print(f"LEVEL A — NIGHTLY REGRESSION  (n={len(summ)})")
print("=" * 60)

FEATURES = {
    "inside_humidity_mean":     "Indoor Humidity",
    "inside_temperature_mean":  "Indoor Temp",
    "outdoor_temperature_mean": "Outdoor Temp",
    "wind_speed_mean":          "Wind Speed",
}
TARGET = "mean_sleep_score"

X_raw = summ[list(FEATURES.keys())].values
y     = summ[TARGET].values
n     = len(y)

# --- Standardise for coefficient comparison ---
scaler = StandardScaler()
X_std  = scaler.fit_transform(X_raw)

# --- Full OLS (standardised) ---
ols = LinearRegression().fit(X_std, y)
y_pred_ols = ols.predict(X_std)
r2_ols = r2_score(y, y_pred_ols)
adj_r2 = 1 - (1 - r2_ols) * (n - 1) / (n - len(FEATURES) - 1)

print(f"\nFull OLS  R2={r2_ols:.3f}  Adj-R2={adj_r2:.3f}")
print(f"{'Feature':<22} {'Std Coeff':>12}  {'Raw Coeff':>12}")
print("-" * 48)
for name, label in FEATURES.items():
    idx = list(FEATURES.keys()).index(name)
    raw_coef = ols.coef_[idx] / scaler.scale_[idx]
    print(f"  {label:<20} {ols.coef_[idx]:>+12.4f}  {raw_coef:>+12.4f}")

# --- Univariate OLS for each feature ---
print(f"\n{'Feature':<22} {'r':>8}  {'β':>8}  {'p-val':>8}  {'R2':>8}")
print("-" * 60)
uni_results = {}
for name, label in FEATURES.items():
    x_col = summ[name].values
    slope, intercept, r, p, se = stats.linregress(x_col, y)
    uni_results[name] = dict(slope=slope, intercept=intercept, r=r, p=p, r2=r**2)
    sig = "**" if p < 0.01 else ("*" if p < 0.05 else ("~" if p < 0.10 else ""))
    print(f"  {label:<20} {r:>+8.3f}  {slope:>+8.4f}  {p:>8.4f}{sig}  {r**2:>8.3f}")

# Fig 10: nightly regression
fig10 = plt.figure(figsize=(14, 9), constrained_layout=True)
fig10.suptitle(f"Fig 10 · Nightly-Level Regression Analysis  (n = {len(summ)})",
               fontsize=13, fontweight="bold")
gs10 = gridspec.GridSpec(2, 4, figure=fig10)

colors10 = ["#42A5F5", "#EF5350", "#FF7043", "#26A69A"]

for idx, (name, label) in enumerate(FEATURES.items()):
    ax = fig10.add_subplot(gs10[0, idx])
    res = uni_results[name]
    x_col = summ[name].values

    ax.scatter(x_col, y, color=colors10[idx], s=80, zorder=3, alpha=0.9)
    for xi, yi, lbl in zip(x_col, y, xlabs):
        ax.annotate(lbl, (xi, yi), textcoords="offset points",
                    xytext=(4, 4), fontsize=7, color="#555")

    x_line = np.linspace(x_col.min(), x_col.max(), 100)
    ax.plot(x_line, res["slope"] * x_line + res["intercept"],
            color=colors10[idx], lw=2)

    sig = "**" if res["p"] < 0.01 else ("*" if res["p"] < 0.05 else
          ("~" if res["p"] < 0.10 else ""))
    ax.set_title(f"{label}\nr={res['r']:+.2f}{sig}  p={res['p']:.3f}", fontsize=9)
    ax.set_xlabel(label, fontsize=8)
    ax.set_ylabel("Mean Sleep Score" if idx == 0 else "", fontsize=8)
    ax.grid(True, alpha=0.3)

# Standardised coefficients bar
ax_coef = fig10.add_subplot(gs10[1, :2])
feat_labels = list(FEATURES.values())
std_coefs   = ols.coef_
bar_cols     = [colors10[i] for i in range(len(FEATURES))]
bars = ax_coef.barh(feat_labels, std_coefs, color=bar_cols, alpha=0.85,
                    edgecolor="white")
ax_coef.axvline(0, color="black", lw=0.8)
for bar, v in zip(bars, std_coefs):
    ax_coef.text(v + (0.005 if v >= 0 else -0.005), bar.get_y() + bar.get_height() / 2,
                 f"{v:+.3f}", va="center", ha="left" if v >= 0 else "right", fontsize=9)
ax_coef.set_xlabel("Standardised Coefficient (β)", fontsize=9)
ax_coef.set_title(f"Full OLS Standardised Coefficients  (R2={r2_ols:.2f}, Adj-R2={adj_r2:.2f})",
                  fontsize=10)
ax_coef.grid(axis="x", alpha=0.3)

# Actual vs Predicted
ax_avp = fig10.add_subplot(gs10[1, 2:])
ax_avp.scatter(y, y_pred_ols, color="#5C6BC0", s=80, zorder=3)
for yi, ypi, lbl in zip(y, y_pred_ols, xlabs):
    ax_avp.annotate(lbl, (yi, ypi), textcoords="offset points",
                    xytext=(4, 4), fontsize=7)
lims = [min(y.min(), y_pred_ols.min()) - 0.03,
        max(y.max(), y_pred_ols.max()) + 0.03]
ax_avp.plot(lims, lims, "k--", lw=1, alpha=0.5, label="Perfect fit")
ax_avp.set_xlabel("Actual Sleep Score", fontsize=9)
ax_avp.set_ylabel("Predicted Sleep Score", fontsize=9)
ax_avp.set_title("Actual vs Predicted (Full OLS)", fontsize=10)
ax_avp.legend(fontsize=8)
ax_avp.grid(True, alpha=0.3)

fig10.savefig(os.path.join(PLOTS, "fig10_nightly_regression.png"),
              dpi=150, bbox_inches="tight")
print("\nSaved: fig10_nightly_regression.png")


# Level A2 — Ridge + Leave-One-Out CV
print("\n" + "=" * 60)
print(f"RIDGE + LOOCV  (unbiased performance with n={len(summ)})")
print("=" * 60)

alphas     = [0.01, 0.1, 1, 10, 100]
loo        = LeaveOneOut()
loocv_res  = {}

for alpha in alphas:
    y_loo, y_true_loo = [], []
    for train_idx, test_idx in loo.split(X_std):
        ridge = Ridge(alpha=alpha)
        ridge.fit(X_std[train_idx], y[train_idx])
        y_loo.append(ridge.predict(X_std[test_idx])[0])
        y_true_loo.append(y[test_idx][0])
    rmse  = np.sqrt(mean_squared_error(y_true_loo, y_loo))
    loocv_res[alpha] = {"rmse": rmse, "pred": y_loo, "true": y_true_loo}
    print(f"  α={alpha:>6}  LOOCV RMSE={rmse:.4f}")

best_alpha = min(loocv_res, key=lambda a: loocv_res[a]["rmse"])
print(f"\n  Best α={best_alpha}  RMSE={loocv_res[best_alpha]['rmse']:.4f}")

ridge_best = Ridge(alpha=best_alpha).fit(X_std, y)

# Fig 11: LOOCV results
fig11, axes11 = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
fig11.suptitle(f"Fig 11 · Ridge Regression + LOOCV  (best α={best_alpha})",
               fontsize=13, fontweight="bold")

# RMSE vs alpha
rmse_vals = [loocv_res[a]["rmse"] for a in alphas]
axes11[0].plot(alphas, rmse_vals, "o-", color="#5C6BC0", lw=2, ms=8)
axes11[0].axvline(best_alpha, color="#EF5350", ls="--", lw=1.5,
                  label=f"Best α={best_alpha}")
axes11[0].set_xscale("log")
axes11[0].set_xlabel("Regularisation α (log scale)", fontsize=9)
axes11[0].set_ylabel("LOOCV RMSE", fontsize=9)
axes11[0].set_title("Alpha Selection via LOOCV", fontsize=10)
axes11[0].legend(fontsize=8)
axes11[0].grid(True, alpha=0.3)

# LOOCV actual vs predicted
y_true_b = loocv_res[best_alpha]["true"]
y_pred_b = loocv_res[best_alpha]["pred"]
axes11[1].scatter(y_true_b, y_pred_b, color="#26A69A", s=80, zorder=3)
for yt, yp, lbl in zip(y_true_b, y_pred_b, xlabs):
    axes11[1].annotate(lbl, (yt, yp), textcoords="offset points",
                       xytext=(4, 4), fontsize=7)
lims2 = [min(y_true_b + y_pred_b) - 0.03, max(y_true_b + y_pred_b) + 0.03]
axes11[1].plot(lims2, lims2, "k--", lw=1, alpha=0.5)
axes11[1].set_xlabel("Actual", fontsize=9)
axes11[1].set_ylabel("LOOCV Predicted", fontsize=9)
axes11[1].set_title("LOOCV: Actual vs Predicted", fontsize=10)
axes11[1].grid(True, alpha=0.3)

# Ridge coefficients vs alpha
coef_by_alpha = []
for alpha in alphas:
    Ridge(alpha=alpha).fit(X_std, y)
    coef_by_alpha.append(Ridge(alpha=alpha).fit(X_std, y).coef_)
coef_by_alpha = np.array(coef_by_alpha)

for i, label in enumerate(feat_labels):
    axes11[2].plot(alphas, coef_by_alpha[:, i], "o-", label=label,
                   lw=1.8, ms=6)
axes11[2].axvline(best_alpha, color="#EF5350", ls="--", lw=1.2)
axes11[2].axhline(0, color="black", lw=0.6)
axes11[2].set_xscale("log")
axes11[2].set_xlabel("α (log scale)", fontsize=9)
axes11[2].set_ylabel("Standardised Coefficient", fontsize=9)
axes11[2].set_title("Coefficient Shrinkage Path", fontsize=10)
axes11[2].legend(fontsize=7.5)
axes11[2].grid(True, alpha=0.3)

fig11.savefig(os.path.join(PLOTS, "fig11_loocv_ridge.png"),
              dpi=150, bbox_inches="tight")
print("Saved: fig11_loocv_ridge.png")


plt.show()
print("\nStep 5 complete.")
