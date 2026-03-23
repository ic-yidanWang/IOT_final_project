"""
Step 6 - Streamlit Dashboard
"""

import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from scipy.stats import spearmanr, norm as norm_dist
from collections import defaultdict
import streamlit as st

st.set_page_config(
    page_title="Sleep x Microclimate",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE = os.path.dirname(os.path.abspath(__file__))

@st.cache_data
def load_data():
    raw  = pd.read_csv(os.path.join(BASE, "data", "all_nights_raw.csv"))
    summ = pd.read_csv(os.path.join(BASE, "data", "nightly_summary.csv"))
    raw["timestamp"]   = pd.to_datetime(raw["timestamp"])
    summ["night_date"] = pd.to_datetime(summ["night_date"])
    raw["stage_numeric"] = raw["sleep_stage"].map({"REM": 1, "Core": 2, "Deep": 3})
    summ["label"] = summ["night_date"].apply(lambda d: f"{d.month}/{d.day}")
    return raw, summ

raw, summ = load_data()
N = len(summ)
STAGE_COLOR = {"Deep": "#1565C0", "Core": "#42A5F5", "REM": "#CE93D8"}
STAGE_Y     = {"Deep": 3, "Core": 2, "REM": 1}

ENV_VARS = [
    ("inside_temperature_mean",  "Indoor Temp"),
    ("inside_humidity_mean",     "Indoor Humidity"),
    ("outdoor_temperature_mean", "Outdoor Temp"),
    ("outdoor_humidity_mean",    "Outdoor Humidity"),
    ("wind_speed_mean",          "Wind Speed"),
]
SLEEP_VARS = [
    ("mean_sleep_score", "Sleep Score"),
    ("pct_deep",         "Deep %"),
    ("pct_rem",          "REM %"),
    ("sleep_duration_h", "Duration (h)"),
]

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "",
    ["Overview", "Night Explorer", "Correlation", "Sleep Structure", "Power Analysis"],
)
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Study period:** Feb 26 - Mar 11, 2026")
st.sidebar.markdown(f"**Nights recorded:** {N}")
st.sidebar.markdown(f"**Total epochs:** {len(raw)}")

# ============================================================
# PAGE 1 - OVERVIEW
# ============================================================
if page == "Overview":
    st.title("Nocturnal Microclimate x Sleep Architecture")
    st.caption("Personal IoT sensing study · London · Feb-Mar 2026")
    st.markdown("---")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Nights Recorded", N)
    c2.metric("Avg Sleep Duration", f"{summ['sleep_duration_h'].mean():.1f} h")
    c3.metric("Avg Deep Sleep", f"{summ['pct_deep'].mean():.1f} %")
    c4.metric("Avg Indoor Temp", f"{summ['inside_temperature_mean'].mean():.1f} C")
    c5.metric("Avg Outdoor Temp", f"{summ['outdoor_temperature_mean'].mean():.1f} C")
    st.markdown("---")

    st.subheader("Nightly Summary Table")
    display_df = summ[[
        "night_date","sleep_duration_h","mean_sleep_score",
        "pct_deep","pct_core","pct_rem",
        "inside_temperature_mean","inside_humidity_mean",
        "outdoor_temperature_mean","wind_speed_mean"
    ]].copy()
    display_df.columns = [
        "Night","Duration (h)","Sleep Score",
        "Deep %","Core %","REM %",
        "Indoor Temp (C)","Indoor Humidity (%)",
        "Outdoor Temp (C)","Wind Speed (m/s)"
    ]
    display_df["Night"] = display_df["Night"].dt.strftime("%b %d")
    st.dataframe(
        display_df.style
            .format({"Duration (h)":"{:.2f}","Sleep Score":"{:.3f}",
                     "Deep %":"{:.1f}","Core %":"{:.1f}","REM %":"{:.1f}",
                     "Indoor Temp (C)":"{:.1f}","Indoor Humidity (%)":"{:.1f}",
                     "Outdoor Temp (C)":"{:.1f}","Wind Speed (m/s)":"{:.2f}"})
            .background_gradient(subset=["Deep %"], cmap="Blues")
            .background_gradient(subset=["Sleep Score"], cmap="Greens"),
        use_container_width=True, height=530
    )
    st.markdown("---")

    st.subheader("Cross-Night Trends")
    col_left, col_right = st.columns(2)
    with col_left:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=summ["label"], y=summ["sleep_duration_h"],
            name="Duration (h)", mode="lines+markers",
            line=dict(color="#5C6BC0", width=2.5), marker=dict(size=8)))
        fig.add_trace(go.Scatter(x=summ["label"], y=summ["mean_sleep_score"],
            name="Sleep Score", mode="lines+markers",
            line=dict(color="#26A69A", width=2.5), marker=dict(size=8), yaxis="y2"))
        fig.update_layout(title="Sleep Duration & Score",
            yaxis=dict(title="Hours"),
            yaxis2=dict(title="Score", overlaying="y", side="right"),
            legend=dict(x=0.01, y=0.99), height=320, margin=dict(t=40,b=30))
        st.plotly_chart(fig, use_container_width=True)
    with col_right:
        fig2 = go.Figure()
        for stage, col in [("Deep","#1565C0"),("Core","#42A5F5"),("REM","#CE93D8")]:
            fig2.add_trace(go.Bar(name=stage, x=summ["label"],
                y=summ[f"pct_{stage.lower()}"], marker_color=col, opacity=0.87))
        fig2.update_layout(title="Sleep Stage Composition",
            barmode="stack", yaxis_title="%",
            legend=dict(x=0.01,y=0.99), height=320, margin=dict(t=40,b=30))
        st.plotly_chart(fig2, use_container_width=True)

    col_l2, col_r2 = st.columns(2)
    with col_l2:
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=summ["label"], y=summ["inside_temperature_mean"],
            name="Indoor Temp", mode="lines+markers",
            line=dict(color="#EF5350", width=2), marker=dict(size=7)))
        fig3.add_trace(go.Scatter(x=summ["label"], y=summ["outdoor_temperature_mean"],
            name="Outdoor Temp", mode="lines+markers",
            line=dict(color="#FF7043", width=2, dash="dash"), marker=dict(size=7)))
        fig3.update_layout(title="Temperature Comparison",
            yaxis_title="C", height=300, margin=dict(t=40,b=30))
        st.plotly_chart(fig3, use_container_width=True)
    with col_r2:
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(x=summ["label"], y=summ["inside_humidity_mean"],
            name="Indoor Humidity", mode="lines+markers",
            line=dict(color="#42A5F5", width=2), marker=dict(size=7)))
        fig4.add_trace(go.Scatter(x=summ["label"], y=summ["outdoor_humidity_mean"],
            name="Outdoor Humidity", mode="lines+markers",
            line=dict(color="#5C6BC0", width=2, dash="dash"), marker=dict(size=7)))
        fig4.update_layout(title="Humidity Comparison",
            yaxis_title="% RH", height=300, margin=dict(t=40,b=30))
        st.plotly_chart(fig4, use_container_width=True)

# ============================================================
# PAGE 2 - NIGHT EXPLORER
# ============================================================
elif page == "Night Explorer":
    st.title("Night Explorer")
    st.caption("Select a night to view its hypnogram and environmental conditions")

    night_options = {pd.to_datetime(n).strftime("%b %d, %Y"): n
                     for n in sorted(raw["night_date"].unique())}
    selected_label = st.selectbox("Select night:", list(night_options.keys()))
    selected_night = night_options[selected_label]

    grp  = raw[raw["night_date"] == selected_night].copy()
    srow = summ[summ["night_date"] == pd.to_datetime(selected_night)].iloc[0]

    st.markdown("---")
    m1,m2,m3,m4,m5,m6 = st.columns(6)
    m1.metric("Duration",      f"{srow['sleep_duration_h']:.2f} h")
    m2.metric("Sleep Score",   f"{srow['mean_sleep_score']:.3f}")
    m3.metric("Deep Sleep",    f"{srow['pct_deep']:.1f} %")
    m4.metric("REM Sleep",     f"{srow['pct_rem']:.1f} %")
    m5.metric("Indoor Temp",   f"{srow['inside_temperature_mean']:.1f} C")
    m6.metric("Indoor Humid.", f"{srow['inside_humidity_mean']:.1f} %")
    st.markdown("---")

    st.subheader("Sleep Hypnogram")
    t = grp["elapsed_min"].values
    stages = grp["sleep_stage"].values

    fig_hyp = go.Figure()
    for k in range(len(t)):
        x0, x1 = t[k], t[k]+15
        y_val = STAGE_Y.get(stages[k], 0)
        fig_hyp.add_shape(type="rect", x0=x0, x1=x1, y0=0, y1=y_val,
            fillcolor=STAGE_COLOR.get(stages[k], "#ccc"), opacity=0.55, line_width=0)
    y_step, x_step = [], []
    for k in range(len(t)):
        x_step += [t[k], t[k]+15]
        y_step += [STAGE_Y.get(stages[k],0), STAGE_Y.get(stages[k],0)]
    fig_hyp.add_trace(go.Scatter(x=x_step, y=y_step, mode="lines",
        line=dict(color="#263238", width=2), showlegend=False))
    for stage, col in STAGE_COLOR.items():
        fig_hyp.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
            marker=dict(size=12, color=col, symbol="square"), name=stage))
    fig_hyp.update_layout(
        yaxis=dict(tickvals=[1,2,3], ticktext=["REM","Core","Deep"], range=[0.3,3.7]),
        xaxis_title="Minutes from sleep onset",
        height=260, margin=dict(t=10,b=40),
        legend=dict(orientation="h", x=0.5, xanchor="center", y=1.1))
    st.plotly_chart(fig_hyp, use_container_width=True)

    col_env1, col_env2 = st.columns(2)
    with col_env1:
        st.subheader("Indoor Environment")
        fig_ind = make_subplots(specs=[[{"secondary_y": True}]])
        fig_ind.add_trace(go.Scatter(x=grp["elapsed_min"], y=grp["inside_temperature"],
            name="Temp (C)", mode="lines+markers",
            line=dict(color="#EF5350", width=2.5), marker=dict(size=6)), secondary_y=False)
        fig_ind.add_trace(go.Scatter(x=grp["elapsed_min"], y=grp["inside_humidity"],
            name="Humidity (%)", mode="lines+markers",
            line=dict(color="#42A5F5", width=2, dash="dot"), marker=dict(size=6)), secondary_y=True)
        fig_ind.update_yaxes(title_text="Temp (C)", secondary_y=False)
        fig_ind.update_yaxes(title_text="Humidity (%)", secondary_y=True)
        fig_ind.update_layout(height=300, margin=dict(t=20,b=40),
            legend=dict(x=0.01,y=0.99), xaxis_title="Minutes from sleep onset")
        st.plotly_chart(fig_ind, use_container_width=True)
    with col_env2:
        st.subheader("Outdoor Environment")
        fig_out = make_subplots(specs=[[{"secondary_y": True}]])
        fig_out.add_trace(go.Scatter(x=grp["elapsed_min"], y=grp["outdoor_temperature"],
            name="Outdoor Temp (C)", mode="lines+markers",
            line=dict(color="#FF7043", width=2.5), marker=dict(size=6)), secondary_y=False)
        fig_out.add_trace(go.Scatter(x=grp["elapsed_min"], y=grp["wind_speed"],
            name="Wind Speed (m/s)", mode="lines+markers",
            line=dict(color="#26A69A", width=2, dash="dot"), marker=dict(size=6)), secondary_y=True)
        fig_out.update_yaxes(title_text="Temp (C)", secondary_y=False)
        fig_out.update_yaxes(title_text="Wind Speed (m/s)", secondary_y=True)
        fig_out.update_layout(height=300, margin=dict(t=20,b=40),
            legend=dict(x=0.01,y=0.99), xaxis_title="Minutes from sleep onset")
        st.plotly_chart(fig_out, use_container_width=True)

# ============================================================
# PAGE 3 - CORRELATION
# ============================================================
elif page == "Correlation":
    st.title("Correlation Analysis")
    st.caption(f"Pearson r and Spearman ρ reported · n = {N} nights · alpha = 0.05")

    st.info(
        "Both Pearson r and Spearman ρ are computed for all variable pairs. "
        "Environmental variables passed Shapiro-Wilk normality tests (p > 0.05), "
        "while Sleep Score and Deep Sleep proportion showed mild non-normality "
        "(p = 0.033 and p = 0.035 respectively) — likely attributable in part to "
        "the limited sample size (n = 14) rather than a fundamental departure from normality. "
        "Both methods are therefore reported as a robustness check; "
        "their close agreement throughout confirms that conclusions are not sensitive "
        "to the choice of correlation measure."
    )

    st.subheader("Pearson r")
    rows_p = []
    for ev, elbl in ENV_VARS:
        row = {"Variable": elbl}
        for sv, slbl in SLEEP_VARS:
            r, p = stats.pearsonr(summ[ev], summ[sv])
            sig = "**" if p<0.01 else ("*" if p<0.05 else ("~" if p<0.10 else "ns"))
            row[slbl] = f"{r:+.3f} ({sig}, p={p:.3f})"
        rows_p.append(row)
    st.dataframe(pd.DataFrame(rows_p).set_index("Variable"), use_container_width=True)

    st.subheader("Spearman ρ")
    rows_s = []
    for ev, elbl in ENV_VARS:
        row = {"Variable": elbl}
        for sv, slbl in SLEEP_VARS:
            rho, p = spearmanr(summ[ev], summ[sv])
            sig = "**" if p<0.01 else ("*" if p<0.05 else ("~" if p<0.10 else "ns"))
            row[slbl] = f"{rho:+.3f} ({sig}, p={p:.3f})"
        rows_s.append(row)
    st.dataframe(pd.DataFrame(rows_s).set_index("Variable"), use_container_width=True)

    st.caption("** p<0.01  · * p<0.05 · ~ p<0.10 · ns = not significant")

    st.markdown("---")
    st.subheader("Scatter: Environment vs Sleep Outcome")
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        env_choice = st.selectbox("Environmental variable:", [e[1] for e in ENV_VARS])
    with col_s2:
        slp_choice = st.selectbox("Sleep outcome:", [s[1] for s in SLEEP_VARS])

    ev_col = next(e[0] for e in ENV_VARS if e[1] == env_choice)
    sv_col = next(s[0] for s in SLEEP_VARS if s[1] == slp_choice)
    x_vals = summ[ev_col].values
    y_vals = summ[sv_col].values
    rho, p_s = spearmanr(x_vals, y_vals)
    r_p, p_p = stats.pearsonr(x_vals, y_vals)

    fig_sc = go.Figure()
    fig_sc.add_trace(go.Scatter(x=x_vals, y=y_vals, mode="markers+text",
        text=summ["label"], textposition="top right",
        marker=dict(size=11, color="#5C6BC0", opacity=0.85), name="Nights"))
    m, b = np.polyfit(x_vals, y_vals, 1)
    x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
    fig_sc.add_trace(go.Scatter(x=x_line, y=m*x_line+b, mode="lines",
        line=dict(color="#EF5350", width=2, dash="dash"), name="Linear fit"))
    fig_sc.update_layout(
        title=f"{env_choice} vs {slp_choice}  |  Pearson r = {r_p:+.3f} (p={p_p:.3f})  ·  Spearman ρ = {rho:+.3f} (p={p_s:.3f})",
        xaxis_title=env_choice, yaxis_title=slp_choice,
        height=420, margin=dict(t=50,b=40))
    st.plotly_chart(fig_sc, use_container_width=True)

    st.markdown("---")
    st.subheader("Lag Cross-Correlation (Within-Night)")
    st.caption("Positive lag = environment leads sleep · Each step = 15 min")

    LAG_VARS = [
        ("inside_temperature",  "Indoor Temp",    "#EF5350"),
        ("inside_humidity",     "Indoor Humidity","#FF7043"),
        ("outdoor_temperature", "Outdoor Temp",   "#42A5F5"),
        ("wind_speed",          "Wind Speed",     "#26A69A"),
    ]
    lags_steps = list(range(-4, 5))
    lags_min   = [s*15 for s in lags_steps]

    fig_lag = go.Figure()
    lag_summary = []
    for col_name, lbl, color in LAG_VARS:
        rs = []
        for lag in lags_steps:
            night_rs = []
            for _, grp2 in raw.groupby("night_date"):
                grp2 = grp2.sort_values("elapsed_min").reset_index(drop=True)
                e_s = grp2[col_name].values
                s_s = grp2["stage_numeric"].values
                if lag >= 0:
                    e = e_s[:-lag] if lag > 0 else e_s
                    s = s_s[lag:]  if lag > 0 else s_s
                else:
                    e = e_s[-lag:]
                    s = s_s[:len(s_s)+lag]
                if len(e) > 3:
                    r, _ = stats.pearsonr(e, s)
                    night_rs.append(r)
            rs.append(np.mean(night_rs) if night_rs else 0)
        best = int(np.argmax(np.abs(rs)))
        fig_lag.add_trace(go.Scatter(x=lags_min, y=rs, mode="lines+markers",
            name=lbl, line=dict(color=color, width=2.5), marker=dict(size=7)))
        fig_lag.add_trace(go.Scatter(x=[lags_min[best]], y=[rs[best]], mode="markers",
            marker=dict(size=13, color=color, symbol="star"), showlegend=False,
            hovertemplate=f"{lbl}<br>Peak lag={lags_min[best]} min<br>r={rs[best]:.3f}"))
        direction = "Env leads Sleep" if lags_min[best]>0 else ("Sleep leads Env" if lags_min[best]<0 else "Contemporaneous")
        lag_summary.append({"Variable": lbl, "Peak lag (min)": lags_min[best],
            "Peak r": f"{rs[best]:+.3f}", "Direction": direction})

    fig_lag.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.4)
    fig_lag.add_vline(x=0, line_dash="dot", line_color="gray", opacity=0.4)
    fig_lag.update_layout(xaxis_title="Lag (minutes)", yaxis_title="Mean Pearson r",
        height=360, margin=dict(t=20,b=40), legend=dict(x=0.01,y=0.99))
    st.plotly_chart(fig_lag, use_container_width=True)
    st.dataframe(pd.DataFrame(lag_summary).set_index("Variable"), use_container_width=True)

# ============================================================
# PAGE 4 - SLEEP STRUCTURE
# ============================================================
elif page == "Sleep Structure":
    st.title("Sleep Architecture")
    st.caption("Stage transition dynamics · Cycle structure · Efficiency metrics")

    STAGES = ["Deep","Core","REM"]
    tc = defaultdict(lambda: defaultdict(int))
    for _, grp2 in raw.groupby("night_date"):
        s = grp2["sleep_stage"].tolist()
        for k in range(len(s)-1):
            tc[s[k]][s[k+1]] += 1

    mat, annot = [], []
    for frm in STAGES:
        row, arow = [], []
        total = sum(tc[frm].values())
        for to in STAGES:
            p_v = tc[frm][to]/total if total else 0
            row.append(round(p_v,3))
            arow.append(f"{p_v:.2f}<br>n={tc[frm][to]}")
        mat.append(row); annot.append(arow)

    st.subheader("Stage Transition Matrix (Pooled across all nights)")
    fig_tm = go.Figure(go.Heatmap(
        z=mat, x=[f"to {s}" for s in STAGES], y=STAGES,
        colorscale="Blues", zmin=0, zmax=1,
        text=annot, texttemplate="%{text}", textfont={"size":14},
        hovertemplate="From %{y} to %{x}<br>Prob: %{z:.3f}<extra></extra>"))
    fig_tm.update_layout(xaxis_title="Next Stage", yaxis_title="Current Stage",
        height=350, margin=dict(t=20,b=40))
    st.plotly_chart(fig_tm, use_container_width=True)

    st.markdown("---")

    def identify_cycles(grp2):
        stages2 = grp2["sleep_stage"].tolist()
        elapsed2 = grp2["elapsed_min"].tolist()
        cycles, cycle_no, i, nrem_start = [], 0, 0, None
        while i < len(stages2):
            if stages2[i] in ("Core","Deep") and nrem_start is None:
                nrem_start = elapsed2[i]
            if stages2[i] == "REM" and nrem_start is not None:
                rem_start = elapsed2[i]
                j = i
                while j < len(stages2) and stages2[j] == "REM":
                    j += 1
                rem_end = elapsed2[j-1]+15
                cycle_no += 1
                cycles.append({"cycle_no":cycle_no,"nrem_start":nrem_start,
                               "rem_start":rem_start,"rem_end":rem_end})
                nrem_start = None; i = j; continue
            i += 1
        return cycles

    eff_rows = []
    for night, grp2 in raw.groupby("night_date"):
        grp2 = grp2.sort_values("elapsed_min")
        stages2 = grp2["sleep_stage"].values
        cycles = identify_cycles(grp2)
        rem_r  = grp2[grp2["sleep_stage"]=="REM"]
        deep_r = grp2[grp2["sleep_stage"]=="Deep"]
        label = f"{pd.to_datetime(night).month}/{pd.to_datetime(night).day}"
        eff_rows.append({"Night": label,
            "Deep (min)": int((stages2=="Deep").sum()*15),
            "Core (min)": int((stages2=="Core").sum()*15),
            "REM (min)":  int((stages2=="REM").sum()*15),
            "Cycles": len(cycles),
            "First REM (min)": int(rem_r["elapsed_min"].iloc[0]) if len(rem_r) else None,
            "First Deep (min)": int(deep_r["elapsed_min"].iloc[0]) if len(deep_r) else None,
        })
    eff_df = pd.DataFrame(eff_rows).set_index("Night")

    st.subheader("Efficiency Metrics per Night")
    st.dataframe(
        eff_df.style
            .background_gradient(subset=["Deep (min)"], cmap="Blues")
            .background_gradient(subset=["REM (min)"],  cmap="Purples")
            .background_gradient(subset=["Cycles"],     cmap="Greens"),
        use_container_width=True)

    st.markdown("---")
    st.subheader("Stage Duration Composition")
    fig_stack = go.Figure()
    for stage, col in [("Deep","#1565C0"),("Core","#42A5F5"),("REM","#CE93D8")]:
        fig_stack.add_trace(go.Bar(name=stage, x=eff_df.index,
            y=eff_df[f"{stage} (min)"], marker_color=col, opacity=0.87))
    fig_stack.update_layout(barmode="stack", yaxis_title="Minutes",
        height=320, margin=dict(t=20,b=40),
        legend=dict(orientation="h", x=0.5, xanchor="center", y=1.06))
    st.plotly_chart(fig_stack, use_container_width=True)

# ============================================================
# PAGE 5 - POWER ANALYSIS
# ============================================================
elif page == "Power Analysis":
    st.title("Statistical Power Analysis")
    st.caption("Fisher z-transformation method · alpha = 0.05 · Two-tailed")

    def fisher_z(r):     return np.arctanh(r)
    def power_from_n(r, n, alpha=0.05):
        if n <= 3: return 0.0
        ncp = fisher_z(r) * np.sqrt(n-3)
        za2 = norm_dist.ppf(1 - alpha/2)
        return norm_dist.cdf(ncp-za2) + norm_dist.cdf(-ncp-za2)
    def min_n_req(r, alpha=0.05, power=0.80):
        za2 = norm_dist.ppf(1-alpha/2)
        zb  = norm_dist.ppf(power)
        return int(np.ceil(((za2+zb)/fisher_z(r))**2 + 3))

    EFFECTS = [0.3, 0.5, 0.7]
    LABELS  = ["r = 0.30 (small)", "r = 0.50 (medium)", "r = 0.70 (large)"]
    COLORS  = ["#EF5350","#FF9800","#4CAF50"]

    st.subheader("A Priori: Minimum Sample Size Required (power = 0.80)")
    ap_rows = []
    for r, lbl in zip(EFFECTS, LABELS):
        n_req = min_n_req(r)
        pwr_cur = power_from_n(r, N)
        ap_rows.append({
            "Effect size": lbl,
            "Fisher z_r": f"{fisher_z(r):.4f}",
            "Min n required": n_req,
            f"Achieved power (n={N})": f"{pwr_cur:.3f}",
            "Status": "Sufficient" if N >= n_req else f"Need {n_req-N} more nights"
        })
    st.dataframe(pd.DataFrame(ap_rows).set_index("Effect size"), use_container_width=True)

    st.markdown("---")
    st.subheader("Power-Sample Size Sensitivity Curve")
    n_range = np.arange(4, 81)
    fig_pwr = go.Figure()
    for r, lbl, col in zip(EFFECTS, LABELS, COLORS):
        pwr_curve = [power_from_n(r, n) for n in n_range]
        fig_pwr.add_trace(go.Scatter(x=n_range, y=pwr_curve, name=lbl,
            mode="lines", line=dict(color=col, width=2.5)))
        n_req = min_n_req(r)
        fig_pwr.add_trace(go.Scatter(x=[n_req], y=[0.80], mode="markers",
            marker=dict(size=10, color=col, symbol="circle"), showlegend=False,
            hovertemplate=f"{lbl}<br>Min n = {n_req}"))
    fig_pwr.add_vline(x=N, line_dash="dash", line_color="#5C6BC0", line_width=2,
        annotation_text=f"Current study (n={N})", annotation_position="top right")
    fig_pwr.add_hline(y=0.80, line_dash="dot", line_color="#333", line_width=1.5,
        annotation_text="Target power = 0.80", annotation_position="right")
    fig_pwr.add_hrect(y0=0, y1=0.80, fillcolor="red", opacity=0.04, line_width=0)
    fig_pwr.update_layout(
        xaxis_title="Sample size n (number of nights)",
        yaxis_title="Statistical Power (1 - beta)",
        yaxis=dict(range=[0,1.05], dtick=0.1),
        xaxis=dict(range=[4,80]),
        height=420, margin=dict(t=20,b=40),
        legend=dict(x=0.65, y=0.15))
    st.plotly_chart(fig_pwr, use_container_width=True)

    st.markdown("---")
    st.subheader("Interpretation")
    p05 = power_from_n(0.5, N)
    p03 = power_from_n(0.3, N)
    st.info(
        f"At n = {N} nights, this study is adequately powered (1-beta >= 0.80) "
        f"only to detect large correlations (r >= 0.70). "
        f"Medium (r=0.50) and small (r=0.30) effects remain underpowered "
        f"with achieved power of {p05:.2f} and {p03:.2f} respectively. "
        f"Results should be interpreted as exploratory and hypothesis-generating."
    )
