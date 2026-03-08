"""
Step 6 — Interactive Dashboard (Plotly Dash)
Project: Nocturnal Microclimate vs Sleep Architecture
------------------------------------------------------
Run:  python step6_dashboard.py
Open: http://127.0.0.1:8050

Tabs:
  Tab 1 — Night Explorer   : hypnogram + env variables for selected night
  Tab 2 — Cross-Night View : trends, stage proportions, scatter correlations
  Tab 3 — Model Insights   : correlation heatmap + RF feature importance + transition matrix
"""

import os
import numpy as np
import pandas as pd
from scipy import stats
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, dcc, html, Input, Output, callback

# ── Load ──────────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
raw  = pd.read_csv(os.path.join(BASE, "data", "all_nights_raw.csv"))
summ = pd.read_csv(os.path.join(BASE, "data", "nightly_summary.csv"))
raw["timestamp"]   = pd.to_datetime(raw["timestamp"])
summ["night_date"] = pd.to_datetime(summ["night_date"])

STAGE_COLOR = {"Deep": "#1565C0", "Core": "#42A5F5", "REM": "#CE93D8"}
raw["stage_numeric"] = raw["sleep_stage"].map({"REM": 0, "Core": 1, "Deep": 2})

nights  = sorted(raw["night_date"].unique())
xlabs   = [f"{pd.to_datetime(d).month}/{pd.to_datetime(d).day}" for d in summ["night_date"]]

# Pre-compute RF importances
B_FEATURES = ["inside_temperature","inside_humidity",
               "outdoor_temperature","wind_speed","elapsed_min"]
B_LABELS   = ["Indoor Temp","Indoor Humidity","Outdoor Temp","Wind Speed","Elapsed Min"]
X_b = StandardScaler().fit_transform(raw[B_FEATURES].values)
rf  = RandomForestClassifier(n_estimators=200, max_depth=6,
                              random_state=42, class_weight="balanced")
rf.fit(X_b, raw["sleep_stage"].values)
importances = rf.feature_importances_

# ══════════════════════════════════════════════════════════════════════════════
# Theme helpers
# ══════════════════════════════════════════════════════════════════════════════
PAGE_BG = "#13131f"
CARD_BG = "#1e1e2e"
TEXT    = "#e0e0f0"
ACC     = "#7c83fd"

CARD = {"backgroundColor": CARD_BG, "borderRadius": "10px",
        "padding": "16px", "marginBottom": "16px"}

def dark(**kw):
    return dict(paper_bgcolor=CARD_BG, plot_bgcolor=PAGE_BG,
                font=dict(color=TEXT, size=11),
                margin=dict(l=50, r=20, t=35, b=40), **kw)

# ══════════════════════════════════════════════════════════════════════════════
# Static figure builders (called once at startup)
# ══════════════════════════════════════════════════════════════════════════════
def make_trend_fig():
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=xlabs, y=summ["sleep_duration_h"], name="Duration (h)",
                             line=dict(color="#5C6BC0", width=2.5), mode="lines+markers",
                             marker=dict(size=7)), secondary_y=False)
    fig.add_trace(go.Scatter(x=xlabs, y=summ["mean_sleep_score"], name="Sleep Score",
                             line=dict(color="#26A69A", width=2.5), mode="lines+markers",
                             marker=dict(size=7)), secondary_y=True)
    fig.update_layout(**dark(legend=dict(x=0.01, y=0.99), height=300))
    fig.update_yaxes(title_text="Hours", secondary_y=False)
    fig.update_yaxes(title_text="Score", secondary_y=True)
    return fig

def make_stage_stack():
    fig = go.Figure()
    for stage, col in [("Deep","#1565C0"),("Core","#42A5F5"),("REM","#CE93D8")]:
        fig.add_trace(go.Bar(name=stage, x=xlabs,
                             y=summ[f"pct_{stage.lower()}"],
                             marker_color=col, opacity=0.87))
    fig.update_layout(**dark(barmode="stack", yaxis_title="%",
                              legend=dict(x=0.01, y=0.99), height=300))
    return fig

def make_corr_heatmap():
    cols   = ["inside_temperature_mean","inside_humidity_mean",
               "outdoor_temperature_mean","wind_speed_mean",
               "mean_sleep_score","pct_deep","pct_rem","sleep_duration_h"]
    labels = ["Indoor Temp","Indoor Humid.","Outdoor Temp","Wind Speed",
               "Sleep Score","Deep %","REM %","Duration"]
    n = len(cols)
    mat = np.zeros((n, n))
    for i, c1 in enumerate(cols):
        for j, c2 in enumerate(cols):
            r, _ = stats.spearmanr(summ[c1], summ[c2])
            mat[i, j] = round(r, 2)
    fig = go.Figure(go.Heatmap(z=mat, x=labels, y=labels,
                                colorscale="RdBu_r", zmin=-1, zmax=1,
                                text=mat, texttemplate="%{text:.2f}",
                                textfont={"size": 10}))
    fig.update_layout(**dark(height=420))
    return fig

def make_feat_imp():
    idx = np.argsort(importances)
    bar_colors = ["#EF5350","#42A5F5","#FF7043","#26A69A","#5C6BC0"]
    fig = go.Figure(go.Bar(
        x=importances[idx],
        y=[B_LABELS[i] for i in idx],
        orientation="h",
        marker_color=[bar_colors[i % len(bar_colors)] for i in idx],
        text=[f"{importances[i]:.3f}" for i in idx],
        textposition="outside",
    ))
    fig.update_layout(**dark(xaxis_title="Feature Importance (Gini)", height=300))
    return fig

def make_transition():
    STAGES = ["Deep","Core","REM"]
    tc = defaultdict(lambda: defaultdict(int))
    for _, grp in raw.groupby("night_date"):
        s = grp["sleep_stage"].tolist()
        for k in range(len(s)-1):
            tc[s[k]][s[k+1]] += 1
    mat, annot = [], []
    for frm in STAGES:
        row, arow = [], []
        total = sum(tc[frm].values())
        for to in STAGES:
            p = tc[frm][to] / total if total else 0
            row.append(round(p, 3))
            arow.append(f"{p:.2f}<br>(n={tc[frm][to]})")
        mat.append(row); annot.append(arow)
    fig = go.Figure(go.Heatmap(
        z=mat, x=[f"to {s}" for s in STAGES], y=[f"{s}" for s in STAGES],
        colorscale="Blues", zmin=0, zmax=1,
        text=annot, texttemplate="%{text}", textfont={"size": 13},
    ))
    fig.update_layout(**dark(height=320,
                              xaxis_title="Next Stage", yaxis_title="Current Stage"))
    return fig

# Pre-build static figures
FIG_TREND      = make_trend_fig()
FIG_STACK      = make_stage_stack()
FIG_CORR       = make_corr_heatmap()
FIG_FEAT       = make_feat_imp()
FIG_TRANSITION = make_transition()

# ══════════════════════════════════════════════════════════════════════════════
# Layout
# ══════════════════════════════════════════════════════════════════════════════
night_opts = [{"label": pd.to_datetime(n).strftime("%b %d, %Y"), "value": n}
              for n in nights]

TAB_STYLE  = {"color": TEXT, "backgroundColor": CARD_BG}
TAB_SEL    = {"color": ACC,  "backgroundColor": CARD_BG, "fontWeight": "bold"}

app = Dash(__name__)
app.title = "Sleep x Microclimate"

app.layout = html.Div(
    style={"backgroundColor": PAGE_BG, "minHeight": "100vh",
           "fontFamily": "Segoe UI, sans-serif", "color": TEXT, "padding": "20px"},
    children=[
        html.H2("Nocturnal Microclimate × Sleep Architecture",
                style={"textAlign": "center", "color": ACC, "marginBottom": "4px"}),
        html.P("Personal IoT Sensing  ·  9 nights  ·  Feb 26 – Mar 6, 2026",
               style={"textAlign": "center", "color": "#888", "marginTop": 0, "marginBottom": "18px"}),

        dcc.Tabs(value="tab-night",
                 colors={"border": "#333", "primary": ACC, "background": CARD_BG},
                 children=[

            # ── Tab 1 ─────────────────────────────────────────────────────────
            dcc.Tab(label="Night Explorer", value="tab-night",
                    style=TAB_STYLE, selected_style=TAB_SEL, children=[
                html.Div(style=CARD, children=[
                    html.Label("Select a night:", style={"color":"#aaa","fontSize":"13px"}),
                    dcc.Dropdown(id="night-select", options=night_opts,
                                 value=str(nights[0]),
                                 clearable=False,
                                 style={"width":"300px","color":"#111"}),
                ]),
                html.Div(style=CARD, children=[
                    html.H4("Sleep Hypnogram", style={"color":ACC,"marginTop":0}),
                    dcc.Graph(id="hypnogram", config={"displayModeBar":False}),
                ]),
                html.Div(style={**CARD,"display":"grid",
                                "gridTemplateColumns":"1fr 1fr","gap":"16px"}, children=[
                    html.Div([html.H4("Indoor Environment",style={"color":ACC,"marginTop":0}),
                              dcc.Graph(id="indoor-env",config={"displayModeBar":False})]),
                    html.Div([html.H4("Outdoor Environment",style={"color":ACC,"marginTop":0}),
                              dcc.Graph(id="outdoor-env",config={"displayModeBar":False})]),
                ]),
            ]),

            # ── Tab 2 ─────────────────────────────────────────────────────────
            dcc.Tab(label="Cross-Night View", value="tab-cross",
                    style=TAB_STYLE, selected_style=TAB_SEL, children=[
                html.Div(style={**CARD,"display":"grid",
                                "gridTemplateColumns":"1fr 1fr","gap":"16px"}, children=[
                    html.Div([html.H4("Duration & Score Trend",style={"color":ACC,"marginTop":0}),
                              dcc.Graph(figure=FIG_TREND, config={"displayModeBar":False})]),
                    html.Div([html.H4("Stage Composition per Night",style={"color":ACC,"marginTop":0}),
                              dcc.Graph(figure=FIG_STACK, config={"displayModeBar":False})]),
                ]),
                html.Div(style=CARD, children=[
                    html.H4("Env Variable vs Sleep Score (Spearman)",
                            style={"color":ACC,"marginTop":0}),
                    dcc.Dropdown(id="env-var-select", clearable=False,
                        options=[
                            {"label":"Indoor Humidity",     "value":"inside_humidity_mean"},
                            {"label":"Indoor Temperature",  "value":"inside_temperature_mean"},
                            {"label":"Outdoor Temperature", "value":"outdoor_temperature_mean"},
                            {"label":"Wind Speed",          "value":"wind_speed_mean"},
                            {"label":"Outdoor Humidity",    "value":"outdoor_humidity_mean"},
                        ],
                        value="inside_humidity_mean",
                        style={"width":"300px","color":"#111","marginBottom":"12px"},
                    ),
                    dcc.Graph(id="scatter-corr", config={"displayModeBar":False}),
                ]),
            ]),

            # ── Tab 3 ─────────────────────────────────────────────────────────
            dcc.Tab(label="Model Insights", value="tab-model",
                    style=TAB_STYLE, selected_style=TAB_SEL, children=[
                html.Div(style={**CARD,"display":"grid",
                                "gridTemplateColumns":"1fr 1fr","gap":"16px"}, children=[
                    html.Div([html.H4("Nightly Correlation Heatmap (Spearman)",
                                      style={"color":ACC,"marginTop":0}),
                              dcc.Graph(figure=FIG_CORR, config={"displayModeBar":False})]),
                    html.Div([html.H4("RF Feature Importance → Sleep Stage",
                                      style={"color":ACC,"marginTop":0}),
                              dcc.Graph(figure=FIG_FEAT, config={"displayModeBar":False})]),
                ]),
                html.Div(style=CARD, children=[
                    html.H4("Stage Transition Matrix (pooled, 9 nights)",
                            style={"color":ACC,"marginTop":0}),
                    dcc.Graph(figure=FIG_TRANSITION, config={"displayModeBar":False}),
                ]),
            ]),
        ]),
    ]
)

# ══════════════════════════════════════════════════════════════════════════════
# Callbacks
# ══════════════════════════════════════════════════════════════════════════════
@callback(
    Output("hypnogram",  "figure"),
    Output("indoor-env", "figure"),
    Output("outdoor-env","figure"),
    Input("night-select","value"),
)
def update_night(night_val):
    grp   = raw[raw["night_date"] == night_val].copy()
    t     = grp["elapsed_min"].values
    label = pd.to_datetime(night_val).strftime("%b %d")
    STAGE_Y = {"REM": 1, "Core": 2, "Deep": 3}

    # Hypnogram
    fig_h = go.Figure()
    for _, row in grp.iterrows():
        fig_h.add_shape(type="rect",
            x0=row["elapsed_min"], x1=row["elapsed_min"]+15,
            y0=0.55, y1=STAGE_Y[row["sleep_stage"]]+0.45,
            fillcolor=STAGE_COLOR[row["sleep_stage"]], opacity=0.4, line_width=0)
    y_vals = [STAGE_Y[s] for s in grp["sleep_stage"]]
    fig_h.add_trace(go.Scatter(x=t, y=y_vals, mode="lines",
        line=dict(color="#eceff1", width=2),
        hovertemplate="<b>%{text}</b><br>%{x} min<extra></extra>",
        text=grp["sleep_stage"].tolist(), showlegend=False))
    for stage, col in STAGE_COLOR.items():
        fig_h.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
            marker=dict(size=11, color=col, symbol="square"), name=stage))
    fig_h.update_layout(**dark(height=240,
        title=dict(text=f"{label} · {grp['sleep time'].iloc[0]}h · "
                        f"score {grp['sleep_score'].mean():.2f}", font=dict(size=12)),
        xaxis_title="Minutes from sleep onset",
        yaxis=dict(tickvals=[1,2,3], ticktext=["REM","Core","Deep"], range=[0.4,3.75]),
        legend=dict(orientation="h", x=0.65, y=1.15),
    ))

    # Indoor
    fig_i = make_subplots(specs=[[{"secondary_y":True}]])
    fig_i.add_trace(go.Scatter(x=t, y=grp["inside_temperature"], name="Temp (°C)",
        line=dict(color="#EF5350",width=2), mode="lines+markers", marker=dict(size=5)),
        secondary_y=False)
    fig_i.add_trace(go.Scatter(x=t, y=grp["inside_humidity"], name="Humidity (%)",
        line=dict(color="#42A5F5",width=2,dash="dash"), mode="lines+markers", marker=dict(size=5)),
        secondary_y=True)
    fig_i.update_layout(**dark(height=280, xaxis_title="Elapsed (min)",
                                legend=dict(x=0.01,y=0.99)))
    fig_i.update_yaxes(title_text="°C", secondary_y=False)
    fig_i.update_yaxes(title_text="%",  secondary_y=True)

    # Outdoor
    fig_o = make_subplots(specs=[[{"secondary_y":True}]])
    fig_o.add_trace(go.Scatter(x=t, y=grp["outdoor_temperature"], name="Temp (°C)",
        line=dict(color="#FF7043",width=2), mode="lines+markers", marker=dict(size=5)),
        secondary_y=False)
    fig_o.add_trace(go.Scatter(x=t, y=grp["wind_speed"], name="Wind (m/s)",
        line=dict(color="#26A69A",width=2,dash="dash"), mode="lines+markers", marker=dict(size=5)),
        secondary_y=True)
    fig_o.update_layout(**dark(height=280, xaxis_title="Elapsed (min)",
                                legend=dict(x=0.01,y=0.99)))
    fig_o.update_yaxes(title_text="°C", secondary_y=False)
    fig_o.update_yaxes(title_text="m/s", secondary_y=True)

    return fig_h, fig_i, fig_o


@callback(Output("scatter-corr","figure"), Input("env-var-select","value"))
def update_scatter(col):
    env_labels = {
        "inside_humidity_mean":     "Indoor Humidity (%)",
        "inside_temperature_mean":  "Indoor Temperature (°C)",
        "outdoor_temperature_mean": "Outdoor Temperature (°C)",
        "wind_speed_mean":          "Wind Speed (m/s)",
        "outdoor_humidity_mean":    "Outdoor Humidity (%)",
    }
    x_v = summ[col].values
    y_v = summ["mean_sleep_score"].values
    r, p = stats.spearmanr(x_v, y_v)
    m, b = np.polyfit(x_v, y_v, 1)
    x_l  = np.linspace(x_v.min(), x_v.max(), 100)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_v, y=y_v, mode="markers+text",
        text=xlabs, textposition="top right", textfont=dict(size=10),
        marker=dict(size=11, color=ACC),
        hovertemplate=f"<b>%{{text}}</b><br>{env_labels.get(col,col)}=%{{x:.2f}}"
                      "<br>Score=%{y:.3f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=x_l, y=m*x_l+b, mode="lines",
        line=dict(color="#EF5350", width=2, dash="dash"),
        name=f"r={r:+.2f}  p={p:.3f}"))
    fig.update_layout(**dark(height=380,
        xaxis_title=env_labels.get(col, col),
        yaxis_title="Mean Sleep Score",
        legend=dict(x=0.01, y=0.99)))
    return fig


if __name__ == "__main__":
    print("\n  Dashboard ready at  http://127.0.0.1:8050\n")
    app.run(debug=False, port=8050)
