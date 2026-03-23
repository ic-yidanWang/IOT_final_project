# Nocturnal Microclimate × Sleep Architecture
**Personal IoT Sensing Study** — Imperial College London, IoT Coursework 2026

## Research Question
Does the nocturnal indoor/outdoor microclimate (temperature, humidity, wind speed) measurably affect sleep architecture (Deep / Core / REM stages) and overall sleep quality?

## Data Sources
| Source | Variables | Frequency |
|--------|-----------|-----------|
| Arduino + DHT22 sensor | Indoor temperature & humidity | Every 15 min |
| OpenWeatherMap API | Outdoor temp, humidity, precipitation, wind speed | Every 15 min |
| Apple Watch (Sleep Stages) | Core / Deep / REM + sleep score | Every 15 min |

**Coverage:** 14 nights · Feb 26 – Mar 11, 2026 · 385 total observations
Note: Mar 3 data had a 3-hour gap filled via linear regression imputation.

## Project Structure
```
├── data/
│   ├── sleep_logs/              # Raw nightly CSV files (sleep_2026-MM-DD.csv)
│   ├── all_nights_raw.csv       # Merged 385-row dataset
│   └── nightly_summary.csv      # Per-night summary (14 rows)
├── plots/                       # All generated figures
├── scripts/                     # Data capture & utility scripts
│   ├── data_capture.py          # Alternative capture script
│   └── visualize_sleep.py       # Early-stage visualisation utility
├── step0_descriptive.py         # Descriptive stats & normality tests (fig0a-fig0c)
├── step1_preprocess.py          # Data merging & feature engineering
├── step2_visualize.py           # Scatter plots env vs sleep stages 
├── step3_correlation.py         # Correlation & lag analysis (fig4–fig6)
├── step5_modelling.py           # Regression modelling (fig10–fig11)
├── step6_dashboard.py           # Interactive Streamlit dashboard
├── power_analysis.py            # Statistical power analysis (power_analysis.png)
└── fill_sleep_regression.py     # Mar 3 gap imputation (sleep_visualization.png)
```

## Running the Dashboard
```bash
pip install -r requirements.txt
python -m streamlit run step6_dashboard.py
# Open http://localhost:8501
```

## Running the Full Analysis Pipeline
```bash
python step1_preprocess.py   # Must run first — generates data/
python step0_descriptive.py
python step2_visualize.py
python step3_correlation.py
python step5_modelling.py
python power_analysis.py
# step6_dashboard.py is the interactive front-end, run separately
```

## Analytical Journey

**Data collection & cleaning**
Raw nightly sleep exports from Apple Watch were merged with 15-minute environmental readings from the Arduino DHT22 sensor and OpenWeatherMap API. The Mar 3 recording had a 3-hour gap; missing values were imputed using linear regression on adjacent observations before the night was included in the dataset.

**Descriptive exploration**
Nightly distributions and boxplots were inspected to establish the range and variability of each variable. Indoor temperature was relatively stable (18.6–22.0°C), while outdoor temperature and wind speed showed considerably more night-to-night variation.

**Normality testing → method selection (Fig 0c–0e)**
Shapiro-Wilk tests revealed that Sleep Score and Deep Sleep proportion were mildly non-normal (p = 0.033 and p = 0.035). Given the small sample, the Central Limit Theorem cannot be assumed to hold, so both Spearman rank correlation and Pearson r were computed throughout as complementary measures.

**Cross-night correlation matrix (Fig 4)**
Bivariate correlations between all environmental predictors and sleep outcomes showed directional trends — indoor humidity negatively associated with sleep score (ρ = −0.32), indoor temperature positively associated with deep sleep (ρ = +0.38) — but none reached statistical significance (all p > 0.20). This raised the immediate hypothesis that the study was underpowered rather than the effects being absent.

**Stage-level scatter analysis (Fig 3a–3c)**
Breaking the outcome down by sleep stage showed that REM proportion was insensitive to all environmental variables (all |ρ| ≤ 0.34), whereas sleep score and deep sleep showed the moderate trends noted above. This narrowed the plausible mechanism to slow-wave sleep rather than REM architecture.

**Within-night lag cross-correlation (Fig 5)**
The most informative signal emerged at the within-night timescale. Outdoor temperature coupled most strongly with sleep stage depth at a +15-minute lag (mean r = +0.31), suggesting that the thermal environment *precedes* sleep stage shifts rather than coinciding with them — the closest finding to a causal signal in the study. Indoor temperature peaked at −45 minutes (r = +0.22). Per-night analysis (Fig 6) confirmed that these associations were present on several individual nights but were suppressed by cross-night averaging.

**Exploratory regression (Fig 10–11)**
OLS regression (adjusted R² = 0.27, n = 14) showed signs of multicollinearity and overfitting. Ridge regression with LOOCV was applied as a corrective, confirming indoor humidity and wind speed as the most directionally stable predictors under shrinkage. Both results are treated as exploratory given the sample size.

**Power analysis**
A post-hoc power analysis confirmed that at n = 14, the study has only ~45% power to detect medium-sized effects (|ρ| ≈ 0.35). The entire study is therefore characterised as hypothesis-generating: the directional trends are consistent and interpretable, but confirmation requires a sample of at least 30 nights.
