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

**Coverage:** 9 nights · Feb 26 – Mar 6, 2026 · 241 total observations

## Project Structure
```
├── sleep_2026-MM-DD.csv     # Raw nightly data files
├── data/
│   ├── all_nights_raw.csv   # Merged 241-row dataset
│   └── nightly_summary.csv  # Per-night summary (9 rows)
├── plots/                   # All generated figures (fig1–fig12)
├── step1_preprocess.py      # Data merging & feature engineering
├── step2_visualize.py       # EDA visualisations (fig1–fig3)
├── step3_correlation.py     # Correlation & lag analysis (fig4–fig6)
├── step4_sleep_structure.py # Sleep architecture analysis (fig7–fig9)
├── step5_modelling.py       # Regression & classification (fig10–fig12)
└── step6_dashboard.py       # Interactive Plotly Dash dashboard
```

## Running the Dashboard
```bash
pip install -r requirements.txt
python step6_dashboard.py
# Open http://127.0.0.1:8050
```

## Running the Full Analysis Pipeline
```bash
python step1_preprocess.py      # Must run first — generates data/
python step2_visualize.py
python step3_correlation.py
python step4_sleep_structure.py
python step5_modelling.py
python step6_dashboard.py
```

## Key Findings
- **Indoor humidity** showed the strongest negative correlation with sleep quality (Spearman ρ = −0.717, p = 0.030)
- **Indoor temperature** showed a positive correlation with sleep score (r = +0.698, p = 0.037)
- **Sleep stage transitions**: Core acts as the central hub — REM always returns to Core (100%), Deep never transitions directly to REM (0%)
- **Random Forest** (75% accuracy) significantly outperformed Logistic Regression (57%) for within-night sleep stage classification, suggesting non-linear relationships between environment and sleep
- **Elapsed time** was the dominant feature (43% importance) in within-night classification, confirming the circadian progression of sleep stages
