# wc2026-ml-predictor

A machine learning pipeline to predict the FIFA World Cup 2026 outcomes — built as a portfolio project demonstrating structured ML workflows, feature engineering with business logic, and iterative model improvement.

---

## Overview

This project covers two prediction tasks using the same data ecosystem:

- **Classification** — predict match outcome (Win / Draw / Loss) via Logistic Regression and Random Forest ensemble
- **Regression** — predict exact goal counts per team via Poisson Regression

The classification model powers a full tournament simulator that generates group stage standings and a complete knockout bracket following the official FIFA 2026 structure.

---

## Repository Structure

```
wc2026-ml-predictor/
│
├── data/
│   ├── results_clean.csv       # Cleaned historical match data (49,476 matches)
│   ├── fifa_ranking.csv        # FIFA ranking scraped in real time
│   ├── df_train.csv            # Training dataset (45,524 matches, post-1990, no friendlies)
│   ├── df_full.csv             # Full dataset including future WC matches
│   └── df_wc_predict.csv       # WC 2026 fixture list with frozen pre-tournament features
│
├── models/
│   ├── model_v4_clf_lr.pkl     # Logistic Regression V4 (.predict())
│   ├── model_v4_clf_rf.pkl     # Random Forest V4 (.predict_proba())
│   ├── model_v4_reg_home.pkl   # Poisson Regressor — home goals
│   └── model_v4_reg_away.pkl   # Poisson Regressor — away goals
│
├── notebooks/
│   ├── 01_scraping_ranking.ipynb
│   ├── 02_eda_dataset.ipynb
│   ├── 03_build_dataset.ipynb
│   ├── 04_model_training_v1.ipynb
│   ├── 04b_model_training_v2.ipynb
│   ├── 04c_model_training_v3.ipynb
│   ├── 04d_model_training_v4.ipynb   ← production model
│   ├── 05_simulator.ipynb
│   ├── 05a_simulator_predictive.ipynb
│   └── 05b_simulator_frozen.ipynb    ← production simulator
│
├── docs/
│   └── progresso_projeto_ml_mundial.txt   # Full project log (PT)
│
└── README.md
```

---

## Data Sources

| Source | Description |
|---|---|
| `results.csv` | Historical international football results since 1872 (Kaggle) |
| FIFA Rankings API | Live ranking scraped from `api.fifa.com` via DevTools endpoint discovery |

The FIFA ranking endpoint (`/api/v3/fifarankings/rankings/live`) was identified by inspecting network requests in Chrome DevTools — the `__NEXT_DATA__` script tag on the official page does not contain the ranking list directly.

---

## Getting Started

1. Download `results.csv` from [Kaggle](LINK) and place it in `data/`
2. Run notebooks in order: `01` → `02` → `03` → `04d` → `05b`
3. All datasets and models are generated automatically

---

## Exploratory Data Analysis

EDA was conducted across three notebooks and directly informed modelling decisions:

**Dataset quality (02_eda_dataset.ipynb)**
- 1 duplicate match removed (Gibraltar vs Cayman Islands, same game registered twice)
- 44 NaN scores confirmed as future WC matches — expected and correct
- 306 matches with score ≥ 10 goals — all legitimate historical records, mostly non-FIFA territories excluded by ranking filter

**Match outcome patterns (02_eda_dataset.ipynb)**
- Home win rate: 49% overall, dropping to 44.2% on neutral ground (+7pp away win rate)
- This confirmed the need for `Fator_Casa = 0` (forced neutral) for all World Cup matches in the simulator
- Draw rate increased progressively from ~15% (1870s) to ~24% (post-1990) — modern football is more tactically balanced
- World Cup specifically: H=45.8%, D=22.6%, A=31.7% — more balanced than any other competition, validating the neutral ground assumption

**Feature analysis (03_build_dataset.ipynb)**
- `Diff_Ranking` showed the strongest correlation with outcome (-0.42) but treats all ranking positions as equal — a known limitation
- `Diff_Points` was added as a superior alternative: captures real distance between teams (e.g. Argentina 1902pts vs Qatar 1220pts = 682 points gap vs just -56 positions)
- Correlation between `Diff_Ranking` and `Diff_Points`: **-0.97** — near-perfect multicollinearity → `Diff_Ranking` removed, `Diff_Points` retained
- No other feature pairs exceeded 0.70 correlation — no further redundancy issues
- `Forma_Golos_Sofridos` (goals conceded form) entered the model at ~12% importance — second only to `Diff_Points`

**Confederation analysis (01_scraping_ranking.ipynb)**
- FIFA points distribution is approximately symmetric, not right-skewed
- CONMEBOL: smallest confederation (10 teams) but highest average points (1582) — highest quality density
- UEFA: 25 of the top 50 ranked teams
- Largest points cliff: England (#4) to Brazil (#5) = 75 points — a 1-position difference that `Diff_Ranking` treats identically to any other

---

## Feature Engineering

| Feature | Description |
|---|---|
| `Diff_Points` | FIFA points difference (home − away) — captures real quality gap |
| `Fator_Casa` | 1 = home team playing in own country; -1 = away team at home; 0 = neutral |
| `Forma_Golos_Home/Away` | Mean goals scored in last 10 matches (shift=1, no leakage) |
| `Forma_Pts_Home/Away` | Mean points (3/1/0) in last 10 matches |
| `Forma_Golos_Sofridos_Home/Away` | Mean goals conceded in last 10 matches |
| `Tipo_Competicao` | Categorical: World Cup / Continental / Qualification / Friendly |

**Design decisions:**
- Rolling window of N=10 with `shift(1)` prevents data leakage — each match only sees the 10 matches before it
- Goals conceded capped at 5.0 to remove pre-modern historical outliers (e.g. Denmark averaging 17 goals/game in the 1890s)
- All World Cup matches forced to `Fator_Casa = 0` and `Tipo_Competicao = "Mundial"` during simulation

---

## Model Iterations

The project evolved through four model versions, each motivated by data analysis findings:

| Version | Key change | Accuracy | Log-loss |
|---|---|---|---|
| V1 | Baseline — all data, all features | 57.1% | 0.988 |
| V2 | Removed friendlies + post-1990 cutoff | 57.1% | 0.906 |
| V3 | Added goals conceded features + asymmetric ensemble | **60.7%** | **0.856** |
| V4 | Diff_Points replaces Diff_Ranking + sample weights | 60.7% | 0.885 |

**Baseline (naive — always predict H):** 53.6%

**Production model: V4 Ensemble**
- `.predict()` → Logistic Regression (best accuracy)
- `.predict_proba()` → Random Forest (best log-loss, needed for knockout simulation)

**Training data:** 17,563 competitive matches (post-1990, no friendlies)

**Sample weights by competition type:**
- World Cup: 10 × | Continental: 3 × | Qualification: 2 × | Friendly: 1 ×

---

## Tournament Simulator

Three simulator versions were built, each addressing a different level of data leakage:

| Simulator | Description | Data leakage |
|---|---|---|
| `05_simulator.ipynb` | Uses real results from already-played WC matches | High |
| `05a_simulator_predictive.ipynb` | Predicts all 72 group matches, ignores real results | Medium |
| `05b_simulator_frozen.ipynb` | Features frozen at 2026-06-10, fully blind prediction | **Zero** |

**Production simulator: 05b**

Features are read from `df_wc_predict.csv` — a pre-computed fixture file with each team's form metrics frozen the day before the tournament started. This eliminates contamination from World Cup results entirely.

The knockout bracket follows the official FIFA 2026 structure (32 → 16 → 8 → 4 → Final). The 8 best third-placed teams qualify using a simplified assignment rule (documented limitation).

**Simulator results comparison:**

| Simulator | Champion | Runner-up | 3rd |
|---|---|---|---|
| 05 (hybrid) | Mexico | France | Belgium |
| 05a (predictive) | France | Morocco | Belgium |
| 05b V3 (frozen) | Argentina | Spain | England |
| **05b V4 (frozen — final)** | **Argentina** | **Spain** | **England** |

The difference between simulators illustrates the concrete impact of data leakage — Mexico won with inflated form from 2 early WC victories; Argentina won when features reflected the pre-tournament state.

---

## Key Limitations

**1. Form window (N=10)**
With only 10 matches in the rolling window, 2-3 World Cup results represent 20-30% of the form signal. A larger window (N=20) would reduce this effect without ignoring recent results. Addressed in 05b by freezing features pre-tournament.

**2. Draws are structurally hard to predict**
All model versions predicted 0-1 draws correctly out of 10 in the test set. This is a known open problem in football prediction literature — professional models (FiveThirtyEight, Opta) achieve ~50-55% accuracy on individual matches.

**3. Deterministic convergence**
Random Forest and Logistic Regression learn average patterns. They will always converge toward the most statistically likely outcome — Argentina (#1, best pre-tournament form) wins because it is objectively the strongest team in the data. Introducing genuine variance requires stochastic simulation (e.g. Monte Carlo), which was out of scope for this project.

**4. Small test set**
28 matches (first week of WC 2026) is statistically insufficient for robust conclusions. One match represents 3.6% of accuracy. The dataset grows dynamically as the tournament progresses — full evaluation possible at ~64 matches.

**5. No head-to-head feature**
Historical direct matchups (e.g. Argentina vs Mexico) are not captured. This would be implementable with the existing `results_clean.csv`.

---

## Future Work

- Increase rolling window from N=10 to N=20
- Add head-to-head historical feature
- Implement Elo rating system for dynamic team strength
- Full re-evaluation at tournament completion (~64 matches)
- Dynamic update of `results_clean.csv` with real results as the tournament progresses

---

## Tech Stack

Python · Pandas · Scikit-learn · Requests · BeautifulSoup · Joblib · Matplotlib · Seaborn · Jupyter Notebooks · VS Code
