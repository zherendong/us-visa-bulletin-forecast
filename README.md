# us-visa-bulletin-forecast

Predict US employment-based green card timelines using deep learning. Given a country of chargeability, EB category, and priority date, the model estimates when the Final Action Date and Date for Filing will reach that priority date.

## Features

- **Multi-target prediction**: Trains on all EB categories (EB1-EB4) and countries (China, India, Mexico, Philippines, Rest of World) simultaneously, allowing the model to learn cross-category spillover effects
- **Multiple data sources**: Visa bulletin data, I-485 pending inventory, USCIS processing times, with automated cross-validation between sources
- **Quantile regression**: Outputs optimistic/median/pessimistic time ranges instead of point estimates
- **Time series cross-validation**: Expanding window CV across 7 fiscal year folds for robust evaluation

## Quick Start

### Setup

```bash
# Requires Python 3.13+ and uv
uv venv
uv pip install -e ".[dev]"
```

### Run the Pipeline

```bash
# 1. Fetch all data sources
.venv/bin/python scripts/00_fetch_all_data.py

# 2. Cross-validate data sources
.venv/bin/python scripts/01_validate_data.py

# 3. Process and merge into feature dataset
.venv/bin/python scripts/02_process_data.py

# 4. Train models with time series cross-validation
.venv/bin/python scripts/03_train.py

# 5. Predict a timeline
.venv/bin/python scripts/04_predict.py --country China --eb EB3 --priority-date 2022-01-15
```

### Example Output

```
============================================================
GREEN CARD TIMELINE PREDICTION
============================================================
Country:       China
Category:      EB3
Priority Date: 2022-01-15

Last Known Final Action Date: 2021-06-15
Last Known Filing Date:       2022-01-01

--- FINAL ACTION DATE ---
  Gap to close: 214 days
  Predicted monthly movement: 27.6 days/month (range: -0.0 to 38.5)
  Estimated wait:
    Optimistic:  5 months
    Median:      7 months
    Pessimistic: 25+ years

--- DATE FOR FILING ---
  Status: CURRENT - Your priority date is already current!
```

## Project Structure

```
gc-predict/
├── src/us_visa_bulletin_forecast/
│   ├── data/                        # Data collection and processing
│   │   ├── fetch_visa_bulletin.py       # Download CSV data (DavidBellamy/visa_dates)
│   │   ├── fetch_visa_bulletin_gov.py   # Scrape travel.state.gov (Filing Dates + newer months)
│   │   ├── fetch_i485_inventory.py      # Download USCIS I-485 pending inventory
│   │   ├── fetch_processing_times.py    # Download USCIS processing times (jzebedee/uscis)
│   │   ├── validate_sources.py          # Cross-validate data sources
│   │   └── merge_datasets.py            # Merge into unified monthly dataset
│   ├── features/
│   │   └── engineer.py                  # Feature engineering and train/val/test splits
│   ├── model/
│   │   ├── architecture.py              # LSTM and Temporal Fusion Transformer models
│   │   ├── dataset.py                   # PyTorch Dataset and DataLoader
│   │   ├── train.py                     # Training loop with early stopping
│   │   ├── evaluate.py                  # Metrics and baselines
│   │   └── cross_validate.py            # Time series cross-validation
│   ├── predict/
│   │   └── inference.py                 # Timeline prediction given (country, EB, priority_date)
│   └── viz/
│       └── plots.py                     # Visualization functions
├── scripts/                         # Pipeline scripts (run in order)
├── data/                            # Raw and processed data (gitignored)
├── models/                          # Saved checkpoints (gitignored)
└── figures/                         # Generated plots (gitignored)
```

## Data Sources

| Source | Description | Frequency | Coverage |
|--------|-------------|-----------|----------|
| [DavidBellamy/visa_dates](https://github.com/DavidBellamy/visa_dates) | Visa bulletin Final Action Dates as CSV | Monthly | 2003 - present |
| [travel.state.gov](https://travel.state.gov/content/travel/en/legal/visa-law0/visa-bulletin.html) | Official visa bulletin (Final Action + Filing Dates) | Monthly | 2015 - present |
| [USCIS I-485 Inventory](https://www.uscis.gov/tools/reports-and-studies/immigration-and-citizenship-data) | Pending I-485 cases by category/country | Quarterly | 2020 - present |
| [jzebedee/uscis](https://github.com/jzebedee/uscis) | USCIS processing times (daily snapshots) | Daily | 2020 - present |

Data is cross-validated before merging: CSV data is compared against government-scraped HTML for sample months, with discrepancies flagged in a validation report.

## Model Architecture

Two models are trained and compared:

**LSTM Baseline**: 2-layer LSTM (hidden=128) with quantile output heads.

**Temporal Fusion Transformer (TFT)**: Input projection, 2-layer LSTM encoder, multi-head self-attention (4 heads), gated residual network, and quantile output.

Both models:
- Take 24 months of historical features as input
- Predict 6 months of movement for all EB x country combinations simultaneously
- Output quantile predictions (q10, q50, q90) for confidence intervals
- Use pinball/quantile loss for training

### Features (46 input features)

- **Action days**: Final Action Date position for each EB x country combination (~20)
- **Filing days**: Date for Filing position for each EB x country (~20, from Oct 2015)
- **Calendar**: Month (sin/cos encoded), fiscal year quarter

### Training

Time series cross-validation with expanding window:

```
Fold 1: Train [2003───2018] Val [FY2019] Test [FY2020]
Fold 2: Train [2003────2019] Val [FY2020] Test [FY2021]
...
Fold 7: Train [2003─────────2024] Val [FY2025] Test [FY2026]
```

The final model (from the last fold) is saved for inference, trained on the most recent data available.

## Model Performance

Cross-validation results (7 folds, all targets averaged):

| Model | MAE (days) | Coverage (80% CI) | Directional Accuracy |
|-------|-----------|-------------------|---------------------|
| LSTM | 53.0 ± 9.8 | 60% ± 15% | 53% ± 16% |
| TFT | 53.1 ± 10.1 | 46% ± 14% | 52% ± 16% |

EB-3 China specifically (final fold, FY2026 test):

| Model | MAE | Coverage |
|-------|-----|----------|
| LSTM | 37.5 days | 81% |
| TFT | 36.7 days | 72% |

## Limitations

- **Small dataset**: ~240 monthly data points spanning 22 years. Limits model complexity.
- **Policy sensitivity**: Visa bulletin movement is heavily influenced by policy changes (executive orders, congressional legislation, COVID-19 processing halts) which cannot be predicted from historical patterns.
- **Filing date predictions**: The model struggles with Date for Filing predictions for categories where filing dates are historically stagnant.
- **Pessimistic tail**: The q10 (pessimistic) predictions often show near-zero or negative movement, leading to "25+ years" estimates. This reflects genuine retrogression risk but may overstate it.

## License

MIT
