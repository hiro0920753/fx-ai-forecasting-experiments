# FX AI Forecasting Experiments

Reproducible Python experiments for evaluating whether an LSTM can forecast
short-horizon foreign-exchange price movements.

The project is organized as scripts that can be executed in numerical order.
It focuses on chronological evaluation rather than claims of profitability.

## Pipeline

```text
load an exported price CSV
        ↓
validate it and create sequences
        ↓
train an LSTM on one chronological split
        ↓
create a PNG chart and Excel workbook
        ↓
run an expanding walk-forward evaluation
```

## Setup

```powershell
git clone https://github.com/hiro0920753/fx-ai-forecasting-experiments.git
cd fx-ai-forecasting-experiments
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
```

## 1. Provide a price CSV

Export historical data from a source such as a trading terminal or market-data
provider and save it as `data/USDJPY_M5.csv`. This repository deliberately does
not tie the experiment to one vendor or publishing site.

## 2. Validate the CSV and create a dataset

```powershell
python scripts/01_prepare_dataset.py `
  --csv data/USDJPY_M5.csv `
  --lookback 288 `
  --horizon 6 `
  --output data/dataset_24h_30m.npz `
  --metadata data/dataset_24h_30m.json
```

For five-minute bars, 288 bars represent 24 hours and six bars represent a
30-minute forecast horizon. The NPZ file contains `x`, `y`, current prices, and
timestamps. The JSON file records shapes, periods, and source information.

## 3. Train and evaluate one split

```powershell
python scripts/02_train_lstm.py `
  --dataset data/dataset_24h_30m.npz `
  --epochs 20 `
  --output artifacts/baseline
```

Outputs:

- `metrics.json`: LSTM and baseline metrics
- `predictions.csv`: all test predictions
- `model.pt`: selected model weights

## 4. Create a chart and Excel report

```powershell
python scripts/03_make_report.py `
  --input artifacts/baseline `
  --output reports/baseline
```

Outputs:

- `prediction_chart.png`
- `experiment_report.xlsx`

The workbook contains Summary, Predictions, and Configuration sheets.

## 5. Run Walk-Forward evaluation

```powershell
python scripts/04_walk_forward.py `
  --csv data/USDJPY_M5.csv `
  --lookback 288 `
  --horizon 6 `
  --train-rows 10000 `
  --test-rows 2000 `
  --step-rows 2000 `
  --folds 3 `
  --epochs 10 `
  --output artifacts/walk_forward
```

Each fold trains on all data available before that fold, reserves the newest
15% of the training portion for early stopping, and evaluates the following
unused block. Preprocessing and model weights are rebuilt in every fold.

Outputs:

- `fold_summary.csv`: period and metrics for each fold
- `oos_predictions.csv`: concatenated out-of-sample predictions
- `summary.json`: aggregate OOS metrics

## Expected CSV format

At minimum, a timestamp and close-price column are required.

```csv
timestamp,open,high,low,close
2024-01-02 00:00:00,140.90,140.93,140.88,140.91
2024-01-02 00:05:00,140.91,140.96,140.90,140.94
```

Rows are sorted by timestamp. Duplicate timestamps and invalid close prices are
rejected rather than silently repaired.

## Evaluation policy

- Preprocessing statistics are fitted on training data only.
- Validation data selects the early-stopping checkpoint.
- Test and walk-forward blocks are not used to fit model weights.
- LSTM results are compared with random-walk and recent-return baselines.
- RMSE, MAE, and directional accuracy are reported separately.

## Disclaimer

This project is for research and education. It is not financial advice and does
not guarantee trading profitability.

