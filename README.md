# FX AI Forecasting Experiments

Reproducible experiments for evaluating whether neural networks can forecast
short-horizon foreign-exchange price movements.

This repository focuses on evaluation rather than claims of profitability. It
uses chronological data splits, fits preprocessing only on training data, and
compares neural-network predictions with simple baselines.

## Scope

- CSV input with a timestamp column and OHLC prices
- LSTM price-regression experiment
- Random-walk and moving-average baselines
- RMSE, MAE, and directional accuracy
- Chronological train/validation/test split
- Reproducible seeds and machine-readable result files

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
fx-forecast --csv path\to\USDJPY_M5.csv --output artifacts\baseline
```

The timestamp and close-price columns are detected automatically when common
names are used. They can also be specified explicitly:

```powershell
fx-forecast `
  --csv path\to\prices.csv `
  --timestamp-column time `
  --close-column bid_close `
  --lookback 288 `
  --horizon 6 `
  --output artifacts\lstm_24h_30m
```

For five-minute bars, `--lookback 288` represents 24 hours and `--horizon 6`
represents 30 minutes.

## Expected CSV format

At minimum, the file must contain a timestamp and a close-price column.

```csv
timestamp,open,high,low,close
2024-01-02 00:00:00,140.90,140.93,140.88,140.91
2024-01-02 00:05:00,140.91,140.96,140.90,140.94
```

Rows are sorted by timestamp. Duplicate timestamps and non-numeric close prices
are rejected instead of being silently repaired.

## Evaluation policy

The default split is 70% training, 15% validation, and 15% test in chronological
order. The test period is not used for model selection. Directional accuracy is
the percentage of samples for which the predicted and actual price changes have
the same sign.

Results are saved as JSON and predictions as CSV under the requested output
directory.

## Disclaimer

This project is for research and education. It is not financial advice and does
not guarantee trading profitability.

