"""Create a PNG chart and an Excel workbook from experiment outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from openpyxl.formatting.rule import ColorScaleRule
from openpyxl.styles import Font


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="artifacts/baseline")
    parser.add_argument("--output", default="reports/baseline")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = Path(args.input)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    predictions = pd.read_csv(source / "predictions.csv", parse_dates=["timestamp"])
    metrics = json.loads((source / "metrics.json").read_text(encoding="utf-8"))

    chart_frame = predictions.tail(min(500, len(predictions))).copy()
    figure, axes = plt.subplots(2, 1, figsize=(12, 8))
    axes[0].plot(chart_frame["timestamp"], chart_frame["actual_log_return"], label="Actual")
    axes[0].plot(
        chart_frame["timestamp"], chart_frame["predicted_log_return"], label="LSTM", alpha=0.8
    )
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_title("Actual vs predicted log return (latest test samples)")
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Log return")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].scatter(
        predictions["actual_log_return"],
        predictions["predicted_log_return"],
        s=12,
        alpha=0.45,
    )
    lower = min(predictions["actual_log_return"].min(), predictions["predicted_log_return"].min())
    upper = max(predictions["actual_log_return"].max(), predictions["predicted_log_return"].max())
    axes[1].plot([lower, upper], [lower, upper], linestyle="--", color="black", linewidth=1)
    axes[1].set_title("All test samples")
    axes[1].set_xlabel("Actual log return")
    axes[1].set_ylabel("Predicted log return")
    axes[1].grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output / "prediction_chart.png", dpi=180)
    plt.close(figure)

    summary_rows = []
    for model_name in ("lstm", "random_walk", "moving_average_return"):
        row = {"model": model_name}
        row.update(metrics[model_name])
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    workbook_path = output / "experiment_report.xlsx"
    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="Summary", index=False)
        predictions.to_excel(writer, sheet_name="Predictions", index=False)
        configuration = pd.DataFrame(
            [{"key": key, "value": value} for key, value in metrics["configuration"].items()]
        )
        configuration.to_excel(writer, sheet_name="Configuration", index=False)
        for sheet in writer.book.worksheets:
            sheet.freeze_panes = "A2"
            sheet.auto_filter.ref = sheet.dimensions
            for cell in sheet[1]:
                cell.font = Font(bold=True)
            for column in sheet.columns:
                width = min(max(len(str(cell.value or "")) for cell in column) + 2, 45)
                sheet.column_dimensions[column[0].column_letter].width = width
        prediction_sheet = writer.book["Predictions"]
        prediction_sheet.conditional_formatting.add(
            f"B2:C{prediction_sheet.max_row}",
            ColorScaleRule(
                start_type="min", start_color="F8696B",
                mid_type="percentile", mid_value=50, mid_color="FFEB84",
                end_type="max", end_color="63BE7B",
            ),
        )
    print(f"chart={output / 'prediction_chart.png'}")
    print(f"excel={workbook_path}")


if __name__ == "__main__":
    main()

