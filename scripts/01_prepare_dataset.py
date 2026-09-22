"""Validate a price CSV and create model-ready sequences."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fx_ai_forecasting.data import load_prices, make_sequences


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/USDJPY_M5.csv")
    parser.add_argument("--lookback", type=int, default=288)
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--output", default="data/dataset_24h_30m.npz")
    parser.add_argument("--metadata", default="data/dataset_24h_30m.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = load_prices(args.csv)
    data = make_sequences(frame, args.lookback, args.horizon)
    output = Path(args.output)
    metadata_path = Path(args.metadata)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        x=data.x,
        y=data.y,
        current=data.current,
        timestamps=data.timestamps,
        lookback=np.asarray(args.lookback),
        horizon=np.asarray(args.horizon),
    )
    metadata = {
        "source_csv": str(Path(args.csv)),
        "rows": len(frame),
        "samples": len(data.y),
        "lookback_bars": args.lookback,
        "horizon_bars": args.horizon,
        "first_timestamp": str(frame["timestamp"].iloc[0]),
        "last_timestamp": str(frame["timestamp"].iloc[-1]),
        "x_shape": list(data.x.shape),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

