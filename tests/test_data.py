import numpy as np
import pandas as pd

from fx_ai_forecasting.data import chronological_split, make_sequences


def test_sequences_and_chronological_split() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=100, freq="5min", tz="UTC"),
            "close": 150.0 + np.arange(100) * 0.01,
        }
    )
    data = make_sequences(frame, lookback=12, horizon=6)
    train, valid, test = chronological_split(data)
    assert data.x.shape == (83, 12, 1)
    assert train.timestamps[-1] < valid.timestamps[0] < test.timestamps[0]

