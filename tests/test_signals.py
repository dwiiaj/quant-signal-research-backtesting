import pandas as pd

from src.signals import (
    calculate_reversal_signal,
)


def test_reversal_signal_direction():
    """
    A stock with negative recent residual returns
    should receive a positive reversal signal.

    A stock with positive recent residual returns
    should receive a negative reversal signal.
    """

    dates = pd.date_range(
        "2026-01-01",
        periods=4,
        freq="D",
    )

    data = pd.DataFrame(
        {
            "date":
                list(dates)
                + list(dates),

            "ticker":
                ["LOSER"] * 4
                + ["WINNER"] * 4,

            "residual_return":
                [
                    -0.01,
                    -0.02,
                    -0.01,
                    -0.02,
                    0.01,
                    0.02,
                    0.01,
                    0.02,
                ],
        }
    )

    result = (
        calculate_reversal_signal(
            data=data,
            signal_horizon=2,
            volatility_window=2,
        )
    )

    loser_signal = (
        result[
            result["ticker"]
            == "LOSER"
        ][
            "raw_signal"
        ]
        .dropna()
        .iloc[-1]
    )

    winner_signal = (
        result[
            result["ticker"]
            == "WINNER"
        ][
            "raw_signal"
        ]
        .dropna()
        .iloc[-1]
    )

    assert loser_signal > 0

    assert winner_signal < 0