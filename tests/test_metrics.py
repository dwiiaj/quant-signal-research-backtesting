import numpy as np
import pandas as pd

from src.metrics import (
    calculate_drawdown,
    calculate_performance_metrics,
)


def test_drawdown_calculation():
    """
    Verify simple drawdown calculation.
    """

    returns = pd.Series(
        [
            0.10,
            -0.10,
        ]
    )

    drawdown = calculate_drawdown(
        returns
    )

    assert np.isclose(
        drawdown.iloc[0],
        0.0,
    )

    assert np.isclose(
        drawdown.iloc[1],
        -0.10,
    )


def test_performance_metric_keys():
    """
    Performance function should return
    all required portfolio statistics.
    """

    returns = pd.Series(
        [
            0.01,
            -0.005,
            0.008,
            -0.003,
            0.006,
        ]
    )

    metrics = (
        calculate_performance_metrics(
            returns
        )
    )

    expected_keys = {
        "observations",
        "annualised_return",
        "cagr",
        "annualised_volatility",
        "sharpe",
        "sortino",
        "maximum_drawdown",
        "calmar",
        "positive_day_pct",
        "cumulative_return",
    }

    assert (
        set(metrics.keys())
        == expected_keys
    )

    assert (
        metrics[
            "observations"
        ]
        == 5
    )

    assert (
        metrics[
            "maximum_drawdown"
        ]
        <= 0
    )