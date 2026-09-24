import numpy as np
import pandas as pd

from src.costs import (
    add_weight_changes,
    calculate_daily_turnover,
    calculate_transaction_costs,
)


def test_turnover_calculation():
    """
    Verify portfolio turnover from
    changes in executed weights.
    """

    data = pd.DataFrame(
        {
            "date": [
                "2026-01-01",
                "2026-01-02",
                "2026-01-01",
                "2026-01-02",
            ],

            "ticker": [
                "A",
                "A",
                "B",
                "B",
            ],

            "execution_weight": [
                0.20,
                0.10,
                -0.20,
                -0.10,
            ],
        }
    )

    data["date"] = pd.to_datetime(
        data["date"]
    )

    result = add_weight_changes(
        data
    )

    turnover = (
        calculate_daily_turnover(
            result
        )
        .sort_values("date")
        .reset_index(drop=True)
    )

    assert np.isclose(
        turnover.loc[
            0,
            "turnover",
        ],
        0.40,
    )

    assert np.isclose(
        turnover.loc[
            1,
            "turnover",
        ],
        0.20,
    )


def test_transaction_cost_calculation():
    """
    Verify linear transaction-cost model.

    5 bps = 0.0005 per dollar traded.
    """

    turnover = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2026-01-01",
                    "2026-01-02",
                ]
            ),

            "turnover": [
                0.40,
                0.20,
            ],
        }
    )

    result = (
        calculate_transaction_costs(
            turnover=turnover,
            cost_bps=5.0,
        )
    )

    assert np.isclose(
        result.loc[
            0,
            "transaction_cost",
        ],
        0.00020,
    )

    assert np.isclose(
        result.loc[
            1,
            "transaction_cost",
        ],
        0.00010,
    )