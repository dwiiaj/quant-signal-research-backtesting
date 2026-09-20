import pandas as pd


def add_weight_changes(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate position changes for each stock.
    """

    data = data.copy()

    data = (
        data
        .sort_values(
            ["ticker", "date"]
        )
        .reset_index(drop=True)
    )

    data["previous_weight"] = (
        data
        .groupby("ticker")[
            "execution_weight"
        ]
        .shift(1)
        .fillna(0.0)
    )

    data["weight_change"] = (
        data[
            "execution_weight"
        ]
        - data[
            "previous_weight"
        ]
    )

    data[
        "absolute_weight_change"
    ] = (
        data[
            "weight_change"
        ]
        .abs()
    )

    return data


def calculate_daily_turnover(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate total dollars traded per
    dollar of portfolio capital each day.

    Example:

        turnover = 0.25

    means 25 cents were traded for every
    dollar of portfolio capital.
    """

    turnover = (
        data
        .groupby("date")[
            "absolute_weight_change"
        ]
        .sum()
        .rename(
            "turnover"
        )
        .reset_index()
    )

    return turnover


def calculate_transaction_costs(
    turnover: pd.DataFrame,
    cost_bps: float,
) -> pd.DataFrame:
    """
    Apply linear transaction costs.

    cost_bps is interpreted as cost per
    dollar traded.
    """

    turnover = turnover.copy()

    cost_rate = (
        cost_bps
        / 10000.0
    )

    turnover[
        "transaction_cost"
    ] = (
        turnover["turnover"]
        * cost_rate
    )

    return turnover