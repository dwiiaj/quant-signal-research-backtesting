import numpy as np
import pandas as pd


TRADING_DAYS = 252


def apply_rebalance_schedule(
    data: pd.DataFrame,
    rebalance_every_n_days: int,
) -> pd.DataFrame:
    """
    Convert daily target weights into a
    lower-frequency rebalance schedule.

    Example:
        rebalance_every_n_days = 5

    means portfolio target weights are
    refreshed every fifth trading day and
    otherwise held constant.
    """

    if rebalance_every_n_days < 1:
        raise ValueError(
            "rebalance_every_n_days must be >= 1."
        )

    data = data.copy()

    data = (
        data
        .sort_values(
            ["ticker", "date"]
        )
        .reset_index(drop=True)
    )

    trading_dates = (
        pd.Series(
            data["date"].unique()
        )
        .sort_values()
        .reset_index(drop=True)
    )

    rebalance_dates = set(
        trading_dates.iloc[
            ::rebalance_every_n_days
        ]
    )

    data["rebalance_flag"] = (
        data["date"]
        .isin(
            rebalance_dates
        )
    )

    data[
        "scheduled_target_weight"
    ] = np.where(
        data["rebalance_flag"],
        data["target_weight"],
        np.nan,
    )

    data[
        "scheduled_target_weight"
    ] = (
        data
        .groupby("ticker")[
            "scheduled_target_weight"
        ]
        .ffill()
        .fillna(0.0)
    )

    return data


def apply_scheduled_execution_lag(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Apply one full trading-observation lag
    to scheduled portfolio weights.
    """

    data = data.copy()

    data[
        "execution_weight"
    ] = (
        data
        .groupby("ticker")[
            "scheduled_target_weight"
        ]
        .shift(1)
        .fillna(0.0)
    )

    return data


def calculate_volatility_scaler(
    daily_returns: pd.DataFrame,
    target_volatility: float,
    window: int = 60,
    minimum_scale: float = 0.25,
    maximum_scale: float = 1.0,
) -> pd.DataFrame:
    """
    Calculate an ex-ante volatility scaling
    factor using only historical portfolio
    returns.

    maximum_scale defaults to 1.0, so this
    risk control only de-levers the portfolio
    rather than increasing leverage.
    """

    daily = daily_returns[
        [
            "date",
            "gross_return",
        ]
    ].copy()

    daily = (
        daily
        .sort_values("date")
        .reset_index(drop=True)
    )

    trailing_volatility = (
        daily[
            "gross_return"
        ]
        .rolling(
            window=window,
            min_periods=20,
        )
        .std()
        * np.sqrt(
            TRADING_DAYS
        )
    )

    daily[
        "ex_ante_volatility"
    ] = (
        trailing_volatility
        .shift(1)
    )

    raw_scale = (
        target_volatility
        / daily[
            "ex_ante_volatility"
        ]
    )

    daily[
        "volatility_scale"
    ] = (
        raw_scale
        .clip(
            lower=minimum_scale,
            upper=maximum_scale,
        )
        .fillna(1.0)
    )

    return daily[
        [
            "date",
            "ex_ante_volatility",
            "volatility_scale",
        ]
    ]


def apply_volatility_scaling(
    positions: pd.DataFrame,
    scaler: pd.DataFrame,
) -> pd.DataFrame:
    """
    Uniformly scale all positions on each
    date.

    Uniform scaling preserves both dollar
    neutrality and beta neutrality.
    """

    positions = positions.copy()

    positions = positions.merge(
        scaler,
        on="date",
        how="left",
        validate="many_to_one",
    )

    positions[
        "volatility_scale"
    ] = (
        positions[
            "volatility_scale"
        ]
        .fillna(1.0)
    )

    positions[
        "execution_weight"
    ] = (
        positions[
            "execution_weight"
        ]
        * positions[
            "volatility_scale"
        ]
    )

    return positions