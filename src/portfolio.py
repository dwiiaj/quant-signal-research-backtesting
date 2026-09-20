import numpy as np
import pandas as pd

from src.data import load_config


def neutralise_weights(
    selected: pd.DataFrame,
    raw_weights: np.ndarray,
) -> np.ndarray:
    """
    Remove dollar and market-beta exposure
    from a set of raw portfolio weights.

    The adjustment projects the raw weights
    onto the space orthogonal to:

        1. the constant vector
        2. stock market beta

    Therefore the resulting portfolio has
    approximately:

        sum(weights) = 0
        sum(weights * beta) = 0
    """

    beta = (
        selected["beta"]
        .to_numpy(
            dtype=float
        )
    )

    exposure_matrix = np.column_stack(
        [
            np.ones(len(selected)),
            beta,
        ]
    )

    # Projection of weights onto the
    # exposure space.
    coefficients = (
        np.linalg.pinv(
            exposure_matrix.T
            @ exposure_matrix
        )
        @ exposure_matrix.T
        @ raw_weights
    )

    neutral_weights = (
        raw_weights
        - exposure_matrix
        @ coefficients
    )

    return neutral_weights


def construct_daily_weights(
    group: pd.DataFrame,
    long_quantile: float,
    short_quantile: float,
    gross_target: float,
    max_position_weight: float,
) -> pd.DataFrame:
    """
    Construct one day's market-neutral
    long-short portfolio.

    Long:
        strongest reversal signals

    Short:
        weakest reversal signals
    """

    group = group.copy()

    group["target_weight"] = 0.0

    eligible = group[
        group["candidate_rank"].notna()
        & group["beta"].notna()
    ].copy()

    if len(eligible) < 10:
        return group

    long_cutoff = (
        1.0 - long_quantile
    )

    short_cutoff = (
        short_quantile
    )

    longs = eligible[
        eligible["candidate_rank"]
        >= long_cutoff
    ]

    shorts = eligible[
        eligible["candidate_rank"]
        <= short_cutoff
    ]

    if (
        len(longs) == 0
        or len(shorts) == 0
    ):
        return group

    selected = pd.concat(
        [
            longs,
            shorts,
        ]
    ).copy()

    raw_weights = np.zeros(
        len(selected)
    )

    long_mask = (
        selected[
            "candidate_rank"
        ].to_numpy()
        >= long_cutoff
    )

    short_mask = (
        selected[
            "candidate_rank"
        ].to_numpy()
        <= short_cutoff
    )

    # Begin with a 50/50 long-short
    # dollar-neutral portfolio.
    raw_weights[
        long_mask
    ] = (
        gross_target
        / 2.0
        / long_mask.sum()
    )

    raw_weights[
        short_mask
    ] = (
        -gross_target
        / 2.0
        / short_mask.sum()
    )

    # Remove residual dollar and beta
    # exposure simultaneously.
    neutral_weights = (
        neutralise_weights(
            selected,
            raw_weights,
        )
    )

    gross_exposure = (
        np.abs(
            neutral_weights
        ).sum()
    )

    if gross_exposure > 0:

        neutral_weights = (
            neutral_weights
            * (
                gross_target
                / gross_exposure
            )
        )

    # Position-limit risk control.
    #
    # Scaling every position by the same
    # constant preserves neutrality.
    largest_position = (
        np.abs(
            neutral_weights
        ).max()
    )

    if (
        largest_position
        > max_position_weight
    ):

        scale = (
            max_position_weight
            / largest_position
        )

        neutral_weights = (
            neutral_weights
            * scale
        )

    selected[
        "target_weight"
    ] = neutral_weights

    weight_map = dict(
        zip(
            selected["ticker"],
            selected[
                "target_weight"
            ],
        )
    )

    group["target_weight"] = (
        group["ticker"]
        .map(weight_map)
        .fillna(0.0)
    )

    return group


def construct_portfolio_weights(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Construct target portfolio weights
    across every date.
    """

    config = load_config()

    long_quantile = (
        config["signal"][
            "long_quantile"
        ]
    )

    short_quantile = (
        config["signal"][
            "short_quantile"
        ]
    )

    gross_target = (
        config["portfolio"][
            "gross_exposure"
        ]
    )

    max_position_weight = (
        config["portfolio"][
            "max_position_weight"
        ]
    )

    frames = []

    for date, group in data.groupby(
        "date",
        sort=True,
    ):

        daily = (
            construct_daily_weights(
                group=group,
                long_quantile=(
                    long_quantile
                ),
                short_quantile=(
                    short_quantile
                ),
                gross_target=(
                    gross_target
                ),
                max_position_weight=(
                    max_position_weight
                ),
            )
        )

        frames.append(
            daily
        )

    result = pd.concat(
        frames,
        ignore_index=True,
    )

    result = (
        result
        .sort_values(
            ["ticker", "date"]
        )
        .reset_index(drop=True)
    )

    return result


def apply_execution_lag(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Lag portfolio weights by one full
    trading day.

    This prevents the backtest from assuming
    that a signal calculated using today's
    close could also be executed at that
    same closing price.

    execution_weight[t]
        = target_weight[t - 1]
    """

    data = data.copy()

    data[
        "execution_weight"
    ] = (
        data
        .groupby("ticker")[
            "target_weight"
        ]
        .shift(1)
        .fillna(0.0)
    )

    return data