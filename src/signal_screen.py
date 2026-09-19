import numpy as np
import pandas as pd

from src.data import (
    PROJECT_ROOT,
    load_config,
)

from src.features import (
    FEATURE_DATA_PATH,
)


TABLE_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
)

FIGURE_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "figures"
)


def load_development_data() -> pd.DataFrame:
    """
    Load engineered features and restrict
    the research screen to the development
    sample defined in config.yaml.
    """

    if not FEATURE_DATA_PATH.exists():
        raise FileNotFoundError(
            "features.parquet was not found. "
            "Run scripts/build_features.py first."
        )

    config = load_config()

    development_end = pd.Timestamp(
        config["research"][
            "development_end_date"
        ]
    )

    data = pd.read_parquet(
        FEATURE_DATA_PATH
    )

    data["date"] = pd.to_datetime(
        data["date"]
    )

    data = (
        data
        .sort_values(
            ["ticker", "date"]
        )
        .reset_index(drop=True)
    )

    # Future return used only as the
    # evaluation target.
    data["forward_return_1d"] = (
        data
        .groupby("ticker")[
            "return_1d"
        ]
        .shift(-1)
    )

    data = data[
        data["date"]
        <= development_end
    ].copy()

    return data


def construct_candidate_signal(
    data: pd.DataFrame,
    horizon: int,
    volatility_window: int,
    direction: str,
) -> pd.DataFrame:
    """
    Construct either a residual reversal
    or residual momentum candidate.

    direction:
        "reversal"
        "momentum"
    """

    result = data.copy()

    result["candidate_move"] = (
        result
        .groupby("ticker")[
            "residual_return"
        ]
        .transform(
            lambda x:
            x.rolling(
                window=horizon,
                min_periods=horizon,
            ).sum()
        )
    )

    result[
        "candidate_residual_volatility"
    ] = (
        result
        .groupby("ticker")[
            "residual_return"
        ]
        .transform(
            lambda x:
            x.rolling(
                window=volatility_window,
                min_periods=volatility_window,
            ).std()
        )
    )

    denominator = (
        result[
            "candidate_residual_volatility"
        ]
        * np.sqrt(horizon)
    )

    base_signal = (
        result["candidate_move"]
        / denominator
    )

    if direction == "reversal":

        result[
            "candidate_signal"
        ] = -base_signal

    elif direction == "momentum":

        result[
            "candidate_signal"
        ] = base_signal

    else:

        raise ValueError(
            "direction must be "
            "'reversal' or 'momentum'."
        )

    result[
        "candidate_signal"
    ] = (
        result[
            "candidate_signal"
        ]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
    )

    result[
        "candidate_rank"
    ] = (
        result
        .groupby("date")[
            "candidate_signal"
        ]
        .rank(
            pct=True,
            method="average",
        )
    )

    return result


def calculate_daily_ic(
    data: pd.DataFrame,
) -> pd.Series:
    """
    Calculate daily Spearman IC.
    """

    valid = data.dropna(
        subset=[
            "candidate_signal",
            "forward_return_1d",
        ]
    )

    records = []

    for date, group in valid.groupby(
        "date"
    ):

        if len(group) < 10:
            continue

        if (
            group[
                "candidate_signal"
            ].nunique() < 2
        ):
            continue

        ic = group[
            "candidate_signal"
        ].corr(
            group[
                "forward_return_1d"
            ],
            method="spearman",
        )

        records.append(ic)

    return pd.Series(
        records,
        dtype=float,
    )


def calculate_long_short_spread(
    data: pd.DataFrame,
) -> pd.Series:
    """
    Create daily equal-weight Q5-minus-Q1
    return spread.
    """

    valid = data.dropna(
        subset=[
            "candidate_rank",
            "forward_return_1d",
        ]
    ).copy()

    valid["quintile"] = np.ceil(
        valid["candidate_rank"]
        * 5
    )

    valid["quintile"] = (
        valid["quintile"]
        .clip(
            lower=1,
            upper=5,
        )
        .astype(int)
    )

    daily_returns = (
        valid
        .groupby(
            [
                "date",
                "quintile",
            ]
        )[
            "forward_return_1d"
        ]
        .mean()
        .unstack(
            "quintile"
        )
    )

    if (
        1 not in daily_returns.columns
        or
        5 not in daily_returns.columns
    ):
        return pd.Series(
            dtype=float
        )

    spread = (
        daily_returns[5]
        - daily_returns[1]
    )

    return spread.dropna()


def calculate_t_stat(
    series: pd.Series,
) -> float:
    """
    Calculate t-statistic of the
    sample mean.
    """

    series = series.dropna()

    n = len(series)

    if n < 2:
        return np.nan

    std = series.std(
        ddof=1
    )

    if (
        pd.isna(std)
        or std == 0
    ):
        return np.nan

    return (
        series.mean()
        /
        (
            std
            / np.sqrt(n)
        )
    )


def evaluate_candidate(
    data: pd.DataFrame,
    horizon: int,
    volatility_window: int,
    direction: str,
) -> dict:
    """
    Evaluate one signal candidate.
    """

    candidate = (
        construct_candidate_signal(
            data=data,
            horizon=horizon,
            volatility_window=(
                volatility_window
            ),
            direction=direction,
        )
    )

    daily_ic = calculate_daily_ic(
        candidate
    )

    spread = (
        calculate_long_short_spread(
            candidate
        )
    )

    mean_ic = daily_ic.mean()

    ic_t_stat = calculate_t_stat(
        daily_ic
    )

    mean_daily_spread = (
        spread.mean()
    )

    annualised_spread_return = (
        mean_daily_spread
        * 252
    )

    annualised_spread_volatility = (
        spread.std(
            ddof=1
        )
        * np.sqrt(252)
    )

    if (
        annualised_spread_volatility
        > 0
    ):

        spread_sharpe = (
            annualised_spread_return
            /
            annualised_spread_volatility
        )

    else:

        spread_sharpe = np.nan

    spread_t_stat = (
        calculate_t_stat(
            spread
        )
    )

    positive_spread_days = (
        (spread > 0).mean()
    )

    return {
        "horizon_days":
            horizon,

        "direction":
            direction,

        "mean_ic":
            mean_ic,

        "ic_t_stat":
            ic_t_stat,

        "annualised_spread_return":
            annualised_spread_return,

        "annualised_spread_volatility":
            annualised_spread_volatility,

        "spread_sharpe":
            spread_sharpe,

        "spread_t_stat":
            spread_t_stat,

        "positive_spread_day_pct":
            positive_spread_days,

        "ic_observations":
            len(daily_ic),

        "spread_observations":
            len(spread),
    }


def run_signal_screen() -> pd.DataFrame:
    """
    Evaluate the deliberately small
    candidate signal grid.
    """

    config = load_config()

    horizons = (
        config["research"][
            "candidate_horizons"
        ]
    )

    volatility_window = (
        config["signal"][
            "residual_volatility_window"
        ]
    )

    data = load_development_data()

    records = []

    for horizon in horizons:

        for direction in [
            "reversal",
            "momentum",
        ]:

            print(
                f"Testing "
                f"{direction:8s} | "
                f"{horizon:3d}-day horizon"
            )

            result = evaluate_candidate(
                data=data,
                horizon=horizon,
                volatility_window=(
                    volatility_window
                ),
                direction=direction,
            )

            records.append(
                result
            )

    summary = pd.DataFrame(
        records
    )

    return summary


def save_signal_screen(
    summary: pd.DataFrame,
):
    """
    Save development-period
    screening results.
    """

    TABLE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        TABLE_DIR
        / "development_signal_screen.csv"
    )

    summary.to_csv(
        output_path,
        index=False,
    )

    return output_path