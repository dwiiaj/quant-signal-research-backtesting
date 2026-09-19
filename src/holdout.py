import numpy as np
import pandas as pd

from src.data import (
    PROJECT_ROOT,
    load_config,
)

from src.features import (
    FEATURE_DATA_PATH,
)

from src.signal_screen import (
    calculate_daily_ic,
    calculate_long_short_spread,
    calculate_t_stat,
    construct_candidate_signal,
)


TABLE_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
)


def load_full_feature_data() -> pd.DataFrame:
    """
    Load the full engineered feature dataset.

    The signal is constructed using the complete
    historical sequence, but performance is
    evaluated only in the frozen holdout period.
    """

    if not FEATURE_DATA_PATH.exists():
        raise FileNotFoundError(
            "features.parquet was not found. "
            "Run scripts/build_features.py first."
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

    # Future return is used only as the
    # evaluation target.
    data["forward_return_1d"] = (
        data
        .groupby("ticker")[
            "return_1d"
        ]
        .shift(-1)
    )

    return data


def build_frozen_signal() -> pd.DataFrame:
    """
    Construct the pre-selected signal using
    parameters frozen after development research.
    """

    config = load_config()

    horizon = (
        config["research"][
            "selected_horizon"
        ]
    )

    direction = (
        config["research"][
            "selected_direction"
        ]
    )

    volatility_window = (
        config["signal"][
            "residual_volatility_window"
        ]
    )

    data = load_full_feature_data()

    candidate = construct_candidate_signal(
        data=data,
        horizon=horizon,
        volatility_window=volatility_window,
        direction=direction,
    )

    return candidate


def restrict_to_holdout(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Restrict evaluation to the untouched
    post-development period.
    """

    config = load_config()

    validation_start = pd.Timestamp(
        config["research"][
            "validation_start_date"
        ]
    )

    holdout = data[
        data["date"]
        >= validation_start
    ].copy()

    return holdout


def calculate_holdout_summary(
    holdout: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate IC and long-short spread metrics
    for the frozen holdout period.
    """

    daily_ic = calculate_daily_ic(
        holdout
    )

    spread = calculate_long_short_spread(
        holdout
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

    if annualised_spread_volatility > 0:
        spread_sharpe = (
            annualised_spread_return
            / annualised_spread_volatility
        )
    else:
        spread_sharpe = np.nan

    spread_t_stat = calculate_t_stat(
        spread
    )

    positive_spread_day_pct = (
        (spread > 0).mean()
    )

    summary = pd.DataFrame(
        {
            "metric": [
                "Mean IC",
                "IC t-statistic",
                "Annualised spread return",
                "Annualised spread volatility",
                "Spread Sharpe",
                "Spread t-statistic",
                "Positive spread day percentage",
                "IC observations",
                "Spread observations",
            ],
            "value": [
                mean_ic,
                ic_t_stat,
                annualised_spread_return,
                annualised_spread_volatility,
                spread_sharpe,
                spread_t_stat,
                positive_spread_day_pct,
                len(daily_ic),
                len(spread),
            ],
        }
    )

    return summary


def calculate_yearly_results(
    holdout: pd.DataFrame,
) -> pd.DataFrame:
    """
    Evaluate whether signal behaviour is
    persistent or concentrated in a few years.
    """

    records = []

    holdout = holdout.copy()

    holdout["year"] = (
        holdout["date"].dt.year
    )

    for year, group in holdout.groupby(
        "year"
    ):

        daily_ic = calculate_daily_ic(
            group
        )

        spread = (
            calculate_long_short_spread(
                group
            )
        )

        annualised_return = (
            spread.mean()
            * 252
        )

        annualised_volatility = (
            spread.std(
                ddof=1
            )
            * np.sqrt(252)
        )

        if annualised_volatility > 0:
            sharpe = (
                annualised_return
                / annualised_volatility
            )
        else:
            sharpe = np.nan

        records.append(
            {
                "year": year,
                "mean_ic":
                    daily_ic.mean(),
                "ic_t_stat":
                    calculate_t_stat(
                        daily_ic
                    ),
                "annualised_spread_return":
                    annualised_return,
                "spread_sharpe":
                    sharpe,
                "spread_t_stat":
                    calculate_t_stat(
                        spread
                    ),
                "positive_spread_day_pct":
                    (spread > 0).mean(),
                "observations":
                    len(spread),
            }
        )

    return pd.DataFrame(
        records
    )


def save_holdout_results(
    summary: pd.DataFrame,
    yearly: pd.DataFrame,
):
    """
    Save holdout research tables.
    """

    TABLE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path = (
        TABLE_DIR
        / "holdout_summary.csv"
    )

    yearly_path = (
        TABLE_DIR
        / "holdout_yearly_results.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    yearly.to_csv(
        yearly_path,
        index=False,
    )

    return (
        summary_path,
        yearly_path,
    )