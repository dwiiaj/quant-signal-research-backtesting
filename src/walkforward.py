from pathlib import Path

import matplotlib.pyplot as plt
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

FIGURE_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "figures"
)


def load_walk_forward_data() -> pd.DataFrame:
    """
    Load engineered features for
    walk-forward analysis.
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

    # Future return is an evaluation target,
    # never an input to the signal.
    data["forward_return_1d"] = (
        data
        .groupby("ticker")[
            "return_1d"
        ]
        .shift(-1)
    )

    return data


def evaluate_candidate_period(
    full_data: pd.DataFrame,
    horizon: int,
    direction: str,
    volatility_window: int,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> tuple[
    dict,
    pd.Series,
]:
    """
    Evaluate one candidate signal over
    a specified period.

    Additional observations immediately
    before start_date are retained only
    so rolling signal calculations have
    sufficient historical context.
    """

    # 180 calendar days is comfortably
    # longer than our maximum 60-trading-day
    # signal formation horizon.
    context_start = (
        start_date
        - pd.Timedelta(
            days=180
        )
    )

    context = full_data[
        (
            full_data["date"]
            >= context_start
        )
        &
        (
            full_data["date"]
            <= end_date
        )
    ].copy()

    candidate = (
        construct_candidate_signal(
            data=context,
            horizon=horizon,
            volatility_window=(
                volatility_window
            ),
            direction=direction,
        )
    )

    evaluation = candidate[
        (
            candidate["date"]
            >= start_date
        )
        &
        (
            candidate["date"]
            <= end_date
        )
    ].copy()

    daily_ic = calculate_daily_ic(
        evaluation
    )

    spread = (
        calculate_long_short_spread(
            evaluation
        )
    )

    mean_ic = (
        daily_ic.mean()
    )

    ic_t_stat = (
        calculate_t_stat(
            daily_ic
        )
    )

    annualised_spread_return = (
        spread.mean()
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

    positive_spread_day_pct = (
        (spread > 0)
        .mean()
    )

    summary = {
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
            positive_spread_day_pct,

        "ic_observations":
            len(daily_ic),

        "spread_observations":
            len(spread),
    }

    return (
        summary,
        spread,
    )


def select_training_candidate(
    data: pd.DataFrame,
    training_start: pd.Timestamp,
    training_end: pd.Timestamp,
    horizons: list[int],
    volatility_window: int,
) -> pd.DataFrame:
    """
    Evaluate the candidate grid using
    training data only.

    Selection rule:
        highest training IC t-statistic.
    """

    records = []

    for horizon in horizons:

        for direction in [
            "reversal",
            "momentum",
        ]:

            result, _ = (
                evaluate_candidate_period(
                    full_data=data,
                    horizon=horizon,
                    direction=direction,
                    volatility_window=(
                        volatility_window
                    ),
                    start_date=(
                        training_start
                    ),
                    end_date=(
                        training_end
                    ),
                )
            )

            records.append(
                {
                    "horizon":
                        horizon,

                    "direction":
                        direction,

                    **result,
                }
            )

    candidates = pd.DataFrame(
        records
    )

    valid = candidates.dropna(
        subset=[
            "ic_t_stat",
        ]
    )

    if valid.empty:
        raise RuntimeError(
            "No valid signal candidates "
            "were available in the "
            "training period."
        )

    # --------------------------------------------------
    # PRE-DEFINED SELECTION RULE
    #
    # Select the signal with the largest
    # training-period IC t-statistic.
    # --------------------------------------------------

    valid = (
        valid
        .sort_values(
            [
                "ic_t_stat",
                "mean_ic",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .reset_index(drop=True)
    )

    return valid


def run_walk_forward() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Run rolling four-year training /
    one-year testing validation.
    """

    config = load_config()

    data = (
        load_walk_forward_data()
    )

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

    training_years = (
        config["validation"][
            "training_years"
        ]
    )

    min_year = int(
        data["date"]
        .dt.year
        .min()
    )

    max_year = int(
        data["date"]
        .dt.year
        .max()
    )

    first_test_year = (
        min_year
        + training_years
    )

    fold_records = []

    out_of_sample_returns = []

    for test_year in range(
        first_test_year,
        max_year + 1,
    ):

        training_start = pd.Timestamp(
            year=(
                test_year
                - training_years
            ),
            month=1,
            day=1,
        )

        training_end = pd.Timestamp(
            year=(
                test_year - 1
            ),
            month=12,
            day=31,
        )

        test_start = pd.Timestamp(
            year=test_year,
            month=1,
            day=1,
        )

        test_end = pd.Timestamp(
            year=test_year,
            month=12,
            day=31,
        )

        # Do not request observations beyond
        # the end of our dataset.
        data_end = (
            data["date"].max()
        )

        test_end = min(
            test_end,
            data_end,
        )

        test_rows = data[
            (
                data["date"]
                >= test_start
            )
            &
            (
                data["date"]
                <= test_end
            )
        ]

        if test_rows.empty:
            continue

        print(
            f"\nWalk-forward fold: "
            f"{test_year}"
        )

        print(
            f"Training: "
            f"{training_start.date()} "
            f"to "
            f"{training_end.date()}"
        )

        print(
            f"Testing : "
            f"{test_start.date()} "
            f"to "
            f"{test_end.date()}"
        )

        candidates = (
            select_training_candidate(
                data=data,
                training_start=(
                    training_start
                ),
                training_end=(
                    training_end
                ),
                horizons=horizons,
                volatility_window=(
                    volatility_window
                ),
            )
        )

        selected = (
            candidates.iloc[0]
        )

        selected_horizon = int(
            selected[
                "horizon"
            ]
        )

        selected_direction = str(
            selected[
                "direction"
            ]
        )

        print(
            f"Selected: "
            f"{selected_horizon}-day "
            f"{selected_direction}"
        )

        print(
            f"Training IC t-stat: "
            f"{selected['ic_t_stat']:.3f}"
        )

        # --------------------------------------------------
        # TEST PERIOD
        # --------------------------------------------------

        test_result, test_spread = (
            evaluate_candidate_period(
                full_data=data,
                horizon=(
                    selected_horizon
                ),
                direction=(
                    selected_direction
                ),
                volatility_window=(
                    volatility_window
                ),
                start_date=(
                    test_start
                ),
                end_date=(
                    test_end
                ),
            )
        )

        fold_records.append(
            {
                "test_year":
                    test_year,

                "training_start":
                    training_start,

                "training_end":
                    training_end,

                "test_start":
                    test_start,

                "test_end":
                    test_end,

                "selected_horizon":
                    selected_horizon,

                "selected_direction":
                    selected_direction,

                "train_mean_ic":
                    selected[
                        "mean_ic"
                    ],

                "train_ic_t_stat":
                    selected[
                        "ic_t_stat"
                    ],

                "train_spread_sharpe":
                    selected[
                        "spread_sharpe"
                    ],

                "test_mean_ic":
                    test_result[
                        "mean_ic"
                    ],

                "test_ic_t_stat":
                    test_result[
                        "ic_t_stat"
                    ],

                "test_annualised_spread_return":
                    test_result[
                        "annualised_spread_return"
                    ],

                "test_spread_sharpe":
                    test_result[
                        "spread_sharpe"
                    ],

                "test_spread_t_stat":
                    test_result[
                        "spread_t_stat"
                    ],

                "test_positive_spread_day_pct":
                    test_result[
                        "positive_spread_day_pct"
                    ],

                "test_observations":
                    test_result[
                        "spread_observations"
                    ],
            }
        )

        spread_frame = (
            test_spread
            .rename(
                "spread_return"
            )
            .reset_index()
        )

        spread_frame.columns = [
            "date",
            "spread_return",
        ]

        spread_frame[
            "test_year"
        ] = test_year

        spread_frame[
            "selected_horizon"
        ] = selected_horizon

        spread_frame[
            "selected_direction"
        ] = selected_direction

        out_of_sample_returns.append(
            spread_frame
        )

    folds = pd.DataFrame(
        fold_records
    )

    if not out_of_sample_returns:
        raise RuntimeError(
            "Walk-forward validation "
            "produced no test returns."
        )

    oos_returns = pd.concat(
        out_of_sample_returns,
        ignore_index=True,
    )

    oos_returns = (
        oos_returns
        .sort_values("date")
        .reset_index(drop=True)
    )

    return (
        folds,
        oos_returns,
    )


def calculate_walk_forward_summary(
    oos_returns: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate aggregate walk-forward
    research-spread performance.
    """

    returns = (
        oos_returns[
            "spread_return"
        ]
        .dropna()
    )

    annualised_return = (
        returns.mean()
        * 252
    )

    annualised_volatility = (
        returns.std(
            ddof=1
        )
        * np.sqrt(252)
    )

    if annualised_volatility > 0:

        sharpe = (
            annualised_return
            /
            annualised_volatility
        )

    else:

        sharpe = np.nan

    t_stat = (
        calculate_t_stat(
            returns
        )
    )

    positive_day_pct = (
        (returns > 0)
        .mean()
    )

    cumulative_return = (
        (
            1.0
            + returns
        )
        .prod()
        - 1.0
    )

    summary = pd.DataFrame(
        {
            "metric": [
                "OOS observations",
                "Annualised spread return",
                "Annualised spread volatility",
                "OOS spread Sharpe",
                "OOS spread t-statistic",
                "Positive spread day percentage",
                "Cumulative spread return",
            ],
            "value": [
                len(returns),
                annualised_return,
                annualised_volatility,
                sharpe,
                t_stat,
                positive_day_pct,
                cumulative_return,
            ],
        }
    )

    return summary


def create_walk_forward_plot(
    oos_returns: pd.DataFrame,
) -> Path:
    """
    Plot cumulative walk-forward
    out-of-sample research spread.
    """

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = (
        oos_returns.copy()
    )

    data[
        "cumulative_return"
    ] = (
        (
            1.0
            + data[
                "spread_return"
            ]
        )
        .cumprod()
        - 1.0
    )

    plt.figure(
        figsize=(12, 6)
    )

    plt.plot(
        data["date"],
        data[
            "cumulative_return"
        ],
        linewidth=1.4,
    )

    plt.axhline(
        0,
        linewidth=1,
    )

    plt.title(
        "Walk-Forward Out-of-Sample "
        "Signal Spread"
    )

    plt.xlabel(
        "Date"
    )

    plt.ylabel(
        "Cumulative Gross Return"
    )

    plt.tight_layout()

    output_path = (
        FIGURE_DIR
        / "08_walk_forward_spread.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
    )

    plt.close()

    return output_path


def create_walk_forward_yearly_plot(
    folds: pd.DataFrame,
) -> Path:
    """
    Plot test-period spread Sharpe
    for each walk-forward year.
    """

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.figure(
        figsize=(11, 6)
    )

    plt.bar(
        folds[
            "test_year"
        ].astype(str),
        folds[
            "test_spread_sharpe"
        ],
    )

    plt.axhline(
        0,
        linewidth=1,
    )

    plt.title(
        "Walk-Forward Test Sharpe "
        "by Year"
    )

    plt.xlabel(
        "Test Year"
    )

    plt.ylabel(
        "Gross Research-Spread Sharpe"
    )

    plt.xticks(
        rotation=45
    )

    plt.tight_layout()

    output_path = (
        FIGURE_DIR
        / "09_walk_forward_yearly_sharpe.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
    )

    plt.close()

    return output_path


def save_walk_forward_outputs(
    folds: pd.DataFrame,
    oos_returns: pd.DataFrame,
    summary: pd.DataFrame,
):
    """
    Save walk-forward research outputs.
    """

    TABLE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    folds_path = (
        TABLE_DIR
        / "walk_forward_folds.csv"
    )

    returns_path = (
        TABLE_DIR
        / "walk_forward_oos_returns.csv"
    )

    summary_path = (
        TABLE_DIR
        / "walk_forward_summary.csv"
    )

    folds.to_csv(
        folds_path,
        index=False,
    )

    oos_returns.to_csv(
        returns_path,
        index=False,
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    return (
        folds_path,
        returns_path,
        summary_path,
    )