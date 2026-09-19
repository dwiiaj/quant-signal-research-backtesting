from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data import PROJECT_ROOT
from src.signals import SIGNAL_DATA_PATH


TABLE_DIR = (
    PROJECT_ROOT / "outputs" / "tables"
)

FIGURE_DIR = (
    PROJECT_ROOT / "outputs" / "figures"
)


def load_signal_data() -> pd.DataFrame:
    """
    Load the completed signal research dataset.
    """

    if not SIGNAL_DATA_PATH.exists():
        raise FileNotFoundError(
            "signals.parquet was not found. "
            "Run scripts/build_signals.py first."
        )

    data = pd.read_parquet(
        SIGNAL_DATA_PATH
    )

    data["date"] = pd.to_datetime(
        data["date"]
    )

    data = (
        data
        .sort_values(
            ["date", "ticker"]
        )
        .reset_index(drop=True)
    )

    return data


def calculate_daily_ic(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate daily Spearman Information
    Coefficient (IC).

    IC measures the cross-sectional relationship
    between today's signal ranking and tomorrow's
    stock returns.

    Positive IC means:
        higher signal
        -> higher subsequent return
    """

    valid = data.dropna(
        subset=[
            "signal_zscore",
            "forward_return_1d",
        ]
    )

    ic_records = []

    for date, group in valid.groupby(
        "date"
    ):

        if (
            group["signal_zscore"].nunique() < 2
            or
            group["forward_return_1d"].nunique() < 2
        ):
            continue

        ic = group[
            "signal_zscore"
        ].corr(
            group["forward_return_1d"],
            method="spearman",
        )

        ic_records.append(
            {
                "date": date,
                "ic": ic,
            }
        )

    daily_ic = pd.DataFrame(
        ic_records
    )

    return daily_ic


def summarize_ic(
    daily_ic: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarise Information Coefficient statistics.
    """

    ic = (
        daily_ic["ic"]
        .dropna()
    )

    observations = len(ic)

    mean_ic = ic.mean()

    ic_std = ic.std(
        ddof=1
    )

    if (
        observations > 1
        and ic_std > 0
    ):

        ic_t_stat = (
            mean_ic
            /
            (
                ic_std
                / np.sqrt(observations)
            )
        )

    else:
        ic_t_stat = np.nan

    positive_ic_pct = (
        (ic > 0).mean()
    )

    summary = pd.DataFrame(
        {
            "metric": [
                "IC observations",
                "Mean IC",
                "IC standard deviation",
                "IC t-statistic",
                "Positive IC percentage",
            ],
            "value": [
                observations,
                mean_ic,
                ic_std,
                ic_t_stat,
                positive_ic_pct,
            ],
        }
    )

    return summary


def assign_signal_quintiles(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Divide stocks into five signal portfolios
    every day.

    Q1 = weakest / most negative signal
    Q5 = strongest / most positive signal
    """

    data = data.copy()

    quintile = np.ceil(
        data["signal_rank"]
        * 5
    )

    quintile = quintile.clip(
        lower=1,
        upper=5,
    )

    data["signal_quintile"] = (
        quintile.astype("Int64")
    )

    return data


def calculate_quintile_returns(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate equal-weight next-day returns
    for each signal quintile.
    """

    valid = data.dropna(
        subset=[
            "signal_quintile",
            "forward_return_1d",
        ]
    )

    quintile_returns = (
        valid
        .groupby(
            [
                "date",
                "signal_quintile",
            ]
        )[
            "forward_return_1d"
        ]
        .mean()
        .unstack(
            "signal_quintile"
        )
    )

    quintile_returns = (
        quintile_returns
        .reindex(
            columns=[
                1,
                2,
                3,
                4,
                5,
            ]
        )
    )

    quintile_returns.columns = [
        "Q1",
        "Q2",
        "Q3",
        "Q4",
        "Q5",
    ]

    # Reversal strategy hypothesis:
    #
    # Long strongest positive signals
    # Short strongest negative signals
    #
    # Therefore spread = Q5 - Q1

    quintile_returns[
        "Q5_minus_Q1"
    ] = (
        quintile_returns["Q5"]
        - quintile_returns["Q1"]
    )

    return (
        quintile_returns
        .reset_index()
    )


def summarize_quintile_returns(
    quintile_returns: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate annualised statistics for
    the five signal portfolios and
    Q5-minus-Q1 research spread.
    """

    portfolios = [
        "Q1",
        "Q2",
        "Q3",
        "Q4",
        "Q5",
        "Q5_minus_Q1",
    ]

    records = []

    for portfolio in portfolios:

        returns = (
            quintile_returns[
                portfolio
            ]
            .dropna()
        )

        mean_daily_return = (
            returns.mean()
        )

        annualised_return = (
            mean_daily_return
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
                / annualised_volatility
            )

        else:
            sharpe = np.nan

        positive_day_pct = (
            (returns > 0).mean()
        )

        records.append(
            {
                "portfolio": portfolio,
                "mean_daily_return":
                    mean_daily_return,
                "annualised_return":
                    annualised_return,
                "annualised_volatility":
                    annualised_volatility,
                "sharpe":
                    sharpe,
                "positive_day_pct":
                    positive_day_pct,
                "observations":
                    len(returns),
            }
        )

    return pd.DataFrame(
        records
    )


def create_ic_plot(
    daily_ic: pd.DataFrame,
) -> Path:
    """
    Plot daily IC and 63-day rolling IC.
    """

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    plot_data = daily_ic.copy()

    plot_data[
        "rolling_ic_63d"
    ] = (
        plot_data["ic"]
        .rolling(
            window=63,
            min_periods=20,
        )
        .mean()
    )

    plt.figure(
        figsize=(12, 6)
    )

    plt.plot(
        plot_data["date"],
        plot_data["ic"],
        alpha=0.30,
        linewidth=0.7,
        label="Daily IC",
    )

    plt.plot(
        plot_data["date"],
        plot_data[
            "rolling_ic_63d"
        ],
        linewidth=1.5,
        label="63-Day Rolling Mean IC",
    )

    plt.axhline(
        0,
        linewidth=1,
    )

    plt.title(
        "Daily Cross-Sectional "
        "Information Coefficient"
    )

    plt.xlabel(
        "Date"
    )

    plt.ylabel(
        "Spearman IC"
    )

    plt.legend()

    plt.tight_layout()

    output_path = (
        FIGURE_DIR
        / "01_information_coefficient.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
    )

    plt.close()

    return output_path


def create_quintile_plot(
    quintile_summary: pd.DataFrame,
) -> Path:
    """
    Plot annualised average returns
    from Q1 to Q5.
    """

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    plot_data = (
        quintile_summary[
            quintile_summary[
                "portfolio"
            ].isin(
                [
                    "Q1",
                    "Q2",
                    "Q3",
                    "Q4",
                    "Q5",
                ]
            )
        ]
    )

    plt.figure(
        figsize=(9, 6)
    )

    plt.bar(
        plot_data["portfolio"],
        plot_data[
            "annualised_return"
        ],
    )

    plt.axhline(
        0,
        linewidth=1,
    )

    plt.title(
        "Signal Quintile "
        "Annualised Average Returns"
    )

    plt.xlabel(
        "Signal Quintile"
    )

    plt.ylabel(
        "Annualised Average Return"
    )

    plt.tight_layout()

    output_path = (
        FIGURE_DIR
        / "02_signal_quintile_returns.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
    )

    plt.close()

    return output_path


def create_spread_plot(
    quintile_returns: pd.DataFrame,
) -> Path:
    """
    Plot cumulative gross Q5-minus-Q1
    signal spread.

    This is a research spread rather than
    our final portfolio backtest.
    """

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    plot_data = (
        quintile_returns[
            [
                "date",
                "Q5_minus_Q1",
            ]
        ]
        .dropna()
        .copy()
    )

    plot_data[
        "cumulative_spread"
    ] = (
        (
            1
            + plot_data[
                "Q5_minus_Q1"
            ]
        )
        .cumprod()
        - 1
    )

    plt.figure(
        figsize=(12, 6)
    )

    plt.plot(
        plot_data["date"],
        plot_data[
            "cumulative_spread"
        ],
        linewidth=1.3,
    )

    plt.axhline(
        0,
        linewidth=1,
    )

    plt.title(
        "Cumulative Gross "
        "Q5 Minus Q1 Signal Spread"
    )

    plt.xlabel(
        "Date"
    )

    plt.ylabel(
        "Cumulative Return"
    )

    plt.tight_layout()

    output_path = (
        FIGURE_DIR
        / "03_signal_spread_cumulative.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
    )

    plt.close()

    return output_path


def save_research_outputs(
    daily_ic: pd.DataFrame,
    ic_summary: pd.DataFrame,
    quintile_returns: pd.DataFrame,
    quintile_summary: pd.DataFrame,
) -> None:
    """
    Save research tables to CSV.
    """

    TABLE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    daily_ic.to_csv(
        TABLE_DIR
        / "daily_information_coefficient.csv",
        index=False,
    )

    ic_summary.to_csv(
        TABLE_DIR
        / "information_coefficient_summary.csv",
        index=False,
    )

    quintile_returns.to_csv(
        TABLE_DIR
        / "daily_quintile_returns.csv",
        index=False,
    )

    quintile_summary.to_csv(
        TABLE_DIR
        / "quintile_return_summary.csv",
        index=False,
    )