from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.backtest import (
    run_backtest,
)

from src.data import (
    PROJECT_ROOT,
)

from src.metrics import (
    calculate_performance_metrics,
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


def calculate_cost_sensitivity(
    daily: pd.DataFrame,
    cost_levels_bps: list[float],
) -> pd.DataFrame:
    """
    Re-evaluate the exact same portfolio
    under different transaction-cost
    assumptions.

    No signal or portfolio parameter is
    changed here.
    """

    records = []

    for cost_bps in cost_levels_bps:

        cost_rate = (
            cost_bps
            / 10000.0
        )

        transaction_cost = (
            daily["turnover"]
            * cost_rate
        )

        net_return = (
            daily["gross_return"]
            - transaction_cost
        )

        metrics = (
            calculate_performance_metrics(
                net_return
            )
        )

        annualised_cost_drag = (
            transaction_cost.mean()
            * 252
        )

        records.append(
            {
                "cost_bps":
                    cost_bps,

                "annualised_cost_drag":
                    annualised_cost_drag,

                "annualised_return":
                    metrics[
                        "annualised_return"
                    ],

                "cagr":
                    metrics[
                        "cagr"
                    ],

                "annualised_volatility":
                    metrics[
                        "annualised_volatility"
                    ],

                "sharpe":
                    metrics[
                        "sharpe"
                    ],

                "maximum_drawdown":
                    metrics[
                        "maximum_drawdown"
                    ],

                "cumulative_return":
                    metrics[
                        "cumulative_return"
                    ],
            }
        )

    return pd.DataFrame(
        records
    )


def calculate_yearly_backtest(
    daily: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate gross and net performance
    independently for each calendar year.
    """

    data = daily.copy()

    data["year"] = (
        pd.to_datetime(
            data["date"]
        )
        .dt.year
    )

    records = []

    for year, group in data.groupby(
        "year"
    ):

        gross = (
            calculate_performance_metrics(
                group["gross_return"]
            )
        )

        net = (
            calculate_performance_metrics(
                group["net_return"]
            )
        )

        records.append(
            {
                "year":
                    year,

                "gross_return":
                    gross[
                        "annualised_return"
                    ],

                "gross_sharpe":
                    gross[
                        "sharpe"
                    ],

                "net_return":
                    net[
                        "annualised_return"
                    ],

                "net_sharpe":
                    net[
                        "sharpe"
                    ],

                "average_turnover":
                    group[
                        "turnover"
                    ].mean(),

                "annualised_turnover":
                    group[
                        "turnover"
                    ].mean()
                    * 252,

                "average_gross_exposure":
                    group[
                        "gross_exposure"
                    ].mean(),

                "average_beta_exposure":
                    group[
                        "beta_exposure"
                    ].mean(),
            }
        )

    return pd.DataFrame(
        records
    )


def calculate_rolling_metrics(
    daily: pd.DataFrame,
    window: int = 252,
) -> pd.DataFrame:
    """
    Calculate rolling one-year gross and
    net Sharpe ratios.
    """

    data = daily.copy()

    data["date"] = pd.to_datetime(
        data["date"]
    )

    gross_mean = (
        data["gross_return"]
        .rolling(
            window=window,
            min_periods=126,
        )
        .mean()
    )

    gross_std = (
        data["gross_return"]
        .rolling(
            window=window,
            min_periods=126,
        )
        .std()
    )

    net_mean = (
        data["net_return"]
        .rolling(
            window=window,
            min_periods=126,
        )
        .mean()
    )

    net_std = (
        data["net_return"]
        .rolling(
            window=window,
            min_periods=126,
        )
        .std()
    )

    data[
        "rolling_gross_sharpe"
    ] = (
        gross_mean
        / gross_std
        * np.sqrt(252)
    )

    data[
        "rolling_net_sharpe"
    ] = (
        net_mean
        / net_std
        * np.sqrt(252)
    )

    return data


def create_equity_curve_plot(
    daily: pd.DataFrame,
) -> Path:
    """
    Compare gross and net cumulative wealth.
    """

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    plot_data = daily.copy()

    plot_data["date"] = pd.to_datetime(
        plot_data["date"]
    )

    plot_data[
        "gross_wealth"
    ] = (
        1.0
        + plot_data[
            "gross_return"
        ]
    ).cumprod()

    plot_data[
        "net_wealth"
    ] = (
        1.0
        + plot_data[
            "net_return"
        ]
    ).cumprod()

    plt.figure(
        figsize=(12, 6)
    )

    plt.plot(
        plot_data["date"],
        plot_data[
            "gross_wealth"
        ],
        label="Gross",
        linewidth=1.5,
    )

    plt.plot(
        plot_data["date"],
        plot_data[
            "net_wealth"
        ],
        label="Net of costs",
        linewidth=1.5,
    )

    plt.axhline(
        1.0,
        linewidth=1,
    )

    plt.title(
        "Gross vs Net Portfolio Wealth"
    )

    plt.xlabel(
        "Date"
    )

    plt.ylabel(
        "Growth of $1"
    )

    plt.legend()

    plt.tight_layout()

    output_path = (
        FIGURE_DIR
        / "04_gross_vs_net_equity_curve.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
    )

    plt.close()

    return output_path


def create_turnover_plot(
    daily: pd.DataFrame,
) -> Path:
    """
    Plot daily and rolling average turnover.
    """

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    plot_data = daily.copy()

    plot_data["date"] = pd.to_datetime(
        plot_data["date"]
    )

    plot_data[
        "turnover_63d"
    ] = (
        plot_data["turnover"]
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
        plot_data["turnover"],
        alpha=0.25,
        linewidth=0.7,
        label="Daily turnover",
    )

    plt.plot(
        plot_data["date"],
        plot_data[
            "turnover_63d"
        ],
        linewidth=1.5,
        label="63-day average",
    )

    plt.title(
        "Portfolio Turnover"
    )

    plt.xlabel(
        "Date"
    )

    plt.ylabel(
        "Turnover"
    )

    plt.legend()

    plt.tight_layout()

    output_path = (
        FIGURE_DIR
        / "05_turnover.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
    )

    plt.close()

    return output_path


def create_cost_sensitivity_plot(
    cost_summary: pd.DataFrame,
) -> Path:
    """
    Plot Sharpe against transaction cost.
    """

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.figure(
        figsize=(9, 6)
    )

    plt.plot(
        cost_summary[
            "cost_bps"
        ],
        cost_summary[
            "sharpe"
        ],
        marker="o",
    )

    plt.axhline(
        0,
        linewidth=1,
    )

    plt.title(
        "Transaction-Cost Sensitivity"
    )

    plt.xlabel(
        "Transaction Cost (bps per dollar traded)"
    )

    plt.ylabel(
        "Net Sharpe Ratio"
    )

    plt.tight_layout()

    output_path = (
        FIGURE_DIR
        / "06_transaction_cost_sensitivity.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
    )

    plt.close()

    return output_path


def create_rolling_sharpe_plot(
    rolling: pd.DataFrame,
) -> Path:
    """
    Plot rolling gross and net Sharpe.
    """

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.figure(
        figsize=(12, 6)
    )

    plt.plot(
        rolling["date"],
        rolling[
            "rolling_gross_sharpe"
        ],
        label="Gross Sharpe",
        linewidth=1.3,
    )

    plt.plot(
        rolling["date"],
        rolling[
            "rolling_net_sharpe"
        ],
        label="Net Sharpe",
        linewidth=1.3,
    )

    plt.axhline(
        0,
        linewidth=1,
    )

    plt.title(
        "252-Day Rolling Sharpe Ratio"
    )

    plt.xlabel(
        "Date"
    )

    plt.ylabel(
        "Sharpe Ratio"
    )

    plt.legend()

    plt.tight_layout()

    output_path = (
        FIGURE_DIR
        / "07_rolling_sharpe.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
    )

    plt.close()

    return output_path


def save_robustness_tables(
    cost_summary: pd.DataFrame,
    yearly: pd.DataFrame,
    rolling: pd.DataFrame,
) -> None:
    """
    Save robustness tables.
    """

    TABLE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    cost_summary.to_csv(
        TABLE_DIR
        / "transaction_cost_sensitivity.csv",
        index=False,
    )

    yearly.to_csv(
        TABLE_DIR
        / "yearly_backtest_results.csv",
        index=False,
    )

    rolling[
        [
            "date",
            "rolling_gross_sharpe",
            "rolling_net_sharpe",
        ]
    ].to_csv(
        TABLE_DIR
        / "rolling_sharpe.csv",
        index=False,
    )