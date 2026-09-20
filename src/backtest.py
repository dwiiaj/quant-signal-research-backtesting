from pathlib import Path

import pandas as pd

from src.costs import (
    add_weight_changes,
    calculate_daily_turnover,
    calculate_transaction_costs,
)

from src.data import (
    PROJECT_ROOT,
    load_config,
)

from src.holdout import (
    build_frozen_signal,
)

from src.metrics import (
    calculate_performance_metrics,
)

from src.portfolio import (
    apply_execution_lag,
    construct_portfolio_weights,
)


BACKTEST_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "daily_backtest.csv"
)


def prepare_backtest_data() -> pd.DataFrame:
    """
    Build the frozen signal and restrict
    the trading simulation to the
    holdout period.
    """

    config = load_config()

    validation_start = pd.Timestamp(
        config["research"][
            "validation_start_date"
        ]
    )

    data = build_frozen_signal()

    data = data[
        data["date"]
        >= validation_start
    ].copy()

    return data


def calculate_daily_portfolio_returns(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate gross long-short portfolio
    return for every day.
    """

    data = data.copy()

    data[
        "weighted_return"
    ] = (
        data[
            "execution_weight"
        ]
        * data[
            "forward_return_1d"
        ]
    )

    daily_returns = (
        data
        .groupby("date")
        .agg(
            gross_return=(
                "weighted_return",
                "sum",
            ),
            gross_exposure=(
                "execution_weight",
                lambda x:
                x.abs().sum(),
            ),
            net_exposure=(
                "execution_weight",
                "sum",
            ),
        )
        .reset_index()
    )

    beta_exposure = (
        data.assign(
            weighted_beta=(
                data[
                    "execution_weight"
                ]
                * data["beta"]
            )
        )
        .groupby("date")[
            "weighted_beta"
        ]
        .sum()
        .rename(
            "beta_exposure"
        )
        .reset_index()
    )

    daily_returns = (
        daily_returns
        .merge(
            beta_exposure,
            on="date",
            how="left",
        )
    )

    return daily_returns


def run_backtest():
    """
    Run complete transaction-cost-aware
    holdout backtest.
    """

    config = load_config()

    cost_bps = (
        config["costs"][
            "transaction_cost_bps"
        ]
    )

    print("=" * 70)
    print(
        "QUANT SIGNAL RESEARCH "
        "- PORTFOLIO BACKTEST"
    )
    print("=" * 70)

    print(
        f"Transaction cost: "
        f"{cost_bps:.1f} bps "
        f"per dollar traded"
    )

    print(
        "Execution assumption: "
        "one-day signal lag"
    )

    data = prepare_backtest_data()

    print(
        f"Backtest rows: "
        f"{len(data):,}"
    )

    print(
        f"Backtest stocks: "
        f"{data['ticker'].nunique()}"
    )

    # --------------------------------------------------
    # Portfolio construction
    # --------------------------------------------------

    data = (
        construct_portfolio_weights(
            data
        )
    )

    # --------------------------------------------------
    # Conservative execution lag
    # --------------------------------------------------

    data = apply_execution_lag(
        data
    )

    # --------------------------------------------------
    # Trading activity
    # --------------------------------------------------

    data = add_weight_changes(
        data
    )

    turnover = (
        calculate_daily_turnover(
            data
        )
    )

    turnover = (
        calculate_transaction_costs(
            turnover,
            cost_bps=cost_bps,
        )
    )

    # --------------------------------------------------
    # Portfolio returns
    # --------------------------------------------------

    daily = (
        calculate_daily_portfolio_returns(
            data
        )
    )

    daily = daily.merge(
        turnover,
        on="date",
        how="left",
    )

    daily[
        "transaction_cost"
    ] = (
        daily[
            "transaction_cost"
        ]
        .fillna(0.0)
    )

    daily[
        "net_return"
    ] = (
        daily["gross_return"]
        - daily[
            "transaction_cost"
        ]
    )

    # --------------------------------------------------
    # Performance statistics
    # --------------------------------------------------

    gross_metrics = (
        calculate_performance_metrics(
            daily["gross_return"]
        )
    )

    net_metrics = (
        calculate_performance_metrics(
            daily["net_return"]
        )
    )

    return (
        data,
        daily,
        gross_metrics,
        net_metrics,
    )


def save_backtest(
    daily: pd.DataFrame,
) -> Path:
    """
    Save daily portfolio backtest.
    """

    BACKTEST_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    daily.to_csv(
        BACKTEST_PATH,
        index=False,
    )

    return BACKTEST_PATH