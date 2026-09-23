import pandas as pd

from src.backtest import (
    calculate_daily_portfolio_returns,
    prepare_backtest_data,
)

from src.costs import (
    add_weight_changes,
    calculate_daily_turnover,
    calculate_transaction_costs,
)

from src.data import (
    PROJECT_ROOT,
    load_config,
)

from src.metrics import (
    calculate_performance_metrics,
)

from src.portfolio import (
    construct_portfolio_weights,
)

from src.risk import (
    apply_rebalance_schedule,
    apply_scheduled_execution_lag,
    apply_volatility_scaling,
    calculate_volatility_scaler,
)


TABLE_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
)


def prepare_daily_targets() -> pd.DataFrame:
    """
    Construct the frozen signal portfolio
    target weights once.

    All robustness scenarios use the same
    frozen signal and portfolio logic.
    """

    data = prepare_backtest_data()

    data = (
        construct_portfolio_weights(
            data
        )
    )

    return data


def finish_scenario(
    positions: pd.DataFrame,
    cost_bps: float,
) -> tuple[
    pd.DataFrame,
    dict,
]:
    """
    Calculate turnover, costs and returns
    for one completed set of execution weights.
    """

    positions = (
        add_weight_changes(
            positions
        )
    )

    turnover = (
        calculate_daily_turnover(
            positions
        )
    )

    turnover = (
        calculate_transaction_costs(
            turnover,
            cost_bps=cost_bps,
        )
    )

    daily = (
        calculate_daily_portfolio_returns(
            positions
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

    daily["net_return"] = (
        daily["gross_return"]
        - daily[
            "transaction_cost"
        ]
    )

    metrics = (
        calculate_performance_metrics(
            daily["net_return"]
        )
    )

    return (
        daily,
        metrics,
    )


def run_rebalance_scenario(
    base_targets: pd.DataFrame,
    rebalance_every_n_days: int,
    cost_bps: float,
    volatility_targeting: bool = False,
) -> tuple[
    dict,
    pd.DataFrame,
]:
    """
    Run one implementation scenario.
    """

    config = load_config()

    positions = (
        apply_rebalance_schedule(
            data=base_targets,
            rebalance_every_n_days=(
                rebalance_every_n_days
            ),
        )
    )

    positions = (
        apply_scheduled_execution_lag(
            positions
        )
    )

    if volatility_targeting:

        # First obtain unscaled gross returns.
        preliminary_daily = (
            calculate_daily_portfolio_returns(
                positions
            )
        )

        scaler = (
            calculate_volatility_scaler(
                daily_returns=(
                    preliminary_daily
                ),
                target_volatility=(
                    config[
                        "portfolio"
                    ][
                        "target_volatility"
                    ]
                ),
                window=60,
                minimum_scale=0.25,
                maximum_scale=1.0,
            )
        )

        positions = (
            apply_volatility_scaling(
                positions=positions,
                scaler=scaler,
            )
        )

    daily, metrics = finish_scenario(
        positions=positions,
        cost_bps=cost_bps,
    )

    scenario = {
        "rebalance_days":
            rebalance_every_n_days,

        "volatility_targeting":
            volatility_targeting,

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

        "average_daily_turnover":
            daily[
                "turnover"
            ].mean(),

        "annualised_turnover":
            daily[
                "turnover"
            ].mean()
            * 252,

        "average_gross_exposure":
            daily[
                "gross_exposure"
            ].mean(),

        "average_net_exposure":
            daily[
                "net_exposure"
            ].mean(),

        "average_beta_exposure":
            daily[
                "beta_exposure"
            ].mean(),

        "max_abs_beta_exposure":
            daily[
                "beta_exposure"
            ].abs().max(),
    }

    return (
        scenario,
        daily,
    )


def run_operational_scenarios(
) -> pd.DataFrame:
    """
    Compare rebalancing frequencies with
    and without volatility risk control.
    """

    config = load_config()

    cost_bps = (
        config["costs"][
            "transaction_cost_bps"
        ]
    )

    base_targets = (
        prepare_daily_targets()
    )

    rebalance_frequencies = [
        1,
        2,
        5,
        10,
    ]

    records = []

    for rebalance_days in (
        rebalance_frequencies
    ):

        for vol_targeting in [
            False,
            True,
        ]:

            print(
                f"Running "
                f"{rebalance_days:2d}-day "
                f"rebalance | "
                f"vol targeting = "
                f"{vol_targeting}"
            )

            scenario, daily = (
                run_rebalance_scenario(
                    base_targets=(
                        base_targets
                    ),
                    rebalance_every_n_days=(
                        rebalance_days
                    ),
                    cost_bps=cost_bps,
                    volatility_targeting=(
                        vol_targeting
                    ),
                )
            )

            records.append(
                scenario
            )

    results = pd.DataFrame(
        records
    )

    return results


def save_operational_scenarios(
    results: pd.DataFrame,
):
    """
    Save implementation-robustness table.
    """

    TABLE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        TABLE_DIR
        / "operational_robustness.csv"
    )

    results.to_csv(
        output_path,
        index=False,
    )

    return output_path