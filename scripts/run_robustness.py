from pathlib import Path
import sys


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from src.backtest import (
    run_backtest,
)

from src.robustness import (
    calculate_cost_sensitivity,
    calculate_rolling_metrics,
    calculate_yearly_backtest,
    create_cost_sensitivity_plot,
    create_equity_curve_plot,
    create_rolling_sharpe_plot,
    create_turnover_plot,
    save_robustness_tables,
)


def run() -> None:

    print("=" * 70)
    print(
        "QUANT SIGNAL RESEARCH "
        "- ROBUSTNESS ANALYSIS"
    )
    print("=" * 70)

    (
        positions,
        daily,
        gross_metrics,
        net_metrics,
    ) = run_backtest()

    cost_levels = [
        0,
        2,
        5,
        10,
        20,
    ]

    cost_summary = (
        calculate_cost_sensitivity(
            daily=daily,
            cost_levels_bps=cost_levels,
        )
    )

    yearly = (
        calculate_yearly_backtest(
            daily
        )
    )

    rolling = (
        calculate_rolling_metrics(
            daily,
            window=252,
        )
    )

    save_robustness_tables(
        cost_summary=cost_summary,
        yearly=yearly,
        rolling=rolling,
    )

    equity_plot = (
        create_equity_curve_plot(
            daily
        )
    )

    turnover_plot = (
        create_turnover_plot(
            daily
        )
    )

    cost_plot = (
        create_cost_sensitivity_plot(
            cost_summary
        )
    )

    sharpe_plot = (
        create_rolling_sharpe_plot(
            rolling
        )
    )

    print("\n" + "=" * 70)
    print("TRANSACTION-COST SENSITIVITY")
    print("=" * 70)

    print(
        cost_summary.to_string(
            index=False
        )
    )

    print("\n" + "=" * 70)
    print("YEARLY PORTFOLIO PERFORMANCE")
    print("=" * 70)

    print(
        yearly.to_string(
            index=False
        )
    )

    print("\n" + "=" * 70)
    print("FIGURES CREATED")
    print("=" * 70)

    print(equity_plot)
    print(turnover_plot)
    print(cost_plot)
    print(sharpe_plot)

    print(
        "\nRobustness analysis "
        "completed successfully."
    )


if __name__ == "__main__":
    run()