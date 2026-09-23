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


from src.walkforward import (
    calculate_walk_forward_summary,
    create_walk_forward_plot,
    create_walk_forward_yearly_plot,
    run_walk_forward,
    save_walk_forward_outputs,
)


def run() -> None:

    print("=" * 70)
    print(
        "QUANT SIGNAL RESEARCH "
        "- WALK-FORWARD VALIDATION"
    )
    print("=" * 70)

    print(
        "Training window: 4 years"
    )

    print(
        "Testing window: 1 year"
    )

    print(
        "Selection criterion: "
        "highest training IC t-statistic"
    )

    print(
        "\nEach test year is evaluated "
        "using parameters selected only "
        "from preceding years."
    )

    folds, oos_returns = (
        run_walk_forward()
    )

    summary = (
        calculate_walk_forward_summary(
            oos_returns
        )
    )

    (
        folds_path,
        returns_path,
        summary_path,
    ) = save_walk_forward_outputs(
        folds=folds,
        oos_returns=oos_returns,
        summary=summary,
    )

    cumulative_plot = (
        create_walk_forward_plot(
            oos_returns
        )
    )

    yearly_plot = (
        create_walk_forward_yearly_plot(
            folds
        )
    )

    display_columns = [
        "test_year",
        "selected_horizon",
        "selected_direction",
        "train_mean_ic",
        "train_ic_t_stat",
        "test_mean_ic",
        "test_ic_t_stat",
        "test_annualised_spread_return",
        "test_spread_sharpe",
    ]

    print("\n" + "=" * 70)
    print("WALK-FORWARD FOLDS")
    print("=" * 70)

    print(
        folds[
            display_columns
        ]
        .to_string(
            index=False
        )
    )

    print("\n" + "=" * 70)
    print("AGGREGATE WALK-FORWARD RESULTS")
    print("=" * 70)

    print(
        summary.to_string(
            index=False
        )
    )

    print("\n" + "=" * 70)
    print("OUTPUTS")
    print("=" * 70)

    print(folds_path)
    print(returns_path)
    print(summary_path)
    print(cumulative_plot)
    print(yearly_plot)

    print(
        "\nWalk-forward validation "
        "completed successfully."
    )


if __name__ == "__main__":
    run()