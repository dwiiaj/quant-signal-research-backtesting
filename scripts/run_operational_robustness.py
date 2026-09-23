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


from src.scenarios import (
    run_operational_scenarios,
    save_operational_scenarios,
)


def run() -> None:

    print("=" * 70)
    print(
        "QUANT SIGNAL RESEARCH "
        "- OPERATIONAL ROBUSTNESS"
    )
    print("=" * 70)

    print(
        "Frozen signal parameters "
        "remain unchanged."
    )

    print(
        "These scenarios test implementation "
        "choices only.\n"
    )

    results = (
        run_operational_scenarios()
    )

    output_path = (
        save_operational_scenarios(
            results
        )
    )

    display_columns = [
        "rebalance_days",
        "volatility_targeting",
        "annualised_return",
        "annualised_volatility",
        "sharpe",
        "maximum_drawdown",
        "average_daily_turnover",
        "annualised_turnover",
        "average_gross_exposure",
        "average_beta_exposure",
    ]

    print("\n" + "=" * 70)
    print("OPERATIONAL ROBUSTNESS RESULTS")
    print("=" * 70)

    print(
        results[
            display_columns
        ]
        .to_string(
            index=False
        )
    )

    print(
        f"\nResults saved to:\n"
        f"{output_path}"
    )

    print(
        "\nOperational robustness "
        "completed successfully."
    )


if __name__ == "__main__":
    run()