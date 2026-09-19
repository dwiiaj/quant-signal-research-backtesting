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


from src.data import load_config

from src.signal_screen import (
    run_signal_screen,
    save_signal_screen,
)


def run() -> None:
    """
    Run the development-period
    signal hypothesis screen.
    """

    config = load_config()

    development_end = (
        config["research"][
            "development_end_date"
        ]
    )

    print("=" * 70)
    print(
        "QUANT SIGNAL RESEARCH "
        "- HYPOTHESIS SCREEN"
    )
    print("=" * 70)

    print(
        f"Development sample ends: "
        f"{development_end}"
    )

    print(
        "\nTesting residual reversal "
        "versus residual momentum.\n"
    )

    summary = (
        run_signal_screen()
    )

    output_path = (
        save_signal_screen(
            summary
        )
    )

    display_columns = [
        "horizon_days",
        "direction",
        "mean_ic",
        "ic_t_stat",
        "annualised_spread_return",
        "spread_sharpe",
        "spread_t_stat",
        "positive_spread_day_pct",
    ]

    print("\n" + "=" * 70)
    print("DEVELOPMENT SIGNAL SCREEN")
    print("=" * 70)

    print(
        summary[
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
        "\nHypothesis screen "
        "completed successfully."
    )


if __name__ == "__main__":
    run()