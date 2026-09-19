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

from src.holdout import (
    build_frozen_signal,
    calculate_holdout_summary,
    calculate_yearly_results,
    restrict_to_holdout,
    save_holdout_results,
)


def run() -> None:
    """
    Run the untouched out-of-sample
    validation of the frozen signal.
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

    start_date = (
        config["research"][
            "validation_start_date"
        ]
    )

    print("=" * 70)
    print(
        "QUANT SIGNAL RESEARCH "
        "- FROZEN HOLDOUT TEST"
    )
    print("=" * 70)

    print(
        f"Frozen signal: "
        f"{horizon}-day {direction}"
    )

    print(
        f"Holdout begins: "
        f"{start_date}"
    )

    print(
        "\nNo parameters are selected "
        "using this holdout period."
    )

    signal_data = (
        build_frozen_signal()
    )

    holdout = restrict_to_holdout(
        signal_data
    )

    summary = (
        calculate_holdout_summary(
            holdout
        )
    )

    yearly = (
        calculate_yearly_results(
            holdout
        )
    )

    (
        summary_path,
        yearly_path,
    ) = save_holdout_results(
        summary=summary,
        yearly=yearly,
    )

    print("\n" + "=" * 70)
    print("HOLDOUT SUMMARY")
    print("=" * 70)

    print(
        summary.to_string(
            index=False
        )
    )

    print("\n" + "=" * 70)
    print("YEAR-BY-YEAR HOLDOUT RESULTS")
    print("=" * 70)

    print(
        yearly.to_string(
            index=False
        )
    )

    print("\nResults saved to:")

    print(summary_path)

    print(yearly_path)

    print(
        "\nFrozen holdout test "
        "completed successfully."
    )


if __name__ == "__main__":
    run()