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


from src.signals import (
    build_signals,
    save_signals,
)


def run() -> None:
    """
    Build and save the quantitative signal dataset.
    """

    signals = build_signals()

    output_path = save_signals(
        signals
    )

    usable_signals = (
        signals[
            "signal_zscore"
        ]
        .notna()
        .sum()
    )

    usable_forward_returns = (
        signals[
            "forward_return_1d"
        ]
        .notna()
        .sum()
    )

    print("\n" + "=" * 70)
    print("SIGNAL CONSTRUCTION SUMMARY")
    print("=" * 70)

    print(
        f"Rows: "
        f"{len(signals):,}"
    )

    print(
        f"Stocks: "
        f"{signals['ticker'].nunique()}"
    )

    print(
        f"Usable signals: "
        f"{usable_signals:,}"
    )

    print(
        f"Usable forward returns: "
        f"{usable_forward_returns:,}"
    )

    print("\nSample signals:")

    sample_columns = [
        "date",
        "ticker",
        "residual_move",
        "residual_volatility",
        "raw_signal",
        "signal_zscore",
        "signal_rank",
        "forward_return_1d",
    ]

    sample = (
        signals[
            sample_columns
        ]
        .dropna()
        .tail(15)
    )

    print(
        sample.to_string(
            index=False
        )
    )

    print(
        f"\nSaved to:\n"
        f"{output_path}"
    )

    print(
        "\nSignal pipeline "
        "completed successfully."
    )


if __name__ == "__main__":
    run()