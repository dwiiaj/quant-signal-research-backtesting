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
    save_backtest,
)


def print_metrics(
    title: str,
    metrics: dict,
) -> None:
    """
    Pretty-print performance metrics.
    """

    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

    for key, value in metrics.items():

        if isinstance(
            value,
            float,
        ):

            print(
                f"{key:25s}: "
                f"{value: .6f}"
            )

        else:

            print(
                f"{key:25s}: "
                f"{value}"
            )


def run() -> None:

    (
        positions,
        daily,
        gross_metrics,
        net_metrics,
    ) = run_backtest()

    output_path = save_backtest(
        daily
    )

    print_metrics(
        "GROSS PERFORMANCE",
        gross_metrics,
    )

    print_metrics(
        "NET PERFORMANCE",
        net_metrics,
    )

    print("\n" + "=" * 70)
    print("TRADING AND RISK")
    print("=" * 70)

    print(
        f"Average daily turnover    : "
        f"{daily['turnover'].mean():.6f}"
    )

    print(
        f"Annualised turnover       : "
        f"{daily['turnover'].mean() * 252:.2f}x"
    )

    print(
        f"Average transaction cost  : "
        f"{daily['transaction_cost'].mean():.6f}"
    )

    print(
        f"Average gross exposure    : "
        f"{daily['gross_exposure'].mean():.6f}"
    )

    print(
        f"Average net exposure      : "
        f"{daily['net_exposure'].mean():.8f}"
    )

    print(
        f"Average beta exposure     : "
        f"{daily['beta_exposure'].mean():.8f}"
    )

    print(
        f"Maximum |beta exposure|   : "
        f"{daily['beta_exposure'].abs().max():.8f}"
    )

    print(
        f"\nDaily results saved to:\n"
        f"{output_path}"
    )

    print(
        "\nPortfolio backtest "
        "completed successfully."
    )


if __name__ == "__main__":
    run()