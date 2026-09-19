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


from src.research import (
    assign_signal_quintiles,
    calculate_daily_ic,
    calculate_quintile_returns,
    create_ic_plot,
    create_quintile_plot,
    create_spread_plot,
    load_signal_data,
    save_research_outputs,
    summarize_ic,
    summarize_quintile_returns,
)


def run() -> None:
    """
    Evaluate predictive quality of
    the residual reversal signal.
    """

    print("=" * 70)
    print(
        "QUANT SIGNAL RESEARCH "
        "- SIGNAL EVALUATION"
    )
    print("=" * 70)

    data = load_signal_data()

    print(
        f"Rows loaded: "
        f"{len(data):,}"
    )

    print(
        f"Stocks: "
        f"{data['ticker'].nunique()}"
    )

    # --------------------------------------------------
    # Information Coefficient
    # --------------------------------------------------

    daily_ic = calculate_daily_ic(
        data
    )

    ic_summary = summarize_ic(
        daily_ic
    )

    # --------------------------------------------------
    # Signal portfolios
    # --------------------------------------------------

    data = assign_signal_quintiles(
        data
    )

    quintile_returns = (
        calculate_quintile_returns(
            data
        )
    )

    quintile_summary = (
        summarize_quintile_returns(
            quintile_returns
        )
    )

    # --------------------------------------------------
    # Save tables
    # --------------------------------------------------

    save_research_outputs(
        daily_ic=daily_ic,
        ic_summary=ic_summary,
        quintile_returns=quintile_returns,
        quintile_summary=quintile_summary,
    )

    # --------------------------------------------------
    # Create figures
    # --------------------------------------------------

    ic_plot = create_ic_plot(
        daily_ic
    )

    quintile_plot = (
        create_quintile_plot(
            quintile_summary
        )
    )

    spread_plot = (
        create_spread_plot(
            quintile_returns
        )
    )

    # --------------------------------------------------
    # Console results
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("INFORMATION COEFFICIENT")
    print("=" * 70)

    print(
        ic_summary.to_string(
            index=False
        )
    )

    print("\n" + "=" * 70)
    print("SIGNAL QUINTILE PERFORMANCE")
    print("=" * 70)

    print(
        quintile_summary.to_string(
            index=False
        )
    )

    print("\n" + "=" * 70)
    print("OUTPUT FILES")
    print("=" * 70)

    print(
        f"IC plot:\n{ic_plot}"
    )

    print(
        f"\nQuintile plot:\n"
        f"{quintile_plot}"
    )

    print(
        f"\nSpread plot:\n"
        f"{spread_plot}"
    )

    print(
        "\nSignal evaluation "
        "completed successfully."
    )


if __name__ == "__main__":
    run()