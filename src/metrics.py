import numpy as np
import pandas as pd


TRADING_DAYS = 252


def calculate_drawdown(
    returns: pd.Series,
) -> pd.Series:
    """
    Calculate portfolio drawdown series.
    """

    wealth = (
        1.0 + returns
    ).cumprod()

    running_max = (
        wealth.cummax()
    )

    drawdown = (
        wealth
        / running_max
        - 1.0
    )

    return drawdown


def calculate_performance_metrics(
    returns: pd.Series,
) -> dict:
    """
    Calculate standard portfolio
    performance statistics.
    """

    returns = (
        returns
        .dropna()
    )

    if len(returns) == 0:

        raise ValueError(
            "No returns supplied."
        )

    mean_daily = (
        returns.mean()
    )

    daily_volatility = (
        returns.std(
            ddof=1
        )
    )

    annualised_return = (
        mean_daily
        * TRADING_DAYS
    )

    annualised_volatility = (
        daily_volatility
        * np.sqrt(
            TRADING_DAYS
        )
    )

    if annualised_volatility > 0:

        sharpe = (
            annualised_return
            / annualised_volatility
        )

    else:

        sharpe = np.nan

    downside_returns = (
        returns[
            returns < 0
        ]
    )

    downside_deviation = (
        np.sqrt(
            (
                downside_returns ** 2
            ).mean()
        )
        * np.sqrt(
            TRADING_DAYS
        )
    )

    if downside_deviation > 0:

        sortino = (
            annualised_return
            / downside_deviation
        )

    else:

        sortino = np.nan

    cumulative_growth = (
        (1.0 + returns)
        .prod()
    )

    years = (
        len(returns)
        / TRADING_DAYS
    )

    if (
        years > 0
        and cumulative_growth > 0
    ):

        cagr = (
            cumulative_growth
            ** (1.0 / years)
            - 1.0
        )

    else:

        cagr = np.nan

    drawdown = (
        calculate_drawdown(
            returns
        )
    )

    max_drawdown = (
        drawdown.min()
    )

    if (
        max_drawdown < 0
        and not pd.isna(cagr)
    ):

        calmar = (
            cagr
            / abs(
                max_drawdown
            )
        )

    else:

        calmar = np.nan

    hit_rate = (
        (returns > 0)
        .mean()
    )

    cumulative_return = (
        cumulative_growth
        - 1.0
    )

    return {
        "observations":
            len(returns),

        "annualised_return":
            annualised_return,

        "cagr":
            cagr,

        "annualised_volatility":
            annualised_volatility,

        "sharpe":
            sharpe,

        "sortino":
            sortino,

        "maximum_drawdown":
            max_drawdown,

        "calmar":
            calmar,

        "positive_day_pct":
            hit_rate,

        "cumulative_return":
            cumulative_return,
    }