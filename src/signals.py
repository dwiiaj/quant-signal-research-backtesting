import numpy as np
import pandas as pd

from src.data import (
    PROCESSED_DATA_DIR,
    load_config,
)

from src.features import (
    FEATURE_DATA_PATH,
)


SIGNAL_DATA_PATH = (
    PROCESSED_DATA_DIR / "signals.parquet"
)


def load_features() -> pd.DataFrame:
    """
    Load engineered stock features.
    """

    if not FEATURE_DATA_PATH.exists():
        raise FileNotFoundError(
            "features.parquet was not found. "
            "Run scripts/build_features.py first."
        )

    data = pd.read_parquet(
        FEATURE_DATA_PATH
    )

    data["date"] = pd.to_datetime(
        data["date"]
    )

    data = (
        data
        .sort_values(
            ["ticker", "date"]
        )
        .reset_index(drop=True)
    )

    return data


def calculate_reversal_signal(
    data: pd.DataFrame,
    signal_horizon: int,
    volatility_window: int,
) -> pd.DataFrame:
    """
    Construct a volatility-standardised
    residual reversal signal.

    Positive signal:
        unusually negative recent residual move
        -> potential long candidate

    Negative signal:
        unusually positive recent residual move
        -> potential short candidate
    """

    data = data.copy()

    # --------------------------------------------------
    # 1. Sum recent idiosyncratic returns
    # --------------------------------------------------

    data["residual_move"] = (
        data
        .groupby("ticker")[
            "residual_return"
        ]
        .transform(
            lambda x:
            x.rolling(
                window=signal_horizon,
                min_periods=signal_horizon,
            ).sum()
        )
    )

    # --------------------------------------------------
    # 2. Estimate residual-return volatility
    # --------------------------------------------------

    data["residual_volatility"] = (
        data
        .groupby("ticker")[
            "residual_return"
        ]
        .transform(
            lambda x:
            x.rolling(
                window=volatility_window,
                min_periods=volatility_window,
            ).std()
        )
    )

    # --------------------------------------------------
    # 3. Standardise recent residual move
    #
    # Negative sign creates reversal interpretation:
    #
    # previous loser  -> positive signal
    # previous winner -> negative signal
    # --------------------------------------------------

    data["raw_signal"] = (
        -data["residual_move"]
        /
        (
            data["residual_volatility"]
            * np.sqrt(signal_horizon)
        )
    )

    data["raw_signal"] = (
        data["raw_signal"]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
    )

    return data


def cross_sectional_standardisation(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Standardise the raw signal across stocks
    independently for each trading day.
    """

    data = data.copy()

    def z_score(
        series: pd.Series,
    ) -> pd.Series:

        standard_deviation = (
            series.std()
        )

        if (
            pd.isna(standard_deviation)
            or standard_deviation == 0
        ):
            return pd.Series(
                np.nan,
                index=series.index,
            )

        return (
            series - series.mean()
        ) / standard_deviation

    data["signal_zscore"] = (
        data
        .groupby("date")[
            "raw_signal"
        ]
        .transform(z_score)
    )

    data["signal_rank"] = (
        data
        .groupby("date")[
            "signal_zscore"
        ]
        .rank(
            pct=True,
            method="average",
        )
    )

    return data


def add_forward_returns(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add next-day stock return.

    This is the future outcome we will use
    to evaluate whether today's signal has
    predictive information.

    It is NOT used to construct the signal.
    """

    data = data.copy()

    data["forward_return_1d"] = (
        data
        .groupby("ticker")[
            "return_1d"
        ]
        .shift(-1)
    )

    return data


def build_signals() -> pd.DataFrame:
    """
    Run the full signal construction pipeline.
    """

    config = load_config()

    signal_horizon = (
        config["signal"][
            "signal_horizon"
        ]
    )

    volatility_window = (
        config["signal"][
            "residual_volatility_window"
        ]
    )

    print("=" * 70)
    print(
        "QUANT SIGNAL RESEARCH "
        "- SIGNAL CONSTRUCTION"
    )
    print("=" * 70)

    print(
        f"Residual reversal horizon: "
        f"{signal_horizon} days"
    )

    print(
        f"Residual volatility window: "
        f"{volatility_window} days"
    )

    data = load_features()

    data = calculate_reversal_signal(
        data,
        signal_horizon=signal_horizon,
        volatility_window=volatility_window,
    )

    data = cross_sectional_standardisation(
        data
    )

    data = add_forward_returns(
        data
    )

    return data


def save_signals(
    data: pd.DataFrame,
):
    """
    Save signal research dataset locally.
    """

    PROCESSED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    data.to_parquet(
        SIGNAL_DATA_PATH,
        index=False,
    )

    return SIGNAL_DATA_PATH