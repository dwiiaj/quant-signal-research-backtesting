from pathlib import Path

import numpy as np
import pandas as pd

from src.data import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    load_config,
)


RAW_MARKET_DATA_PATH = (
    RAW_DATA_DIR / "market_data.parquet"
)

FEATURE_DATA_PATH = (
    PROCESSED_DATA_DIR / "features.parquet"
)


def load_market_data() -> pd.DataFrame:
    """
    Load raw downloaded market data.
    """

    if not RAW_MARKET_DATA_PATH.exists():
        raise FileNotFoundError(
            "Raw market data was not found. "
            "Run scripts/download_data.py first."
        )

    data = pd.read_parquet(
        RAW_MARKET_DATA_PATH
    )

    data["date"] = pd.to_datetime(
        data["date"]
    )

    data["ticker"] = (
        data["ticker"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    data = (
        data
        .sort_values(
            ["ticker", "date"]
        )
        .drop_duplicates(
            subset=["ticker", "date"]
        )
        .reset_index(drop=True)
    )

    return data


def calculate_returns(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate daily arithmetic returns
    separately for every ticker.

    Returns are calculated before removing
    missing prices so that a missing trading
    observation is not silently bridged.
    """

    data = data.copy()

    data["return_1d"] = (
        data
        .groupby("ticker")["close"]
        .pct_change(
            fill_method=None
        )
    )

    return data


def clean_market_data(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Remove observations with unusable prices
    after returns have been calculated.
    """

    data = data.copy()

    initial_rows = len(data)

    missing_close_before = (
        data["close"].isna().sum()
    )

    data = data.dropna(
        subset=["close"]
    )

    data = data[
        data["close"] > 0
    ]

    data = (
        data
        .sort_values(
            ["ticker", "date"]
        )
        .drop_duplicates(
            subset=["ticker", "date"]
        )
        .reset_index(drop=True)
    )

    removed_rows = (
        initial_rows - len(data)
    )

    print("=" * 70)
    print("MARKET DATA CLEANING")
    print("=" * 70)

    print(
        f"Initial rows: "
        f"{initial_rows:,}"
    )

    print(
        f"Missing close prices found: "
        f"{missing_close_before:,}"
    )

    print(
        f"Rows removed: "
        f"{removed_rows:,}"
    )

    print(
        f"Clean rows remaining: "
        f"{len(data):,}"
    )

    return data


def add_market_returns(
    data: pd.DataFrame,
    market_ticker: str,
) -> pd.DataFrame:
    """
    Add SPY market return to every stock row.
    """

    data = data.copy()

    market = (
        data.loc[
            data["ticker"] == market_ticker,
            ["date", "return_1d"]
        ]
        .rename(
            columns={
                "return_1d":
                "market_return"
            }
        )
    )

    data = data.merge(
        market,
        on="date",
        how="left",
        validate="many_to_one",
    )

    return data


def calculate_rolling_volatility(
    data: pd.DataFrame,
    window: int,
) -> pd.DataFrame:
    """
    Calculate trailing annualised stock volatility.

    Each estimate is shifted by one trading day
    within each ticker so today's volatility
    uses only previously available information.
    """

    data = data.copy()

    data["volatility_20d"] = (
        data
        .groupby("ticker")["return_1d"]
        .transform(
            lambda x:
            x.rolling(
                window=window,
                min_periods=window,
            )
            .std()
            .shift(1)
        )
        * np.sqrt(252)
    )

    return data


def calculate_rolling_beta(
    data: pd.DataFrame,
    market_ticker: str,
    window: int,
) -> pd.DataFrame:
    """
    Estimate rolling market beta using
    historical observations only.

    beta =
        Cov(stock return, market return)
        --------------------------------
             Var(market return)
    """

    stock_data = data[
        data["ticker"] != market_ticker
    ].copy()

    stock_data = stock_data.sort_values(
        ["ticker", "date"]
    )

    beta_frames = []

    for ticker, group in stock_data.groupby(
        "ticker",
        sort=False,
    ):

        group = group.copy()

        covariance = (
            group["return_1d"]
            .rolling(
                window=window,
                min_periods=window,
            )
            .cov(
                group["market_return"]
            )
        )

        market_variance = (
            group["market_return"]
            .rolling(
                window=window,
                min_periods=window,
            )
            .var()
        )

        group["beta"] = (
            covariance
            / market_variance
        ).shift(1)

        beta_frames.append(
            group
        )

    result = pd.concat(
        beta_frames,
        ignore_index=True,
    )

    result["beta"] = (
        result["beta"]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
    )

    return result


def calculate_residual_returns(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate idiosyncratic/residual return.

    residual return
        = stock return
        - beta * market return
    """

    data = data.copy()

    data["market_component"] = (
        data["beta"]
        * data["market_return"]
    )

    data["residual_return"] = (
        data["return_1d"]
        - data["market_component"]
    )

    return data


def build_features() -> pd.DataFrame:
    """
    Run the full feature-engineering pipeline.
    """

    config = load_config()

    market_ticker = (
        config["data"]["market_ticker"]
    )

    beta_window = (
        config["risk"]["beta_window"]
    )

    volatility_window = (
        config["risk"]["volatility_window"]
    )

    print("=" * 70)
    print(
        "QUANT SIGNAL RESEARCH "
        "- FEATURE ENGINEERING"
    )
    print("=" * 70)

    data = load_market_data()

    # Calculate returns before dropping
    # missing close observations.
    data = calculate_returns(
        data
    )

    data = clean_market_data(
        data
    )

    data = add_market_returns(
        data,
        market_ticker=market_ticker,
    )

    data = calculate_rolling_volatility(
        data,
        window=volatility_window,
    )

    data = calculate_rolling_beta(
        data,
        market_ticker=market_ticker,
        window=beta_window,
    )

    data = calculate_residual_returns(
        data
    )

    data = (
        data
        .sort_values(
            ["ticker", "date"]
        )
        .reset_index(drop=True)
    )

    return data


def save_features(
    data: pd.DataFrame,
) -> Path:
    """
    Save engineered features.
    """

    PROCESSED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    data.to_parquet(
        FEATURE_DATA_PATH,
        index=False,
    )

    return FEATURE_DATA_PATH


if __name__ == "__main__":

    features = build_features()

    output_path = save_features(
        features
    )

    print("\n" + "=" * 70)
    print("FEATURE ENGINEERING SUMMARY")
    print("=" * 70)

    print(
        f"Rows: "
        f"{len(features):,}"
    )

    print(
        f"Stocks: "
        f"{features['ticker'].nunique()}"
    )

    print(
        f"First date: "
        f"{features['date'].min().date()}"
    )

    print(
        f"Last date: "
        f"{features['date'].max().date()}"
    )

    print("\nFeature columns:")

    print(
        features.columns.tolist()
    )

    print(
        "\nNon-missing residual returns:"
    )

    print(
        features[
            "residual_return"
        ]
        .notna()
        .sum()
    )

    print("\nSample:")

    sample_columns = [
        "date",
        "ticker",
        "close",
        "return_1d",
        "market_return",
        "volatility_20d",
        "beta",
        "residual_return",
    ]

    print(
        features[
            sample_columns
        ]
        .dropna()
        .head(10)
        .to_string(
            index=False
        )
    )

    print(
        "\nSaved locally to:"
    )

    print(output_path)

    print(
        "\nFeature engineering "
        "completed successfully."
    )