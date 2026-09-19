from pathlib import Path
import sys

import pandas as pd
import yfinance as yf


# ---------------------------------------------------------
# Allow this script to import modules from the project root
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------
# Import our own project functions
# ---------------------------------------------------------

from src.data import (
    RAW_DATA_DIR,
    ensure_data_directories,
    load_config,
    load_universe,
)


def download_market_data() -> pd.DataFrame:
    """
    Download daily OHLCV data for the equity universe
    plus the SPY market proxy.

    Returns
    -------
    pd.DataFrame
        Long-format market data with one row per
        date/ticker combination.
    """

    config = load_config()

    stock_tickers = load_universe()

    market_ticker = config["data"]["market_ticker"]

    # Add SPY to our 50-stock universe
    tickers = stock_tickers + [market_ticker]

    start_date = config["data"]["start_date"]
    end_date = config["data"]["end_date"]

    print("=" * 70)
    print("QUANT SIGNAL RESEARCH - MARKET DATA DOWNLOAD")
    print("=" * 70)

    print(f"Number of equities: {len(stock_tickers)}")
    print(f"Market proxy: {market_ticker}")
    print(f"Total tickers requested: {len(tickers)}")
    print(f"Start date: {start_date}")
    print(f"End date: {end_date}")

    print("\nDownloading historical market data...")
    print("Please wait. This may take a little while.\n")

    data = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        group_by="ticker",
        auto_adjust=True,
        threads=True,
        progress=True,
    )

    if data.empty:
        raise RuntimeError(
            "Yahoo Finance returned an empty dataset."
        )

    print("\nDownload completed.")

    # -----------------------------------------------------
    # Convert Yahoo's wide MultiIndex format
    # into a clean long-format table
    # -----------------------------------------------------

    if isinstance(data.columns, pd.MultiIndex):

        long_data = (
            data
            .stack(level=0, future_stack=True)
            .rename_axis(index=["date", "ticker"])
            .reset_index()
        )

    else:

        # This branch is mainly protection in case
        # only one ticker is ever downloaded.
        long_data = data.reset_index()

        long_data["ticker"] = tickers[0]

    # -----------------------------------------------------
    # Standardise column names
    # -----------------------------------------------------

    long_data.columns = [
        str(column)
        .strip()
        .lower()
        .replace(" ", "_")
        for column in long_data.columns
    ]

    # Make sure Date became date
    if "date" not in long_data.columns:
        raise ValueError(
            "Expected a 'date' column after processing."
        )

    # -----------------------------------------------------
    # Keep the columns we actually need
    # -----------------------------------------------------

    desired_columns = [
        "date",
        "ticker",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    available_columns = [
        column
        for column in desired_columns
        if column in long_data.columns
    ]

    long_data = long_data[available_columns]

    # -----------------------------------------------------
    # Clean and sort
    # -----------------------------------------------------

    long_data["date"] = pd.to_datetime(
        long_data["date"]
    )

    long_data["ticker"] = (
        long_data["ticker"]
        .astype(str)
        .str.upper()
    )

    long_data = (
        long_data
        .sort_values(["ticker", "date"])
        .drop_duplicates(
            subset=["date", "ticker"]
        )
        .reset_index(drop=True)
    )

    return long_data


def validate_market_data(
    data: pd.DataFrame,
    requested_tickers: list[str],
) -> None:
    """
    Run basic quality-control checks on downloaded data.
    """

    print("\n" + "=" * 70)
    print("DATA QUALITY CHECK")
    print("=" * 70)

    downloaded_tickers = sorted(
        data["ticker"].dropna().unique()
    )

    missing_tickers = sorted(
        set(requested_tickers)
        - set(downloaded_tickers)
    )

    duplicate_rows = data.duplicated(
        subset=["date", "ticker"]
    ).sum()

    missing_close = data["close"].isna().sum()

    print(
        f"Tickers requested: "
        f"{len(requested_tickers)}"
    )

    print(
        f"Tickers downloaded: "
        f"{len(downloaded_tickers)}"
    )

    print(
        f"Duplicate date/ticker rows: "
        f"{duplicate_rows}"
    )

    print(
        f"Missing close prices: "
        f"{missing_close}"
    )

    if missing_tickers:
        print("\nWARNING - Missing tickers:")

        for ticker in missing_tickers:
            print(f"  - {ticker}")

    else:
        print("\nAll requested tickers were downloaded.")


def save_market_data(
    data: pd.DataFrame,
) -> Path:
    """
    Save market data locally in Parquet format.
    """

    ensure_data_directories()

    output_path = (
        RAW_DATA_DIR / "market_data.parquet"
    )

    data.to_parquet(
        output_path,
        index=False,
    )

    return output_path


def run() -> None:
    """
    Run the complete market-data pipeline.
    """

    config = load_config()

    stock_tickers = load_universe()

    market_ticker = config["data"]["market_ticker"]

    requested_tickers = (
        stock_tickers + [market_ticker]
    )

    market_data = download_market_data()

    validate_market_data(
        market_data,
        requested_tickers,
    )

    output_path = save_market_data(
        market_data
    )

    print("\n" + "=" * 70)
    print("MARKET DATA SUMMARY")
    print("=" * 70)

    print(
        f"Total rows: "
        f"{len(market_data):,}"
    )

    print(
        f"Unique tickers: "
        f"{market_data['ticker'].nunique()}"
    )

    print(
        f"First date: "
        f"{market_data['date'].min().date()}"
    )

    print(
        f"Last date: "
        f"{market_data['date'].max().date()}"
    )

    print("\nColumns:")
    print(
        market_data.columns.tolist()
    )

    print("\nFirst five rows:")
    print(
        market_data.head()
    )

    print("\nSaved locally to:")

    print(
        output_path
    )

    print(
        "\nMarket data pipeline "
        "completed successfully."
    )


if __name__ == "__main__":
    run()