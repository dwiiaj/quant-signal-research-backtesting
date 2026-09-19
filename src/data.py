from pathlib import Path

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

UNIVERSE_PATH = DATA_DIR / "universe.csv"


def load_config() -> dict:
    """
    Load project configuration from config/config.yaml.
    """
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    return config


def load_universe() -> list[str]:
    """
    Load the equity universe from data/universe.csv.
    """
    universe = pd.read_csv(UNIVERSE_PATH)

    if "ticker" not in universe.columns:
        raise ValueError(
            "universe.csv must contain a column named 'ticker'."
        )

    tickers = (
        universe["ticker"]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )

    return tickers


def ensure_data_directories() -> None:
    """
    Ensure that required project data directories exist.
    """
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    ensure_data_directories()

    config = load_config()
    tickers = load_universe()

    print("=" * 60)
    print("QUANT SIGNAL RESEARCH - DATA CONFIGURATION CHECK")
    print("=" * 60)

    print(f"Project: {config['project']['name']}")
    print(f"Start date: {config['data']['start_date']}")
    print(f"End date: {config['data']['end_date']}")
    print(f"Market proxy: {config['data']['market_ticker']}")
    print(f"Number of stocks: {len(tickers)}")

    print("\nFirst 10 tickers:")
    print(tickers[:10])

    print("\nData configuration loaded successfully.")