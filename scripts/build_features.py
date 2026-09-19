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


from src.features import (
    build_features,
    save_features,
)


def run() -> None:

    features = build_features()

    output_path = save_features(
        features
    )

    print("\n" + "=" * 70)
    print("FEATURE PIPELINE COMPLETE")
    print("=" * 70)

    print(
        f"Rows created: "
        f"{len(features):,}"
    )

    print(
        f"Stocks: "
        f"{features['ticker'].nunique()}"
    )

    print(
        f"Residual return observations: "
        f"{features['residual_return'].notna().sum():,}"
    )

    print(
        f"\nSaved to:\n"
        f"{output_path}"
    )


if __name__ == "__main__":
    run()