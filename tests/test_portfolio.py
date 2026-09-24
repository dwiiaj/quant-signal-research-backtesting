import numpy as np
import pandas as pd

from src.portfolio import (
    neutralise_weights,
)


def test_neutralise_weights():
    """
    Neutralised weights should have
    approximately zero:

    1. dollar exposure
    2. market-beta exposure
    """

    selected = pd.DataFrame(
        {
            "ticker": [
                "A",
                "B",
                "C",
                "D",
            ],
            "beta": [
                0.8,
                1.0,
                1.2,
                1.4,
            ],
        }
    )

    raw_weights = np.array(
        [
            0.25,
            0.25,
            -0.25,
            -0.25,
        ]
    )

    weights = neutralise_weights(
        selected=selected,
        raw_weights=raw_weights,
    )

    dollar_exposure = (
        weights.sum()
    )

    beta_exposure = (
        weights
        @ selected[
            "beta"
        ].to_numpy()
    )

    assert np.isclose(
        dollar_exposure,
        0.0,
        atol=1e-10,
    )

    assert np.isclose(
        beta_exposure,
        0.0,
        atol=1e-10,
    )