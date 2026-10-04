"""Accuracy measures.

The three measures the brief asks for travel together as one lightweight
:class:`MetricSuite` tuple, so a function that scores a model returns a single
value that still unpacks like ``mae, rmse, mape = suite``.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np
from numpy.typing import ArrayLike


def _aligned(actual: ArrayLike, predicted: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Coerce a pair of array-likes and insist they line up."""
    a = np.asarray(actual, dtype=float).ravel()
    p = np.asarray(predicted, dtype=float).ravel()
    if a.size == 0:
        raise ValueError("actual must not be empty")
    if a.shape != p.shape:
        raise ValueError(f"actual ({a.size}) and predicted ({p.size}) must match")
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(p))):
        raise ValueError("actual and predicted must be finite")
    return a, p


class MetricSuite(NamedTuple):
    """MAE, RMSE and MAPE for one model on one test window.

    Being a tuple, it unpacks and compares naturally; being named, a caller can
    still ask for ``suite.mape`` without remembering the order.
    """

    mae: float
    rmse: float
    mape: float

    @classmethod
    def compare(cls, actual: ArrayLike, predicted: ArrayLike) -> "MetricSuite":
        """Score a forecast against the truth.

        MAE is in the units of the series, RMSE punishes large misses harder,
        and MAPE is unit-free so districts of very different sizes are
        comparable. MAPE is undefined if any actual value is zero.
        """
        a, p = _aligned(actual, predicted)
        residual = a - p
        if np.any(a == 0):
            raise ValueError("MAPE is undefined when an actual value is zero")
        return cls(mae=float(np.mean(np.abs(residual))),
                   rmse=float(np.sqrt(np.mean(residual ** 2))),
                   mape=float(np.mean(np.abs(residual / a)) * 100))

    def as_row(self, model: str) -> dict[str, float | str]:
        """A dict ready to become one row of a comparison table."""
        return {"model": model, "MAE": self.mae, "RMSE": self.rmse, "MAPE_%": self.mape}

    def __str__(self) -> str:
        return f"MAE {self.mae:,.3f} | RMSE {self.rmse:,.3f} | MAPE {self.mape:,.3f}%"
