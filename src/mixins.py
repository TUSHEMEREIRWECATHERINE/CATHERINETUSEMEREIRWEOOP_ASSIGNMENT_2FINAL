"""Behaviour that several unrelated classes need, supplied as mixins.

Each mixin does one job and knows nothing about the others. A forecasting
model inherits all three; a container that only needs input checking inherits
just :class:`SeriesValidationMixin`. That keeps the capability and the class
hierarchy separate, so a new container does not have to become a model to
re-use the validation.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import ArrayLike

from src.metrics import MetricSuite


class SeriesValidationMixin:
    """Coercion and rejection rules for one-dimensional numeric series.

    Attributes
    ----------
    min_obs : int
        Fewest observations the inheriting class can work with.
    """

    min_obs: int = 2

    @staticmethod
    def clean(values: ArrayLike, label: str = "values", minimum: int = 1) -> np.ndarray:
        """Return ``values`` as a 1-D float array, or raise ``ValueError``."""
        arr = np.asarray(values, dtype=float).ravel()
        if arr.size < minimum:
            raise ValueError(f"{label} needs at least {minimum} value(s), got {arr.size}")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{label} must be finite (no NaN or inf)")
        return arr

    @staticmethod
    def whole_number(value: object, label: str) -> int:
        """Return ``value`` as a positive int, or raise ``ValueError``."""
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{label} must be an integer, got {type(value).__name__}")
        if value < 1:
            raise ValueError(f"{label} must be >= 1, got {value}")
        return int(value)

    def accept(self, y: ArrayLike,
               t: ArrayLike | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Validate a series and its time index together."""
        y_arr = self.clean(y, "y", self.min_obs)
        t_arr = (np.arange(y_arr.size, dtype=float) if t is None
                 else self.clean(t, "t", self.min_obs))
        if t_arr.size != y_arr.size:
            raise ValueError(f"t ({t_arr.size}) and y ({y_arr.size}) must match")
        return y_arr, t_arr


class ScoringMixin:
    """Gives anything with a ``predict`` method the ability to score itself."""

    def score(self, actual: ArrayLike) -> MetricSuite:
        """Forecast ``len(actual)`` steps ahead and measure the error."""
        truth = np.asarray(actual, dtype=float).ravel()
        if truth.size == 0:
            raise ValueError("actual must not be empty")
        return MetricSuite.compare(truth, self.predict(truth.size))


class WalkForwardMixin:
    """Rolling-origin backtesting, shared by mini-projects 1 and 5.

    At each origin the model is refitted on everything observed so far and
    asked for one step, which is how the forecast would actually have been
    produced at the time. Mini-project 5 reuses this unchanged.
    """

    def walk_forward(self, y: ArrayLike, first_origin: int | None = None) -> np.ndarray:
        """One-step-ahead forecasts from ``first_origin`` to the end of ``y``.

        Parameters
        ----------
        y : array-like
            The full series.
        first_origin : int, optional
            Index of the first point to forecast. Defaults to ``min_obs``, the
            earliest point at which the model can be fitted at all.
        """
        series = np.asarray(y, dtype=float).ravel()
        start = self.min_obs if first_origin is None else int(first_origin)
        if not 0 < start < series.size:
            raise ValueError(
                f"first_origin must be in (0, {series.size}), got {start}")
        out = np.empty(series.size - start, dtype=float)
        for i, origin in enumerate(range(start, series.size)):
            self.fit(series[:origin])
            out[i] = self.predict(1)[0]
        return out

    def walk_forward_score(self, y: ArrayLike,
                           first_origin: int | None = None) -> MetricSuite:
        """Accuracy of the rolling-origin forecasts against what happened."""
        series = np.asarray(y, dtype=float).ravel()
        predicted = self.walk_forward(series, first_origin)
        return MetricSuite.compare(series[series.size - predicted.size:], predicted)


def grid_search(build: Callable[[float], "WalkForwardMixin"], values: ArrayLike,
                y: ArrayLike, first_origin: int | None = None) -> tuple[float, dict]:
    """Pick the parameter value with the lowest walk-forward MAE.

    Written here rather than in a project module because mini-project 5 tunes
    its smoothing constant the same way.
    """
    candidates = np.asarray(values, dtype=float).ravel()
    if candidates.size == 0:
        raise ValueError("values must not be empty")
    scores = {float(v): build(float(v)).walk_forward_score(y, first_origin).mae
              for v in candidates}
    best = min(scores, key=scores.__getitem__)
    return best, scores
