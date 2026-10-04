"""The abstract forecasting model, assembled from mixins.

:class:`Forecaster` inherits validation, scoring and backtesting rather than
implementing them, and adds only the fit/predict lifecycle itself. Concrete
models fill in two hooks, ``_estimate`` and ``_extrapolate``; everything
public is handled here.

Models also add together. ``trend + growth`` returns an
:class:`EnsembleForecaster` that fits both and averages their forecasts, which
is a genuinely useful operation when two models disagree in opposite
directions and neither is clearly right.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import ArrayLike

from src.mixins import ScoringMixin, SeriesValidationMixin, WalkForwardMixin


class Forecaster(SeriesValidationMixin, ScoringMixin, WalkForwardMixin, ABC):
    """Base class for every univariate forecasting model in this repository.

    The method resolution order is
    ``Forecaster -> SeriesValidationMixin -> ScoringMixin -> WalkForwardMixin``,
    so ``self.clean``, ``self.score`` and ``self.walk_forward`` are all
    available to a subclass that implements nothing but its own mathematics.
    """

    min_obs: int = 2
    label: str = "forecaster"

    def __init__(self) -> None:
        self.y_: np.ndarray | None = None
        self.t_: np.ndarray | None = None

    # ----- lifecycle ---------------------------------------------------
    def fit(self, y: ArrayLike, t: ArrayLike | None = None) -> "Forecaster":
        """Validate the series, store it and estimate the parameters."""
        self.y_, self.t_ = self.accept(y, t)
        self._estimate()
        return self

    def predict(self, horizon: int) -> np.ndarray:
        """Forecast the next ``horizon`` values beyond the training data."""
        if self.y_ is None:
            raise RuntimeError(f"call fit() before predict() on {type(self).__name__}")
        steps = self.whole_number(horizon, "horizon")
        return np.asarray(self._extrapolate(steps), dtype=float)

    def fitted(self) -> np.ndarray:
        """In-sample fitted values; NaN where a model does not define them."""
        if self.y_ is None:
            raise RuntimeError(f"call fit() before fitted() on {type(self).__name__}")
        return np.full_like(self.y_, np.nan, dtype=float)

    def residuals(self) -> np.ndarray:
        """Training residuals (actual minus fitted)."""
        return self.y_ - self.fitted()

    # ----- hooks for subclasses ----------------------------------------
    @abstractmethod
    def _estimate(self) -> None:
        """Estimate parameters from ``self.y_`` and ``self.t_``."""

    @abstractmethod
    def _extrapolate(self, horizon: int) -> np.ndarray:
        """Return ``horizon`` forecasts; the horizon is already validated."""

    # ----- dunder methods ---------------------------------------------
    def __add__(self, other: "Forecaster") -> "EnsembleForecaster":
        """``a + b`` builds an equally-weighted ensemble of the two models."""
        if not isinstance(other, Forecaster):
            return NotImplemented
        left = list(self) if isinstance(self, EnsembleForecaster) else [self]
        right = list(other) if isinstance(other, EnsembleForecaster) else [other]
        return EnsembleForecaster(left + right)

    def __repr__(self) -> str:
        state = "fitted" if self.y_ is not None else "unfitted"
        return f"{type(self).__name__}({state}, n={len(self)})"

    def __len__(self) -> int:
        """Observations the model was trained on; 0 before ``fit``."""
        return 0 if self.y_ is None else int(self.y_.size)


class EnsembleForecaster(Forecaster):
    """Averages the forecasts of several models.

    Built by adding models together rather than constructed directly::

        combined = PolyfitTrend() + CompoundGrowth()

    Parameters
    ----------
    members : list of Forecaster
        The models to combine; at least one.
    """

    def __init__(self, members: list[Forecaster]) -> None:
        super().__init__()
        if not members:
            raise ValueError("an ensemble needs at least one member")
        if not all(isinstance(m, Forecaster) for m in members):
            raise ValueError("every member must be a Forecaster")
        self.members = list(members)
        self.min_obs = max(m.min_obs for m in self.members)
        self.label = " + ".join(m.label for m in self.members)

    def _estimate(self) -> None:
        for member in self.members:
            member.fit(self.y_, self.t_)

    def _extrapolate(self, horizon: int) -> np.ndarray:
        return np.mean([m.predict(horizon) for m in self.members], axis=0)

    def fitted(self) -> np.ndarray:
        if self.y_ is None:
            raise RuntimeError("call fit() before fitted() on EnsembleForecaster")
        return np.mean([m.fitted() for m in self.members], axis=0)

    # An ensemble is first of all a collection of models, so its container
    # dunders describe its membership. Use ``n_observations`` for the training
    # length that ``len()`` reports on an ordinary model.
    def __iter__(self):
        return iter(self.members)

    def __getitem__(self, index: int) -> Forecaster:
        return self.members[index]

    def __len__(self) -> int:
        """How many models are being averaged."""
        return len(self.members)

    @property
    def n_observations(self) -> int:
        """Observations the ensemble was trained on; 0 before ``fit``."""
        return 0 if self.y_ is None else int(self.y_.size)

    def __repr__(self) -> str:
        return f"EnsembleForecaster({self.label!r}, n_members={len(self.members)})"
