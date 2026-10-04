"""Mini-Project 2: Solar Micro-Grid Dispatch Planner.

A rural health centre in Kasese meets two daily loads from solar panels (x)
and batteries (y)::

    3x + 2y = D1   (daytime load, kWh)
    4x +  y = D2   (critical-equipment load, kWh)

Classes
-------
:class:`MicroGrid`
    The coefficient matrix, its conditioning and the solvers.
:class:`HybridMicroGrid`
    Adds a diesel generator and a third load (the extension).
:class:`RepairStrategy`
    How an infeasible day is dealt with. Pluggable: the grid is handed a
    strategy rather than hard-coding one, so the choice can be compared
    rather than assumed.
:class:`DemandSource`
    The two input modes -- a seeded CSV, and interactive entry.
:class:`EnergyTariff`
    Costs a dispatch at UGX 150/kWh solar, UGX 450/kWh battery.

The strategy used by default is :class:`BoundedLeastSquares`, which re-solves
an infeasible day as a non-negative least-squares problem: instead of simply
discarding the negative component, it finds the closest feasible dispatch and
reports how far short it falls.
"""
from __future__ import annotations

import statistics
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Sequence

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy import linalg
from scipy.optimize import lsq_linear

from src.mixins import SeriesValidationMixin

__all__ = ["BoundedLeastSquares", "ClipToZero", "DemandSource", "DispatchResult",
           "EnergyTariff", "HybridMicroGrid", "MicroGrid", "RepairStrategy",
           "perturbation_study"]


class RepairStrategy(ABC):
    """What to do with a day whose exact solution is physically impossible.

    A negative allocation means the algebra wants a source to *absorb* energy
    to balance the equations, which no panel or generator can do. Every
    strategy must return a non-negative dispatch; they differ in how much
    demand they manage to serve while doing so.
    """

    name: str = "repair"

    @abstractmethod
    def apply(self, A: np.ndarray, demands: np.ndarray,
              raw: np.ndarray) -> np.ndarray:
        """Return a non-negative dispatch for every day in ``raw``."""

    def shortfall(self, A: np.ndarray, demands: np.ndarray,
                  repaired: np.ndarray) -> np.ndarray:
        """Per-day ``norm(A @ repaired - demand)``: the energy left unserved."""
        return np.linalg.norm(A @ repaired - demands, axis=0)

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class ClipToZero(RepairStrategy):
    """The blunt option: set any negative allocation to zero and move on.

    Cheap and obvious, but it makes no attempt to recover the demand it
    breaks, so it is the benchmark the better strategy has to beat.
    """

    name = "clip"

    def apply(self, A: np.ndarray, demands: np.ndarray,
              raw: np.ndarray) -> np.ndarray:
        return np.clip(raw, 0.0, None)


class BoundedLeastSquares(RepairStrategy):
    """Re-solve the day subject to ``x >= 0``, via ``scipy.optimize.lsq_linear``.

    Rather than discarding the impossible component, this asks for the
    *closest feasible dispatch*: the non-negative allocation that minimises
    ``norm(A x - d)``. It therefore serves as much of the demand as physics
    allows, and the residual it reports is a genuine lower bound on what
    cannot be delivered.
    """

    name = "bounded-lsq"

    def apply(self, A: np.ndarray, demands: np.ndarray,
              raw: np.ndarray) -> np.ndarray:
        repaired = np.array(raw, dtype=float, copy=True)
        infeasible = np.any(raw < -1e-9, axis=0)
        for day in np.flatnonzero(infeasible):
            solution = lsq_linear(A, demands[:, day], bounds=(0.0, np.inf))
            repaired[:, day] = solution.x
        return np.clip(repaired, 0.0, None)


@dataclass
class DispatchResult:
    """Everything that came out of dispatching a batch of days."""

    table: pd.DataFrame
    raw: np.ndarray
    infeasible: np.ndarray
    strategy: RepairStrategy
    sources: tuple[str, ...]

    def __len__(self) -> int:
        return len(self.table)

    def __repr__(self) -> str:
        return (f"DispatchResult(days={len(self)}, repaired={self.n_repaired}, "
                f"strategy={self.strategy.name}, "
                f"unserved={self.unserved_kwh:,.1f} kWh)")

    @property
    def n_repaired(self) -> int:
        return int(self.infeasible.sum())

    @property
    def unserved_kwh(self) -> float:
        return float(self.table["unserved_kwh"].sum())

    def repaired_rows(self) -> pd.DataFrame:
        return self.table.loc[self.infeasible]

    def allocation(self) -> np.ndarray:
        """The repaired dispatch as an ``(n_sources, n_days)`` array."""
        return self.table[list(self.sources)].to_numpy(dtype=float).T


class MicroGrid(SeriesValidationMixin):
    """Linear dispatch model ``A s = d``.

    Inherits only the validation mixin: a linear system needs the input rules
    but none of the forecasting machinery that mini-projects 1 and 5 use.

    Parameters
    ----------
    matrix : array-like, optional
        Square coefficient matrix; defaults to the brief's 2x2 system.
    sources : sequence of str
        Column names -- the energy sources.
    loads : sequence of str
        Row names -- the demand constraints.
    repair : RepairStrategy, optional
        How to handle infeasible days. Defaults to
        :class:`BoundedLeastSquares`.
    """

    MATRIX: list[list[float]] = [[3.0, 2.0], [4.0, 1.0]]
    SOURCE_NAMES: tuple[str, ...] = ("solar", "battery")
    LOAD_NAMES: tuple[str, ...] = ("D1", "D2")

    #: A condition number above this is treated as unusable.
    MAX_CONDITION = 1e8

    def __init__(self, matrix: ArrayLike | None = None,
                 sources: Sequence[str] | None = None,
                 loads: Sequence[str] | None = None,
                 repair: RepairStrategy | None = None) -> None:
        A = np.asarray(self.MATRIX if matrix is None else matrix, dtype=float)
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError(f"coefficient matrix must be square, got {A.shape}")
        if not np.all(np.isfinite(A)):
            raise ValueError("coefficient matrix must be finite")
        self.sources = tuple(self.SOURCE_NAMES if sources is None else sources)
        self.loads = tuple(self.LOAD_NAMES if loads is None else loads)
        if len(self.sources) != A.shape[1] or len(self.loads) != A.shape[0]:
            raise ValueError(
                f"need {A.shape[1]} source names and {A.shape[0]} load names")
        self.A = A
        self.repair = repair if repair is not None else BoundedLeastSquares()

    # ----- dunder methods ---------------------------------------------
    def __repr__(self) -> str:
        return (f"{type(self).__name__}(size={self.size}, det={self.det:.4g}, "
                f"cond={self.cond:.4g}, repair={self.repair.name})")

    def __len__(self) -> int:
        return self.size

    # ----- task 1 ---------------------------------------------------------
    @property
    def size(self) -> int:
        return int(self.A.shape[0])

    @property
    def det(self) -> float:
        """Non-zero means the loads are independent and the mix is unique."""
        return float(np.linalg.det(self.A))

    @property
    def cond(self) -> float:
        """Bounds how far a relative error in demand can grow in the answer."""
        return float(np.linalg.cond(self.A))

    @property
    def rank(self) -> int:
        return int(np.linalg.matrix_rank(self.A))

    @property
    def well_posed(self) -> bool:
        """A unique answer exists and can be trusted."""
        return (not np.isclose(self.det, 0.0)
                and np.isfinite(self.cond)
                and self.cond < self.MAX_CONDITION)

    def conditioning(self) -> pd.Series:
        """Every diagnostic in one place."""
        return pd.Series({"size": self.size, "determinant": self.det,
                          "condition_number": self.cond, "rank": self.rank,
                          "full_rank": self.rank == self.size,
                          "well_posed": self.well_posed})

    def _demands(self, demands: ArrayLike) -> np.ndarray:
        d = np.asarray(demands, dtype=float)
        if d.ndim == 1:
            d = d.reshape(self.size, 1)
        if d.ndim != 2 or d.shape[0] != self.size:
            raise ValueError(f"demands need {self.size} rows, got shape {d.shape}")
        if not np.all(np.isfinite(d)):
            raise ValueError("demands must be finite")
        if np.any(d < 0):
            raise ValueError("demands cannot be negative")
        return d

    def _guard(self) -> None:
        if not self.well_posed:
            raise ValueError(
                f"refusing to solve an ill-posed system "
                f"(det={self.det:.4g}, cond={self.cond:.4g})")

    # ----- tasks 2-3 --------------------------------------------------------
    def solve_day(self, *demands: float) -> np.ndarray:
        """Dispatch one day, one argument per load."""
        if len(demands) != self.size:
            raise ValueError(f"expected {self.size} demands, got {len(demands)}")
        return self.solve_batch(np.asarray(demands, dtype=float))[:, 0]

    def solve_batch(self, demands: ArrayLike) -> np.ndarray:
        """Every day in a single ``scipy.linalg.solve`` call."""
        self._guard()
        return linalg.solve(self.A, self._demands(demands))

    def solve_each(self, demands: ArrayLike) -> np.ndarray:
        """The same answer, one day per loop iteration, for the timing comparison."""
        self._guard()
        d = self._demands(demands)
        return np.column_stack([linalg.solve(self.A, d[:, i])
                                for i in range(d.shape[1])])

    # ----- task 4 ------------------------------------------------------------
    @staticmethod
    def infeasible_days(allocation: ArrayLike, tol: float = 1e-9) -> np.ndarray:
        """Mask of days asking some source to supply a negative amount."""
        return np.any(np.asarray(allocation, dtype=float) < -abs(tol), axis=0)

    def dispatch(self, demands: ArrayLike,
                 repair: RepairStrategy | None = None) -> DispatchResult:
        """Solve, repair whatever is impossible, and report the batch."""
        d = self._demands(demands)
        raw = self.solve_batch(d)
        flagged = self.infeasible_days(raw)
        strategy = repair if repair is not None else self.repair
        fixed = strategy.apply(self.A, d, raw)
        unserved = strategy.shortfall(self.A, d, fixed)

        table = pd.DataFrame(fixed.T, columns=list(self.sources))
        for row, load in enumerate(self.loads):
            table[load] = d[row]
        table["repaired"] = flagged
        table["unserved_kwh"] = unserved
        table.index.name = "day"
        return DispatchResult(table, raw, flagged, strategy, self.sources)

    def compare_repairs(self, demands: ArrayLike,
                        strategies: Sequence[RepairStrategy]) -> pd.DataFrame:
        """How much demand each repair strategy manages to serve."""
        if not strategies:
            raise ValueError("need at least one strategy to compare")
        rows = []
        for strategy in strategies:
            result = self.dispatch(demands, repair=strategy)
            rows.append({"strategy": strategy.name,
                         "days repaired": result.n_repaired,
                         "unserved kWh": result.unserved_kwh,
                         "worst day kWh": float(result.table["unserved_kwh"].max())})
        return pd.DataFrame(rows).set_index("strategy")

    # ----- task 5 --------------------------------------------------------------
    def volatility(self, allocation: ArrayLike) -> pd.DataFrame:
        """Per-source mean, variance, sd and CV, computed with ``statistics``.

        The CV is the column that answers the question. Variance is in
        kWh-squared and grows with the size of the source, so on its own it
        says more about how much a source supplies than about how erratic it is.
        """
        s = np.asarray(allocation, dtype=float)
        if s.ndim != 2 or s.shape[0] != self.size:
            raise ValueError(f"expected a ({self.size}, n_days) allocation")
        if s.shape[1] < 2:
            raise ValueError("need at least two days to measure volatility")
        rows = []
        for name, values in zip(self.sources, s):
            data = values.tolist()
            mean, sd = statistics.mean(data), statistics.stdev(data)
            rows.append({"source": name, "mean_kwh": mean,
                         "variance": statistics.variance(data), "stdev": sd,
                         "cv": sd / mean if mean else float("inf")})
        return pd.DataFrame(rows).set_index("source")


class HybridMicroGrid(MicroGrid):
    """Solar, battery and diesel, under a third load.

    The third constraint here is the **vaccine cold chain**, which runs around
    the clock and which the brief leaves open::

        3x + 2y      = D1   (daytime load, unchanged)
        4x +  y      = D2   (critical-equipment load, unchanged)
         x + 2y + 5z = D3   (cold chain)

    The coefficients say diesel is the efficient way to hold a fridge
    overnight, which is exactly why one gets installed.
    """

    MATRIX = [[3.0, 2.0, 0.0], [4.0, 1.0, 0.0], [1.0, 2.0, 5.0]]
    SOURCE_NAMES = ("solar", "battery", "diesel")
    LOAD_NAMES = ("D1", "D2", "D3")

    def __init__(self, matrix: ArrayLike | None = None,
                 sources: Sequence[str] | None = None,
                 loads: Sequence[str] | None = None,
                 repair: RepairStrategy | None = None) -> None:
        super().__init__(matrix, sources, loads, repair)
        if self.size != 3:
            raise ValueError("HybridMicroGrid describes a 3x3 system")

    @classmethod
    def linearly_dependent(cls) -> "HybridMicroGrid":
        """A broken variant whose third load is the sum of the first two.

        Used to show what happens when the extra equation carries no new
        information: the rank drops, the determinant vanishes and no unique
        dispatch exists.
        """
        A = np.array(cls.MATRIX, dtype=float)
        A[2] = A[0] + A[1]
        return cls(A)


@dataclass(frozen=True)
class EnergyTariff:
    """UGX per kWh by source. Frozen, so a costing cannot drift mid-analysis."""

    solar: float = 150.0
    battery: float = 450.0
    diesel: float = 900.0

    def __post_init__(self) -> None:
        if min(self.solar, self.battery, self.diesel) < 0:
            raise ValueError("tariff rates must be non-negative")

    def rates(self) -> dict[str, float]:
        return {"solar": self.solar, "battery": self.battery, "diesel": self.diesel}

    def per_day(self, dispatch: pd.DataFrame) -> pd.Series:
        """Cost of each day of a dispatch table, in UGX."""
        known = [name for name in self.rates() if name in dispatch.columns]
        if not known:
            raise ValueError(
                f"none of {sorted(self.rates())} appear in {list(dispatch.columns)}")
        rates = self.rates()
        total = sum(dispatch[name] * rates[name] for name in known)
        return pd.Series(total, index=dispatch.index, name="cost_ugx")

    def over_period(self, dispatch: pd.DataFrame) -> float:
        return float(self.per_day(dispatch).sum())

    def run_rate(self, dispatch: pd.DataFrame, days: int = 30) -> float:
        """Mean daily cost projected over a ``days``-day month."""
        if days < 1:
            raise ValueError("days must be at least 1")
        return float(self.per_day(dispatch).mean()) * int(days)

    def split(self, dispatch: pd.DataFrame) -> pd.DataFrame:
        """Share of energy and of cost taken by each source."""
        rates = self.rates()
        known = [n for n in rates if n in dispatch.columns]
        energy = dispatch[known].sum()
        cost = energy * pd.Series({n: rates[n] for n in known})
        return pd.DataFrame({"kwh": energy, "ugx": cost,
                             "energy_share": energy / energy.sum(),
                             "cost_share": cost / cost.sum()})


class DemandSource(SeriesValidationMixin):
    """The two input modes: a generated CSV, and typed entry.

    Parameters
    ----------
    columns : sequence of str
        One name per load.
    """

    def __init__(self, columns: Sequence[str] = ("D1", "D2")) -> None:
        if not columns:
            raise ValueError("need at least one demand column")
        self.columns = tuple(columns)

    def __repr__(self) -> str:
        return f"DemandSource{self.columns}"

    def __iter__(self) -> Iterator[str]:
        return iter(self.columns)

    # ----- mode (b): the CSV ------------------------------------------------
    def write_csv(self, path: str | Path, days: int = 30, seed: int = 1234,
                  means: Sequence[float] = (100.0, 120.0),
                  weekly: float = 0.14, noise: float = 0.06) -> pd.DataFrame:
        """Generate ``days`` days of demand and save them.

        A health centre is busier on weekdays, so the series carries a weekly
        sinusoid multiplied by lognormal-ish noise. Everything is drawn from
        ``np.random.default_rng(seed)``, so the file is identical on every run.
        """
        n = self.whole_number(days, "days")
        if len(means) != len(self.columns):
            raise ValueError(f"need {len(self.columns)} means, got {len(means)}")
        if not 0 <= weekly < 1 or noise < 0:
            raise ValueError("weekly must be in [0, 1) and noise non-negative")
        rng = np.random.default_rng(seed)
        index = np.arange(n)
        rhythm = 1 + weekly * np.sin(2 * np.pi * index / 7 + 0.4)
        frame = pd.DataFrame({"day": index + 1})
        for name, level in zip(self.columns, means):
            frame[name] = np.round(level * rhythm * rng.normal(1.0, noise, n), 2)
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(destination, index=False)
        return frame

    def read_csv(self, path: str | Path) -> pd.DataFrame:
        """Load a demand file, rejecting anything the grid cannot use."""
        source = Path(path)
        if not source.exists():
            raise FileNotFoundError(f"no demand file at {source}")
        frame = pd.read_csv(source)
        absent = [c for c in self.columns if c not in frame.columns]
        if absent:
            raise ValueError(f"{source.name} is missing column(s) {absent}")
        block = frame[list(self.columns)]
        if block.isna().any().any():
            raise ValueError(f"{source.name} contains blanks")
        if (block < 0).any().any():
            raise ValueError(f"{source.name} contains negative demand")
        return frame

    def to_matrix(self, frame: pd.DataFrame) -> np.ndarray:
        """Demand columns as an ``(n_loads, n_days)`` array."""
        return frame[list(self.columns)].to_numpy(dtype=float).T

    # ----- mode (a): typed entry ---------------------------------------------
    @staticmethod
    def read_value(text: str) -> float:
        """Turn one typed line into a demand figure, or explain what is wrong."""
        if text is None or not str(text).strip():
            raise ValueError("nothing entered; please type a number")
        cleaned = str(text).strip().replace(",", "")
        try:
            value = float(cleaned)
        except ValueError:
            raise ValueError(f"{cleaned!r} is not a number") from None
        if not np.isfinite(value):
            raise ValueError("the demand must be finite")
        if value < 0:
            raise ValueError("the demand cannot be negative")
        return value

    @classmethod
    def prompt(cls, label: str, reader: Callable[[str], str] = input,
               writer: Callable[[str], None] = print, tries: int = 3) -> float:
        """Ask until a usable number arrives, or give up after ``tries``.

        The reader and writer are injected so this can be driven by the
        notebook and the tests without a human at a keyboard.
        """
        if tries < 1:
            raise ValueError("tries must be at least 1")
        for remaining in range(int(tries), 0, -1):
            try:
                return cls.read_value(reader(f"{label}: "))
            except ValueError as err:
                writer(f"  {err} ({remaining - 1} left)")
        raise ValueError(f"gave up after {tries} attempts")

    def prompt_all(self, reader: Callable[[str], str] = input,
                   writer: Callable[[str], None] = print) -> np.ndarray:
        """Collect one demand per load."""
        return np.array([self.prompt(f"{name} (kWh)", reader, writer)
                         for name in self.columns])


def perturbation_study(grid: MicroGrid, demands: ArrayLike, spread: float = 0.05,
                       draws: int = 1_000, seed: int = 1234) -> dict[str, object]:
    """Extension: re-solve under +/- ``spread`` demand error, ``draws`` times.

    Compares the observed error amplification against the condition number,
    which bounds it.
    """
    if not 0 < spread < 1:
        raise ValueError("spread must be in (0, 1)")
    n = int(draws)
    if n < 1:
        raise ValueError("draws must be at least 1")
    nominal = np.asarray(demands, dtype=float).ravel()
    if nominal.size != grid.size:
        raise ValueError(f"expected {grid.size} demands, got {nominal.size}")

    base = grid.solve_day(*nominal)
    rng = np.random.default_rng(seed)
    factors = rng.uniform(1 - spread, 1 + spread, size=(n, nominal.size))
    shifted = nominal * factors
    solutions = grid.solve_batch(shifted.T)

    out_error = np.linalg.norm(solutions - base[:, None], axis=0) / np.linalg.norm(base)
    in_error = np.linalg.norm((shifted - nominal).T, axis=0) / np.linalg.norm(nominal)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(in_error > 0, out_error / in_error, np.nan)
    return {"base": base, "solutions": solutions, "amplification": ratio,
            "worst": float(np.nanmax(ratio)), "typical": float(np.nanmedian(ratio)),
            "cond": grid.cond}
