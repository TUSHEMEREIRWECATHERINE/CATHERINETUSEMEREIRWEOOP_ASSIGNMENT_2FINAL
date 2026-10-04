"""Mini-Project 4: Rainfall Pattern & Crop Suitability Analyser.

An agricultural extension officer wants to compare rainfall regimes across
regions and advise farmers on which months suit which crops.

Classes
-------
:class:`Region`
    One region's twelve monthly totals, behaving as a month-to-millimetres
    mapping.
:class:`CropRule`
    A crop's suitable monthly band, with the source of the figures cited.
:class:`SeasonDetector`
    Peak detection and the unimodal/bimodal verdict.
:class:`CropCalendar`
    Composes regions and crop rules into the month-by-region classification.
:class:`RegionComparison`
    The three pairwise measures, as matrices.

The previous version of this question asked for "cosine similarity" via
``math.cos``, which returns the cosine of an *angle* and says nothing about
two rainfall profiles. :func:`cosine_similarity` implements the real thing and
the notebook checks it against SciPy.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Sequence

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy.signal import find_peaks, peak_prominences
from scipy.spatial.distance import cosine as scipy_cosine

from src.mixins import SeriesValidationMixin

__all__ = ["CROPS", "MONTHS", "RAINFALL", "CropCalendar", "CropRule",
           "RegionComparison", "Region", "SeasonDetector", "cosine_similarity",
           "euclidean_distance", "load_power_csv", "load_regions",
           "pearson_correlation"]

MONTHS: tuple[str, ...] = ("Jan", "Feb", "Mar", "Apr", "May", "Jun",
                           "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")

#: Illustrative monthly rainfall in mm, as given in the brief.
RAINFALL: dict[str, tuple[float, ...]] = {
    "Kampala": (120, 140, 180, 200, 220, 180, 90, 70, 60, 100, 110, 130),
    "Gulu": (8, 25, 75, 160, 190, 145, 170, 215, 175, 150, 60, 15),
    "Mbarara": (70, 85, 120, 140, 90, 25, 20, 55, 100, 125, 120, 90),
}

FAO = ("FAO (1986) Irrigation Water Management Training Manual No. 3: "
       "Irrigation Water Needs, Brouwer & Heibloem, Table 4")

#: Labels used by the classification.
DRY, FINE, WET = "drought risk", "good", "waterlogging risk"


class Region(SeriesValidationMixin):
    """One region's monthly rainfall in millimetres.

    Takes only the validation mixin: a rainfall profile needs the input rules
    but none of the forecasting machinery.

    Parameters
    ----------
    name : str
        Region name.
    rainfall : array-like of float
        Twelve monthly totals, January first, non-negative and finite.
    """

    min_obs = 12

    def __init__(self, name: str, rainfall: ArrayLike) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("name must be a non-empty string")
        values = self.clean(rainfall, "rainfall", self.min_obs)
        if values.size != len(MONTHS):
            raise ValueError(f"need exactly {len(MONTHS)} months, got {values.size}")
        if np.any(values < 0):
            raise ValueError("rainfall cannot be negative")
        self.name = name.strip()
        self.rainfall = values

    # ----- container behaviour ------------------------------------------
    def __repr__(self) -> str:
        return (f"<Region {self.name}: {self.annual:,.0f} mm/yr, "
                f"wettest {self.wettest}, driest {self.driest}>")

    def __len__(self) -> int:
        return int(self.rainfall.size)

    def __getitem__(self, month: str | int) -> float:
        """``region["Mar"]`` or ``region[2]``."""
        if isinstance(month, str):
            if month not in MONTHS:
                raise KeyError(f"{month!r} is not a month; expected one of {MONTHS}")
            return float(self.rainfall[MONTHS.index(month)])
        return float(self.rainfall[month])

    def __iter__(self) -> Iterator[tuple[str, float]]:
        return zip(MONTHS, self.rainfall.tolist())

    def __contains__(self, month: object) -> bool:
        return month in MONTHS

    # ----- task 1 ---------------------------------------------------------
    @property
    def annual(self) -> float:
        return float(self.rainfall.sum())

    @property
    def monthly_mean(self) -> float:
        return float(self.rainfall.mean())

    @property
    def monthly_sd(self) -> float:
        return float(self.rainfall.std(ddof=1))

    @property
    def wettest(self) -> str:
        return MONTHS[int(np.argmax(self.rainfall))]

    @property
    def driest(self) -> str:
        return MONTHS[int(np.argmin(self.rainfall))]

    @property
    def cv(self) -> float:
        """Coefficient of variation across the twelve months.

        High means the rain is concentrated into a short season; low means it
        is spread through the year. More informative for planting advice than
        the annual total, which can be identical for very different regimes.
        """
        if self.monthly_mean == 0:
            raise ValueError("CV is undefined for a region with no rainfall")
        return self.monthly_sd / self.monthly_mean

    def profile(self) -> dict[str, float | str]:
        return {"annual_mm": self.annual, "mean_mm": self.monthly_mean,
                "sd_mm": self.monthly_sd, "wettest": self.wettest,
                "driest": self.driest, "cv": self.cv}


@dataclass(frozen=True)
class CropRule:
    """A crop's suitable monthly rainfall band.

    Derived from FAO crop water *needs* over a full growing period: the lower
    bound is the smallest seasonal requirement divided by the longest growing
    period, the upper bound the largest requirement divided by the shortest.
    What results is the monthly rate at which the crop meets its seasonal need.

    Parameters
    ----------
    crop : str
        Crop name.
    low, high : float
        Suitable monthly band, in mm.
    citation : str
        Where the figures come from.
    """

    crop: str
    low: float
    high: float
    citation: str = ""

    def __post_init__(self) -> None:
        if not str(self.crop).strip():
            raise ValueError("the crop must be named")
        if not 0 <= self.low < self.high:
            raise ValueError(
                f"{self.crop}: need 0 <= low < high, got {self.low} and {self.high}")

    def verdict(self, mm: float) -> str:
        """``drought risk`` / ``good`` / ``waterlogging risk``."""
        if mm < 0 or not np.isfinite(mm):
            raise ValueError("rainfall must be finite and non-negative")
        if mm < self.low:
            return DRY
        return WET if mm > self.high else FINE

    def code(self, mm: float) -> int:
        """-1, 0 or +1, for the heatmap."""
        return {DRY: -1, FINE: 0, WET: 1}[self.verdict(mm)]

    def __str__(self) -> str:
        return f"{self.crop} ({self.low:.0f}-{self.high:.0f} mm/month)"


#: The three crops here are the staple, the wetland cereal and the main
#: rotation legume of Uganda's central and eastern farming systems, which is
#: where the advisory note at the end of the notebook is directed.
CROPS: tuple[CropRule, ...] = (
    # 1,200-2,200 mm over 300-365 days (10-12 months)
    CropRule("Banana", 100, 220, FAO + ": 1,200-2,200 mm over 300-365 days"),
    # 450-700 mm over 90-150 days (3.0-5.0 months)
    CropRule("Paddy rice", 90, 233, FAO + ": 450-700 mm over 90-150 days"),
    # 450-700 mm over 135-150 days (4.4-4.9 months)
    CropRule("Soybean", 92, 159, FAO + ": 450-700 mm over 135-150 days"),
)


# ----------------------------------------------------------------------
# Tasks 3-4: similarity
# ----------------------------------------------------------------------
def _aligned(a: ArrayLike, b: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    left = np.asarray(a, dtype=float).ravel()
    right = np.asarray(b, dtype=float).ravel()
    if left.size == 0:
        raise ValueError("vectors must not be empty")
    if left.shape != right.shape:
        raise ValueError(f"lengths differ: {left.size} and {right.size}")
    if not (np.all(np.isfinite(left)) and np.all(np.isfinite(right))):
        raise ValueError("vectors must be finite")
    return left, right


def cosine_similarity(a: ArrayLike, b: ArrayLike) -> float:
    """``a . b / (|a| |b|)`` -- the cosine of the angle *between two vectors*.

    This is what the previous version of the question got wrong.
    ``math.cos(x)`` takes a single angle in radians; it cannot express the
    relationship between two rainfall profiles at all. The quantity wanted is
    the normalised dot product, which equals 1 whenever two profiles have the
    same shape, however different their magnitudes.
    """
    left, right = _aligned(a, b)
    scale = np.linalg.norm(left) * np.linalg.norm(right)
    if scale == 0:
        raise ValueError("cosine similarity is undefined for a zero vector")
    return float(np.dot(left, right) / scale)


def pearson_correlation(a: ArrayLike, b: ArrayLike) -> float:
    """Cosine similarity after subtracting each mean.

    Centring is what lets it answer the question an extension officer cares
    about: do these regions get their rain in the *same months*?
    """
    left, right = _aligned(a, b)
    if left.size < 2:
        raise ValueError("correlation needs at least two observations")
    if left.std() == 0 or right.std() == 0:
        raise ValueError("correlation is undefined for a constant series")
    return float(np.corrcoef(left, right)[0, 1])


def euclidean_distance(a: ArrayLike, b: ArrayLike) -> float:
    """Straight-line distance in mm -- keeps the magnitudes cosine discards."""
    left, right = _aligned(a, b)
    return float(np.linalg.norm(left - right))


class RegionComparison:
    """Every pair of regions under three measures.

    Parameters
    ----------
    regions : sequence of Region
        At least two.
    """

    MEASURES: dict[str, Callable[[ArrayLike, ArrayLike], float]] = {
        "cosine": cosine_similarity,
        "pearson": pearson_correlation,
        "euclidean": euclidean_distance,
    }

    def __init__(self, regions: Sequence[Region]) -> None:
        if len(regions) < 2:
            raise ValueError("need at least two regions to compare")
        self.regions = list(regions)

    def __len__(self) -> int:
        return len(self.regions)

    def __repr__(self) -> str:
        return f"RegionComparison({[r.name for r in self.regions]})"

    def matrix(self, measure: str = "cosine") -> pd.DataFrame:
        if measure not in self.MEASURES:
            raise ValueError(f"unknown measure {measure!r}; "
                             f"expected one of {sorted(self.MEASURES)}")
        fn = self.MEASURES[measure]
        names = [r.name for r in self.regions]
        values = np.array([[fn(x.rainfall, y.rainfall) for y in self.regions]
                           for x in self.regions])
        return pd.DataFrame(values, index=names, columns=names)

    def every_matrix(self) -> dict[str, pd.DataFrame]:
        return {name: self.matrix(name) for name in self.MEASURES}

    def pairs(self) -> pd.DataFrame:
        """One row per pair, with all three measures side by side."""
        rows = []
        for i, x in enumerate(self.regions):
            for y in self.regions[i + 1:]:
                rows.append({
                    "pair": f"{x.name} / {y.name}",
                    "cosine": cosine_similarity(x.rainfall, y.rainfall),
                    "pearson": pearson_correlation(x.rainfall, y.rainfall),
                    "euclidean_mm": euclidean_distance(x.rainfall, y.rainfall),
                    "annual_gap_mm": abs(x.annual - y.annual),
                })
        return pd.DataFrame(rows).set_index("pair")

    def cross_check(self) -> pd.DataFrame:
        """Verify every cosine figure against ``scipy.spatial.distance.cosine``."""
        rows = []
        for i, x in enumerate(self.regions):
            for y in self.regions[i + 1:]:
                ours = cosine_similarity(x.rainfall, y.rainfall)
                scipys = 1 - float(scipy_cosine(x.rainfall, y.rainfall))
                rows.append({"pair": f"{x.name} / {y.name}", "ours": ours,
                             "scipy": scipys, "difference": abs(ours - scipys)})
        return pd.DataFrame(rows).set_index("pair")


# ----------------------------------------------------------------------
# Task 5: seasons
# ----------------------------------------------------------------------
class SeasonDetector:
    """Finds rainy seasons in a twelve-month profile.

    Two decisions make the result defensible.

    The calendar **wraps** -- December is next to January -- so the series is
    tiled three times and peaks are read from the middle copy. A plain
    twelve-point array would put an artificial cliff at both ends.

    A month counts as a season only if it clears two tests at once: its
    rainfall exceeds the **annual mean** (so a peak in an otherwise dry stretch
    does not qualify), and its **prominence exceeds one standard deviation** of
    the monthly totals. Scaling the prominence rule by the region's own
    variability rather than by a fixed fraction means the same detector suits a
    uniformly wet region and a sharply seasonal one.

    Parameters
    ----------
    region : Region
        The profile to analyse.
    sd_multiple : float
        How many standard deviations of prominence a peak must clear.
    gap : int
        Minimum months between peaks, so one broad season is not split in two.
    """

    def __init__(self, region: Region, sd_multiple: float = 1.0,
                 gap: int = 2) -> None:
        if sd_multiple <= 0:
            raise ValueError("sd_multiple must be positive")
        if gap < 1:
            raise ValueError("gap must be at least 1")
        self.region = region
        self.sd_multiple = float(sd_multiple)
        self.gap = int(gap)

    def __repr__(self) -> str:
        return (f"SeasonDetector({self.region.name!r}, peaks={self.months()}, "
                f"{self.verdict()})")

    def _find(self) -> tuple[np.ndarray, np.ndarray]:
        values = self.region.rainfall
        if self.region.monthly_sd == 0:
            return np.array([], dtype=int), np.array([])
        tiled = np.tile(values, 3)
        found, _ = find_peaks(
            tiled,
            height=self.region.monthly_mean,
            prominence=self.sd_multiple * self.region.monthly_sd,
            distance=self.gap)
        middle = found[(found >= values.size) & (found < 2 * values.size)]
        strength = peak_prominences(tiled, middle)[0] if middle.size else np.array([])
        return middle - values.size, strength

    def indices(self) -> np.ndarray:
        return self._find()[0]

    def months(self) -> list[str]:
        return [MONTHS[i] for i in self.indices()]

    def strengths(self) -> np.ndarray:
        return self._find()[1]

    def verdict(self) -> str:
        """``unimodal``, ``bimodal`` or ``polymodal``."""
        count = self.indices().size
        if count <= 1:
            return "unimodal"
        return "bimodal" if count == 2 else "polymodal"

    def summary(self) -> dict[str, object]:
        indices, strength = self._find()
        return {"region": self.region.name,
                "peaks": ", ".join(MONTHS[i] for i in indices) or "none",
                "n_peaks": int(indices.size),
                "threshold_mm": self.sd_multiple * self.region.monthly_sd,
                "strongest_mm": float(strength.max()) if strength.size else 0.0,
                "verdict": self.verdict()}


# ----------------------------------------------------------------------
# Tasks 2 and 6: the calendar
# ----------------------------------------------------------------------
class CropCalendar:
    """Classifies every month of every region against every crop.

    Composition: the calendar *has* regions and crop rules, because neither is
    a kind of the other and both stand alone.
    """

    def __init__(self, regions: Sequence[Region],
                 crops: Sequence[CropRule] = CROPS) -> None:
        if not regions:
            raise ValueError("need at least one region")
        if not crops:
            raise ValueError("need at least one crop rule")
        self.regions = list(regions)
        self.crops = list(crops)

    def __repr__(self) -> str:
        return (f"CropCalendar(regions={[r.name for r in self.regions]}, "
                f"crops={[c.crop for c in self.crops]})")

    def __len__(self) -> int:
        return len(self.regions) * len(self.crops)

    def __iter__(self) -> Iterator[CropRule]:
        return iter(self.crops)

    def _rule(self, crop: str) -> CropRule:
        for rule in self.crops:
            if rule.crop.lower() == crop.lower():
                return rule
        raise ValueError(f"unknown crop {crop!r}; "
                         f"have {[c.crop for c in self.crops]}")

    def verdicts(self, crop: str) -> pd.DataFrame:
        """Month-by-region table of labels for one crop."""
        rule = self._rule(crop)
        return pd.DataFrame(
            {r.name: [rule.verdict(v) for v in r.rainfall] for r in self.regions},
            index=list(MONTHS))

    def codes(self, crop: str) -> pd.DataFrame:
        """The same table as -1/0/+1."""
        rule = self._rule(crop)
        return pd.DataFrame(
            {r.name: [rule.code(v) for v in r.rainfall] for r in self.regions},
            index=list(MONTHS))

    def good_months(self, crop: str) -> pd.Series:
        return (self.codes(crop) == 0).sum()

    def summary(self) -> pd.DataFrame:
        """Count of suitable months for every crop and region."""
        return pd.DataFrame({c.crop: self.good_months(c.crop) for c in self.crops})

    def planting_window(self, crop: str, region: str) -> list[str]:
        """The months in which a given crop suits a given region."""
        column = self.verdicts(crop)[region]
        return [month for month, label in zip(MONTHS, column) if label == FINE]


def load_regions(names: Sequence[str] | None = None) -> list[Region]:
    """Build :class:`Region` objects from the illustrative data."""
    wanted = list(RAINFALL) if names is None else list(names)
    absent = [n for n in wanted if n not in RAINFALL]
    if absent:
        raise ValueError(f"unknown region(s) {absent}; have {sorted(RAINFALL)}")
    return [Region(n, RAINFALL[n]) for n in wanted]


def load_power_csv(path: str | Path) -> pd.DataFrame:
    """Read a NASA POWER monthly export into a year x month table of mm.

    Used by the optional extension; the notebook falls back to the
    illustrative data when no file is present, so it always runs.
    """
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"no NASA POWER file at {source}")
    lines = source.read_text(encoding="utf-8").splitlines()
    skip = next((i + 1 for i, line in enumerate(lines) if "-END HEADER-" in line), 0)
    frame = pd.read_csv(source, skiprows=skip)
    frame.columns = [str(c).strip().upper() for c in frame.columns]
    months = [c for c in frame.columns if c[:3].title() in MONTHS]
    if "YEAR" not in frame.columns or len(months) != 12:
        raise ValueError(f"{source.name} is not a POWER monthly export")
    table = frame.set_index("YEAR")[months]
    table.columns = list(MONTHS)
    return table.astype(float)
