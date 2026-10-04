"""Shared fixtures, and the path setup that lets ``import src...`` work.

Running ``pytest`` from the repository root is enough; nothing is installed.
"""
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.population import DistrictPopulation, load_series  # noqa: E402


@pytest.fixture
def kampala() -> DistrictPopulation:
    """The brief's Kampala series, 2015-2024."""
    return load_series(["Kampala"])[0]


@pytest.fixture
def all_districts() -> list[DistrictPopulation]:
    """All five districts, including the two added for task 1."""
    return load_series()


@pytest.fixture
def arithmetic() -> DistrictPopulation:
    """A perfectly straight series: the linear model should be exact on it."""
    return DistrictPopulation("Arithmetic", list(range(2015, 2025)),
                              [100 + 10 * step for step in range(10)])


@pytest.fixture
def geometric() -> DistrictPopulation:
    """A perfectly compounding series: 5% a year, every year."""
    return DistrictPopulation("Geometric", list(range(2015, 2025)),
                              [100 * 1.05 ** step for step in range(10)])
