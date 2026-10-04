"""Tests for Mini-Project 4."""
import math

import numpy as np
import pytest
from scipy.spatial.distance import cosine as scipy_cosine

from src.rainfall import (
    CROPS, DRY, FINE, MONTHS, WET, CropCalendar, CropRule, Region,
    RegionComparison, SeasonDetector, cosine_similarity, euclidean_distance,
    load_power_csv, load_regions, pearson_correlation)


@pytest.fixture
def regions() -> list[Region]:
    return load_regions()


@pytest.fixture
def kampala(regions) -> Region:
    return regions[0]


class TestRegion:
    """Task 1."""

    def test_summary_statistics(self, kampala):
        assert kampala.annual == pytest.approx(1_600.0)
        assert kampala.monthly_mean == pytest.approx(1_600 / 12)
        assert kampala.wettest == "May" and kampala.driest == "Sep"
        assert 0 < kampala.cv < 1

    def test_container_behaviour(self, kampala):
        assert kampala["May"] == 220.0 and kampala[4] == 220.0
        assert len(kampala) == 12
        assert list(kampala)[0] == ("Jan", 120.0)
        assert "Mar" in kampala and "Smarch" not in kampala

    def test_unknown_month_raises(self, kampala):
        with pytest.raises(KeyError):
            kampala["Smarch"]

    @pytest.mark.parametrize("rainfall", [
        [100] * 11, [100] * 13, [100] * 11 + [-5],
        [100] * 11 + [math.nan], []])
    def test_bad_profiles_rejected(self, rainfall):
        with pytest.raises(ValueError):
            Region("Test", rainfall)

    @pytest.mark.parametrize("name", ["", "  ", None, 7])
    def test_name_required(self, name):
        with pytest.raises(ValueError):
            Region(name, [100] * 12)

    def test_cv_identifies_the_most_seasonal_region(self, regions):
        by_cv = {r.name: r.cv for r in regions}
        assert max(by_cv, key=by_cv.__getitem__) == "Gulu"


class TestCropRule:
    """Task 2."""

    @pytest.mark.parametrize("mm,expected", [
        (50, DRY), (99, DRY), (100, FINE), (220, FINE), (221, WET), (500, WET)])
    def test_banana_band_boundaries(self, mm, expected):
        banana = CROPS[0]
        assert banana.crop == "Banana"
        assert banana.verdict(mm) == expected

    def test_codes_map_to_minus_one_zero_one(self):
        banana = CROPS[0]
        assert banana.code(50) == -1 and banana.code(150) == 0 and banana.code(500) == 1

    @pytest.mark.parametrize("kwargs", [
        {"low": 200, "high": 100}, {"low": -10}, {"low": 100, "high": 100},
        {"crop": ""}])
    def test_impossible_bands_rejected(self, kwargs):
        base = {"crop": "Test", "low": 100.0, "high": 200.0}
        with pytest.raises(ValueError):
            CropRule(**{**base, **kwargs})

    def test_rules_are_frozen_and_cited(self):
        assert all(rule.citation for rule in CROPS)
        assert all("FAO" in rule.citation for rule in CROPS)
        with pytest.raises(Exception):
            CROPS[0].low = 0

    def test_negative_rainfall_rejected(self):
        with pytest.raises(ValueError):
            CROPS[0].verdict(-5)


class TestSimilarity:
    """Tasks 3-4."""

    def test_cosine_matches_scipy_on_every_pair(self, regions):
        for i, a in enumerate(regions):
            for b in regions[i + 1:]:
                assert cosine_similarity(a.rainfall, b.rainfall) == pytest.approx(
                    1 - float(scipy_cosine(a.rainfall, b.rainfall)), abs=1e-12)

    def test_cosine_ignores_magnitude_entirely(self):
        assert cosine_similarity([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)
        assert cosine_similarity([1, 2, 3], [10, 20, 30]) == pytest.approx(1.0)

    def test_cosine_of_orthogonal_vectors_is_zero(self):
        assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)

    def test_cosine_needs_a_non_zero_vector(self):
        with pytest.raises(ValueError):
            cosine_similarity([0, 0, 0], [1, 2, 3])

    @pytest.mark.parametrize("fn", [cosine_similarity, pearson_correlation,
                                    euclidean_distance])
    def test_measures_reject_mismatched_input(self, fn):
        with pytest.raises(ValueError):
            fn([1, 2, 3], [1, 2])
        with pytest.raises(ValueError):
            fn([], [])

    def test_positive_data_forces_cosine_high_while_pearson_does_not(self, regions):
        """The heart of task 4."""
        comparison = RegionComparison(regions)
        off_diagonal = comparison.matrix("cosine").to_numpy()[
            ~np.eye(len(regions), dtype=bool)]
        assert off_diagonal.min() > 0.7
        pearson = comparison.matrix("pearson").to_numpy()[
            ~np.eye(len(regions), dtype=bool)]
        assert abs(pearson).max() < 0.5

    def test_matrices_are_symmetric_with_expected_diagonals(self, regions):
        comparison = RegionComparison(regions)
        for name in ("cosine", "pearson", "euclidean"):
            matrix = comparison.matrix(name).to_numpy()
            assert matrix == pytest.approx(matrix.T)
        assert np.diag(comparison.matrix("cosine")) == pytest.approx([1.0] * 3)
        assert np.diag(comparison.matrix("euclidean")) == pytest.approx([0.0] * 3)

    def test_cross_check_shows_no_difference(self, regions):
        checks = RegionComparison(regions).cross_check()
        assert len(checks) == 3 and checks["difference"].max() < 1e-12

    def test_comparison_validates_its_inputs(self, regions):
        with pytest.raises(ValueError):
            RegionComparison(regions).matrix("manhattan")
        with pytest.raises(ValueError):
            RegionComparison(regions[:1])


class TestSeasonDetector:
    """Task 5: the standard-deviation prominence rule."""

    def test_verdicts_match_known_climatology(self, regions):
        verdicts = {r.name: SeasonDetector(r).verdict() for r in regions}
        assert verdicts["Gulu"] == "unimodal"
        assert verdicts["Mbarara"] == "bimodal"

    def test_mbarara_peaks_in_april_and_october(self, regions):
        mbarara = next(r for r in regions if r.name == "Mbarara")
        assert SeasonDetector(mbarara).months() == ["Apr", "Oct"]

    def test_gulu_peaks_once_in_august(self, regions):
        gulu = next(r for r in regions if r.name == "Gulu")
        assert SeasonDetector(gulu).months() == ["Aug"]

    def test_the_threshold_scales_with_the_regions_own_variability(self, regions):
        """Why the rule uses sd rather than a fixed fraction."""
        thresholds = {r.name: SeasonDetector(r).summary()["threshold_mm"]
                      for r in regions}
        # Gulu is the most variable region, so it faces the strictest test
        assert thresholds["Gulu"] > thresholds["Kampala"] > thresholds["Mbarara"]

    def test_detection_wraps_the_calendar(self):
        wrapping = Region("Wrap", [200, 60, 20, 20, 20, 20, 20, 20, 20, 20, 60, 190])
        assert SeasonDetector(wrapping).months() == ["Jan"]

    def test_a_flat_profile_has_no_peaks(self):
        assert SeasonDetector(Region("Flat", [100] * 12)).indices().size == 0
        assert SeasonDetector(Region("Flat", [100] * 12)).verdict() == "unimodal"

    def test_a_shallow_notch_is_not_a_second_season(self):
        """May and Jul are both local maxima, but Jun barely dips."""
        notched = Region("Notched", [20, 40, 90, 150, 200, 185,
                                     195, 160, 100, 50, 25, 20])
        assert SeasonDetector(notched, sd_multiple=1.0).months() == ["May"]
        assert SeasonDetector(notched, sd_multiple=0.1).verdict() == "bimodal"

    def test_a_peak_below_the_annual_mean_does_not_count(self):
        """The height test: a bump in a dry stretch is not a rainy season."""
        lopsided = Region("Lopsided", [5, 10, 40, 10, 5, 5,
                                       5, 300, 280, 300, 10, 5])
        detector = SeasonDetector(lopsided)
        assert "Mar" not in detector.months()

    @pytest.mark.parametrize("kwargs", [{"sd_multiple": 0}, {"sd_multiple": -1},
                                        {"gap": 0}])
    def test_impossible_settings_rejected(self, kampala, kwargs):
        with pytest.raises(ValueError):
            SeasonDetector(kampala, **kwargs)


class TestCropCalendar:
    """Tasks 2 and 6."""

    def test_verdict_table_covers_every_month_and_region(self, regions):
        table = CropCalendar(regions).verdicts("Banana")
        assert table.shape == (12, 3)
        assert list(table.index) == list(MONTHS)
        assert set(table.to_numpy().ravel()) <= {DRY, FINE, WET}

    def test_codes_agree_with_verdicts(self, regions):
        calendar = CropCalendar(regions)
        assert ((calendar.verdicts("Soybean") == FINE)
                == (calendar.codes("Soybean") == 0)).all().all()

    def test_summary_counts_suitable_months(self, regions):
        summary = CropCalendar(regions).summary()
        assert list(summary.columns) == ["Banana", "Paddy rice", "Soybean"]
        assert (summary >= 0).all().all() and (summary <= 12).all().all()

    def test_planting_window_lists_the_good_months(self, regions):
        window = CropCalendar(regions).planting_window("Banana", "Kampala")
        assert all(month in MONTHS for month in window)
        assert len(window) == CropCalendar(regions).good_months("Banana")["Kampala"]

    def test_crop_lookup_is_case_insensitive(self, regions):
        calendar = CropCalendar(regions)
        assert calendar.verdicts("banana").equals(calendar.verdicts("Banana"))

    def test_bad_inputs_rejected(self, regions):
        with pytest.raises(ValueError):
            CropCalendar(regions).verdicts("Quinoa")
        with pytest.raises(ValueError):
            CropCalendar([])
        with pytest.raises(ValueError):
            CropCalendar(regions, crops=[])


class TestLoading:
    def test_unknown_region_rejected(self):
        with pytest.raises(ValueError):
            load_regions(["Narnia"])

    def test_power_loader_reads_a_real_shaped_export(self, tmp_path):
        path = tmp_path / "power.csv"
        path.write_text(
            "-BEGIN HEADER-\nmeta\n-END HEADER-\n"
            "YEAR,JAN,FEB,MAR,APR,MAY,JUN,JUL,AUG,SEP,OCT,NOV,DEC\n"
            "2020,10,20,30,40,50,60,70,80,90,100,110,120\n"
            "2021,15,25,35,45,55,65,75,85,95,105,115,125\n", encoding="utf-8")
        table = load_power_csv(path)
        assert list(table.columns) == list(MONTHS)
        assert table.loc[2020, "Jan"] == pytest.approx(10.0)
        assert len(table) == 2

    def test_power_loader_rejects_other_files(self, tmp_path):
        path = tmp_path / "wrong.csv"
        path.write_text("a,b\n1,2\n", encoding="utf-8")
        with pytest.raises(ValueError):
            load_power_csv(path)
        with pytest.raises(FileNotFoundError):
            load_power_csv(tmp_path / "absent.csv")
