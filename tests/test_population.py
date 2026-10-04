"""Tests for Mini-Project 1.

Style note: tests are grouped into classes, one per unit under test, so a
failure report names the component as well as the behaviour.
"""
import math
import statistics

import numpy as np
import pytest

from src.forecasting import EnsembleForecaster, Forecaster
from src.metrics import MetricSuite
from src.mixins import grid_search
from src.population import (
    CompoundGrowth, DistrictPopulation, GoldenRatioProjection, PolyfitTrend,
    SchoolCapacityModel, ValidationBench, fibonacci, load_series, projection,
    residual_bootstrap, spread_check)


class TestDistrictPopulation:
    """The container: construction, rejection and container behaviour."""

    def test_reports_its_own_shape(self, kampala):
        assert len(kampala) == 10
        assert "Kampala" in repr(kampala)
        assert kampala.district == "Kampala"

    def test_behaves_like_a_mapping_of_year_to_population(self, kampala):
        assert kampala[2015] == 1200
        assert kampala[2024] == 1800
        assert 2020 in kampala and 2031 not in kampala

    def test_iterates_as_year_population_pairs(self, kampala):
        pairs = list(kampala)
        assert len(pairs) == 10
        assert pairs[0] == (2015, 1200.0)
        assert all(isinstance(year, int) for year, _ in pairs)

    def test_unknown_year_raises_key_error(self, kampala):
        with pytest.raises(KeyError):
            kampala[1999]

    @pytest.mark.parametrize("years,thousands", [
        ([2020, 2021], [1, 2, 3]),            # lengths differ
        ([2020, 2021], [100, -5]),            # negative
        ([], []),                             # empty: the brief's edge case
        ([2021, 2020], [100, 110]),           # not increasing
        ([2020, 2021], [100, math.inf]),      # non-finite
    ])
    def test_rejects_unusable_input(self, years, thousands):
        with pytest.raises(ValueError):
            DistrictPopulation("Test", years, thousands)

    @pytest.mark.parametrize("name", ["", "  ", None, 7])
    def test_requires_a_real_name(self, name):
        with pytest.raises(ValueError):
            DistrictPopulation(name, [2020, 2021], [100, 110])


class TestDescriptiveStatistics:
    """Task 2: the stdlib/NumPy variance difference and what ddof does."""

    def test_stdlib_matches_numpy_at_ddof_one(self, kampala):
        stdlib = kampala.summary_stdlib()
        assert stdlib["variance"] == pytest.approx(kampala.summary_numpy(1)["variance"])
        assert stdlib["stdev"] == pytest.approx(kampala.summary_numpy(1)["stdev"])

    def test_numpy_defaults_to_the_population_variance(self, kampala):
        n = len(kampala)
        sample = kampala.summary_stdlib()["variance"]
        assert kampala.summary_numpy(0)["variance"] == pytest.approx(sample * (n - 1) / n)

    def test_mean_and_median_agree_with_the_stdlib(self, kampala):
        values = kampala.thousands.tolist()
        summary = kampala.summary_stdlib()
        assert summary["mean"] == pytest.approx(statistics.mean(values))
        assert summary["median"] == pytest.approx(statistics.median(values))

    def test_variance_needs_two_points(self):
        single = DistrictPopulation("One", [2020], [100])
        with pytest.raises(ValueError):
            single.summary_stdlib()


class TestGrowth:
    """Task 3: year-on-year change and the compound annual rate."""

    def test_annual_changes_are_one_shorter_than_the_series(self, kampala):
        assert kampala.annual_changes().size == len(kampala) - 1

    def test_cagr_reproduces_the_final_value(self, kampala):
        span = int(kampala.years[-1] - kampala.years[0])
        grown = kampala.thousands[0] * (1 + kampala.cagr()) ** span
        assert grown == pytest.approx(kampala.thousands[-1])

    def test_cagr_is_exact_on_a_geometric_series(self, geometric):
        assert geometric.cagr() == pytest.approx(0.05)

    def test_wakiso_grows_fastest_in_relative_terms(self, all_districts):
        rates = {s.district: s.cagr() for s in all_districts}
        assert max(rates, key=rates.__getitem__) == "Wakiso"

    def test_growth_after_a_zero_is_undefined(self):
        series = DistrictPopulation("Zero", [2020, 2021, 2022], [0, 50, 60])
        with pytest.raises(ValueError):
            series.annual_changes()


class TestModels:
    """Task 4: the three forecasting models and the shared lifecycle."""

    @pytest.mark.parametrize("builder", [PolyfitTrend, CompoundGrowth,
                                         GoldenRatioProjection])
    def test_every_model_is_a_forecaster(self, builder):
        assert issubclass(builder, Forecaster)
        assert hasattr(builder(), "fit") and hasattr(builder(), "predict")

    def test_polyfit_is_exact_on_a_straight_line(self, arithmetic):
        model = PolyfitTrend().fit(arithmetic.thousands, arithmetic.years)
        assert model.predict(2) == pytest.approx([200.0, 210.0])
        assert model.fitted() == pytest.approx(arithmetic.thousands)

    def test_compound_growth_recovers_a_known_rate(self, geometric):
        model = CompoundGrowth().fit(geometric.thousands, geometric.years)
        assert model.rate_ == pytest.approx(0.05)
        assert model.predict(1)[0] == pytest.approx(geometric.thousands[-1] * 1.05)

    def test_golden_ratio_starts_from_the_unsettled_ratios(self, kampala):
        model = GoldenRatioProjection(from_term=2).fit(kampala.thousands)
        # F(3)/F(2) = 3/2 = 1.5 is the first ratio applied at from_term=2
        assert model.predict(1)[0] / kampala.thousands[-1] == pytest.approx(1.5)
        assert model.ratios_[0] == pytest.approx(1.5)

    def test_fibonacci_ratios_converge_on_phi(self):
        sequence = fibonacci(30)
        assert sequence[:6].tolist() == [1, 1, 2, 3, 5, 8]
        assert sequence[-1] / sequence[-2] == pytest.approx((1 + math.sqrt(5)) / 2)

    @pytest.mark.parametrize("count", [0, -2, 2.5, "4"])
    def test_fibonacci_rejects_a_bad_count(self, count):
        with pytest.raises(ValueError):
            fibonacci(count)

    @pytest.mark.parametrize("builder", [PolyfitTrend, CompoundGrowth,
                                         GoldenRatioProjection])
    def test_predicting_before_fitting_raises(self, builder):
        with pytest.raises(RuntimeError):
            builder().predict(2)

    @pytest.mark.parametrize("horizon", [0, -3, 1.5, "2", True])
    def test_horizon_must_be_a_positive_integer(self, kampala, horizon):
        model = CompoundGrowth().fit(kampala.thousands)
        with pytest.raises(ValueError):
            model.predict(horizon)

    def test_fit_returns_self_so_calls_chain(self, kampala):
        assert CompoundGrowth().fit(kampala.thousands).predict(2).size == 2

    def test_fit_rejects_a_series_that_is_too_short(self):
        with pytest.raises(ValueError):
            PolyfitTrend().fit([100.0])


class TestEnsemble:
    """The ``__add__`` dunder and the ensemble it builds."""

    def test_adding_two_models_builds_an_ensemble(self):
        combined = PolyfitTrend() + CompoundGrowth()
        assert isinstance(combined, EnsembleForecaster)
        assert len(combined) == 2
        assert combined.label == "polyfit-trend + compound-growth"

    def test_adding_a_third_model_flattens(self):
        combined = PolyfitTrend() + CompoundGrowth() + GoldenRatioProjection()
        assert len(combined) == 3
        assert all(not isinstance(member, EnsembleForecaster) for member in combined)

    def test_ensemble_averages_its_members(self, kampala):
        trend, growth = PolyfitTrend(), CompoundGrowth()
        combined = PolyfitTrend() + CompoundGrowth()
        for model in (trend, growth, combined):
            model.fit(kampala.thousands, kampala.years.astype(float))
        expected = (trend.predict(3) + growth.predict(3)) / 2
        assert combined.predict(3) == pytest.approx(expected)

    def test_adding_a_non_model_is_not_supported(self):
        with pytest.raises(TypeError):
            PolyfitTrend() + 5

    def test_an_empty_ensemble_is_rejected(self):
        with pytest.raises(ValueError):
            EnsembleForecaster([])


class TestMixins:
    """Behaviour inherited rather than implemented."""

    def test_scoring_mixin_returns_all_three_measures(self, kampala):
        train, test = kampala.partition(2021)
        model = CompoundGrowth().fit(train.thousands, train.years.astype(float))
        suite = model.score(test.thousands)
        assert isinstance(suite, MetricSuite)
        assert suite.mae > 0 and suite.rmse >= suite.mae and suite.mape > 0

    def test_walk_forward_produces_one_forecast_per_origin(self, kampala):
        forecasts = CompoundGrowth().walk_forward(kampala.thousands, first_origin=4)
        assert forecasts.size == len(kampala) - 4

    @pytest.mark.parametrize("origin", [0, -1, 10, 99])
    def test_walk_forward_rejects_an_impossible_origin(self, kampala, origin):
        with pytest.raises(ValueError):
            CompoundGrowth().walk_forward(kampala.thousands, first_origin=origin)

    def test_grid_search_picks_the_lowest_error(self, kampala):
        best, scores = grid_search(lambda term: GoldenRatioProjection(int(term)),
                                   [2, 3, 4], kampala.thousands, first_origin=4)
        assert best in scores
        assert scores[best] == min(scores.values())

    def test_metric_suite_rejects_mismatched_input(self):
        with pytest.raises(ValueError):
            MetricSuite.compare([1, 2, 3], [1, 2])
        with pytest.raises(ValueError):
            MetricSuite.compare([0, 100], [1, 101])   # MAPE undefined at zero


class TestValidationBench:
    """Tasks 5-6: the split, the comparison and the selection."""

    def test_split_keeps_the_boundary_year_in_training(self, kampala):
        train, test = kampala.partition(2021)
        assert train.years[-1] == 2021 and test.years[0] == 2022
        assert len(train) == 7 and len(test) == 3

    @pytest.mark.parametrize("year", [2014, 2024, 2040])
    def test_a_split_that_empties_one_side_raises(self, kampala, year):
        with pytest.raises(ValueError):
            kampala.partition(year)

    def test_compound_growth_wins_on_every_district(self, all_districts):
        bench = ValidationBench()
        assert {bench.winner(s) for s in all_districts} == {"compound-growth"}

    def test_table_is_ordered_most_accurate_first(self, kampala):
        table = ValidationBench().table(kampala)
        assert list(table.columns) == ["MAE", "RMSE", "MAPE_%"]
        assert table["MAPE_%"].is_monotonic_increasing
        assert table.index[0] == "compound-growth"

    def test_bench_needs_at_least_one_model(self):
        with pytest.raises(ValueError):
            ValidationBench(builders={})

    def test_forecast_variance_is_below_observed(self, kampala):
        bench = ValidationBench()
        path = projection(kampala, bench.refit_winner(kampala))
        assert 0 < spread_check(kampala, path.to_numpy())["ratio"] < 1

    def test_projection_is_indexed_by_calendar_year(self, kampala):
        path = projection(kampala, ValidationBench().refit_winner(kampala))
        assert list(path.index) == [2025, 2026, 2027, 2028, 2029]


class TestResidualBootstrap:
    """The prediction-interval extension."""

    def test_interval_brackets_the_point_forecast(self, kampala):
        model = ValidationBench().refit_winner(kampala)
        band = residual_bootstrap(model, draws=1000, seed=1234)
        assert (band["lower"] <= band["upper"]).all()
        assert len(band) == 5

    def test_a_fixed_seed_reproduces_the_interval(self, kampala):
        model = ValidationBench().refit_winner(kampala)
        first = residual_bootstrap(model, draws=500, seed=1234)
        second = residual_bootstrap(model, draws=500, seed=1234)
        assert first["lower"].to_numpy() == pytest.approx(second["lower"].to_numpy())

    @pytest.mark.parametrize("coverage", [0.0, 1.0, -0.5, 2.0])
    def test_coverage_must_be_a_proper_probability(self, kampala, coverage):
        model = ValidationBench().refit_winner(kampala)
        with pytest.raises(ValueError):
            residual_bootstrap(model, coverage=coverage)


class TestSchoolCapacityModel:
    """Task 8: turning growth into classrooms."""

    @pytest.mark.parametrize("kwargs", [
        {"primary_share": 0.0}, {"primary_share": 1.2},
        {"pupils_per_room": 0}, {"pupils_per_room": -5},
    ])
    def test_impossible_assumptions_are_rejected(self, kwargs):
        with pytest.raises(ValueError):
            SchoolCapacityModel(**kwargs)

    def test_matches_a_hand_calculation(self):
        planner = SchoolCapacityModel()
        # 100k more people x 18% = 18,000 pupils; 18,000 / 53 = 339.6 -> 340
        assert planner.rooms_to_add(1_000, 1_100) == 340
        assert planner.enrolment(100) == pytest.approx(18_000)

    def test_no_rooms_needed_when_population_falls(self):
        assert SchoolCapacityModel().rooms_to_add(900, 850) == 0

    def test_the_planner_is_immutable(self):
        planner = SchoolCapacityModel()
        with pytest.raises(Exception):
            planner.primary_share = 0.25

    def test_negative_population_is_rejected(self):
        with pytest.raises(ValueError):
            SchoolCapacityModel().enrolment(-1)
