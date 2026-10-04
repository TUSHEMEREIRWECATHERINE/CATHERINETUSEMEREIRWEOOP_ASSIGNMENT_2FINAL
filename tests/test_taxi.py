"""Tests for Mini-Project 5."""
import numpy as np
import pytest

from src.forecasting import EnsembleForecaster, Forecaster
from src.taxi import (
    ExponentialSmoothing, FareMarket, FleetSizer, MovingAverage, Route,
    SeasonalNaive, StraightLine, WeightedAverage, compare_models, load_routes,
    simulate_weekly_pattern, tune_smoothing)


@pytest.fixture
def ntinda() -> Route:
    return load_routes(["Kampala-Ntinda"])[0]


@pytest.fixture
def market() -> FareMarket:
    return FareMarket()


@pytest.fixture
def builders():
    return {"moving-average": lambda: MovingAverage(3),
            "weighted-average": lambda: WeightedAverage(3),
            "ses": lambda: ExponentialSmoothing(0.5),
            "straight-line": StraightLine}


class TestRoute:
    """Task 1."""

    def test_revenue_matches_a_hand_calculation(self, ntinda):
        assert ntinda.counts.sum() == 474
        assert ntinda.total_revenue == pytest.approx(474 * 2_000)
        assert ntinda.revenue_by_day[0] == pytest.approx(35 * 2_000)

    def test_statistics(self, ntinda):
        stats = ntinda.describe()
        assert stats["mean"] == pytest.approx(47.4)
        assert stats["cv"] == pytest.approx(stats["stdev"] / stats["mean"])

    def test_container_behaviour(self, ntinda):
        assert len(ntinda) == 10
        assert ntinda[1] == 35.0 and ntinda[10] == 45.0
        assert list(ntinda)[0] == (1, 35.0)

    def test_day_outside_the_range_raises(self, ntinda):
        with pytest.raises(KeyError):
            ntinda[0]
        with pytest.raises(KeyError):
            ntinda[11]

    @pytest.mark.parametrize("kwargs", [
        {"counts": []}, {"counts": [10, -5]}, {"fare": 0}, {"fare": -1},
        {"fare": np.inf}, {"name": ""}])
    def test_bad_input_rejected(self, kwargs):
        base = {"name": "Test", "counts": [10, 20], "fare": 1_000.0}
        with pytest.raises(ValueError):
            Route(**{**base, **kwargs})

    def test_statistics_need_two_days(self):
        with pytest.raises(ValueError):
            Route("One", [40], 2_000).describe()


class TestFareMarket:
    """Task 2."""

    def test_clearing_point_matches_the_algebra(self, market):
        # 120 - 0.02P = 10 + 0.03P  ->  110 = 0.05P  ->  P* = 2,200, Q* = 76
        price, quantity = market.clearing()
        assert price == pytest.approx(2_200.0)
        assert quantity == pytest.approx(76.0)
        assert price == pytest.approx(110 / 0.05)

    def test_both_curves_agree_at_the_clearing_price(self, market):
        price, quantity = market.clearing()
        assert market.demanded(price) == pytest.approx(quantity)
        assert market.supplied(price) == pytest.approx(quantity)

    def test_the_current_fare_leaves_a_shortage(self, market):
        gap = market.imbalance(2_000)
        assert gap["demanded"] == pytest.approx(80.0)
        assert gap["supplied"] == pytest.approx(70.0)
        assert gap["shortage"] == pytest.approx(10.0)
        assert gap["below_clearing_by"] == pytest.approx(200.0)

    def test_above_clearing_the_shortage_becomes_a_surplus(self, market):
        assert market.imbalance(2_500)["shortage"] < 0

    def test_the_system_is_well_conditioned(self, market):
        assert market.det == pytest.approx(0.05)
        assert market.cond < 100

    @pytest.mark.parametrize("kwargs", [
        {"demand_slope": 0}, {"supply_slope": -0.01},
        {"demand_at_zero": 5, "supply_at_zero": 10}])
    def test_impossible_curves_rejected(self, kwargs):
        with pytest.raises(ValueError):
            FareMarket(**kwargs)

    def test_negative_price_rejected(self, market):
        with pytest.raises(ValueError):
            market.imbalance(-10)


class TestModels:
    """Task 3."""

    @pytest.mark.parametrize("build", [
        lambda: MovingAverage(3), lambda: WeightedAverage(3),
        lambda: ExponentialSmoothing(0.5), StraightLine, lambda: SeasonalNaive(7)])
    def test_every_model_is_a_forecaster(self, build):
        assert isinstance(build(), Forecaster)

    def test_moving_average_is_the_window_mean(self, ntinda):
        model = MovingAverage(3).fit(ntinda.counts)
        assert model.predict(1)[0] == pytest.approx(np.mean([52, 47, 45]))
        assert model.predict(3) == pytest.approx([model.level] * 3)

    def test_weighted_average_leans_on_the_latest_day(self, ntinda):
        plain = MovingAverage(3).fit(ntinda.counts).predict(1)[0]
        weighted = WeightedAverage(3).fit(ntinda.counts).predict(1)[0]
        # the last three are 52, 47, 45 -- falling, so the weighted mean is lower
        assert weighted < plain
        assert WeightedAverage(3).weights.sum() == pytest.approx(1.0)

    def test_ses_recursion_by_hand(self):
        model = ExponentialSmoothing(0.5).fit([10.0, 20.0, 30.0])
        # l0 = 10; l1 = .5*20 + .5*10 = 15; l2 = .5*30 + .5*15 = 22.5
        assert model.predict(1)[0] == pytest.approx(22.5)

    def test_ses_at_alpha_one_is_naive(self, ntinda):
        assert ExponentialSmoothing(1.0).fit(ntinda.counts).predict(1)[0] == \
            pytest.approx(ntinda.counts[-1])

    @pytest.mark.parametrize("alpha", [0.0, -0.1, 1.5])
    def test_bad_alpha_rejected(self, alpha):
        with pytest.raises(ValueError):
            ExponentialSmoothing(alpha)

    def test_straight_line_is_exact_on_a_line(self):
        model = StraightLine().fit([10.0, 20.0, 30.0, 40.0])
        assert model.predict(2) == pytest.approx([50.0, 60.0])

    def test_seasonal_naive_repeats_the_previous_week(self):
        series = np.arange(1.0, 15.0)
        model = SeasonalNaive(7).fit(series)
        assert model.predict(7) == pytest.approx(series[-7:])

    @pytest.mark.parametrize("build", [lambda: MovingAverage(3), StraightLine,
                                       lambda: SeasonalNaive(7)])
    def test_predicting_before_fitting_raises(self, build):
        with pytest.raises(RuntimeError):
            build().predict(1)

    @pytest.mark.parametrize("build,series", [
        (lambda: MovingAverage(3), [40.0, 42.0]),
        (lambda: SeasonalNaive(7), list(np.arange(5.0))),
        (StraightLine, [40.0])])
    def test_models_insist_on_enough_history(self, build, series):
        with pytest.raises(ValueError):
            build().fit(series)


class TestBacktesting:
    """Task 4, using the mixin shared with mini-project 1."""

    def test_backtesting_comes_from_the_mixin_not_this_module(self, ntinda):
        """Nothing in src/taxi.py reimplements walk-forward scoring."""
        model = MovingAverage(3)
        assert hasattr(model, "walk_forward") and hasattr(model, "walk_forward_score")
        forecasts = model.walk_forward(ntinda.counts, first_origin=3)
        assert forecasts.size == len(ntinda) - 3

    def test_comparison_covers_days_four_to_ten(self, ntinda, builders):
        table = compare_models(ntinda.counts, builders, first_origin=3)
        assert set(table.index) == set(builders)
        assert table["MAE"].is_monotonic_increasing

    def test_comparison_needs_a_model(self, ntinda):
        with pytest.raises(ValueError):
            compare_models(ntinda.counts, {}, first_origin=3)

    @pytest.mark.parametrize("origin", [0, -1, 10, 50])
    def test_impossible_origin_rejected(self, ntinda, origin):
        with pytest.raises(ValueError):
            MovingAverage(3).walk_forward(ntinda.counts, first_origin=origin)

    def test_tuning_finds_the_minimising_alpha(self, ntinda):
        best, scores = tune_smoothing(ntinda.counts)
        assert 0 < best <= 1
        assert scores[best] == pytest.approx(scores.min())

    @pytest.mark.parametrize("grid", [[], [0.0, 0.5], [0.5, 1.5]])
    def test_bad_alpha_grids_rejected(self, ntinda, grid):
        with pytest.raises(ValueError):
            tune_smoothing(ntinda.counts, grid=grid)


class TestEnsemble:
    """The __add__ operator, applied to a second mini-project."""

    def test_models_still_add_together_here(self):
        combined = MovingAverage(3) + SeasonalNaive(7)
        assert isinstance(combined, EnsembleForecaster)
        assert len(combined) == 2
        assert combined.min_obs == 7           # the stricter of the two

    def test_the_ensemble_averages_its_members(self):
        series = simulate_weekly_pattern(40, seed=1234)
        moving, seasonal = MovingAverage(3), SeasonalNaive(7)
        combined = MovingAverage(3) + SeasonalNaive(7)
        for model in (moving, seasonal, combined):
            model.fit(series)
        assert combined.predict(3) == pytest.approx(
            (moving.predict(3) + seasonal.predict(3)) / 2)

    def test_the_ensemble_lands_between_its_members(self):
        """Consistent with mini-project 1: averaging dilutes the better model."""
        series = simulate_weekly_pattern(60, seed=1234)
        table = compare_models(series, {
            "moving": lambda: MovingAverage(3),
            "seasonal": lambda: SeasonalNaive(7),
            "ensemble": lambda: MovingAverage(3) + SeasonalNaive(7),
        }, first_origin=14)
        assert (table.loc["seasonal", "MAE"] < table.loc["ensemble", "MAE"]
                < table.loc["moving", "MAE"])


class TestFleetSizer:
    """Task 6."""

    def test_capacity_matches_the_brief(self):
        assert FleetSizer().daily_capacity == 14 * 8 == 112

    def test_rounding_to_nearest_with_a_utilisation_floor(self):
        sizer = FleetSizer()
        # 50 * 1.15 = 57.5, / 112 = 0.51 -> rounds to 1 (floored at 1 anyway)
        assert sizer.vehicles(50) == 1
        # 200 * 1.15 = 230, / 112 = 2.05 -> nearest is 2
        assert sizer.vehicles(200) == 2
        # 300 * 1.15 = 345, / 112 = 3.08 -> nearest is 3
        assert sizer.vehicles(300) == 3

    def test_the_floor_removes_a_badly_underused_vehicle(self):
        """A second vehicle is not added if it would run under 60% full."""
        strict = FleetSizer(min_utilisation=0.95)
        relaxed = FleetSizer(min_utilisation=0.10)
        assert strict.vehicles(200) <= relaxed.vehicles(200)

    def test_utilisation_is_reported(self):
        sizer = FleetSizer()
        assert sizer.utilisation(112, 1) == pytest.approx(1.0)
        assert sizer.utilisation(56, 1) == pytest.approx(0.5)

    def test_zero_and_one_passenger(self):
        sizer = FleetSizer()
        assert sizer.vehicles(0) == 0
        assert sizer.vehicles(1) == 1

    def test_plan_reports_a_row_per_route(self):
        plan = FleetSizer().plan({"A": 45.0, "B": 600.0})
        assert list(plan.index) == ["A", "B"]
        assert (plan["vehicles"] >= 1).all()
        assert (plan["utilisation"] > 0).all()

    @pytest.mark.parametrize("kwargs", [
        {"seats": 0}, {"trips": 0}, {"buffer": -0.1},
        {"min_utilisation": 0}, {"min_utilisation": 1.5}])
    def test_impossible_settings_rejected(self, kwargs):
        with pytest.raises(ValueError):
            FleetSizer(**kwargs)

    def test_negative_passengers_rejected(self):
        with pytest.raises(ValueError):
            FleetSizer().vehicles(-1)


class TestWeeklyExtension:
    def test_simulation_is_reproducible_and_weekly(self):
        first = simulate_weekly_pattern(60, seed=1234)
        assert first == pytest.approx(simulate_weekly_pattern(60, seed=1234))
        by_weekday = [first[i::7].mean() for i in range(7)]
        assert int(np.argmax(by_weekday)) == 4      # Friday
        assert int(np.argmin(by_weekday)) == 6      # Sunday

    @pytest.mark.parametrize("kwargs", [{"days": 0}, {"base": 0}, {"noise": -1},
                                        {"multipliers": [1.0] * 5}])
    def test_bad_settings_rejected(self, kwargs):
        with pytest.raises(ValueError):
            simulate_weekly_pattern(**{"days": 60, **kwargs})

    def test_seasonal_naive_beats_the_moving_average(self):
        series = simulate_weekly_pattern(60, seed=1234)
        table = compare_models(series, {
            "moving": lambda: MovingAverage(3),
            "seasonal": lambda: SeasonalNaive(7)}, first_origin=14)
        assert table.loc["seasonal", "MAE"] < table.loc["moving", "MAE"] * 0.6
