"""Tests for Mini-Project 3."""
import numpy as np
import pytest

from src.fishery import (
    KG_PER_TONNE, FisheryCase, FishStock, HarvestPolicy, PriceWalk, RiskProfile,
    fibonacci_series)


@pytest.fixture
def stock() -> FishStock:
    return FishStock()


@pytest.fixture
def case() -> FisheryCase:
    return FisheryCase(stock=FishStock(), price=PriceWalk(seed=1234))


class TestFibonacciBaseline:
    """Task 1: the baseline, kept only to be criticised."""

    def test_sequence_is_right(self):
        assert fibonacci_series(15).astype(int).tolist()[:7] == [1, 1, 2, 3, 5, 8, 13]
        assert fibonacci_series(15).size == 15

    def test_growth_never_slows(self):
        long = fibonacci_series(40)
        assert long[-1] / long[-2] > 1.6
        assert long[-1] > 1_000 * fibonacci_series(15)[-1]

    @pytest.mark.parametrize("count", [0, -3, 2.5, True])
    def test_bad_counts_rejected(self, count):
        with pytest.raises(ValueError):
            fibonacci_series(count)


class TestHarvestPolicy:
    """The policy object that separates the decision from the biology."""

    def test_default_is_open_all_year(self):
        policy = HarvestPolicy(0.10)
        assert policy.open_mask(52).all()
        assert len(policy) == 0
        assert "open all year" in str(policy)

    def test_closed_season_shuts_the_right_weeks(self):
        policy = HarvestPolicy.with_closed_season(0.10, start=0, length=8)
        mask = policy.open_mask(52)
        assert (~mask).sum() == 8
        assert not mask[0] and not mask[7] and mask[8]
        assert len(policy) == 8

    def test_rate_in_a_closed_week_is_zero(self):
        # start=3, length=4 shuts weeks 3, 4, 5 and 6 -- week 7 is open again
        policy = HarvestPolicy.with_closed_season(0.25, start=3, length=4)
        assert policy.closed_weeks == (3, 4, 5, 6)
        assert policy.rate_in(0) == pytest.approx(0.25)
        assert policy.rate_in(3) == pytest.approx(0.0)
        assert policy.rate_in(6) == pytest.approx(0.0)
        assert policy.rate_in(7) == pytest.approx(0.25)

    @pytest.mark.parametrize("kwargs", [{"rate": 1.0}, {"rate": -0.1},
                                        {"closed_weeks": (-1,)}])
    def test_impossible_policies_rejected(self, kwargs):
        with pytest.raises(ValueError):
            HarvestPolicy(**{"rate": 0.1, **kwargs})

    def test_closed_weeks_must_fit_the_year(self):
        with pytest.raises(ValueError):
            HarvestPolicy.with_closed_season(0.1, start=50, length=8).open_mask(52)

    def test_policy_is_frozen(self):
        with pytest.raises(Exception):
            HarvestPolicy(0.1).rate = 0.2


class TestFishStock:
    """Task 2: the logistic model and its reference points."""

    def test_defaults_match_the_brief(self, stock):
        assert (stock.r, stock.K, stock.N0) == (0.4, 10_000.0, 4_000.0)

    @pytest.mark.parametrize("kwargs", [{"r": 0}, {"K": -1}, {"N0": 0},
                                        {"N0": 20_000}, {"r": np.inf}])
    def test_impossible_parameters_rejected(self, kwargs):
        with pytest.raises(ValueError):
            FishStock(**kwargs)

    def test_msy_matches_the_closed_form(self, stock):
        assert stock.msy == pytest.approx(0.4 * 10_000 / 4)
        assert stock.msy_rate == pytest.approx(0.2)
        assert stock.sustainable_yield(stock.msy_rate) == pytest.approx(stock.msy)

    @pytest.mark.parametrize("rate,expected", [(0.0, 10_000), (0.1, 7_500),
                                               (0.2, 5_000), (0.4, 0.0)])
    def test_equilibrium_biomass(self, stock, rate, expected):
        assert stock.settles_at(rate) == pytest.approx(expected)

    def test_yield_peaks_at_the_msy_rate(self, stock):
        rates = np.linspace(0.01, 0.39, 200)
        yields = [stock.sustainable_yield(r) for r in rates]
        assert rates[int(np.argmax(yields))] == pytest.approx(0.2, abs=0.01)

    def test_simulation_converges_on_the_equilibrium(self, stock):
        run = stock.run(HarvestPolicy(0.1), weeks=200)
        assert run.closing_biomass == pytest.approx(stock.settles_at(0.1), rel=1e-6)

    def test_simulation_accounting(self, stock):
        run = stock.run(HarvestPolicy(0.1), weeks=52)
        assert len(run) == 52 and run.biomass.size == 53
        assert run.catch == pytest.approx(0.1 * run.biomass[:-1])
        assert run.landed == pytest.approx(run.catch.sum())

    def test_biomass_never_goes_negative(self, stock):
        assert stock.run(HarvestPolicy(0.39), weeks=300).biomass.min() >= 0

    def test_overfishing_lands_less_than_msy(self, stock):
        at_msy = stock.run(HarvestPolicy(0.20), 52)
        over = stock.run(HarvestPolicy(0.30), 52)
        assert over.landed < at_msy.landed
        assert over.closing_biomass < at_msy.closing_biomass

    def test_no_catch_during_a_closure(self, stock):
        run = stock.run(HarvestPolicy.with_closed_season(0.1, 0, 8), 52)
        assert run.catch[:8] == pytest.approx(np.zeros(8))
        assert (run.catch[8:] > 0).all()

    def test_a_closure_leaves_more_fish(self, stock):
        heavy = HarvestPolicy(0.35)
        rested = HarvestPolicy.with_closed_season(0.35, 0, 8)
        assert (stock.run(rested, 52).closing_biomass
                > stock.run(heavy, 52).closing_biomass)


class TestPriceWalk:
    """Task 3: the bounded random walk."""

    def test_path_stays_inside_the_bounds(self):
        walk = PriceWalk(weekly_sd=3_000, seed=1234)
        values = walk.simulate(weeks=500, paths=20)
        assert values.min() >= walk.floor - 1e-9
        assert values.max() <= walk.ceiling + 1e-9

    def test_a_fixed_seed_reproduces_the_path(self):
        assert PriceWalk(seed=1234).one(52) == pytest.approx(PriceWalk(seed=1234).one(52))
        assert not np.allclose(PriceWalk(seed=1).one(52), PriceWalk(seed=2).one(52))

    def test_shape(self):
        assert PriceWalk(seed=1234).simulate(52, 100).shape == (100, 52)

    def test_boundary_time_detects_clipping(self):
        """A tiny step size should almost never reach a bound; a huge one should."""
        calm = PriceWalk(weekly_sd=20, seed=1234).boundary_time(52, 200)
        wild = PriceWalk(weekly_sd=5_000, seed=1234).boundary_time(52, 200)
        assert calm < 0.01 < wild

    @pytest.mark.parametrize("kwargs", [
        {"floor": 16_000, "ceiling": 9_000}, {"start": 20_000},
        {"start": 1_000}, {"weekly_sd": 0}, {"weekly_sd": -5}])
    def test_impossible_settings_rejected(self, kwargs):
        with pytest.raises(ValueError):
            PriceWalk(**kwargs)

    def test_revenue_converts_tonnes_to_kilograms(self, stock):
        run = stock.run(HarvestPolicy(0.1), weeks=3)
        assert run.revenue([10_000.0] * 3) == pytest.approx(
            run.catch * KG_PER_TONNE * 10_000.0)

    def test_revenue_needs_one_price_per_week(self, stock):
        with pytest.raises(ValueError):
            stock.run(HarvestPolicy(0.1), weeks=5).revenue([12_000.0, 12_000.0])


class TestRiskProfile:
    """Tasks 4-5: the CV rule and the downside measures."""

    def test_describe_reports_five_statistics(self):
        summary = RiskProfile.describe([100.0, 110.0, 90.0, 105.0])
        assert set(summary) == {"mean", "median", "variance", "stdev", "cv"}
        assert summary["cv"] == pytest.approx(summary["stdev"] / summary["mean"])

    def test_describe_needs_two_points(self):
        with pytest.raises(ValueError):
            RiskProfile.describe([100.0])

    @pytest.mark.parametrize("cv,expected", [
        (0.05, "low"), (0.119, "low"), (0.12, "moderate"),
        (0.279, "moderate"), (0.28, "high"), (1.0, "high")])
    def test_cv_bands(self, cv, expected):
        assert RiskProfile().band(cv) == expected

    def test_cv_survives_a_change_of_units_but_variance_does_not(self):
        """Exactly why the rule is written on CV, not variance."""
        ugx = np.array([1e9, 1.2e9, 0.9e9, 1.1e9])
        thousands = ugx / 1_000
        assert (RiskProfile.describe(ugx)["cv"]
                == pytest.approx(RiskProfile.describe(thousands)["cv"]))
        assert (RiskProfile.describe(ugx)["variance"]
                == pytest.approx(RiskProfile.describe(thousands)["variance"] * 1e6))

    @pytest.mark.parametrize("kwargs", [{"low_above": 0}, {"low_above": -0.1},
                                        {"low_above": 0.4, "high_above": 0.2}])
    def test_impossible_thresholds_rejected(self, kwargs):
        with pytest.raises(ValueError):
            RiskProfile(**kwargs)

    def test_var_is_the_right_quantile(self):
        draws = np.arange(1, 1001, dtype=float)
        assert RiskProfile.value_at_risk(draws, 0.10) == pytest.approx(
            np.percentile(draws, 10))

    @pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, 2.0])
    def test_var_rejects_a_bad_alpha(self, alpha):
        with pytest.raises(ValueError):
            RiskProfile.value_at_risk([1.0, 2.0, 3.0], alpha)

    def test_semi_deviation_only_counts_the_downside(self):
        """A series with a long upper tail has sd far above its semi-deviation."""
        skewed = np.array([100.0, 100.0, 100.0, 100.0, 500.0])
        assert RiskProfile.semi_deviation(skewed) < float(np.std(skewed))

    def test_semi_deviation_of_a_flat_series_is_zero(self):
        assert RiskProfile.semi_deviation([100.0] * 5) == pytest.approx(0.0)

    def test_tail_reports_everything_consistently(self):
        draws = np.random.default_rng(1234).normal(1_000, 100, size=10_000)
        tail = RiskProfile().tail(draws, alpha=0.05)
        assert tail["var"] < tail["mean"]
        assert tail["shortfall"] == pytest.approx(tail["mean"] - tail["var"])
        assert tail["semi_deviation"] > 0


class TestFisheryCase:
    """Task 6 and the extension."""

    def test_the_case_composes_three_objects(self, case):
        assert isinstance(case.stock, FishStock)
        assert isinstance(case.price, PriceWalk)
        assert isinstance(case.risk, RiskProfile)

    def test_a_year_is_internally_consistent(self, case):
        outcome = case.year(HarvestPolicy(0.10), weeks=52)
        assert outcome["revenue"].size == 52 and outcome["prices"].size == 52
        assert outcome["band"] in {"low", "moderate", "high"}

    def test_monte_carlo_shape_and_reproducibility(self, case):
        first = case.annual_revenue_paths(HarvestPolicy(0.1), paths=200, seed=1234)
        second = case.annual_revenue_paths(HarvestPolicy(0.1), paths=200, seed=1234)
        assert first.shape == (200,)
        assert first == pytest.approx(second)

    def test_sweep_reproduces_the_msy_result(self, case):
        table = case.rate_sweep([0.05, 0.10, 0.20, 0.30], paths=200)
        assert table["landed_t"].idxmax() == 0.20
        assert table["sustainable_t_per_week"].idxmax() == 0.20
        assert table.loc[0.30, "landed_t"] < table.loc[0.20, "landed_t"]
        assert table.loc[0.30, "closing_stock_t"] < table.loc[0.20, "closing_stock_t"]

    def test_sweep_closing_stock_tracks_the_equilibrium(self, case):
        table = case.rate_sweep([0.10, 0.20], paths=50)
        assert table["closing_stock_t"].to_numpy() == pytest.approx(
            table["equilibrium_t"].to_numpy(), rel=1e-3)

    def test_sweep_needs_a_rate(self, case):
        with pytest.raises(ValueError):
            case.rate_sweep([])

    def test_closure_helps_only_when_overfished(self, case):
        """The headline extension result, in one assertion each way."""
        light = HarvestPolicy(0.10)
        light_closed = HarvestPolicy.with_closed_season(0.10, 0, 8)
        heavy = HarvestPolicy(0.35)
        heavy_closed = HarvestPolicy.with_closed_season(0.35, 0, 8)
        assert case.multi_year(light_closed, 5) < case.multi_year(light, 5)
        assert case.multi_year(heavy_closed, 5) > case.multi_year(heavy, 5)

    def test_multi_year_needs_at_least_one_year(self, case):
        with pytest.raises(ValueError):
            case.multi_year(HarvestPolicy(0.1), years=0)
