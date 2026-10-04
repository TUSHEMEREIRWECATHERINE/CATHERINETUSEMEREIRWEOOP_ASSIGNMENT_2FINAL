"""Tests for Mini-Project 2."""
import numpy as np
import pandas as pd
import pytest

from src.microgrid import (
    BoundedLeastSquares, ClipToZero, DemandSource, EnergyTariff, HybridMicroGrid,
    MicroGrid, RepairStrategy, perturbation_study)


@pytest.fixture
def grid() -> MicroGrid:
    return MicroGrid()


@pytest.fixture
def thirty_days(tmp_path):
    source = DemandSource()
    path = tmp_path / "demand.csv"
    source.write_csv(path, days=30, seed=1234)
    return source, source.to_matrix(source.read_csv(path))


class TestConditioning:
    """Task 1: determinant, condition number and what they mean."""

    def test_default_matrix_is_the_brief(self, grid):
        assert grid.A.tolist() == [[3.0, 2.0], [4.0, 1.0]]
        assert grid.sources == ("solar", "battery")
        assert len(grid) == grid.size == 2

    def test_determinant_by_hand(self, grid):
        assert grid.det == pytest.approx(3 * 1 - 2 * 4)
        assert grid.det == pytest.approx(-5.0)

    def test_condition_number_is_small(self, grid):
        assert grid.cond == pytest.approx(5.828427, abs=1e-5)
        assert grid.well_posed
        assert grid.rank == 2

    def test_conditioning_series_reports_everything(self, grid):
        report = grid.conditioning()
        assert report["full_rank"] and report["well_posed"]

    def test_a_singular_system_is_refused(self):
        singular = MicroGrid([[1.0, 2.0], [2.0, 4.0]])
        assert singular.det == pytest.approx(0.0)
        assert not singular.well_posed
        with pytest.raises(ValueError):
            singular.solve_day(10, 20)

    @pytest.mark.parametrize("matrix", [
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[1.0, np.nan], [2.0, 3.0]], [1.0, 2.0],
    ])
    def test_bad_matrices_are_rejected(self, matrix):
        with pytest.raises(ValueError):
            MicroGrid(matrix)

    def test_name_counts_must_match(self):
        with pytest.raises(ValueError):
            MicroGrid([[3.0, 2.0], [4.0, 1.0]], sources=("solar",))


class TestSolvers:
    """Tasks 2-3: the two solving routes must agree."""

    def test_single_day_matches_the_algebra(self, grid):
        # x = (2*D2 - D1)/5, y = (4*D1 - 3*D2)/5
        assert grid.solve_day(100, 120) == pytest.approx([28.0, 8.0])
        assert grid.solve_day(90, 110) == pytest.approx([26.0, 6.0])

    def test_wrong_argument_count_is_caught(self, grid):
        with pytest.raises(ValueError):
            grid.solve_day(100)

    def test_the_solution_satisfies_the_system(self, grid, thirty_days):
        _, demands = thirty_days
        assert grid.A @ grid.solve_batch(demands) == pytest.approx(demands)

    def test_batch_and_loop_agree(self, grid, thirty_days):
        _, demands = thirty_days
        assert grid.solve_batch(demands) == pytest.approx(grid.solve_each(demands))

    @pytest.mark.parametrize("demands", [
        [[100.0], [120.0], [90.0]], [[-10.0], [120.0]], [[np.inf], [120.0]],
    ])
    def test_bad_demands_are_rejected(self, grid, demands):
        with pytest.raises(ValueError):
            grid.solve_batch(demands)


class TestRepairStrategies:
    """Task 4: infeasible days, and the strategy that handles them."""

    def test_infeasibility_happens_when_d2_exceeds_four_thirds_d1(self, grid):
        raw = grid.solve_batch([[100.0], [140.0]])
        assert raw[1, 0] < 0                       # the battery goes negative
        assert grid.infeasible_days(raw).tolist() == [True]
        assert grid.infeasible_days(grid.solve_batch([[100.0], [120.0]])).tolist() \
            == [False]

    @pytest.mark.parametrize("strategy", [ClipToZero(), BoundedLeastSquares()])
    def test_every_strategy_returns_a_feasible_dispatch(self, grid, thirty_days,
                                                        strategy):
        _, demands = thirty_days
        result = grid.dispatch(demands, repair=strategy)
        assert (result.allocation() >= -1e-9).all()
        assert len(result) == 30

    def test_bounded_least_squares_serves_more_demand_than_clipping(self, grid,
                                                                    thirty_days):
        """The reason the default strategy is the least-squares one."""
        _, demands = thirty_days
        comparison = grid.compare_repairs(demands,
                                          [ClipToZero(), BoundedLeastSquares()])
        assert (comparison.loc["bounded-lsq", "unserved kWh"]
                < comparison.loc["clip", "unserved kWh"])
        # and both repair exactly the same days
        assert (comparison["days repaired"].nunique() == 1)

    def test_a_feasible_day_is_left_untouched(self, grid):
        result = grid.dispatch([[100.0], [120.0]])
        assert result.allocation().ravel() == pytest.approx([28.0, 8.0])
        assert result.unserved_kwh == pytest.approx(0.0, abs=1e-9)
        assert result.n_repaired == 0

    def test_shortfall_is_the_residual_norm(self, grid):
        demands = np.array([[100.0], [140.0]])
        result = grid.dispatch(demands, repair=ClipToZero())
        expected = float(np.linalg.norm(grid.A @ result.allocation() - demands))
        assert result.unserved_kwh == pytest.approx(expected)

    def test_comparing_no_strategies_is_an_error(self, grid, thirty_days):
        _, demands = thirty_days
        with pytest.raises(ValueError):
            grid.compare_repairs(demands, [])

    def test_strategies_share_a_base_class(self):
        assert issubclass(ClipToZero, RepairStrategy)
        assert issubclass(BoundedLeastSquares, RepairStrategy)
        with pytest.raises(TypeError):
            RepairStrategy()                       # abstract


class TestVolatility:
    """Task 5: which source swings more."""

    def test_the_battery_is_the_volatile_one(self, grid, thirty_days):
        _, demands = thirty_days
        volatility = grid.volatility(grid.dispatch(demands).allocation())
        assert volatility.loc["battery", "cv"] > volatility.loc["solar", "cv"]

    def test_one_day_is_not_enough(self, grid):
        with pytest.raises(ValueError):
            grid.volatility(np.array([[28.0], [8.0]]))


class TestTariff:
    """Task 6: costing the dispatch."""

    def test_costs_match_a_hand_calculation(self):
        tariff = EnergyTariff()
        table = pd.DataFrame({"solar": [10.0, 20.0], "battery": [1.0, 2.0]})
        # 10*150 + 1*450 = 1,950;  20*150 + 2*450 = 3,900
        assert tariff.per_day(table).tolist() == pytest.approx([1950.0, 3900.0])
        assert tariff.over_period(table) == pytest.approx(5850.0)
        assert tariff.run_rate(table, 30) == pytest.approx(2925.0 * 30)

    def test_split_reports_energy_and_cost_shares(self):
        split = EnergyTariff().split(pd.DataFrame({"solar": [100.0],
                                                   "battery": [50.0]}))
        assert split["energy_share"].sum() == pytest.approx(1.0)
        assert split["cost_share"].sum() == pytest.approx(1.0)
        # the battery takes a bigger share of cost than of energy
        assert (split.loc["battery", "cost_share"]
                > split.loc["battery", "energy_share"])

    def test_negative_rates_and_unknown_columns_are_rejected(self):
        with pytest.raises(ValueError):
            EnergyTariff(solar=-1)
        with pytest.raises(ValueError):
            EnergyTariff().per_day(pd.DataFrame({"wind": [1.0]}))
        with pytest.raises(ValueError):
            EnergyTariff().run_rate(pd.DataFrame({"solar": [1.0]}), days=0)

    def test_tariff_is_frozen(self):
        with pytest.raises(Exception):
            EnergyTariff().solar = 1.0


class TestDemandSource:
    """The two input modes."""

    def test_generated_file_is_reproducible(self, tmp_path):
        source = DemandSource()
        first = source.write_csv(tmp_path / "a.csv", 30, seed=1234)
        second = source.write_csv(tmp_path / "b.csv", 30, seed=1234)
        assert first.equals(second)
        assert (first[["D1", "D2"]] > 0).all().all()

    def test_generated_file_has_a_weekly_rhythm(self, tmp_path):
        frame = DemandSource().write_csv(tmp_path / "c.csv", 70, seed=1234,
                                         weekly=0.3, noise=0.0)
        means = frame.groupby((frame["day"] - 1) % 7)["D1"].mean()
        assert means.max() / means.min() > 1.3

    def test_broken_files_are_rejected(self, tmp_path):
        source = DemandSource()
        with pytest.raises(FileNotFoundError):
            source.read_csv(tmp_path / "absent.csv")
        (tmp_path / "cols.csv").write_text("day,X\n1,100\n", encoding="utf-8")
        with pytest.raises(ValueError):
            source.read_csv(tmp_path / "cols.csv")
        (tmp_path / "neg.csv").write_text("day,D1,D2\n1,100,-5\n", encoding="utf-8")
        with pytest.raises(ValueError):
            source.read_csv(tmp_path / "neg.csv")

    @pytest.mark.parametrize("text", ["", "   ", "abc", "-5", "nan", "inf", None])
    def test_typed_rubbish_is_rejected(self, text):
        with pytest.raises(ValueError):
            DemandSource.read_value(text)

    @pytest.mark.parametrize("text,expected", [
        ("150", 150.0), (" 99.5 ", 99.5), ("1,250.5", 1250.5), ("0", 0.0)])
    def test_typed_numbers_are_accepted(self, text, expected):
        assert DemandSource.read_value(text) == pytest.approx(expected)

    def test_prompt_reprompts_then_accepts(self):
        answers = iter(["", "nope", "-1", "104.2"])
        complaints = []
        value = DemandSource.prompt("D1", reader=lambda _: next(answers),
                                    writer=complaints.append, tries=4)
        assert value == pytest.approx(104.2)
        assert len(complaints) == 3

    def test_prompt_eventually_gives_up(self):
        with pytest.raises(ValueError, match="gave up"):
            DemandSource.prompt("D1", reader=lambda _: "no",
                                writer=lambda _: None, tries=2)

    def test_prompt_all_collects_one_per_load(self):
        answers = iter(["100", "120"])
        values = DemandSource().prompt_all(reader=lambda _: next(answers),
                                           writer=lambda _: None)
        assert values == pytest.approx([100.0, 120.0])


class TestHybridExtension:
    """The 3x3 system and the sensitivity study."""

    def test_hybrid_inherits_everything_and_solves(self):
        hybrid = HybridMicroGrid()
        assert isinstance(hybrid, MicroGrid)
        assert hybrid.size == 3
        assert hybrid.det == pytest.approx(-25.0)
        allocation = hybrid.solve_day(120, 140, 200)
        assert hybrid.A @ allocation == pytest.approx([120.0, 140.0, 200.0])

    def test_hybrid_reuses_the_parent_dispatch(self):
        demands = np.array([[120.0, 118.0], [140.0, 139.0], [200.0, 205.0]])
        result = HybridMicroGrid().dispatch(demands)
        assert list(result.table.columns)[:3] == ["solar", "battery", "diesel"]

    def test_hybrid_rejects_the_wrong_size(self):
        with pytest.raises(ValueError):
            HybridMicroGrid([[3.0, 2.0], [4.0, 1.0]], sources=("a", "b"),
                            loads=("D1", "D2"))

    def test_a_dependent_third_load_destroys_uniqueness(self):
        broken = HybridMicroGrid.linearly_dependent()
        assert broken.det == pytest.approx(0.0)
        assert broken.rank == 2
        assert not broken.well_posed
        with pytest.raises(ValueError):
            broken.solve_day(100, 120, 220)

    def test_amplification_stays_under_the_condition_number(self, grid):
        study = perturbation_study(grid, [100.0, 120.0], draws=1000, seed=1234)
        assert study["solutions"].shape == (2, 1000)
        assert study["worst"] <= study["cond"] + 1e-9
        assert study["typical"] > 0

    def test_study_is_reproducible(self, grid):
        first = perturbation_study(grid, [100.0, 120.0], draws=200, seed=1234)
        second = perturbation_study(grid, [100.0, 120.0], draws=200, seed=1234)
        assert first["solutions"] == pytest.approx(second["solutions"])

    @pytest.mark.parametrize("kwargs", [{"spread": 0.0}, {"spread": 1.0},
                                        {"draws": 0}])
    def test_study_rejects_bad_settings(self, grid, kwargs):
        with pytest.raises(ValueError):
            perturbation_study(grid, [100.0, 120.0], **kwargs)
