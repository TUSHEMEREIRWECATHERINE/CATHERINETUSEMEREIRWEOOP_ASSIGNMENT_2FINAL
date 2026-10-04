# --- markdown
# # Mini-Project 1: UBOS District Population Forecaster
#
# **What the planning unit asked.** Give us five-year population forecasts so
# we can budget primary-school classrooms. How many extra classrooms does each
# district need by 2029?
#
# **The data.** Population estimates in **thousands**, 2015-2024. Kampala,
# Wakiso and Gulu are the figures supplied in the brief and are *illustrative*,
# not official UBOS releases. **Masaka** and **Lira** are the two additional
# districts required by task 1. Both are regional service centres rather than
# dormitory suburbs of Kampala, so they act as a counterweight to the central
# growth belt instead of repeating it.
#
# **Where the code lives.** `src/population.py` holds the domain classes.
# `src/forecasting.py` holds the abstract `Forecaster`, assembled out of the
# three mixins in `src/mixins.py`, and `src/metrics.py` holds the error
# measures. This notebook only runs the analysis and interprets it.

# --- code
import sys
from pathlib import Path

ROOT = Path.cwd() if (Path.cwd() / "src").exists() else Path.cwd().parent
sys.path.insert(0, str(ROOT))

import statistics

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.charts import SERIES_COLOURS, store, use_house_style
from src.forecasting import EnsembleForecaster, Forecaster
from src.metrics import MetricSuite
from src.mixins import grid_search
from src.population import (
    PLANNING_HORIZON, TRAIN_UNTIL, CompoundGrowth, DistrictPopulation,
    GoldenRatioProjection, PolyfitTrend, SchoolCapacityModel, ValidationBench,
    fibonacci, load_series, projection, residual_bootstrap, spread_check)

use_house_style()
SEED = 1234
rng = np.random.default_rng(SEED)
pd.set_option("display.float_format", "{:,.3f}".format)

series = load_series()
print("loaded:", ", ".join(s.district for s in series))

# --- markdown
# ## Task 1: the `DistrictPopulation` class
#
# The class stores years and populations as NumPy arrays and refuses input it
# could not sensibly analyse. Beyond the `__repr__` and `__len__` the brief
# asks for, it behaves like a read-only mapping: `series[2020]` looks a year
# up, iterating yields `(year, population)` pairs, and `in` tests coverage.
# That means the rest of the analysis can treat a district as data rather than
# reaching into its attributes.

# --- code
for s in series:
    print(repr(s))

kampala = series[0]
print("\nkampala[2020] =", kampala[2020])
print("first three pairs:", list(kampala)[:3])
print("2018 covered:", 2018 in kampala, "| 2035 covered:", 2035 in kampala)

# --- markdown
# Validation matters because each rejection below would otherwise surface much
# later as a confidently wrong forecast.

# --- code
rejections = {
    "lengths differ": dict(years=[2020, 2021], thousands=[1, 2, 3]),
    "negative population": dict(years=[2020, 2021], thousands=[100, -5]),
    "empty series": dict(years=[], thousands=[]),
    "years out of order": dict(years=[2021, 2020], thousands=[100, 110]),
    "non-finite value": dict(years=[2020, 2021], thousands=[100, np.inf]),
}
for description, kwargs in rejections.items():
    try:
        DistrictPopulation("Invalid", **kwargs)
    except ValueError as error:
        print(f"{description:<22} -> {error}")

# --- markdown
# ## Task 2: the same statistics from two libraries
#
# `statistics.variance` and `np.var` disagree because they estimate different
# things.
#
# * `statistics.variance` is the **sample** variance and divides by `n - 1`.
# * `np.var` divides by `n - ddof`, and `ddof` defaults to **0**, so it returns
#   the **population** variance unless told otherwise.
#
# `ddof` is the "delta degrees of freedom" subtracted from `n` in the
# denominator. One degree of freedom has already been spent estimating the
# mean, so `ddof=1` removes it and makes the estimator unbiased when the ten
# observations are treated as a *sample* from a process rather than as the
# entire population of interest. With n = 10 the NumPy default therefore
# reports a variance 10% smaller.

# --- code
rows = []
for s in series:
    stdlib, pop_var, sample_var = s.summary_stdlib(), s.summary_numpy(0), s.summary_numpy(1)
    rows.append({
        "district": s.district,
        "mean": stdlib["mean"],
        "median": stdlib["median"],
        "var (statistics)": stdlib["variance"],
        "var (np, ddof=0)": pop_var["variance"],
        "var (np, ddof=1)": sample_var["variance"],
        "sd (statistics)": stdlib["stdev"],
    })
pd.DataFrame(rows).set_index("district")

# --- code
n = len(kampala)
sample = kampala.summary_stdlib()["variance"]
population = kampala.summary_numpy(ddof=0)["variance"]
assert np.isclose(population, sample * (n - 1) / n)
assert np.isclose(sample, kampala.summary_numpy(ddof=1)["variance"])
print(f"sample variance     (/ n-1) : {sample:>12,.2f}")
print(f"population variance (/ n)   : {population:>12,.2f}")
print(f"ratio                       : {population / sample:.4f}   (= 9/10)")

# --- markdown
# ## Task 3: year-on-year growth and the CAGR
#
# Year-on-year growth says how fast a district grew in a particular year. The
# CAGR collapses the whole decade into the single constant rate that would
# have carried it from its 2015 value to its 2024 value.

# --- code
changes = pd.DataFrame({s.district: np.round(100 * s.annual_changes(), 2) for s in series},
                       index=[f"{y}->{y + 1}" for y in range(2015, 2024)])
changes.loc["CAGR"] = [round(100 * s.cagr(), 2) for s in series]
changes

# --- code
rates = {s.district: s.cagr() for s in series}
for name in sorted(rates, key=rates.__getitem__, reverse=True):
    print(f"{name:<10} {rates[name]:>7.2%} a year")
print(f"\nfastest in relative terms: {max(rates, key=rates.__getitem__)}")
print(f"slowest in relative terms: {min(rates, key=rates.__getitem__)}")
print("\nRelative and absolute growth point to different districts: Kampala adds")
print("the most people (600k over the decade) but Wakiso compounds fastest, and")
print("it is the compounding that drives the classroom bill.")

# --- markdown
# ## Task 4: three models, three assumptions
#
# All three subclass the abstract `Forecaster`, which fills in `fit()` and
# `predict(horizon)` once and leaves subclasses two hooks, `_estimate` and
# `_extrapolate`. What makes this build distinctive is where the rest of the
# behaviour comes from: `Forecaster` inherits it from three independent mixins
# rather than implementing it.
#
# | Mixin | What it contributes |
# |---|---|
# | `SeriesValidationMixin` | coercion and the rejection rules |
# | `ScoringMixin` | `.score(actual)` returning a `MetricSuite` |
# | `WalkForwardMixin` | rolling-origin backtesting and `grid_search` |
#
# The payoff is that a class can take only what it needs. `DistrictPopulation`
# inherits `SeriesValidationMixin` alone -- it gets the input rules without
# having to pretend to be a forecasting model.

# --- code
print("Forecaster MRO:")
for klass in Forecaster.__mro__:
    print("   ", klass.__name__)

print("\nDistrictPopulation takes only the validation mixin:")
print("   ", [k.__name__ for k in DistrictPopulation.__mro__])

# --- code
demo = CompoundGrowth().fit(kampala.thousands, kampala.years.astype(float))
print(repr(demo), "| implied growth:", f"{demo.rate_:.2%}")
print("chained:", np.round(CompoundGrowth().fit(kampala.thousands).predict(3), 1))

sequence = fibonacci(16)
ratios = sequence[1:] / sequence[:-1]
print("\nFibonacci:", sequence.astype(int).tolist())
print("ratios   :", np.round(ratios, 4).tolist())
print(f"phi = {(1 + np.sqrt(5)) / 2:.6f}, i.e. {(np.sqrt(5) - 1) / 2:.1%} growth a year")

# --- markdown
# This build starts the Fibonacci ratios at term 2, so the first forecast steps
# use the ratios while they are still unsettled -- 1.5, 1.667, 1.6, 1.625 --
# before they converge. It makes no material difference to the verdict, but it
# shows the convergence happening inside the forecast rather than hiding it.

# --- markdown
# ## Task 5: validate before forecasting
#
# Models are trained on **2015-2021** and scored on the held-out
# **2022-2024**. Three measures are reported together because any one of them
# can flatter a model: MAE is in thousands of people, RMSE punishes large
# misses, and MAPE is unit-free so districts of different sizes compare.

# --- code
bench = ValidationBench(train_until=TRAIN_UNTIL)
print(bench)

train, test = kampala.partition(TRAIN_UNTIL)
print(f"\ntrain {train.years[0]}-{train.years[-1]} (n={len(train)})")
print(f"test  {test.years[0]}-{test.years[-1]} (n={len(test)})")

# --- code
for s in series:
    print(f"\n=== {s.district} " + "=" * (46 - len(s.district)))
    print(bench.table(s).to_string())
    print(f"--> selected: {bench.winner(s)}")

# --- code
overview = pd.DataFrame([
    {"district": s.district,
     **{name: suite.mape for name, suite in bench.scores(s).items()},
     "selected": bench.winner(s)}
    for s in series]).set_index("district")
overview.columns = [c if c == "selected" else f"MAPE% {c}" for c in overview.columns]
overview

# --- markdown
# **Result.** Compound growth wins on all five districts, with MAPE between
# 0.35% and 2.03%. The straight line is consistently second (0.60-4.01%): it is
# not ridiculous, but it cannot bend, so it falls further behind each year.
# The golden-ratio model is wrong by 132-150%.

# --- markdown
# ## Models add together
#
# `Forecaster.__add__` returns an `EnsembleForecaster` that fits every member
# and averages their forecasts. Averaging is worth trying whenever two models
# err in opposite directions and there is no clear reason to trust one over the
# other. Whether it actually helps here is an empirical question, so it is
# worth answering rather than assuming.

# --- code
combined = PolyfitTrend() + CompoundGrowth()
combined.fit(train.thousands, train.years.astype(float))
print(repr(combined), "| members:", len(combined))

contest = {"polyfit-trend": PolyfitTrend(), "compound-growth": CompoundGrowth()}
for model in contest.values():
    model.fit(train.thousands, train.years.astype(float))
comparison = {name: model.score(test.thousands).mape for name, model in contest.items()}
comparison["ensemble (average)"] = combined.score(test.thousands).mape
pd.Series(comparison, name="MAPE %").to_frame().round(3)

# --- markdown
# **The ensemble does not help here, and it is worth being clear why.**
# Averaging pays off when members miss in opposite directions, so their errors
# partly cancel. On this data both models under-forecast -- the straight line
# badly, compound growth slightly -- so averaging simply drags the better model
# towards the worse one. The ensemble lands at 0.81% MAPE, between its two
# members rather than below either. The right conclusion is to keep the single
# best model and to treat the `__add__` operator as a tool that earns its place
# only when the errors are genuinely uncorrelated.

# --- markdown
# ## Task 6: forecast 2025-2029, and compare the variances
#
# The winning model is refitted on the **full** 2015-2024 series before the
# final forecast, so nothing observed is thrown away.

# --- code
chosen = {s.district: bench.refit_winner(s) for s in series}
paths = {s.district: projection(s, chosen[s.district], PLANNING_HORIZON) for s in series}
pd.DataFrame(paths).round(1)

# --- code
pd.DataFrame([{"district": s.district, **spread_check(s, paths[s.district].to_numpy())}
              for s in series]).set_index("district")

# --- markdown
# **Reading the variance difference.** The forecast series is less variable
# than the observed one everywhere (ratios of 0.42-0.67). That is a fact about
# the estimator, not a prediction that the future will be calmer. The observed
# series contains the trend *plus* a decade of year-to-year scatter; a fitted
# compound-growth curve is a smooth deterministic path with the scatter removed
# by construction. Quoting the forecast variance as a measure of uncertainty
# would understate the risk considerably. Genuine uncertainty has to be rebuilt
# separately, which the bootstrap below attempts.

# --- markdown
# ## Task 7: observed, fitted and forecast

# --- code
fig, axes = plt.subplots(2, 3, figsize=(13.5, 7), sharex=True)
future = np.arange(2025, 2030)

for ax, s in zip(axes.flat, series):
    model, path = chosen[s.district], paths[s.district]
    ax.plot(s.years, s.thousands, "o-", color=SERIES_COLOURS[0], label="observed")
    ax.plot(s.years, model.fitted(), "--", color=SERIES_COLOURS[1], lw=1.4,
            label="fitted")
    ax.plot(future, path.to_numpy(), "s--", color=SERIES_COLOURS[2], label="forecast")
    ax.axvspan(2021.5, 2024.5, color="0.5", alpha=0.10)
    ax.axvline(2021.5, color="0.35", lw=1, ls="-.")
    ax.set_title(f"{s.district} ({s.cagr():.2%} a year)")
    ax.set_ylabel("population (thousands)")

axes.flat[-1].axis("off")
axes.flat[0].legend(loc="upper left", fontsize=8)
axes[0, 2].tick_params(labelbottom=True)
for ax in (axes[0, 2], *axes[1, :2]):
    ax.set_xlabel("year")
fig.suptitle("Observed, fitted and forecast population by district, 2015-2029\n"
             "(shaded band = the held-out 2022-2024 test window)", fontsize=12)
fig.tight_layout()
print("saved:", store(fig, "p1_population_paths.png").name)
plt.show()

# --- markdown
# ## Task 8: classrooms needed by 2029
#
# The brief's assumptions: 18% of the population is of primary-school age and a
# classroom seats 53 pupils. `SchoolCapacityModel` is a frozen dataclass, so a
# scenario cannot be edited mid-analysis -- a different assumption means a new,
# named object.

# --- code
planner = SchoolCapacityModel()
print(planner)

plan = pd.DataFrame([{
    "district": s.district,
    "2024 (k)": s.thousands[-1],
    "2029 (k)": round(float(paths[s.district].iloc[-1]), 1),
    "growth (k)": round(float(paths[s.district].iloc[-1]) - s.thousands[-1], 1),
    "new pupils": round(planner.enrolment(
        float(paths[s.district].iloc[-1]) - s.thousands[-1])),
    "extra classrooms": planner.rooms_to_add(
        s.thousands[-1], float(paths[s.district].iloc[-1])),
} for s in series]).set_index("district").sort_values("extra classrooms",
                                                      ascending=False)
plan

# --- code
rate = kampala.cagr()
by_hand_2029 = kampala.thousands[-1] * (1 + rate) ** 5
by_hand_rooms = int(np.ceil((by_hand_2029 - kampala.thousands[-1]) * 1_000 * 0.18 / 53))
assert by_hand_rooms == plan.loc["Kampala", "extra classrooms"]
print(f"by hand: 1800 x {1 + rate:.5f}^5 = {by_hand_2029:,.1f}k")
print(f"         ({by_hand_2029:,.1f} - 1800) x 1000 x 0.18 / 53 = {by_hand_rooms:,}")
print(f"classes: {plan.loc['Kampala', 'extra classrooms']:,}   [match]")
print(f"\ntotal across the five districts: {plan['extra classrooms'].sum():,} classrooms")

# --- markdown
# ## Extension 1: prediction intervals from resampled residuals
#
# A point forecast carries no uncertainty of its own. The residual bootstrap
# reconstructs some: draw the model's training residuals with replacement
# (2,000 resamples, above the 1,000 the brief requires), add them to the
# forecast path, and read the 2.5th and 97.5th percentiles.

# --- code
bands = {s.district: residual_bootstrap(chosen[s.district], PLANNING_HORIZON,
                                        draws=2000, coverage=0.95, seed=SEED)
         for s in series}

pd.DataFrame([{
    "district": s.district,
    "2029 point": round(float(bands[s.district]["point"].iloc[-1]), 1),
    "2029 lower": round(float(bands[s.district]["lower"].iloc[-1]), 1),
    "2029 upper": round(float(bands[s.district]["upper"].iloc[-1]), 1),
    "rooms (point)": planner.rooms_to_add(
        s.thousands[-1], float(bands[s.district]["point"].iloc[-1])),
    "rooms (upper)": planner.rooms_to_add(
        s.thousands[-1], float(bands[s.district]["upper"].iloc[-1])),
} for s in series]).set_index("district")

# --- code
fig, axes = plt.subplots(2, 3, figsize=(13.5, 7), sharex=True)
for ax, s in zip(axes.flat, series):
    band = bands[s.district]
    ax.plot(s.years, s.thousands, "o-", color=SERIES_COLOURS[0], label="observed")
    ax.plot(future, band["point"], "s--", color=SERIES_COLOURS[2], label="forecast")
    ax.fill_between(future, band["lower"], band["upper"], color=SERIES_COLOURS[2],
                    alpha=0.18, label="95% interval")
    ax.axvline(2021.5, color="0.35", lw=1, ls="-.")
    ax.set_title(s.district)
    ax.set_ylabel("population (thousands)")

axes.flat[-1].axis("off")
axes.flat[0].legend(loc="upper left", fontsize=8)
axes[0, 2].tick_params(labelbottom=True)
for ax in (axes[0, 2], *axes[1, :2]):
    ax.set_xlabel("year")
fig.suptitle("Forecasts with 95% residual-bootstrap prediction intervals", fontsize=12)
fig.tight_layout()
print("saved:", store(fig, "p1_prediction_intervals.png").name)
plt.show()

# --- markdown
# The bands are narrow: the upper bound adds only about 1% to the classroom
# total. That is a statement about the **method**, not reassurance about the
# future. A compound-growth curve is anchored on the first and last
# observations, so it passes through both exactly and its training residuals
# are small by construction. Resampling them measures how well the curve fits
# points it was pinned to -- it cannot measure the risk that constant growth is
# the wrong shape altogether, which is the larger danger over five years.

# --- markdown
# ## Extension 2: is the Fibonacci-ratio model ever defensible?
#
# Successive Fibonacci ratios converge on the golden ratio phi = 1.618..., so
# after a few terms the model simply multiplies by 1.618 each year: a standing
# assumption of **61.8% annual population growth**.

# --- code
assessment = pd.DataFrame([{
    "district": s.district,
    "MAPE% golden-ratio": bench.scores(s)["golden-ratio"].mape,
    "MAPE% compound": bench.scores(s)["compound-growth"].mape,
    "actual CAGR %": 100 * s.cagr(),
} for s in series]).set_index("district")
assessment

# --- code
implied = (1 + np.sqrt(5)) / 2 - 1
fastest = max(rates.values())
print(f"growth implied by phi      : {implied:>7.1%} a year")
print(f"fastest district observed  : {fastest:>7.1%} a year")
print(f"overstatement              : {implied / fastest:>7.0f}x")
print(f"\ndoubling time at 61.8%     : {np.log(2) / np.log(1 + implied):>5.1f} years")
print(f"doubling time at Wakiso's  : {np.log(2) / np.log(1 + fastest):>5.1f} years")

# --- markdown
# **Assessment.** The model is defensible only where the quantity really does
# grow at about 61.8% per period *and* each value really is the sum of the
# previous two. Neither holds for a district population: people are not
# produced by adding the last two years together, and no Ugandan district has
# ever sustained 61.8% annual growth. It has two narrow virtues -- it is
# reproducible, and for a few periods it would be a plausible shape for an
# early-stage epidemic or a viral adoption curve before saturation takes hold.
# As a population model it fails three ways: the growth rate is never estimated
# from the data, the series is unbounded, and it ignores the carrying capacity
# imposed by land, water and services. It still earns its place in the
# comparison as the control that proves the validation step is doing real work.

# --- markdown
# ## Findings & Limitations
#
# **Findings.** All five districts grow close to exponentially, so compound
# growth wins the 2022-2024 validation outright, with MAPE from 0.35% (Lira)
# to 2.03% (Gulu). The straight-line trend is a credible second (0.60-4.01%)
# but drifts low because it cannot curve. The golden-ratio model misses by
# 132-150%, since its ratios converge on phi and so assume 61.8% annual growth
# -- about ten times the fastest rate observed. Wakiso compounds fastest at
# 6.47% a year, Masaka slowest at 2.90%. Projected to 2029, the five districts
# need roughly **4,660 extra primary classrooms**: Wakiso 2,088, Kampala 1,545,
# Lira 419, Gulu 412 and Masaka 196. Wakiso needing more rooms than Kampala,
# despite being smaller, is the result that should shape the budget. Averaging
# the two sensible models made things worse (0.81% against 0.55%), because both
# err in the same direction and the average simply dilutes the better one.
#
# **Limitations.** The data are illustrative, so magnitudes are indicative
# only. A compound rate fitted to two endpoints ignores everything between
# them and assumes the next five years resemble the last nine, which excludes
# boundary changes, migration shocks and the fertility decline already visible
# in UBOS projections. The bootstrap intervals capture fit error alone: because
# the curve is pinned to its first and last observations, the residuals cannot
# express how wrong the constant-growth *shape* might be. The 18% primary-age
# share and 53-pupil classroom are national averages applied uniformly, though
# both plausibly differ between Kampala and Gulu. Finally, the figures count
# rooms for *additional* pupils only and say nothing about the backlog already
# standing in 2024.
