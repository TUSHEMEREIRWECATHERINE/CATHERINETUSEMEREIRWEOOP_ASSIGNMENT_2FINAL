# OOP with Python — Advent 2026 Mini-Projects

**Student:** _<your name>_ · **Reg. no.:** _<your registration number>_ · **Programme:** MSCS / MSDS

Five mini-projects in Ugandan settings. Each is modelled with classes, backed by reusable
modules in `src/`, run from its own Jupyter notebook and covered by `pytest`.

| # | Mini-project | Notebook | Module |
|---|---|---|---|
| 1 | UBOS District Population Forecaster | `notebooks/project1_population.ipynb` | `src/population.py` |
| 2 | Solar Micro-Grid Dispatch Planner (Kasese) | `notebooks/project2_microgrid.ipynb` | `src/microgrid.py` |
| 3 | Lake Victoria Fish Stock & Export Risk | `notebooks/project3_fishery.ipynb` | `src/fishery.py` |
| 4 | Rainfall Pattern & Crop Suitability | `notebooks/project4_rainfall.ipynb` | `src/rainfall.py` |
| 5 | Taxi Route Revenue, Pricing & Fleet Planner | `notebooks/project5_taxi.ipynb` | `src/taxi.py` |

## Repository layout

```
├── README.md
├── Makefile                  # make test / make notebooks
├── requirements.txt
├── src/
│   ├── mixins.py             # SeriesValidationMixin, ScoringMixin, WalkForwardMixin
│   ├── forecasting.py        # abstract Forecaster, assembled from the mixins
│   ├── metrics.py            # MetricSuite: MAE / RMSE / MAPE as one named tuple
│   ├── charts.py             # shared chart style and figure paths
│   ├── population.py   ├── microgrid.py   ├── fishery.py
│   ├── rainfall.py     └── taxi.py
├── notebooks/                # one notebook per mini-project, committed with outputs
├── tests/                    # pytest suites, grouped into classes (335 tests)
├── data/                     # the generated demand CSV; optional NASA POWER files
├── figures/                  # PNGs written by the notebooks
└── tools/
    ├── make_notebooks.py     # banner-delimited .py  ->  executed .ipynb
    └── nb_src/               # the notebooks' real source, in plain text
```

### Design decisions worth asking about

* **Capability by mixin, not by base class.** `Forecaster` inherits validation, scoring and
  walk-forward backtesting from three independent mixins and adds only the fit/predict
  lifecycle. The payoff is that a class takes just what it needs: `DistrictPopulation`,
  `Region`, `Route`, `MicroGrid` and `DemandSource` all inherit `SeriesValidationMixin`
  alone, so they get the input rules without pretending to be forecasting models.
* **Mini-project 5 reimplements no backtesting at all.** `WalkForwardMixin` and
  `grid_search` already exist for mini-project 1, so every taxi model backtests and tunes
  itself with no new code. That reuse is the reason the capability lives in a mixin.
* **Two hooks, not six.** Concrete models implement `_estimate` and `_extrapolate`; the
  public `fit`/`predict` pair handles validation, state and chaining once, in the base.
* **Models add together.** `Forecaster.__add__` builds an `EnsembleForecaster` that fits
  all members and averages them, flattening when a third is added. Both P1 and P5 then
  *test* whether the ensemble helps — it does not, in both cases for the same reason, and
  the notebooks say why rather than quietly dropping the result.
* **Repair as a strategy object.** P2's infeasible-day handling is a pluggable
  `RepairStrategy`, so clipping and bounded least-squares are compared on identical data
  instead of one being assumed. The comparison is the finding.
* **Policy separated from biology.** P3's `HarvestPolicy` holds the harvest rate and the
  closed season; `FishStock` holds the logistic dynamics. The closed-season extension is
  therefore a different policy object, not a second simulation routine.
* **Containers behave like containers.** `DistrictPopulation`, `Region` and `Route`
  implement `__getitem__`, `__iter__`, `__contains__`, `__len__` and `__repr__`, so domain
  objects read as data rather than through attribute access.
* **Composition where that is the honest relationship.** `FisheryCase` combines a
  `FishStock`, a `PriceWalk` and a `RiskProfile`; `CropCalendar` combines `Region` and
  `CropRule` objects.
* **Validation everywhere.** Every constructor rejects empty, negative, non-finite or
  mismatched input with a `ValueError`; type hints and docstrings on all public API.

## Setup

```bash
git clone <this-repo-url>
cd advent2026-b
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt    # or: make install
```

Developed on Python 3.13.7 with NumPy 2.4.4, SciPy 1.18.1, pandas 3.0.2 and
Matplotlib 3.11.2. Python 3.10 or newer should work.

## Running

```bash
make test                                      # 335 tests, or: pytest -q
make notebooks                                 # rebuild and execute every notebook
make notebooks-dry                             # assemble without executing

jupyter notebook notebooks/                    # then Kernel ▸ Restart & Run All
python tools/make_notebooks.py project3_fishery
```

The notebooks find `src/` whether Jupyter starts at the repository root or inside
`notebooks/`. Their real source is the banner-delimited scripts in `tools/nb_src/`, which
keeps diffs readable; `make_notebooks.py` regenerates the `.ipynb` files and runs them,
refusing to write one whose cells raised.

**Reproducibility.** All randomness goes through `np.random.default_rng(seed)`, with the
seed fixed at the top of each notebook (`SEED = 1234`) or passed into the object that
needs it. The P2 demand CSV is regenerated from its seed on every run.

## Summary of findings

**P1 — Population.** All five districts grow close to exponentially, so the compound-growth
model wins the 2022–24 validation outright (MAPE 0.35–2.03%). The straight line is second
(0.60–4.01%) but cannot curve; the golden-ratio model misses by 132–150%, since its ratios
converge on φ and so assume 61.8% annual growth. Wakiso compounds fastest at 6.47% a year,
Masaka slowest at 2.90%. By 2029 the five districts need about **4,660 extra primary
classrooms**: Wakiso 2,088, Kampala 1,545, Lira 419, Gulu 412, Masaka 196 — Wakiso needing
more than Kampala despite being smaller is the result that should shape the budget.
Averaging the two sensible models made accuracy **worse** (0.81% against 0.55%): both err
in the same direction, so the average dilutes the better one rather than cancelling error.

**P2 — Micro-grid.** det A = −5 and κ ≈ 5.83, so the system is well posed, and Monte Carlo
perturbation at ±5% produced a worst-case amplification just under κ — though the error
falls almost entirely on the battery rather than being shared. Batching the 30-day solve
beats looping several times over, on per-call overhead rather than arithmetic. Five of 30
days were infeasible (D2 > 4/3·D1). Treating the repair as a pluggable strategy made the
choice measurable: clipping leaves ~60 kWh unserved across the month, bounded least-squares
leaves ~27 kWh — **a 55% improvement on exactly the same days**. The battery is far more
volatile (CV 0.75 vs solar's 0.17) and at 3× the tariff takes about **half the bill on a
quarter of the energy**: ~UGX 8,300 a day, UGX 250,000 a month. Diesel under a cold-chain
constraint keeps the system well conditioned (det = −25, κ ≈ 7.0).

**P3 — Fishery.** The logistic model replaces unbounded Fibonacci growth and reproduces
N\* = K(1−h/r) to six decimals. MSY is rK/4 = **1,000 t/week at h = 0.20**. The current
h = 0.10 is safe (stock settles at 7,500 t) but lands **27% below** the sustainable yield;
h = 0.30 is strictly worse than MSY — 17% less fish *and* half the standing stock. The old
"variance > 50,000" rule is meaningless, not just mis-tuned: revenue variance is in UGX², so
re-expressing the same revenue in thousands changes it by 10⁶ while the risk is identical.
On the dimensionless CV, h = 0.10 scores 0.144 → **moderate**, and the 5% VaR sits ~18%
below expectation. The downside semi-deviation is well under the full standard deviation,
so the distribution is right-skewed and budgeting on total volatility would over-reserve.
The clipped price walk pins ~7% of simulated weeks to a bound — reported rather than hidden.
An eight-week closure costs 9–13% of five-year revenue at h ≤ 0.20 and only pays once
over-exploited (+6.1% at h = 0.30, +37.5% at h = 0.35).

**P4 — Rainfall.** Cosine similarity, correctly implemented and matching SciPy to machine
precision, rates every pair 0.75–0.89 — an artefact of rainfall being non-negative, so all
vectors sit in the positive orthant where cosine cannot go below zero. Pearson, which
centres first, shows seasonal timing is essentially unrelated (−0.17 to 0.21). Season
detection using a height test at the annual mean and a prominence test at **one standard
deviation of that region's own totals** classifies Gulu unimodal and Mbarara bimodal, both
matching documented climate zones; Mbarara holds from 0.25 to 2.0 sd. Kampala is called
unimodal only because the illustrative series omits the June–July dip. All three crops need
roughly 90–100 mm/month at the lower bound despite seasonal requirements differing
fourfold, because smaller-requirement crops also have shorter growing periods — so the
classification is driven by the ceiling, and soybean's narrow 92–159 mm band restricts it
to 5 months in Kampala against banana's 9.

**P5 — Taxis.** The Ntinda market clears at **P\* = UGX 2,200, Q\* = 76**, so UGX 2,000
sustains a shortage of ~10 passengers per trip-hour. Revenue follows fare, not volume
(Entebbe 3.6× the revenue on 1.4× the passengers). SES and the weighted average edge out
the plain moving average, but the α grid search lands on **α = 1.0** everywhere — smoothing
collapsing to the naive forecast, meaning ten observations hold nothing exploitable. Fleet
size depends on a definition the brief leaves open: **1 vehicle** per route as daily
totals, **6–8** as hourly stage counts. Reporting utilisation makes the ambiguity
decidable — 40–57% full on the daily reading, 80–89% on the hourly, and only the second
resembles a working route. Over 60 simulated days with a weekly pattern, seasonal-naïve
cuts MAE from 10.4 to 4.4 passengers (**58%**); the ensemble again landed between its
members rather than below them.

## Data sources and assumptions

* Population, rainfall, route and demand data are **illustrative**, as the brief states.
  **Masaka** and **Lira** (P1) are the two additional districts required by task 1; the P2
  demand CSV is generated here from a fixed seed.
* Crop water needs: Brouwer, C. & Heibloem, M. (1986) *Irrigation Water Management
  Training Manual No. 3: Irrigation Water Needs*, FAO, Table 4.
  <https://www.fao.org/4/s2022e/s2022e02.htm>
* **Optional P4 extension (real data).** Download monthly precipitation from the
  [NASA POWER Data Access Viewer](https://power.larc.nasa.gov/data-access-viewer/):
  community *Agroclimatology*, temporal *Monthly & Annual*, parameter *Precipitation
  Corrected (PRECTOTCORR)*, 2010–2023, CSV. Points: Kampala (0.35, 32.58), Gulu
  (2.78, 32.30), Mbarara (−0.61, 30.65). Save as `data/nasa_power_Kampala.csv` and re-run
  notebook 4 for monthly box plots. The notebook runs fine without them.

## AI-use declaration

_Edit this section so that it describes accurately how **you** used AI._

This repository was developed with the help of an AI coding assistant. It was used to
scaffold the repository structure, draft the class designs, implementations, tests and
notebook text, and to check results. I have reviewed, run and tested all of the code:
`pytest` passes and every notebook runs top to bottom with Restart & Run All. I can
explain each method, formula and design decision. Key results are checked against hand
calculations, which appear as `assert` statements in the notebooks.
