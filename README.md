# Twin-Aware Power Toolkit

Twin trials demand special handling for intra-pair correlation and mixed singleton/twin cohorts. This repository packages the
core Monte Carlo + analytic engines for two ongoing studies along with a unified Streamlit UI so the science and stats teams can
share one interface.

The codebase brings together:

* **Biological Age trial** – within-pair randomized twin RCT (DunedinPACE, GrimAge, custom endpoints).
* **Sleep (ISI) trial** – individually randomized RCT with a large twin fraction and zygosity-specific clustering.

Both projects expose CLIs and Streamlit apps; the top-level `streamlit_app.py` unifies them behind a single sidebar selector.


## Repository Layout

```
.
├── streamlit_app.py              # Unified Streamlit launcher
├── biologicol_age/               # Bio-age paired-RCT tooling
│   ├── power_twin_age.py         # Analytic & simulation engines + CLI
│   ├── streamlit_app.py          # Study-specific Streamlit shell
│   └── requirements.txt          # Narrow dependency pin set
├── sleep/                        # Sleep ISI simulation tooling
│   ├── power_twin_sleep.py       # Simulation engine + CLI
│   ├── streamlit_app.py          # Study-specific Streamlit shell
│   └── requirements.txt
└── tests/
    └── test_power_twin_age.py    # Analytic + CLI regression suite for bio-age module
```


## Installation

Both subprojects share a consistent Python stack. Use whatever workflow you prefer (`pip`, `uv`, etc.); the examples below cover
the common options.

### Using `pip`

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r biologicol_age/requirements.txt -r sleep/requirements.txt
```

### Using [`uv`](https://github.com/astral-sh/uv)

```bash
uv venv .venv
source .venv/bin/activate  # optional; uv run will auto-activate
uv pip install -r biologicol_age/requirements.txt -r sleep/requirements.txt
```

The requirements files intentionally avoid heavyweight geospatial/ML dependencies so installs finish quickly on macOS, Linux, or
Windows.


## Unified Streamlit App

From the repository root (with your virtual environment active):

```bash
streamlit run streamlit_app.py
```

In the sidebar you can:

1. Choose **Biological Age** or **Sleep (ISI)**.
2. Set effect sizes, ICCs, attrition, contamination, seed, and Monte Carlo iterations.
3. Select an analysis goal (power for fixed N, solve for N, minimal detectable effect, or bio-age co-primary joint power).

Results surfaces include enrollment inflation, contamination-adjusted effects, and Monte Carlo standard errors when simulations
are enabled.


## Command-Line Interfaces

Both analysis engines can be scripted for batch workflows or reproducible reports.

* **Biological Age** – see [`biologicol_age/README.md`](biologicol_age/README.md) for detailed CLI examples covering analytic and
  simulation modes, co-primary endpoints, and helper utilities (e.g., `sd_change_from_pre_post`).
* **Sleep (ISI)** – see [`sleep/README.md`](sleep/README.md) for Monte Carlo examples with cluster-robust OLS and MixedLM
  analyses, contamination, and attrition modelling.

Typical invocations look like:

```bash
python biologicol_age/power_twin_age.py --mode power --n-pairs 700 --endpoint dunedinpace --effect-pct 3 --sd-change 0.10 --icc-mz 0.55 --icc-dz 0.55 --prop-mz 0.5
python sleep/power_twin_sleep.py --mode power --n-total 220 --effect-points 6 --sd-change 7 --prop-twins 0.9 --prop-mz 0.5 --icc-mz 0.5 --icc-dz 0.25 --sims 2000
```

Each script prints a structured summary (power, required enrollment, contamination note, etc.) that can be captured in reports or
downstream tooling.


## Testing

The biological-age module ships with a regression suite that exercises validation helpers, analytic solutions, simulation
stability, and the CLI wrapper. Run it from the repo root:

```bash
PYTHONPATH=. pytest tests/test_power_twin_age.py
```

(`PYTHONPATH=.` ensures the `biologicol_age` package is importable when running tests from the repo root.) Add similar coverage
for the sleep module as new features land.


## Troubleshooting & Tips

* **SciPy / statsmodels optionality** – analytic paths rely on NumPy/SciPy only; simulation gracefully degrades if MixedLM is not
  present by using cluster-robust OLS.
* **Monte Carlo precision** – increase `--sims` (CLI) or the slider (UI) for publication-quality precision. Monte Carlo SE is
  roughly `sqrt(p·(1−p)/sims)`.
* **Seeds** – set a seed to reproduce simulation runs. Changing seeds provides a sense of Monte Carlo variability.
* **Attrition** – reported power is for *completing* participants/pairs; enrollment counts are inflated by `1/(1−attrition)`.
* **Contamination** – observed effects are attenuated as `effect × (1 − rate × effect_fraction)`.

For study-specific defaults, rationale, and interpretation guidance, continue with the subproject READMEs linked above.

