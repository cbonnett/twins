# Twin-Aware Power Toolkit

Twin trials demand special handling for intra-pair correlation and mixed singleton/twin cohorts. This repository packages the
core Monte Carlo + analytic engines for two ongoing studies along with a unified Streamlit UI so the science and stats teams can
share one interface.

The codebase brings together:

- Biological Age trial — within-pair randomized twin RCT (DunedinPACE, GrimAge, custom endpoints).
- Biological Age trial — within-pair randomized twin RCT (DunedinPACE; custom secondary endpoint).
- Sleep (ISI) trial — individually randomized RCT with a large twin fraction and zygosity-specific clustering.

Both projects expose CLIs, and the top-level `streamlit_app.py` provides a single, unified Streamlit UI with a sidebar selector.


## Repository Layout

```
.
├── .streamlit/
│   └── config.toml               # App theming
├── requirements.txt              # Shared dependencies for CLI + UI
├── streamlit_app.py              # Unified Streamlit launcher (bio-age + sleep)
├── biological_age/
│   ├── power_twin_age.py         # Analytic + simulation engine + CLI
│   └── README.md                 # Bio-age module docs
├── sleep/
│   ├── power_twin_sleep.py       # Simulation engine + CLI
│   └── README.md                 # Sleep module docs
└── tests/
    ├── test_power_twin_age.py            # Regression suite for bio-age module
    ├── test_sleep_study.py               # Protocol-oriented tests for sleep module
    └── test_power_twin_sleep_parallel.py # Parallel determinism + search tests
```


## Installation

Use a single environment for both modules (pip or uv are fine).

- pip

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

- uv

```bash
uv venv .venv
source .venv/bin/activate  # optional; uv run auto-activates
uv pip install -r requirements.txt
```

The dependency set is intentionally lean so installs complete quickly on macOS, Linux, or Windows.


## Unified Streamlit App

From the repository root (with your virtual environment active):

```bash
streamlit run streamlit_app.py
```

Sidebar options:

1. Pick Biological Age or Sleep (ISI).
2. Configure effect sizes, ICCs, zygosity mix, contamination, attrition, seed, and sims.
3. Choose a goal:
   - Biological Age: power at fixed pairs, pairs for target power, MDE, or co‑primary joint power.
   - Sleep (ISI): power at fixed N or N for target power.

Results include enrollment inflation, contamination-adjusted effects, and Monte Carlo standard errors when simulation is used.

Recent UI enhancements:

- Default to simulation mode (realistic, enables co-primary).
- 95% Monte Carlo intervals for power where applicable.
- Design summaries alongside results (e.g., effective ICC, observed effect).
- Scenario JSON download (inputs + outputs) for sharing.
- Theming via `.streamlit/config.toml`.

Shareable URL params:

- `?study=bio` or `?study=sleep` to preselect the study.
- `&use_sim=1` to default the bio-age panel to simulation.


## Command-Line Interfaces

Both analysis engines can be scripted for batch workflows or reproducible reports.

- Biological Age — see `biological_age/README.md` for CLI examples covering analytic vs simulation, co-primary endpoints, and helpers like `sd_change_from_pre_post`.
- Sleep (ISI) — see `sleep/README.md` for Monte Carlo examples with cluster-robust OLS or MixedLM, contamination, and attrition modelling.

Typical invocations:

```bash
python biological_age/power_twin_age.py --mode power --n-pairs 700 --endpoint dunedinpace --effect-pct 3 --sd-change 0.10 --icc-mz 0.55 --icc-dz 0.55 --prop-mz 0.5
python sleep/power_twin_sleep.py --mode power --n-total 220 --effect-points 6 --sd-change 7 --prop-twins 0.9 --prop-mz 0.5 --icc-mz 0.5 --icc-dz 0.25 --sims 2000 --n-jobs -1
 
# Co‑primary biological age example
python biological_age/power_twin_age.py --mode co-primary-power --n-pairs 700 \
  --endpoint dunedinpace --effect-pct 3 --sd-change 0.10 \
  --endpoint2 custom --effect2-abs 0.75 --sd2-change 3.0 \
  --use-simulation --sims 5000 --pair-effect-corr 0.8

# Individual‑level (two‑sample) biological age examples
python biological_age/power_twin_age.py --mode two-sample-power \
  --n-per-group 150 --endpoint dunedinpace --effect-pct 2.5 --sd-change 0.10 \
  --ancova-r2 0.50 --alpha 0.05 --attrition-rate 0.27 --deff-icc 0.50 --deff-m 2

python biological_age/power_twin_age.py --mode two-sample-n-for-power \
  --endpoint dunedinpace --effect-pct 2.5 --sd-change 0.10 --ancova-r2 0.50 \
  --alpha 0.05 --target-power 0.80 --deff-icc 0.50 --deff-m 2 --attrition-rate 0.27
```

Each script prints a structured summary (power, required enrollment, contamination note, etc.) suitable for reports or downstream tooling.


## Testing

Run all tests from the repository root:

```bash
PYTHONPATH=. pytest -q
```

Suites included:
- `tests/test_power_twin_age.py` — validation helpers, analytic formulas, simulation stability, CLI behavior (biological age).
- `tests/test_sleep_study.py` — protocol-oriented checks and simulation behavior (sleep/ISI).
- `tests/test_power_twin_sleep_parallel.py` — regression suite for multiprocessing determinism and parallel `find_n_for_power`.

`PYTHONPATH=.` ensures the subpackages are importable when running from the root.


## Troubleshooting & Tips

- SciPy/statsmodels optionality — analytic bio‑age paths use NumPy/SciPy (with safe fallbacks). The sleep module uses cluster‑robust OLS by default; MixedLM is optional and, if it fails to converge, the CLI falls back to cluster‑robust OLS.
- Monte Carlo precision — increase `--sims` (CLI) or the slider (UI) for publication-quality precision. MC SE ≈ `sqrt(p·(1−p)/sims)`.
- Seeds — set a seed to reproduce simulation runs. Vary to gauge Monte Carlo variability.
- Attrition — power is for completing participants/pairs; enrollment inflates by `1/(1−attrition)`.
- Contamination — observed effects shrink as `effect × (1 − rate × fraction)`.

For study-specific defaults, rationale, and interpretation, see the subproject READMEs.
