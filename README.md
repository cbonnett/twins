# Twin-Aware Power Toolkit

Twin trials demand special handling for intra-pair correlation and mixed singleton/twin cohorts. This repository packages the
core Monte Carlo + analytic engines for two ongoing studies along with a unified Streamlit UI so the science and stats teams can
share one interface.

The codebase brings together:

- Biological Age trial — within-pair randomized twin RCT (DunedinPACE, GrimAge, custom endpoints).
- Sleep (ISI) trial — individually randomized RCT with a large twin fraction and zygosity-specific clustering.

Both projects expose CLIs, and the top-level `streamlit_app.py` provides a single, unified Streamlit UI with a sidebar selector.


## Repository Layout

```
.
├── .streamlit/
│   └── config.toml               # App theming
├── requirements.txt              # Shared dependencies for CLI + UI
├── streamlit_app.py              # Unified Streamlit launcher (bio-age + sleep)
├── biologicol_age/
│   ├── power_twin_age.py         # Analytic + simulation engine + CLI
│   └── README.md                 # Bio-age module docs
├── sleep/
│   ├── power_twin_sleep.py       # Simulation engine + CLI
│   ├── raw_sleep_design.md       # Design notes
│   └── README.md                 # Sleep module docs
└── tests/
    └── test_power_twin_age.py    # Regression suite for bio-age module
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
3. Choose a goal (power for fixed N, solve for N, MDE, or bio-age co-primary joint power).

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

- Biological Age — see `biologicol_age/README.md` for CLI examples covering analytic vs simulation, co-primary endpoints, and helpers like `sd_change_from_pre_post`.
- Sleep (ISI) — see `sleep/README.md` for Monte Carlo examples with cluster-robust OLS or MixedLM, contamination, and attrition modelling.

Typical invocations:

```bash
python biologicol_age/power_twin_age.py --mode power --n-pairs 700 --endpoint dunedinpace --effect-pct 3 --sd-change 0.10 --icc-mz 0.55 --icc-dz 0.55 --prop-mz 0.5
python sleep/power_twin_sleep.py --mode power --n-total 220 --effect-points 6 --sd-change 7 --prop-twins 0.9 --prop-mz 0.5 --icc-mz 0.5 --icc-dz 0.25 --sims 2000
```

Each script prints a structured summary (power, required enrollment, contamination note, etc.) suitable for reports or downstream tooling.


## Testing

The biological-age module ships with a regression suite covering validation helpers, analytic solutions, simulation stability, and the CLI wrapper. Run from the repo root:

```bash
PYTHONPATH=. pytest tests/test_power_twin_age.py
```

`PYTHONPATH=.` ensures `biologicol_age` is importable when running tests from the root. Add similar coverage for the sleep module as new features land.


## Troubleshooting & Tips

- SciPy/statsmodels optionality — analytic bio-age paths use NumPy/SciPy (with safe fallbacks); the sleep simulation uses cluster-robust OLS by default and falls back to GLS if MixedLM is unavailable.
- Monte Carlo precision — increase `--sims` (CLI) or the slider (UI) for publication-quality precision. MC SE ≈ `sqrt(p·(1−p)/sims)`.
- Seeds — set a seed to reproduce simulation runs. Vary to gauge Monte Carlo variability.
- Attrition — power is for completing participants/pairs; enrollment inflates by `1/(1−attrition)`.
- Contamination — observed effects shrink as `effect × (1 − rate × fraction)`.

For study-specific defaults, rationale, and interpretation, see the subproject READMEs.
