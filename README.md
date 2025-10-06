# Twin-Aware Power Toolkit

Tools for planning and interrogating randomized trials that enroll twins. The repository bundles two
statistical engines – a within-pair biological age trial and an individually randomized sleep trial –
plus a Streamlit front end so statisticians and domain scientists can explore scenarios with a shared UI.

The codebase is intentionally lightweight: everything is pure Python, tested, and installable with a
single `requirements.txt` file.

## Components

- **Biological Age (within-pair RCT).** Analytic and Monte Carlo power calculations for DunedinPACE,
  GrimAge, or custom change scores with contamination, attrition, and co-primary endpoint support.
- **Sleep / ISI (individual RCT with many twins).** Monte Carlo engine that models mixed
  singleton/twin cohorts, zygosity-specific ICCs, contamination, and attrition.
- **Unified Streamlit app.** `streamlit_app.py` exposes both engines through a single sidebar-driven UI.
- **Tests.** `pytest` suites regression-test the analytic formulas and simulation behaviour for both
  modules.

## Repository Layout

```
.
├── README.md
├── requirements.txt
├── streamlit_app.py
├── biological_age/
│   ├── README.md
│   └── power_twin_age.py
├── sleep/
│   ├── README.md
│   ├── power_twin_sleep.py
│   └── raw_sleep_design.md
└── tests/
    ├── test_power_twin_age.py
    └── test_sleep_study.py
```

## Quick Start

1. Create and activate a virtual environment (Python 3.10+ recommended).
2. Install the shared dependencies.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The requirements cover both command-line engines, the Streamlit UI, and the test suite.

## Unified Streamlit App

Launch the Streamlit UI from the repository root after installing dependencies:

```bash
streamlit run streamlit_app.py
```

The sidebar lets you switch between the biological age and sleep modules, configure effect sizes,
ICCs, contamination/attrition assumptions, and Monte Carlo settings, and download scenario JSON
summaries. Monte Carlo power intervals are computed from the simulation standard error.

## Command-Line Interfaces

Both statistical engines can be run directly from the command line for scripted workflows or report
reproducibility. Each README within `biological_age/` and `sleep/` contains detailed usage notes; the
typical invocation pattern is:

```bash
python biological_age/power_twin_age.py --mode power --n-pairs 700 --endpoint dunedinpace \
    --effect-pct 3 --sd-change 0.10 --icc-mz 0.55 --icc-dz 0.55 --prop-mz 0.5

python sleep/power_twin_sleep.py --mode power --n-total 220 --effect-points 6 --sd-change 7 \
    --prop-twins 0.9 --prop-mz 0.5 --icc-mz 0.5 --icc-dz 0.25 --sims 2000
```

The CLIs expose modes for power at fixed N, solving for N, minimal detectable effect, power curves,
and co-primary joint power (biological age).

## Testing

Run the regression suite from the repository root:

```bash
pytest -q
```

The tests validate analytic formulas, Monte Carlo accuracy, and CLI behaviour across both modules.

## Troubleshooting & Tips

- Monte Carlo precision improves with larger `--sims` values; the UI reports the Monte Carlo standard
  error so you can gauge stability.
- Set `--seed` (or the UI seed input) to make simulation runs reproducible.
- Attrition and contamination inputs shrink observed effects; the engines report both the observed
  and true magnitudes so you can plan enrollment inflation accordingly.
- MixedLM analysis in the sleep module falls back to cluster-robust OLS if `statsmodels` cannot fit
  the random-effects model.

For deeper parameter guidance and worked examples, see the module-level READMEs in
`biological_age/` and `sleep/`.
