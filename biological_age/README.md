# Twin-Aware Power — Biological Age (Within-Pair RCT)

This module implements analytic and Monte Carlo calculations for a within-pair randomized twin RCT targeting biological age
reversal. It supports DunedinPACE (pace of aging), GrimAge (years), and custom endpoints with zygosity-specific ICCs,
contamination, attrition, and co-primary testing.


## Contents

- `power_twin_age.py` — analytic + simulation engines and CLI
- `README.md` — this documentation

The unified Streamlit UI lives at the repository root (`streamlit_app.py`) and includes a Biological Age panel.


## Installation

Install dependencies from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r ../requirements.txt
```

or with [`uv`](https://github.com/astral-sh/uv):

```bash
uv venv .venv
source .venv/bin/activate  # optional when using `uv run`
uv pip install -r ../requirements.txt
```


## Streamlit UI

Launch from the repository root and select “Biological Age” in the sidebar:

```bash
streamlit run streamlit_app.py
```

Sidebar highlights:

1. Endpoint (DunedinPACE %, GrimAge years, or custom).
2. Effect inputs — absolute magnitude, standardized paired `d`, or endpoint-specific helpers (percent/years).
3. ICCs (MZ/DZ), proportion MZ, attrition, contamination, alpha, and Monte Carlo controls.
4. Goal — power for fixed pairs, pairs for target power, minimal detectable effect (MDE), or co-primary joint power.

Enable simulation for heterogeneous ICC draws and co-primary joint power; Monte Carlo summaries include the standard error of
power estimates.


## Command-Line

`power_twin_age.py` mirrors the UI features for scripting or reproducible reports. You can run it from this folder as
`python power_twin_age.py` or from the repository root as `python biological_age/power_twin_age.py`.

Key modes:

- `power` — power for a fixed number of completing pairs.
- `pairs-for-power` — smallest pair count achieving a target power.
- `mde` — minimal detectable effect (beneficial magnitude) at a target power.
- `curve` — power curve CSV for a range of pairs.
- `co-primary-power` — joint probability both endpoints are significant (Bonferroni-adjusted by default).

Examples:

```bash
# Analytic power (paired t) at 700 completing pairs for DunedinPACE
python power_twin_age.py --mode power --n-pairs 700 --endpoint dunedinpace --effect-pct 3 \
    --sd-change 0.10 --icc-mz 0.55 --icc-dz 0.55 --prop-mz 0.5

# Solve for pairs needed for 90% power on GrimAge with attrition planning
python power_twin_age.py --mode pairs-for-power --target-power 0.90 --endpoint grimage \
    --effect-years 2.0 --sd-change 3.0 --icc-mz 0.6 --icc-dz 0.3 --prop-mz 0.5 \
    --attrition-rate 0.40

# Minimal detectable effect (analytic)
python power_twin_age.py --mode mde --n-pairs 700 --endpoint dunedinpace --sd-change 0.10 \
    --icc-mz 0.55 --icc-dz 0.55 --target-power 0.80

# Co-primary joint power with simulation
python power_twin_age.py --mode co-primary-power --n-pairs 700 --endpoint dunedinpace \
    --effect-pct 3 --sd-change 0.10 --endpoint2 grimage --effect2-years 2.0 \
    --sd2-change 3.0 --use-simulation --sims 5000

# Power curve CSV (pairs 200..1000)
python power_twin_age.py --mode curve --endpoint dunedinpace --effect-pct 3 \
    --sd-change 0.10 --icc-mz 0.55 --icc-dz 0.55 --prop-mz 0.5 \
    --curve-start 200 --curve-stop 1000 --curve-step 50 > dpace_curve.csv
```

Helpful flags:

- `--use-simulation --sims 3000` — enable Monte Carlo (recommended for co-primary or heterogeneous ICC scenarios).
- `--pair-effect-corr` — correlation of endpoint-specific pair effects in co-primary mode (default 0.8).
- `--seed` — reproducible random draws.
- `--contamination-rate`, `--contamination-effect` — model effect attenuation.
- `--attrition-rate` — compute inflated enrollment alongside completing pairs.
- `--d-std` — specify standardized paired d instead of absolute magnitude (requires SD(change) and ICC to be provided).


## Parameter Guidance (Protocol-Aligned)

- Effects (absolute scale)
  - DunedinPACE: specify a positive slowing fraction (e.g., `0.03` = 3% slower).
  - GrimAge: specify years younger (e.g., `2.0`).
  - Custom: enter the beneficial magnitude in native units. Internally the treated change is reduced by this amount so the
    paired difference (treated − control) is negative; reported magnitudes are positive.
- Standardized paired `d` — `d = effect_abs / sqrt(2·(1−ICC)·sd_change²)`. Use CLI `--d-std` when comparing across endpoints.
- SD of change — typical ranges: DunedinPACE `0.08–0.12`, GrimAge `2–4` years. Helper: `sd_change_from_pre_post(sd_pre, sd_post, rho)`.
- ICCs — specify separately for MZ and DZ. The effective ICC used in analytic mode is
  `1 − [prop_mz·(1−ICC_MZ) + (1−prop_mz)·(1−ICC_DZ)]`.
- Alpha / co-primary — single endpoint defaults to `α=0.05`. In co-primary mode the CLI auto-applies Bonferroni (`α=0.025` per endpoint) unless overridden.
- Monte Carlo precision — MC SE ≈ `sqrt(p·(1−p)/sims)`.


## Outputs

- Power, required pairs, or minimal detectable effect (with simulation-based standard errors when applicable).
- Effective ICC, SD of within-pair difference, and paired Cohen’s `d` summaries.
- Enrollment inflation when attrition is supplied.
- Co-primary results report marginal and joint power along with per-endpoint alpha.
  Printed summaries assume the biological age primary endpoint at Week 24.


## Worked Examples

* **Contamination** – if 30% of controls adopt 50% of the intervention effect, the observed magnitude is
  `effect × (1 − 0.30 × 0.50) = 0.85 × effect`. A 3% DunedinPACE slowing input therefore displays as 2.55% observed.
* **Attrition** – enrollment required for `n` completing pairs at attrition rate `a` equals `ceil(n / (1−a))`. For 700 completing
  pairs and 40% attrition the CLI reports 1,167 pairs (2,334 individuals) to enroll.


## Testing

The top-level `tests/test_power_twin_age.py` suite covers validation helpers, analytic calculations, Monte Carlo stability, and CLI behaviour. Run from the repository root:

```bash
PYTHONPATH=. pytest tests/test_power_twin_age.py
```


## Troubleshooting

- Missing dependencies — confirm your virtual environment is active, then reinstall with `pip install -r ../requirements.txt` from the repo root.
- SciPy unavailable — analytic paths fall back to accurate normal approximations; simulation does not require `statsmodels` for this module.
- Streamlit — install via `pip install streamlit` if not already present and run `streamlit run streamlit_app.py` from the repo root.


## Reproducibility Tips

- Fix `--seed` (CLI) or the UI seed input for reproducible Monte Carlo runs.
- Increase `--sims` (or the app slider) once scenarios are finalised to reduce Monte Carlo error.
