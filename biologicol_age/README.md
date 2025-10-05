# Twin-Aware Power — Biological Age (Within-Pair RCT)

This module implements analytic and Monte Carlo calculations for a within-pair randomized twin RCT targeting biological age
reversal. It supports DunedinPACE (pace of aging), GrimAge (years), and custom endpoints with zygosity-specific ICCs,
contamination, attrition, and co-primary testing.


## Contents

* `power_twin_age.py` – core engines + CLI
* `streamlit_app.py` – study-specific Streamlit entry point
* `requirements.txt` – slim dependency list shared by CLI + UI


## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Or, with [`uv`](https://github.com/astral-sh/uv):

```bash
uv venv .venv
source .venv/bin/activate  # optional when using `uv run`
uv pip install -r requirements.txt
```


## Streamlit UI

Launch from this directory:

```bash
streamlit run streamlit_app.py
```

The sidebar exposes:

1. Endpoint selection (DunedinPACE %, GrimAge years, or custom).
2. Effect scale inputs – absolute magnitude, standardized paired `d`, or endpoint-specific helpers (percent/years).
3. Timepoint labels for quick scenario switching (Week 12 interim, Week 24 primary, Month 12 follow-up).
4. ICCs for MZ/DZ, proportion MZ, attrition, contamination, alpha, and Monte Carlo controls.
5. Analysis mode (power for fixed pairs, solve for pairs, minimal detectable effect, or co-primary joint power).

Enable simulation for heterogeneous ICC draws or co-primary joint power; Monte Carlo summaries include the standard error of the
estimated power.


## Command-Line Usage

`power_twin_age.py` mirrors the UI features for scripting or reproducible reports. Key modes:

| Mode | Description |
| --- | --- |
| `power` | Power for a fixed number of completing pairs. |
| `pairs-for-power` | Smallest pair count achieving a target power. |
| `mde` | Minimal detectable effect (beneficial magnitude) at a target power. |
| `curve` | Power curve CSV for a range of pairs. |
| `co-primary-power` | Joint probability both endpoints are significant (Bonferroni-adjusted by default). |

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
```

Helpful flags:

* `--use-simulation --sims 3000` – enable Monte Carlo (recommended for co-primary or heterogeneous ICC scenarios).
* `--pair-effect-corr` – correlation between endpoint-specific pair effects in co-primary mode (defaults to 0.8).
* `--seed` – reproducible random draws.
* `--contamination-rate`, `--contamination-effect` – apply treatment leakage attenuation.
* `--attrition-rate` – compute inflated enrollment counts alongside completing pairs.


## Parameter Guidance (Protocol-Aligned)

* **Effects (absolute scale)**
  * *DunedinPACE*: specify a positive slowing fraction (e.g., `0.03` = 3% slower). Week 24 primary target ≈ 2–3%.
  * *GrimAge*: specify years younger (e.g., `2.0`). Treat as exploratory until stronger evidence emerges.
  * *Custom*: enter the beneficial magnitude in native units. Internally the treated change is reduced by this amount so the
    paired difference (treated – control) is negative; reported magnitudes are positive.
* **Standardized paired `d`** – `d = effect_abs / sqrt(2·(1−ICC)·sd_change²)`. Use the CLI flag `--d-std` or UI toggle to input
  directly when comparing across endpoints.
* **SD of change** – typical ranges: DunedinPACE `0.08–0.12`, GrimAge `2–4` years. Use the helper
  `sd_change_from_pre_post(sd_pre, sd_post, rho)` when only pre/post SDs and correlation are available.
* **ICCs** – specify separately for MZ and DZ. The effective ICC used in analytic mode is
  `1 − [prop_mz·(1−ICC_MZ) + (1−prop_mz)·(1−ICC_DZ)]`.
* **Alpha / co-primary** – single endpoint defaults to `α=0.05`. In co-primary mode the script auto-applies Bonferroni (`α=0.025`
  per endpoint) unless overridden.
* **Monte Carlo precision** – Monte Carlo SE ≈ `sqrt(p·(1−p)/sims)`. At `p=0.85` and `sims=2000`, SE ≈ 0.009 (±1.8% at 95% CI).


## Outputs

* Power, required pairs, or minimal detectable effect (with simulation-based standard errors when applicable).
* Effective ICC, SD of within-pair difference, and paired Cohen’s `d` summaries.
* Enrollment inflation when attrition is supplied.
* Co-primary results report marginal and joint power along with per-endpoint alpha.


## Worked Examples

* **Contamination** – if 30% of controls adopt 50% of the intervention effect, the observed magnitude is
  `effect × (1 − 0.30 × 0.50) = 0.85 × effect`. A 3% DunedinPACE slowing input therefore displays as 2.55% observed.
* **Attrition** – enrollment required for `n` completing pairs at attrition rate `a` equals `ceil(n / (1−a))`. For 700 completing
  pairs and 40% attrition the CLI reports 1,167 pairs (2,334 individuals) to enroll.


## Testing

The top-level `tests/test_power_twin_age.py` suite covers validation helpers, analytic calculations, Monte Carlo stability, and
CLI behaviour. Run from the repository root:

```bash
pytest tests/test_power_twin_age.py
```


## Troubleshooting

* **Missing dependencies** – confirm your virtual environment is active, then reinstall with `pip install -r requirements.txt`.
* **SciPy/statsmodels unavailable** – analytic functions operate with NumPy/SciPy; simulation falls back to stable approximations
  if `statsmodels` MixedLM is missing.
* **Streamlit** – install via `pip install streamlit` if not already present.


## Reproducibility Tips

* Fix `--seed` (CLI) or the UI seed input for reproducible Monte Carlo runs.
* Increase `--sims` (or the app slider) once scenarios are finalised to reduce Monte Carlo error.
