# SIESTA-LLM — Sleep ISI Twin-Aware Power Analysis

The sleep module simulates an individually randomized RCT with a high proportion of twins (both monozygotic and dizygotic).
Within-pair correlation is explicitly modelled, enabling realistic power and sample-size planning for the Insomnia Severity Index
(ISI) change at Week 8.


## Contents

* `power_twin_sleep.py` – Monte Carlo engine + CLI
* `streamlit_app.py` – Streamlit UI dedicated to the sleep study
* `requirements.txt` – dependency list shared by CLI + UI


## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Or use [`uv`](https://github.com/astral-sh/uv) for faster installs:

```bash
uv venv .venv
source .venv/bin/activate  # optional with `uv run`
uv pip install -r requirements.txt
```


## Streamlit UI

Launch from this directory:

```bash
streamlit run streamlit_app.py
```

Key sidebar controls:

1. Mode selection – estimate power at fixed `N` or solve for `N` that meets a target power.
2. Effect size (ISI points), standard deviation of change, and alpha level.
3. Twin structure – proportion twins vs singletons, proportion MZ, ICCs for MZ/DZ.
4. Analysis method – cluster-robust OLS (default) or MixedLM random intercept.
5. Monte Carlo settings – number of simulations and RNG seed.
6. Optional contamination (rate × effect fraction) and attrition assumptions.

Outputs include estimated power or required total `N`, contamination-adjusted effects, attrition-inflated enrollment counts, and
Monte Carlo standard errors.


## Command-Line Usage

The CLI mirrors the UI and is convenient for scripted sweeps or reproducible reports.

### Core modes

| Mode | Description |
| --- | --- |
| `power` | Estimate power for a fixed total sample size. |
| `n-for-power` | Solve for the smallest total `N` achieving a target power. |

### Examples

```bash
# Power at N=200 with heavy twin composition
python power_twin_sleep.py --mode power --n-total 200 --effect-points 6 --sd-change 7 \
    --prop-twins 0.9 --prop-mz 0.5 --icc-mz 0.5 --icc-dz 0.25 --alpha 0.05 --sims 2000

# Required N for 90% power
python power_twin_sleep.py --mode n-for-power --target-power 0.90 --effect-points 6 \
    --sd-change 7 --prop-twins 0.9 --prop-mz 0.5 --icc-mz 0.5 --icc-dz 0.25 \
    --alpha 0.05 --sims 1500

# Include contamination and attrition considerations
python power_twin_sleep.py --mode power --n-total 220 --effect-points 6 --sd-change 7 \
    --prop-twins 0.9 --prop-mz 0.5 --icc-mz 0.5 --icc-dz 0.25 --alpha 0.05 --sims 2000 \
    --contamination-rate 0.30 --contamination-effect 0.50 --attrition-rate 0.10

# MixedLM analysis (falls back gracefully if statsmodels MixedLM is unavailable)
python power_twin_sleep.py --mode power --n-total 160 --effect-points 5 --sd-change 6 \
    --prop-twins 0.8 --prop-mz 0.5 --icc-mz 0.45 --icc-dz 0.20 --analysis mixedlm --sims 1200
```

Helpful flags:

* `--analysis {cluster_robust,mixedlm}` – choose estimator; MixedLM automatically reverts to GLS if the model fails to converge.
* `--seed` – reproducible Monte Carlo draws.
* `--contamination-rate` / `--contamination-effect` – apply effect attenuation (`effect_obs = effect × (1 − rate × fraction)`).
* `--attrition-rate` – compute inflated enrollment counts (power is always on completing participants).


## Parameter Guidance

* **Effect (ISI points)** – specify the beneficial magnitude (control minus treatment). The simulation applies the sign so
  reported treatment effects are negative when beneficial. MCID ≈ 6 points.
* **Standard deviation of change** – typical values range from 5–7; use baseline-adjusted values if planning ANCOVA on post-ISI.
  Helper function: `sd_change_from_pre_post(sd_pre, sd_post, rho)`.
* **Twin mix** – `prop_twins` is the fraction of participants who are in twin pairs; `prop_mz` is the monozygotic share among
  twins. The remainder are dizygotic.
* **ICCs** – separate within-pair correlations by zygosity; singletons implicitly have ICC=0.
* **Alpha** – two-sided test level (default 0.05).
* **Simulations (`--sims`)** – more iterations reduce Monte Carlo error at the cost of runtime.


## Outputs

* Estimated power (fixed `N`) or required total sample size (`n-for-power`).
* Monte Carlo standard error, average estimated treatment effect, and Kish effective sample size.
* Contamination-adjusted effect magnitude and attrition-inflated enrollment counts when those features are enabled.


## Assumptions & Model Notes

* Individual randomization; co-twins are randomized independently.
* Outcome model: change = treatment effect + pair random effect + residual.
* Pair random effect variance is calibrated to match the specified ICCs; singletons omit the random effect.
* Zygosity indicators enter the analysis model as fixed effects.
* Normality assumed; no informative dropout beyond attrition scalar.


## Reproducibility & Troubleshooting

* Activate your virtual environment before installing requirements.
* If `statsmodels` or MixedLM is missing, stick with `--analysis cluster_robust` (default). The CLI automatically falls back to a
  GLS solution when MixedLM fails to converge.
* Install Streamlit separately if needed: `pip install streamlit` then re-run `streamlit run streamlit_app.py`.
* Change the Streamlit port if 8501 is busy: `streamlit run streamlit_app.py --server.port 8502`.
* Set `--seed` (CLI) or the UI seed slider to make Monte Carlo estimates reproducible. Increase `--sims` for final numbers.


## Sensitivity Exploration

* Vary ICCs – e.g., MZ `0.40–0.60`, DZ `0.20–0.35` – to assess robustness to correlation assumptions.
* Explore different twin mixes; the Kish effective sample size summary highlights the design effect from clustering.
* Adjust `sd_change` in line with expected baseline-post correlations (smaller SD means higher power at fixed `N`).
