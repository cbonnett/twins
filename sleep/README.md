# SIESTA-LLM — Sleep ISI Twin-Aware Power Analysis

The sleep module simulates an individually randomized RCT with a high proportion of twins (both monozygotic and dizygotic).
Within-pair correlation is explicitly modeled, enabling realistic power and sample-size planning for the Insomnia Severity Index
(ISI) change at Week 8.


## Contents

- `power_twin_sleep.py` — Monte Carlo engine + CLI
- `README.md` — this documentation

The unified Streamlit UI lives at the repository root (`streamlit_app.py`) and includes a Sleep (ISI) panel.


## Installation

Install dependencies from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r ../requirements.txt
```

Or with [`uv`](https://github.com/astral-sh/uv):

```bash
uv venv .venv
source .venv/bin/activate  # optional with `uv run`
uv pip install -r ../requirements.txt
```


## Streamlit UI

Launch from the repository root and select “Sleep (ISI)” in the sidebar:

```bash
streamlit run streamlit_app.py
```

Key sidebar controls:

1. Mode — estimate power at fixed `N` or solve for `N` that meets a target power.
2. Effect size (ISI points), standard deviation of change, and alpha level.
3. Twin structure — proportion twins vs singletons, proportion MZ, ICCs for MZ/DZ.
4. Analysis method — cluster-robust OLS (default) or MixedLM random intercept.
5. Monte Carlo settings — number of simulations and RNG seed.
6. Optional contamination (rate × fraction) and attrition assumptions.

Outputs include estimated power or required total `N`, contamination‑adjusted effects, attrition‑inflated enrollment counts,
95% Monte Carlo confidence intervals for power, and Monte Carlo standard errors.


## Command-Line

The CLI mirrors the UI and is convenient for scripted sweeps or reproducible reports. You can run it from this folder as
`python power_twin_sleep.py` or from the repository root as `python sleep/power_twin_sleep.py`.

Modes:

- `power` — estimate power for a fixed total sample size.
- `n-for-power` — solve for the smallest total `N` achieving a target power.

Examples:

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

- `--analysis {cluster_robust,mixedlm}` — choose estimator. MixedLM is optional and, if it fails to converge, the CLI falls back to cluster‑robust OLS.
- `--seed` — reproducible Monte Carlo draws.
- `--n-jobs` — parallel worker processes; use `-1` for all CPU cores. Combine with `--chunk-size` to tune workload per process.
- `--chunk-size` — number of simulations per task when multiprocessing is enabled (default 64).
- `--contamination-rate` / `--contamination-effect` — apply effect attenuation (`effect_obs = effect × (1 − rate × fraction)`).
- `--attrition-rate` — compute inflated enrollment counts (power is always on completing participants).


## Parameter Guidance

- Effect (ISI points) — specify the beneficial magnitude. The simulation applies the sign so estimated treatment effects are negative when beneficial. MCID ≈ 6 points.
- Standard deviation of change — typical values range from 5–7; use baseline-adjusted values if planning ANCOVA on post-ISI. Helper: `sd_change_from_pre_post(sd_pre, sd_post, rho)`.
- Twin mix — `prop_twins` is the fraction of participants in twin pairs; `prop_mz` is the monozygotic share among twins.
- ICCs — separate within-pair correlations by zygosity; singletons implicitly have ICC=0.
- Alpha — two-sided test level (default 0.05).
- Simulations (`--sims`) — more iterations reduce Monte Carlo error at the cost of runtime.


## Outputs

- Estimated power (fixed `N`) or required total sample size (`n-for-power`).
- 95% Monte Carlo confidence interval for power.
- Monte Carlo standard error, average estimated treatment effect, and Kish effective sample size.
- Parallel simulation support with reproducible seeds across worker counts; speed-ups scale with available cores for large `--sims`.
- Contamination-adjusted effect magnitude and attrition-inflated enrollment counts when those features are enabled.


## Assumptions & Model Notes

- Individual randomization; co-twins are randomized independently.
- Outcome model: change = treatment effect + pair random effect + residual.
- Pair random effect variance is calibrated to match the specified ICCs; singletons omit the random effect.
- Zygosity indicators enter the analysis model as fixed effects.
- Normality assumed; no informative dropout beyond attrition scalar.


## Reproducibility & Troubleshooting

- Activate your virtual environment before installing requirements.
- If `statsmodels` or MixedLM is missing, stick with `--analysis cluster_robust` (default). If MixedLM fails to converge, the CLI falls back to cluster‑robust OLS.
- Install Streamlit separately if needed: `pip install streamlit` then run from the repo root: `streamlit run streamlit_app.py`.
- Change the Streamlit port if 8501 is busy: `streamlit run streamlit_app.py --server.port 8502`.
- Set `--seed` (CLI) or the UI seed slider to make Monte Carlo estimates reproducible. Increase `--sims` for final numbers.


## Sensitivity Exploration

- Vary ICCs — e.g., MZ `0.40–0.60`, DZ `0.20–0.35` — to assess robustness to correlation assumptions.
- Explore different twin mixes; the Kish effective sample size summary highlights the design effect from clustering.
- Adjust `sd_change` in line with expected baseline-post correlations (smaller SD means higher power at fixed `N`).


## Testing

Run from the repository root so subpackages are importable:

```bash
PYTHONPATH=. pytest tests/test_sleep_study.py -q
PYTHONPATH=. pytest tests/test_power_twin_sleep_parallel.py -q
```

The suite focuses on SIESTA‑LLM protocol parameters and verifies simulation behavior, continuity, and robustness.


## Parallel Execution Notes

- The Monte Carlo engine uses `ProcessPoolExecutor` when `--n-jobs > 1`, and automatically falls back to thread-based execution if the host blocks process creation (common on managed CI or notebooks). Chunk seeds are generated deterministically from the top-level seed, so results are identical regardless of worker count.
- Adjust `--chunk-size` to balance CPU utilisation and scheduling overhead. Larger chunks reduce coordination cost for very large `--sims`, while smaller chunks keep the workload evenly distributed across heterogeneous cores.
- The parallel test (`tests/test_power_twin_sleep_parallel.py`) exercises serial vs parallel equivalence, deterministic chunking, and the `find_n_for_power` binary search when multiprocessing is enabled.
- For exploratory runs stick to the default single process; switch to `--n-jobs -1` (or an explicit core count) for production sweeps where wall-clock time dominates.
