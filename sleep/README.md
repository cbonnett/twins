# SIESTA-LLM — Twin-Aware Power Analysis (ISI Change)

This repo contains a simulation-based power analysis for an individually randomized RCT with a high proportion of twin participants (MZ/DZ). It estimates power for the primary endpoint (Insomnia Severity Index, ISI, change to Week 8) or finds the minimum total N to achieve a target power while modeling within-pair correlations by zygosity.

**Files**
- `streamlit_app.py` — Interactive Streamlit UI to run simulations, adjust parameters, and read assumptions/explanations.
- `power_twin_sleep.py` — Core simulation and analysis utilities (twin structure, ICCs, cluster-robust OLS/MixedLM/GLS fallback).
- `requirements.txt` — Python dependencies.

**What it models**
- Individual 1:1 randomization, including co-twins.
- Separate within-pair ICCs for MZ and DZ twins; singletons have ICC=0.
- Mix of twins and singletons; analysis includes zygosity indicators.
- Default analysis: OLS with cluster-robust SEs clustered by pair; MixedLM optional.
- Optional contamination modeling and attrition-based enrollment inflation.

## Quick Start

- Install deps (recommend a fresh virtualenv):
  - `python -m venv .venv && source .venv/bin/activate`
  - `pip install -r requirements.txt`
- Launch the app:
  - `streamlit run streamlit_app.py`

Then use the UI to:
- Choose mode: estimate power for a fixed `N` or find `N` for a target power.
- Adjust effect size (ISI points), SD of change, twin mix (MZ/DZ), ICCs, alpha, and number of simulations.
- Pick analysis method: `cluster_robust` or `mixedlm` (falls back to GLS if unavailable).
- Optionally model contamination (controls adopting part of the intervention) and plan for attrition.

## Quick Start (uv)

Use [uv](https://github.com/astral-sh/uv) for fast installs and runs.

- Install uv (macOS/Linux):
  - `curl -LsSf https://astral.sh/uv/install.sh | sh`
  - Or via Homebrew (if available): `brew install uv`
- Create and activate a virtual environment:
  - `uv venv .venv`
  - macOS/Linux: `source .venv/bin/activate`
  - Windows (PowerShell): `.\.venv\Scripts\Activate.ps1`
- Install dependencies with uv:
  - `uv pip install -r requirements.txt`
  - (Alternative sync): `uv pip sync requirements.txt`
- Run the Streamlit app (activation optional with uv run):
  - `uv run streamlit run streamlit_app.py`

CLI examples with uv (no manual activation needed):
- Estimate power for a fixed N:
  - `uv run python power_twin_sleep.py --mode power --n-total 200 --effect-points 6 --sd-change 7 --prop-twins 0.9 --prop-mz 0.5 --icc-mz 0.5 --icc-dz 0.25 --alpha 0.05 --sims 2000`
- Find N for target power:
  - `uv run python power_twin_sleep.py --mode n-for-power --target-power 0.9 --effect-points 6 --sd-change 7 --prop-twins 0.9 --prop-mz 0.5 --icc-mz 0.5 --icc-dz 0.25 --alpha 0.05 --sims 1500`
- Include contamination and attrition (example):
  - `uv run python power_twin_sleep.py --mode power --n-total 220 --effect-points 6 --sd-change 7 --prop-twins 0.9 --prop-mz 0.5 --icc-mz 0.5 --icc-dz 0.25 --alpha 0.05 --sims 2000 --contamination-rate 0.30 --contamination-effect 0.50 --attrition-rate 0.10`

## Command-Line (Non-interactive)

You can also run the simulation directly:
- Estimate power for a fixed `N`:
  - `python power_twin_sleep.py --mode power --n-total 200 --effect-points 6 --sd-change 7 --prop-twins 0.9 --prop-mz 0.5 --icc-mz 0.5 --icc-dz 0.25 --alpha 0.05 --sims 2000`
- Find `N` for a target power:
  - `python power_twin_sleep.py --mode n-for-power --target-power 0.9 --effect-points 6 --sd-change 7 --prop-twins 0.9 --prop-mz 0.5 --icc-mz 0.5 --icc-dz 0.25 --alpha 0.05 --sims 1500`

Optional flags:
- `--analysis {cluster_robust,mixedlm}` (default `cluster_robust`)
- `--seed 12345` (reproducibility)
- `--contamination-rate` and `--contamination-effect` to attenuate the observed effect: `effect_obs = effect_points × (1 − rate × effect)`
- `--attrition-rate` to inflate enrollment counts in the printed output

## Parameters (brief)

- Effect size (ISI points): Difference in ISI change (LLM – control). MCID ≈ 6.
- SD of ISI change: Typical range ~5–7; adjust to your context.
- Proportion twins: Fraction of all participants who are twins; rest are singletons.
- Proportion MZ: Among twin pairs, fraction who are monozygotic; rest DZ.
- ICC MZ / ICC DZ: Within-pair correlations in ISI change for MZ/DZ.
- Alpha: Two-sided test level (default 0.05).
- Simulations: Monte Carlo iterations; higher → slower but tighter estimates.
- Analysis method: `cluster_robust` (OLS+cluster SE) or `mixedlm` (random intercept).

Note on effect direction: The simulation treats beneficial effects as more negative ISI change in treatment than control. You specify a positive `--effect-points` magnitude; the code applies the sign so reported treatment-effect estimates are negative when beneficial.

Tip: If you plan baseline-adjusted ANCOVA on post-ISI, you can enter a smaller `SD(change)` reflecting pre/post correlation. A helper is included:
- `sd_change_from_pre_post(sd_pre, sd_post, rho)` in `power_twin_sleep.py`.

## Outputs

- Estimated power (for fixed N) or required N (for target power).
- Approximate effective N (Kish) for a quick sanity check.
- Average estimated treatment effect from simulations.
- If contamination/attrition flags are set: observed effect after contamination and required enrollment after attrition.

## Assumptions

- Individual randomization with co-twins randomized independently.
- Outcome model: change = treatment effect + pair random effect + residual.
- Pair effect variance set to match the specified ICC by zygosity; singletons have no pair effect.
- Zygosity included as fixed effects in analysis.
- Normality assumptions; no informative dropout modeled.

## Troubleshooting

- ImportError for `numpy`/`pandas`/`streamlit`/`statsmodels`/`scipy`:
  - Ensure your virtualenv is active and run `pip install -r requirements.txt`.
- `statsmodels` not installed or MixedLM convergence issues:
  - Use `analysis=cluster_robust` in the app, or the script will automatically fall back to an analytical GLS.
- Streamlit not found:
  - `pip install streamlit` then run `streamlit run streamlit_app.py`.
- Port already in use:
  - `streamlit run streamlit_app.py --server.port 8502` (or another port).

## Reproducibility

- Set a fixed RNG seed in the app or via `--seed` on the CLI.
- Increase `--sims` (or the app slider) to reduce Monte Carlo error for final estimates.

## Notes

- The approximate effective N shown is a conservative sanity check and ignores across-arm co-twin correlation created by independent randomization.
- Explore sensitivity: vary ICCs (e.g., MZ 0.4–0.6; DZ 0.2–0.3) and `SD(change)` (e.g., 5–7).
