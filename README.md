# Twin-Aware Power — Unified Streamlit App

This repository contains a single Streamlit app that lets you run power and sample‑size calculations for two studies in one place:

- Biological Age (within‑pair randomized twin RCT; DunedinPACE and GrimAge)
- Sleep (ISI change; individually randomized with a high proportion of twins)

The unified app is a thin UI over the existing analysis modules:

- `biologicol_age/power_twin_age.py` (paired t analytic + Monte Carlo; co‑primary support)
- `sleep/power_twin_sleep.py` (individual randomization with twin clustering; cluster‑robust or MixedLM)


## Quick Start

- Create/activate a virtual environment (recommended)
- Install dependencies (both subprojects share the same stack):
  - `pip install -r biologicol_age/requirements.txt -r sleep/requirements.txt`
- Launch the app from the repo root:
  - `streamlit run streamlit_app.py`

In the sidebar, choose the study (Biological Age or Sleep), set parameters, and select a goal (estimate power, find pairs/N for a target power, find MDE, or co‑primary power for the bio‑age panel).


## What You Can Do

- Biological Age panel
  - Endpoints: DunedinPACE (absolute slowing; e.g., 0.03 = 3%), GrimAge (years), or custom
  - Modes: estimate power (fixed pairs), find pairs for a target power, find MDE, co‑primary joint power (DunedinPACE + GrimAge)
  - Parameters: SD(change), ICCs for MZ/DZ, proportion MZ, contamination (rate × effect fraction), attrition (enrollment inflation), alpha
  - Engines: paired‑t analytic with robust normal fallback; Monte Carlo simulation (recommended for final numbers or co‑primary)

- Sleep panel (ISI)
  - Modes: estimate power for fixed N (individuals), find N for target power
  - Parameters: effect (ISI points), SD(change), proportion twins, proportion MZ among twins, ICCs for MZ/DZ, contamination, attrition, alpha
  - Engines: simulation with cluster‑robust OLS by default (MixedLM available in core module)


## Tips and Defaults

- Co‑primary (bio‑age): when global alpha is 0.05, the app uses 0.025 per endpoint (Bonferroni) in the co‑primary panel.
- Simulation: increase `sims` (e.g., 3000–5000) for publication‑quality precision; Monte Carlo SE ≈ sqrt(p·(1−p)/sims).
- Seeds: set a fixed seed to reproduce estimates; change seeds to gauge Monte Carlo variability.
- Attrition: results are for completing pairs/individuals; the app reports inflated enrollment counts.
- Contamination: observed effect = specified effect × (1 − rate × fraction).


## Files of Interest

- Unified app entrypoint: `streamlit_app.py`
- Bio‑age module + CLI: `biologicol_age/power_twin_age.py`
- Bio‑age Streamlit app (standalone): `biologicol_age/streamlit_app.py`
- Sleep module + CLI: `sleep/power_twin_sleep.py`
- Sleep Streamlit app (standalone): `sleep/streamlit_app.py`

For deeper background and CLI examples, see:
- `biologicol_age/README.md`
- `sleep/README.md`


## Troubleshooting

- Missing packages: ensure your environment is active; run the `pip install` command above.
- SciPy/statsmodels issues: the app falls back to stable approximations where possible; increase sims for simulation‑based results.
- Streamlit not found: `pip install streamlit`, then `streamlit run streamlit_app.py`.

