# Twin-Aware Power — Biological Age (Within-Pair RCT)

This folder provides analytic and simulation tools for power and sample-size planning in a within-pair randomized twin RCT targeting biological age reversal. It supports DunedinPACE (pace-of-aging), GrimAge (years), and custom endpoints, and models MZ/DZ correlations directly.

**Files**
- `streamlit_app.py` — Interactive Streamlit UI (paired design with optional simulation).
- `power_twin_age.py` — CLI tool and core methods (analytic paired t-test + Monte Carlo simulation; contamination and attrition support; co-primary endpoints).
- `requirements.txt` — Python dependencies.

**What it models**
- Within-pair randomization (one twin treated, co‑twin control).
- Endpoint options: `dunedinpace`, `grimage`, or `custom` (absolute effect scale).
- Separate ICCs for MZ and DZ pairs; mixes via variance‑weighted effective ICC.
- Contamination (control twins partially adopting intervention) and attrition.

## Quick Start

- Create a virtualenv and install deps:
  - `python -m venv .venv && source .venv/bin/activate`
  - `pip install -r requirements.txt`
- Launch the app:
  - `streamlit run streamlit_app.py`

Use the UI to:
- Choose endpoint and effect input (absolute or paired standardized `d`).
- Select timepoint label (Week 12 interim, Week 24 primary, Month 12 follow‑up).
- Set SD of individual change and ICCs for MZ/DZ; set proportion MZ.
- Model contamination (rate and fraction) and view observed effect attenuation.
- Plan for attrition with enrollment inflation hints.
- Pick goal: estimate power (fixed pairs), find pairs for target power, or find MDE (with and without contamination).
- Optionally enable simulation for higher fidelity with heterogeneous ICCs.
- Compute co‑primary joint power (DunedinPACE + GrimAge) with Bonferroni alpha (0.025 per endpoint) and adjustable cross‑endpoint pair‑effect correlation.

## Quick Start (uv)

Use `uv` for fast installs and runs.

- Install uv (macOS/Linux):
  - `curl -LsSf https://astral.sh/uv/install.sh | sh`
  - Or via Homebrew: `brew install uv`
- Create and activate a virtual environment:
  - `uv venv .venv`
  - macOS/Linux: `source .venv/bin/activate`
- Install dependencies:
  - `uv pip install -r requirements.txt`
- Run the Streamlit app (activation optional with uv run):
  - ` tooltips `

## Command-Line (CLI)

Core modes are implemented in `power_twin_age.py`:
- Power for fixed pairs: `--mode power`
- Pairs for target power: `--mode pairs-for-power`
- Minimal detectable effect: `--mode mde`
- Power curve CSV: `--mode curve`
- Co‑primary endpoints (joint power): `--mode co-primary-power`

Examples:
- Power at 700 completing pairs, DunedinPACE 3% slowing, ICC=0.55:
  - `python3 power_twin_age.py --mode power --n-pairs 700 --endpoint dunedinpace --effect-pct 3 --sd-change 0.10 --icc-mz 0.55 --icc-dz 0.55 --prop-mz 0.5`
- Required pairs for 90% power, GrimAge −2.0 years, with 40% attrition:
  - `python3 power_twin_age.py --mode pairs-for-power --target-power 0.90 --endpoint grimage --effect-years 2.0 --sd-change 3.0 --icc-mz 0.6 --icc-dz 0.3 --prop-mz 0.5 --attrition-rate 0.40`
- MDE at 700 pairs (analytic):
  - `python3 power_twin_age.py --mode mde --n-pairs 700 --endpoint dunedinpace --sd-change 0.10 --icc-mz 0.55 --icc-dz 0.55 --target-power 0.80`
- Use simulation (recommended for final calcs or co‑primary):
  - Add `--use-simulation --sims 3000`
- Co‑primary joint power (DunedinPACE + GrimAge) with Bonferroni alpha 0.025:
  - `python3 power_twin_age.py --mode co-primary-power --n-pairs 700 --alpha 0.025 --endpoint dunedinpace --effect-pct 3 --sd-change 0.10 --endpoint2 grimage --effect2-years 2.0 --sd2-change 3.0 --use-simulation --sims 5000`
  - Optional: correlate pair effects across endpoints (default 0.8): add `--pair-effect-corr 0.8`
  - Endpoint-2 flags accept both styles: `--effect2-years`/`--sd2-change` and `--effect-years2`/`--sd-change2`.

Common options:
- `--icc-mz`, `--icc-dz`, `--prop-mz` — twin correlation structure.
- `--contamination-rate`, `--contamination-effect` — observed effect = specified effect × (1 − rate × effect).
- `--attrition-rate` — inflates required enrollment; power is for completing pairs.
- `--seed`, `--sims` — reproducibility and Monte Carlo precision.

## Parameter Guidance (protocol‑aligned)

- Effect (absolute):
  - DunedinPACE (pace-of-aging): enter absolute slowing. Example: `0.03` = 3% slower pace (beneficial). Protocol target at Week 24 is 2–3% slowing; use 0.02–0.03 for planning and sensitivity.
  - GrimAge (years): enter absolute reduction in years. Example: `2.0` = 2 years younger (beneficial). Treat as exploratory; CALERIE showed null GrimAge response while DunedinPACE responded.
  - Custom: enter the beneficial magnitude in your endpoint’s native units. Internally, treated change is reduced by this amount, so treated − control paired difference is negative.

- Standardized effect `d` (paired):
  - Use `--d-std` or the UI’s “standardized d” input when you prefer unit‑less comparisons across endpoints.
  - Definition: `d = effect_abs / SD(pair_diff)` with `SD(pair_diff) = sqrt(2 · (1 − ICC) · sd_change^2)`.
  - Interpretation: In paired designs, even small `d` can be detectable at moderate N if ICC is high (reduces within‑pair noise).

- SD of individual change (`sd_change`):
  - DunedinPACE: typical `0.08–0.12` for short trials; use the helper `sd_change_from_pre_post(sd_pre, sd_post, rho)` if you have pre/post SDs and correlation.
  - GrimAge: typical `2.0–4.0` years for short trials (varies by cohort and lab processing).
  - If using ANCOVA on post values with baseline as a covariate, the implied SD(change) is smaller than unadjusted; reflect your planned analysis.

- ICCs and zygosity mix:
  - ICC within MZ pairs is usually higher than within DZ (e.g., MZ `0.5–0.7`, DZ `0.3–0.6`).
  - Effective ICC with an MZ/DZ mix is approximately `ICC_eff = 1 − [prop_mz·(1−ICC_MZ) + (1−prop_mz)·(1−ICC_DZ)]`.
  - Higher ICC → smaller `SD(pair_diff)` → higher power at fixed pairs.

- Alpha and co‑primary endpoints:
  - Use `alpha = 0.05` for a single primary endpoint.
  - For two co‑primary endpoints (DunedinPACE + GrimAge), use Bonferroni `alpha = 0.025` per endpoint. The Streamlit app’s co‑primary panel defaults to 0.025 when the global alpha is 0.05; the CLI auto‑sets 0.025 in co‑primary mode unless overridden.

- Simulation controls:
  - `sims`: Monte Carlo iterations. Monte Carlo SE for a power estimate `p` is `sqrt(p·(1−p)/sims)`. Example: at `p=0.85`, `sims=2000` → SE ≈ 0.0087 (≈±1.7% at 95% CI).
  - `seed`: Fix to compare scenarios reproducibly.

## Reasonable Defaults and Ranges

- DunedinPACE example defaults
  - Effect: `0.03` (3% slowing; planning range 0.02–0.03)
  - SD(change): `0.10` (typical 0.08–0.12)
  - ICCs: `ICC_MZ=0.55`, `ICC_DZ=0.55` (swap DZ to 0.30–0.55 for sensitivity)
  - Mix: `prop_mz=0.5`
  - Derived: `SD(pair_diff) = sqrt(2·(1−ICC)·sd_change^2)` = `sqrt(2·0.45·0.10^2)` ≈ `0.095`; `d ≈ 0.03 / 0.095 ≈ 0.32`.

- GrimAge example defaults
  - Effect: `2.0` years (conservative 1.0–2.0; optimistic 2.0–3.0)
  - SD(change): `3.0` years (typical 2.0–4.0)
  - ICCs: `ICC_MZ=0.60`, `ICC_DZ=0.30` (sensitivity 0.3–0.6)
  - Mix: `prop_mz=0.5`
  - Derived: `SD(pair_diff) = sqrt(2·(1−0.45)·3^2)` = `sqrt(2·0.55·9)` ≈ `3.15`; `d ≈ 2.0 / 3.15 ≈ 0.63`.

Tip: If `d` seems implausibly large/small, revisit `sd_change` and `ICC` — they directly determine `SD(pair_diff)` and therefore standardized magnitude.

## Outputs

- Power or required pairs (with CI if simulated), MDE, and power curves.
- Effective ICC, SD of within‑pair difference, and paired Cohen’s d.
- Enrollment counts when `--attrition-rate` > 0.

## Notes

- Internally, beneficial effects reduce the treated change by `effect_abs` (treated − control is negative). Reported effects are positive magnitudes for clarity.
- Contamination attenuates the observed effect: `effect_obs = effect_abs × (1 − rate × effect)`.

### Contamination and Attrition Examples

- Contamination attenuation:
  - If 30% of controls adopt 50% of the intervention effect, the observed effect becomes `0.85 ×` the specified absolute effect (`1 − 0.30·0.50 = 0.85`).
  - Example: specified `0.03` → observed `0.0255` (DunedinPACE) or specified `2.0` → observed `1.7` years (GrimAge).

- Attrition inflation:
  - Required enrollment for `n_completing` pairs at attrition `a` is `ceil(n_completing / (1 − a))`.
  - Example: need 700 completing pairs at 40% attrition → enroll `ceil(700/0.6) = 1167` pairs (2334 individuals).

## Troubleshooting

- Missing packages: ensure your virtualenv is active; run `pip install -r requirements.txt`.
- SciPy/statsmodels unavailable: analytic mode and paired tests still run; simulation does not require MixedLM.
- Streamlit not found: `pip install streamlit`; then `streamlit run streamlit_app.py`.

## Reproducibility

- Set `--seed` (CLI) or the UI seed. Increase `--sims` for tighter Monte Carlo precision.
