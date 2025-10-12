# Twin-Aware Power — Biological Age Module

Analytic and simulation tooling for a within-pair randomized twin RCT targeting biological age
slowing. The module supports DunedinPACE (percent slowing), GrimAge (years), and arbitrary custom
change scores with contamination, attrition, and co-primary endpoint options.

## Files

- `power_twin_age.py` – analytic + simulation engines and CLI entry point.
- `README.md` – this guide.

## Installation

Install dependencies from the repository root (see `requirements.txt`). A typical setup is:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r ../requirements.txt
```

## Using the CLI

Run the script either from this folder (`python power_twin_age.py`) or from the repository root
(`python biological_age/power_twin_age.py`). Core modes:

- `power` – estimate power for a fixed number of completing pairs.
- `pairs-for-power` – solve for the smallest number of pairs that achieves a target power.
- `mde` – minimal detectable effect at a target power.
- `curve` – emit a CSV power curve over a range of pair counts.
- `co-primary-power` – joint power for two endpoints with optional Monte Carlo simulation.
- `study` – preset summary for the LLM multi-domain twin RCT (DunedinPACE primary) with a small grid over
  effect sizes and completing pairs, plus enrollment inflation for an attrition range.
- `two-sample-power` – individual-level (parallel groups) power on change scores with optional ANCOVA variance reduction.
- `two-sample-n-for-power` – individual-level n per group for target power; optional design-effect (DEFF) reporting.

Example invocations:

```bash
# Analytic power at 700 completing pairs for DunedinPACE (effect specified as percent slowing)
python biological_age/power_twin_age.py --mode power --n-pairs 700 --endpoint dunedinpace \
    --effect-pct 3 --sd-change 0.10 --icc-mz 0.55 --icc-dz 0.55 --prop-mz 0.5

# Required pairs for 90% power on GrimAge (years) with attrition and contamination planning
python biological_age/power_twin_age.py --mode pairs-for-power --target-power 0.90 --endpoint grimage \
    --effect-years 2.0 --sd-change 3.0 --icc-mz 0.60 --icc-dz 0.30 --prop-mz 0.5 \
    --attrition-rate 0.40 --contamination-rate 0.30 --contamination-effect 0.50

# Co-primary joint power for DunedinPACE and GrimAge via simulation
python biological_age/power_twin_age.py --mode co-primary-power --n-pairs 700 --endpoint dunedinpace \
    --effect-pct 3 --sd-change 0.10 --endpoint2 grimage --effect2-years 2.0 --sd2-change 3.0 \
    --pair-effect-corr 0.8 --use-simulation --sims 5000

# Study preset: LLM multi-domain twin RCT (DunedinPACE primary)
# Quick summary across 2–3% slowing and 150/165/185 completing pairs, with 25–30% attrition range
python biological_age/power_twin_age.py --mode study --study-preset twins-llm \
    --use-simulation --sims 3000 \
    --contamination-rate 0.10 --contamination-effect 0.30

# Override defaults (e.g., different grids, ICCs)
python biological_age/power_twin_age.py --mode study --study-preset twins-llm \
    --study-effect-pct "2,2.5,3.0,3.5" --study-n-pairs "150,165,185,200" \
    --icc-mz 0.60 --icc-dz 0.50 --prop-mz 0.5

# Individual-level (two-sample) power – DunedinPACE primary
# Power at fixed N using ANCOVA variance reduction (R^2=0.50)
python biological_age/power_twin_age.py --mode two-sample-power \
    --n-per-group 150 --endpoint dunedinpace --effect-pct 2.5 --sd-change 0.10 \
    --ancova-r2 0.50 --alpha 0.05 --attrition-rate 0.27 --deff-icc 0.50 --deff-m 2

# n per group for 80% power (ANCOVA R^2=0.50)
python biological_age/power_twin_age.py --mode two-sample-n-for-power \
    --endpoint dunedinpace --effect-pct 2.5 --sd-change 0.10 --ancova-r2 0.50 \
    --alpha 0.05 --target-power 0.80 --deff-icc 0.50 --deff-m 2 --attrition-rate 0.27
```

Helpful flags:

- `--use-simulation --sims N` – run Monte Carlo instead of analytic formulas; required for
  co-primary joint power and heterogeneous ICC scenarios.
- `--d-std` – specify a standardized paired difference (Cohen's d) instead of absolute magnitude.
- `--seed` – make Monte Carlo runs reproducible.
- `--contamination-rate` / `--contamination-effect` – attenuate the true effect for ITT analysis.
- `--attrition-rate` – report inflated enrollment alongside completing pairs.

## Parameter Guidance

- **Effect magnitude.** Use `--effect-pct`, `--effect-years`, or `--effect-abs` to set the beneficial
  change on the outcome scale. Internally the treated twin is reduced by this amount so paired
  differences are negative; summaries report positive magnitudes for clarity.
- **SD of change.** Supply `--sd-change` directly or derive it via
  `--sd-pre`, `--sd-post`, and `--rho-pre-post` (see helper function `sd_change_from_pre_post`).
- **ICCs.** Provide zygosity-specific ICCs (`--icc-mz`, `--icc-dz`) and the expected MZ mix
  (`--prop-mz`). Analytic mode converts these to an effective ICC; simulation draws the mix explicitly.
- **Alpha / co-primary.** Default alpha is 0.05. In co-primary mode, the script applies Bonferroni
  (0.025 per endpoint) unless `--alpha` is overridden.
- **Monte Carlo precision.** Standard error ≈ `sqrt(p·(1−p)/sims)`. Increase `--sims` for final
  estimates or when power is near the decision threshold.

## Outputs

Runs report power (or required pairs/MDE), contamination-adjusted effects, effective ICCs, attrition
inflation, and Monte Carlo standard errors when simulation is active. Co-primary runs also show
per-endpoint alpha and marginal power alongside the joint probability of success.

## Streamlit UI

The top-level `streamlit_app.py` includes a Biological Age tab that wraps this module. Start it from
the repository root with `streamlit run streamlit_app.py`.

## Testing

Regression tests for this module live in `tests/test_power_twin_age.py`. Execute them from the
repository root:

```bash
pytest tests/test_power_twin_age.py -q
```
