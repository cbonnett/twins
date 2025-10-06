# Twin-Aware Power — Sleep / ISI Module

Monte Carlo power analysis for an individually randomized insomnia trial that includes a large
fraction of twins. The module models zygosity-specific ICCs, contamination, attrition, and optional
MixedLM analysis so protocol teams can evaluate design trade-offs for Insomnia Severity Index (ISI)
change scores.

## Files

- `power_twin_sleep.py` – Monte Carlo simulator + CLI.
- `raw_sleep_design.md` – additional design notes.
- `README.md` – this guide.

## Installation

Use the shared repository requirements so the biological age and sleep modules stay in sync:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r ../requirements.txt
```

## Using the CLI

Run the script from this directory (`python power_twin_sleep.py`) or from the repository root
(`python sleep/power_twin_sleep.py`). Available modes:

- `power` – estimate power for a fixed total sample size (completers).
- `n-for-power` – solve for the smallest total `N` that meets a target power.

Example invocations:

```bash
# Power at N=220 with heavy twin composition and cluster-robust analysis
python sleep/power_twin_sleep.py --mode power --n-total 220 --effect-points 6 --sd-change 7 \
    --prop-twins 0.9 --prop-mz 0.5 --icc-mz 0.5 --icc-dz 0.25 --analysis cluster_robust \
    --contamination-rate 0.30 --contamination-effect 0.50 --attrition-rate 0.10 --sims 2000

# Required N for 90% power under MixedLM analysis
python sleep/power_twin_sleep.py --mode n-for-power --target-power 0.90 --effect-points 6 \
    --sd-change 7 --prop-twins 0.9 --prop-mz 0.5 --icc-mz 0.5 --icc-dz 0.25 \
    --analysis mixedlm --sims 1500
```

Helpful flags:

- `--analysis {cluster_robust,mixedlm}` – choose the estimator; MixedLM falls back to cluster-robust
  OLS if fitting fails.
- `--seed` – reproducible Monte Carlo draws.
- `--contamination-rate` / `--contamination-effect` – attenuate the observed treatment effect.
- `--attrition-rate` – compute inflated enrollment counts; power is always on completers.
- `--rho-pre-post`, `--sd-pre`, `--sd-post` – derive SD(change) via `sd_change_from_pre_post` helper.

## Parameter Guidance

- **Effect (ISI points).** Specify the beneficial magnitude (positive numbers mean larger reductions
  for treated participants). The simulation applies the sign internally.
- **SD of change.** Provide the change-score SD or derive it from pre/post SDs and their correlation.
- **Twin mix.** `--prop-twins` controls the fraction of participants who are twins; `--prop-mz` is the
  MZ share among twins.
- **ICCs.** Separate ICCs by zygosity. Singletons implicitly have ICC=0.
- **Alpha.** Two-sided alpha defaults to 0.05; override with `--alpha` if multiplicity adjustments
  are required.
- **Monte Carlo precision.** Standard error ≈ `sqrt(p·(1−p)/sims)`; increase `--sims` for decision
  points near the power threshold.

## Outputs

The CLI reports estimated power (or required N), 95% Monte Carlo confidence intervals, Monte Carlo
standard errors, contamination-adjusted effect sizes, and attrition-inflated enrollment counts. MixedLM
runs include convergence diagnostics; failures automatically fall back to cluster-robust OLS.

## Streamlit UI

`sleep` is exposed via the Sleep tab in the root `streamlit_app.py`. Launch the app with
`streamlit run streamlit_app.py`.

## Testing

Module-specific regression tests live in `tests/test_sleep_study.py`. Run them from the repository
root:

```bash
pytest tests/test_sleep_study.py -q
```
