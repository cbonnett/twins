"""
Streamlit app: Twin-aware power analysis for SIESTA-LLM (ISI change).

Adjust trial parameters (effect size, SD, twin mix, MZ/DZ ICCs, alpha, sims),
and estimate power for a fixed N or find the minimum N that achieves a target
power. Explanations of parameters and key assumptions are included inline.
"""

from __future__ import annotations

import time
import math
from typing import Optional

import streamlit as st
import numpy as np
import pandas as pd

try:
    from power_twin_sleep import (
        TwinTrialSpec,
        simulate_power,
        simulate_distribution,
        power_curve,
        binomial_wilson_ci,
        find_n_for_power,
        approximate_effective_n,
        sd_change_from_pre_post,
        inflate_for_attrition,
    )
except Exception as e:  # pragma: no cover
    st.error(
        "Could not import power functions. Please ensure `power_twin_sleep.py` is present "
        "in the same directory and its dependencies are installed.\n\n"
        f"Import error: {e}"
    )
    st.stop()


st.set_page_config(page_title="Twin-Aware Power Analysis (ISI)", layout="centered")
st.title("Twin-Aware Power Analysis — SIESTA-LLM (ISI Change)")
st.caption("Interactive simulation accounting for MZ/DZ co-twin correlations and individual randomization")

with st.expander("What this tool does"):
    st.markdown(
        "- Estimates statistical power for the primary endpoint (ISI change to Week 8) in an individually randomized RCT with a large fraction of twins.\n"
        "- Models within-pair correlation separately for MZ and DZ twins.\n"
        "- Supports a mix of twins and singletons, and stratifies by zygosity via covariates in the analysis.\n"
        "- Uses cluster-robust OLS by default, MixedLM if selected, or an analytical GLS fallback if those libraries are unavailable.\n"
        "- Can either: (1) estimate power for a fixed total N, or (2) search for the minimum N to reach a target power." 
    )

st.divider()

# Sidebar: global settings
st.sidebar.header("Global Settings")

# Inline hover help (expanded, highly detailed)
HELP = {
    "mode": (
        "Choose your objective:\n"
        "- Estimate power for a fixed total N (individuals) to check feasibility of your current plan.\n"
        "- Find N for a target power to plan recruitment and budget.\n"
        "Tip: You can iterate between modes to hone in on design trade‑offs."
    ),
    "alpha": (
        "Two-sided significance level (Type I error rate). Common: 0.05 for a single primary endpoint.\n"
        "If you will test multiple co‑primaries with Bonferroni, use 0.025 per endpoint.\n"
        "Lower alpha reduces false positives but increases required N. Align with your multiplicity plan."
    ),
    "sims": (
        "Number of Monte Carlo iterations. More sims → smaller Monte Carlo noise, longer runtime.\n"
        "Monte Carlo standard error (SE) for an estimated power p is ≈ sqrt(p·(1−p)/sims).\n"
        "Examples: p=0.85, sims=2000 → SE≈0.009 (±1.8% at 95% CI); sims=5000 → SE≈0.006 (±1.2%).\n"
        "Recommendation: Use ≥3000 for final reported figures."
    ),
    "seed": (
        "Random seed ensures reproducibility. With the same seed, you should see nearly identical results.\n"
        "Vary the seed to validate robustness to Monte Carlo noise. The seed does not affect non‑simulation fallbacks."
    ),
    "analysis": (
        "Analysis method for the simulated datasets:\n"
        "- cluster_robust: OLS with cluster‑robust SEs by pair (recommended for speed/robustness).\n"
        "- mixedlm: Random‑intercept MixedLM by pair (slower; similar inference if model converges).\n"
        "If libraries are missing or models fail, the code falls back to a GLS with known covariance."
    ),
    "effect_points": (
        "Effect magnitude in ISI points at Week 8. Interpretation: how many points more negative the treatment change is compared to control.\n"
        "Example: effect=6 means treatment improves 6 points more than control (MCID≈6)."
    ),
    "sd_change": (
        "SD of individual ISI change (Week 8 − baseline). If you plan an ANCOVA on post‑ISI, you can derive a smaller SD(change) using the pre/post SDs and their correlation ρ.\n"
        "Use realistic SDs/ρ from prior data to avoid over‑optimistic power."
    ),
    "prop_twins": (
        "Proportion of all participants who are twins (appear in pairs). Remaining participants are singletons.\n"
        "This affects clustering and the precision of estimates."
    ),
    "prop_mz": (
        "Among twin pairs, fraction that are monozygotic (MZ). The remainder are dizygotic (DZ).\n"
        "Higher MZ fraction typically increases within‑pair correlation."
    ),
    "icc_mz": (
        "Within‑pair ICC for change among MZ twins. Larger ICC generally increases power for a fixed N under individual randomization with clustering."
    ),
    "icc_dz": (
        "Within‑pair ICC for change among DZ twins. MZ and DZ ICCs need not be equal."
    ),
    "contamination_rate": (
        "Proportion of controls expected to adopt intervention‑like behaviors (0–1)."
    ),
    "contamination_effect": (
        "Fraction of full effect those contaminated controls receive (0–1). Observed effect = effect × (1 − rate × fraction).\n"
        "Example: rate=0.30, fraction=0.50 → observed effect = 0.85 × specified effect."
    ),
    "attrition": (
        "Expected dropout rate (0–1). Power is reported for completing individuals.\n"
        "Enrollment should be inflated by 1/(1−attrition) to maintain the target completing N."
    ),
    "n_total": (
        "Total number of randomized individuals contributing to the primary analysis (post‑attrition)."
    ),
    "target_power": (
        "Desired statistical power, typically 0.80–0.95. Higher targets require larger N."
    ),
    "n_min": (
        "Lower bound for the search grid (N‑for‑power). Choose low enough to include the underpowered region for context."
    ),
    "n_max": (
        "Upper bound for the search grid (N‑for‑power). Ensure it’s high enough to include the adequately powered region."
    ),
    "sd_prepost": (
        "Compute SD(change) from SD(pre), SD(post), and their correlation ρ via Var(change)=Var(pre)+Var(post)−2ρ·sd_pre·sd_post.\n"
        "This mirrors baseline adjustment when pre/post are correlated."
    ),
}

mode = st.sidebar.radio(
    "Mode",
    ("Estimate power for fixed N", "Find N for target power"),
    index=0,
    help=HELP["mode"],
)

alpha = st.sidebar.number_input("Alpha (two-sided)", min_value=0.0001, max_value=0.2, value=0.05, step=0.005, format="%.3f", help=HELP["alpha"])
sims = st.sidebar.number_input("Monte Carlo simulations", min_value=200, max_value=20000, value=2000, step=200, help=HELP["sims"])
analysis = st.sidebar.selectbox(
    "Analysis method",
    options=["cluster_robust", "mixedlm"],
    index=0,
    help=HELP["analysis"],
)
seed = st.sidebar.number_input("Random seed", min_value=0, max_value=10**9, value=12345, step=1, help=HELP["seed"])

st.sidebar.caption(
    "Tips:\n"
    "- Increase simulations for tighter precision (use ≥3000 for final estimates).\n"
    "- Results vary slightly due to Monte Carlo error; set a fixed seed to compare scenarios reproducibly.\n"
    "- Consider sensitivity checks on ICCs, effect, and SD(change)."
)


# Main parameter controls
st.subheader("Treatment Effect and Outcome Variability")
col1, col2 = st.columns(2)
with col1:
    effect_points = st.number_input(
        "Effect size (ISI points, LLM vs control)",
        min_value=0.0, max_value=20.0, value=6.0, step=0.5,
        help=(
            "Difference in ISI change at Week 8 (treatment − control) in points. "
            "Set to the MCID (~6) to test a clinically meaningful effect. "
            "Internally, beneficial effects are applied as more negative change in treatment, "
            "so estimated coefficients are negative when the intervention helps."
        ),
    )
with col2:
    sd_change = st.number_input(
        "SD of ISI change",
        min_value=0.5, max_value=20.0, value=7.0, step=0.5,
        help=(
            "SD of individual change (Week 8 − baseline). If you plan ANCOVA on post-ISI, use a smaller SD(change) "
            "reflecting pre/post correlation. Use the helper below to derive SD(change) from SD(pre), SD(post), and ρ."
        ),
    )

with st.expander("Compute change SD from pre/post SDs and correlation (optional)"):
    c1, c2, c3 = st.columns(3)
    with c1:
        sd_pre = st.number_input("Baseline SD", min_value=0.5, max_value=30.0, value=7.0, step=0.5, key="sd_pre")
    with c2:
        sd_post = st.number_input("Week 8 SD", min_value=0.5, max_value=30.0, value=7.0, step=0.5, key="sd_post")
    with c3:
        rho = st.number_input("Corr(pre, post)", min_value=-0.99, max_value=0.99, value=0.5, step=0.05, format="%.2f")
    if st.button(
        "Use computed SD(change)",
        help=(
            "Derive SD(change) from SD(pre), SD(post), and their correlation ρ using the standard variance identity. "
            "Clicking this will overwrite the SD(change) input above with the computed value. "
            "This approximation mirrors baseline adjustment when pre/post are correlated. "
            "Use a realistic ρ from prior data to avoid overstating precision."
        ),
    ):
        sd_change = sd_change_from_pre_post(sd_pre, sd_post, rho)
        st.success(f"Updated SD(change) = {sd_change:.3f}")

with st.expander("Contamination modeling (optional)"):
    c1, c2 = st.columns(2)
    with c1:
        contamination_rate = st.number_input(
            "Controls adopting intervention (rate)",
            min_value=0.0, max_value=1.0, value=0.0, step=0.05, format="%.2f",
            help=(
                "Proportion of controls expected to adopt intervention-like behaviors (0–1). "
                "Observed effect = effect_points × (1 − rate × fraction). "
                "Higher contamination dilutes between-arm differences and reduces power."
            ),
        )
    with c2:
        contamination_effect = st.number_input(
            "Fraction of full effect in contaminated controls",
            min_value=0.0, max_value=1.0, value=0.0, step=0.05, format="%.2f",
            help=(
                "If controls contaminate, fraction of the treatment effect they receive (0–1). "
                "Example: rate=0.30, fraction=0.50 → observed effect = 0.85 × specified effect. "
                "Use sensitivity analysis across plausible contamination scenarios."
            ),
        )

st.subheader("Twin Structure and Correlations")
col3, col4, col5 = st.columns(3)
with col3:
    prop_twins = st.number_input(
        "Proportion twins in sample",
        min_value=0.0, max_value=1.0, value=0.9, step=0.05, format="%.2f",
        help=(
            "Fraction of all participants who are twins (pairs of 2). The remainder are singletons. "
            "Typical values depend on recruitment; in a twin-focused trial this can be 0.7–0.95. "
            "Affects clustering and the approximate effective N (Kish)."
        ),
    )

# Practical guidance, sanity checks, and reporting notes
with st.expander("Guidance • Assumptions • Sanity checks • Reporting", expanded=False):
    st.markdown(
        "- Under the null (effect≈0), estimated power should be near α.\n"
        "- Power should increase with larger effects, larger N, and higher ICC (within reasonable ranges).\n"
        "- Attrition inflates enrollment only; power pertains to completing individuals.\n"
        "- Monte Carlo precision: report the number of simulations and consider adding binomial CIs for power estimates.\n"
        "- Sensitivity checklist: vary ICC_MZ/DZ, proportion of twins, effect, SD(change), and contamination.\n"
        "- Analysis notes: cluster_robust is faster and robust; MixedLM requires convergence checks.\n"
        "- Document seeds and sims used to facilitate reproducibility in protocols and reports."
    )
with col4:
    prop_mz = st.number_input(
        "Proportion MZ among twin pairs",
        min_value=0.0, max_value=1.0, value=0.5, step=0.05, format="%.2f",
        help=(
            "Among twin pairs, fraction that are MZ (identical). Typical 0.4–0.6 depending on cohort. "
            "Effective ICC ≈ 1 − [prop_mz·(1−ICC_MZ) + (1−prop_mz)·(1−ICC_DZ)]."
        ),
    )
with col5:
    icc_mz = st.number_input(
        "ICC within MZ pairs",
        min_value=0.0, max_value=0.99, value=0.5, step=0.05, format="%.2f",
        help=(
            "Within-pair correlation for MZ twins on ISI change. Typical 0.4–0.7. "
            "Higher ICC reduces SD(pair-diff) = sqrt(2·(1−ICC)·sd_change^2) and increases power."
        ),
    )
icc_dz = st.number_input(
    "ICC within DZ pairs",
    min_value=0.0, max_value=0.99, value=0.25, step=0.05, format="%.2f",
    help=(
        "Within-pair correlation for DZ twins on ISI change (often 0.2–0.5, lower than MZ). "
        "Affects SD(pair-diff) via 1−ICC, similar to MZ. "
        "Vary DZ ICC in sensitivity sweeps because it materially changes effective N."
    ),
)

st.subheader("Design Choice")
if mode == "Estimate power for fixed N":
    n_total = st.number_input(
        "Total N (individuals)",
        min_value=20, max_value=10000, value=200, step=10,
        help=(
            "Total number of randomized completers (twins and singletons combined). "
            "Enrollment may need to be inflated for attrition (set below). "
            "This N is the denominator for power; it does not include those who drop out."
        ),
    )
    attrition_rate = st.number_input(
        "Attrition (dropout) rate",
        min_value=0.0, max_value=0.95, value=0.0, step=0.05, format="%.2f",
        help=(
            "Used to inflate required enrollment: required_enrollment = ceil(completers / (1 − attrition)). "
            "Power is computed on completing N and does not model informative dropout. "
            "Include buffers for screening failures and missing data as needed."
        ),
    )
else:
    target_power = st.number_input(
        "Target power",
        min_value=0.60, max_value=0.99, value=0.90, step=0.01, format="%.2f",
        help=(
            "Desired probability to detect the specified effect at the chosen alpha (1−β). "
            "Typical choices are 0.80 or 0.90; higher targets increase required N. "
            "Consider feasibility, endpoint variability, and effect size realism when choosing this value."
        ),
    )
    cmin, cmax = st.columns(2)
    with cmin:
        n_min = st.number_input(
            "Search N min", min_value=20, max_value=5000, value=80, step=10,
            help=(
                "Lower bound for the binary search over N when finding the minimum sample size. "
                "Set low enough to include underpowered regions for proper bracketing. "
                "Tighten bounds after an initial run to speed up subsequent searches."
            ),
        )
    with cmax:
        n_max = st.number_input(
            "Search N max", min_value=40, max_value=20000, value=2000, step=20,
            help=(
                "Upper bound for the binary search over N when finding the minimum sample size. "
                "Ensure this is high enough that the target power is achievable under conservative assumptions. "
                "If the algorithm reaches this bound, raise it and rerun."
            ),
        )
    attrition_rate = st.number_input(
        "Attrition (dropout) rate",
        min_value=0.0, max_value=0.95, value=0.0, step=0.05, format="%.2f",
        help=(
            "Used to inflate required enrollment: required_enrollment = ceil(N_req / (1 − attrition)). "
            "Power is computed on completing N_req."
        ),
    )


run = st.button(
    "Run power analysis",
    type="primary",
    help=(
        "Estimate power for a fixed N or find the minimum N for a target power. "
        "Monte Carlo simulation is used by default; analytical fallbacks may apply depending on settings. "
        "Inspect the reported Monte Carlo uncertainty and consider sensitivity sweeps for ICCs and SD(change)."
    ),
)

if run:
    spec = TwinTrialSpec(
        n_total=int(n_total if mode == "Estimate power for fixed N" else max(n_min, 20)),
        effect_points=float(effect_points),
        sd_change=float(sd_change),
        prop_twins=float(prop_twins),
        prop_mz=float(prop_mz),
        icc_mz=float(icc_mz),
        icc_dz=float(icc_dz),
        alpha=float(alpha),
        analysis=analysis,
        seed=int(seed),
        contamination_rate=float(contamination_rate),
        contamination_effect=float(contamination_effect),
        attrition_rate=float(attrition_rate),
    )

    start = time.time()
    error_msg: Optional[str] = None
    try:
        with st.spinner("Running simulations…"):
            if mode == "Estimate power for fixed N":
                pw, avg = simulate_power(spec, sims=int(sims))
                eff_n = approximate_effective_n(spec.n_total, spec.prop_twins, spec.prop_mz, spec.icc_mz, spec.icc_dz)
                n_enroll = inflate_for_attrition(int(spec.n_total), float(attrition_rate)) if attrition_rate > 0 else int(spec.n_total)
                effect_obs = spec.observed_effect()
            else:
                n_req, pw = find_n_for_power(float(target_power), spec, sims=int(sims), n_min=int(n_min), n_max=int(n_max))
                eff_n = approximate_effective_n(n_req, spec.prop_twins, spec.prop_mz, spec.icc_mz, spec.icc_dz)
                n_enroll = inflate_for_attrition(int(n_req), float(attrition_rate)) if attrition_rate > 0 else int(n_req)
                effect_obs = spec.observed_effect()
    except Exception as e:
        error_msg = str(e)
    elapsed = time.time() - start

    if error_msg:
        st.error(f"Error: {error_msg}")
        st.stop()
    else:
        st.success("Done.")
    if mode == "Estimate power for fixed N":
        # MC 95% CI via Wilson (approximate): successes ~ sims * pw
        k = int(round(pw * int(sims)))
        ci_low, ci_high = binomial_wilson_ci(k, int(sims), alpha=0.05)
        st.metric(label="Estimated power", value=f"{pw:.3f}")
        st.write(
            f"- Total N (completing): {spec.n_total}\n"
            + (f"- Expected attrition: {attrition_rate:.1%}\n- Required enrollment: {n_enroll}\n" if attrition_rate > 0 else "")
            + f"- Approx. effective N (Kish): {eff_n:.1f}\n"
            + f"- Specified effect: {effect_points:.2f} points; Observed (after contamination): {effect_obs:.2f}\n"
            + (f"  (contam rate={contamination_rate:.1%}, fraction={contamination_effect:.1%})\n" if contamination_rate > 0 else "")
            + f"- Avg estimated treatment effect: {avg:.2f} points\n"
            + f"- Sims: {sims}, alpha={alpha}, analysis={analysis}\n"
            + f"- MC 95% CI for power (Wilson): [{ci_low:.3f}, {ci_high:.3f}]"
        )
    else:
        st.metric(label="Required N (approx)", value=str(n_req))
        st.write(
            f"- Achieved power at N_req: {pw:.3f}\n"
            + (f"- With attrition {attrition_rate:.1%}, required enrollment: {n_enroll}\n" if attrition_rate > 0 else "")
            + (f"- Specified effect: {effect_points:.2f} points; Observed (after contamination): {effect_obs:.2f}\n" if contamination_rate > 0 else "")
            + f"- Approx. effective N (Kish): {eff_n:.1f}\n"
            + f"- Sims: {sims}, target power={target_power:.2f}, alpha={alpha}, analysis={analysis}"
        )

    st.caption(f"Runtime: {elapsed:.2f}s. Monte Carlo error decreases as simulations increase.")


with st.expander("Parameter definitions and guidance"):
    st.markdown(
        "- Effect size: Difference in ISI change (Week 8 minus baseline) between LLM and control, in points. MCID is about 6.\n"
        "- SD of change: Standard deviation of individual change scores. If using ANCOVA on post-ISI, SD(change) can be reduced relative to unadjusted change; adjust accordingly.\n"
        "- Proportion twins: Fraction of all randomized participants who are twins (in pairs of 2). The rest are singletons.\n"
        "- Proportion MZ: Among twin pairs, the fraction that are monozygotic (identical); others are dizygotic.\n"
        "- ICC MZ / DZ: Within-pair correlation in ISI change for MZ and DZ pairs, respectively. MZ typically > DZ.\n"
        "- Alpha: Two-sided test level for the primary endpoint.\n"
        "- Simulations: Number of Monte Carlo replicates; higher is slower but more precise.\n"
        "- Analysis method: \n"
        "  - cluster_robust: OLS with cluster-robust SEs clustered by pair, including zygosity indicators.\n"
        "  - mixedlm: Random intercept per pair (falls back if not available).\n"
        "  Fallback analytical GLS is used if required libraries are missing."
    )

with st.expander("Key assumptions in this power model"):
    st.markdown(
        "- Individual randomization (including co-twins), 1:1 allocation.\n"
        "- Outcome model: change = treatment effect + pair random effect + residual.\n"
        "- Pair random effect variance is chosen to match the specified ICC by zygosity (MZ/DZ).\n"
        "- Singletons have no pair effect (ICC=0) and variance = SD(change)^2.\n"
        "- Analysis includes zygosity fixed effects to adjust for systematic MZ/DZ differences.\n"
        "- Attrition is handled as enrollment inflation (not modeled as informative dropout).\n"
        "- Normality of effects and residuals; large-sample approximations are reasonable at typical Ns."
    )

with st.expander("Tips and sanity checks"):
    st.markdown(
        "- Use the reported MCID (~6 ISI points) to benchmark clinical relevance.\n"
        "- Inspect the approximate effective N (Kish) as a quick sense-check; it ignores across-arm co-twin correlation and is conservative.\n"
        "- If you plan baseline-adjusted analysis (ANCOVA), enter a smaller SD(change) based on expected pre/post correlation.\n"
        "- Consider sensitivity sweeps: vary ICCs (e.g., MZ 0.4–0.6, DZ 0.2–0.3) and SD(change) (e.g., 5–7)."
    )

st.divider()
st.subheader("Plots")

with st.expander("Power vs N curve"):
    cols = st.columns(4)
    with cols[0]:
        n_min_curve = st.number_input(
            "N min", min_value=20, max_value=20000, value=100, step=10, key="nmin_curve",
            help=(
                "Start of the N grid used to draw the power curve. "
                "Choose a low enough value to show the underpowered region for context. "
                "Use with N max and Points to control curve resolution."
            )
        )
    with cols[1]:
        n_max_curve = st.number_input(
            "N max", min_value=40, max_value=50000, value=1000, step=20, key="nmax_curve",
            help=(
                "End of the N grid used to draw the power curve. "
                "Set high enough to include the region where power plateaus above your target. "
                "Increase Points for a smoother curve if needed."
            )
        )
    with cols[2]:
        n_points = st.number_input(
            "Points", min_value=3, max_value=100, value=10, step=1, key="npoints_curve",
            help=(
                "Number of grid points between N min and N max when drawing the curve. "
                "More points produce a smoother curve but increase runtime. "
                "Match with Sims/point to balance smoothness and speed."
            )
        )
    with cols[3]:
        sims_curve = st.number_input(
            "Sims/point", min_value=200, max_value=10000, value=max(500, int(sims)//2), step=100, key="sims_curve",
            help=(
                "Monte Carlo iterations used at each N on the curve. "
                "Higher values produce smoother, less noisy curves at the cost of speed. "
                "Consider 1000+ per point for final figures."
            )
        )

    gen_curve = st.button(
        "Generate power curve",
        help=(
            "Simulate power across a grid of N and plot the curve to visualize how power grows with sample size. "
            "Use more points and more sims per point for smooth, publication-quality curves. "
            "Compare curves under multiple effect sizes and ICC assumptions to guide design."
        ),
    )
    if gen_curve:
        spec_curve = TwinTrialSpec(
            n_total=int(n_min_curve),  # placeholder; per-point spec is set in power_curve
            effect_points=float(effect_points),
            sd_change=float(sd_change),
            prop_twins=float(prop_twins),
            prop_mz=float(prop_mz),
            icc_mz=float(icc_mz),
            icc_dz=float(icc_dz),
            alpha=float(alpha),
            analysis=analysis,
            seed=int(seed),
            contamination_rate=float(contamination_rate),
            contamination_effect=float(contamination_effect),
            attrition_rate=float(attrition_rate),
        )
        ns = np.linspace(int(n_min_curve), int(n_max_curve), int(n_points)).astype(int)
        start_c = time.time()
        with st.spinner("Simulating power across N…"):
            df_curve = power_curve(spec_curve, ns, sims=int(sims_curve))
        st.caption(f"Curve runtime: {time.time()-start_c:.2f}s")
        st.line_chart(df_curve.set_index("N")["power"], height=260)
        st.dataframe(df_curve, use_container_width=True)

with st.expander("Distributions (estimates and p-values)"):
    sims_dist = st.number_input(
        "Simulations (distributions)", min_value=200, max_value=20000, value=int(sims), step=200, key="sims_dist",
        help=(
            "Number of Monte Carlo iterations used to build the histograms of estimated effects and p-values. "
            "More iterations yield smoother histograms and more stable estimates. "
            "Adjust to balance runtime and clarity."
        )
    )
    run_dist = st.button(
        "Simulate distributions",
        help=(
            "Run repeated simulations to visualize the sampling distributions of estimated treatment effects and p-values. "
            "Use these plots to check estimator bias/variance and p-value calibration. "
            "Increase the number of simulations for smoother, more informative histograms."
        ),
    )
    if run_dist:
        spec_d = TwinTrialSpec(
            n_total=int(n_total if mode == "Estimate power for fixed N" else max(80, int(n_min))),
            effect_points=float(effect_points),
            sd_change=float(sd_change),
            prop_twins=float(prop_twins),
            prop_mz=float(prop_mz),
            icc_mz=float(icc_mz),
            icc_dz=float(icc_dz),
            alpha=float(alpha),
            analysis=analysis,
            seed=int(seed),
            contamination_rate=float(contamination_rate),
            contamination_effect=float(contamination_effect),
            attrition_rate=float(attrition_rate),
        )
        with st.spinner("Running simulations…"):
            pw_d, coefs, pvals = simulate_distribution(spec_d, sims=int(sims_dist))
        st.write(f"Estimated power from these runs: {pw_d:.3f}")

        # Histogram for treatment effect estimates
        counts, edges = np.histogram(coefs[~np.isnan(coefs)], bins=30)
        centers = 0.5 * (edges[:-1] + edges[1:])
        df_hist_eff = pd.DataFrame({"estimate": centers, "count": counts})
        st.write("Treatment effect estimates (histogram)")
        st.bar_chart(df_hist_eff.set_index("estimate"), height=200)
        st.caption(f"Mean est: {np.nanmean(coefs):.2f}, SD est: {np.nanstd(coefs):.2f}, True: {effect_points:.2f}")

        # Histogram for p-values
        counts_p, edges_p = np.histogram(pvals[~np.isnan(pvals)], bins=20, range=(0, 1))
        centers_p = 0.5 * (edges_p[:-1] + edges_p[1:])
        df_hist_p = pd.DataFrame({"pvalue": centers_p, "count": counts_p})
        st.write("P-values (histogram)")
        st.bar_chart(df_hist_p.set_index("pvalue"), height=200)
