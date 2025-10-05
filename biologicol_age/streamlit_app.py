"""
Streamlit app: Twin-aware power for biological age (DunedinPACE / GrimAge).
Within-pair randomized twin RCT (paired design), analytic and simulation.
"""

from __future__ import annotations

import time
import math
from typing import Optional

import streamlit as st
import numpy as np

try:
    from .power_twin_age import (
        AgeTwinSpec,
        sd_change_from_pre_post,
        sd_diff_from_sd_change_icc,
        analytic_power_paired,
        analytic_pairs_for_power,
        analytic_mde,
        _simulate_pairs,
        _simulate_co_primary,
    )
except Exception:
    from power_twin_age import (  # type: ignore
        AgeTwinSpec,
        sd_change_from_pre_post,
        sd_diff_from_sd_change_icc,
        analytic_power_paired,
        analytic_pairs_for_power,
        analytic_mde,
        _simulate_pairs,
        _simulate_co_primary,
    )


st.set_page_config(page_title="Twin Power — BioAge", layout="centered")
st.title("Twin-Aware Power — Biological Age")
st.caption("Within-pair randomization (co-twin control). Paired-t analytic with optional simulation.")

# Inline hover help for controls (expanded)
HELP = {
    "alpha": (
        "Type I error rate (two-sided). Controls false positives. "
        "For co-primary endpoints, use 0.025 per endpoint (Bonferroni)."
    ),
    "use_sim": (
        "Use Monte Carlo on paired differences to mimic the trial many times and estimate power empirically. "
        "Simulation captures heterogeneous ICCs, MZ/DZ mixtures, and non-linear behaviors the analytic approximation may miss. "
        "Analytic paired t-test is fast and interpretable, but it assumes idealized conditions; simulation is recommended for final figures. "
        "Expect small Monte Carlo error that shrinks as you increase the number of simulations."
    ),
    "sims": (
        "Number of Monte Carlo iterations to run. More iterations reduce Monte Carlo noise but increase runtime. "
        "The Monte Carlo standard error of a power estimate p is approximately sqrt(p·(1−p)/sims). "
        "Example: with p≈0.85 and sims=2000, SE≈0.009 (≈±1.8% for a 95% CI). "
        "Increase to 3000–5000 for publication-quality precision."
    ),
    "seed": (
        "Random seed for reproducibility so repeated runs are comparable. "
        "Change the seed if you want to verify robustness to Monte Carlo variation. "
        "The seed has no effect on analytic (non-simulation) calculations. "
        "Use different seeds when stress-testing edge cases."
    ),
    "endpoint": (
        "Choose the outcome scale: DunedinPACE (unitless pace; smaller is slower aging), GrimAge (years), or a custom endpoint. "
        "For DunedinPACE, beneficial effects are positive slowings (e.g., 0.05 = 5% slower pace). For GrimAge, beneficial effects are year reductions. "
        "Custom endpoints should be entered as the absolute magnitude of improvement on the change scale. "
        "Pick the effect input mode (absolute vs paired d) that matches your data and reporting preferences."
    ),
    "mode_eff": (
        "Choose whether to enter the effect as an absolute magnitude in endpoint units or as a standardized paired effect size d. "
        "Paired d = effect_abs / SD(pair-diff) uses the variability of the within-pair difference, not the raw outcome. "
        "Using d is helpful when comparing across endpoints or when units differ. "
        "You can switch modes to sanity‑check implied magnitudes and power."
    ),
    "effect_abs_dpace": (
        "Enter absolute DunedinPACE slowing. For example, 0.03 means a 3% slower pace of aging versus control (beneficial). "
        "Protocol target: 2–3% slowing at Week 24 (primary). "
        "This is the additional slowing in treated twins relative to co‑twins. "
        "Sensitivity analysis across 0.02–0.04 is recommended."
    ),
    "effect_abs_grimage": (
        "Enter absolute GrimAge reduction in years (beneficial). Example: 2.0 years younger versus control. "
        "Note: CALERIE showed no GrimAge response; treat this as exploratory. Any reduction would be novel evidence. "
        "This parameter is the between‑arm difference in change, not a post value."
    ),
    "effect_abs_custom": (
        "Enter the absolute beneficial effect in your endpoint's native units (on the change scale). "
        "Internally, the app applies this as a reduction to the treated twin's change so that treated−control differences are negative. "
        "Report the magnitude as positive for readability; the sign is handled under the hood. "
        "Use paired d mode to confirm that the implied standardized magnitude is reasonable."
    ),
    "effect_d": (
        "Paired standardized effect size d = effect_abs / SD(pair-diff), where SD(pair-diff) reflects within‑pair noise. "
        "Guidelines: d≈0.1 small, 0.3 moderate, 0.5+ large for paired designs (context dependent). "
        "Using d makes scenarios comparable across endpoints and labs. "
        "Convert back to absolute units using SD(pair-diff) shown in the results panel."
    ),
    "sd_change": (
        "Standard deviation of an individual's change (follow‑up − baseline). This parameter drives the noise component of SD(pair‑diff). "
        "If unknown, use the pre/post helper: Var(change) = Var(post) + Var(pre) − 2·ρ·sd_post·sd_pre. "
        "ANCOVA on post values with baseline as a covariate typically implies a smaller effective SD(change). "
        "Check sensitivity to a plausible range informed by prior data or pilot studies."
    ),
    "icc_mz": (
        "Within‑pair correlation for MZ twins on the change outcome; higher values imply stronger shared variance. "
        "Typical ranges are 0.5–0.7+ for stable biomarkers measured closely in time. "
        "Higher ICC lowers SD(pair‑diff) = sqrt(2·(1−ICC)·sd_change^2) and therefore increases power for a given number of pairs. "
        "Use conservative values when external reliability data are limited."
    ),
    "icc_dz": (
        "Within‑pair correlation for DZ twins on the change outcome; usually lower than MZ due to lower genetic sharing. "
        "A common range is 0.3–0.6 depending on the measure and cohort. "
        "This parameter affects SD(pair‑diff) via the same 1−ICC factor as MZ, but the effective ICC depends on the MZ/DZ mix. "
        "Explore sensitivity across plausible DZ values when planning."
    ),
    "prop_mz": (
        "Proportion of twin pairs that are MZ (identical); the remainder are DZ (fraternal). "
        "This mix determines the effective ICC: ICC_eff ≈ 1 − [prop_mz·(1−ICC_MZ) + (1−prop_mz)·(1−ICC_DZ)]. "
        "Higher prop MZ increases the effective ICC and can reduce required sample size. "
        "Use cohort data if available; otherwise test 0.4–0.6 as a reasonable range."
    ),
    "goal": (
        "Select the objective for this session: estimate power for a fixed number of completing pairs, "
        "find the smallest number of pairs to reach a target power, or compute the minimal detectable effect (MDE). "
        "Estimating power helps assess feasibility under current assumptions. "
        "Searching for pairs or MDE helps with budgeting and endpoint selection."
    ),
    "n_pairs": (
        "Number of completing twin pairs analyzed after attrition (each pair contributes a treated and a control twin). "
        "Power increases with more pairs and with higher ICC (lower within‑pair noise). "
        "This is the analysis denominator; enrollment may need to be larger to account for dropout. "
        "Use the attrition calculator in the CLI to plan enrollment."
    ),
    "target_power": (
        "Desired probability (1−β) to detect the specified effect at the chosen α. "
        "Typical choices are 0.80 or 0.90; higher targets increase required pairs. "
        "Consider feasibility and expected effect size when choosing this value. "
        "Use simulation to verify that analytic and empirical estimates agree."
    ),
    "n_min": (
        "Lower bound for the binary search over pairs in sample-size mode. "
        "Set this low enough to include underpowered regions for a proper bracket. "
        "Larger ranges increase runtime but ensure the target is reachable. "
        "Use prior runs to tighten bounds and speed up."
    ),
    "n_max": (
        "Upper bound for the binary search over pairs in sample-size mode. "
        "Set high enough that the target power is achievable under conservative assumptions. "
        "If the search hits the upper bound, increase it and retry. "
        "Use incremental refinement to find exact minima."
    ),
    "sd_from_prepost": (
        "Derive SD(change) from SD(pre), SD(post), and their correlation ρ using the standard variance identity. "
        "Clicking this will overwrite the SD(change) input above with the computed value. "
        "This approach approximates the effect of baseline adjustment when pre/post are correlated. "
        "Use realistic ρ from prior data to avoid overstating precision."
    ),
}

st.sidebar.header("Settings")
alpha = st.sidebar.number_input("Alpha (two-sided)", 0.0001, 0.2, 0.05, 0.005, format="%.3f", help=HELP["alpha"])
use_sim = st.sidebar.checkbox("Use simulation", value=True, help=HELP["use_sim"])
sims = st.sidebar.number_input("Simulations", 200, 20000, 2000, 200, help=HELP["sims"])
seed = st.sidebar.number_input("Seed", 0, 10**9, 12345, 1, help=HELP["seed"])

st.sidebar.subheader("Contamination & Attrition")
contam_rate = st.sidebar.number_input("Contamination rate (controls)", 0.0, 1.0, 0.0, 0.05, format="%.2f")
contam_frac = st.sidebar.number_input("Effect fraction in contaminated controls", 0.0, 1.0, 0.0, 0.05, format="%.2f")
attrition = st.sidebar.number_input("Attrition rate", 0.0, 0.95, 0.0, 0.05, format="%.2f")

st.subheader("Endpoint and effect")
timepoint = st.selectbox("Timepoint", ["Week 12 (interim)", "Week 24 (primary)", "Month 12 (follow‑up)"], index=1)
endpoint = st.selectbox("Endpoint", ["dunedinpace", "grimage", "custom"], index=0, help=HELP["endpoint"])
mode_eff = st.selectbox("Effect input", ["absolute", "standardized d"], index=0, help=HELP["mode_eff"])

if endpoint == "dunedinpace" and mode_eff == "absolute":
    effect_abs = st.number_input("Absolute slowing (0.03 = 3%)", 0.0, 1.0, 0.03, 0.005, format="%.3f", help=HELP["effect_abs_dpace"])
elif endpoint == "grimage" and mode_eff == "absolute":
    effect_abs = st.number_input("Years reduction (absolute)", 0.0, 20.0, 2.0, 0.1, help=HELP["effect_abs_grimage"])
elif mode_eff == "absolute":
    effect_abs = st.number_input("Effect (absolute, beneficial)", 0.0, 100.0, 1.0, 0.1, help=HELP["effect_abs_custom"])
else:
    effect_d = st.number_input("Paired standardized effect d", 0.0, 5.0, 0.11, 0.01, format="%.2f", help=HELP["effect_d"])
    effect_abs = None

st.subheader("Variability and ICC")
sd_default = 0.10 if endpoint == "dunedinpace" else 3.0
sd_step = 0.01 if endpoint == "dunedinpace" else 0.1
sd_fmt = "%.3f" if endpoint == "dunedinpace" else "%.2f"
sd_change = st.number_input("SD of individual change", 1e-6, 100.0, sd_default, sd_step, format=sd_fmt, help=HELP["sd_change"])
icc_mz = st.number_input("ICC (MZ)", 0.0, 0.99, 0.55, 0.05, format="%.2f", help=HELP["icc_mz"])
icc_dz = st.number_input("ICC (DZ)", 0.0, 0.99, 0.55, 0.05, format="%.2f", help=HELP["icc_dz"])
prop_mz = st.number_input("Proportion MZ pairs", 0.0, 1.0, 0.5, 0.05, format="%.2f", help=HELP["prop_mz"])

with st.expander("Compute SD(change) from pre/post SDs and correlation"):
    c1, c2, c3 = st.columns(3)
    with c1:
        sd_pre = st.number_input("Baseline SD", 0.0, 100.0, sd_change, sd_step, key="sd_pre")
    with c2:
        sd_post = st.number_input("Follow-up SD", 0.0, 100.0, sd_change, sd_step, key="sd_post")
    with c3:
        rho = st.number_input("Corr(pre, post)", -0.99, 0.99, 0.5, 0.05, format="%.2f")
    if st.button("Use computed SD(change)", help=HELP["sd_from_prepost"]):
        sd_change = sd_change_from_pre_post(sd_pre, sd_post, rho)
        st.success(f"Updated SD(change) = {sd_change:.4f}")

st.subheader("Goal")
goal = st.radio("Choose", ["Estimate power (fixed pairs)", "Find pairs for target power", "Find MDE"], index=0, help=HELP["goal"])

if goal == "Estimate power (fixed pairs)":
    n_pairs = st.number_input("Completing pairs", 2, 100000, 700, 10, help=HELP["n_pairs"])
    if st.button(
        "Estimate power",
        type="primary",
        help=(
            "Compute statistical power under the current assumptions using either the analytic paired t-test or Monte Carlo simulation. "
            "Analytic mode is instantaneous and useful for quick iteration. Simulation provides a more faithful estimate when ICCs differ by zygosity or effects are borderline. "
            "Review the reported Monte Carlo uncertainty (via sims) before making final decisions."
        ),
    ):
        spec = AgeTwinSpec(
            n_pairs=int(n_pairs), effect_abs=float(effect_abs or 0.0), sd_change=float(sd_change),
            prop_mz=float(prop_mz), icc_mz=float(icc_mz), icc_dz=float(icc_dz), alpha=float(alpha), seed=int(seed),
            contamination_rate=float(contam_rate), contamination_effect=float(contam_frac)
        )
        icc_eff = spec.effective_icc()
        sd_d = sd_diff_from_sd_change_icc(sd_change, icc_eff)
        eff_abs = float(effect_abs) if mode_eff == "absolute" else float(effect_d) * sd_d  # type: ignore
        spec.effect_abs = eff_abs
        reduction = (1.0 - float(contam_rate) * float(contam_frac))
        eff_obs = eff_abs * reduction
        start = time.time()
        with st.spinner("Running…"):
            if use_sim:
                pw, avg_mag, se_pw = _simulate_pairs(spec, sims=int(sims))
            else:
                pw = analytic_power_paired(spec.n_pairs, eff_obs, spec.sd_change, icc_eff, spec.alpha)
                avg_mag = abs(eff_obs)
        st.success("Done.")
        st.metric("Estimated power", f"{pw:.3f}")
        st.write(
            f"- Timepoint: {timepoint}\n"
            f"- Endpoint: {endpoint}\n"
            f"- Pairs: {spec.n_pairs}\n"
            f"- Effect(abs): {eff_abs:.4f}  | Observed after contamination: {eff_obs:.4f}\n"
            f"- SD(change): {sd_change}\n"
            f"- ICC_eff: {icc_eff:.3f}  [MZ={icc_mz}, DZ={icc_dz}, prop_mz={prop_mz:.2f}]\n"
            f"- SD(pair-diff): {sd_d:.4f}\n"
            f"- alpha={alpha}, method={'simulation' if use_sim else 'analytic'}\n"
            f"- Expected mean paired diff (treated-control): {-eff_abs:.4f}"
        )
        if attrition > 0:
            n_enroll = math.ceil(int(n_pairs) / (1.0 - float(attrition)))
            st.info(f"Enrollment needed with attrition {attrition:.1%}: ~{n_enroll} pairs ({n_enroll*2} individuals)")

elif goal == "Find pairs for target power":
    target_power = st.number_input("Target power", 0.60, 0.995, 0.80, 0.01, format="%.2f", help=HELP["target_power"])
    n_min = st.number_input("Search min", 2, 100000, 50, 10, help=HELP["n_min"])
    n_max = st.number_input("Search max", 4, 200000, 5000, 20, help=HELP["n_max"])
    if st.button(
        "Find required pairs",
        type="primary",
        help=(
            "Run a binary search over the number of completing pairs to find the smallest N that achieves the target power. "
            "Results depend on effect size, SD(change), ICCs, and α; small changes in inputs can shift the minimum. "
            "Use a generous upper bound initially and then refine to pinpoint the exact threshold."
        ),
    ):
        spec0 = AgeTwinSpec(
            n_pairs=int(n_min), effect_abs=float(effect_abs or 0.0), sd_change=float(sd_change),
            prop_mz=float(prop_mz), icc_mz=float(icc_mz), icc_dz=float(icc_dz), alpha=float(alpha), seed=int(seed),
            contamination_rate=float(contam_rate), contamination_effect=float(contam_frac)
        )
        icc_eff = spec0.effective_icc()
        sd_d = sd_diff_from_sd_change_icc(sd_change, icc_eff)
        eff_abs = float(effect_abs) if mode_eff == "absolute" else float(effect_d) * sd_d  # type: ignore
        if use_sim:
            low, high = int(n_min), int(n_max)
            best_n, best_pw = high, 0.0
            while low <= high:
                mid = (low + high) // 2
                spec = AgeTwinSpec(
                    n_pairs=int(mid), effect_abs=float(eff_abs), sd_change=float(sd_change),
                    prop_mz=float(prop_mz), icc_mz=float(icc_mz), icc_dz=float(icc_dz), alpha=float(alpha), seed=int(seed),
                    contamination_rate=float(contam_rate), contamination_effect=float(contam_frac)
                )
                pw, _, _ = _simulate_pairs(spec, sims=max(500, int(sims)//2))
                if pw >= float(target_power):
                    best_n, best_pw = mid, pw
                    high = mid - 1
                else:
                    low = mid + 1
        else:
            eff_obs = float(eff_abs) * (1.0 - float(contam_rate) * float(contam_frac))
            best_n, best_pw = analytic_pairs_for_power(float(target_power), float(eff_obs), float(sd_change), float(icc_eff), float(alpha))
        st.success("Done.")
        st.metric("Required pairs (approx)", f"{best_n}")
        st.write(
            f"- Achieved power at N*: {best_pw:.3f}\n"
            f"- Endpoint: {endpoint}, Effect(abs): {eff_abs}, Observed after contamination: {eff_obs:.4f}, SD(change): {sd_change}\n"
            f"- ICC_eff={icc_eff:.3f}  [MZ={icc_mz}, DZ={icc_dz}, prop_mz={prop_mz:.2f}], alpha={alpha}, method={'simulation' if use_sim else 'analytic'}"
        )
        if attrition > 0:
            n_enroll = math.ceil(int(best_n) / (1.0 - float(attrition)))
            st.info(f"Enrollment needed with attrition {attrition:.1%}: ~{n_enroll} pairs ({n_enroll*2} individuals)")

else:
    n_pairs = st.number_input("Completing pairs", 2, 100000, 700, 10, help=HELP["n_pairs"])
    target_power = st.number_input("Target power", 0.60, 0.995, 0.80, 0.01, format="%.2f", help=HELP["target_power"])

# Glossary and assumptions
with st.expander("Parameter definitions and guidance", expanded=True):
    st.markdown(
        "- Effect (absolute): For DunedinPACE, enter absolute slowing (e.g., 0.05 = 5% slower pace; beneficial). For GrimAge, enter absolute year reduction (e.g., 2.0 years; beneficial). For custom endpoints, use the absolute magnitude on the change scale. Internally, we apply a reduction to the treated twin so the paired difference (treated−control) is negative, but we report magnitudes for clarity.\n"
        "- Paired d (standardized): d = effect_abs / SD(pair-diff). SD(pair-diff) = sqrt(2·(1−ICC_eff)·sd_change^2). Using d facilitates comparison across endpoints and labs. As a rough guide for paired designs: d≈0.10 small, 0.30 moderate, 0.50+ large (context matters).\n"
        "- SD(change): Standard deviation of each twin’s change (follow‑up − baseline). If unknown, derive via sd_change_from_pre_post(sd_pre, sd_post, ρ) using Var(change) = Var(post) + Var(pre) − 2·ρ·sd_post·sd_pre. ANCOVA on post with baseline covariate often implies a smaller effective SD(change) than unadjusted change.\n"
        "- ICCs (MZ/DZ) and effective ICC: ICC reflects within‑pair correlation for the change outcome. Higher ICC → smaller SD(pair‑diff) → higher power at fixed pairs. With a mix of MZ/DZ, ICC_eff ≈ 1 − [prop_mz·(1−ICC_MZ) + (1−prop_mz)·(1−ICC_DZ)]. Typical ranges: MZ 0.5–0.7+, DZ 0.3–0.6.\n"
        "- Proportion MZ: Fraction of pairs that are MZ (identical). Higher prop_mz tends to increase ICC_eff and reduce required pairs. If cohort data are unavailable, sensitivity test 0.4–0.6.\n"
        "- Alpha and co‑primary endpoints: Use α=0.05 for a single primary endpoint. For co‑primary endpoints (e.g., DunedinPACE + GrimAge), use α≈0.025 per endpoint (Bonferroni) and consider the co‑primary CLI mode for joint power.\n"
        "- Simulations: Monte Carlo power has sampling error ≈ sqrt(p·(1−p)/sims). At p=0.85 and sims=2000, SE≈0.009 (~±1.8% for 95% CI). Increase sims (e.g., 3000–5000) for publication‑quality estimates."
    )

with st.expander("Key assumptions", expanded=True):
    st.markdown(
        "- Randomization and independence: Within each pair, one twin is randomized to treatment and the co‑twin to control. Pairs are treated as independent clusters (SUTVA holds across pairs).\n"
        "- Outcome model: Change = pair random effect + residual; the intervention reduces the treated twin’s change by the specified effect. This yields paired differences with variance determined by SD(change) and ICC.\n"
        "- ICC and zygosity: Pair‑effect variance is set to match ICC by zygosity (MZ/DZ). When both zygosities are present, ICC_eff depends on prop_mz. Accurate zygosity classification improves planning.\n"
        "- Distributional assumptions: Analytic paired t‑test assumes approximate normality of paired differences and large‑sample behavior; simulation relaxes these by resampling from the specified model.\n"
        "- Measurement context: SD(change) and ICC reflect both biological variability and measurement noise (e.g., lab platform). If reliability is lower than expected, SD(pair‑diff) grows and power falls.\n"
        "- Missingness/attrition: Power is computed for completing pairs. Enrollment often needs inflation to reach the desired number of completers; plan for non‑response and protocol deviations outside of this model."
    )

with st.expander("Tips and sanity checks", expanded=True):
    st.markdown(
        "- Sanity‑check magnitudes: Ensure SD(pair‑diff) and the implied d are plausible given prior data. If d seems too large/small, revisit SD(change) and ICC assumptions.\n"
        "- Sensitivity sweeps: Vary effect (e.g., ±20%), SD(change), and ICCs (MZ/DZ) across reasonable ranges. Power should change smoothly and monotonically with N and effect size.\n"
        "- Simulation precision: For borderline designs (power near 0.80/0.90), increase sims and try multiple seeds to confirm stability. Use 3000–5000 sims for final decisions.\n"
        "- Endpoints and ANCOVA: If you will analyze post values with baseline adjustment, reduce SD(change) accordingly or translate to an equivalent d.\n"
        "- Effective ICC intuition: Higher ICC reduces within‑pair noise; prop_mz shifts ICC_eff. Use the printed SD(pair‑diff) and ICC_eff as quick diagnostics to understand power drivers."
    )
    if st.button(
        "Compute MDE",
        type="primary",
        help=(
            "Calculate the minimal detectable (beneficial) effect at the specified number of pairs, α, and target power. "
            "MDE helps assess endpoint sensitivity and whether expected effects are realistically observable. "
            "Use this alongside feasibility (pairs available) and measurement reliability when selecting endpoints."
        ),
    ):
        spec0 = AgeTwinSpec(
            n_pairs=int(n_pairs), effect_abs=0.0, sd_change=float(sd_change),
            prop_mz=float(prop_mz), icc_mz=float(icc_mz), icc_dz=float(icc_dz), alpha=float(alpha), seed=int(seed)
        )
        icc_eff = spec0.effective_icc()
        mde = analytic_mde(int(n_pairs), float(target_power), float(sd_change), float(icc_eff), float(alpha))
        st.success("Done.")
        # Adjust for contamination to report the underlying effect required without contamination
        reduction = (1.0 - float(contam_rate) * float(contam_frac))
        mde_no_contam = mde / reduction if reduction > 0 else float('inf')
        st.metric("MDE (observed under contamination)", f"{mde:.4f}")
        st.write(
            f"- Underlying MDE without contamination: {mde_no_contam:.4f} (rate={contam_rate:.1%}, fraction={contam_frac:.1%})\n"
            f"- Endpoint: {endpoint}, Pairs: {n_pairs}, Target power: {target_power:.2f}\n"
            f"- SD(change): {sd_change}, ICC_eff={icc_eff:.3f}  [MZ={icc_mz}, DZ={icc_dz}, prop_mz={prop_mz:.2f}], alpha={alpha}"
        )

# Advanced: co-primary endpoints
with st.expander("Co-primary power (DunedinPACE + GrimAge)"):
    st.write("Alpha per endpoint defaults to 0.025 here (Bonferroni).")
    st.caption(
        "Note: Treat GrimAge as exploratory. Its SD(change) is often much larger than DunedinPACE, "
        "so marginal power for GrimAge can be substantially lower for the same absolute effect."
    )
    c1, c2 = st.columns(2)
    with c1:
        dpace_eff = st.number_input("DunedinPACE slowing (abs)", 0.0, 1.0, 0.03, 0.005, format="%.3f")
        dpace_sd = st.number_input("DunedinPACE SD(change)", 0.0, 1.0, 0.10, 0.01, format="%.2f")
    with c2:
        grim_eff = st.number_input("GrimAge years reduction", 0.0, 20.0, 2.0, 0.1)
        grim_sd = st.number_input("GrimAge SD(change)", 0.0, 20.0, 3.0, 0.1)
    pair_corr = st.slider("Pair-effect correlation across endpoints", 0.0, 1.0, 0.8, 0.05)
    n_pairs_co = st.number_input("Completing pairs (co-primary)", 2, 100000, 700, 10)
    sims_co = st.number_input("Simulations (co-primary)", 500, 20000, int(sims), 500)
    # Adjust alpha per endpoint if user left global at 0.05
    alpha_co = 0.025 if abs(alpha - 0.05) < 1e-9 else alpha
    if st.button("Estimate co-primary joint power", type="primary"):
        spec1 = AgeTwinSpec(
            n_pairs=int(n_pairs_co), effect_abs=float(dpace_eff), sd_change=float(dpace_sd),
            prop_mz=float(prop_mz), icc_mz=float(icc_mz), icc_dz=float(icc_dz), alpha=float(alpha_co), seed=int(seed),
            contamination_rate=float(contam_rate), contamination_effect=float(contam_frac)
        )
        spec2 = AgeTwinSpec(
            n_pairs=int(n_pairs_co), effect_abs=float(grim_eff), sd_change=float(grim_sd),
            prop_mz=float(prop_mz), icc_mz=float(icc_mz), icc_dz=float(icc_dz), alpha=float(alpha_co), seed=int(seed)+1,
            contamination_rate=float(contam_rate), contamination_effect=float(contam_frac)
        )
        with st.spinner("Running co-primary simulation…"):
            pw_joint, pw1, pw2 = _simulate_co_primary(spec1, spec2, sims=int(sims_co), alpha=float(alpha_co), pair_effect_corr=float(pair_corr))
        st.success("Done.")
        st.metric("Joint power (both significant)", f"{pw_joint:.3f}")
        st.write(
            f"- Marginal power — DunedinPACE: {pw1:.3f}; GrimAge: {pw2:.3f}\n"
            f"- Alpha per endpoint: {alpha_co:.3f}; Sims: {int(sims_co)}\n"
            f"- Contamination: rate={contam_rate:.1%}, fraction={contam_frac:.1%}; Timepoint: {timepoint}"
        )
