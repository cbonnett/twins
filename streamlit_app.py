"""
Unified Streamlit app: run power/sample-size calculations for
1) Biological Age (within-pair twin RCT; DunedinPACE/GrimAge)
2) Sleep (ISI change; individually randomized with twin mix)

This app is a thin UI that calls into the existing analysis modules:
- biologicol_age.power_twin_age
- sleep.power_twin_sleep
"""

from __future__ import annotations

import math
from typing import Optional

import streamlit as st

# Biological age imports
from biologicol_age.power_twin_age import (
    AgeTwinSpec,
    sd_change_from_pre_post as age_sd_change_from_pre_post,
    sd_diff_from_sd_change_icc,
    analytic_power_paired,
    analytic_pairs_for_power,
    analytic_mde,
    _simulate_pairs,
    _simulate_co_primary,
)

# Sleep imports
from sleep.power_twin_sleep import (
    TwinTrialSpec,
    simulate_power,
    find_n_for_power,
    sd_change_from_pre_post as sleep_sd_change_from_pre_post,
)


st.set_page_config(page_title="Twin Power — Unified", layout="wide")
st.title("Twin-Aware Power — Unified App")
st.caption("Run power and sample-size calculations for Biological Age and Sleep studies in one place.")

# Global help text blocks to show as hover tooltips (help=...) and inline guidance
HELP_BIO = {
    "alpha": (
        "Two-sided significance level (Type I error rate). For two co‑primary endpoints, use 0.025 per endpoint (Bonferroni). "
        "Lower alpha reduces false positives but requires larger samples."
    ),
    "use_sim": (
        "Monte Carlo simulation estimates power by simulating many trials under your assumptions (ICCs, MZ/DZ mix). "
        "Analytic mode (paired t) is fast and interpretable but assumes ideal conditions; simulation is recommended for final numbers, edge cases, and co‑primary analysis."
    ),
    "sims": (
        "Number of Monte Carlo iterations. Larger values reduce Monte Carlo noise but take longer. "
        "Monte Carlo SE ≈ sqrt(p·(1−p)/sims). Example: p=0.85, sims=3000 → SE≈0.0067 (≈±1.3% at 95% CI)."
    ),
    "seed": (
        "Random seed for reproducibility of simulation results. Changing the seed produces slightly different estimates due to Monte Carlo noise."
    ),
    "endpoint": (
        "Outcome scale. DunedinPACE is unitless pace‑of‑aging (beneficial = slower pace). GrimAge is years (beneficial = fewer years). Custom is any absolute change scale you provide."
    ),
    "eff_abs_dpace": (
        "Absolute slowing for DunedinPACE. Example: 0.03 = 3% slower pace in treatment vs control (beneficial)."
    ),
    "eff_abs_grimage": (
        "Absolute years reduction for GrimAge. Example: 2.0 = treatment is 2 years younger than control at follow‑up."
    ),
    "eff_abs_custom": (
        "Absolute beneficial effect on your change scale. Internally, treated change is reduced by this amount so paired differences are negative; magnitudes are reported positive for clarity."
    ),
    "sd_change": (
        "SD of individual change (post − pre). If you will adjust for baseline (ANCOVA), you can derive a smaller SD(change) from SD(pre), SD(post), and corr(pre,post) using the helper expander."
    ),
    "icc_mz": (
        "Within‑pair ICC for change in MZ twins. Higher ICC reduces SD of the paired difference and increases power at fixed N."
    ),
    "icc_dz": (
        "Within‑pair ICC for change in DZ twins. The effective ICC is a variance‑weighted blend of MZ/DZ per your MZ proportion."
    ),
    "prop_mz": (
        "Proportion of twin pairs that are MZ. Effective ICC ≈ 1 − [prop_mz·(1−ICC_MZ) + (1−prop_mz)·(1−ICC_DZ)]."
    ),
    "contam_rate": (
        "Proportion of controls expected to adopt intervention‑like behaviors (0–1)."
    ),
    "contam_frac": (
        "Fraction of the full intervention effect that contaminated controls receive (0–1). Observed effect = effect × (1 − rate × fraction)."
    ),
    "attrition": (
        "Expected dropout rate (0–1). Reported power and required pairs refer to completing pairs; enrollment is inflated by 1/(1−attrition)."
    ),
    "goal": (
        "Choose your task: (1) Estimate power for a fixed number of completing pairs; (2) Find the minimum pairs to reach a target power; (3) Compute the minimal detectable effect (MDE); (4) Estimate co‑primary joint power."
    ),
    "n_pairs": (
        "Number of completing twin pairs contributing to primary analysis. Enrollment must be larger if attrition > 0."
    ),
    "target_power": (
        "Desired statistical power (e.g., 0.80 or 0.90). Higher targets require more pairs."
    ),
    "n_min": (
        "Lower bound of the search range for pairs‑for‑power. Use a small value to include the underpowered region."
    ),
    "n_max": (
        "Upper bound of the search range for pairs‑for‑power. Ensure it’s high enough to include the adequately powered region."
    ),
    "pair_corr": (
        "Correlation (0–1) between the pair random effects across endpoints (co‑primary). Higher correlation increases the chance both endpoints are significant together."
    ),
    "n_pairs_co": (
        "Number of completing pairs for the co‑primary simulation."
    ),
    "sims_co": (
        "Number of Monte Carlo iterations used for co‑primary joint power. Increase for tighter precision."
    ),
    "sd_prepost": (
        "Helper to compute SD(change) from SD(pre), SD(post), and their correlation ρ using Var(change)=Var(pre)+Var(post)−2ρ·sd_pre·sd_post."
    ),
}

HELP_SLEEP = {
    "alpha": (
        "Two-sided significance level (Type I error rate). If you later adopt multiple primaries, adjust alpha accordingly (e.g., Bonferroni)."
    ),
    "sims": (
        "Number of Monte Carlo iterations. Monte Carlo SE ≈ sqrt(p·(1−p)/sims). Increase for final estimates."
    ),
    "seed": (
        "Random seed for reproducibility of simulated results."
    ),
    "effect_points": (
        "Effect size in ISI points: additional reduction (more negative change) in treatment vs control by Week 8. MCID is ~6 points."
    ),
    "sd_change": (
        "SD of the individual ISI change (Week 8 − baseline). If using ANCOVA on post‑ISI, derive a smaller SD(change) from SD(pre), SD(post), and corr using the helper."
    ),
    "prop_twins": (
        "Proportion of the full sample who are twins (pairs). Remaining participants are singletons. Affects clustering."
    ),
    "prop_mz": (
        "Among twins, fraction who are MZ. Remaining are DZ. Influences effective correlation and precision."
    ),
    "icc_mz": (
        "Within‑pair ICC for MZ twins on change."
    ),
    "icc_dz": (
        "Within‑pair ICC for DZ twins on change."
    ),
    "contamination_rate": (
        "Proportion of controls expected to adopt intervention‑like behaviors (0–1)."
    ),
    "contamination_effect": (
        "Fraction of full effect those contaminated controls receive (0–1). Observed effect = effect × (1 − rate × fraction)."
    ),
    "attrition": (
        "Expected dropout rate (0–1). Enrollment is inflated by 1/(1−attrition); power refers to completing individuals."
    ),
    "mode": (
        "Choose to estimate power for a fixed total N (individuals) or search for the minimum N achieving a target power."
    ),
    "n_total": (
        "Total number of randomized individuals contributing to the primary analysis."
    ),
    "target_power": (
        "Desired statistical power (e.g., 0.80–0.95)."
    ),
    "n_min": (
        "Lower bound of the search range for N."
    ),
    "n_max": (
        "Upper bound of the search range for N."
    ),
    "sd_prepost": (
        "Helper to compute SD(change) from SD(pre), SD(post), and their correlation ρ using the standard variance identity."
    ),
}

study = st.sidebar.radio("Study", ["Biological Age", "Sleep (ISI)"], help="Select which study framework to use in the main panel.")


def panel_bioage():
    st.header("Biological Age — Within-Pair Twin RCT")
    c1, c2, c3 = st.columns(3)
    with c1:
        alpha = st.number_input("Alpha (two-sided)", 0.0001, 0.2, 0.05, 0.005, format="%.3f", help=HELP_BIO["alpha"]) 
    with c2:
        use_sim = st.checkbox("Use simulation", value=False, help=HELP_BIO["use_sim"]) 
    with c3:
        sims = st.number_input("Simulations", 200, 20000, 2000, 200, help=HELP_BIO["sims"]) 
    seed = st.number_input("Random seed", 0, 10**9, 12345, 1, help=HELP_BIO["seed"]) 

    st.subheader("Endpoint and effect")
    endpoint = st.selectbox("Endpoint", ["dunedinpace", "grimage", "custom"], index=0, help=HELP_BIO["endpoint"]) 
    if endpoint == "dunedinpace":
        eff_abs = st.number_input("Absolute slowing (e.g., 0.03 for 3%)", 0.0, 1.0, 0.03, 0.005, format="%.3f", help=HELP_BIO["eff_abs_dpace"]) 
        sd_change = st.number_input("SD of change", 1e-6, 1.0, 0.10, 0.01, format="%.3f", help=HELP_BIO["sd_change"]) 
    elif endpoint == "grimage":
        eff_abs = st.number_input("Years reduction (absolute)", 0.0, 20.0, 2.0, 0.1, help=HELP_BIO["eff_abs_grimage"]) 
        sd_change = st.number_input("SD of change (years)", 0.0, 20.0, 3.0, 0.1, help=HELP_BIO["sd_change"]) 
    else:
        eff_abs = st.number_input("Effect (absolute, beneficial)", 0.0, 100.0, 1.0, 0.1, help=HELP_BIO["eff_abs_custom"]) 
        sd_change = st.number_input("SD of change", 0.0, 100.0, 1.0, 0.1, help=HELP_BIO["sd_change"]) 

    st.subheader("ICC and zygosity mix")
    icc_mz = st.number_input("ICC (MZ)", 0.0, 0.99, 0.55, 0.05, format="%.2f", help=HELP_BIO["icc_mz"]) 
    icc_dz = st.number_input("ICC (DZ)", 0.0, 0.99, 0.55, 0.05, format="%.2f", help=HELP_BIO["icc_dz"]) 
    prop_mz = st.number_input("Proportion MZ", 0.0, 1.0, 0.5, 0.05, format="%.2f", help=HELP_BIO["prop_mz"]) 

    with st.expander("Compute SD(change) from pre/post SDs and correlation (optional)"):
        c1, c2, c3 = st.columns(3)
        with c1:
            sd_pre = st.number_input("Baseline SD", 0.0, 100.0, sd_change, 0.01, key="bio_sd_pre")
        with c2:
            sd_post = st.number_input("Follow-up SD", 0.0, 100.0, sd_change, 0.01, key="bio_sd_post")
        with c3:
            rho = st.number_input("Corr(pre, post)", -0.99, 0.99, 0.5, 0.05, format="%.2f", help=HELP_BIO["sd_prepost"]) 
        if st.button("Use computed SD(change)", key="bio_use_sd_from_prepost"):
            sd_change = age_sd_change_from_pre_post(sd_pre, sd_post, rho)
            st.success(f"Updated SD(change) = {sd_change:.4f}")

    st.subheader("Contamination and attrition")
    contam_rate = st.number_input("Contamination rate (controls)", 0.0, 1.0, 0.0, 0.05, format="%.2f", help=HELP_BIO["contam_rate"]) 
    contam_frac = st.number_input("Effect fraction in contaminated controls", 0.0, 1.0, 0.0, 0.05, format="%.2f", help=HELP_BIO["contam_frac"]) 
    attrition = st.number_input("Attrition rate", 0.0, 0.95, 0.0, 0.05, format="%.2f", help=HELP_BIO["attrition"]) 

    st.subheader("Goal")
    goal = st.radio("Choose", ["Estimate power (fixed pairs)", "Find pairs for target power", "Find MDE", "Co-primary joint power"], index=0, help=HELP_BIO["goal"]) 

    if goal == "Estimate power (fixed pairs)":
        n_pairs = st.number_input("Completing pairs", 2, 100000, 700, 10, help=HELP_BIO["n_pairs"]) 
        if st.button("Estimate power", type="primary"):
            spec = AgeTwinSpec(
                n_pairs=int(n_pairs), effect_abs=float(eff_abs), sd_change=float(sd_change),
                prop_mz=float(prop_mz), icc_mz=float(icc_mz), icc_dz=float(icc_dz), alpha=float(alpha), seed=int(seed),
                contamination_rate=float(contam_rate), contamination_effect=float(contam_frac)
            )
            icc_eff = spec.effective_icc()
            eff_obs = spec.observed_effect()
            if use_sim:
                pw, avg_mag, se_pw = _simulate_pairs(spec, sims=int(sims))
            else:
                pw = analytic_power_paired(spec.n_pairs, eff_obs, spec.sd_change, icc_eff, spec.alpha)
            st.metric("Estimated power", f"{pw:.3f}")
            if attrition > 0:
                n_enroll = math.ceil(int(n_pairs) / (1.0 - float(attrition)))
                st.info(f"Enrollment with attrition {attrition:.1%}: ~{n_enroll} pairs ({n_enroll*2} individuals)")

    elif goal == "Find pairs for target power":
        target_power = st.number_input("Target power", 0.60, 0.995, 0.80, 0.01, format="%.2f", help=HELP_BIO["target_power"]) 
        n_min = st.number_input("Search min", 2, 100000, 50, 10, help=HELP_BIO["n_min"]) 
        n_max = st.number_input("Search max", 4, 200000, 5000, 20, help=HELP_BIO["n_max"]) 
        if st.button("Find required pairs", type="primary"):
            spec0 = AgeTwinSpec(
                n_pairs=int(n_min), effect_abs=float(eff_abs), sd_change=float(sd_change),
                prop_mz=float(prop_mz), icc_mz=float(icc_mz), icc_dz=float(icc_dz), alpha=float(alpha), seed=int(seed),
                contamination_rate=float(contam_rate), contamination_effect=float(contam_frac)
            )
            icc_eff = spec0.effective_icc()
            eff_obs = spec0.observed_effect()
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
                best_n, best_pw = analytic_pairs_for_power(float(target_power), float(eff_obs), float(sd_change), float(icc_eff), float(alpha))
            st.metric("Required pairs (approx)", f"{best_n}")
            st.write(f"Achieved power at N*: {best_pw:.3f}")
            if attrition > 0:
                n_enroll = math.ceil(int(best_n) / (1.0 - float(attrition)))
                st.info(f"Enrollment with attrition {attrition:.1%}: ~{n_enroll} pairs ({n_enroll*2} individuals)")

    elif goal == "Find MDE":
        n_pairs = st.number_input("Completing pairs", 2, 100000, 700, 10, help=HELP_BIO["n_pairs"]) 
        target_power = st.number_input("Target power", 0.60, 0.995, 0.80, 0.01, format="%.2f", help=HELP_BIO["target_power"]) 
        if st.button("Compute MDE", type="primary"):
            spec0 = AgeTwinSpec(
                n_pairs=int(n_pairs), effect_abs=float(eff_abs), sd_change=float(sd_change),
                prop_mz=float(prop_mz), icc_mz=float(icc_mz), icc_dz=float(icc_dz), alpha=float(alpha), seed=int(seed),
                contamination_rate=float(contam_rate), contamination_effect=float(contam_frac)
            )
            icc_eff = spec0.effective_icc()
            mde = analytic_mde(int(n_pairs), float(target_power), float(sd_change), float(icc_eff), float(alpha))
            if contam_rate > 0:
                rf = 1.0 - float(contam_rate) * float(contam_frac)
                mde_no_contam = mde / rf
                st.write(f"MDE (beneficial abs): {mde:.4f}; No contamination: {mde_no_contam:.4f}")
            else:
                st.write(f"MDE (beneficial abs): {mde:.4f}")

    else:  # Co-primary
        st.info("Alpha per endpoint defaults to 0.025 if global alpha=0.05.")
        alpha_co = 0.025 if abs(alpha - 0.05) < 1e-9 else alpha
        colA, colB = st.columns(2)
        with colA:
            dpace_eff = st.number_input("DunedinPACE slowing (abs)", 0.0, 1.0, 0.03, 0.005, format="%.3f", help=HELP_BIO["eff_abs_dpace"]) 
            dpace_sd = st.number_input("DunedinPACE SD(change)", 0.0, 1.0, 0.10, 0.01, format="%.3f", help=HELP_BIO["sd_change"]) 
        with colB:
            grim_eff = st.number_input("GrimAge years reduction", 0.0, 20.0, 2.0, 0.1, help=HELP_BIO["eff_abs_grimage"]) 
            grim_sd = st.number_input("GrimAge SD(change)", 0.0, 20.0, 3.0, 0.1, help=HELP_BIO["sd_change"]) 
        pair_corr = st.slider("Pair-effect correlation across endpoints", 0.0, 1.0, 0.8, 0.05, help=HELP_BIO["pair_corr"]) 
        n_pairs_co = st.number_input("Completing pairs (co-primary)", 2, 100000, 700, 10, help=HELP_BIO["n_pairs_co"]) 
        sims_co = st.number_input("Simulations (co-primary)", 500, 20000, int(sims), 500, help=HELP_BIO["sims_co"]) 
        if st.button("Estimate joint power", type="primary"):
            spec1 = AgeTwinSpec(
                n_pairs=int(n_pairs_co), effect_abs=float(dpace_eff), sd_change=float(dpace_sd),
                prop_mz=float(prop_mz), icc_mz=float(icc_mz), icc_dz=float(icc_dz), alpha=float(alpha_co), seed=int(seed),
                contamination_rate=float(contam_rate), contamination_effect=float(contam_frac),
            )
            spec2 = AgeTwinSpec(
                n_pairs=int(n_pairs_co), effect_abs=float(grim_eff), sd_change=float(grim_sd),
                prop_mz=float(prop_mz), icc_mz=float(icc_mz), icc_dz=float(icc_dz), alpha=float(alpha_co), seed=int(seed) + 1,
                contamination_rate=float(contam_rate), contamination_effect=float(contam_frac),
            )
            pw_joint, pw1, pw2 = _simulate_co_primary(spec1, spec2, sims=int(sims_co), alpha=float(alpha_co), pair_effect_corr=float(pair_corr))
            st.metric("Joint power (both significant)", f"{pw_joint:.3f}")
            st.write(f"Marginal — DunedinPACE: {pw1:.3f}; GrimAge: {pw2:.3f}")


def panel_sleep():
    st.header("Sleep — ISI Change (Individually Randomized RCT with Twins)")
    c1, c2, c3 = st.columns(3)
    with c1:
        alpha = st.number_input("Alpha (two-sided)", min_value=0.0001, max_value=0.2, value=0.05, step=0.005, format="%.3f", help=HELP_SLEEP["alpha"]) 
    with c2:
        sims = st.number_input("Monte Carlo simulations", min_value=200, max_value=20000, value=2000, step=200, help=HELP_SLEEP["sims"]) 
    with c3:
        seed = st.number_input("Random seed", min_value=0, max_value=10**9, value=12345, step=1, help=HELP_SLEEP["seed"]) 

    st.subheader("Effect and variability")
    col1, col2 = st.columns(2)
    with col1:
        effect_points = st.number_input("Effect (ISI points)", min_value=0.0, max_value=20.0, value=6.0, step=0.5, help=HELP_SLEEP["effect_points"]) 
    with col2:
        sd_change = st.number_input("SD of ISI change", min_value=0.5, max_value=20.0, value=7.0, step=0.5, help=HELP_SLEEP["sd_change"]) 

    with st.expander("Compute SD(change) from pre/post SDs and correlation (optional)"):
        c1, c2, c3 = st.columns(3)
        with c1:
            sd_pre = st.number_input("Baseline SD", min_value=0.5, max_value=30.0, value=sd_change, step=0.5, key="sleep_sd_pre")
        with c2:
            sd_post = st.number_input("Follow-up SD", min_value=0.5, max_value=30.0, value=sd_change, step=0.5, key="sleep_sd_post")
        with c3:
            rho = st.number_input("Corr(pre, post)", min_value=-0.99, max_value=0.99, value=0.5, step=0.05, format="%.2f", help=HELP_SLEEP["sd_prepost"]) 
        if st.button("Use computed SD(change)", key="sleep_use_sd_from_prepost"):
            sd_change = sleep_sd_change_from_pre_post(sd_pre, sd_post, rho)
            st.success(f"Updated SD(change) = {sd_change:.3f}")

    st.subheader("Twin structure and ICCs")
    col3, col4, col5 = st.columns(3)
    with col3:
        prop_twins = st.number_input("Proportion twins", min_value=0.0, max_value=1.0, value=0.9, step=0.05, format="%.2f", help=HELP_SLEEP["prop_twins"]) 
    with col4:
        prop_mz = st.number_input("Proportion MZ among twins", min_value=0.0, max_value=1.0, value=0.5, step=0.05, format="%.2f", help=HELP_SLEEP["prop_mz"]) 
    with col5:
        icc_mz = st.number_input("ICC (MZ)", min_value=0.0, max_value=0.99, value=0.5, step=0.05, format="%.2f", help=HELP_SLEEP["icc_mz"]) 
    icc_dz = st.number_input("ICC (DZ)", min_value=0.0, max_value=0.99, value=0.25, step=0.05, format="%.2f", help=HELP_SLEEP["icc_dz"]) 

    st.subheader("Contamination and attrition (optional)")
    col6, col7 = st.columns(2)
    with col6:
        contamination_rate = st.number_input("Contamination rate (controls)", min_value=0.0, max_value=1.0, value=0.0, step=0.05, format="%.2f", help=HELP_SLEEP["contamination_rate"]) 
    with col7:
        contamination_effect = st.number_input("Fraction of full effect in contaminated controls", min_value=0.0, max_value=1.0, value=0.0, step=0.05, format="%.2f", help=HELP_SLEEP["contamination_effect"]) 
    attrition = st.number_input("Attrition rate", min_value=0.0, max_value=0.95, value=0.0, step=0.05, format="%.2f", help=HELP_SLEEP["attrition"]) 

    st.subheader("Goal")
    mode = st.radio("Choose", ("Estimate power for fixed N", "Find N for target power"), index=0, help=HELP_SLEEP["mode"]) 

    if mode == "Estimate power for fixed N":
        n_total = st.number_input("Total N (individuals)", min_value=10, max_value=200000, value=220, step=10, help=HELP_SLEEP["n_total"]) 
        if st.button("Estimate power", type="primary"):
            spec = TwinTrialSpec(
                n_total=int(n_total), effect_points=float(effect_points), sd_change=float(sd_change),
                prop_twins=float(prop_twins), prop_mz=float(prop_mz), icc_mz=float(icc_mz), icc_dz=float(icc_dz),
                alpha=float(alpha), analysis="cluster_robust", seed=int(seed),
                contamination_rate=float(contamination_rate), contamination_effect=float(contamination_effect), attrition_rate=float(attrition)
            )
            pw, se = simulate_power(spec, sims=int(sims))
            st.metric("Estimated power", f"{pw:.3f}")
            if attrition > 0:
                n_enroll = math.ceil(int(n_total) / (1.0 - float(attrition)))
                st.info(f"Enrollment with attrition {attrition:.1%}: ~{n_enroll} individuals")

    else:
        target_power = st.number_input("Target power", min_value=0.60, max_value=0.995, value=0.90, step=0.01, format="%.2f", help=HELP_SLEEP["target_power"]) 
        n_min = st.number_input("Search min", min_value=20, max_value=100000, value=100, step=10, help=HELP_SLEEP["n_min"]) 
        n_max = st.number_input("Search max", min_value=40, max_value=400000, value=10000, step=20, help=HELP_SLEEP["n_max"]) 

    # Inline guidance and best practices
    with st.expander("Guidance, assumptions, and sanity checks", expanded=False):
        st.markdown(
            "- Under null (effect≈0), power≈alpha; use this to sanity‑check inputs.\n"
            "- Power should increase with larger effects, larger N, and higher ICC (for bio‑age paired design).\n"
            "- Attrition inflates ENROLLMENT only; power is computed on completing pairs/individuals.\n"
            "- For co‑primary, alpha per endpoint defaults to 0.025 if global alpha=0.05.\n"
            "- Monte Carlo precision: use sims≥3000 for final results; report 95% CIs where appropriate.\n"
            "- For bio‑age, SD(pair‑diff)=sqrt(2·(1−ICC_eff))·SD(change). Paired d = effect_abs / SD(pair‑diff).\n"
            "- For sleep, analysis defaults to cluster‑robust OLS on individuals clustered by pair; MixedLM available in the module for sensitivity checks."
        )
        if st.button("Find N for target power", type="primary"):
            spec0 = TwinTrialSpec(
                n_total=int(n_min), effect_points=float(effect_points), sd_change=float(sd_change),
                prop_twins=float(prop_twins), prop_mz=float(prop_mz), icc_mz=float(icc_mz), icc_dz=float(icc_dz),
                alpha=float(alpha), analysis="cluster_robust", seed=int(seed),
                contamination_rate=float(contamination_rate), contamination_effect=float(contamination_effect), attrition_rate=float(attrition)
            )
            best_n, best_pw = find_n_for_power(spec0, float(target_power), low=int(n_min), high=int(n_max), sims=int(sims))
            st.metric("Required N (individuals)", f"{best_n}")
            st.write(f"Achieved power at N*: {best_pw:.3f}")
            if attrition > 0:
                n_enroll = math.ceil(int(best_n) / (1.0 - float(attrition)))
                st.info(f"Enrollment with attrition {attrition:.1%}: ~{n_enroll} individuals")


if study == "Biological Age":
    panel_bioage()
else:
    panel_sleep()
