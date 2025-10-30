"""
Unified Streamlit app: run power/sample-size calculations for
1) Biological Age (within-pair twin RCT; DunedinPACE)
2) Sleep (ISI change; individually randomized with twin mix)

This app is a thin UI that calls into the existing analysis modules:
- biological_age.power_twin_age
- sleep.power_twin_sleep
"""

from __future__ import annotations

import math
import json
from typing import Dict, Any

import streamlit as st

# Biological age imports
from biological_age.power_twin_age import (
    AgeTwinSpec,
    sd_change_from_pre_post as age_sd_change_from_pre_post,
    analytic_power_paired,
    analytic_pairs_for_power,
    analytic_mde,
    _simulate_pairs,
    _simulate_co_primary,
    analytic_power_two_sample,
    analytic_n_per_group_for_power,
    design_effect,
)

# Sleep imports
from sleep.power_twin_sleep import (
    TwinTrialSpec,
    simulate_power,
    find_n_for_power,
    sd_change_from_pre_post as sleep_sd_change_from_pre_post,
)

# (Lifestyle module available separately; not wired into the app UI)


st.set_page_config(page_title="Twin Power — Unified", layout="wide")
st.title("Twin-Aware Power — Unified App")
st.caption("Run power and sample-size calculations for Biological Age and Sleep studies in one place.")
st.warning(
    "This application is intended for educational and exploratory purposes only. "
    "The analyses have not undergone expert statistical review."
)


def _power_ci(p: float, se: float) -> tuple[float, float]:
    # If se is zero/None (e.g., power is exactly 0 or 1), return a degenerate interval at p
    if se is None or se <= 0:
        return (max(0.0, min(1.0, float(p))), max(0.0, min(1.0, float(p))))
    lo = max(0.0, p - 1.96 * se)
    hi = min(1.0, p + 1.96 * se)
    return (lo, hi)


def _approx_se(p: float, sims: int) -> float:
    sims = max(1, int(sims))
    v = max(p * (1.0 - p), 1e-12)
    return math.sqrt(v / sims)


def _download_button(label: str, payload: Dict[str, Any], key: str) -> None:
    st.download_button(
        label=label,
        data=json.dumps(payload, indent=2),
        file_name=f"{key}.json",
        mime="application/json",
        key=key,
    )

# Global help text blocks to show as hover tooltips (help=...) and inline guidance
HELP_BIO = {
    "alpha": (
        "Two‑sided Type I error rate for hypothesis tests. Use 0.05 for a single primary endpoint; for two co‑primary endpoints a simple Bonferroni split uses 0.025 per endpoint. Lower alpha reduces false positives but increases the sample size required for the same power. If you have a strong, pre‑registered directional hypothesis, a one‑sided test could be justified, but most regulatory and peer‑review contexts expect two‑sided tests to guard against unanticipated harm. For multiple outcomes, consider Holm or Hochberg procedures as alternatives to Bonferroni; they can be less conservative while still controlling family‑wise error."
    ),
    "use_sim": (
        "Choose Monte Carlo simulation to reflect your exact design (MZ/DZ mix, ICCs, contamination, attrition) and to handle co‑primary endpoints. Analytic mode (paired t‑test) is very fast and transparent but relies on stronger assumptions (e.g., normality, homogeneous variance, no contamination). Simulation naturally captures finite‑sample behavior, incorporates dilution from contamination, and estimates joint power across endpoints. Use analytic mode for rapid what‑ifs and intuition building; rely on simulation for final planning, sensitivity analyses, and when assumptions are questionable."
    ),
    "sims": (
        "How many trial replicates to simulate for power. Larger values reduce Monte Carlo noise at the cost of time. Rough precision: SE(power) ≈ sqrt(p·(1−p)/sims). Example: p=0.85 with 3000 sims → SE≈0.0067 (≈±1.3% at 95% CI). As a rule of thumb: 1000 sims for quick checks, 3000–5000 for drafts, 10k+ for publication or critical decisions. When comparing scenarios, keep ‘sims’ and the random seed fixed so differences reflect inputs, not randomness."
    ),
    "seed": (
        "Random seed controlling the pseudo‑random draws so results are reproducible. Changing the seed changes the Monte Carlo realization but not the underlying expected power. Log the seed in your protocol or report to allow exact replication of your power numbers; use a different seed only when deliberately re‑sampling to verify stability."
    ),
    "endpoint": (
        "Outcome scale to analyze. DunedinPACE is a unitless pace‑of‑aging index (beneficial = slower pace). ‘Custom’ lets you supply any absolute change scale. Choose the option that matches your scientific endpoint, reporting units, and how SD(change) is measured. Be consistent: the ‘effect’ and ‘SD of change’ must be on the same scale. If you also track standardized effects (Cohen’s d), you can convert between absolute and standardized via the paired‑difference SD shown below."
    ),
    "eff_abs_dpace": (
        "Absolute slowing on DunedinPACE (beneficial). For example, 0.03 means the treated twin slows aging by 3% relative to their co‑twin. Larger values increase power; use prior studies, meta‑analyses, or pilot estimates to anchor plausibility. Consider duration: shorter interventions tend to yield smaller absolute slowing. If you have a standardized target (paired d), multiply by SD(pair‑diff) to back out the absolute effect here."
    ),
    # (GrimAge-specific helper removed from UI)
    "eff_abs_custom": (
        "Absolute beneficial effect on your chosen change scale. Internally the treated change is reduced by this amount, so paired differences are negative; we report magnitudes positively for clarity. Set to 0 to sanity‑check that power ≈ alpha. If you have a standardized effect size (d), convert by multiplying d × SD(pair‑diff) where SD(pair‑diff)=sqrt(2·(1−ICC_eff))·SD(change)."
    ),
    "sd_change": (
        "Standard deviation of individual change (post − pre) on the chosen scale. Smaller variability increases power. If you plan baseline adjustment (ANCOVA), derive a reduced SD(change) using SD(pre), SD(post), and their correlation in the helper—this often yields meaningfully smaller variance than using raw change. SD(change) is one of the most influential inputs; where uncertain, run a sensitivity grid (e.g., ±20%) to assess robustness of N and MDE."
    ),
    "icc_mz": (
        "Within‑pair intraclass correlation (ICC) for MZ twins on the change measure. Higher ICC tightens paired differences: Var(pair‑diff)=2·(1−ICC)·Var(change), so power rises with ICC at fixed N. Typical change‑score ICCs may range from ~0.3 to 0.8 depending on measurement reliability and shared environment. Over‑ or under‑estimating ICC directly shifts precision; prefer estimates from cohorts measured with the same assay and follow‑up interval."
    ),
    "icc_dz": (
        "Within‑pair ICC for DZ twins on the change measure. Because DZ pairs are less correlated than MZ, lower ICC typically inflates the paired‑difference variance. The overall effective ICC blends MZ and DZ per your mix. If DZ ICC is close to MZ ICC, the incremental precision from MZ over DZ is smaller; if much lower, prioritizing MZ pairs (when feasible) can markedly reduce required N."
    ),
    "prop_mz": (
        "Proportion of twin pairs who are MZ. This controls the effective ICC: ICC_eff ≈ prop_mz·ICC_MZ + (1−prop_mz)·ICC_DZ (on the correlation scale). More MZ pairs generally increase precision at fixed N. If your recruitment mix is uncertain, run sensitivity (e.g., prop_mz in {0.3, 0.5, 0.7}) and plan to monitor the accrued mix to update projections during enrollment."
    ),
    "contam_rate": (
        "Share of control twins expected to adopt intervention‑like behaviors (0–1). This ‘contamination’ dilutes the observed treatment‑control contrast. With contamination fraction f, the observed effect becomes effect_true × (1 − rate × f). Use this for pragmatic trials where spillover, cross‑talk between co‑twins, or self‑initiated behavior change is likely. If contamination is differential (e.g., more common in one zygosity), simulate a range; even modest rates can materially shrink power."
    ),
    "contam_frac": (
        "Fraction (0–1) of the full intervention effect that contaminated controls experience. A value of 0.3 with a 20% contamination rate reduces the observed effect by 6% (0.2×0.3). If contaminated controls approach full adherence (fraction near 1), intent‑to‑treat effects will be heavily diluted. If your analysis is per‑protocol or uses instrumental variables, the estimand changes; here we assume ITT, so set realistic values and document sensitivity runs."
    ),
    "attrition": (
        "Proportion of randomized pairs expected to be missing from the primary analysis. Power and N are defined for completing pairs; plan enrollment as required_pairs/(1−attrition). In within‑pair analyses, a pair ‘completes’ only if both twins provide follow‑up; losing one twin removes the pair. Anticipate higher attrition for longer follow‑up, invasive sampling, or burdensome visits; include operational buffers."
    ),
    "goal": (
        "Choose the design task: (i) estimate power at a fixed number of completing pairs; (ii) solve for the minimum pairs needed to achieve a target power; (iii) compute the minimal detectable effect (MDE) at a given N; or (iv) estimate joint power when both co‑primary endpoints must be significant. Use MDE when effect size is uncertain to understand what magnitudes your design can credibly detect."
    ),
    "n_pairs": (
        "Number of completing twin pairs contributing data to the analysis. Enrollment should be inflated for attrition so that the expected number of completers matches this value. If you expect heterogeneous ICCs or measurement variance across sites, consider simulating with a slightly larger N than the analytic minimum to protect against misspecification."
    ),
    "target_power": (
        "Desired probability of detecting the specified effect (e.g., 0.80 or 0.90). Higher targets are more conservative but require more pairs. Many funders expect ≥0.80; choose ≥0.90 when stakes are high or assumptions are uncertain. Remember that contamination, lower ICC, or larger SD(change) all reduce power at fixed N."
    ),
    "n_min": (
        "Lower bound for the search over completing pairs when solving for sample size. Set low enough to include the underpowered region so the binary search can bracket the solution. If unsure, start near 20–50 pairs to ensure the algorithm explores low‑power regimes."
    ),
    "n_max": (
        "Upper bound for the search over completing pairs. Set high enough that the design is clearly well‑powered somewhere in the range. If the result hits this bound, increase it and re‑run—this indicates the initial bracket was too low for your assumptions."
    ),
    "pair_corr": (
        "Correlation (0–1) between the latent pair effects across the two endpoints for co‑primary analysis. Higher correlation increases the probability that both endpoints are significant together at the same N. At ρ≈0 the joint power ≈ product of marginal powers; at ρ→1 the joint success rate rises toward the weaker of the two marginal powers. Use pilot or literature to set ρ and report sensitivity (e.g., 0.5, 0.8, 0.95)."
    ),
    "n_pairs_co": (
        "Number of completing pairs used when estimating joint power for two endpoints. This plays the same role as ‘n_pairs’ but in a bivariate setting; power depends on both marginal effects/variances and their correlation structure."
    ),
    "sims_co": (
        "Number of Monte Carlo iterations for estimating co‑primary joint power. Because joint probabilities are often lower than marginal ones, use more iterations (e.g., ≥5000) for stable estimates. Keep the seed fixed when comparing scenarios so changes reflect inputs rather than Monte Carlo noise."
    ),
    "sd_prepost": (
        "Helper to derive SD(change) from SD(pre), SD(post), and their correlation ρ using Var(change)=Var(pre)+Var(post)−2ρ·sd_pre·sd_post. Positive correlations (common in longitudinal measures) reduce Var(change); negative correlations increase it. When ρ is uncertain, try 0.3–0.7; report which value you assumed to keep calculations transparent and reproducible."
    ),
}

HELP_SLEEP = {
    "alpha": (
        "Two‑sided Type I error rate. 0.05 is standard for a single primary endpoint; if you plan multiple primary outcomes, adjust alpha (e.g., Bonferroni or Holm) to control the family‑wise error rate. Lower alpha protects against false positives but increases required N. Use one‑sided tests only with a pre‑specified directional hypothesis and clear justification; two‑sided tests remain the norm to guard against unexpected harm."
    ),
    "sims": (
        "Number of Monte Carlo iterations used to estimate power. Larger values reduce Monte Carlo error but take longer. Approximate precision: SE(power) ≈ sqrt(p·(1−p)/sims). Use ~1000 for quick iteration, 3000–5000 for robust planning, and 10k+ when presenting final numbers. Keep the seed fixed to isolate the effect of input changes across scenarios."
    ),
    "seed": (
        "Random seed for reproducibility. Keeps simulated power results stable across runs for the same inputs. Record the seed alongside assumptions in analysis plans to facilitate independent replication and auditing of your power calculations."
    ),
    "n_jobs": (
        "Number of worker processes for Monte Carlo simulation. Set to 1 for deterministic single-core runs; use -1 to leverage all available cores (the app falls back to threads automatically if the environment blocks multiprocessing). Increasing workers shortens runtime for large `sims`, but be mindful of laptop thermals and concurrent workloads."
    ),
    "chunk_size": (
        "Simulations per task handed to each worker. Larger chunks reduce scheduling overhead for massive `sims`; smaller chunks provide finer-grained load balancing on heterogeneous hardware. Defaults to 64—tune upward (e.g., 256) for high-core servers, or downward (e.g., 16) when interactive responsiveness matters."
    ),
    "effect_points": (
        "Treatment–control difference in ISI change at Week 8, in points (beneficial = larger additional reduction for treatment). A commonly cited minimal clinically important difference (MCID) is ≈6 points—use this as a reference when selecting a plausible effect. Align your assumed effect with intervention intensity, adherence expectations, and follow‑up length; shorter or lower‑touch interventions often yield smaller point changes."
    ),
    "sd_change": (
        "Standard deviation of individual ISI change (Week 8 − baseline). Lower variability improves power. If analyzing post‑ISI with baseline as a covariate (ANCOVA), use the helper to convert SD(pre), SD(post), and their correlation into an implied SD(change), which is often smaller. When uncertain, perform sensitivity analysis (e.g., ±1–2 points) as SD assumptions can dominate sample size requirements."
    ),
    "prop_twins": (
        "Proportion of participants who are twins (i.e., enrolled as members of a pair). The remainder are singletons. More twins increase within‑cluster correlation and affect precision; the analysis uses cluster‑robust SEs to account for this. If your twin fraction is uncertain, explore a range (e.g., 0.5–0.9) and plan recruitment to achieve the target mix."
    ),
    "prop_mz": (
        "Among the twin participants, the fraction who are MZ (the rest are DZ). This shapes the effective clustering because MZ pairs typically have higher ICC than DZ pairs, influencing the standard errors. If MZ and DZ proportions drift during enrollment, update projections—precision at fixed N may change if the realized mix differs from planning assumptions."
    ),
    "icc_mz": (
        "Within‑pair ICC among MZ twins for ISI change. Higher ICC means twins’ outcomes are more similar, increasing effective clustering and influencing the precision of treatment effects. Use prior ISI datasets or pilot twin data to set this; ICCs for change often sit below those for raw scores due to regression to the mean and measurement error."
    ),
    "icc_dz": (
        "Within‑pair ICC among DZ twins for ISI change. Typically lower than MZ; together with the MZ fraction this determines the effective clustering in the mixed twin/singleton sample. Sensitivity to ICC assumptions can be high near the design boundary; if feasible, plan a short pilot to estimate ICCs and refine the main trial’s N."
    ),
    "contamination_rate": (
        "Proportion of control participants likely to adopt intervention‑like behaviors (0–1). Contamination reduces the observed treatment–control contrast. With fraction f, the observed effect becomes effect_true × (1 − rate × f). Use >0 when pragmatic conditions encourage self‑help or information spillover (including between co‑twins). If contamination is anticipated to vary by site or cohort, run scenario analyses to bound possible losses in power."
    ),
    "contamination_effect": (
        "Fraction (0–1) of the full treatment effect that contaminated controls experience. For example, a 25% contamination rate and 0.5 fraction shrinks the observed effect by 12.5%. If controls receive nearly the full intervention effect (fraction→1), intent‑to‑treat effects will be heavily diluted. Consider registering sensitivity analyses at several fraction levels to transparently communicate robustness of power to contamination."
    ),
    "attrition": (
        "Proportion of randomized individuals expected to be missing from the primary analysis. Power and N refer to completers; inflate planned enrollment by 1/(1−attrition) so enough participants finish. Attrition may not be random; while the simulation assumes missing completely at random for planning, operational strategies (reminders, flexible scheduling) should target at‑risk subgroups to maintain power."
    ),
    "mode": (
        "Pick whether to estimate power for a fixed total N (what is my power if I can enroll N?) or to search for the minimum N that achieves a target power (how many people do I need?). Use the search mode when negotiating budgets or timelines; use fixed‑N mode when N is constrained by feasibility."
    ),
    "n_total": (
        "Total number of randomized individuals contributing to the primary analysis (twins and singletons combined). Adjust planned enrollment upward if attrition is non‑zero. Note that pairs count as two individuals; this app’s analysis accounts for clustering via cluster‑robust standard errors or mixed models in the underlying engine."
    ),
    "target_power": (
        "Desired probability of detecting your specified effect (e.g., 0.80–0.95). Higher targets are more conservative and require larger N. Consider 0.90+ if assumptions (ICC, SD, contamination) are uncertain or if the trial is costly to repeat."
    ),
    "n_min": (
        "Lower bound for the sample‑size search. Set low enough that designs in this region are clearly underpowered; this helps the search bracket the solution and avoid boundary effects. If unsure, start at 100 and adjust based on returned results."
    ),
    "n_max": (
        "Upper bound for the sample‑size search. Set high enough that adequate power is reachable within the range. If the algorithm returns the upper bound, increase it and re‑run—your initial bracket was too low for the assumed effect and variability."
    ),
    "sd_prepost": (
        "Helper to derive SD(change) from SD(pre), SD(post), and their correlation ρ using Var(change)=Var(pre)+Var(post)−2ρ·sd_pre·sd_post. Positive correlations reduce change variance; negative correlations increase it. If you only know SD(pre) and an R² from baseline regression, you can approximate ρ ≈ sqrt(R²) under equal‑variance assumptions."
    ),
}

# Basic URL state for convenience (new API)
qp = st.query_params
_study_q = (qp.get("study", "bio") or "bio")
study_index = 0 if _study_q.lower() in ("bio", "biological", "biological age") else 1
study = st.sidebar.radio(
    "Study",
    ["Biological Age", "Sleep (ISI)"],
    index=study_index,
    help="Select which study framework to use in the main panel.",
)


def panel_bioage():
    st.header("Biological Age — Within-Pair Twin RCT")
    c1, c2, c3 = st.columns(3)
    with c1:
        alpha = st.number_input("Alpha (two-sided)", 0.0001, 0.2, 0.05, 0.005, format="%.3f", help=HELP_BIO["alpha"]) 
    # Allow URL to set default for use_sim on first load
    default_use_sim = str(qp.get("use_sim", "1")).lower() in ("1", "true")
    with c2:
        # Default to simulation mode checked on first load for realism and joint-endpoint support
        use_sim = st.checkbox("Use simulation", value=default_use_sim, help=HELP_BIO["use_sim"]) 
    with c3:
        sims = st.number_input("Simulations", 200, 20000, 2000, 200, help=HELP_BIO["sims"]) 
    seed = st.number_input("Random seed", 0, 10**9, 12345, 1, help=HELP_BIO["seed"]) 

    st.subheader("Endpoint and effect")
    endpoint = st.selectbox("Endpoint", ["dunedinpace", "custom"], index=0, help=HELP_BIO["endpoint"]) 
    if endpoint == "dunedinpace":
        eff_abs = st.number_input("Absolute slowing (e.g., 0.03 for 3%)", 0.0, 1.0, 0.03, 0.005, format="%.3f", help=HELP_BIO["eff_abs_dpace"]) 
        sd_change = st.number_input("SD of change", 1e-6, 1.0, 0.10, 0.01, format="%.3f", help=HELP_BIO["sd_change"]) 
    else:
        eff_abs = st.number_input("Effect (absolute, beneficial)", 0.0, 100.0, 1.0, 0.1, help=HELP_BIO["eff_abs_custom"]) 
        sd_change = st.number_input("SD of change", 0.0, 100.0, 1.0, 0.1, help=HELP_BIO["sd_change"]) 

    st.subheader("ICC and zygosity mix")
    icc_mz = st.number_input("ICC (MZ)", 0.0, 0.99, 0.60, 0.05, format="%.2f", help=HELP_BIO["icc_mz"]) 
    icc_dz = st.number_input("ICC (DZ)", 0.0, 0.99, 0.40, 0.05, format="%.2f", help=HELP_BIO["icc_dz"]) 
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
    contam_rate = st.number_input("Contamination rate (controls)", 0.0, 1.0, 0.30, 0.05, format="%.2f", help=HELP_BIO["contam_rate"]) 
    contam_frac = st.number_input("Effect fraction in contaminated controls", 0.0, 1.0, 0.40, 0.05, format="%.2f", help=HELP_BIO["contam_frac"]) 
    attrition = st.number_input("Attrition rate", 0.0, 0.95, 0.25, 0.05, format="%.2f", help=HELP_BIO["attrition"]) 

    st.subheader("Goal")
    goal = st.radio(
        "Choose",
        [
            "Estimate power (fixed pairs)",
            "Find pairs for target power",
            "Find MDE",
            "Co-primary joint power",
            "Two-sample power (individual-level, secondary)",
            "Two-sample N for target power (secondary)",
        ],
        index=0,
        help=HELP_BIO["goal"],
    ) 

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
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Effective ICC", f"{icc_eff:.3f}")
            with m2:
                st.metric("Observed effect (diluted)", f"{eff_obs:.4f}")
            with m3:
                # Relative efficiency and effective independent N
                from biological_age.power_twin_age import relative_efficiency as _re
                re = _re(icc_eff)
                deff = 1.0 + icc_eff
                n_eff = int((int(n_pairs) * 2) / max(1e-12, deff))
                st.metric("Relative efficiency (RE)", f"{re:.2f}×")
            if use_sim:
                with st.spinner("Running simulation…"):
                    pw, avg_mag, se_pw = _simulate_pairs(spec, sims=int(sims))
            else:
                pw = analytic_power_paired(spec.n_pairs, eff_obs, spec.sd_change, icc_eff, spec.alpha)
            st.metric("Estimated power", f"{pw:.3f}")
            if use_sim:
                lo, hi = _power_ci(pw, se_pw)
                st.caption(f"95% MC CI for power: [{lo:.3f}, {hi:.3f}] (sims={int(sims)})")
            # Design summary and share
            st.subheader("Design Summary")
            st.markdown(
                f"- Endpoint: `{endpoint}`\n"
                f"- Completing pairs: `{int(n_pairs)}`\n"
                f"- Effect (abs): `{float(eff_abs):.4f}`\n"
                f"- SD(change): `{float(sd_change):.4f}`\n"
                f"- ICC MZ/DZ: `{float(icc_mz):.2f}` / `{float(icc_dz):.2f}` (eff={icc_eff:.3f})\n"
                f"- Proportion MZ: `{float(prop_mz):.2f}`\n"
                f"- Contamination: rate `{float(contam_rate):.2f}`, frac `{float(contam_frac):.2f}`\n"
                f"- Attrition: `{float(attrition):.2f}`\n"
                f"- Alpha: `{float(alpha):.3f}`; Simulation: `{bool(use_sim)}` (sims={int(sims)})\n"
                f"- Relative efficiency (RE): `{re:.2f}×`; Effective independent N (DEFF=1+ICC): `{n_eff}`\n"
            )
            _download_button(
                "Download scenario (JSON)",
                payload={
                    "study": "bio_age",
                    "endpoint": endpoint,
                    "inputs": {
                        "n_pairs": int(n_pairs),
                        "effect_abs": float(eff_abs),
                        "sd_change": float(sd_change),
                        "icc_mz": float(icc_mz),
                        "icc_dz": float(icc_dz),
                        "prop_mz": float(prop_mz),
                        "contamination_rate": float(contam_rate),
                        "contamination_effect": float(contam_frac),
                        "attrition": float(attrition),
                        "alpha": float(alpha),
                        "seed": int(seed),
                        "sims": int(sims),
                        "use_sim": bool(use_sim),
                    },
                    "results": {
                        "power": float(pw),
                        "power_ci": _power_ci(pw, se_pw) if use_sim else None,
                        "effective_icc": float(icc_eff),
                        "observed_effect": float(eff_obs),
                        "relative_efficiency": float(re),
                        "effective_independent_N": int(n_eff),
                        "design_effect": float(deff),
                        "sd_pair_diff": float((2.0 * max(0.0, 1.0 - float(icc_eff)) * (float(sd_change) ** 2)) ** 0.5),
                    },
                },
                key="bio_fixed_power",
            )
            # Update URL (new API)
            st.query_params["study"] = "bio"
            st.query_params["use_sim"] = "1" if use_sim else "0"
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
                with st.spinner("Searching N via simulation…"):
                    low, high = int(n_min), int(n_max)
                    best_n, best_pw = high, 0.0
                    # Rough iteration bound for binary search
                    max_iter = 1 + int(math.ceil(math.log2(max(2, high - low + 1))))
                    prog = st.progress(0)
                    it = 0
                    while low <= high:
                        mid = (low + high) // 2
                        spec = AgeTwinSpec(
                            n_pairs=int(mid), effect_abs=float(eff_abs), sd_change=float(sd_change),
                            prop_mz=float(prop_mz), icc_mz=float(icc_mz), icc_dz=float(icc_dz), alpha=float(alpha), seed=int(seed),
                            contamination_rate=float(contam_rate), contamination_effect=float(contam_frac)
                        )
                        sims_mid = max(500, int(sims)//2)
                        pw, _, _ = _simulate_pairs(spec, sims=sims_mid)
                        if pw >= float(target_power):
                            best_n, best_pw = mid, pw
                            high = mid - 1
                        else:
                            low = mid + 1
                        it += 1
                        prog.progress(min(1.0, it / max_iter))
                    prog.empty()
            else:
                best_n, best_pw = analytic_pairs_for_power(float(target_power), float(eff_obs), float(sd_change), float(icc_eff), float(alpha))
            st.metric("Required pairs (approx)", f"{best_n}")
            # Efficiency summary at N*
            from biological_age.power_twin_age import relative_efficiency as _re
            re = _re(icc_eff)
            deff = 1.0 + icc_eff
            n_eff = int((int(best_n) * 2) / max(1e-12, deff))
            st.caption(f"Efficiency: RE={re:.2f}×; Effective independent N≈{n_eff}")
            st.write(f"Achieved power at N*: {best_pw:.3f}")
            if use_sim:
                se = _approx_se(best_pw, max(500, int(sims)//2))
                lo, hi = _power_ci(best_pw, se)
                st.caption(f"95% MC CI for power at N*: [{lo:.3f}, {hi:.3f}]")
            # Update URL (new API)
            st.query_params["study"] = "bio"
            st.query_params["use_sim"] = "1" if use_sim else "0"
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

    elif goal == "Co-primary joint power":
        st.info("Alpha per endpoint defaults to 0.025 if global alpha≈0.05.")
        alpha_co = 0.025 if abs(alpha - 0.05) < 1e-6 else alpha
        colA, colB = st.columns(2)
        with colA:
            dpace_eff = st.number_input("DunedinPACE slowing (abs)", 0.0, 1.0, 0.03, 0.005, format="%.3f", help=HELP_BIO["eff_abs_dpace"]) 
            dpace_sd = st.number_input("DunedinPACE SD(change)", 0.0, 1.0, 0.10, 0.01, format="%.3f", help=HELP_BIO["sd_change"]) 
        with colB:
            grim_eff = st.number_input("Secondary endpoint effect (abs)", 0.0, 20.0, 0.75, 0.1) 
            grim_sd = st.number_input("Secondary endpoint SD(change)", 0.0, 20.0, 3.0, 0.1) 
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
            with st.spinner("Simulating joint power…"):
                pw_joint, pw1, pw2 = _simulate_co_primary(
                    spec1, spec2, sims=int(sims_co), alpha=float(alpha_co), pair_effect_corr=float(pair_corr)
                )
            st.metric("Joint power (both significant)", f"{pw_joint:.3f}")
            st.write(f"Marginal — DunedinPACE: {pw1:.3f}; Secondary: {pw2:.3f}")
            # CI approximations
            se_joint = _approx_se(pw_joint, int(sims_co))
            lo_j, hi_j = _power_ci(pw_joint, se_joint)
            st.caption(f"95% MC CI for joint power: [{lo_j:.3f}, {hi_j:.3f}] (sims={int(sims_co)})")
            # Share
            _download_button(
                "Download scenario (JSON)",
                payload={
                    "study": "bio_age_coprimary",
                    "inputs": {
                        "n_pairs": int(n_pairs_co),
                        "dpace_effect": float(dpace_eff),
                        "dpace_sd": float(dpace_sd),
                        "secondary_effect": float(grim_eff),
                        "secondary_sd": float(grim_sd),
                        "pair_corr": float(pair_corr),
                        "icc_mz": float(icc_mz),
                        "icc_dz": float(icc_dz),
                        "prop_mz": float(prop_mz),
                        "contamination_rate": float(contam_rate),
                        "contamination_effect": float(contam_frac),
                        "alpha_each": float(alpha_co),
                        "seed": int(seed),
                        "sims": int(sims_co),
                    },
                    "results": {
                        "power_joint": float(pw_joint),
                        "power_joint_ci": (lo_j, hi_j),
                        "power_dpace": float(pw1),
                        "power_secondary": float(pw2),
                    },
                },
                key="bio_coprimary",
            )
            st.query_params["study"] = "bio"
            st.query_params["use_sim"] = "1"

    elif goal == "Two-sample power (individual-level, secondary)":
        n_pg = st.number_input("n per group", 2, 100000, 150, 2)
        ancova_r2 = st.number_input("ANCOVA baseline R² (optional)", 0.0, 0.99, 0.50, 0.05, format="%.2f")
        deff_icc = st.number_input("Design effect ICC (optional)", 0.0, 0.99, 0.50, 0.05, format="%.2f")
        deff_m = st.number_input("Cluster size m (for DEFF)", 1, 10, 2, 1)
        if st.button("Estimate two-sample power", type="primary"):
            sd_ts = sd_change * (1.0 - ancova_r2) ** 0.5 if 0.0 <= ancova_r2 < 1.0 else sd_change
            eff_obs = float(eff_abs) * (1.0 - float(contam_rate) * float(contam_frac))
            pw = analytic_power_two_sample(int(n_pg), float(eff_obs), float(sd_ts), float(alpha))
            st.metric("Two-sample power", f"{pw:.3f}")
            n_total = 2 * int(n_pg)
            try:
                deff = design_effect(float(deff_icc), int(deff_m)) if deff_icc > 0 else None
            except Exception:
                deff = None
            if deff is not None:
                n_infl = int(math.ceil(n_total * deff))
                st.caption(f"With DEFF={deff:.3f}, inflated total N={n_infl}")
            if attrition > 0:
                n_enroll = math.ceil(n_total / (1.0 - float(attrition)))
                st.info(f"Enrollment with attrition {attrition:.1%}: ~{n_enroll} individuals")

    else:  # Two-sample N for target power (secondary)
        target_power = st.number_input("Target power", 0.60, 0.995, 0.80, 0.01, format="%.2f") 
        ancova_r2 = st.number_input("ANCOVA baseline R² (optional)", 0.0, 0.99, 0.50, 0.05, format="%.2f", key="ts_r2")
        deff_icc = st.number_input("Design effect ICC (optional)", 0.0, 0.99, 0.50, 0.05, format="%.2f", key="ts_icc")
        deff_m = st.number_input("Cluster size m (for DEFF)", 1, 10, 2, 1, key="ts_m")
        if st.button("Find n per group", type="primary"):
            sd_ts = sd_change * (1.0 - ancova_r2) ** 0.5 if 0.0 <= ancova_r2 < 1.0 else sd_change
            eff_obs = float(eff_abs) * (1.0 - float(contam_rate) * float(contam_frac))
            n_pg = analytic_n_per_group_for_power(float(target_power), float(eff_obs), float(sd_ts), float(alpha))
            n_total = 2 * int(n_pg)
            st.metric("Required n per group", f"{n_pg}")
            try:
                deff = design_effect(float(deff_icc), int(deff_m)) if deff_icc > 0 else None
            except Exception:
                deff = None
            if deff is not None:
                n_infl = int(math.ceil(n_total * deff))
                st.caption(f"With DEFF={deff:.3f}, inflated total N={n_infl}")
            if attrition > 0:
                n_enroll = math.ceil(n_total / (1.0 - float(attrition)))
                st.info(f"Enrollment with attrition {attrition:.1%}: ~{n_enroll} individuals")

    st.divider()
    st.subheader("Study preset (LLM multi-domain twin RCT)")
    st.caption("Quick grid: 2–3% DunedinPACE slowing across 150/165/185 completing pairs with enrollment inflation.")
    c1, c2, c3 = st.columns(3)
    with c1:
        eff_grid = st.text_input("Effect % grid", value="2,2.5,3", key="study_eff_grid")
    with c2:
        n_grid = st.text_input("Completing pairs grid", value="150,165,185", key="study_n_grid")
    with c3:
        attr_grid = st.text_input("Attrition range", value="0.25,0.30", key="study_attr_grid")
    if st.button("Run study summary", key="study_run"):
        try:
            eff_vals = [float(x.strip()) for x in eff_grid.split(',') if x.strip()]
            n_vals = [int(float(x.strip())) for x in n_grid.split(',') if x.strip()]
            ar = [float(x.strip()) for x in attr_grid.split(',') if x.strip()]
            assert len(eff_vals) > 0 and len(n_vals) > 0 and len(ar) >= 2
        except Exception:
            st.error("Please provide valid CSVs for effects, N, and attrition (e.g., 2,2.5,3 | 150,165,185 | 0.25,0.30)")
        else:
            # Effective ICC for current design controls
            icc_eff = 1.0 - (float(prop_mz) * (1.0 - float(icc_mz)) + (1.0 - float(prop_mz)) * (1.0 - float(icc_dz)))
            icc_eff = max(0.0, min(0.999999, icc_eff))
            import pandas as pd
            rows = []
            for n in n_vals:
                row = {"n_pairs": int(n)}
                for ep in eff_vals:
                    eff_abs = float(ep) / 100.0
                    # Apply contamination for observed effect
                    eff_obs = eff_abs * (1.0 - float(contam_rate) * float(contam_frac))
                    if use_sim:
                        spec = AgeTwinSpec(
                            n_pairs=int(n), effect_abs=eff_obs, sd_change=float(sd_change),
                            prop_mz=float(prop_mz), icc_mz=float(icc_mz), icc_dz=float(icc_dz), alpha=float(alpha), seed=int(seed),
                            contamination_rate=0.0, contamination_effect=0.0,
                        )
                        pw, _, _ = _simulate_pairs(spec, sims=int(sims))
                    else:
                        pw = analytic_power_paired(int(n), float(eff_obs), float(sd_change), float(icc_eff), float(alpha))
                    row[f"{ep:.2f}%".rstrip('0').rstrip('.')] = round(float(pw), 3)
                rows.append(row)
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True)
            lo_ar = max(0.0, min(0.99, float(ar[0])))
            hi_ar = max(0.0, min(0.99, float(ar[-1])))
            st.subheader("Enrollment")
            for n in n_vals:
                el = math.ceil(int(n) / (1.0 - lo_ar))
                eh = math.ceil(int(n) / (1.0 - hi_ar))
                st.write(f"Completing {int(n)} → Enroll {el}–{eh} pairs ({2*el}–{2*eh} individuals)")

    st.subheader("Sensitivity grid (Δ × ρ)")
    st.caption("Quick sensitivity across within-pair effect Δ and twin correlation ρ. Uses analytic paired t-test.")
    grid_mode = st.radio("Grid mode", ["Pairs for target power", "Power for fixed pairs"], index=0, horizontal=True)
    colg1, colg2, colg3, colg4 = st.columns(4)
    with colg1:
        deltas_csv = st.text_input("Δ values (abs)", value="0.020,0.025,0.030")
    with colg2:
        rhos_csv = st.text_input("ρ values (ICC)", value="0.40,0.50,0.60")
    with colg3:
        apply_contam = st.checkbox("Apply contamination to Δ", value=True)
    if grid_mode == "Pairs for target power":
        with colg4:
            target_power = st.number_input("Target power", 0.50, 0.999, 0.80, 0.01, format="%.2f")
    else:
        with colg4:
            n_pairs_grid = st.number_input("Completing pairs", 2, 100000, 150, 10)
    if st.button("Compute sensitivity grid", key="bio_sens_grid"):
        try:
            deltas = [float(x.strip()) for x in deltas_csv.split(',') if x.strip()]
            rhos = [float(x.strip()) for x in rhos_csv.split(',') if x.strip()]
            assert deltas and rhos
        except Exception:
            st.error("Provide valid CSVs for Δ and ρ, e.g., 0.020,0.025,0.030 and 0.40,0.50,0.60")
        else:
            import pandas as pd
            rows = []
            for dlt in deltas:
                row = {"Δ": dlt}
                for rho in rhos:
                    eff = float(dlt)
                    if apply_contam:
                        eff *= (1.0 - float(contam_rate) * float(contam_frac))
                    if grid_mode == "Pairs for target power":
                        n_req, _ = analytic_pairs_for_power(float(target_power), float(eff), float(sd_change), float(rho), float(alpha))
                        row[f"ρ={rho:.2f}"] = int(n_req)
                    else:
                        pw = analytic_power_paired(int(n_pairs_grid), float(eff), float(sd_change), float(rho), float(alpha))
                        row[f"ρ={rho:.2f}"] = round(float(pw), 3)
                rows.append(row)
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True)
            if grid_mode == "Pairs for target power" and attrition > 0:
                st.caption(f"Enrollment inflation with attrition {attrition:.1%}: enroll N_pairs / (1-attrition)")


def panel_sleep():
    st.header("Sleep — ISI Change (Individually Randomized RCT with Twins)")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        alpha = st.number_input("Alpha (two-sided)", min_value=0.0001, max_value=0.2, value=0.05, step=0.005, format="%.3f", help=HELP_SLEEP["alpha"]) 
    with c2:
        sims = st.number_input("Monte Carlo simulations", min_value=200, max_value=20000, value=2000, step=200, help=HELP_SLEEP["sims"]) 
    with c3:
        seed = st.number_input("Random seed", min_value=0, max_value=10**9, value=12345, step=1, help=HELP_SLEEP["seed"]) 
    with c4:
        n_jobs = st.number_input("Worker processes", min_value=-1, max_value=64, value=1, step=1, help=HELP_SLEEP["n_jobs"], format="%d") 
    with c5:
        chunk_size = st.number_input("Chunk size", min_value=1, max_value=2048, value=64, step=16, help=HELP_SLEEP["chunk_size"], format="%d") 

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
    r1c1, r1c2 = st.columns(2)
    with r1c1:
        prop_twins = st.number_input("Proportion twins", min_value=0.0, max_value=1.0, value=0.9, step=0.05, format="%.2f", help=HELP_SLEEP["prop_twins"]) 
    with r1c2:
        prop_mz = st.number_input("Proportion MZ among twins", min_value=0.0, max_value=1.0, value=0.5, step=0.05, format="%.2f", help=HELP_SLEEP["prop_mz"]) 
    r2c1, r2c2 = st.columns(2)
    with r2c1:
        icc_mz = st.number_input("ICC (MZ)", min_value=0.0, max_value=0.99, value=0.5, step=0.05, format="%.2f", help=HELP_SLEEP["icc_mz"]) 
    with r2c2:
        icc_dz = st.number_input("ICC (DZ)", min_value=0.0, max_value=0.99, value=0.25, step=0.05, format="%.2f", help=HELP_SLEEP["icc_dz"]) 

    st.subheader("Contamination and attrition (optional)")
    col6, col7 = st.columns(2)
    with col6:
        contamination_rate = st.number_input("Contamination rate (controls)", min_value=0.0, max_value=1.0, value=0.30, step=0.05, format="%.2f", help=HELP_SLEEP["contamination_rate"]) 
    with col7:
        contamination_effect = st.number_input("Fraction of full effect in contaminated controls", min_value=0.0, max_value=1.0, value=0.40, step=0.05, format="%.2f", help=HELP_SLEEP["contamination_effect"]) 
    attrition = st.number_input("Attrition rate", min_value=0.0, max_value=0.95, value=0.25, step=0.05, format="%.2f", help=HELP_SLEEP["attrition"]) 

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
            with st.spinner("Running simulation…"):
                pw, avg_est = simulate_power(
                    spec,
                    sims=int(sims),
                    n_jobs=int(n_jobs),
                    chunk_size=int(chunk_size),
                )
            st.metric("Estimated power", f"{pw:.3f}")
            se_pw = _approx_se(pw, int(sims))
            lo, hi = _power_ci(pw, se_pw)
            st.caption(f"95% MC CI for power: [{lo:.3f}, {hi:.3f}] (sims={int(sims)})")
            # Design summary and share
            st.subheader("Design Summary")
            st.markdown(
                f"- Total N (individuals): `{int(n_total)}`\n"
                f"- Effect (ISI points): `{float(effect_points):.2f}`; SD(change): `{float(sd_change):.2f}`\n"
                f"- Twin proportion: `{float(prop_twins):.2f}`; MZ among twins: `{float(prop_mz):.2f}`\n"
                f"- ICC MZ/DZ: `{float(icc_mz):.2f}` / `{float(icc_dz):.2f}`\n"
                f"- Contamination: rate `{float(contamination_rate):.2f}`, frac `{float(contamination_effect):.2f}`\n"
                f"- Attrition: `{float(attrition):.2f}`; Alpha: `{float(alpha):.3f}`; Sims: `{int(sims)}`\n"
                f"- Parallelism: workers `{int(n_jobs)}`, chunk size `{int(chunk_size)}`\n"
            )
            _download_button(
                "Download scenario (JSON)",
                payload={
                    "study": "sleep_isi",
                    "inputs": {
                        "n_total": int(n_total),
                        "effect_points": float(effect_points),
                        "sd_change": float(sd_change),
                        "prop_twins": float(prop_twins),
                        "prop_mz": float(prop_mz),
                        "icc_mz": float(icc_mz),
                        "icc_dz": float(icc_dz),
                        "contamination_rate": float(contamination_rate),
                        "contamination_effect": float(contamination_effect),
                        "attrition": float(attrition),
                        "alpha": float(alpha),
                        "seed": int(seed),
                        "sims": int(sims),
                        "n_jobs": int(n_jobs),
                        "chunk_size": int(chunk_size),
                    },
                    "results": {
                        "power": float(pw),
                        "power_ci": (lo, hi),
                        "avg_estimated_effect": float(avg_est),
                    },
                },
                key="sleep_fixed_power",
            )
            if attrition > 0:
                n_enroll = math.ceil(int(n_total) / (1.0 - float(attrition)))
                st.info(f"Enrollment with attrition {attrition:.1%}: ~{n_enroll} individuals")

    else:
        target_power = st.number_input("Target power", min_value=0.60, max_value=0.995, value=0.90, step=0.01, format="%.2f", help=HELP_SLEEP["target_power"]) 
        n_min = st.number_input("Search min", min_value=20, max_value=100000, value=20, step=10, help=HELP_SLEEP["n_min"]) 
        n_max = st.number_input("Search max", min_value=40, max_value=400000, value=200, step=20, help=HELP_SLEEP["n_max"]) 

    # Inline guidance and best practices (kept separate from the action button)
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

    # Action button only in the 'Find N' mode (kept outside expander for visibility)
    if mode == "Find N for target power":
        if st.button("Find N for target power", type="primary"):
            spec0 = TwinTrialSpec(
                n_total=int(n_min), effect_points=float(effect_points), sd_change=float(sd_change),
                prop_twins=float(prop_twins), prop_mz=float(prop_mz), icc_mz=float(icc_mz), icc_dz=float(icc_dz),
                alpha=float(alpha), analysis="cluster_robust", seed=int(seed),
                contamination_rate=float(contamination_rate), contamination_effect=float(contamination_effect), attrition_rate=float(attrition)
            )
            with st.spinner("Searching N via simulation…"):
                best_n, best_pw = find_n_for_power(
                    float(target_power),
                    spec0,
                    sims=int(sims),
                    n_min=int(n_min),
                    n_max=int(n_max),
                    n_jobs=int(n_jobs),
                    chunk_size=int(chunk_size),
                )
            st.metric("Required N (individuals)", f"{best_n}")
            st.write(f"Achieved power at N*: {best_pw:.3f}")
            se = _approx_se(best_pw, int(sims))
            lo, hi = _power_ci(best_pw, se)
            st.caption(f"95% MC CI for power at N*: [{lo:.3f}, {hi:.3f}]")
            _download_button(
                "Download scenario (JSON)",
                payload={
                    "study": "sleep_isi_find_n",
                    "inputs": {
                        "target_power": float(target_power),
                        "search_min": int(n_min),
                        "search_max": int(n_max),
                        "effect_points": float(effect_points),
                        "sd_change": float(sd_change),
                        "prop_twins": float(prop_twins),
                        "prop_mz": float(prop_mz),
                        "icc_mz": float(icc_mz),
                        "icc_dz": float(icc_dz),
                        "contamination_rate": float(contamination_rate),
                        "contamination_effect": float(contamination_effect),
                        "attrition": float(attrition),
                        "alpha": float(alpha),
                        "seed": int(seed),
                        "sims": int(sims),
                        "n_jobs": int(n_jobs),
                        "chunk_size": int(chunk_size),
                    },
                    "results": {
                        "n_required": int(best_n),
                        "power": float(best_pw),
                        "power_ci": (lo, hi),
                    },
                },
                key="sleep_find_n",
            )
            if attrition > 0:
                n_enroll = math.ceil(int(best_n) / (1.0 - float(attrition)))
                st.info(f"Enrollment with attrition {attrition:.1%}: ~{n_enroll} individuals")


if study == "Biological Age":
    panel_bioage()
else:
    panel_sleep()
