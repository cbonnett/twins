"""
Revised power analysis for a within-pair randomized twin RCT on biological age.

This module centers the within-pair (paired) analysis as the PRIMARY basis for
sample size and power calculations, consistent with the study design. It also
includes individual‑level (two‑sample) calculations as SECONDARY/confirmatory
checks.

Endpoints
---------
- DunedinPACE (unitless pace-of-aging; baseline ≈ 1.0). Effects usually
  expressed as percent slowing (e.g., 2–3%).

Primary Analysis (Within-Pair)
------------------------------
- Paired twin design with within-pair randomization (one treated, one control).
- Primary test is a paired t‑test on within‑pair differences in change:
  d_i = (Δ_treated) − (Δ_control).
- Under a random‑intercept twin model for change:
    y = −Δ * I[treat] + b_pair + e,
  with Var(b_pair) = ICC·σ² and Var(e) = (1−ICC)·σ², the difference variance is
    Var(d) = 2·(1−ICC)·σ².
- The paired t‑test is equivalent to an OLS with pair fixed effects.

Secondary Analysis (Individual-Level)
------------------------------------
- Two‑sample t‑test on change scores treating individuals as independent.
- Intended as a robustness/confirmatory analysis, not for powering the study.

Key Improvements
----------------
- Contamination modeling: Accounts for control twins adopting intervention behaviors
- Attrition adjustment: Inflates sample size for expected dropout
- Co‑primary endpoint support: Joint power calculation and alpha adjustment
- Input validation and consistent parameterization
- Reporting emphasizes within‑pair ICC, SD of pair differences, and relative efficiency

What you need to specify
------------------------
- Effect magnitude on the change scale (absolute and beneficial):
  * DunedinPACE: absolute slowing (e.g., 0.02–0.03 for 2–3%).
  * GrimAge: absolute change in years (e.g., 2.0).
- SD of the individual change (sd_change). If unknown, use
  sd_change_from_pre_post().
- Within‑pair ICC for the change outcome (can differ by zygosity). If a mix is
  provided, analytic formulas use a variance‑weighted effective ICC.
- Optional contamination: rate and fractional effect in controls.

Features
--------
- Analytic paired‑t power, sample size, and minimal detectable effect (MDE)
- Monte Carlo simulation with optional MZ/DZ mix and different ICCs
- CLI modes: power, pairs-for-power, mde, curve, co-primary-power
- Automatic attrition adjustment for enrollment planning

Notes
-----
- Effect magnitude inputs represent absolute beneficial magnitudes (positive).
- Internally, treated outcomes use y_treat = baseline − effect_abs (reduction).
- Outputs report beneficial effects as positive values for clarity.
- RECOMMENDATION: Use --use-simulation for final trial calculations when MZ/DZ
  ICCs differ substantially or for co‑primary endpoint analysis.

Usage examples
--------------
1) Power at 173 completing pairs for DunedinPACE (2.5% slowing), assuming
   sd_change=0.10, ICC_eff≈0.50, alpha=0.05:
   python3 power_twin_age.py --mode power --n-pairs 173 \
     --endpoint dunedinpace --effect-abs 0.025 --sd-change 0.10 \
     --icc-mz 0.60 --icc-dz 0.40 --prop-mz 0.5 --alpha 0.05

2) Required pairs for 90% power at 2.0‑year GrimAge reduction with 27% attrition:
   python3 power_twin_age.py --mode pairs-for-power --target-power 0.90 \
     --endpoint grimage --effect-abs 2.0 --sd-change 3.0 \
     --icc-mz 0.60 --icc-dz 0.40 --prop-mz 0.5 --attrition-rate 0.27

3) Joint power for co‑primary endpoints (alpha=0.025 per endpoint):
   python3 power_twin_age.py --mode co-primary-power --n-pairs 173 \
     --endpoint dunedinpace --effect-abs 0.025 --sd-change 0.10 \
     --endpoint2 grimage --effect2-abs 2.0 --sd2-change 3.0 \
     --icc-mz 0.60 --icc-dz 0.40 --alpha 0.025 --pair-effect-corr 0.8 --use-simulation --sims 5000

4) MDE with simulation (recommended):
   python3 power_twin_age.py --mode mde --n-pairs 173 \
     --endpoint dunedinpace --sd-change 0.10 --icc-mz 0.60 --icc-dz 0.40 \
     --target-power 0.80 --use-simulation --sims 3000

"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Optional, Tuple

import math
import numpy as np

import scipy.stats as sps
import statsmodels.api as sm  # noqa: F401
import statsmodels.formula.api as smf  # noqa: F401


# ---------- Utility functions ----------

def validate_probability(value: float, name: str, allow_zero: bool = True, allow_one: bool = True):
    """Validate that value is a valid probability in [0,1]."""
    lower = 0.0 if allow_zero else 0.0 + 1e-9
    upper = 1.0 if allow_one else 1.0 - 1e-9
    if not (lower <= value <= upper):
        raise ValueError(f"{name} must be in [{lower}, {upper}], got {value}")


def validate_positive(value: float, name: str, allow_zero: bool = False):
    """Validate that value is positive (or non-negative if allow_zero)."""
    threshold = 0.0 if allow_zero else 1e-12
    if value < threshold:
        raise ValueError(f"{name} must be {'non-negative' if allow_zero else 'positive'}, got {value}")


def validate_icc(value: float, name: str):
    """Validate ICC is in [0, 1)."""
    if not (0.0 <= value < 1.0):
        raise ValueError(f"{name} must be in [0, 1), got {value}")


def sd_change_from_pre_post(sd_pre: float, sd_post: float, rho_pre_post: float) -> float:
    """SD of change = post - pre given SD(pre), SD(post) and their correlation.

    Var(change) = Var(post) + Var(pre) - 2*rho*sd_post*sd_pre.
    """
    validate_positive(sd_pre, "sd_pre")
    validate_positive(sd_post, "sd_post")
    validate_probability(abs(rho_pre_post), "abs(rho_pre_post)", allow_one=True)
    var_change = sd_post**2 + sd_pre**2 - 2.0 * rho_pre_post * sd_post * sd_pre
    if var_change < 0:
        raise ValueError(f"Derived var_change is negative ({var_change}). Check correlation value.")
    return float(math.sqrt(max(1e-12, var_change)))


def sd_diff_from_sd_change_icc(sd_change: float, icc: float) -> float:
    """SD of within-pair difference in change: s_d = sqrt(2 * (1-ICC) * sd_change^2)."""
    validate_positive(sd_change, "sd_change")
    validate_icc(icc, "icc")
    return float(math.sqrt(max(1e-12, 2.0 * max(0.0, 1.0 - icc) * (sd_change ** 2))))


def relative_efficiency(icc: float) -> float:
    """Relative efficiency of paired vs independent individuals: RE = 1 / [2(1-ICC)].

    Values > 1 indicate fewer pairs than independent individuals are required
    for the same power. At ICC=0.50, RE≈1.0 (parity). For ICC>0.50, paired design
    becomes increasingly efficient.
    """
    validate_icc(icc, "icc")
    denom = 2.0 * max(1e-12, 1.0 - float(icc))
    return 1.0 / denom


def apply_contamination(effect_abs: float, contamination_rate: float, contamination_effect: float) -> float:
    """Reduce observed effect due to contamination.
    
    If contamination_rate fraction of controls adopt contamination_effect fraction
    of intervention, the observed between-group difference becomes:
    effect_obs = effect_abs * (1 - contamination_rate * contamination_effect)
    
    Example: 30% of controls adopt 50% of intervention behaviors
    -> effect_obs = effect_abs * (1 - 0.30 * 0.50) = 0.85 * effect_abs
    """
    validate_probability(contamination_rate, "contamination_rate", allow_one=True)
    validate_probability(contamination_effect, "contamination_effect", allow_one=True)
    reduction_factor = 1.0 - (contamination_rate * contamination_effect)
    return float(max(0.0, effect_abs * reduction_factor))


def inflate_for_attrition(n_completing: int, attrition_rate: float) -> int:
    """Calculate enrollment sample size needed to achieve n_completing after attrition."""
    validate_probability(attrition_rate, "attrition_rate", allow_one=False)
    return int(math.ceil(n_completing / (1.0 - attrition_rate)))


def _z_alpha_two_sided(alpha: float) -> float:
    """z for two-sided alpha (via SciPy)."""
    return float(sps.norm.ppf(1.0 - alpha / 2.0))


def _t_alpha_two_sided(alpha: float, df: int) -> float:
    """t critical for two-sided alpha with df (via SciPy)."""
    return float(sps.t.ppf(1.0 - alpha / 2.0, df))


def _two_sided_power_normal(lambda_nc: float, alpha: float) -> float:
    """Two-sided power using normal approx for noncentral mean lambda.

    T ~ Normal(lambda_nc, 1). Reject if |T| > z_{alpha/2}.
    Power = P(T > z) + P(T < -z) = 1 - Phi(z - lambda) + Phi(-z - lambda).
    """
    z = float(sps.norm.ppf(1.0 - alpha / 2.0))
    Phi = sps.norm.cdf
    return float((1.0 - Phi(z - lambda_nc)) + Phi(-z - lambda_nc))


def d_paired(effect_abs: float, sd_change: float, icc: float) -> float:
    """Cohen's d for paired design using SD of the within-pair difference.

    d = effect_abs / s_d, where s_d = sqrt(2 * (1-ICC) * sd_change^2).
    """
    sd_d = sd_diff_from_sd_change_icc(sd_change, icc)
    return float(effect_abs / max(1e-12, sd_d))


def analytic_power_paired(n_pairs: int, effect_abs: float, sd_change: float, icc_eff: float, alpha: float) -> float:
    """Analytic two-sided power for paired t-test with robust fallbacks.

    Prefers noncentral t (SciPy) but falls back to a high-quality normal
    approximation if parameters are extreme or SciPy returns NaN.
    """
    if n_pairs <= 1:
        return 0.0
    d = d_paired(effect_abs, sd_change, icc_eff)
    lam = math.sqrt(n_pairs) * d
    df = n_pairs - 1

    # For very large noncentrality, the normal approximation is extremely accurate
    # and more numerically stable than nct.cdf in some SciPy builds.
    if lam > 20.0:
        pw = _two_sided_power_normal(lam, alpha)
        return float(max(0.0, min(1.0, pw)))

    try:
        tcrit = float(sps.t.ppf(1.0 - alpha / 2.0, df))
        # Power = P(T > tcrit) + P(T < -tcrit), T ~ nct(df, lam)
        cdf_pos = sps.nct.cdf(tcrit, df, lam)
        cdf_neg = sps.nct.cdf(-tcrit, df, lam)
        pw = float((1.0 - cdf_pos) + cdf_neg)
        if math.isnan(pw):
            pw = _two_sided_power_normal(lam, alpha)
        return float(max(0.0, min(1.0, pw)))
    except Exception:
        # Fallback to normal approx on any SciPy error
        pw = _two_sided_power_normal(lam, alpha)
        return float(max(0.0, min(1.0, pw)))


def analytic_pairs_for_power(target_power: float, effect_abs: float, sd_change: float, icc_eff: float,
                             alpha: float, n_lo: int = 10, n_hi: int = 50000) -> Tuple[int, float]:
    """Find minimum number of pairs for target power using binary search."""
    validate_probability(target_power, "target_power", allow_zero=False, allow_one=False)
    low, high = n_lo, n_hi
    best_n, best_pw = high, 0.0
    while low <= high:
        mid = (low + high) // 2
        pw = analytic_power_paired(mid, effect_abs, sd_change, icc_eff, alpha)
        if pw >= target_power:
            best_n, best_pw = mid, pw
            high = mid - 1
        else:
            low = mid + 1
    return best_n, best_pw


def analytic_mde(n_pairs: int, target_power: float, sd_change: float, icc_eff: float, alpha: float,
                 max_iter: int = 60) -> float:
    """Minimal detectable absolute effect for given n_pairs and target power.

    Uses bisection on effect_abs with relative tolerance.
    """
    validate_probability(target_power, "target_power", allow_zero=False, allow_one=False)
    # Set broad search bounds based on SD of differences
    sd_d = sd_diff_from_sd_change_icc(sd_change, icc_eff)
    lo, hi = 1e-6, 10.0 * sd_d  # 10 SDs is generous upper bound
    for iteration in range(max_iter):
        mid = 0.5 * (lo + hi)
        pw = analytic_power_paired(n_pairs, mid, sd_change, icc_eff, alpha)
        # Use relative tolerance scaled to current estimate
        rel_tol = 1e-4 * max(mid, 0.01)
        if abs(hi - lo) < rel_tol:
            return mid
        if pw >= target_power:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


# ---------- Simulation for within-pair RCT ----------

@dataclass
class AgeTwinSpec:
    n_pairs: int
    # Absolute beneficial effect on the individual change outcome (always positive input)
    # DunedinPACE: absolute slowing (e.g., 0.05 for 5%)
    # GrimAge: years reduction (e.g., 2.0)
    effect_abs: float
    sd_change: float
    # Zygosity mix (for simulation and effective ICC computation)
    prop_mz: float = 0.5
    # Defaults align with moderate assumptions: ICC_MZ≈0.60, ICC_DZ≈0.40
    icc_mz: float = 0.60
    icc_dz: float = 0.40
    alpha: float = 0.05
    seed: Optional[int] = 12345
    # Contamination parameters
    contamination_rate: float = 0.0
    contamination_effect: float = 0.0

    def __post_init__(self):
        """Validate inputs after initialization."""
        if self.n_pairs < 1:
            raise ValueError(f"n_pairs must be >= 1, got {self.n_pairs}")
        validate_positive(self.effect_abs, "effect_abs")
        validate_positive(self.sd_change, "sd_change")
        validate_probability(self.prop_mz, "prop_mz", allow_one=True)
        validate_icc(self.icc_mz, "icc_mz")
        validate_icc(self.icc_dz, "icc_dz")
        validate_probability(self.alpha, "alpha", allow_zero=False, allow_one=False)
        validate_probability(self.contamination_rate, "contamination_rate", allow_one=True)
        validate_probability(self.contamination_effect, "contamination_effect", allow_one=True)

    def effective_icc(self) -> float:
        """Variance-weighted effective ICC for analytic formulas under a mix of ICCs.

        Approximates Var(d) by weighted average across pair types:
        Var(d) = 2 * sd^2 * [prop_mz*(1-icc_mz) + (1-prop_mz)*(1-icc_dz)].
        Equivalently, ICC_eff = 1 - [prop_mz*(1-icc_mz) + (1-prop_mz)*(1-icc_dz)].
        """
        w = max(0.0, min(1.0, self.prop_mz))
        one_minus_icc = w * (1.0 - self.icc_mz) + (1.0 - w) * (1.0 - self.icc_dz)
        return max(0.0, min(0.999999, 1.0 - one_minus_icc))

    def observed_effect(self) -> float:
        """Return effect size accounting for contamination."""
        return apply_contamination(self.effect_abs, self.contamination_rate, self.contamination_effect)


def _compute_paired_pval(d: np.ndarray) -> float:
    """Compute two-sided p-value for paired differences."""
    n = len(d)
    if n <= 1:
        return 1.0
    d_bar = float(np.mean(d))
    s_d = float(np.std(d, ddof=1))
    se = s_d / math.sqrt(n)
    if se <= 0:
        # Degenerate case: zero variance in differences
        # If the mean difference is also (near) zero, this should not be significant.
        if abs(d_bar) < 1e-12:
            return 1.0
        # Non-zero mean with zero SE implies infinite t-stat (significant)
        return 0.0
    t_stat = abs(d_bar / se)
    df = n - 1
    try:
        return 2.0 * sps.t.sf(t_stat, df)
    except Exception:
        # Normal approximation if SciPy errors
        Phi = lambda x: 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
        return 2.0 * (1.0 - Phi(t_stat))


def _simulate_pairs(spec: AgeTwinSpec, sims: int = 2000) -> Tuple[float, float, float]:
    """Monte Carlo power, returning (power, avg_estimated_effect_magnitude, std_err).

    Simulates within-pair randomization explicitly and estimates via paired t-test.
    Returns the absolute magnitude of estimated effects (always positive).
    """
    rng = np.random.default_rng(spec.seed)
    hits = 0
    ests = []
    alpha = spec.alpha
    effect_obs = spec.observed_effect()  # Account for contamination
    
    for s in range(sims):
        # Sample zygosity for each pair
        z_is_mz = rng.random(spec.n_pairs) < spec.prop_mz
        # Pair random intercept b ~ N(0, sqrt(ICC * sd^2)), residual e ~ N(0, sqrt((1-ICC) * sd^2))
        # Treatment assignment: for each pair, randomly pick which twin is treated
        # We simulate change outcome y where treated twins have mean -effect_obs (reduction/slowing)
        y_treat = np.empty(spec.n_pairs, dtype=float)
        y_ctrl = np.empty(spec.n_pairs, dtype=float)
        
        for i in range(spec.n_pairs):
            icc = spec.icc_mz if z_is_mz[i] else spec.icc_dz
            sigma2 = spec.sd_change ** 2
            sd_b = math.sqrt(max(0.0, icc) * sigma2)
            sd_e = math.sqrt(max(0.0, 1.0 - icc) * sigma2)
            b = rng.normal(0.0, sd_b)
            e1 = rng.normal(0.0, sd_e)
            e2 = rng.normal(0.0, sd_e)
            # Randomly assign which twin is treated
            if rng.integers(0, 2) == 1:
                # Twin 1 treated: mean change is -effect_obs (beneficial = reduction)
                y_treat[i] = -effect_obs + b + e1
                y_ctrl[i] = 0.0 + b + e2
            else:
                # Twin 2 treated
                y_treat[i] = -effect_obs + b + e2
                y_ctrl[i] = 0.0 + b + e1

        # Paired differences: d = y_treat - y_ctrl (negative when treatment beneficial)
        d = y_treat - y_ctrl
        pval = _compute_paired_pval(d)

        if pval < alpha:
            hits += 1
        # Store absolute magnitude of estimated effect (convert negative to positive)
        d_bar = float(np.mean(d))
        ests.append(abs(d_bar))

    power = hits / sims
    avg_effect_mag = float(np.mean(ests)) if ests else float('nan')
    se_power = math.sqrt(power * (1 - power) / sims)  # Standard error of power estimate
    return power, avg_effect_mag, se_power


def _simulate_co_primary(
    spec1: AgeTwinSpec,
    spec2: AgeTwinSpec,
    sims: int = 2000,
    alpha: float = 0.025,
    pair_effect_corr: float = 0.8,
) -> Tuple[float, float, float]:
    """Simulate joint power for co-primary endpoints with shared assignment and correlated pair effects.

    - Uses one random treatment assignment per pair shared across endpoints (same trial randomization).
    - Shares zygosity across endpoints.
    - Correlates pair random effects across endpoints with correlation ``pair_effect_corr``.

    Returns (joint_power, power1_alone, power2_alone) where joint_power is P(p1 < alpha and p2 < alpha).
    """
    if spec1.n_pairs != spec2.n_pairs:
        raise ValueError("Both endpoints must have same n_pairs for co-primary analysis")
    if spec1.seed is None or spec2.seed is None:
        raise ValueError("Both specs must have seeds for co-primary simulation")
    
    # Use same random seed to maintain correlation structure across endpoints
    rng = np.random.default_rng(spec1.seed)
    hits_joint = 0
    hits1 = 0
    hits2 = 0
    
    effect_obs1 = spec1.observed_effect()
    effect_obs2 = spec2.observed_effect()
    
    for s in range(sims):
        # Sample shared zygosity and shared treatment assignment per pair
        z_is_mz = rng.random(spec1.n_pairs) < spec1.prop_mz
        treat_first = rng.integers(0, 2, size=spec1.n_pairs) == 1

        # Prepare arrays
        y1_treat = np.empty(spec1.n_pairs, dtype=float)
        y1_ctrl = np.empty(spec1.n_pairs, dtype=float)
        y2_treat = np.empty(spec2.n_pairs, dtype=float)
        y2_ctrl = np.empty(spec2.n_pairs, dtype=float)

        # Clamp correlation to [0, 0.999] for stability and biological plausibility
        rho = max(0.0, min(0.999, float(pair_effect_corr)))

        for i in range(spec1.n_pairs):
            # Endpoint-specific ICCs and scales
            icc1 = spec1.icc_mz if z_is_mz[i] else spec1.icc_dz
            icc2 = spec2.icc_mz if z_is_mz[i] else spec2.icc_dz
            s2_1 = spec1.sd_change ** 2
            s2_2 = spec2.sd_change ** 2
            sd_b1 = math.sqrt(max(0.0, icc1) * s2_1)
            sd_b2 = math.sqrt(max(0.0, icc2) * s2_2)
            sd_e1 = math.sqrt(max(0.0, 1.0 - icc1) * s2_1)
            sd_e2 = math.sqrt(max(0.0, 1.0 - icc2) * s2_2)

            # Correlated pair effects via shared z's
            z1 = rng.normal(0.0, 1.0)
            z2 = rng.normal(0.0, 1.0)
            b1 = sd_b1 * z1
            b2 = sd_b2 * (rho * z1 + math.sqrt(max(0.0, 1.0 - rho * rho)) * z2)

            # Residuals for two twins per endpoint
            e1a = rng.normal(0.0, sd_e1)
            e1b = rng.normal(0.0, sd_e1)
            e2a = rng.normal(0.0, sd_e2)
            e2b = rng.normal(0.0, sd_e2)

            if treat_first[i]:
                # Twin A treated
                y1_treat[i] = -effect_obs1 + b1 + e1a
                y1_ctrl[i] = 0.0 + b1 + e1b
                y2_treat[i] = -effect_obs2 + b2 + e2a
                y2_ctrl[i] = 0.0 + b2 + e2b
            else:
                # Twin B treated
                y1_treat[i] = -effect_obs1 + b1 + e1b
                y1_ctrl[i] = 0.0 + b1 + e1a
                y2_treat[i] = -effect_obs2 + b2 + e2b
                y2_ctrl[i] = 0.0 + b2 + e2a

        # Paired differences
        d1 = y1_treat - y1_ctrl
        d2 = y2_treat - y2_ctrl
        pval1 = _compute_paired_pval(d1)
        pval2 = _compute_paired_pval(d2)
        
        if pval1 < alpha:
            hits1 += 1
        if pval2 < alpha:
            hits2 += 1
        if pval1 < alpha and pval2 < alpha:
            hits_joint += 1
    
    return hits_joint / sims, hits1 / sims, hits2 / sims


# ---------- CLI ----------

# ----- Individual-level (two-sample) helpers -----

def _apply_ancova_sd(sd_change: float, ancova_r2: Optional[float]) -> float:
    if ancova_r2 is None:
        return float(sd_change)
    validate_probability(ancova_r2, "ancova_r2", allow_one=False)
    factor = math.sqrt(max(0.0, 1.0 - float(ancova_r2)))
    return float(sd_change) * factor


def design_effect(icc: float, m: int = 2) -> float:
    validate_icc(icc, "icc")
    if m < 1:
        raise ValueError("m must be >= 1 for design effect")
    return 1.0 + (int(m) - 1) * float(icc)


def analytic_power_two_sample(n_per_group: int, effect_abs: float, sd_change: float, alpha: float) -> float:
    """Analytic two-sided power for equal-size two-sample t-test on change scores.

    Assumes pooled SD = sd_change in both arms, n_per_group individuals in each arm.
    Uses noncentral t with ncp = |Δ| / (σ * sqrt(2/n)).
    """
    if n_per_group <= 1:
        return 0.0
    validate_positive(effect_abs, "effect_abs")
    validate_positive(sd_change, "sd_change")
    validate_probability(alpha, "alpha", allow_zero=False, allow_one=False)

    # Noncentrality
    se = sd_change * math.sqrt(2.0 / float(n_per_group))
    lam = float(effect_abs) / max(1e-12, se)
    df = max(1, 2 * n_per_group - 2)

    if lam > 20.0:
        # Normal approx stable for very large ncp
        z = float(sps.norm.ppf(1.0 - alpha / 2.0))
        Phi = sps.norm.cdf
        return float((1.0 - Phi(z - lam)) + Phi(-z - lam))

    try:
        tcrit = float(sps.t.ppf(1.0 - alpha / 2.0, df))
        cdf_pos = sps.nct.cdf(tcrit, df, lam)
        cdf_neg = sps.nct.cdf(-tcrit, df, lam)
        pw = float((1.0 - cdf_pos) + cdf_neg)
        if math.isnan(pw):
            z = float(sps.norm.ppf(1.0 - alpha / 2.0))
            Phi = sps.norm.cdf
            pw = float((1.0 - Phi(z - lam)) + Phi(-z - lam))
        return float(max(0.0, min(1.0, pw)))
    except Exception:
        z = float(sps.norm.ppf(1.0 - alpha / 2.0))
        Phi = sps.norm.cdf
        pw = float((1.0 - Phi(z - lam)) + Phi(-z - lam))
        return float(max(0.0, min(1.0, pw)))


def analytic_n_per_group_for_power(target_power: float, effect_abs: float, sd_change: float, alpha: float,
                                   n_lo: int = 2, n_hi: int = 100000) -> int:
    """Binary search for minimal n per group achieving target power for two-sample test."""
    validate_probability(target_power, "target_power", allow_zero=False, allow_one=False)
    low, high = max(2, int(n_lo)), int(n_hi)
    best = high
    while low <= high:
        mid = (low + high) // 2
        pw = analytic_power_two_sample(mid, effect_abs, sd_change, alpha)
        if pw >= target_power:
            best = mid
            high = mid - 1
        else:
            low = mid + 1
    return int(best)

def _parse_csv_numbers(s: Optional[str], cast=float) -> Optional[list]:
    """Parse a comma-separated list of numbers into a list with the given cast.

    Returns None if s is None. Strips whitespace and ignores empty items.
    """
    if s is None:
        return None
    items = []
    for part in str(s).split(','):
        part = part.strip()
        if not part:
            continue
        items.append(cast(part))
    return items

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Power analysis for within-pair twin RCT (biological age)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    p.add_argument("--mode", choices=[
        "power",
        "pairs-for-power",
        "mde",
        "curve",
        "co-primary-power",
        "study",
        "two-sample-power",
        "two-sample-n-for-power",
    ],
                   default="power", help="Analysis mode")
    p.add_argument("--endpoint", choices=["dunedinpace", "grimage", "custom"], default="dunedinpace",
                   help="Primary endpoint type (within-pair analysis is primary)")
    p.add_argument("--n-pairs", type=int, default=700,
                   help="Number of COMPLETING twin pairs (mode=power/mde/curve)")
    p.add_argument("--target-power", type=float, default=0.80,
                   help="Target power (mode=pairs-for-power/mde)")
    
    # Effect specification (absolute on change scale)
    p.add_argument("--effect-abs", type=float, default=None,
                   help="Absolute beneficial effect (e.g., 0.025 DunedinPACE or 2.0 years GrimAge)")
    p.add_argument("--d-std", type=float, default=None,
                   help="Paired standardized effect size d (overrides --effect-abs if set)")
    
    # Endpoint-friendly helpers
    p.add_argument("--effect-pct", type=float, default=None,
                   help="DunedinPACE percent slowing, e.g., 2.5 for 2.5% (converted to 0.025)")
    p.add_argument("--effect-years", type=float, default=None,
                   help="GrimAge years reduction (alternative to --effect-abs)")
    
    # Variability
    p.add_argument("--sd-change", type=float, default=None,
                   help="SD of individual change (required unless --sd-pre/--sd-post provided)")
    p.add_argument("--sd-pre", type=float, default=None,
                   help="SD at baseline (to derive sd_change with sd-post and rho)")
    p.add_argument("--sd-post", type=float, default=None,
                   help="SD at follow-up (to derive sd_change with sd-pre and rho)")
    p.add_argument("--rho-pre-post", type=float, default=None,
                   help="Correlation between pre and post (to derive sd_change)")
    
    # ICC and zygosity mix
    p.add_argument("--icc-mz", type=float, default=0.60,
                   help="Within-pair ICC for change (monozygotic twins)")
    p.add_argument("--icc-dz", type=float, default=0.40,
                   help="Within-pair ICC for change (dizygotic twins)")
    p.add_argument("--prop-mz", type=float, default=0.5,
                   help="Proportion of twin pairs that are MZ")
    
    # Contamination modeling
    p.add_argument("--contamination-rate", type=float, default=0.0,
                   help="Expected proportion of controls adopting intervention (0-1)")
    p.add_argument("--contamination-effect", type=float, default=0.0,
                   help="Fraction of full effect controls receive if contaminated (0-1)")
    
    # Attrition
    p.add_argument("--attrition-rate", type=float, default=0.0,
                   help="Expected dropout rate (0-1). Inflates sample size for enrollment.")
    
    # Inference
    p.add_argument("--alpha", type=float, default=0.05,
                   help="Significance level (two-sided). Use 0.025 for co-primary endpoints.")
    
    # Simulation controls
    p.add_argument("--sims", type=int, default=2000,
                   help="Monte Carlo iterations (when using simulation)")
    p.add_argument("--seed", type=int, default=12345, help="Random seed")
    p.add_argument("--use-simulation", action="store_true",
                   help="Use Monte Carlo simulation instead of analytic formulas (recommended for final calculations)")
    
    # Curve mode
    p.add_argument("--curve-start", type=int, default=50)
    p.add_argument("--curve-stop", type=int, default=2000)
    p.add_argument("--curve-step", type=int, default=50)
    
    # Co-primary mode
    p.add_argument("--endpoint2", choices=["dunedinpace", "grimage", "custom"], default=None,
                   help="Secondary endpoint for co-primary analysis")
    # Accept both naming variants for co-primary endpoint 2
    p.add_argument("--effect2-abs", dest="effect2_abs", type=float, default=None,
                   help="Effect size for endpoint2")
    p.add_argument("--effect-abs2", dest="effect2_abs", type=float, default=None,
                   help="Effect size for endpoint2 (alternate naming)")
    p.add_argument("--effect2-pct", dest="effect2_pct", type=float, default=None)
    p.add_argument("--effect-pct2", dest="effect_pct2", type=float, default=None)
    p.add_argument("--effect2-years", dest="effect2_years", type=float, default=None)
    p.add_argument("--effect-years2", dest="effect_years2", type=float, default=None)
    p.add_argument("--sd2-change", dest="sd2_change", type=float, default=None,
                   help="SD of change for endpoint2")
    p.add_argument("--sd-change2", dest="sd2_change", type=float, default=None,
                   help="SD of change for endpoint2 (alternate naming)")
    p.add_argument("--pair-effect-corr", type=float, default=0.8,
                   help="Correlation (0-1) of pair random effects across endpoints in co-primary simulation")

    # Individual-level (two-sample) analysis options
    p.add_argument("--n-per-group", type=int, default=None,
                   help="Individuals per group for two-sample power (mode=two-sample-power)")
    p.add_argument("--n-total", type=int, default=None,
                   help="Total individuals for two-sample power (alternative to --n-per-group)")
    p.add_argument("--ancova-r2", type=float, default=None,
                   help="ANCOVA baseline R^2 to reduce SD(change): sd *= sqrt(1-R^2)")
    p.add_argument("--deff-icc", type=float, default=None,
                   help="Optional ICC for design effect inflation in individual-level N (DEFF = 1 + (m-1)*ICC)")
    p.add_argument("--deff-m", type=int, default=2,
                   help="Cluster size m for design effect (twins: m=2). Used only when --deff-icc is provided")

    # Study preset (LLM multi-domain twin RCT)
    p.add_argument("--study-preset", choices=["twins-llm"], default=None,
                   help="Apply study defaults for the LLM multi-domain twin RCT (DunedinPACE primary)")
    p.add_argument("--study-effect-pct", type=str, default=None,
                   help="Comma-separated DunedinPACE percent slowing to grid over in study mode (e.g., '2,2.5,3')")
    p.add_argument("--study-n-pairs", type=str, default=None,
                   help="Comma-separated completing pairs to grid over in study mode (e.g., '150,165,185')")
    p.add_argument("--study-attrition-range", type=str, default="0.25,0.30",
                   help="Comma-separated attrition range to display enrollment inflation (e.g., '0.25,0.30')")

    return p


def _resolve_effect_abs(args: argparse.Namespace, sd_change: Optional[float], 
                       icc_eff: Optional[float], suffix: str = "") -> float:
    """Resolve effect size from various input options.

    Supports both naming conventions for co-primary endpoint 2:
    - Dasherized with index after base (e.g., --effect-abs2 → effect_abs2)
    - Indexed before base (e.g., --effect2-abs → effect2_abs)
    """
    attr_name = lambda base: base + suffix

    def _first_attr(*names):
        for n in names:
            if hasattr(args, n):
                v = getattr(args, n)
                if v is not None:
                    return v
        return None

    # Try direct specification first (accept both syntaxes)
    effect_abs = _first_attr(
        attr_name("effect_abs"),
        "effect_abs2", "effect2_abs"
    )
    if effect_abs is not None:
        return float(effect_abs)

    # Endpoint-aware helpers
    endpoint = _first_attr(attr_name("endpoint"), "endpoint2")
    effect_pct = _first_attr(attr_name("effect_pct"), "effect_pct2", "effect2_pct")
    effect_years = _first_attr(attr_name("effect_years"), "effect_years2", "effect2_years")

    if endpoint == "dunedinpace" and effect_pct is not None:
        return float(effect_pct) / 100.0
    if endpoint == "grimage" and effect_years is not None:
        return float(effect_years)

    # Standardized effect size
    if getattr(args, "d_std", None) is not None:
        if sd_change is None or icc_eff is None:
            raise SystemExit("--d-std requires sd_change and ICC to compute effect_abs.")
        sd_d = sd_diff_from_sd_change_icc(sd_change, icc_eff)
        return float(args.d_std) * sd_d

    raise SystemExit(
        f"Effect magnitude{suffix} required. Use --effect-abs{suffix} or "
        f"endpoint-specific --effect-pct{suffix}/--effect-years{suffix}"
    )


def _resolve_sd_change(args: argparse.Namespace, suffix: str = "") -> float:
    """Resolve SD of change from various input options.

    Accepts both sd-change2 and sd2-change naming for co-primary endpoint 2.
    """
    attr_name = lambda base: base + suffix

    def _first_attr(*names):
        for n in names:
            if hasattr(args, n):
                v = getattr(args, n)
                if v is not None:
                    return v
        return None

    sd_change = _first_attr(attr_name("sd_change"), "sd_change2", "sd2_change")
    if sd_change is not None:
        return float(sd_change)

    sd_pre = getattr(args, attr_name("sd_pre"), None)
    sd_post = getattr(args, attr_name("sd_post"), None)

    if sd_pre is not None and sd_post is not None and getattr(args, "rho_pre_post", None) is not None:
        return sd_change_from_pre_post(float(sd_pre), float(sd_post), float(args.rho_pre_post))

    raise SystemExit(
        f"SD of change{suffix} required. Use --sd-change{suffix} or "
        f"(--sd-pre{suffix}, --sd-post{suffix}, --rho-pre-post)"
    )


def _print_contamination_note(contamination_rate: float, contamination_effect: float, 
                             effect_original: float, effect_observed: float):
    """Print contamination adjustment summary."""
    if contamination_rate > 0:
        reduction_pct = (1 - effect_observed / effect_original) * 100
        print(f"\n  CONTAMINATION ADJUSTMENT:")
        print(f"    Rate: {contamination_rate:.1%} of controls")
        print(f"    Effect in controls: {contamination_effect:.1%} of intervention effect")
        print(f"    Observed effect (adjusted): {effect_observed:.4f} ({reduction_pct:.1f}% reduction)")
        print(f"    Original effect (no contamination): {effect_original:.4f}")


def run_cli():
    args = _build_parser().parse_args()
    
    # Resolve common parameters
    try:
        # Apply study preset defaults before resolution (non-destructive if user overrides)
        if getattr(args, "study_preset", None) == "twins-llm":
            # Set SD(change) default if not provided via any path; endpoint default already set by argparse
            if all(getattr(args, k, None) is None for k in ("sd_change", "sd_pre", "sd_post")):
                args.sd_change = 0.10
            # ICCs and mix defaults if user left as CLI defaults (already set), leave as-is
            # Alpha default remains 0.05 unless overridden
            # For study mode, seed and sims are used as provided
            # Provide defaults for study grids if user didn't set them
            if getattr(args, "mode", None) == "study":
                if getattr(args, "study_effect_pct", None) is None:
                    args.study_effect_pct = "2,2.5,3"
                if getattr(args, "study_n_pairs", None) is None:
                    args.study_n_pairs = "150,165,185"

        # Protocol-consistent default for co-primary mode:
        # If user didn't override alpha and selected co-primary mode, set alpha=0.025 per endpoint.
        if getattr(args, "mode", None) == "co-primary-power" and abs(float(args.alpha) - 0.05) < 1e-6:
            args.alpha = 0.025
            print("[info] Co-primary mode detected: setting alpha per endpoint to 0.025 (Bonferroni)", file=sys.stderr)

        # Validate co-primary correlation is in [0,1]
        if getattr(args, "mode", None) == "co-primary-power":
            try:
                corr = float(args.pair_effect_corr)
            except Exception:
                raise SystemExit("--pair-effect-corr must be a number in [0,1]")
            if not (0.0 <= corr <= 1.0):
                raise SystemExit(f"--pair-effect-corr must be in [0,1], got {args.pair_effect_corr}")

        sd_change = _resolve_sd_change(args, "")
        
        # Preliminary spec to compute effective ICC
        spec_temp = AgeTwinSpec(
            n_pairs=args.n_pairs,
            effect_abs=1.0,  # placeholder
            sd_change=sd_change,
            prop_mz=args.prop_mz,
            icc_mz=args.icc_mz,
            icc_dz=args.icc_dz,
            alpha=args.alpha,
            seed=args.seed,
            contamination_rate=args.contamination_rate,
            contamination_effect=args.contamination_effect,
        )
        icc_eff = spec_temp.effective_icc()

        # Resolve effect size only for modes that require it
        if args.mode in {"power", "pairs-for-power", "curve"}:
            effect_abs = _resolve_effect_abs(args, sd_change, icc_eff, "")
        elif args.mode == "co-primary-power":
            # Primary endpoint effect is needed as well
            effect_abs = _resolve_effect_abs(args, sd_change, icc_eff, "")
        else:  # mde
            effect_abs = 1.0  # placeholder; not used for MDE computation

        # Build final spec
        spec = AgeTwinSpec(
            n_pairs=args.n_pairs,
            effect_abs=effect_abs,
            sd_change=sd_change,
            prop_mz=args.prop_mz,
            icc_mz=args.icc_mz,
            icc_dz=args.icc_dz,
            alpha=args.alpha,
            seed=args.seed,
            contamination_rate=args.contamination_rate,
            contamination_effect=args.contamination_effect,
        )
        
    except (ValueError, SystemExit) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Calculate derived quantities
    icc_eff = spec.effective_icc()
    effect_obs = spec.observed_effect()
    sd_d = sd_diff_from_sd_change_icc(spec.sd_change, icc_eff)
    
    # MODE: power
    if args.mode == "power":
        if args.use_simulation:
            pw, avg_mag, se_pw = _simulate_pairs(spec, sims=args.sims)
        else:
            pw = analytic_power_paired(spec.n_pairs, effect_obs, spec.sd_change, icc_eff, spec.alpha)
            avg_mag = effect_obs
            se_pw = math.sqrt(pw * (1 - pw) / args.sims) if args.sims > 0 else 0
        
        n_enroll = inflate_for_attrition(spec.n_pairs, args.attrition_rate)
        
        print("=" * 70)
        print("POWER ANALYSIS: Within-Pair Twin RCT")
        print("=" * 70)
        print(f"\nEndpoint: {args.endpoint}")
        print(f"Method: {'Monte Carlo simulation' if args.use_simulation else 'Analytic (paired t-test)'}")
        
        print(f"\nSAMPLE SIZE:")
        print(f"  Completing pairs: {spec.n_pairs}")
        if args.attrition_rate > 0:
            print(f"  Expected attrition: {args.attrition_rate:.1%}")
            print(f"  Required enrollment: {n_enroll} pairs ({n_enroll * 2} individuals)")
        
        print(f"\nEFFECT SIZE:")
        print(f"  Specified effect (beneficial): {spec.effect_abs:.4f}")
        _print_contamination_note(args.contamination_rate, args.contamination_effect, 
                                 spec.effect_abs, effect_obs)
        print(f"  SD(individual change): {spec.sd_change:.4f}")
        print(f"  Cohen's d (paired): {d_paired(effect_obs, spec.sd_change, icc_eff):.3f}")
        
        print(f"\nDESIGN PARAMETERS:")
        print(f"  ICCs: MZ={spec.icc_mz:.3f}, DZ={spec.icc_dz:.3f}")
        print(f"  Proportion MZ: {spec.prop_mz:.2f}")
        print(f"  Effective ICC (variance-weighted): {icc_eff:.3f}")
        print(f"  SD(within-pair difference): {sd_d:.4f}")
        # Relative efficiency vs. independent individuals
        re = relative_efficiency(icc_eff)
        deff = 1.0 + icc_eff  # m=2 twins
        n_eff = int(math.floor((spec.n_pairs * 2) / max(1e-12, deff)))
        print(f"  Relative efficiency (RE = 1/[2(1-ICC)]): {re:.2f}×")
        print(f"  Effective independent N (DEFF=1+ICC): {n_eff}")
        print(f"  Significance level (two-sided): α={spec.alpha:.3f}")
        print(f"  Timepoint: Baseline → Week 24 (primary endpoint)")
        
        print(f"\nRESULTS:")
        print(f"  Statistical power: {pw:.3f} ({pw*100:.1f}%)")
        if args.use_simulation:
            ci_lower = max(0, pw - 1.96 * se_pw)
            ci_upper = min(1, pw + 1.96 * se_pw)
            print(f"  95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")
            print(f"  Avg estimated effect magnitude: {avg_mag:.4f}")
        print("=" * 70)
    
    # MODE: pairs-for-power
    elif args.mode == "pairs-for-power":
        if args.use_simulation:
            # Binary search using simulation
            low, high = 10, 50000
            best_n, best_pw = high, 0.0
            while low <= high:
                mid = (low + high) // 2
                spec_mid = AgeTwinSpec(
                    n_pairs=mid, effect_abs=spec.effect_abs, sd_change=spec.sd_change,
                    prop_mz=spec.prop_mz, icc_mz=spec.icc_mz, icc_dz=spec.icc_dz,
                    alpha=spec.alpha, seed=spec.seed,
                    contamination_rate=args.contamination_rate,
                    contamination_effect=args.contamination_effect
                )
                pw, _, _ = _simulate_pairs(spec_mid, sims=max(500, args.sims // 2))
                if pw >= args.target_power:
                    best_n, best_pw = mid, pw
                    high = mid - 1
                else:
                    low = mid + 1
        else:
            best_n, best_pw = analytic_pairs_for_power(
                args.target_power, effect_obs, spec.sd_change, icc_eff, spec.alpha
            )
        
        n_enroll = inflate_for_attrition(best_n, args.attrition_rate)
        
        print("=" * 70)
        print("SAMPLE SIZE CALCULATION: Within-Pair Twin RCT")
        print("=" * 70)
        print(f"\nEndpoint: {args.endpoint}")
        print(f"Target power: {args.target_power:.3f} ({args.target_power*100:.1f}%)")
        
        print(f"\nRESULTS:")
        print(f"  Required COMPLETING pairs: {best_n}")
        print(f"  Achieved power: {best_pw:.3f} ({best_pw*100:.1f}%)")
        if args.attrition_rate > 0:
            print(f"  With {args.attrition_rate:.1%} attrition:")
            print(f"    Required ENROLLMENT: {n_enroll} pairs ({n_enroll * 2} individuals)")
        
        print(f"\nEFFECT SIZE:")
        print(f"  Specified effect: {spec.effect_abs:.4f}")
        _print_contamination_note(args.contamination_rate, args.contamination_effect,
                                 spec.effect_abs, effect_obs)
        print(f"  SD(individual change): {spec.sd_change:.4f}")
        print(f"  Cohen's d (paired): {d_paired(effect_obs, spec.sd_change, icc_eff):.3f}")
        
        print(f"\nDESIGN:")
        print(f"  ICCs: MZ={spec.icc_mz:.3f}, DZ={spec.icc_dz:.3f}, prop_MZ={spec.prop_mz:.2f}")
        print(f"  Effective ICC: {icc_eff:.3f}, SD(pair-diff): {sd_d:.4f}")
        # Efficiency and effective N at the required size
        re = relative_efficiency(icc_eff)
        deff = 1.0 + icc_eff
        n_eff = int(math.floor((best_n * 2) / max(1e-12, deff)))
        print(f"  Relative efficiency (RE): {re:.2f}×; Effective independent N: {n_eff}")
        print(f"  Alpha: {spec.alpha:.3f}, Method: {'simulation' if args.use_simulation else 'analytic'}")
        print("=" * 70)
    
    # MODE: mde
    elif args.mode == "mde":
        # Compute the observed MDE (what the paired test 'sees')
        mde_obs = analytic_mde(spec.n_pairs, args.target_power, spec.sd_change, icc_eff, spec.alpha)

        print("=" * 70)
        print("MINIMAL DETECTABLE EFFECT: Within-Pair Twin RCT")
        print("=" * 70)
        print(f"\nEndpoint: {args.endpoint}")
        print(f"Completing pairs: {spec.n_pairs}")
        print(f"Target power: {args.target_power:.3f}, Alpha: {spec.alpha:.3f}")

        print(f"\nRESULTS:")
        print(f"  Observed MDE (on test scale): {mde_obs:.4f}")

        if args.contamination_rate > 0:
            reduction = 1.0 - (args.contamination_rate * args.contamination_effect)
            reduction = max(1e-12, reduction)  # guard
            mde_true_needed = mde_obs / reduction
            print(f"  Underlying true effect required (given contamination): {mde_true_needed:.4f}")
            print(f"    Reduction factor from contamination: {reduction:.3f}")
            print(f"    (If there were no contamination, the required true effect equals the observed MDE.)")

        print(f"\nDESIGN:")
        print(f"  SD(individual change): {spec.sd_change:.4f}")
        print(f"  ICCs: MZ={spec.icc_mz:.3f}, DZ={spec.icc_dz:.3f}, prop_MZ={spec.prop_mz:.2f}")
        print(f"  Effective ICC: {icc_eff:.3f}, SD(pair-diff): {sd_d:.4f}")
        print("=" * 70)
    
    # MODE: co-primary-power
    elif args.mode == "co-primary-power":
        if not args.use_simulation:
            print("WARNING: Co-primary power requires simulation. Enabling --use-simulation.", 
                  file=sys.stderr)
            args.use_simulation = True
        
        if args.endpoint2 is None:
            raise SystemExit("--endpoint2 required for co-primary mode")
        
        try:
            sd_change2 = _resolve_sd_change(args, "2")
            
            spec2_temp = AgeTwinSpec(
                n_pairs=args.n_pairs,
                effect_abs=1.0,
                sd_change=sd_change2,
                prop_mz=args.prop_mz,
                icc_mz=args.icc_mz,
                icc_dz=args.icc_dz,
                alpha=args.alpha,
                seed=args.seed + 1,  # Different seed for endpoint 2
                contamination_rate=args.contamination_rate,
                contamination_effect=args.contamination_effect,
            )
            icc_eff2 = spec2_temp.effective_icc()
            
            effect_abs2 = _resolve_effect_abs(args, sd_change2, icc_eff2, "2")
            
            spec2 = AgeTwinSpec(
                n_pairs=args.n_pairs,
                effect_abs=effect_abs2,
                sd_change=sd_change2,
                prop_mz=args.prop_mz,
                icc_mz=args.icc_mz,
                icc_dz=args.icc_dz,
                alpha=args.alpha,
                seed=args.seed + 1,
                contamination_rate=args.contamination_rate,
                contamination_effect=args.contamination_effect,
            )
        except (ValueError, SystemExit) as e:
            print(f"Error with endpoint2: {e}", file=sys.stderr)
            sys.exit(1)
        
        # Clamp correlation bounds (allow 0..1 at CLI level)
        rho = max(0.0, min(0.999, float(args.pair_effect_corr)))
        pw_joint, pw1, pw2 = _simulate_co_primary(
            spec, spec2, sims=args.sims, alpha=args.alpha, pair_effect_corr=rho
        )
        
        print("=" * 70)
        print("CO-PRIMARY ENDPOINT POWER ANALYSIS")
        print("=" * 70)
        print(f"\nSample: {spec.n_pairs} completing pairs")
        print(f"Alpha per endpoint: {args.alpha:.3f} (Bonferroni-adjusted)")
        print(f"Simulations: {args.sims:,}")
        
        print(f"\nENDPOINT 1: {args.endpoint}")
        print(f"  Effect: {spec.effect_abs:.4f}, SD(change): {spec.sd_change:.4f}")
        print(f"  Marginal power: {pw1:.3f} ({pw1*100:.1f}%)")
        
        print(f"\nENDPOINT 2: {args.endpoint2}")
        print(f"  Effect: {spec2.effect_abs:.4f}, SD(change): {spec2.sd_change:.4f}")
        print(f"  Marginal power: {pw2:.3f} ({pw2*100:.1f}%)")
        
        print(f"\nJOINT POWER (both significant):")
        print(f"  Probability both p < {args.alpha:.3f}: {pw_joint:.3f} ({pw_joint*100:.1f}%)")
        print(f"  Timepoint: Baseline → Week 24 (primary endpoint)")
        
        if args.contamination_rate > 0:
            print(f"\nNote: Powers account for {args.contamination_rate:.1%} contamination "
                  f"at {args.contamination_effect:.1%} effect")
        print("=" * 70)
    
    # MODE: study (preset grid for the LLM twin RCT)
    elif args.mode == "study":
        # Validate preset
        if args.study_preset != "twins-llm":
            print("Error: --mode study requires --study-preset twins-llm", file=sys.stderr)
            sys.exit(1)

        # Parse grids
        eff_pcts = _parse_csv_numbers(args.study_effect_pct, cast=float) or []
        n_vals = _parse_csv_numbers(args.study_n_pairs, cast=int) or []
        if not eff_pcts or not n_vals:
            print("Error: Provide --study-effect-pct and --study-n-pairs (comma-separated)", file=sys.stderr)
            sys.exit(1)

        # Compute powers across grid
        print("=" * 70)
        print("STUDY SUMMARY — LLM Multi‑Domain Twin RCT (DunedinPACE)")
        print("=" * 70)
        print(f"\nAssumptions (can override):")
        print(f"  Endpoint: DunedinPACE (pace of aging)")
        print(f"  SD(individual change): {spec.sd_change:.4f}")
        print(f"  ICCs: MZ={spec.icc_mz:.3f}, DZ={spec.icc_dz:.3f}, prop_MZ={spec.prop_mz:.2f}")
        print(f"  Effective ICC: {icc_eff:.3f}")
        print(f"  Alpha (two‑sided): {spec.alpha:.3f}")
        print(f"  Method: {'Monte Carlo simulation' if args.use_simulation else 'Analytic (paired t-test)'}")
        if args.use_simulation:
            print(f"  Simulations: {args.sims}")
        if args.contamination_rate > 0:
            print("\n  CONTAMINATION:")
            print(f"    Rate in controls: {args.contamination_rate:.1%}")
            print(f"    Fraction of effect in contaminated controls: {args.contamination_effect:.1%}")
            print("    (Applied to each grid effect before power calculation)")
        print(f"  Timepoints: Baseline → Week 24 (primary); Week 12 interim; Month 12 durability")

        print("\nPOWER GRID (rows = completing pairs; cols = effect % slowing):")
        # Header
        headers = ["n_pairs"] + [f"{ep:.2f}%".rstrip('0').rstrip('.') for ep in eff_pcts]
        print("  " + "  ".join([f"{h:>10s}" for h in headers]))

        for n in n_vals:
            row_vals = [f"{int(n):>10d}"]
            for ep in eff_pcts:
                eff_abs = float(ep) / 100.0
                if args.use_simulation:
                    spec_n = AgeTwinSpec(
                        n_pairs=int(n), effect_abs=eff_abs, sd_change=spec.sd_change,
                        prop_mz=spec.prop_mz, icc_mz=spec.icc_mz, icc_dz=spec.icc_dz,
                        alpha=spec.alpha, seed=spec.seed,
                        contamination_rate=args.contamination_rate,
                        contamination_effect=args.contamination_effect,
                    )
                    pw, _, _ = _simulate_pairs(spec_n, sims=args.sims)
                else:
                    pw = analytic_power_paired(int(n), eff_abs, spec.sd_change, icc_eff, spec.alpha)
                row_vals.append(f"{pw:>10.3f}")
            print("  " + "  ".join(row_vals))

        # Enrollment inflation for the suggested attrition range
        ar = _parse_csv_numbers(args.study_attrition_range, cast=float) or [0.25, 0.30]
        lo_ar = max(0.0, min(0.99, float(ar[0])))
        hi_ar = max(0.0, min(0.99, float(ar[-1])))

        print("\nENROLLMENT PLANNING (pairs):")
        print(f"  Attrition range: {lo_ar:.0%} – {hi_ar:.0%}")
        for n in n_vals:
            enroll_lo = inflate_for_attrition(int(n), lo_ar)
            enroll_hi = inflate_for_attrition(int(n), hi_ar)
            print(f"  Completing {int(n):3d} → Enroll {enroll_lo:3d} – {enroll_hi:3d} pairs ({2*enroll_lo}–{2*enroll_hi} individuals)")

        print("\nNotes:")
        print("  - Target success: 2–3% DunedinPACE slowing at Week 24 (α=0.05).")
        print("  - Twin within‑pair randomization maximizes power; ICCs reflect MZ/DZ mix.")
        print("  - For secondary mechanistic endpoints (IL‑6/CRP, HbA1c/HOMA‑IR/lipids), use co‑primary mode")
        print("    or separate runs to estimate marginal and joint power; hierarchical testing maps to joint power.")
        print("=" * 70)

    # MODE: curve
    elif args.mode == "curve":  # curve
        n_vals = np.arange(int(args.curve_start), int(args.curve_stop) + 1, int(args.curve_step))
        rows = []
        for n in n_vals:
            if args.use_simulation:
                spec_n = AgeTwinSpec(
                    n_pairs=int(n), effect_abs=spec.effect_abs, sd_change=spec.sd_change,
                    prop_mz=spec.prop_mz, icc_mz=spec.icc_mz, icc_dz=spec.icc_dz,
                    alpha=spec.alpha, seed=spec.seed,
                    contamination_rate=args.contamination_rate,
                    contamination_effect=args.contamination_effect
                )
                pw, _, _ = _simulate_pairs(spec_n, sims=args.sims)
            else:
                pw = analytic_power_paired(int(n), effect_obs, spec.sd_change, icc_eff, spec.alpha)
            rows.append((int(n), pw))
        
        print("n_pairs,power")
        for n, pw in rows:
            # Align precision with power-mode default (3 decimals)
            print(f"{n},{pw:.3f}")

    # MODE: two-sample-power (individual-level, parallel-groups)
    elif args.mode == "two-sample-power":
        # Resolve inputs
        sd_change_ts = _apply_ancova_sd(sd_change, getattr(args, "ancova_r2", None))
        # For two-sample, use endpoint helpers but DO NOT use d_std mapping to paired sd
        effect_abs_ts = _resolve_effect_abs(args, sd_change_ts, icc_eff, "")
        effect_obs_ts = apply_contamination(effect_abs_ts, args.contamination_rate, args.contamination_effect)

        n_per_group = None
        if args.n_per_group is not None:
            n_per_group = int(args.n_per_group)
        elif args.n_total is not None:
            n_per_group = max(1, int(args.n_total) // 2)
        else:
            print("Error: --n-per-group or --n-total required for two-sample-power mode", file=sys.stderr)
            sys.exit(1)

        pw = analytic_power_two_sample(n_per_group, effect_obs_ts, sd_change_ts, args.alpha)
        n_total = 2 * n_per_group

        # Optional design effect
        if args.deff_icc is not None:
            try:
                deff = design_effect(float(args.deff_icc), int(args.deff_m))
            except Exception as e:
                print(f"Design-effect error: {e}", file=sys.stderr)
                deff = None
        else:
            deff = None

        print("=" * 70)
        print("TWO-SAMPLE POWER (individual-level, secondary/confirmatory)")
        print("=" * 70)
        print(f"\nEndpoint: {args.endpoint}")
        print(f"Groups: n_per_group={n_per_group} (total N={n_total})")
        if getattr(args, "ancova_r2", None) is not None:
            print(f"ANCOVA R^2: {float(args.ancova_r2):.2f} (sd_change scaled → {sd_change_ts:.4f})")
        print(f"Effect (beneficial | observed): {effect_obs_ts:.4f} (from specified {effect_abs_ts:.4f})")
        print(f"SD(change): {sd_change_ts:.4f}")
        print(f"Alpha (two-sided): {args.alpha:.3f}")
        print(f"Power: {pw:.3f} ({pw*100:.1f}%)")
        if deff is not None:
            n_infl = int(math.ceil(n_total * float(deff)))
            print(f"\nDesign effect (optional): DEFF={deff:.3f} → Inflated total N={n_infl}")
        if args.attrition_rate > 0:
            n_enroll = inflate_for_attrition(n_total, args.attrition_rate)
            print(f"\nEnrollment with attrition {args.attrition_rate:.1%}: {n_enroll} individuals")
        print("=" * 70)

    # MODE: two-sample-n-for-power
    else:  # two-sample-n-for-power
        if args.mode != "two-sample-n-for-power":
            # Should not happen, guard for future additions
            print(f"Unknown mode: {args.mode}", file=sys.stderr)
            sys.exit(2)

        sd_change_ts = _apply_ancova_sd(sd_change, getattr(args, "ancova_r2", None))
        effect_abs_ts = _resolve_effect_abs(args, sd_change_ts, icc_eff, "")
        effect_obs_ts = apply_contamination(effect_abs_ts, args.contamination_rate, args.contamination_effect)

        n_pg = analytic_n_per_group_for_power(args.target_power, effect_obs_ts, sd_change_ts, args.alpha)
        n_total = 2 * n_pg

        # Optional design effect
        deff = None
        n_total_deff = None
        if args.deff_icc is not None:
            try:
                deff = design_effect(float(args.deff_icc), int(args.deff_m))
                n_total_deff = int(math.ceil(n_total * float(deff)))
            except Exception as e:
                print(f"Design-effect error: {e}", file=sys.stderr)

        print("=" * 70)
        print("TWO-SAMPLE SAMPLE SIZE (individual-level, secondary/confirmatory)")
        print("=" * 70)
        print(f"\nEndpoint: {args.endpoint}")
        print(f"Target power: {args.target_power:.3f} ({args.target_power*100:.1f}%) at α={args.alpha:.3f}")
        if getattr(args, "ancova_r2", None) is not None:
            print(f"ANCOVA R^2: {float(args.ancova_r2):.2f} (sd_change scaled → {sd_change_ts:.4f})")
        print(f"Effect (beneficial | observed): {effect_obs_ts:.4f} (from specified {effect_abs_ts:.4f})")
        print(f"SD(change): {sd_change_ts:.4f}")
        print(f"\nRESULTS:")
        print(f"  Required n per group: {n_pg}")
        print(f"  Total individuals: {n_total}")
        if deff is not None:
            print(f"  With DEFF={deff:.3f}: Inflated total N={n_total_deff}")
        if args.attrition_rate > 0:
            n_enroll = inflate_for_attrition(n_total, args.attrition_rate)
            print(f"  With attrition {args.attrition_rate:.1%}: Enroll {n_enroll} individuals")
        print("=" * 70)


if __name__ == "__main__":
    run_cli()
