"""
Power analysis for a twin-focused individually randomized RCT (ISI change).

This script estimates statistical power (or required sample size) for the
primary endpoint (change in Insomnia Severity Index, ISI) comparing an LLM
digital CBT-I arm vs an attention-matched active control, explicitly modeling
MZ and DZ twin correlations under individual-level randomization.

Approach
--------
- Outcome model: y = delta * I[treatment] + b_pair + e
  where b_pair ~ N(0, sigma_pair^2) with ICC = sigma_pair^2 / sigma_y^2
  and e ~ N(0, sigma_resid^2) with sigma_resid^2 = (1 - ICC) * sigma_y^2.
- ICC differs by zygosity: ICC_MZ, ICC_DZ.
- Individuals are randomized 1:1, including within twin pairs (independent
  randomization per twin). This captures cases where co-twins land in the
  same or opposite arms and therefore captures across-arm correlation.
- Mix of participants: a fraction are twins (in pairs of size 2), the rest
  are singletons (cluster size 1). Among twin pairs, a fraction are MZ and
  the rest DZ.
- Analysis method: by default OLS with cluster-robust standard errors clustered
  on pair_id, including zygosity fixed effects. This is valid for individual
  randomization with within-cluster correlation and can handle heterogeneous
  ICC across clusters. Optionally, a random-intercept MixedLM can be used.

Effect Size Convention
----------------------
The outcome y is the CHANGE in ISI from baseline to 8 weeks (change = follow-up - baseline).
For ISI, LOWER scores indicate LESS insomnia (better sleep). Therefore:
- Improvement = NEGATIVE change (ISI decreased)
- Worsening = POSITIVE change (ISI increased)

The parameter effect_points represents how much MORE NEGATIVE the change is in the
treatment group compared to control. For example:
- effect_points = 6 means treatment group's change is 6 points MORE NEGATIVE than control
- If control changes by -3 points (3-point improvement), treatment changes by -9 points
  (9-point improvement), a 6-point additional benefit.
- This is coded as: y = -effect_points * I[treatment] + b_pair + e
  (note the NEGATIVE sign so positive effect_points means treatment benefit)

Usage
-----
1) Estimate power for a given total N (individuals):
   python3 power_twin_sleep.py --mode power --n-total 200 \
     --effect-points 6 --sd-change 7 --prop-twins 0.9 --prop-mz 0.5 \
     --icc-mz 0.5 --icc-dz 0.25 --alpha 0.05 --sims 2000

2) Find minimum N for target power (e.g., 90%):
   python3 power_twin_sleep.py --mode n-for-power --target-power 0.9 \
     --effect-points 6 --sd-change 7 --prop-twins 0.9 --prop-mz 0.5 \
     --icc-mz 0.5 --icc-dz 0.25 --alpha 0.05 --sims 1500

Notes
-----
- The effect size is in ISI points (e.g., 6 for the MCID). The SD should be
  the SD of the change score at 8 weeks relative to baseline (typical values
  in insomnia trials are around 5–7; adjust to your context).
- ICC_MZ and ICC_DZ are correlations for the change outcome between co-twins
  within a pair. Reasonable starting assumptions: ICC_MZ ≈ 0.4–0.6, ICC_DZ ≈ 0.2–0.3.
- prop_twins is the fraction of all participants that are twins (≈0.9 per design).
- prop_mz is the fraction of twin pairs that are MZ; rest are DZ (≈0.5/0.5).
- The simulation seeds the RNG for reproducibility but each run will still have
  Monte Carlo error; increase --sims to tighten precision.

"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import math

import statsmodels.api as sm
import statsmodels.formula.api as smf
import scipy.stats as sps  # for t distribution p-values


@dataclass
class TwinTrialSpec:
    n_total: int
    effect_points: float = 6.0  # additional ISI reduction (more negative change) in treatment vs control
    sd_change: float = 7.0      # SD of change at 8 weeks
    prop_twins: float = 0.9     # fraction of participants who are twins
    prop_mz: float = 0.5        # fraction of twin pairs that are MZ
    icc_mz: float = 0.5         # within-pair ICC for MZ pairs on change
    icc_dz: float = 0.25        # within-pair ICC for DZ pairs on change
    alpha: float = 0.05
    analysis: str = "cluster_robust"  # or "mixedlm"
    seed: Optional[int] = 12345

    # Contamination and attrition (new)
    contamination_rate: float = 0.0      # fraction of controls adopting intervention (0-1)
    contamination_effect: float = 0.0    # fraction of full effect controls receive if contaminated (0-1)
    attrition_rate: float = 0.0          # expected dropout rate (0-1); used for enrollment inflation only

    def observed_effect(self) -> float:
        """Observed effect after contamination attenuation.

        effect_obs = effect_points * (1 - contamination_rate * contamination_effect)
        """
        reduction = 1.0 - (self.contamination_rate * self.contamination_effect)
        return float(max(0.0, self.effect_points * reduction))

    def validate(self) -> None:
        validate_probability(self.prop_twins, "prop_twins", allow_one=True)
        validate_probability(self.prop_mz, "prop_mz", allow_one=True)
        validate_probability(self.icc_mz, "icc_mz", allow_one=True)
        validate_probability(self.icc_dz, "icc_dz", allow_one=True)
        validate_probability(self.alpha, "alpha", allow_zero=False, allow_one=False)
        validate_probability(self.contamination_rate, "contamination_rate", allow_one=True)
        validate_probability(self.contamination_effect, "contamination_effect", allow_one=True)
        validate_probability(self.attrition_rate, "attrition_rate", allow_one=False)
        if not (self.n_total > 0):
            raise ValueError("n_total must be > 0")
        if not (self.sd_change > 0):
            raise ValueError("sd_change must be > 0")
        if self.effect_points < 0:
            raise ValueError("effect_points should be non-negative (beneficial magnitude)")


def validate_probability(value: float, name: str, allow_zero: bool = True, allow_one: bool = True) -> None:
    """Validate a probability-like value in [0,1] with optional strictness."""
    if value is None or not (value == value):  # NaN check
        raise ValueError(f"{name} must be a real number in [0,1]")
    if (not allow_zero and value <= 0.0) or (allow_zero and value < 0.0):
        raise ValueError(f"{name} must be >= 0{'' if allow_zero else ' (strict)'}")
    if (not allow_one and value >= 1.0) or (allow_one and value > 1.0):
        raise ValueError(f"{name} must be <= 1{'' if allow_one else ' (strict)'}")


def _build_sample(n_total: int, prop_twins: float, prop_mz: float, rng: np.random.Generator) -> pd.DataFrame:
    """Create a sample with pair structure and zygosity labels.

    Returns a dataframe with columns: id, pair_id, zygosity (MZ/DZ/SINGLE), is_singleton.
    """
    # Number of twin participants, ensure even number to form pairs
    n_twins = int(round(n_total * prop_twins))
    if n_twins % 2 == 1:
        # Adjust to even by reducing one (the remainder becomes singleton)
        n_twins -= 1
    n_pairs = n_twins // 2
    n_singleton = n_total - (2 * n_pairs)

    n_mz_pairs = int(round(n_pairs * prop_mz))
    # Ensure counts sum correctly
    n_mz_pairs = min(n_mz_pairs, n_pairs)
    n_dz_pairs = n_pairs - n_mz_pairs

    rows = []
    pid = 0
    uid = 0
    # Add MZ pairs
    for _ in range(n_mz_pairs):
        pid_str = f"P{pid:05d}"
        for j in range(2):
            rows.append({
                "id": f"U{uid:06d}",
                "pair_id": pid_str,
                "zygosity": "MZ",
                "is_singleton": False,
            })
            uid += 1
        pid += 1
    # Add DZ pairs
    for _ in range(n_dz_pairs):
        pid_str = f"P{pid:05d}"
        for j in range(2):
            rows.append({
                "id": f"U{uid:06d}",
                "pair_id": pid_str,
                "zygosity": "DZ",
                "is_singleton": False,
            })
            uid += 1
        pid += 1
    # Add singletons (unique pair_id per singleton)
    for _ in range(n_singleton):
        pid_str = f"S{pid:05d}"
        rows.append({
            "id": f"U{uid:06d}",
            "pair_id": pid_str,
            "zygosity": "SINGLE",
            "is_singleton": True,
        })
        uid += 1
        pid += 1

    df = pd.DataFrame(rows)
    # Randomize treatment 1:1 at the individual level
    df["treat"] = rng.integers(0, 2, size=len(df))
    return df


def sd_change_from_pre_post(sd_pre: float, sd_post: float, rho_pre_post: float) -> float:
    """Compute SD of change = post - pre given SDs and their correlation.

    Var(change) = Var(post) + Var(pre) - 2*rho*sd_post*sd_pre.
    """
    return float(np.sqrt(max(1e-12, sd_post**2 + sd_pre**2 - 2.0 * rho_pre_post * sd_post * sd_pre)))


def _simulate_outcomes(df: pd.DataFrame, effect_points: float, sd_change: float,
                       icc_mz: float, icc_dz: float, rng: np.random.Generator) -> pd.DataFrame:
    """Simulate change outcomes per the specified ICCs by zygosity.

    - For twin pairs (MZ/DZ): y = -delta * I[treat] + b_pair + e
      where var(b_pair) = ICC * sd^2, var(e) = (1-ICC) * sd^2.
      The NEGATIVE sign ensures positive effect_points = treatment benefit (more negative change).
    - For singletons: y = -delta * I[treat] + e, with var(e) = sd^2.
    """
    sigma2 = sd_change ** 2
    
    # Reset index to ensure sequential 0-based indexing
    df = df.reset_index(drop=True)
    y = np.zeros(len(df))

    # Map each pair_id to a random intercept drawn per zygosity
    pair_groups = df.groupby("pair_id")
    b_dict = {}
    for pid, g in pair_groups:
        z = g["zygosity"].iloc[0]
        if z == "MZ":
            icc = icc_mz
            var_pair = icc * sigma2
            b = rng.normal(0.0, np.sqrt(var_pair))
        elif z == "DZ":
            icc = icc_dz
            var_pair = icc * sigma2
            b = rng.normal(0.0, np.sqrt(var_pair))
        else:  # SINGLE
            icc = 0.0
            b = 0.0
        b_dict[pid] = (b, icc)

    # Generate residuals and outcomes - FIXED: use enumerate for safe indexing
    for i in range(len(df)):
        pid = df.loc[i, "pair_id"]
        z = df.loc[i, "zygosity"]
        treat = df.loc[i, "treat"]
        b, icc = b_dict[pid]
        
        if z == "SINGLE":
            var_e = sigma2
        else:
            var_e = (1.0 - icc) * sigma2
        e = rng.normal(0.0, np.sqrt(var_e))
        
        # NEGATIVE sign: positive effect_points means treatment reduces ISI more (beneficial)
        y[i] = -effect_points * treat + b + e

    df = df.copy()
    df["y"] = y
    return df


def _pvalue_for_treat(df: pd.DataFrame, alpha: float, analysis: str = "cluster_robust",
                      sd_change: Optional[float] = None, icc_mz: Optional[float] = None,
                      icc_dz: Optional[float] = None) -> Tuple[float, float]:
    """Fit model and return (pvalue, coef) for treatment indicator.

    analysis in {"cluster_robust", "mixedlm"}.
    
    Note: coef will be negative if treatment is beneficial (reduces ISI more).
    """
    # Ensure categorical coding for zygosity
    df = df.copy()
    df["zygosity"] = pd.Categorical(df["zygosity"], categories=["SINGLE", "DZ", "MZ"], ordered=False)

    if analysis == "mixedlm":
        # Random intercept per pair
        try:
            model = smf.mixedlm("y ~ treat + C(zygosity)", df, groups=df["pair_id"]).fit(reml=False, disp=False)
            pval = model.pvalues.get("treat", np.nan)
            coef = model.params.get("treat", np.nan)
            return pval, coef
        except Exception as e:
            # Fall back to cluster-robust if MixedLM fails to converge
            analysis = "cluster_robust"

    if analysis == "cluster_robust":
        # OLS + cluster-robust SE clustered by pair
        model = smf.ols("y ~ treat + C(zygosity)", df).fit(cov_type="cluster", cov_kwds={"groups": df["pair_id"], "use_correction": True})
        pval = model.pvalues.get("treat", np.nan)
        coef = model.params.get("treat", np.nan)
        return pval, coef

    # Fallback: known-covariance GLS using provided sd_change and ICCs
    if sd_change is None or icc_mz is None or icc_dz is None:
        raise RuntimeError("GLS fallback requires sd_change, icc_mz, icc_dz.")

    return _pvalue_for_treat_gls_known(df, sd_change, icc_mz, icc_dz)


def _normal_two_sided_pvalue(t: float) -> float:
    # Two-sided p-value from standard normal
    from math import erf, sqrt
    z = abs(t)
    # Phi(z) = 0.5*(1 + erf(z/sqrt(2)))
    phi = 0.5 * (1.0 + erf(z / sqrt(2.0)))
    return 2.0 * (1.0 - phi)


def _t_two_sided_pvalue(t: float, df: int) -> float:
    try:
        return 2.0 * sps.t.sf(abs(t), df)
    except Exception:
        # Normal approximation on SciPy error
        return _normal_two_sided_pvalue(t)


def _pvalue_for_treat_gls_known(df: pd.DataFrame, sd_change: float, icc_mz: float, icc_dz: float) -> Tuple[float, float]:
    """Compute p-value for treatment effect via GLS with known covariance.

    - Builds block-diagonal C^{-1} (correlation inverse) per cluster:
      singleton: [1]; pair: (1/(1-rho^2))*[[1,-rho],[-rho,1]]
    - Uses sigma2_hat from GLS residuals to scale var(beta).
    - Two-sided t-test with df=N-p (falls back to normal if SciPy missing).
    """
    # Design matrix X: intercept, treat, I[zygosity==DZ], I[zygosity==MZ]
    z = df["zygosity"].astype(str)
    X = np.column_stack([
        np.ones(len(df), dtype=float),
        df["treat"].astype(float).to_numpy(),
        (z == "DZ").astype(float).to_numpy(),
        (z == "MZ").astype(float).to_numpy(),
    ])
    y = df["y"].to_numpy(dtype=float)

    # Accumulate M = X' C^{-1} X and b = X' C^{-1} y and Q = (y - Xb)' C^{-1} (y - Xb)
    p = X.shape[1]
    M = np.zeros((p, p), dtype=float)
    b = np.zeros((p,), dtype=float)

    # Build per-block contributions
    for pid, g in df.groupby("pair_id"):
        idx = g.index.to_numpy()
        Xg = X[idx, :]
        yg = y[idx]
        zygo = str(g["zygosity"].iloc[0])
        if zygo == "SINGLE":
            Cinv = np.array([[1.0]], dtype=float)
        else:
            rho = icc_mz if zygo == "MZ" else icc_dz
            denom = max(1e-9, 1.0 - rho * rho)
            Cinv = (1.0 / denom) * np.array([[1.0, -rho], [-rho, 1.0]], dtype=float)
        # Accumulate
        Xt_Cinv = Xg.T @ Cinv
        M += Xt_Cinv @ Xg
        b += Xt_Cinv @ yg

    # Solve for beta
    Minv = np.linalg.inv(M)
    beta = Minv @ b
    resid = y - X @ beta

    # Compute sigma2_hat via blockwise residual quadratic form Q = sum r' C^{-1} r
    Q = 0.0
    for pid, g in df.groupby("pair_id"):
        idx = g.index.to_numpy()
        rg = resid[idx]
        zygo = str(g["zygosity"].iloc[0])
        if zygo == "SINGLE":
            Cinv = np.array([[1.0]], dtype=float)
        else:
            rho = icc_mz if zygo == "MZ" else icc_dz
            denom = max(1e-9, 1.0 - rho * rho)
            Cinv = (1.0 / denom) * np.array([[1.0, -rho], [-rho, 1.0]], dtype=float)
        Q += float(rg.T @ Cinv @ rg)

    n = len(df)
    df_dof = max(1, n - p)
    # Unbiased estimate of sigma^2 under GLS with known C: sigma2_hat = Q / (n - p)
    sigma2_hat = Q / float(df_dof)

    # Var(beta) = sigma2_hat * Minv
    Vbeta = sigma2_hat * Minv
    # Treatment is column 1
    se_treat = float(np.sqrt(max(1e-12, Vbeta[1, 1])))
    coef = float(beta[1])
    tval = coef / se_treat if se_treat > 0 else np.inf
    pval = _t_two_sided_pvalue(tval, df_dof)
    return pval, coef


def simulate_power(spec: TwinTrialSpec, sims: int = 2000) -> Tuple[float, float]:
    """Monte Carlo power estimate for a given N and spec.

    Returns (power, avg_estimated_effect).
    Note: avg_estimated_effect will be negative if treatment is beneficial.
    """
    spec.validate()
    rng = np.random.default_rng(spec.seed)
    hits = 0
    ests = []
    effect_obs = spec.observed_effect()
    for s in range(sims):
        df = _build_sample(spec.n_total, spec.prop_twins, spec.prop_mz, rng)
        df = _simulate_outcomes(df, effect_obs, spec.sd_change, spec.icc_mz, spec.icc_dz, rng)
        pval, coef = _pvalue_for_treat(
            df, spec.alpha, analysis=spec.analysis,
            sd_change=spec.sd_change, icc_mz=spec.icc_mz, icc_dz=spec.icc_dz
        )
        if np.isnan(pval):
            continue  # treat as failure; conservative
        hits += (pval < spec.alpha)
        ests.append(coef)
    power = hits / sims
    avg_effect = float(np.mean(ests)) if ests else float("nan")
    return power, avg_effect


def simulate_distribution(spec: TwinTrialSpec, sims: int = 1000):
    """Run simulations and return detailed arrays for plotting.

    Returns (power, coefs, pvals), where coefs and pvals are numpy arrays.
    Power uses denominator=n_sims and counts NaN p-values as non-significant.
    """
    spec.validate()
    rng = np.random.default_rng(spec.seed)
    pvals = np.full(sims, np.nan, dtype=float)
    coefs = np.full(sims, np.nan, dtype=float)
    hits = 0
    effect_obs = spec.observed_effect()
    for s in range(sims):
        df = _build_sample(spec.n_total, spec.prop_twins, spec.prop_mz, rng)
        df = _simulate_outcomes(df, effect_obs, spec.sd_change, spec.icc_mz, spec.icc_dz, rng)
        pval, coef = _pvalue_for_treat(
            df, spec.alpha, analysis=spec.analysis, sd_change=spec.sd_change, icc_mz=spec.icc_mz, icc_dz=spec.icc_dz
        )
        pvals[s] = pval
        coefs[s] = coef
        if not np.isnan(pval) and pval < spec.alpha:
            hits += 1
    power = hits / sims
    return power, coefs, pvals


def _z_two_sided(alpha: float) -> float:
    """Two-sided normal z for given alpha (SciPy)."""
    return float(sps.norm.ppf(1.0 - alpha / 2.0))


def binomial_wilson_ci(k: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """Wilson score interval for binomial proportion k/n (two-sided alpha)."""
    if n <= 0:
        return (float("nan"), float("nan"))
    z = _z_two_sided(alpha)
    phat = k / n
    denom = 1.0 + (z * z) / n
    center = (phat + (z * z) / (2.0 * n)) / denom
    half = (z / denom) * math.sqrt(max(0.0, phat * (1.0 - phat) / n + (z * z) / (4.0 * n * n)))
    low = max(0.0, center - half)
    high = min(1.0, center + half)
    return low, high


def power_curve(base_spec: TwinTrialSpec, n_values: np.ndarray, sims: int = 1000, alpha_ci: float = 0.05):
    """Compute power across a grid of N with Wilson intervals.

    Returns a DataFrame with columns: N, power, ci_low, ci_high.
    """
    rows = []
    for n in n_values:
        spec = TwinTrialSpec(
            n_total=int(n),
            effect_points=base_spec.effect_points,
            sd_change=base_spec.sd_change,
            prop_twins=base_spec.prop_twins,
            prop_mz=base_spec.prop_mz,
            icc_mz=base_spec.icc_mz,
            icc_dz=base_spec.icc_dz,
            alpha=base_spec.alpha,
            analysis=base_spec.analysis,
            seed=base_spec.seed,
        )
        pw, coefs, pvals = simulate_distribution(spec, sims=sims)
        k = int(np.sum((pvals < spec.alpha) & ~np.isnan(pvals)))
        low, high = binomial_wilson_ci(k, sims, alpha=alpha_ci)
        rows.append({"N": int(n), "power": pw, "ci_low": low, "ci_high": high})
    return pd.DataFrame(rows)


def find_n_for_power(target_power: float, base_spec: TwinTrialSpec, sims: int = 1500,
                     n_min: int = 50, n_max: int = 2000, tol: float = 0.01) -> Tuple[int, float]:
    """Binary search for the minimum N achieving target power.

    Returns (n_required, achieved_power_at_n).
    
    Note: Steps by 1 to find exact minimum N.
    """
    assert 0.5 < target_power < 0.999, "target_power should be in (0.5, 0.999)."
    base_spec.validate()

    low = n_min
    high = n_max
    best_n = high
    best_pw = 0.0

    while low <= high:
        mid = (low + high) // 2
        spec = TwinTrialSpec(
            n_total=mid,
            effect_points=base_spec.effect_points,
            sd_change=base_spec.sd_change,
            prop_twins=base_spec.prop_twins,
            prop_mz=base_spec.prop_mz,
            icc_mz=base_spec.icc_mz,
            icc_dz=base_spec.icc_dz,
            alpha=base_spec.alpha,
            analysis=base_spec.analysis,
            seed=base_spec.seed,
            contamination_rate=base_spec.contamination_rate,
            contamination_effect=base_spec.contamination_effect,
            attrition_rate=base_spec.attrition_rate,
        )
        pw, _ = simulate_power(spec, sims=sims)
        # print(f"N={mid}, power={pw:.3f}")  # optional debug
        if pw >= target_power - tol:
            best_n, best_pw = mid, pw
            high = mid - 1  # continue searching for lower N
        else:
            low = mid + 1

    return best_n, best_pw


def approximate_effective_n(n_total: int, prop_twins: float, prop_mz: float, icc_mz: float, icc_dz: float) -> float:
    """Kish effective sample size approximation for mixed singleton/twin sample.

    Treats each singleton as cluster size 1 contributing 1 effective unit, and
    each twin pair of size 2 contributing 2/(1+ICC) effective units (ICC by zygosity).
    Returns the approximate total effective N (ignores across-arm correlation).
    """
    n_twins = int(round(n_total * prop_twins))
    if n_twins % 2 == 1:
        n_twins -= 1
    n_pairs = n_twins // 2
    n_singleton = n_total - (2 * n_pairs)

    n_mz_pairs = int(round(n_pairs * prop_mz))
    n_dz_pairs = n_pairs - n_mz_pairs

    eff_singletons = float(n_singleton)
    eff_mz = n_mz_pairs * (2.0 / (1.0 + max(1e-9, icc_mz)))
    eff_dz = n_dz_pairs * (2.0 / (1.0 + max(1e-9, icc_dz)))
    return eff_singletons + eff_mz + eff_dz


def inflate_for_attrition(n_completing: int, attrition_rate: float) -> int:
    """Inflate completing N to required enrollment given attrition_rate."""
    validate_probability(attrition_rate, "attrition_rate", allow_one=False)
    return int(math.ceil(n_completing / max(1e-9, (1.0 - attrition_rate))))


def main():
    parser = argparse.ArgumentParser(description="Power analysis for twin-focused LLM CBT-I RCT (ISI change)")
    parser.add_argument("--mode", choices=["power", "n-for-power"], default="power")
    parser.add_argument("--n-total", type=int, default=200, help="Total N (individuals) when mode=power")
    parser.add_argument("--target-power", type=float, default=0.9, help="Target power for mode=n-for-power")
    parser.add_argument("--effect-points", type=float, default=6.0, help="Treatment effect: additional ISI reduction in treatment vs control (positive = beneficial)")
    parser.add_argument("--sd-change", type=float, default=7.0, help="SD of ISI change at 8 weeks")
    parser.add_argument("--prop-twins", type=float, default=0.9, help="Proportion of participants who are twins")
    parser.add_argument("--prop-mz", type=float, default=0.5, help="Proportion of twin pairs who are MZ")
    parser.add_argument("--icc-mz", type=float, default=0.5, help="Within-pair ICC for MZ on change")
    parser.add_argument("--icc-dz", type=float, default=0.25, help="Within-pair ICC for DZ on change")
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level")
    parser.add_argument("--sims", type=int, default=2000, help="Monte Carlo iterations per evaluation")
    parser.add_argument("--analysis", choices=["cluster_robust", "mixedlm"], default="cluster_robust", help="Analysis method")
    parser.add_argument("--seed", type=int, default=12345, help="Random seed")
    # Contamination and attrition
    parser.add_argument("--contamination-rate", type=float, default=0.0, help="Proportion of controls adopting intervention (0-1)")
    parser.add_argument("--contamination-effect", type=float, default=0.0, help="Fraction of full effect controls receive if contaminated (0-1)")
    parser.add_argument("--attrition-rate", type=float, default=0.0, help="Expected dropout rate (0-1). Used to inflate enrollment.")

    args = parser.parse_args()

    base_spec = TwinTrialSpec(
        n_total=args.n_total,
        effect_points=args.effect_points,
        sd_change=args.sd_change,
        prop_twins=args.prop_twins,
        prop_mz=args.prop_mz,
        icc_mz=args.icc_mz,
        icc_dz=args.icc_dz,
        alpha=args.alpha,
        analysis=args.analysis,
        seed=args.seed,
        contamination_rate=args.contamination_rate,
        contamination_effect=args.contamination_effect,
        attrition_rate=args.attrition_rate,
    )

    if args.mode == "power":
        pw, avg = simulate_power(base_spec, sims=args.sims)
        n_eff = approximate_effective_n(args.n_total, args.prop_twins, args.prop_mz, args.icc_mz, args.icc_dz)
        # 95% CI for Monte Carlo power
        se_pw = math.sqrt(pw * (1 - pw) / args.sims)
        ci_low = max(0.0, pw - 1.96 * se_pw)
        ci_high = min(1.0, pw + 1.96 * se_pw)
        n_enroll = inflate_for_attrition(args.n_total, args.attrition_rate) if args.attrition_rate > 0 else args.n_total
        effect_obs = base_spec.observed_effect()
        print("Power analysis (simulation)")
        print(f"  N total: {args.n_total}")
        if args.attrition_rate > 0:
            print(f"  Expected attrition: {args.attrition_rate:.1%}")
            print(f"  Required enrollment: {n_enroll}")
        print(f"  Approx effective N (Kish): {n_eff:.1f}")
        print(f"  Effect (additional ISI reduction): {args.effect_points} points")
        if args.contamination_rate > 0:
            print(f"  Observed effect after contamination: {effect_obs:.2f} (rate={args.contamination_rate:.1%}, effect={args.contamination_effect:.1%})")
        print(f"  SD(change): {args.sd_change}")
        print(f"  prop_twins={args.prop_twins}, prop_mz={args.prop_mz}")
        print(f"  ICCs: MZ={args.icc_mz}, DZ={args.icc_dz}")
        print(f"  alpha={args.alpha}, sims={args.sims}, analysis={args.analysis}")
        print(f"  Estimated power: {pw:.3f}")
        print(f"  95% CI: [{ci_low:.3f}, {ci_high:.3f}]")
        print(f"  Avg estimated treatment effect: {avg:.2f} (negative = beneficial)")
    else:
        n_req, pw = find_n_for_power(args.target_power, base_spec, sims=args.sims)
        n_eff = approximate_effective_n(n_req, args.prop_twins, args.prop_mz, args.icc_mz, args.icc_dz)
        n_enroll = inflate_for_attrition(n_req, args.attrition_rate) if args.attrition_rate > 0 else n_req
        print("Sample size for target power (simulation)")
        print(f"  Target power: {args.target_power}")
        print(f"  Required N (approx): {n_req}")
        if args.attrition_rate > 0:
            print(f"  With {args.attrition_rate:.1%} attrition, required enrollment: {n_enroll}")
        print(f"  Approx effective N (Kish) at N_req: {n_eff:.1f}")
        print(f"  Achieved power at N_req: {pw:.3f}")
        print(f"  Effect (additional ISI reduction): {args.effect_points} points")
        if args.contamination_rate > 0:
            print(f"  Observed effect after contamination: {base_spec.observed_effect():.2f} (rate={args.contamination_rate:.1%}, effect={args.contamination_effect:.1%})")
        print(f"  SD(change): {args.sd_change}")
        print(f"  prop_twins={args.prop_twins}, prop_mz={args.prop_mz}, alpha={args.alpha}")
        print(f"  ICCs: MZ={args.icc_mz}, DZ={args.icc_dz}, sims={args.sims}, analysis={args.analysis}")


if __name__ == "__main__":
    main()
