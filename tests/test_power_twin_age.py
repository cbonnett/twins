import math
import types
import argparse
import subprocess

import numpy as np
import pytest

import biological_age.power_twin_age as pta


# ---------- Utility validations ----------

def test_validate_probability_bounds():
    pta.validate_probability(0.0, "p")
    pta.validate_probability(1.0, "p")
    pta.validate_probability(0.5, "p")
    with pytest.raises(ValueError):
        pta.validate_probability(-1e-9, "p")
    with pytest.raises(ValueError):
        pta.validate_probability(1.0000001, "p")


def test_validate_positive_and_zero():
    pta.validate_positive(1e-12, "x")
    with pytest.raises(ValueError):
        pta.validate_positive(0.0, "x")
    pta.validate_positive(0.0, "x", allow_zero=True)


def test_validate_icc():
    pta.validate_icc(0.0, "icc")
    pta.validate_icc(0.999999, "icc")
    with pytest.raises(ValueError):
        pta.validate_icc(-1e-12, "icc")
    with pytest.raises(ValueError):
        pta.validate_icc(1.0, "icc")


def test_sd_change_from_pre_post_basic_symmetry():
    # symmetric case, rho=0
    out = pta.sd_change_from_pre_post(2.0, 2.0, 0.0)
    assert math.isclose(out, math.sqrt(8.0))
    # positive correlation reduces var
    out_pos = pta.sd_change_from_pre_post(2.0, 2.0, 0.8)
    assert out_pos < out
    # negative correlation increases var
    out_neg = pta.sd_change_from_pre_post(2.0, 2.0, -0.8)
    assert out_neg > out


def test_sd_diff_from_sd_change_icc_monotonic():
    sd = 1.23
    s0 = pta.sd_diff_from_sd_change_icc(sd, 0.0)
    s1 = pta.sd_diff_from_sd_change_icc(sd, 0.5)
    s2 = pta.sd_diff_from_sd_change_icc(sd, 0.9)
    assert s0 > s1 > s2  # higher ICC -> smaller pair-diff SD


def test_apply_contamination_edges():
    # No contamination
    assert math.isclose(pta.apply_contamination(1.0, 0.0, 0.5), 1.0)
    # Full contamination at full effect -> zero observed
    assert math.isclose(pta.apply_contamination(2.0, 1.0, 1.0), 0.0)
    # Partial contamination
    assert math.isclose(pta.apply_contamination(1.0, 0.3, 0.5), 1.0 * (1 - 0.15))


def test_inflate_for_attrition():
    assert pta.inflate_for_attrition(100, 0.0) == 100
    assert pta.inflate_for_attrition(100, 0.25) == 134  # 100/(1-0.25)=133.33 -> ceil
    with pytest.raises(ValueError):
        pta.inflate_for_attrition(100, 1.0)


def test_z_t_alpha_two_sided_values():
    # sanity only; rely on SciPy if present or constants when not
    z_005 = pta._z_alpha_two_sided(0.05)
    assert 1.95 < z_005 < 1.97
    z_001 = pta._z_alpha_two_sided(0.01)
    assert 2.57 < z_001 < 2.59


def test_two_sided_power_normal_symmetry():
    # power should depend on |lambda|
    lam = 2.0
    pw_pos = pta._two_sided_power_normal(lam, 0.05)
    pw_neg = pta._two_sided_power_normal(-lam, 0.05)
    assert math.isclose(pw_pos, pw_neg, rel_tol=1e-12)


def test_d_paired_consistency():
    sd_change = 0.1
    icc = 0.5
    sd_d = pta.sd_diff_from_sd_change_icc(sd_change, icc)
    d = pta.d_paired(0.02, sd_change, icc)
    assert math.isclose(d, 0.02 / sd_d)


# ---------- Analytic functions ----------

def test_analytic_power_monotonic_effect_and_n():
    sd_change = 0.10
    icc = 0.55
    alpha = 0.05
    n = 300
    # power increases with effect
    pw_small = pta.analytic_power_paired(n, 0.01, sd_change, icc, alpha)
    pw_med = pta.analytic_power_paired(n, 0.03, sd_change, icc, alpha)
    pw_big = pta.analytic_power_paired(n, 0.05, sd_change, icc, alpha)
    assert pw_small <= pw_med <= pw_big
    # power increases with n
    pw_n1 = pta.analytic_power_paired(100, 0.03, sd_change, icc, alpha)
    pw_n2 = pta.analytic_power_paired(400, 0.03, sd_change, icc, alpha)
    assert pw_n2 > pw_n1


def test_analytic_power_increases_with_icc():
    sd_change = 0.10
    alpha = 0.05
    n = 300
    pw_low_icc = pta.analytic_power_paired(n, 0.03, sd_change, 0.1, alpha)
    pw_high_icc = pta.analytic_power_paired(n, 0.03, sd_change, 0.8, alpha)
    assert pw_high_icc > pw_low_icc


def test_pairs_for_power_binary_search_properties():
    target = 0.8
    effect = 0.03
    sd_change = 0.10
    icc = 0.55
    alpha = 0.05
    n_star, pw_star = pta.analytic_pairs_for_power(target, effect, sd_change, icc, alpha, n_lo=5, n_hi=5000)
    assert pw_star >= target
    # One fewer pair should not meet target
    if n_star > 5:
        pw_before = pta.analytic_power_paired(n_star - 1, effect, sd_change, icc, alpha)
        assert pw_before < target


def test_analytic_mde_consistency():
    n = 300
    target = 0.8
    sd_change = 0.10
    icc = 0.55
    alpha = 0.05
    mde = pta.analytic_mde(n, target, sd_change, icc, alpha)
    pw_at_mde = pta.analytic_power_paired(n, mde, sd_change, icc, alpha)
    assert abs(pw_at_mde - target) < 0.02


# ---------- Dataclass and helpers ----------

def test_effective_icc_limits_and_mix():
    spec = pta.AgeTwinSpec(n_pairs=10, effect_abs=0.1, sd_change=1.0, prop_mz=1.0, icc_mz=0.7, icc_dz=0.3)
    assert math.isclose(spec.effective_icc(), 0.7, rel_tol=1e-12)
    spec.prop_mz = 0.0
    assert math.isclose(spec.effective_icc(), 0.3, rel_tol=1e-12)
    spec.prop_mz = 0.5
    eff = spec.effective_icc()
    assert 0.3 <= eff <= 0.7


def test_observed_effect_matches_apply_contamination():
    spec = pta.AgeTwinSpec(n_pairs=5, effect_abs=1.0, sd_change=1.0, contamination_rate=0.3, contamination_effect=0.5)
    assert math.isclose(spec.observed_effect(), pta.apply_contamination(1.0, 0.3, 0.5))


# ---------- Paired p-value computation ----------

def test_compute_paired_pval_degenerate_all_zero_should_not_be_significant():
    d = np.zeros(10)
    p = pta._compute_paired_pval(d)
    assert math.isclose(p, 1.0)


def test_compute_paired_pval_basic():
    # Large mean relative to sd -> significant
    rng = np.random.default_rng(123)
    d = rng.normal(0.5, 0.1, size=50)
    p = pta._compute_paired_pval(d)
    assert p < 0.001


# ---------- Simulation sanity checks ----------

def test_simulate_pairs_reproducible_and_reasonable():
    spec = pta.AgeTwinSpec(
        n_pairs=200,
        effect_abs=0.03,
        sd_change=0.10,
        prop_mz=0.5,
        icc_mz=0.55,
        icc_dz=0.55,
        alpha=0.05,
        seed=42,
    )
    pw1, avg1, se1 = pta._simulate_pairs(spec, sims=800)
    pw2, avg2, se2 = pta._simulate_pairs(spec, sims=800)
    # same seed/spec -> same results
    assert math.isclose(pw1, pw2)
    assert math.isclose(avg1, avg2)
    assert math.isclose(se1, se2)
    # power should be plausibly high at these parameters
    assert 0.8 < pw1 < 1.0
    assert avg1 > 0


def test_simulate_pairs_vs_analytic_closeness():
    spec = pta.AgeTwinSpec(
        n_pairs=400,
        effect_abs=0.03,
        sd_change=0.10,
        prop_mz=0.5,
        icc_mz=0.55,
        icc_dz=0.55,
        alpha=0.05,
        seed=7,
    )
    icc_eff = spec.effective_icc()
    analytic = pta.analytic_power_paired(spec.n_pairs, spec.effect_abs, spec.sd_change, icc_eff, spec.alpha)
    sim, _, _ = pta._simulate_pairs(spec, sims=1200)
    # Allow some Monte Carlo error
    assert abs(sim - analytic) < 0.05


def test_simulate_coprimary_runs_and_bounds():
    spec1 = pta.AgeTwinSpec(n_pairs=150, effect_abs=0.03, sd_change=0.10, prop_mz=0.5, icc_mz=0.6, icc_dz=0.4, seed=100)
    spec2 = pta.AgeTwinSpec(n_pairs=150, effect_abs=2.0, sd_change=3.0, prop_mz=0.5, icc_mz=0.6, icc_dz=0.4, seed=101)
    jp, p1, p2 = pta._simulate_co_primary(spec1, spec2, sims=600, alpha=0.025, pair_effect_corr=0.9)
    assert 0 <= jp <= 1
    assert 0 <= p1 <= 1
    assert 0 <= p2 <= 1


def test_simulate_extreme_icc_and_corr_do_not_crash():
    spec1 = pta.AgeTwinSpec(n_pairs=50, effect_abs=0.01, sd_change=0.10, icc_mz=0.999, icc_dz=0.999, seed=123)
    spec2 = pta.AgeTwinSpec(n_pairs=50, effect_abs=0.01, sd_change=0.10, icc_mz=0.999, icc_dz=0.999, seed=124)
    # Should run without floating errors/NaNs
    jp, p1, p2 = pta._simulate_co_primary(spec1, spec2, sims=200, alpha=0.025, pair_effect_corr=0.999)
    assert 0 <= jp <= 1


# ---------- CLI resolution helpers ----------

def test_resolve_effect_abs_variants():
    # direct absolute
    args = argparse.Namespace(effect_abs=0.12, endpoint="custom", d_std=None,
                              effect_pct=None, effect_years=None)
    out = pta._resolve_effect_abs(args, sd_change=0.10, icc_eff=0.5, suffix="")
    assert math.isclose(out, 0.12)

    # endpoint-specific helpers
    args_dp = argparse.Namespace(effect_abs=None, endpoint="dunedinpace", effect_pct=5.0,
                                 effect_years=None, d_std=None)
    assert math.isclose(pta._resolve_effect_abs(args_dp, 0.10, 0.5, ""), 0.05)

    args_ga = argparse.Namespace(effect_abs=None, endpoint="grimage", effect_years=2.5,
                                 effect_pct=None, d_std=None)
    assert math.isclose(pta._resolve_effect_abs(args_ga, 0.10, 0.5, ""), 2.5)

    # standardized d with sd_change + ICC
    args_d = argparse.Namespace(effect_abs=None, endpoint="custom", d_std=0.2,
                                effect_pct=None, effect_years=None)
    out_d = pta._resolve_effect_abs(args_d, sd_change=0.10, icc_eff=0.5, suffix="")
    sd_d = pta.sd_diff_from_sd_change_icc(0.10, 0.5)
    assert math.isclose(out_d, 0.2 * sd_d)


def test_resolve_sd_change_variants():
    # direct
    args = argparse.Namespace(sd_change=0.25)
    assert math.isclose(pta._resolve_sd_change(args, ""), 0.25)

    # derived from pre/post
    args2 = argparse.Namespace(sd_change=None, sd_pre=2.0, sd_post=3.0, rho_pre_post=0.8)
    out2 = pta._resolve_sd_change(args2, "")
    assert out2 > 0


# ---------- CLI helpers ----------

def _run_power_cli(args):
    try:
        res = subprocess.run(
            ["python3", "biological_age/power_twin_age.py"] + list(args),
            capture_output=True,
            text=True,
            timeout=60,
        )
        return res.returncode == 0, res.stdout
    except Exception as e:
        return False, str(e)


def _extract_power(output: str) -> float:
    for line in output.splitlines():
        if "Statistical power:" in line or "Marginal power:" in line:
            try:
                return float(line.split(":")[1].split("(")[0].strip())
            except Exception:
                pass
    return -1.0


def _extract_joint_power(output: str) -> float:
    for line in output.splitlines():
        if "Probability both p <" in line:
            try:
                return float(line.split(":")[1].split("(")[0].strip())
            except Exception:
                pass
    return -1.0


def _extract_sample_size(output: str) -> int:
    for line in output.splitlines():
        if "Required COMPLETING pairs:" in line:
            try:
                return int(line.split(":")[1].strip())
            except Exception:
                pass
    return -1


def _extract_mde(output: str) -> float:
    for line in output.splitlines():
        if "MDE (absolute, beneficial):" in line:
            try:
                return float(line.split(":")[1].strip())
            except Exception:
                pass
    return -1.0


def _extract_curve_rows(output: str):
    rows = []
    for line in output.strip().splitlines():
        if line and line[0].isdigit():
            try:
                n_str, pw_str = line.split(',')
                rows.append((int(n_str.strip()), float(pw_str.strip())))
            except Exception:
                pass
    return rows


# ---------- CLI-based tests ----------

def test_cli_coprimary_alpha_auto_adjust():
    ok, out = _run_power_cli([
        "--mode", "co-primary-power", "--n-pairs", "120",
        "--endpoint", "dunedinpace", "--effect-pct", "3.0", "--sd-change", "0.10",
        "--endpoint2", "grimage", "--effect2-years", "1.0", "--sd2-change", "3.0",
        "--use-simulation", "--sims", "500"
    ])
    assert ok
    line_alpha = next((ln for ln in out.splitlines() if ln.strip().startswith("Alpha per endpoint:")), "")
    assert "0.025" in line_alpha


def test_cli_coprimary_naming_variant():
    ok, _ = _run_power_cli([
        "--mode", "co-primary-power", "--n-pairs", "120",
        "--endpoint", "dunedinpace", "--effect-pct", "3.0", "--sd-change", "0.10",
        "--endpoint2", "grimage", "--effect2-years", "1.0", "--sd2-change", "3.0",
        "--alpha", "0.025", "--use-simulation", "--sims", "500"
    ])
    assert ok


def test_cli_curve_monotonicity_and_consistency():
    ok_curve, out_curve = _run_power_cli([
        "--mode", "curve", "--endpoint", "dunedinpace", "--effect-pct", "3.0",
        "--sd-change", "0.10", "--alpha", "0.05",
        "--curve-start", "50", "--curve-stop", "200", "--curve-step", "50"
    ])
    ok_point, out_point = _run_power_cli([
        "--mode", "power", "--n-pairs", "100",
        "--endpoint", "dunedinpace", "--effect-pct", "3.0",
        "--sd-change", "0.10", "--alpha", "0.05"
    ])
    assert ok_curve and ok_point
    rows = _extract_curve_rows(out_curve)
    assert len(rows) >= 2
    assert all(rows[i][1] < rows[i+1][1] for i in range(len(rows)-1))
    pw_point = _extract_power(out_point)
    pw_curve_100 = next((pw for n, pw in rows if n == 100), None)
    assert pw_curve_100 is not None
    assert abs(pw_curve_100 - pw_point) < 1e-6


def test_cli_coprimary_rho_zero_product_bound():
    ok, out = _run_power_cli([
        "--mode", "co-primary-power", "--n-pairs", "160",
        "--endpoint", "dunedinpace", "--effect-pct", "3.0", "--sd-change", "0.10",
        "--endpoint2", "grimage", "--effect2-years", "1.0", "--sd2-change", "3.0",
        "--alpha", "0.025", "--use-simulation", "--sims", "1200", "--pair-effect-corr", "0.0"
    ])
    assert ok
    lines = out.splitlines()
    power1, power2 = None, None
    for idx, ln in enumerate(lines):
        if ln.strip().startswith('ENDPOINT 1:'):
            for l in lines[idx:]:
                if 'Marginal power:' in l:
                    power1 = float(l.split(':')[1].split('(')[0].strip()); break
        if ln.strip().startswith('ENDPOINT 2:'):
            for l in lines[idx:]:
                if 'Marginal power:' in l:
                    power2 = float(l.split(':')[1].split('(')[0].strip()); break
    joint = _extract_joint_power(out)
    assert power1 is not None and power2 is not None and joint >= 0
    assert abs(joint - (power1 * power2)) < 0.05  # allow MC tolerance


def test_cli_contamination_note_and_mde_outputs():
    ok_power, out_power = _run_power_cli([
        "--mode", "power", "--n-pairs", "150",
        "--endpoint", "dunedinpace", "--effect-pct", "3.0",
        "--sd-change", "0.10", "--alpha", "0.05",
        "--contamination-rate", "0.30", "--contamination-effect", "0.50"
    ])
    assert ok_power
    assert any('CONTAMINATION ADJUSTMENT:' in ln for ln in out_power.splitlines())

    ok_mde, out_mde = _run_power_cli([
        "--mode", "mde", "--n-pairs", "150", "--target-power", "0.80",
        "--endpoint", "dunedinpace", "--sd-change", "0.10", "--alpha", "0.05",
        "--contamination-rate", "0.30", "--contamination-effect", "0.50"
    ])
    assert ok_mde
    has_mde_obs = any('Observed MDE (on test scale):' in ln for ln in out_mde.splitlines())
    has_mde_true = any('Underlying true effect required (given contamination):' in ln for ln in out_mde.splitlines())
    assert has_mde_obs and has_mde_true


def test_cli_validation_errors():
    ok_bad_icc, _ = _run_power_cli([
        "--mode", "power", "--n-pairs", "50",
        "--endpoint", "dunedinpace", "--effect-pct", "3.0",
        "--sd-change", "0.10", "--icc-mz", "-0.1", "--icc-dz", "0.3"
    ])
    assert not ok_bad_icc

    ok_bad_attr, _ = _run_power_cli([
        "--mode", "power", "--n-pairs", "100",
        "--endpoint", "dunedinpace", "--effect-pct", "3.0",
        "--sd-change", "0.10", "--attrition-rate", "1.0"
    ])
    assert not ok_bad_attr


def test_cli_pairs_for_power_enrollment_attrition():
    ok, out = _run_power_cli([
        "--mode", "pairs-for-power", "--target-power", "0.80",
        "--endpoint", "dunedinpace", "--effect-pct", "3.0",
        "--sd-change", "0.10", "--alpha", "0.05", "--attrition-rate", "0.25"
    ])
    assert ok
    enrollment = None
    for line in out.splitlines():
        if 'Required ENROLLMENT:' in line:
            try:
                enrollment = int(line.split(':')[1].split('pairs')[0].strip())
            except Exception:
                pass
    assert enrollment == 108


def test_cli_near_null_analytic_alpha():
    ok, out = _run_power_cli([
        "--mode", "power", "--n-pairs", "150",
        "--endpoint", "dunedinpace", "--effect-pct", "0.0001",
        "--sd-change", "0.10", "--alpha", "0.05"
    ])
    assert ok
    pw = _extract_power(out)
    assert abs(pw - 0.05) < 0.001


def test_cli_nan_hotspot_guard_reasonable_n():
    ok, out = _run_power_cli([
        "--mode", "pairs-for-power", "--target-power", "0.80",
        "--endpoint", "dunedinpace", "--effect-pct", "3.0",
        "--sd-change", "0.10", "--alpha", "0.05"
    ])
    assert ok
    n = _extract_sample_size(out)
    assert n > 0 and n < 1000


# ---------- Additional unit tests (expanded coverage) ----------

def test_validate_probability_toggles():
    pta.validate_probability(0.0, "p", allow_zero=True)
    with pytest.raises(ValueError):
        pta.validate_probability(0.0, "p", allow_zero=False)
    pta.validate_probability(1.0, "p", allow_one=True)
    with pytest.raises(ValueError):
        pta.validate_probability(1.0, "p", allow_one=False)


def test_sd_change_from_pre_post_invalid_rho_raises():
    with pytest.raises(ValueError):
        pta.sd_change_from_pre_post(1.0, 1.0, 1.1)
    with pytest.raises(ValueError):
        pta.sd_change_from_pre_post(1.0, 1.0, -1.1)


def test_two_sided_power_normal_monotonic_in_abs_lambda():
    alpha = 0.05
    pw0 = pta._two_sided_power_normal(0.0, alpha)
    pw1 = pta._two_sided_power_normal(1.0, alpha)
    pw2 = pta._two_sided_power_normal(2.0, alpha)
    # At lambda=0, power ~= alpha
    assert abs(pw0 - alpha) < 1e-12
    assert pw0 < pw1 < pw2


def test_analytic_power_paired_normal_fallback_large_lambda():
    # Choose parameters such that sqrt(n)*d > 20 triggers normal approximation path
    n = 100
    sd_change = 1.0
    icc = 0.0
    effect_abs = 3.0  # d = 3/sqrt(2) ~ 2.12; lambda ~ 21.2
    pw = pta.analytic_power_paired(n, effect_abs, sd_change, icc, 0.05)
    assert 0.99 < pw <= 1.0


def test_analytic_mde_monotonic_in_n_and_alpha():
    sd_change = 0.10
    icc = 0.55
    # MDE decreases as n increases
    mde_small_n = pta.analytic_mde(100, 0.80, sd_change, icc, 0.05)
    mde_large_n = pta.analytic_mde(500, 0.80, sd_change, icc, 0.05)
    assert mde_large_n < mde_small_n
    # MDE decreases as alpha increases (less stringent)
    mde_alpha_strict = pta.analytic_mde(300, 0.80, sd_change, icc, 0.01)
    mde_alpha_loose = pta.analytic_mde(300, 0.80, sd_change, icc, 0.10)
    assert mde_alpha_loose < mde_alpha_strict


def test_agetwinspec_validation_errors():
    with pytest.raises(ValueError):
        pta.AgeTwinSpec(n_pairs=0, effect_abs=0.1, sd_change=1.0)
    with pytest.raises(ValueError):
        pta.AgeTwinSpec(n_pairs=2, effect_abs=-0.1, sd_change=1.0)
    with pytest.raises(ValueError):
        pta.AgeTwinSpec(n_pairs=2, effect_abs=0.1, sd_change=0.0)
    with pytest.raises(ValueError):
        pta.AgeTwinSpec(n_pairs=2, effect_abs=0.1, sd_change=1.0, prop_mz=1.1)
    with pytest.raises(ValueError):
        pta.AgeTwinSpec(n_pairs=2, effect_abs=0.1, sd_change=1.0, icc_mz=1.0)


def test_effective_icc_clamps_prop_mz_beyond_bounds():
    spec = pta.AgeTwinSpec(n_pairs=10, effect_abs=0.1, sd_change=1.0, prop_mz=0.5, icc_mz=0.8, icc_dz=0.2)
    spec.prop_mz = 1.5
    assert math.isclose(spec.effective_icc(), 0.8, rel_tol=1e-12)
    spec.prop_mz = -0.5
    assert math.isclose(spec.effective_icc(), 0.2, rel_tol=1e-12)


def test_compute_paired_pval_degenerate_all_constant_nonzero_is_significant():
    d = np.full(10, 0.5)
    p = pta._compute_paired_pval(d)
    assert math.isclose(p, 0.0)


def test_simulate_coprimary_joint_power_monotonic_wrt_corr():
    spec1 = pta.AgeTwinSpec(n_pairs=160, effect_abs=0.03, sd_change=0.10, prop_mz=0.5, icc_mz=0.55, icc_dz=0.55, seed=200)
    spec2 = pta.AgeTwinSpec(n_pairs=160, effect_abs=1.0, sd_change=3.0, prop_mz=0.5, icc_mz=0.55, icc_dz=0.55, seed=201)
    jp_low, p1_low, p2_low = pta._simulate_co_primary(spec1, spec2, sims=1200, alpha=0.025, pair_effect_corr=0.0)
    jp_high, p1_high, p2_high = pta._simulate_co_primary(spec1, spec2, sims=1200, alpha=0.025, pair_effect_corr=0.9)
    # Joint power should not decrease when correlation increases (tolerance for MC noise)
    assert jp_high + 0.02 >= jp_low
    # Joint power is bounded by marginals
    assert jp_low <= min(p1_low, p2_low)
    assert jp_high <= min(p1_high, p2_high)


# ---------- Additional CLI error-case tests ----------

def test_cli_missing_endpoint2_in_coprimary_errors():
    ok, _ = _run_power_cli([
        "--mode", "co-primary-power", "--n-pairs", "120",
        "--endpoint", "dunedinpace", "--effect-pct", "3.0", "--sd-change", "0.10",
        "--alpha", "0.025", "--use-simulation", "--sims", "200"
    ])
    assert not ok


def test_cli_missing_effect_errors_for_power():
    ok, _ = _run_power_cli([
        "--mode", "power", "--n-pairs", "100",
        "--endpoint", "custom", "--sd-change", "0.10", "--alpha", "0.05"
    ])
    assert not ok


def test_cli_missing_sd_change_errors():
    ok, _ = _run_power_cli([
        "--mode", "power", "--n-pairs", "100",
        "--endpoint", "dunedinpace", "--effect-pct", "3.0", "--alpha", "0.05"
    ])
    assert not ok
