"""
Consolidated tests for the Sleep (ISI) study.

Combines protocol validation, effect-size scenarios, and parallel simulation checks.
"""

import math
import sleep.power_twin_sleep as pts


# ---------------- Protocol-oriented tests (from test_sleep_study.py) ----------------


class TestSleepStudyProtocolValidation:
    def test_isi_primary_endpoint_parameters(self):
        # ISI parameters
        sd_pre = 5.5
        sd_post = 5.5
        rho = 0.45
        sd_change = pts.sd_change_from_pre_post(sd_pre, sd_post, rho)

        effect_mcid = 6.0  # MCID
        for n_pairs in [100, 150, 200, 250]:
            pw = pts.analytic_power_paired(n_pairs, effect_mcid, sd_change, icc_eff=0.55, alpha=0.05)
            if n_pairs >= 150:
                assert pw > 0.80, f"Power {pw:.2f} unexpectedly low at n={n_pairs}"

    def test_eight_week_intervention_appropriateness(self):
        effect = 6.0
        sd_change = 4.0
        n_pairs = 150
        icc = 0.55
        pw = pts.analytic_power_paired(n_pairs, effect, sd_change, icc, 0.05)
        assert 0 < pw < 1
        assert pw > 0.90

    def test_sleep_efficiency_secondary_endpoint(self):
        effect = 7.5
        sd_change = 12.0
        n_pairs = 150
        icc = 0.60
        pw = pts.analytic_power_paired(n_pairs, effect, sd_change, icc, 0.05)
        assert 0 < pw < 1
        assert pw > 0.60

    def test_comorbid_symptoms_endpoints(self):
        n_pairs = 150
        icc = 0.55
        # PHQ-9
        effect_phq9 = 3.0
        sd_phq9 = 6.0
        pw_phq9 = pts.analytic_power_paired(n_pairs, effect_phq9, sd_phq9, icc, 0.05)
        # GAD-7
        effect_gad7 = 2.5
        sd_gad7 = 5.0
        pw_gad7 = pts.analytic_power_paired(n_pairs, effect_gad7, sd_gad7, icc, 0.05)
        assert 0.40 < pw_phq9 < 0.90
        assert 0.40 < pw_gad7 < 0.90

    def test_sample_size_for_mcid(self):
        effect = 6.0
        sd_change = 4.0
        icc = 0.55
        target_power = 0.80
        n_required, achieved_pw = pts.analytic_pairs_for_power(target_power, effect, sd_change, icc, 0.05)
        assert 50 < n_required < 250
        assert achieved_pw >= 0.80


class TestSleepStudyEffectSizes:
    def test_large_effect_from_meta_analysis(self):
        sd_baseline = 5.5
        smd = 0.76
        effect_absolute = smd * sd_baseline
        sd_change = pts.sd_change_from_pre_post(5.5, 5.5, 0.45)
        n_pairs = 150
        pw = pts.analytic_power_paired(n_pairs, effect_absolute, sd_change, 0.55, 0.05)
        assert pw > 0.85

    def test_conservative_vs_optimistic_scenarios(self):
        n_pairs = 150
        icc = 0.55
        alpha = 0.05
        effect_conservative = 5.0
        sd_conservative = 5.0
        pw_conservative = pts.analytic_power_paired(n_pairs, effect_conservative, sd_conservative, icc, alpha)
        effect_optimistic = 6.0
        sd_optimistic = 4.0
        pw_optimistic = pts.analytic_power_paired(n_pairs, effect_optimistic, sd_optimistic, icc, alpha)
        assert pw_optimistic > pw_conservative + 0.10
        assert pw_conservative > 0.60
        assert pw_optimistic > 0.85

    def test_noninferiority_margin_for_active_control(self):
        margin = 1.0
        sd_change = 4.0
        icc = 0.55
        target_power = 0.80
        n_required, _ = pts.analytic_pairs_for_power(target_power, margin, sd_change, icc, 0.05)
        assert n_required > 200


# ---------------- Parallel/determinism tests (from test_power_twin_sleep_parallel.py) ----------------


def _default_spec(seed: int = 2024) -> pts.TwinTrialSpec:
    return pts.TwinTrialSpec(
        n_total=120,
        effect_points=5.5,
        sd_change=6.5,
        prop_twins=0.85,
        prop_mz=0.5,
        icc_mz=0.45,
        icc_dz=0.25,
        alpha=0.05,
        analysis="cluster_robust",
        seed=seed,
    )


def test_simulate_power_parallel_reproducible():
    spec = _default_spec()
    power1, avg1 = pts.simulate_power(spec, sims=120, n_jobs=3, chunk_size=32)
    power2, avg2 = pts.simulate_power(spec, sims=120, n_jobs=3, chunk_size=32)
    assert power1 == power2
    assert avg1 == avg2


def test_simulate_power_parallel_matches_serial_within_mc_error():
    spec = _default_spec()
    sims = 160
    power_serial, avg_serial = pts.simulate_power(spec, sims=sims, n_jobs=1)
    power_parallel, avg_parallel = pts.simulate_power(spec, sims=sims, n_jobs=4, chunk_size=32)

    mc_se = math.sqrt(power_serial * (1 - power_serial) / sims)
    tolerance = 3.5 * mc_se + 1e-3
    assert abs(power_parallel - power_serial) <= tolerance
    assert math.isfinite(avg_serial)
    assert math.isfinite(avg_parallel)
    assert abs(avg_parallel - avg_serial) <= 2.0


def test_find_n_for_power_parallel_meets_target():
    base_spec = _default_spec(seed=99)
    target_power = 0.70
    sims = 120
    n_req, pw_est = pts.find_n_for_power(
        target_power,
        base_spec,
        sims=sims,
        n_jobs=2,
        chunk_size=48,
        tol=0.02,
    )
    spec_at_n = pts.TwinTrialSpec(
        n_total=n_req,
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
    pw_check, _ = pts.simulate_power(spec_at_n, sims=240, n_jobs=2, chunk_size=48)
    assert pw_check >= target_power - 0.08
    pw_replay, _ = pts.simulate_power(spec_at_n, sims=sims, n_jobs=2, chunk_size=48)
    assert pw_est == pw_replay

