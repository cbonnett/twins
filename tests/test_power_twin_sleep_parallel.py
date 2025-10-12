import math

import sleep.power_twin_sleep as pts


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
    # Allow three standard errors plus a small cushion for numerical differences
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
