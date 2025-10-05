"""
Advanced and creative tests for power_twin_age.py

These tests go beyond basic validation to check:
- Empirical verification of claimed properties
- Cross-validation against known benchmarks
- Robustness to perturbations
- Pathological edge cases
- Type I error rate calibration
- Consistency across different calculation paths
"""

import sys
import os
import math
import numpy as np
import pytest

# Make biological_age module importable as power_twin_age
HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', 'biological_age')))
import power_twin_age as pta


class TestEmpiricalVerification:
    """Empirically verify claimed statistical properties via simulation."""
    
    def test_type_i_error_rate_calibration(self):
        """Under null hypothesis (effect≈0), rejection rate should equal alpha.
        
        This is a fundamental statistical property: false positive rate = alpha.
        Note: Code requires effect_abs > 0, so we use a tiny effect (0.0001) as proxy for null.
        """
        alpha = 0.05
        spec = pta.AgeTwinSpec(
            n_pairs=100,
            effect_abs=0.0001,  # Tiny effect as proxy for NULL (code requires positive)
            sd_change=0.10,
            icc_mz=0.55,
            icc_dz=0.55,
            alpha=alpha,
            seed=999
        )
        
        # Run many simulations under near-null
        rejection_rate, _, _ = pta._simulate_pairs(spec, sims=5000)
        
        # Rejection rate should be close to alpha
        # Use binomial SE: sqrt(p*(1-p)/n) = sqrt(0.05*0.95/5000) ≈ 0.003
        se = math.sqrt(alpha * (1 - alpha) / 5000)
        tolerance = 3 * se  # 3 standard errors (~99% confidence)
        
        assert abs(rejection_rate - alpha) < tolerance, (
            f"Type I error rate {rejection_rate:.4f} differs from alpha {alpha:.4f} "
            f"by {abs(rejection_rate - alpha):.4f}, exceeds tolerance {tolerance:.4f}"
        )
    
    def test_empirical_power_matches_claimed_power(self):
        """Empirical power from simulation should match analytic prediction.
        
        This validates that our power calculations are not just consistent but accurate.
        """
        spec = pta.AgeTwinSpec(
            n_pairs=150,
            effect_abs=0.04,
            sd_change=0.10,
            icc_mz=0.55,
            icc_dz=0.55,
            seed=777
        )
        
        # Predicted power (analytic)
        power_predicted = pta.analytic_power_paired(
            spec.n_pairs, spec.effect_abs, spec.sd_change,
            spec.effective_icc(), spec.alpha
        )
        
        # Empirical power (simulation)
        power_empirical, _, _ = pta._simulate_pairs(spec, sims=5000)
        
        # Should match within simulation error
        se = math.sqrt(power_empirical * (1 - power_empirical) / 5000)
        tolerance = 3 * se
        
        assert abs(power_empirical - power_predicted) < tolerance, (
            f"Empirical power {power_empirical:.3f} differs from predicted {power_predicted:.3f}"
        )
    
    def test_effect_size_recovery(self):
        """Simulation should recover the true effect size on average."""
        true_effect = 0.035
        spec = pta.AgeTwinSpec(
            n_pairs=200,
            effect_abs=true_effect,
            sd_change=0.10,
            seed=555
        )
        
        _, avg_estimated_effect, _ = pta._simulate_pairs(spec, sims=3000)
        
        # Average estimate should be close to true effect
        # SE of mean ≈ true_effect / sqrt(200) ≈ 0.0025 in each simulation
        # Across 3000 sims, SE of average ≈ 0.0025 / sqrt(3000) ≈ 0.00005
        tolerance = 0.005  # Conservative tolerance
        
        assert abs(avg_estimated_effect - true_effect) < tolerance, (
            f"Estimated effect {avg_estimated_effect:.4f} differs from true {true_effect:.4f}"
        )


class TestRobustnessAndPerturbations:
    """Test that small input changes produce small output changes (Lipschitz continuity)."""
    
    def test_power_continuity_in_effect_size(self):
        """Power should change smoothly with effect size (no discontinuities)."""
        base_effect = 0.03
        epsilon = 0.001  # Small perturbation
        
        pw_base = pta.analytic_power_paired(200, base_effect, 0.10, 0.55, 0.05)
        pw_perturbed = pta.analytic_power_paired(200, base_effect + epsilon, 0.10, 0.55, 0.05)
        
        # Change in power should be small (Lipschitz continuity)
        delta_power = abs(pw_perturbed - pw_base)
        assert delta_power < 0.05, f"Power changed by {delta_power:.4f} for tiny effect change"
    
    def test_power_continuity_in_sample_size(self):
        """Power should increase smoothly with sample size."""
        powers = []
        for n in range(180, 221, 10):  # n = 180, 190, ..., 220
            pw = pta.analytic_power_paired(n, 0.03, 0.10, 0.55, 0.05)
            powers.append(pw)
        
        # Check all consecutive differences are similar (smooth increase)
        diffs = [powers[i+1] - powers[i] for i in range(len(powers)-1)]
        avg_diff = sum(diffs) / len(diffs)
        
        for diff in diffs:
            # No single jump should be 3x the average
            assert diff < 3 * avg_diff, f"Non-smooth power increase detected: {diff:.4f}"
    
    def test_effective_icc_robustness(self):
        """Effective ICC should be stable to small changes in proportions."""
        spec_base = pta.AgeTwinSpec(
            n_pairs=100, effect_abs=0.03, sd_change=0.1,
            prop_mz=0.50, icc_mz=0.6, icc_dz=0.4
        )
        spec_perturbed = pta.AgeTwinSpec(
            n_pairs=100, effect_abs=0.03, sd_change=0.1,
            prop_mz=0.51, icc_mz=0.6, icc_dz=0.4
        )
        
        icc_eff_base = spec_base.effective_icc()
        icc_eff_perturbed = spec_perturbed.effective_icc()
        
        # Should change by at most 1% for 1% change in prop_mz
        assert abs(icc_eff_perturbed - icc_eff_base) < 0.01


class TestConsistencyAcrossPaths:
    """Test that different calculation paths give the same answer."""
    
    def test_mde_gives_target_power(self):
        """MDE should give exactly the target power when used in power calculation."""
        n_pairs = 200
        target_power = 0.85
        sd_change = 0.10
        icc = 0.55
        alpha = 0.05
        
        # Calculate MDE for 85% power
        mde = pta.analytic_mde(n_pairs, target_power, sd_change, icc, alpha)
        
        # Use that MDE to calculate power - should get target_power back
        achieved_power = pta.analytic_power_paired(n_pairs, mde, sd_change, icc, alpha)
        
        assert abs(achieved_power - target_power) < 0.001, (
            f"MDE {mde:.4f} gives power {achieved_power:.3f}, not target {target_power:.3f}"
        )
    
    def test_pairs_for_power_gives_target_power(self):
        """Sample size for target power should actually achieve that power."""
        target_power = 0.90
        effect_abs = 0.03
        sd_change = 0.10
        icc = 0.55
        alpha = 0.05
        
        # Calculate sample size for 90% power
        n_required, achieved_power = pta.analytic_pairs_for_power(
            target_power, effect_abs, sd_change, icc, alpha
        )
        
        # Verify the achieved power is correct by recalculating
        power_check = pta.analytic_power_paired(n_required, effect_abs, sd_change, icc, alpha)
        
        assert abs(power_check - achieved_power) < 0.01, "Sample size calculation inconsistent"
        assert achieved_power >= target_power, f"Failed to achieve target power"
    
    def test_contamination_path_independence(self):
        """Contamination applied before vs after should give same result."""
        # Path 1: Apply contamination, then calculate power
        effect_contaminated = pta.apply_contamination(0.03, 0.2, 0.5)
        pw1 = pta.analytic_power_paired(200, effect_contaminated, 0.10, 0.55, 0.05)
        
        # Path 2: Use AgeTwinSpec which applies contamination internally
        spec = pta.AgeTwinSpec(
            n_pairs=200, effect_abs=0.03, sd_change=0.10,
            contamination_rate=0.2, contamination_effect=0.5
        )
        effect_from_spec = spec.observed_effect()
        pw2 = pta.analytic_power_paired(200, effect_from_spec, 0.10, 0.55, 0.05)
        
        assert abs(pw1 - pw2) < 1e-10, "Different contamination paths give different results"


class TestPathologicalCases:
    """Test deliberately pathological scenarios to find hidden bugs."""
    
    def test_all_mz_twins(self):
        """100% MZ twins should work correctly."""
        spec = pta.AgeTwinSpec(
            n_pairs=150, effect_abs=0.03, sd_change=0.10,
            prop_mz=1.0, icc_mz=0.70, icc_dz=0.40
        )
        
        icc_eff = spec.effective_icc()
        assert abs(icc_eff - 0.70) < 1e-6, "100% MZ should give MZ ICC"
        
        pw = pta.analytic_power_paired(
            spec.n_pairs, spec.effect_abs, spec.sd_change, icc_eff, spec.alpha
        )
        assert 0 < pw < 1, "Power invalid for 100% MZ twins"
    
    def test_all_dz_twins(self):
        """100% DZ twins should work correctly."""
        spec = pta.AgeTwinSpec(
            n_pairs=150, effect_abs=0.03, sd_change=0.10,
            prop_mz=0.0, icc_mz=0.70, icc_dz=0.40
        )
        
        icc_eff = spec.effective_icc()
        assert abs(icc_eff - 0.40) < 1e-6, "100% DZ should give DZ ICC"
        
        pw = pta.analytic_power_paired(
            spec.n_pairs, spec.effect_abs, spec.sd_change, icc_eff, spec.alpha
        )
        assert 0 < pw < 1, "Power invalid for 100% DZ twins"
    
    def test_extreme_mz_dz_icc_difference(self):
        """Very different MZ vs DZ ICCs should be handled."""
        spec = pta.AgeTwinSpec(
            n_pairs=150, effect_abs=0.03, sd_change=0.10,
            prop_mz=0.5, icc_mz=0.95, icc_dz=0.05
        )
        
        icc_eff = spec.effective_icc()
        # Should be somewhere in between
        assert 0.05 <= icc_eff <= 0.95, f"Effective ICC {icc_eff} outside bounds"
        
        # Should still give valid power
        pw = pta.analytic_power_paired(
            spec.n_pairs, spec.effect_abs, spec.sd_change, icc_eff, spec.alpha
        )
        assert 0 < pw < 1, "Power invalid with extreme ICC difference"
    
    def test_very_high_contamination(self):
        """90% contamination at 90% effect magnitude (extreme scenario)."""
        spec = pta.AgeTwinSpec(
            n_pairs=500,  # Need large sample for remaining effect
            effect_abs=0.10,  # Need large effect to have anything left
            sd_change=0.10,
            contamination_rate=0.90,
            contamination_effect=0.90
        )
        
        effect_obs = spec.observed_effect()
        # 90% * 90% = 81% reduction, leaving 19% of effect
        expected = 0.10 * (1 - 0.90 * 0.90)
        assert abs(effect_obs - expected) < 1e-10, "Contamination calculation wrong"
        
        # Should still compute power (even if low)
        pw = pta.analytic_power_paired(
            spec.n_pairs, effect_obs, spec.sd_change, spec.effective_icc(), spec.alpha
        )
        assert 0 <= pw <= 1, "Power invalid with extreme contamination"
    
    def test_perfect_correlation_in_sd_change_derivation(self):
        """Perfect pre-post correlation should give SD change ≈ 0."""
        sd_change = pta.sd_change_from_pre_post(1.0, 1.0, 0.9999)
        assert sd_change < 0.02, f"Near-perfect correlation should give tiny SD, got {sd_change}"
    
    def test_negative_correlation_in_sd_change_derivation(self):
        """Negative correlation should increase SD of change."""
        sd_pos = pta.sd_change_from_pre_post(1.0, 1.0, 0.5)
        sd_neg = pta.sd_change_from_pre_post(1.0, 1.0, -0.5)
        
        assert sd_neg > sd_pos, "Negative correlation should increase SD of change"


class TestCrossValidation:
    """Cross-validate against published literature values and known benchmarks."""
    
    def test_calerie_dunedinpace_benchmark(self):
        """Validate against CALERIE trial results.
        
        CALERIE: 2-year caloric restriction, n≈200, DunedinPACE slowing 2-3%,
        achieved statistical significance (p<0.003).
        
        If we simulate similar conditions, we should get similar power.
        """
        # CALERIE-like parameters
        n_pairs = 100  # ~200 individuals, rough approximation
        effect = 0.025  # 2.5% slowing (middle of 2-3%)
        sd_change = 0.10  # Reasonable assumption
        icc = 0.55  # Protocol assumption
        alpha = 0.05
        
        pw = pta.analytic_power_paired(n_pairs, effect, sd_change, icc, alpha)
        
        # Should have decent power (they got p<0.003, suggesting high power)
        # We expect 70-90% power for detection
        assert 0.50 < pw < 0.95, (
            f"Power {pw:.2f} for CALERIE-like parameters seems unrealistic"
        )
    
    def test_fitzgerald_grimage_benchmark(self):
        """Validate against Fitzgerald 2021 study.
        
        8-week intervention, n=43 treated vs 40 control, achieved 3.23 year
        DNAmAge reduction (p=0.018).
        """
        # Fitzgerald-like parameters (rough approximation)
        n_pairs = 40  # ~40 pairs for 80 individuals
        effect = 3.0  # 3-year reduction
        sd_change = 5.0  # Estimated from biological age SD
        icc = 0.50
        alpha = 0.05
        
        pw = pta.analytic_power_paired(n_pairs, effect, sd_change, icc, alpha)
        
        # They achieved significance (p=0.018), suggesting power >80%
        assert pw > 0.60, (
            f"Power {pw:.2f} too low for Fitzgerald parameters that achieved significance"
        )
    
    def test_typical_rct_power(self):
        """Standard RCT aims for 80% power with Cohen's d ≈ 0.5."""
        # With paired design, what sample size gives 80% power for d=0.5?
        # d = effect / SD(diff), and SD(diff) = sqrt(2*(1-ICC)*sigma^2)
        # If ICC=0.5, SD(diff) = sigma. So effect = 0.5*sigma.
        
        effect = 0.05  # 5% slowing
        sd_change = 0.10
        icc = 0.50
        d = pta.d_paired(effect, sd_change, icc)
        
        # d should be about 0.5
        assert 0.4 < d < 0.6, f"Cohen's d {d:.2f} not in expected range"
        
        # Find sample size for 80% power
        n, pw = pta.analytic_pairs_for_power(0.80, effect, sd_change, icc, 0.05)
        
        # Typical RCTs need 50-100 per group, so 50-100 pairs sounds right
        assert 30 < n < 150, f"Sample size {n} seems unrealistic for medium effect"


class TestAdvancedSimulationProperties:
    """Test advanced statistical properties of the simulation."""
    
    def test_simulation_variance_matches_theory(self):
        """Simulated effect sizes should have correct variance.
        
        Under the model, Var(diff) = 2*(1-ICC)*sigma^2
        Note: Using tiny effect instead of zero due to code validation.
        """
        spec = pta.AgeTwinSpec(
            n_pairs=100,
            effect_abs=0.0001,  # Tiny effect (code requires positive)
            sd_change=0.10,
            icc_mz=0.55,
            icc_dz=0.55,
            seed=333
        )
        
        # Run simulation and collect estimates
        rng = np.random.default_rng(333)
        estimates = []
        effect_obs = spec.observed_effect()  # Nearly zero
        
        for _ in range(1000):  # 1000 simulated trials
            # Generate one trial
            z_is_mz = rng.random(spec.n_pairs) < spec.prop_mz
            y_treat = np.empty(spec.n_pairs)
            y_ctrl = np.empty(spec.n_pairs)
            
            for i in range(spec.n_pairs):
                icc = spec.icc_mz if z_is_mz[i] else spec.icc_dz
                sigma2 = spec.sd_change ** 2
                sd_b = math.sqrt(max(0.0, icc) * sigma2)
                sd_e = math.sqrt(max(0.0, 1.0 - icc) * sigma2)
                b = rng.normal(0.0, sd_b)
                e1 = rng.normal(0.0, sd_e)
                e2 = rng.normal(0.0, sd_e)
                
                # Use tiny effect
                y_treat[i] = -effect_obs + b + e1
                y_ctrl[i] = 0.0 + b + e2
            
            d = y_treat - y_ctrl
            estimates.append(np.mean(d))
        
        # Variance of mean difference
        empirical_var = np.var(estimates, ddof=1)
        
        # Theoretical variance: Var(mean diff) = Var(diff) / n = 2*(1-ICC)*sigma^2 / n
        icc_eff = spec.effective_icc()
        theoretical_var = 2 * (1 - icc_eff) * (spec.sd_change ** 2) / spec.n_pairs
        
        # Should match within ~20% (Monte Carlo error)
        rel_error = abs(empirical_var - theoretical_var) / theoretical_var
        assert rel_error < 0.20, (
            f"Empirical variance {empirical_var:.6f} differs from "
            f"theoretical {theoretical_var:.6f} by {rel_error*100:.1f}%"
        )
    
    def test_coprimary_null_correlation(self):
        """Under null hypothesis, co-primary endpoints should be independent if uncorrelated.
        
        Note: Using tiny effects instead of zero due to code validation.
        """
        spec1 = pta.AgeTwinSpec(
            n_pairs=100, effect_abs=0.0001, sd_change=0.10, seed=111
        )
        spec2 = pta.AgeTwinSpec(
            n_pairs=100, effect_abs=0.0001, sd_change=0.15, seed=112
        )
        
        # With zero correlation and null hypothesis, joint rejection should be ~alpha^2
        pw_joint, pw1, pw2 = pta._simulate_co_primary(
            spec1, spec2, sims=2000, alpha=0.05, pair_effect_corr=0.0
        )
        
        # Under null with independence: P(both reject) ≈ alpha * alpha = 0.0025
        expected_joint = 0.05 * 0.05
        
        # Individual powers should be ≈ alpha under null
        assert abs(pw1 - 0.05) < 0.02, f"Endpoint 1 Type I error {pw1:.3f} not ≈ 0.05"
        assert abs(pw2 - 0.05) < 0.02, f"Endpoint 2 Type I error {pw2:.3f} not ≈ 0.05"
        
        # Joint should be near alpha^2 (with wide tolerance due to small probability)
        assert abs(pw_joint - expected_joint) < 0.015, (
            f"Joint rejection rate {pw_joint:.4f} not ≈ {expected_joint:.4f} under null"
        )


class TestInvarianceProperties:
    """Test that certain transformations preserve expected relationships."""
    
    def test_scale_invariance_of_cohen_d(self):
        """Cohen's d should be invariant to scaling of both effect and SD."""
        d1 = pta.d_paired(0.03, 0.10, 0.55)
        
        # Scale both by 2x
        d2 = pta.d_paired(0.06, 0.20, 0.55)
        
        # Should give same Cohen's d
        assert abs(d1 - d2) < 1e-10, "Cohen's d not scale invariant"
    
    def test_power_increases_if_everything_else_improves(self):
        """If we improve on all dimensions, power should increase."""
        pw_base = pta.analytic_power_paired(100, 0.03, 0.10, 0.50, 0.05)
        
        # Improve: more pairs, larger effect, smaller SD, higher ICC
        pw_improved = pta.analytic_power_paired(150, 0.04, 0.09, 0.60, 0.05)
        
        assert pw_improved > pw_base, "Power didn't increase when all factors improved"
    
    def test_symmetry_of_mz_dz_proportions(self):
        """Swapping MZ/DZ ICC values and proportions should give reciprocal effective ICC."""
        spec1 = pta.AgeTwinSpec(
            n_pairs=100, effect_abs=0.03, sd_change=0.1,
            prop_mz=0.3, icc_mz=0.7, icc_dz=0.4
        )
        spec2 = pta.AgeTwinSpec(
            n_pairs=100, effect_abs=0.03, sd_change=0.1,
            prop_mz=0.7, icc_mz=0.4, icc_dz=0.7  # Swapped ICC values and proportion
        )
        
        icc1 = spec1.effective_icc()
        icc2 = spec2.effective_icc()
        
        # Effective ICCs should be related by the swap
        # eff1 = 1 - [0.3*(1-0.7) + 0.7*(1-0.4)] = 1 - [0.09 + 0.42] = 0.49
        # eff2 = 1 - [0.7*(1-0.4) + 0.3*(1-0.7)] = 1 - [0.42 + 0.09] = 0.49
        assert abs(icc1 - icc2) < 1e-10, "Symmetry broken in effective ICC calculation"


def run_advanced_tests():
    """Run all advanced tests."""
    print("\n" + "=" * 70)
    print("RUNNING ADVANCED/CREATIVE TEST SUITE")
    print("=" * 70)
    
    pytest_args = [
        __file__,
        "-v",
        "--tb=short",
        "-x",  # Stop on first failure
    ]
    
    exit_code = pytest.main(pytest_args)
    
    if exit_code == 0:
        print("\n" + "=" * 70)
        print("ALL ADVANCED TESTS PASSED ✓")
        print("=" * 70)
    else:
        print("\n" + "=" * 70)
        print("SOME ADVANCED TESTS FAILED ✗")
        print("=" * 70)
    
    return exit_code


if __name__ == "__main__":
    exit_code = run_advanced_tests()
    sys.exit(exit_code)

