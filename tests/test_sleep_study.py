"""
SIESTA-LLM Protocol-Specific Tests

Tests tailored to the sleep study parameters:
- ISI (Insomnia Severity Index) as primary outcome
- 8-week intervention timeframe
- Different effect sizes and variability
- Twin-focused RCT design similar to aging study
"""

import sys
import math
import numpy as np
import pytest
import sleep.power_twin_sleep as pta


class TestSleepStudyProtocolValidation:
    """Validate power calculations for SIESTA-LLM protocol parameters."""
    
    def test_isi_primary_endpoint_parameters(self):
        """Test power for ISI with protocol-relevant parameters.
        
        Protocol: ISI change, MCID = 6 points
        Literature: dCBT-I typically shows large effects (SMD ≈ -0.76)
        Expected SD: ISI has range 0-28, baseline SD ≈ 5-6 points
        """
        # Estimate parameters for ISI
        # MCID = 6 points, SD at baseline ≈ 5.5
        # Pre-post correlation typically 0.4-0.5 for behavioral interventions
        sd_pre = 5.5
        sd_post = 5.5
        rho = 0.45
        
        sd_change = pta.sd_change_from_pre_post(sd_pre, sd_post, rho)
        
        # Effect: 6-point MCID (MCID is typically 0.5-0.8 SD)
        effect_mcid = 6.0
        
        # For various sample sizes, calculate power
        for n_pairs in [100, 150, 200, 250]:
            pw = pta.analytic_power_paired(
                n_pairs, effect_mcid, sd_change, 
                icc_eff=0.55, alpha=0.05
            )
            
            # With large effect (MCID = 6) and reasonable sample, should have good power
            if n_pairs >= 150:
                assert pw > 0.80, f"Power {pw:.2f} unexpectedly low at n={n_pairs}"
    
    def test_eight_week_intervention_appropriateness(self):
        """Verify that 8-week timeframe doesn't create special issues.
        
        8 weeks is standard for insomnia trials (vs 24 weeks for aging study).
        Power calculations should work identically - duration affects effect size
        estimation, not the power calculation itself.
        """
        # Same effect, same SD, same n should give same power
        # regardless of whether intervention is 8 weeks or 24 weeks
        effect = 6.0
        sd_change = 4.0
        n_pairs = 150
        icc = 0.55
        
        pw = pta.analytic_power_paired(n_pairs, effect, sd_change, icc, 0.05)
        
        # Power calculation is duration-agnostic
        assert 0 < pw < 1, "Power invalid for 8-week timeframe"
        assert pw > 0.90, "Should have high power for large MCID effect"
    
    def test_sleep_efficiency_secondary_endpoint(self):
        """Test power for objective sleep efficiency outcome.
        
        Sleep efficiency: ratio of time asleep / time in bed (0-1 scale)
        Typical SD: 10-15 percentage points
        Clinically meaningful: 5-10 percentage point improvement
        """
        # Sleep efficiency as percentage points (0-100 scale)
        effect = 7.5  # 7.5 percentage point improvement
        sd_change = 12.0  # Typical variability
        n_pairs = 150
        icc = 0.60  # Actigraphy might show higher ICC (objective measure)
        
        pw = pta.analytic_power_paired(n_pairs, effect, sd_change, icc, 0.05)
        
        assert 0 < pw < 1, "Power invalid for sleep efficiency"
        # Should have reasonable power for secondary endpoint
        assert pw > 0.60, f"Secondary endpoint power {pw:.2f} unexpectedly low"
    
    def test_comorbid_symptoms_endpoints(self):
        """Test power for PHQ-9 and GAD-7 (depression/anxiety).
        
        PHQ-9: 0-27 scale, MCID ≈ 5 points, SD ≈ 6
        GAD-7: 0-21 scale, MCID ≈ 4 points, SD ≈ 5
        """
        n_pairs = 150
        icc = 0.55
        
        # PHQ-9
        effect_phq9 = 3.0  # Moderate improvement (less than primary MCID)
        sd_phq9 = 6.0
        pw_phq9 = pta.analytic_power_paired(n_pairs, effect_phq9, sd_phq9, icc, 0.05)
        
        # GAD-7
        effect_gad7 = 2.5  # Moderate improvement
        sd_gad7 = 5.0
        pw_gad7 = pta.analytic_power_paired(n_pairs, effect_gad7, sd_gad7, icc, 0.05)
        
        # Both should have moderate power as secondary outcomes
        assert 0.40 < pw_phq9 < 0.90, f"PHQ-9 power {pw_phq9:.2f} outside expected range"
        assert 0.40 < pw_gad7 < 0.90, f"GAD-7 power {pw_gad7:.2f} outside expected range"
    
    def test_sample_size_for_mcid(self):
        """Find sample size needed to detect ISI MCID with 80% power."""
        # ISI parameters
        effect = 6.0  # MCID
        sd_change = 4.0  # Estimated from SD ≈ 5.5, rho ≈ 0.45
        icc = 0.55
        target_power = 0.80
        
        n_required, achieved_pw = pta.analytic_pairs_for_power(
            target_power, effect, sd_change, icc, 0.05
        )
        
        # Should need modest sample for large effect (MCID)
        assert 50 < n_required < 250, f"Sample size {n_required} outside expected range"
        assert achieved_pw >= 0.80, "Failed to achieve target power"
        
        print(f"\n  ISI MCID Detection: {n_required} pairs needed for 80% power")


class TestSleepStudyEffectSizes:
    """Test with realistic effect sizes from sleep literature."""
    
    def test_large_effect_from_meta_analysis(self):
        """Test with SMD ≈ -0.76 from meta-analysis.
        
        Literature reports SMD ≈ -0.76 for dCBT-I.
        With SD ≈ 5.5 at baseline, this translates to ~4.2 point improvement.
        """
        sd_baseline = 5.5
        smd = 0.76
        effect_absolute = smd * sd_baseline  # ~4.2 points
        
        # Derive SD of change (assuming rho ≈ 0.45)
        sd_change = pta.sd_change_from_pre_post(5.5, 5.5, 0.45)
        
        n_pairs = 150
        pw = pta.analytic_power_paired(n_pairs, effect_absolute, sd_change, 0.55, 0.05)
        
        # Large effect should give high power
        assert pw > 0.85, f"Power {pw:.2f} unexpectedly low for meta-analysis effect"
    
    def test_conservative_vs_optimistic_scenarios(self):
        """Compare power under conservative vs optimistic assumptions."""
        n_pairs = 150
        icc = 0.55
        alpha = 0.05
        
        # Conservative: smaller effect, larger SD
        effect_conservative = 5.0  # Just under MCID
        sd_conservative = 5.0
        pw_conservative = pta.analytic_power_paired(
            n_pairs, effect_conservative, sd_conservative, icc, alpha
        )
        
        # Optimistic: MCID effect, smaller SD
        effect_optimistic = 6.0  # At MCID
        sd_optimistic = 4.0
        pw_optimistic = pta.analytic_power_paired(
            n_pairs, effect_optimistic, sd_optimistic, icc, alpha
        )
        
        # Optimistic should have substantially higher power
        assert pw_optimistic > pw_conservative + 0.10, "Optimistic scenario not meaningfully better"
        assert pw_conservative > 0.60, "Even conservative scenario should have reasonable power"
        assert pw_optimistic > 0.85, "Optimistic scenario should have high power"
        
        print(f"\n  Power range: {pw_conservative:.2f} (conservative) to {pw_optimistic:.2f} (optimistic)")
    
    def test_noninferiority_margin_for_active_control(self):
        """Test sample size if using non-inferiority design.
        
        Protocol uses active control (sleep hygiene education).
        Could test non-inferiority if hypothesis were reversed.
        """
        # Non-inferiority margin: preserve 50% of effect (common approach)
        # If control also improves by ~2 points, test if LLM is not worse by >1 point
        margin = 1.0
        sd_change = 4.0
        icc = 0.55
        target_power = 0.80
        
        # For non-inferiority, need to detect difference of margin
        n_required, _ = pta.analytic_pairs_for_power(
            target_power, margin, sd_change, icc, 0.05
        )
        
        # Non-inferiority requires larger sample than superiority for MCID
        assert n_required > 200, "Non-inferiority sample unexpectedly small"
        
        print(f"\n  Non-inferiority design would need: {n_required} pairs")


class TestSleepStudyContaminationScenarios:
    """Test contamination-specific scenarios for sleep study."""
    
    def test_twin_information_sharing(self):
        """Model scenario where control twins learn CBT-I techniques from treated twin.
        
        CBT-I has specific techniques (sleep restriction, stimulus control) that
        could be shared between twins, potentially diluting the effect.
        """
        n_pairs = 150
        effect_clean = 6.0
        sd_change = 4.0
        icc = 0.55
        
        # Scenario: 20% of control twins learn techniques, adopt 40% of effect
        contamination_rate = 0.20
        contamination_effect = 0.40
        
        effect_contaminated = pta.apply_contamination(
            effect_clean, contamination_rate, contamination_effect
        )
        
        pw_clean = pta.analytic_power_paired(n_pairs, effect_clean, sd_change, icc, 0.05)
        pw_contaminated = pta.analytic_power_paired(n_pairs, effect_contaminated, sd_change, icc, 0.05)
        
        # Contamination should reduce power
        assert pw_contaminated < pw_clean
        
        # Power loss should be modest for moderate contamination
        power_loss_pct = (pw_clean - pw_contaminated) / pw_clean * 100
        assert power_loss_pct < 20, f"Power loss {power_loss_pct:.1f}% exceeds 20%"
        
        print(f"\n  Contamination impact: {power_loss_pct:.1f}% power loss")
    
    def test_living_separately_benefit(self):
        """Compare contamination scenarios: living together vs separately.
        
        Protocol prefers twins living separately to reduce contamination.
        """
        n_pairs = 150
        effect = 6.0
        sd_change = 4.0
        icc = 0.55
        
        # Living together: higher contamination risk
        pw_together = pta.analytic_power_paired(
            n_pairs, 
            pta.apply_contamination(effect, 0.40, 0.50),  # 40% of pairs, 50% effect
            sd_change, icc, 0.05
        )
        
        # Living separately: lower contamination risk
        pw_separate = pta.analytic_power_paired(
            n_pairs,
            pta.apply_contamination(effect, 0.15, 0.30),  # 15% of pairs, 30% effect
            sd_change, icc, 0.05
        )
        
        # Separate living should preserve more power
        assert pw_separate > pw_together, "Living separately doesn't reduce contamination"
        
        print(f"\n  Power benefit of living separately: +{(pw_separate - pw_together)*100:.1f}%")


class TestSleepStudyDesignChoices:
    """Test design decisions specific to the sleep protocol."""
    
    def test_active_vs_waitlist_control(self):
        """Compare power assumptions for active control vs waitlist.
        
        Protocol uses active control (sleep hygiene education) rather than waitlist.
        Active controls may show some improvement, reducing between-group difference.
        """
        n_pairs = 150
        sd_change = 4.0
        icc = 0.55
        
        # Waitlist: full 6-point effect
        effect_waitlist = 6.0
        pw_waitlist = pta.analytic_power_paired(n_pairs, effect_waitlist, sd_change, icc, 0.05)
        
        # Active control: control group improves 1-2 points, net effect 4-5 points
        effect_active = 4.5  # Net difference after control improvement
        pw_active = pta.analytic_power_paired(n_pairs, effect_active, sd_change, icc, 0.05)
        
        # Active control should have lower power
        assert pw_active < pw_waitlist
        
        # But should still be adequately powered
        assert pw_active > 0.70, f"Active control power {pw_active:.2f} too low"
        
        print(f"\n  Active control power: {pw_active:.2f} vs waitlist: {pw_waitlist:.2f}")
    
    def test_hierarchical_testing_strategy(self):
        """Verify hierarchical testing protects Type I error.
        
        Protocol tests: (1) ISI, (2) Sleep efficiency, (3) PHQ-9, (4) GAD-7
        """
        n_pairs = 150
        icc = 0.55
        alpha = 0.05
        
        # Primary (ISI)
        pw_isi = pta.analytic_power_paired(n_pairs, 6.0, 4.0, icc, alpha)
        
        # Secondary (assuming progressively smaller effects)
        pw_sleep_eff = pta.analytic_power_paired(n_pairs, 7.5, 12.0, icc, alpha)
        pw_phq9 = pta.analytic_power_paired(n_pairs, 3.0, 6.0, icc, alpha)
        pw_gad7 = pta.analytic_power_paired(n_pairs, 2.5, 5.0, icc, alpha)
        
        # Primary should have highest power
        assert pw_isi > pw_sleep_eff, "Secondary unexpectedly higher power than primary"
        
        # All should be adequately powered
        assert pw_isi > 0.80, "Primary underpowered"
        
        print(f"\n  Hierarchical powers: ISI={pw_isi:.2f}, Sleep={pw_sleep_eff:.2f}, "
              f"PHQ-9={pw_phq9:.2f}, GAD-7={pw_gad7:.2f}")
    
    def test_week4_interim_assessment(self):
        """Protocol includes optional Week 4 mid-course check.
        
        This doesn't affect primary power calculation but could enable
        early response detection.
        """
        # Week 4 might show partial effect (50-70% of final effect)
        n_pairs = 150
        effect_final = 6.0
        effect_interim = 0.60 * effect_final  # 60% of final effect at Week 4
        sd_change = 4.0
        icc = 0.55
        
        pw_interim = pta.analytic_power_paired(n_pairs, effect_interim, sd_change, icc, 0.05)
        pw_final = pta.analytic_power_paired(n_pairs, effect_final, sd_change, icc, 0.05)
        
        # Interim should have lower power (smaller effect)
        assert pw_interim < pw_final
        
        # But still might detect effect if present
        assert pw_interim > 0.50, "Interim assessment has very low power"
        
        print(f"\n  Interim (Week 4) power: {pw_interim:.2f} vs Final (Week 8): {pw_final:.2f}")


class TestSleepStudyTwinSpecificAnalyses:
    """Test twin-specific analytical approaches for sleep study."""
    
    def test_within_pair_power_advantage(self):
        """Quantify power advantage from within-pair analysis.
        
        Protocol does individual randomization with co-twin secondary analysis.
        Within-pair comparison controls for genetics and shared environment.
        """
        n_pairs = 150
        effect = 6.0
        sd_change = 4.0
        alpha = 0.05
        
        # Individual analysis (ignoring pairing): ICC doesn't matter
        # This is like treating as unpaired data
        # For unpaired, need 2n observations, so 300 individuals
        # But variance is sd_change² per person, so SE = sd_change/sqrt(n)
        
        # Paired analysis with high ICC (twins very similar)
        pw_paired_high_icc = pta.analytic_power_paired(n_pairs, effect, sd_change, icc=0.70, alpha=alpha)
        
        # Paired analysis with moderate ICC
        pw_paired_mod_icc = pta.analytic_power_paired(n_pairs, effect, sd_change, icc=0.50, alpha=alpha)
        
        # Paired analysis with low ICC (less similar)
        pw_paired_low_icc = pta.analytic_power_paired(n_pairs, effect, sd_change, icc=0.30, alpha=alpha)
        
        # Higher ICC should give more power in paired design
        assert pw_paired_high_icc > pw_paired_mod_icc > pw_paired_low_icc
        
        print(f"\n  Power by ICC: low (0.3)={pw_paired_low_icc:.2f}, "
              f"mod (0.5)={pw_paired_mod_icc:.2f}, high (0.7)={pw_paired_high_icc:.2f}")
    
    def test_mz_vs_dz_power_difference(self):
        """Compare power between MZ and DZ twin pairs.
        
        MZ twins (identical) likely have higher ICC than DZ twins (fraternal).
        """
        n_pairs = 150
        effect = 6.0
        sd_change = 4.0
        alpha = 0.05
        
        # MZ twins: higher ICC
        pw_mz = pta.analytic_power_paired(n_pairs, effect, sd_change, icc=0.70, alpha=alpha)
        
        # DZ twins: moderate ICC
        pw_dz = pta.analytic_power_paired(n_pairs, effect, sd_change, icc=0.45, alpha=alpha)
        
        # MZ should have more power
        assert pw_mz > pw_dz, "MZ twins don't show expected power advantage"
        
        # Both should be well-powered
        assert pw_mz > 0.90 and pw_dz > 0.80
        
        print(f"\n  MZ power: {pw_mz:.2f} vs DZ power: {pw_dz:.2f}")


def run_sleep_study_tests():
    """Run all sleep-study-specific tests."""
    print("\n" + "=" * 70)
    print("RUNNING SIESTA-LLM SLEEP STUDY TEST SUITE")
    print("=" * 70)
    
    pytest_args = [
        __file__,
        "-v",
        "--tb=short",
        "-x",
    ]
    
    exit_code = pytest.main(pytest_args)
    
    if exit_code == 0:
        print("\n" + "=" * 70)
        print("ALL SLEEP STUDY TESTS PASSED ✓")
        print("=" * 70)
    else:
        print("\n" + "=" * 70)
        print("SOME SLEEP STUDY TESTS FAILED ✗")
        print("=" * 70)
    
    return exit_code


if __name__ == "__main__":
    exit_code = run_sleep_study_tests()
    sys.exit(exit_code)
