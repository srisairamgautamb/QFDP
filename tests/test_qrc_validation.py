"""
Rigorous QRC Integration Tests
Verify QRC properly generates factors without hardcoding
"""

import numpy as np
import sys

sys.path.insert(0, '/Volumes/Hippocampus/Quant_Finance/QFDP/QFDP')

from qrc import QuantumRecurrentCircuit


class QRCValidator:
    """Validate QRC generates real, dynamic factors."""
    
    def __init__(self):
        self.qrc = QuantumRecurrentCircuit(n_factors=4, n_qubits=8, n_deep_layers=3)
        print("=" * 80)
        print("🧪 QRC INTEGRATION VALIDATION")
        print("=" * 80)
    
    def test_1_factors_vary_with_input(self):
        """Test: Different inputs produce different factors."""
        
        print("\n" + "=" * 80)
        print("TEST 1: Factors Vary With Input (Not Hardcoded)")
        print("=" * 80)
        
        test_inputs = [
            {'prices': 100, 'volatility': 0.15, 'corr_change': 0.0, 'stress': 0.1},
            {'prices': 100, 'volatility': 0.30, 'corr_change': 0.3, 'stress': 0.5},
            {'prices': 100, 'volatility': 0.50, 'corr_change': 0.7, 'stress': 0.9},
            {'prices': 90,  'volatility': 0.25, 'corr_change': 0.2, 'stress': 0.3},
            {'prices': 110, 'volatility': 0.20, 'corr_change': 0.1, 'stress': 0.2},
        ]
        
        factors_list = []
        
        print(f"\n{'Input':<40} {'F1':<10} {'F2':<10} {'F3':<10} {'F4':<10}")
        print("-" * 80)
        
        for i, data in enumerate(test_inputs):
            # Reset state for independent tests
            self.qrc.reset_hidden_state()
            result = self.qrc.forward(data)
            factors = result.factors
            factors_list.append(factors)
            
            desc = f"vol={data['volatility']:.2f}, corr={data['corr_change']:.1f}, stress={data['stress']:.1f}"
            print(f"{desc:<40} {factors[0]:<10.4f} {factors[1]:<10.4f} {factors[2]:<10.4f} {factors[3]:<10.4f}")
        
        factors_array = np.array(factors_list)
        
        # Check variance across inputs
        variance_per_factor = np.var(factors_array, axis=0)
        total_variance = variance_per_factor.sum()
        
        # Check that factors are not identical
        unique_combinations = len(set([tuple(f.round(4)) for f in factors_list]))
        
        print(f"\n📊 Analysis:")
        print(f"   Variance per factor: {variance_per_factor}")
        print(f"   Total variance: {total_variance:.6f}")
        print(f"   Unique combinations: {unique_combinations}/5")
        
        # Pass criteria: variance > 0 AND multiple unique outputs
        passed = total_variance > 0.0001 and unique_combinations >= 3
        
        if passed:
            print("\n✅ PASS: Factors vary with input (NOT hardcoded)")
        else:
            print("\n❌ FAIL: Factors may be hardcoded")
        
        return passed
    
    def test_2_factors_sum_to_one(self):
        """Test: Factors always sum to approximately 1."""
        
        print("\n" + "=" * 80)
        print("TEST 2: Factors Sum to 1.0 (Valid Probability Distribution)")
        print("=" * 80)
        
        sums = []
        
        for i in range(10):
            data = {
                'prices': 100 + np.random.randn() * 10,
                'volatility': 0.2 + np.random.rand() * 0.3,
                'corr_change': np.random.rand(),
                'stress': np.random.rand()
            }
            
            self.qrc.reset_hidden_state()
            result = self.qrc.forward(data)
            factor_sum = result.factors.sum()
            sums.append(factor_sum)
            print(f"   Trial {i+1}: sum = {factor_sum:.6f}")
        
        mean_sum = np.mean(sums)
        std_sum = np.std(sums)
        
        print(f"\n📊 Analysis:")
        print(f"   Mean sum: {mean_sum:.6f} (target: 1.0)")
        print(f"   Std dev:  {std_sum:.6f}")
        
        passed = abs(mean_sum - 1.0) < 0.01
        
        if passed:
            print("\n✅ PASS: Factors sum to 1.0")
        else:
            print("\n❌ FAIL: Factors don't sum to 1.0")
        
        return passed
    
    def test_3_temporal_memory(self):
        """Test: Same input produces different output over time (recurrent)."""
        
        print("\n" + "=" * 80)
        print("TEST 3: Temporal Memory (Recurrent Behavior)")
        print("=" * 80)
        
        self.qrc.reset_hidden_state()
        
        fixed_input = {
            'prices': 100,
            'volatility': 0.25,
            'corr_change': 0.2,
            'stress': 0.3
        }
        
        factors_over_time = []
        
        print(f"\n{'Call':<8} {'F1':<12} {'F2':<12} {'F3':<12} {'F4':<12}")
        print("-" * 60)
        
        for i in range(5):
            result = self.qrc.forward(fixed_input)
            factors_over_time.append(result.factors.copy())
            print(f"t={i:<5} {result.factors[0]:<12.4f} {result.factors[1]:<12.4f} "
                  f"{result.factors[2]:<12.4f} {result.factors[3]:<12.4f}")
        
        # Check if factors change over time
        factors_array = np.array(factors_over_time)
        diffs = []
        for i in range(1, len(factors_array)):
            diff = np.linalg.norm(factors_array[i] - factors_array[i-1])
            diffs.append(diff)
        
        mean_diff = np.mean(diffs)
        
        print(f"\n📊 Analysis:")
        print(f"   Mean change between calls: {mean_diff:.6f}")
        
        # Pass if factors evolve (even slightly)
        passed = mean_diff > 0.0001
        
        if passed:
            print("\n✅ PASS: Temporal memory working (factors evolve)")
        else:
            print("\n⚠️  WARNING: Factors appear static (no temporal memory)")
        
        return passed
    
    def test_4_response_to_stress(self):
        """Test: Factors change meaningfully when stress increases."""
        
        print("\n" + "=" * 80)
        print("TEST 4: Response to Stress Signal")
        print("=" * 80)
        
        stress_levels = [0.0, 0.25, 0.5, 0.75, 1.0]
        factors_by_stress = []
        
        print(f"\n{'Stress':<10} {'F1':<12} {'F2':<12} {'F3':<12} {'F4':<12}")
        print("-" * 60)
        
        for stress in stress_levels:
            self.qrc.reset_hidden_state()
            data = {
                'prices': 100,
                'volatility': 0.25,
                'corr_change': stress / 2,  # Correlated with stress
                'stress': stress
            }
            
            result = self.qrc.forward(data)
            factors_by_stress.append(result.factors.copy())
            print(f"{stress:<10.2f} {result.factors[0]:<12.4f} {result.factors[1]:<12.4f} "
                  f"{result.factors[2]:<12.4f} {result.factors[3]:<12.4f}")
        
        # Check correlation between stress and factor changes
        factors_array = np.array(factors_by_stress)
        factor_0_vs_stress = np.corrcoef(stress_levels, factors_array[:, 0])[0, 1]
        
        total_change = np.linalg.norm(factors_array[-1] - factors_array[0])
        
        print(f"\n📊 Analysis:")
        print(f"   Factor 1 correlation with stress: {factor_0_vs_stress:.4f}")
        print(f"   Total factor change (low→high stress): {total_change:.4f}")
        
        # Pass if there's measurable response
        passed = total_change > 0.05 or abs(factor_0_vs_stress) > 0.3
        
        if passed:
            print("\n✅ PASS: QRC responds to stress signal")
        else:
            print("\n❌ FAIL: QRC doesn't respond to stress")
        
        return passed
    
    def test_5_all_factors_active(self):
        """Test: All 4 factors are being used (not just F1)."""
        
        print("\n" + "=" * 80)
        print("TEST 5: All Factors Active")
        print("=" * 80)
        
        all_factors = []
        
        for i in range(20):
            self.qrc.reset_hidden_state()
            data = {
                'prices': 100 + np.random.randn() * 15,
                'volatility': 0.15 + np.random.rand() * 0.35,
                'corr_change': np.random.rand(),
                'stress': np.random.rand()
            }
            
            result = self.qrc.forward(data)
            all_factors.append(result.factors)
        
        factors_array = np.array(all_factors)
        
        mean_per_factor = np.mean(factors_array, axis=0)
        std_per_factor = np.std(factors_array, axis=0)
        
        print(f"\n📊 Factor Statistics (over 20 runs):")
        print(f"   Factor 1: mean={mean_per_factor[0]:.4f}, std={std_per_factor[0]:.4f}")
        print(f"   Factor 2: mean={mean_per_factor[1]:.4f}, std={std_per_factor[1]:.4f}")
        print(f"   Factor 3: mean={mean_per_factor[2]:.4f}, std={std_per_factor[2]:.4f}")
        print(f"   Factor 4: mean={mean_per_factor[3]:.4f}, std={std_per_factor[3]:.4f}")
        
        # Check that all factors have non-trivial values and variance
        all_active = np.all(mean_per_factor > 0.05) and np.all(std_per_factor > 0.001)
        
        if all_active:
            print("\n✅ PASS: All 4 factors are active and varying")
        else:
            dead_factors = [i+1 for i, (m, s) in enumerate(zip(mean_per_factor, std_per_factor)) 
                          if m < 0.05 or s < 0.001]
            print(f"\n⚠️  WARNING: Factors {dead_factors} appear inactive")
        
        return all_active
    
    def run_all_tests(self):
        """Run complete test suite."""
        
        tests = [
            ('Factors Vary With Input', self.test_1_factors_vary_with_input),
            ('Factors Sum to 1.0', self.test_2_factors_sum_to_one),
            ('Temporal Memory', self.test_3_temporal_memory),
            ('Response to Stress', self.test_4_response_to_stress),
            ('All Factors Active', self.test_5_all_factors_active),
        ]
        
        results = []
        
        for name, test_fn in tests:
            try:
                result = test_fn()
                results.append((name, result))
            except Exception as e:
                print(f"\n❌ ERROR in {name}: {e}")
                import traceback
                traceback.print_exc()
                results.append((name, False))
        
        # Summary
        print("\n\n" + "=" * 80)
        print("📊 VALIDATION SUMMARY")
        print("=" * 80)
        
        for name, passed in results:
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"   {status}  {name}")
        
        passed_count = sum([r[1] for r in results])
        total = len(results)
        
        print(f"\nTotal: {passed_count}/{total} tests passed")
        
        if passed_count == total:
            print("\n🎉 ALL TESTS PASSED - QRC is generating real factors!")
        elif passed_count >= 4:
            print("\n⚠️  MOSTLY PASSING - Minor issues to investigate")
        else:
            print("\n🚨 CRITICAL FAILURES - QRC may be broken")
        
        return passed_count == total


if __name__ == '__main__':
    validator = QRCValidator()
    all_passed = validator.run_all_tests()
