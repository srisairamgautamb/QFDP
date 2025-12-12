"""
FB-IQFT IBM Quantum Hardware Test
==================================

Deploy FB-IQFT to real IBM Quantum hardware and measure wait time + results.
"""

import numpy as np
import sys
import os
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from qfdp.fb_iqft.pricing import factor_based_qfdp

def test_fb_iqft_hardware():
    """Test FB-IQFT on IBM Quantum hardware."""
    
    print("="*70)
    print("FB-IQFT IBM QUANTUM HARDWARE TEST")
    print("="*70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 5-asset basket (same as demo)
    N = 5
    weights = np.ones(N) / N
    vols = np.array([0.20, 0.25, 0.18, 0.22, 0.19])
    
    # Moderate correlation
    corr = np.array([
        [1.0, 0.5, 0.3, 0.2, 0.1],
        [0.5, 1.0, 0.4, 0.3, 0.2],
        [0.3, 0.4, 1.0, 0.5, 0.3],
        [0.2, 0.3, 0.5, 1.0, 0.4],
        [0.1, 0.2, 0.3, 0.4, 1.0]
    ])
    
    # Option parameters
    spot = 100.0
    strike = 105.0
    rate = 0.05
    maturity = 1.0
    
    print("Portfolio Configuration:")
    print(f"  Assets: {N}")
    print(f"  Spot: ${spot}")
    print(f"  Strike: ${strike}")
    print(f"  Maturity: {maturity}Y")
    print()
    
    print("Deploying to IBM Quantum Hardware...")
    print("  Backend: Auto-selecting least busy (excluding ibm_fez)")
    print("  This may take several minutes (queue wait time)")
    print()
    
    # Track total time
    start_wall_time = time.time()
    
    try:
        # Run FB-IQFT on hardware
        result = factor_based_qfdp(
            weights,
            vols,
            corr,
            spot_value=spot,
            strike=strike,
            risk_free_rate=rate,
            maturity=maturity,
            K=4,  # 4 factors
            use_approximate_iqft=False,
            validate_vs_classical=True,
            run_on_hardware=True,
            backend_name=None  # Auto-select least busy
        )
        
        total_wall_time = time.time() - start_wall_time
        
        print()
        print("="*70)
        print("HARDWARE EXECUTION RESULTS")
        print("="*70)
        print(f"✓ FB-IQFT Price:        ${result.price:.4f}")
        print(f"✓ Classical MC Price:   ${result.classical_price_baseline:.4f}")
        print(f"✓ Error:                {abs(result.price - result.classical_price_baseline) / result.classical_price_baseline * 100:.2f}%")
        print()
        print(f"✓ Circuit Depth:        {result.circuit_depth}")
        print(f"✓ Depth Reduction:      {result.depth_reduction:.1f}×")
        print(f"✓ Qubits:               {result.n_factor_qubits}")
        print(f"✓ Variance Explained:   {result.variance_explained*100:.1f}%")
        print()
        print(f"⏱  Total Wall Time:      {total_wall_time:.1f} seconds ({total_wall_time/60:.1f} minutes)")
        print()
        print("="*70)
        print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70)
        
        # Save results
        with open('fb_iqft_hardware_results.txt', 'w') as f:
            f.write(f"FB-IQFT IBM Quantum Hardware Test\n")
            f.write(f"="*70 + "\n")
            f.write(f"Timestamp: {datetime.now()}\n")
            f.write(f"\nResults:\n")
            f.write(f"  FB-IQFT Price: ${result.price:.4f}\n")
            f.write(f"  Classical MC: ${result.classical_price_baseline:.4f}\n")
            f.write(f"  Error: {abs(result.price - result.classical_price_baseline) / result.classical_price_baseline * 100:.2f}%\n")
            f.write(f"  Circuit Depth: {result.circuit_depth}\n")
            f.write(f"  Wall Time: {total_wall_time:.1f}s\n")
        
        print("✓ Results saved to fb_iqft_hardware_results.txt")
        
        return result
        
    except Exception as e:
        print()
        print(f"❌ Hardware execution failed: {e}")
        print()
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    result = test_fb_iqft_hardware()
