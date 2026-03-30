"""
test_match_human_bonobo.py — PeerJ Validation Test for Backward Compatibility
=============================================================================
Ensures that the newly added 3D physical augmentations (ICR shifting, Capstan Wrap)
can cleanly disable (via `Config`) to mathematically return to the published 2D/3D
rigid skeleton model originally distributed in the PeerJ 7470 study.
"""

import numpy as np
import sys
import os

from climbing_finger_3d import (
    Config, FingerGeometry, GripAngles, solve_all_methods
)

# Reference data extracted from standard test case prior to ICR enhancements
# Rigid link configuration (MinorFlex 'low' load): 
# Total Applied Force = 4.9 N
BASELINE_EXPECTED = {
    'direct': {'F_FDP': 15.3, 'F_FDS': 9.0},
    'emg':    {'F_FDP': 13.6, 'F_FDS': 11.4},
}

def test_backward_compatibility():
    print("Running backwards-compatibility ICR test...")
    
    # 1. Disable all advanced modules
    Config.use_icr_shifting = False
    Config.use_capstan = False

    # 2. Replicate `compare_models.py` MinorFlex condition
    geom = FingerGeometry(L1=45.0 * (23.63/28.0), L2=23.63, L3=22.0 * (23.63/28.0), name='PeerJ')
    grip = GripAngles('MinorFlex', theta_MCP=40, phi_MCP=0, theta_PIP=55, theta_DIP=35, emg_ratio=1.20)
    
    total_tendon_applied = (300 * 9.81 / 1000 * 0.835) * 2
    F_ext = np.array([total_tendon_applied, 0.0, 0.0])

    # 3. Solve
    res = solve_all_methods(grip, geom, F_ext, contact=None)
    
    # 4. Asserts (Verify it falls back exactly to the baseline!)
    direct = res['direct']
    emg = res['emg']
    
    print(f"Direct -> F_FDP: {direct['F_FDP']:.1f}, F_FDS: {direct['F_FDS']:.1f}")
    print(f"EMG    -> F_FDP: {emg['F_FDP']:.1f}, F_FDS: {emg['F_FDS']:.1f}")
    
    assert np.isclose(direct['F_FDP'], BASELINE_EXPECTED['direct']['F_FDP'], atol=0.2), "FDP Direct broken"
    assert np.isclose(direct['F_FDS'], BASELINE_EXPECTED['direct']['F_FDS'], atol=0.2), "FDS Direct broken"
    assert np.isclose(emg['F_FDP'], BASELINE_EXPECTED['emg']['F_FDP'], atol=0.2), "FDP EMG broken"
    assert np.isclose(emg['F_FDS'], BASELINE_EXPECTED['emg']['F_FDS'], atol=0.2), "FDS EMG broken"
    
    print("SUCCESS: Config flags perfectly bypass ICR/Capstan and restore PeerJ exact math.")

if __name__ == '__main__':
    test_backward_compatibility()
