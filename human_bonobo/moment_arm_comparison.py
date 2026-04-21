"""
moment_arm_comparison.py — PeerJ vs Our Model Moment Arm Diagnostic
====================================================================
Computes and compares the moment arms from:
  1. PeerJ calibrated path points (Fingermodel.py's computeMA)
  2. Our climbing_finger_3d.py moment arm functions

For each posture (MinorFlex, MajorFlex, HyperExt, Hook), prints:
  - FDP moment arm at DIP, PIP, MCP
  - FDS moment arm at PIP, MCP
  - The ratio (our / PeerJ) to identify which joints are most discrepant

Usage:
    cd /Users/igorcerovsky/Documents/finger
    .venv/bin/python3 human_bonobo/moment_arm_comparison.py
"""

import sys, os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from climbing_finger_3d import Config, FingerGeometry, GripAngles, kinematics_3d

# ─── PeerJ geometry setup (from peerj_model.py) ──────────────────────────────
O2O3 = 23.63e-3   # metres
SEG_RATIOS = np.array([0.015/O2O3, 0.17, 0.22, 1.62, 0.37])
O0O1 = SEG_RATIOS[0] * O2O3
O1O2 = SEG_RATIOS[1] * O2O3
O3O4 = SEG_RATIOS[2] * O2O3
O4O5 = SEG_RATIOS[3] * O2O3
O5O6 = SEG_RATIOS[4] * O2O3

geom_path = os.path.join(os.path.dirname(__file__), 'Geometry_Middle_Cal_Hum')

# Load PeerJ path points (normalized to O2O3)
DIPPoints = np.loadtxt(os.path.join(geom_path, 'DIP_path.csv'), delimiter=',', skiprows=1)
PIPPoints = np.loadtxt(os.path.join(geom_path, 'PIP_path.csv'), delimiter=',', skiprows=1)
MCPPoints = np.loadtxt(os.path.join(geom_path, 'MCP_path.csv'), delimiter=',', skiprows=1)


def deg2rad(d):
    return d * np.pi / 180.0


def computeRotation(tx, ty, tz):
    """PeerJ rotation: Rx·Ry·Rz"""
    cx, sx = np.cos(tx), np.sin(tx)
    cy, sy = np.cos(ty), np.sin(ty)
    cz, sz = np.cos(tz), np.sin(tz)
    Rx = np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]])
    Ry = np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])
    Rz = np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]])
    return Rx @ Ry @ Rz


def computeMA(r_p, r_d, offset, axis, theta):
    """PeerJ's computeMA: moment arm from path points using generalized force method."""
    R_mat = np.linalg.inv(computeRotation(*theta))
    r_p_ind = R_mat @ r_p + offset
    r_d_ind = r_d
    r_pd_ind = r_p_ind - r_d_ind
    r_pd_norm_ind = r_pd_ind / np.linalg.norm(r_pd_ind)
    mom_ind = np.cross(r_p_ind - offset, r_pd_norm_ind)
    ma = np.dot(mom_ind, R_mat @ axis)
    return ma


def peerj_moment_arms(DIP, PIP, MCP_flex, MCP_abd=0):
    """Compute PeerJ moment arms at a given posture. Returns dict in mm."""
    # Physical path points (scale by O2O3)
    DIPProx = DIPPoints[:, 3:6] * O2O3
    DIPDist = DIPPoints[:, 0:3] * O2O3
    PIPProx = PIPPoints[:, 3:6] * O2O3
    PIPDist = PIPPoints[:, 0:3] * O2O3
    MCPProx = MCPPoints[:, 3:6] * O2O3
    MCPDist = MCPPoints[:, 0:3] * O2O3

    abd_r = deg2rad(MCP_abd)
    mcp_r = deg2rad(MCP_flex)
    pip_r = deg2rad(PIP)
    dip_r = deg2rad(DIP)

    # MCP flexion axis (rotated by abduction)
    MCP_flex_axis = computeRotation(0, abd_r, 0) @ np.array([0, 0, 1])
    MCP_theta = np.array([0, abd_r, mcp_r])

    # DIP: row 0 = TE, row 1 = FDP
    DIP_FDP = computeMA(DIPProx[1], DIPDist[1], np.array([O1O2, 0, 0]),
                        np.array([0, 0, 1]), np.array([0, 0, dip_r]))

    # PIP: row 0 = FDP, row 3 = FDS
    PIP_FDP = computeMA(PIPProx[0], PIPDist[0], np.array([O3O4, 0, 0]),
                        np.array([0, 0, 1]), np.array([0, 0, pip_r]))
    PIP_FDS = computeMA(PIPProx[3], PIPDist[3], np.array([O3O4, 0, 0]),
                        np.array([0, 0, 1]), np.array([0, 0, pip_r]))

    # MCP: row 0 = FDP, row 1 = FDS
    MCP_FDP = computeMA(MCPProx[0], MCPDist[0], np.array([O5O6, 0, 0]),
                        MCP_flex_axis, MCP_theta)
    MCP_FDS = computeMA(MCPProx[1], MCPDist[1], np.array([O5O6, 0, 0]),
                        MCP_flex_axis, MCP_theta)

    # Convert to mm for comparison with our model
    return {
        'DIP_FDP': DIP_FDP * 1000,
        'PIP_FDP': PIP_FDP * 1000,
        'PIP_FDS': PIP_FDS * 1000,
        'MCP_FDP': MCP_FDP * 1000,
        'MCP_FDS': MCP_FDS * 1000,
    }


def our_moment_arms(DIP, PIP, MCP_flex, MCP_abd=0):
    """Compute our model's moment arms at a given posture. Returns dict in mm."""
    from climbing_finger_3d import moment_arms

    grip = GripAngles('test', theta_MCP=MCP_flex, phi_MCP=MCP_abd,
                      theta_PIP=PIP, theta_DIP=DIP, emg_ratio=1.0)
    ma = moment_arms(grip)

    return {
        'DIP_FDP': ma['FDP_DIP'],
        'PIP_FDP': ma['FDP_PIP'],
        'PIP_FDS': ma['FDS_PIP'],
        'MCP_FDP': ma['FDP_MCP'],
        'MCP_FDS': ma['FDS_MCP'],
    }


def main():
    postures = {
        'MinorFlex': {'DIP': 35, 'PIP': 55, 'MCP_flex': 40},
        'MajorFlex': {'DIP': 25, 'PIP': 57, 'MCP_flex': 55},
        'HyperExt':  {'DIP': 45, 'PIP': 50, 'MCP_flex': -20},
        'Hook':      {'DIP': 50, 'PIP': 65, 'MCP_flex': 0},
    }

    ma_keys = ['DIP_FDP', 'PIP_FDP', 'PIP_FDS', 'MCP_FDP', 'MCP_FDS']

    print('=' * 90)
    print('  MOMENT ARM COMPARISON: Our Model vs PeerJ Calibrated Path Points')
    print('  Values in mm. Ratio > 1 means our MA is larger; < 1 means ours is smaller.')
    print('=' * 90)

    for posture_name, angles in postures.items():
        print(f'\n  Posture: {posture_name} (DIP={angles["DIP"]}°, PIP={angles["PIP"]}°, '
              f'MCP={angles["MCP_flex"]}°)')
        print(f'  {"Joint_Tendon":<12} {"Our(mm)":>10} {"PeerJ(mm)":>10} {"Ratio":>8} {"Note":<20}')
        print('  ' + '-' * 65)

        peerj = peerj_moment_arms(**angles)
        ours = our_moment_arms(**angles)

        for mk in ma_keys:
            p_val = peerj[mk]
            o_val = ours[mk]
            if abs(p_val) > 0.01:
                ratio = o_val / p_val
                note = ''
                if abs(ratio) < 0.5:
                    note = '← OURS MUCH SMALLER'
                elif abs(ratio) < 0.8:
                    note = '← ours smaller'
                elif abs(ratio) > 2.0:
                    note = '← OURS MUCH LARGER'
                elif abs(ratio) > 1.2:
                    note = '← ours larger'
            else:
                ratio = float('inf')
                note = '(PeerJ ≈ 0)'
            print(f'  {mk:<12} {o_val:>10.2f} {p_val:>10.2f} {ratio:>8.2f} {note:<20}')

    print('\n' + '=' * 90)
    print('  Summary:')
    print('  - If our moment arms are systematically SMALLER than PeerJ,')
    print('    our model needs MORE tendon force to balance the same external moment.')
    print('    This explains the Pred/Applied > 1.0 overestimate in Iteration 14.')
    print('  - The joints with the largest ratio deviation are the primary')
    print('    calibration targets for improving absolute force accuracy.')
    print('=' * 90)


if __name__ == '__main__':
    main()
