"""
compare_models.py — PeerJ 7470 Validation vs Our 3D Climbing Model
========================================================================
Compares our climbing_finger_3d.py predictions against the cadaver direct-force
measurements from the PeerJ 7470 paper (Synek et al. 2019).
"""

import sys, os
import numpy as np
import csv

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from climbing_finger_3d import (
    Config, FingerGeometry, GripAngles, solve_all_methods
)

PEERJ_POSTURES = {
    'MinorFlex': {'DIP': 35, 'PIP': 55, 'MCP_flex': 40,  'MCP_abd': 0},
    'MajorFlex': {'DIP': 25, 'PIP': 57, 'MCP_flex': 55,  'MCP_abd': 0},
    'HyperExt':  {'DIP': 45, 'PIP': 50, 'MCP_flex': -20, 'MCP_abd': 0},
    'Hook':      {'DIP': 50, 'PIP': 65, 'MCP_flex': 0,   'MCP_abd': 0},
}

PULLEY_EFF = 0.835

VIGOUROUX_RATIOS = {
    'MinorFlex': 1.20,
    'MajorFlex': 1.20,
    'HyperExt':  1.75,
    'Hook':      1.20,
}

O2O3 = 23.63e-3
SEG_RATIOS = np.array([0.015/O2O3, 0.17, 0.22, 1.62, 0.37])
O0O1 = SEG_RATIOS[0] * O2O3
O1O2 = SEG_RATIOS[1] * O2O3
O3O4 = SEG_RATIOS[2] * O2O3
O4O5 = SEG_RATIOS[3] * O2O3
O5O6 = SEG_RATIOS[4] * O2O3

PP_mm = (O4O5 + O5O6) * 1000
MP_mm = (O2O3 + O3O4) * 1000
DP_mm = (O0O1 + O1O2) * 1000

PEERJ_GEOM = FingerGeometry(L1=PP_mm, L2=MP_mm, L3=DP_mm, name='PeerJ-exact')


def load_experimental_fingertip_forces():
    base = os.path.join(os.path.dirname(__file__), 'Experiments', 'Fingertip_forces')
    specimens = ['H01', 'H02', 'H03']

    all_data = {}
    for spec in specimens:
        fname = os.path.join(base, f'{spec}_fingertip_combined.csv')
        if not os.path.exists(fname):
            print(f'  WARNING: {fname} not found, skipping')
            continue
        with open(fname) as f:
            reader = csv.DictReader(f)
            for row in reader:
                posture = row['Posture'].strip()
                load_g = int(row['FDP'])
                ftip = np.array([float(row['Ftip_PD']),
                                 float(row['Ftip_DV']),
                                 float(row['Ftip_RU'])])
                key = (posture, load_g)
                if key not in all_data:
                    all_data[key] = []
                all_data[key].append(ftip)

    mean_data = {}
    for key, vecs in all_data.items():
        arr = np.array(vecs)
        mean_data[key] = {
            'mean': arr.mean(axis=0),
            'std': arr.std(axis=0),
            'n': len(vecs),
        }
    return mean_data


def run_comparison():
    print('=' * 95)
    print('  VALIDATION: Our 3D Model vs PeerJ 7470 Cadaver Measurements')
    print('  Using ACTUAL cadaver force plate data as F_ext (mean of 3 specimens)')
    print(f'  Geometry: PP={PP_mm:.1f}mm  MP={MP_mm:.1f}mm  DP={DP_mm:.1f}mm')
    print('=' * 95)

    exp_data = load_experimental_fingertip_forces()
    if not exp_data:
        print('  ERROR: No experimental data found!')
        return

    posture_order = ['MinorFlex', 'MajorFlex', 'HyperExt', 'Hook']
    load_grams = [300, 950]
    results = []

    for posture_name in posture_order:
        angles = PEERJ_POSTURES[posture_name]
        for gram in load_grams:
            key = (posture_name, gram)
            if key not in exp_data:
                continue

            exp = exp_data[key]
            F_exp_mean = exp['mean']
            F_exp_mag = np.linalg.norm(F_exp_mean)
            F_exp_ang = np.degrees(np.arctan2(F_exp_mean[1], F_exp_mean[0]))

            F_tendon_per = gram * 9.81 / 1000.0 * PULLEY_EFF
            total_tendon = 2 * F_tendon_per  # Applied FDP + FDS

            F_ext_pj = -F_exp_mean
            F_ext = np.array([F_ext_pj[0], -F_ext_pj[1], -F_ext_pj[2]])

            grip = GripAngles(
                name=posture_name,
                theta_MCP=angles['MCP_flex'],
                phi_MCP=angles['MCP_abd'],
                theta_PIP=angles['PIP'],
                theta_DIP=angles['DIP'],
                emg_ratio=VIGOUROUX_RATIOS[posture_name],
            )

            orig_wrist = Config.use_wrist_coupling
            Config.use_wrist_coupling = False
            try:
                r_all = solve_all_methods(grip, PEERJ_GEOM, F_ext, contact=None)
            except Exception as e:
                print(f'  ERROR [{posture_name}/{gram}g]: {e}')
                Config.use_wrist_coupling = orig_wrist
                continue
            Config.use_wrist_coupling = orig_wrist

            for method in ('emg', 'lu_min', 'direct'):
                r = r_all[method]
                pred_ratio = r['ratio'] if not np.isinf(r['ratio']) else 999

                pred_flexor = r['F_FDP'] + r['F_FDS']
                force_ratio = pred_flexor / total_tendon if total_tendon > 0 else 0

                results.append({
                    'posture': posture_name,
                    'gram': gram,
                    'method': method,
                    'F_exp_mag': F_exp_mag,
                    'F_exp_ang': F_exp_ang,
                    'applied_tendon': total_tendon,
                    'pred_FDP': r['F_FDP'],
                    'pred_FDS': r['F_FDS'],
                    'pred_RI': r.get('F_RI', 0.0),
                    'pred_UI': r.get('F_UI', 0.0),
                    'pred_total': pred_flexor,
                    'pred_all_muscles': r['F_total'],
                    'pred_ratio': pred_ratio,
                    'force_ratio': force_ratio,
                })

    print(f"\n  SECTION 1: Experimental Fingertip Reaction Forces (mean ± std, n=3)")
    print(f"  {'Posture':<12} {'Load':<6} {'|F_exp|(N)':<11} {'Dir(°)':<9} "
          f"{'Applied(N)':<11} {'F_exp/F_tendon':<14}")
    print('  ' + '-' * 70)
    seen = set()
    for r in results:
        key = (r['posture'], r['gram'])
        if key in seen:
            continue
        seen.add(key)
        ratio_str = f"{r['F_exp_mag'] / r['applied_tendon']:.3f}"
        print(f"  {r['posture']:<12} {r['gram']:<6} {r['F_exp_mag']:<11.3f} "
              f"{r['F_exp_ang']:<9.1f} {r['applied_tendon']:<11.1f} {ratio_str:<14}")

    print(f"\n  SECTION 2: Model Predictions (F_ext = measured fingertip reaction)")
    print(f"  {'Posture':<12} {'Load':<6} {'Method':<11} "
          f"{'FDP':>6} {'FDS':>6} {'RI':>5} {'UI':>5} {'FlexSum':>8} {'Ratio':>7} {'Pred/App':>8}")
    print('  ' + '-' * 86)
    last_key = None
    for r in results:
        key = (r['posture'], r['gram'])
        if key != last_key and last_key is not None:
            print()
        last_key = key
        print(f"  {r['posture']:<12} {r['gram']:<6} {r['method']:<11} "
              f"{r['pred_FDP']:>6.1f} {r['pred_FDS']:>6.1f} "
              f"{r['pred_RI']:>5.1f} {r['pred_UI']:>5.1f} "
              f"{r['pred_total']:>8.1f} {r['pred_ratio']:>7.2f} "
              f"{r['force_ratio']:>8.2f}")

    print('\n' + '=' * 95)
    print('  SECTION 3: Validation Summary (EMG-constrained Flexor Pred/App Ratio)')
    emg_ratios = [r['force_ratio'] for r in results if r['method'] == 'emg']
    for posture in posture_order:
        posture_ratios = [r['force_ratio'] for r in results
                         if r['posture'] == posture and r['method'] == 'emg']
        if posture_ratios:
            print(f"    {posture:<12}: {np.mean(posture_ratios):.3f}")
    if emg_ratios:
        print(f"    {'Overall':<12}: {np.mean(emg_ratios):.3f} ± {np.std(emg_ratios):.3f}")
    print('=' * 95)


if __name__ == '__main__':
    run_comparison()
