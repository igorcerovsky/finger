"""
sync_paper_tables.py
====================
Runs the climbing_finger_3d simulation engine and updates Tables 2 and 3
in paper/short_vs_long_finger_advantage.md to guarantee 100% numerical reproducibility.
"""

import os
import sys
import numpy as np

# Ensure root directory is on PATH
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, ROOT_DIR)
import climbing_finger_3d as c3d

PAPER_MD = os.path.join(ROOT_DIR, 'paper/short_vs_long_finger_advantage.md')

def generate_table2():
    geom_std = c3d.FingerGeometry(c3d.Config.PP_mm, c3d.Config.MP_mm, c3d.Config.DP_mm, 'Standard')
    ct10 = c3d.ContactGeometry(d_hold=10.0, r_edge=2.0, t_DP=9.0, beta_wall=0.0)
    F_ext_171 = np.array([171.7, 0.0, 0.0])

    rows = []
    for gk, g_title in [('crimp', '**Full Crimp**'), ('half_crimp', '**Half-Crimp**'), ('open_hand', '**Open Hand**')]:
        g = c3d.GRIPS[gk]
        res = c3d.solve_all_methods(g, geom_std, F_ext_171, contact=ct10)
        for idx, (m, m_label) in enumerate([('direct', 'Direct (3×3)'), ('emg', 'EMG-Constrained'), ('lu_min', 'LU-Minimizing')]):
            r = res[m]
            jr = c3d.joint_reactions_3d(r, F_ext_171, geom_std)
            p = jr['pulley']
            col_grip = g_title if idx == 0 else ""
            ratio_str = f"{r['ratio']:.2f}" if r['ratio'] < 99 else "—"
            rows.append(
                f"| {col_grip:<14} | {m_label:<15} | {r['F_FDP']:7.1f} | {r['F_FDS']:7.1f} | {r['F_LU']:6.1f} | "
                f"{r['F_EDC']:7.1f} | {r['F_RI']:6.1f} | {r['F_total']:9.1f} | {ratio_str:^13} | {p['P_A2_MPa']:9.2f} |"
            )
            
    header = (
        "| Grip Type | Solver Method | $F_{FDP}$ (N) | $F_{FDS}$ (N) | $F_{LU}$ (N) | $F_{EDC}$ (N) | $F_{RI}$ (N) | Total Force (N) | FDP/FDS Ratio | A2 Stress (MPa) |\n"
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n"
    )
    return header + "\n".join(rows)

def generate_table3():
    geom_std = c3d.FingerGeometry(c3d.Config.PP_mm, c3d.Config.MP_mm, c3d.Config.DP_mm, 'Standard')
    geom_short = geom_std.scaled(c3d.Config.scale_short, 'Short')
    geom_long = geom_std.scaled(c3d.Config.scale_long, 'Long')
    
    # Ensure isometric scaling is active (realistic anatomy)
    c3d.Config.scale_moment_arms_with_geometry = True

    lines = []
    header = (
        "| Loading Condition | Phenotype | Total Length (mm) | $L_{DP}$ (mm) | Edge Depth $d$ (mm) | $\\rho = d/L_{DP}$ | Contact Mode | Crimp A2 (N) | Half-Crimp A2 (N) |\n"
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n"
    )

    # 1. Fixed 100 N Middle-Finger Load across hold depths
    edges = [
        ("Micro-Edge (6 mm)", 6.0),
        ("Medium Edge (10 mm)", 10.0),
        ("Testing Ledge (20 mm)", 20.0)
    ]
    
    F_100 = np.array([100.0, 0.0, 0.0])

    for e_label, d in edges:
        ct = c3d.ContactGeometry(d_hold=d, r_edge=2.0, t_DP=9.0, beta_wall=0.0)
        for p_idx, (p_name, geom, tot_len, l_dp) in enumerate([
            ("Short (−15%)", geom_short, "80.75", f"{geom_short.L3:.1f}"),
            ("Standard", geom_std, "95.00", f"{geom_std.L3:.1f}"),
            ("Long (+15%)", geom_long, "109.25", f"{geom_long.L3:.1f}")
        ]):
            r_cr = c3d.solve_all_methods(c3d.GRIPS['crimp'], geom, F_100, contact=ct)['emg']
            f_cr = c3d.joint_reactions_3d(r_cr, F_100, geom)['pulley']['F_A2_mag']

            r_hc = c3d.solve_all_methods(c3d.GRIPS['half_crimp'], geom, F_100, contact=ct)['emg']
            f_hc = c3d.joint_reactions_3d(r_hc, F_100, geom)['pulley']['F_A2_mag']

            rho = d / geom.L3
            mode = "Dual (MP+DP)" if d >= geom.L3 else "Single (DP)"
            col_cond = f"**Fixed 100 N Hang**<br>({e_label})" if p_idx == 0 else ""
            lines.append(
                f"| {col_cond:<26} | {p_name:<13} | {tot_len:^17} | {l_dp:^11} | {d:^19.1f} | {rho:^16.2f} | {mode:<12} | {f_cr:12.1f} | {f_hc:17.1f} |"
            )

    # 2. Allometric Bodyweight Hang (mass scaled as L^2, BMI constant)
    # Short = 50.6 kg (72.3 N), Std = 70.0 kg (100.0 N), Long = 92.5 kg (132.3 N)
    bw_forces = [
        ("Short (−15%, 50.6 kg)", geom_short, "80.75", f"{geom_short.L3:.1f}", 72.3),
        ("Standard (70.0 kg)", geom_std, "95.00", f"{geom_std.L3:.1f}", 100.0),
        ("Long (+15%, 92.5 kg)", geom_long, "109.25", f"{geom_long.L3:.1f}", 132.3)
    ]
    ct_10 = c3d.ContactGeometry(d_hold=10.0, r_edge=2.0, t_DP=9.0, beta_wall=0.0)
    for p_idx, (p_name, geom, tot_len, l_dp, bw_F) in enumerate(bw_forces):
        F_bw = np.array([bw_F, 0.0, 0.0])
        r_cr = c3d.solve_all_methods(c3d.GRIPS['crimp'], geom, F_bw, contact=ct_10)['emg']
        f_cr = c3d.joint_reactions_3d(r_cr, F_bw, geom)['pulley']['F_A2_mag']

        r_hc = c3d.solve_all_methods(c3d.GRIPS['half_crimp'], geom, F_bw, contact=ct_10)['emg']
        f_hc = c3d.joint_reactions_3d(r_hc, F_bw, geom)['pulley']['F_A2_mag']

        rho = 10.0 / geom.L3
        mode = "Single (DP)"
        col_cond = "**Bodyweight Hang**<br>(10 mm Edge, Stature $L^2$)" if p_idx == 0 else ""
        lines.append(
            f"| {col_cond:<26} | {p_name:<13} | {tot_len:^17} | {l_dp:^11} | {10.0:^19.1f} | {rho:^16.2f} | {mode:<12} | {f_cr:12.1f} | {f_hc:17.1f} |"
        )

    return header + "\n".join(lines)

def main():
    print("Generating Table 2...")
    t2_md = generate_table2().strip()
    print("Generating Table 3...")
    t3_md = generate_table3().strip()

    print("\nTable 2 Preview:\n", t2_md)
    print("\nTable 3 Preview:\n", t3_md)

    import re
    with open(PAPER_MD, 'r') as f:
        text = f.read()

    # Sync Table 2
    text = re.sub(
        r'(### Table 2:[^\n]+\n\n)\| Grip Type \|.*?(?=\n\n\*Note: Values computed)',
        lambda m: m.group(1) + t2_md,
        text,
        flags=re.DOTALL
    )

    # Update Table 3 heading and content
    t3_heading = "### Table 3: Scientific Evaluation of Phenotypic Scaling Across Hold Geometries (Middle Finger, Sub-Linear Allometric Scaling $ma \\propto L^{0.50}$ & Cantilever Mechanics)"
    text = re.sub(
        r'### Table 3:[^\n]+\n\n\| Loading Condition \|.*?(?=\n\n\*Note: In the Fixed 100 N)',
        lambda m: t3_heading + '\n\n' + t3_md,
        text,
        flags=re.DOTALL
    )

    with open(PAPER_MD, 'w') as f:
        f.write(text)
    print(f"\nSuccessfully synced Tables 2 and 3 into {PAPER_MD}!")

if __name__ == '__main__':
    main()
