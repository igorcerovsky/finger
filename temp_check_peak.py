import numpy as np
import sys
import os

# Put directory in path so we can import climbing_finger_3d
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import climbing_finger_3d as cf

ct_base = cf.ContactGeometry(d_hold=cf.Config.d_hold_mm, r_edge=cf.Config.r_edge_mm, t_DP=cf.Config.t_DP_mm, beta_wall=0.0)

geoms = [
    cf.FingerGeometry(cf.Config.PP_mm, cf.Config.MP_mm, cf.Config.DP_mm, "Standard").scaled(cf.Config.scale_short, "Short"),
    cf.FingerGeometry(cf.Config.PP_mm, cf.Config.MP_mm, cf.Config.DP_mm, "Standard"),
    cf.FingerGeometry(cf.Config.PP_mm, cf.Config.MP_mm, cf.Config.DP_mm, "Standard").scaled(cf.Config.scale_long, "Long")
]

grip = cf.GRIPS["crimp"]
d_range = np.linspace(2.0, 45.0, 50)
F_tip = 171.7
F_ext = np.array([F_tip, 0.0, cf.Config.F_lateral_N])

fdp_s, fds_s = [], []
fdp_l, fds_l = [], []

for d in d_range:
    ct = cf.ContactGeometry(d_hold=d, r_edge=cf.Config.r_edge_mm, t_DP=cf.Config.t_DP_mm, mu=cf.Config.mu_friction, beta_wall=0.0)
    
    r_s = cf.solve_all_methods(grip, geoms[0], F_ext, contact=ct)['emg']
    fdp_s.append(r_s['F_FDP'])
    fds_s.append(r_s['F_FDS'])
    
    r_l = cf.solve_all_methods(grip, geoms[2], F_ext, contact=ct)['emg']
    fdp_l.append(r_l['F_FDP'])
    fds_l.append(r_l['F_FDS'])

fdp_s, fdp_l = np.array(fdp_s), np.array(fdp_l)

pct_fdp = np.where(fdp_s > 1, (fdp_l - fdp_s) / fdp_s * 100, 0)
max_idx = np.argmax(pct_fdp)
print(f"Peak depth: {d_range[max_idx]:.2f} mm")
print(f"Peak % inc: {pct_fdp[max_idx]:.2f} %")

for i, d in enumerate(d_range):
    if d > 15 and d < 45:
        print(f"d={d:.1f}: fdp_s={fdp_s[i]:.1f}, fdp_l={fdp_l[i]:.1f}, pct={pct_fdp[i]:.1f}%")
