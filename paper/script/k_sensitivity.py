"""k-exponent sensitivity of the long-finger penalty (crimp/half-crimp A2, total force)."""
import sys, numpy as np
sys.path.insert(0, __import__('os').path.abspath(__import__('os').path.join(__import__('os').path.dirname(__file__), '../..')))
import climbing_finger_3d as c3d

std = c3d.FingerGeometry(c3d.Config.PP_mm, c3d.Config.MP_mm, c3d.Config.DP_mm, 'Standard')
G = {'S': std.scaled(c3d.Config.scale_short, 'Short'), 'M': std, 'L': std.scaled(c3d.Config.scale_long, 'Long')}

def run(grip, geom, F, d):
    ct = c3d.ContactGeometry(d_hold=d, r_edge=2.0, t_DP=9.0, beta_wall=0.0)
    Fv = np.array([F, 0.0, 0.0])
    r = c3d.solve_all_methods(c3d.GRIPS[grip], geom, Fv, contact=ct)['emg']
    return c3d.joint_reactions_3d(r, Fv, geom)['pulley']['F_A2_mag'], r['F_total']

for k in (1.0, 0.5, 0.0):
    c3d.Config.allometric_ma_exponent = k
    print(f"\n=== k = {k} ===")
    for grip in ('crimp', 'half_crimp'):
        for d in (6.0, 10.0, 20.0):
            v = {p: run(grip, g, 100.0, d) for p, g in G.items()}
            print(f"{grip:10s} d={d:4.0f} 100N  A2 S/M/L = {v['S'][0]:6.1f} {v['M'][0]:6.1f} {v['L'][0]:6.1f}"
                  f"  L/S={100*(v['L'][0]/v['S'][0]-1):+5.1f}%  L/M={100*(v['L'][0]/v['M'][0]-1):+5.1f}%"
                  f" | Ftot L/S={100*(v['L'][1]/v['S'][1]-1):+5.1f}%")
    # bodyweight hang, m ~ L^2, 10 mm
    bw = {'S': 72.3, 'M': 100.0, 'L': 132.3}
    v = {p: run('crimp', G[p], bw[p], 10.0) for p in G}
    print(f"crimp BW 10mm A2 S/M/L = {v['S'][0]:.1f} {v['M'][0]:.1f} {v['L'][0]:.1f}; "
          f"A2/Ftip S/M/L = {v['S'][0]/72.3:.3f} {v['M'][0]/100:.3f} {v['L'][0]/132.3:.3f}")

c3d.Config.allometric_ma_exponent = 0.5
for F in (100.0, 171.7):
    a = {g: run(g, std, F, 10.0)[0] for g in ('crimp', 'half_crimp', 'open_hand')}
    print(f"\nStd 10mm F={F}: A2 crimp {a['crimp']:.1f}, half {a['half_crimp']:.1f}, open {a['open_hand']:.1f}, crimp/open={a['crimp']/a['open_hand']:.2f}")
