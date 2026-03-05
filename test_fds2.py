import numpy as np
f_mag = 100
short_l = (21.0, 22.0, 23.0)
long_l = (31.0, 28.0, 29.0)

s_short = short_l[1] / 26.0
s_long = long_l[1] / 26.0
r_fdp_short = s_short * 4.2  
r_fdp_long  = s_long * 4.2   

# The previous script did this:
# M_pip = F * dist(PIP, contact_x)
# fds = (M_pip - M_pass_pip - fdp * r_fdp_pip) / r_fds_pip
# but wait! Contact point X is NOT dist(PIP, contact_x) = ma_dip + Lm 
# The actual moment at PIP depends on the true X-distance from PIP to the contact point. 
# Previously:
# F_ext = contact_force_vector(pose, f_mag) -> this has a shear component!
# But in Mechanism C:
# F_ext = [0, 100], but wait, it has shear? Yes, F_shear = magnitude_N * MU_EFF * abs(np.cos(theta_d)) * (-u_d)
# If theta_d = 0, F_shear is 100 * 0.3 * (-1, 0) = [-30, 0].
# So total force is F_ext = [-30, 100]. This shear REDUCES the PIP moment!
# Let's use the actual solve_static_equilibrium!

from finger_biomechanics_model import _build_grips, solve_static_equilibrium, SimConfig, get_radius, build_tendon_geometry, posture_from_joint_targets, contact_force_vector, tendon_excursion_moment_arms, cross2

# Wait, we can just use the actual geometry builder
def _forces(lengths, pip_deg, dip_deg, d_tip_mm, d_wall_penalty_mm):
    # posture
    pose = posture_from_joint_targets(lengths, pip_deg, dip_deg, 0.0, 0.0, "half_crimp")
    geo = build_tendon_geometry(pose)
    
    # contact point
    dip_pt = geo["DIP"] * 1e-3
    tip_pt = geo["TIP"] * 1e-3
    Ld_mm = np.linalg.norm(tip_pt - dip_pt) * 1000
    contact_mm = max(Ld_mm - d_tip_mm, 0.0)
    
    # Apply wall penalty by shifting the contact point backwards? No, you shift the *bone* forwards
    # Or shift the contact point backwards relative to the bone by d_wall_penalty
    # Because the wall physically forces the contact point further from the joint
    contact_mm = contact_mm + d_wall_penalty_mm
    
    contact = dip_pt + (contact_mm * 1e-3) * geo["uD"]
    
    F = contact_force_vector(pose, f_mag)
    arms = tendon_excursion_moment_arms(pose)
    
    M_dip = abs(cross2(contact - dip_pt, F))
    M_pip = abs(cross2(contact - geo["PIP"] * 1e-3, F))
    
    fdp = M_dip / max(arms["r_fdp_dip"] * 1e-3, 1e-6)
    fds = max((M_pip - fdp * arms["r_fdp_pip"] * 1e-3) / max(arms["r_fds_pip"] * 1e-3, 1e-6), 0.0)
    
    return fdp, fds

d_tip = 12.0
r_pad_short = 6.0 * (short_l[1] / 22.0)
r_pad_long  = 6.0 * (long_l[1] / 22.0)

for D in [15.0, 10.0, 8.0, 6.0, 4.0]:
    pen_short = max(r_pad_short - D, 0.0)
    pen_long  = max(r_pad_long  - D, 0.0)
    
    # Short finger uses 75/-20 pip/dip
    fdp_s, fds_s = _forces(short_l, 75.0, -20.0, d_tip, pen_short)
    
    # Long finger typically uses smaller pip angle to reach height (e.g. 39 deg)
    fdp_l, fds_l = _forces(long_l, 39.0, -10.0, d_tip, pen_long)
    
    print(f"D={D}: S_FDP={fdp_s:.1f} S_FDS={fds_s:.1f} ratio={fdp_s/fds_s:.2f} | L_FDP={fdp_l:.1f} L_FDS={fds_l:.1f} ratio={fdp_l/fds_l:.2f}")

