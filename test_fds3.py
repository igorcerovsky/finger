import numpy as np
f_mag = 100
short_l = (21.0, 22.0, 23.0)
long_l = (31.0, 28.0, 29.0)

s_short = short_l[1] / 26.0
s_long = long_l[1] / 26.0

r_fdp_short = s_short * 4.2  
r_fdp_long  = s_long * 4.2   
r_fds_short = s_short * 6.5
r_fds_long  = s_long * 6.5
r_fdp_pip_short = s_short * 2.8
r_fdp_pip_long  = s_long * 2.8

# Vigouroux says max FDP/FDS ratio ~ 1.75
# This means FDS shouldn't drop arbitrarily low (which causes the FDP to spike unrealistically without FDS sharing load).
# Conversely, FDS shouldn't spike unrealistically high. FDP and FDS share the moment at the PIP joint.
# At PIP: M_pip = FDS * r_fds + FDP * r_fdp_pip
# M_pip for a half-crimp: The PIP joint is Lm behind the DIP joint horizontally.
lm_short = short_l[1]
lm_long  = long_l[1]
theta_m_rad = np.deg2rad(-20.0) # Angle of middle phalanx relative to horizontal

for D in [15.0, 10.0, 8.0, 6.0, 4.0]:
    # Baseline DIP Moment arm
    r_pad_short = 6.0 * (short_l[1] / 22.0)
    r_pad_long  = 6.0 * (long_l[1] / 22.0)
    
    ma_dip_short = max(r_pad_short - D, 0.0) + 2.0
    ma_dip_long  = max(r_pad_long  - D, 0.0) + 2.0
    
    s_fdp = f_mag * ma_dip_short / r_fdp_short
    l_fdp = f_mag * ma_dip_long  / r_fdp_long

    # True horizontal PIP distance = DIP_dist + Lm * cos(theta_m)
    ma_pip_short = ma_dip_short + lm_short * np.cos(theta_m_rad)
    ma_pip_long  = ma_dip_long + lm_long * np.cos(theta_m_rad)
    
    # We must also account for shear reducing the PIP moment!
    # F_shear = 100 * 0.3 = 30 N pulling backwards (proximal)
    # The PIP joint is Lm * sin(theta_m) ABOVE the DIP joint vertically.
    # sin(-20) = -0.342. 
    # Wait, if theta_m is -20, the PIP is HIGHER than the DIP?
    # In a half crimp, the middle phalanx slants DOWN to the DIP. So PIP is higher. Correct.
    dy_pip_short = -lm_short * np.sin(theta_m_rad)
    dy_pip_long  = -lm_long * np.sin(theta_m_rad)
    
    f_shear = f_mag * 0.3
    
    # M_pip = F_vertical * dx - F_shear * dy
    m_pip_ext_short = (f_mag * ma_pip_short) - (f_shear * dy_pip_short)
    m_pip_ext_long  = (f_mag * ma_pip_long) - (f_shear * dy_pip_long)
    
    s_fds = max((m_pip_ext_short - s_fdp * r_fdp_pip_short) / r_fds_short, 0.0)
    l_fds = max((m_pip_ext_long - l_fdp * r_fdp_pip_long) / r_fds_long, 0.0)

    print(f"D={D} | Short: FDP={s_fdp:.1f} FDS={s_fds:.1f} Ratio={s_fdp/s_fds if s_fds>0 else 999:.2f} | Long: FDP={l_fdp:.1f} FDS={l_fds:.1f} Ratio={l_fdp/l_fds if l_fds>0 else 999:.2f}")
    
