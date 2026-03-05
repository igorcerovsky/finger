import numpy as np
f_mag = 100
short_l = (21.0, 22.0, 23.0)
s_short = short_l[1] / 26.0
r_fdp_short = s_short * 4.2  
r_fds_short = s_short * 6.5
r_fdp_pip_short = s_short * 2.8

# Try 8mm hold
r_pad_short = 6.0 * (short_l[1] / 22.0)
ma_dip_short = max(r_pad_short - 8.0, 0.0) + 2.0
s_fdp = f_mag * ma_dip_short / r_fdp_short
print(f"s_fdp: {s_fdp}")

# In half crimp, angle is 75... horizontal dist from DIP to PIP is Lm * cos(-20)?
# Actually, the angle of the middle phalanx relative to horizontal depends on the wall.
# In a half crimp, the distal phalanx is near 0 abs deg -> horizontal.
# The middle phalanx is at ~-20 relative to distal, so it's not horizontal!
# Wait, theta_d = 0. theta_m = dip_flex + theta_d = -20 deg.
# So horizontal distance from DIP to PIP is Lm * cos(-20) = Lm * 0.94
lm_short = short_l[1]
ma_pip_short = ma_dip_short + lm_short * np.cos(np.deg2rad(-20.0))
print(f"ma_pip_short: {ma_pip_short}")

s_fds = max((f_mag * ma_pip_short - s_fdp * r_fdp_pip_short) / r_fds_short, 0.0)
print(f"s_fds raw: {s_fds}")
print(f"FDP/FDS ratio: {s_fdp/s_fds if s_fds > 0 else 'inf'}")

# M_pip_ext = f_mag * ma_pip_short (100 * 22.6 = 2260 Nmm)
# s_fdp * r_fdp_pip_short = 59.5 * 2.37 = 141 Nmm
# M_fds = 2260 - 141 = 2119 Nmm
# s_fds = 2119 / 5.5 = 385 N
# FDP/FDS = 59.5 / 385 = 0.15!!
