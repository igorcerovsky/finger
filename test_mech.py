from finger_biomechanics_model import SimConfig, AthleteCase
def get_radius(lengths): return 6.0 * (lengths[1] / 22.0)
short_l = (21.0, 22.0, 23.0)
long_l = (31.0, 28.0, 29.0)
f_mag = 100
r_pad_short = get_radius(short_l)
r_pad_long  = get_radius(long_l)
s_short = short_l[1] / 26.0
s_long = long_l[1] / 26.0
r_fdp_short = s_short * 4.2  
r_fdp_long  = s_long * 4.2   
for D in [15.0, 10.0, 8.0, 6.0, 4.0]:
    baseline = 3.0 
    ma_short = max(r_pad_short - D, 0.0) + baseline
    ma_long  = max(r_pad_long  - D, 0.0) + baseline
    fdp_short = f_mag * ma_short / r_fdp_short
    fdp_long  = f_mag * ma_long  / r_fdp_long
    print(f"{D}: {ma_short} {ma_long} {fdp_short:.1f} {fdp_long:.1f} {fdp_long/fdp_short:.2f}x")
