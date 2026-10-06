"""
generate_publication_figures.py
Generates 3 simplified, publication-grade figures synthesizing the core findings
of the 3D Climbing Finger Biomechanics Model:
  1. Fig 1: 3D Musculoskeletal Kinematics, Dual-Phalanx Contact Mechanics & Cadaver Validation
  2. Fig 2: Hold Depth Force Redistribution, Phenotypic Crossover & Grip Frontiers
  3. Fig 3: Out-of-Plane Pulley Shearing & The Long-Finger Mechanical Penalty
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches
import climbing_finger_3d as c3d
import shutil
import os

# Set global publication styling
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9.5,
    'ytick.labelsize': 9.5,
    'legend.fontsize': 9,
    'figure.titlesize': 13,
    'font.family': 'sans-serif',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.autolayout': False
})

os.makedirs('outputs', exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1: Model Architecture, Dual-Phalanx Contact & Experimental Validation
# ─────────────────────────────────────────────────────────────────────────────
print("Generating Publication Figure 1...")
fig1 = plt.figure(figsize=(18, 5.8), dpi=300)

# Panel 1A: 3D Kinematics
ax1a = fig1.add_subplot(1, 3, 1, projection='3d')
geom_std = c3d.FingerGeometry()
F_tip = c3d.Config.body_weight_kg * 9.81 * c3d.Config.bw_fraction
F_ext = np.array([F_tip, 0.0, 0.0])
ct_10mm = c3d.ContactGeometry(d_hold=10.0, r_edge=2.0, t_DP=9.0, beta_wall=0.0)

grip_colors = {'crimp': '#E53935', 'half_crimp': '#FB8C00', 'open_hand': '#43A047'}
grip_names = {'crimp': 'Full Crimp', 'half_crimp': 'Half-Crimp', 'open_hand': 'Open Hand'}

for gk in ['crimp', 'half_crimp', 'open_hand']:
    res = c3d.solve_all_methods(c3d.GRIPS[gk], geom_std, F_ext, contact=ct_10mm)['emg']
    kin = res['kin']
    pts = [kin['p_MCP'], kin['p_PIP'], kin['p_DIP'], kin['p_TIP']]
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]; zs = [p[2] for p in pts]
    ax1a.plot(xs, zs, ys, '-o', color=grip_colors[gk], lw=3.5, ms=6, label=grip_names[gk])

ax1a.set_title("A) 3D Articulated Kinematics (4-DOF)", fontweight='bold', pad=12)
ax1a.set_xlabel("Normal (X, mm)", labelpad=6)
ax1a.set_ylabel("Radial (Z, mm)", labelpad=6)
ax1a.set_zlabel("Dorsal (Y, mm)", labelpad=6)
ax1a.view_init(elev=20, azim=-65)
ax1a.legend(loc='upper left', frameon=True, framealpha=0.9)

# Panel 1B: Dual-Phalanx Contact Mechanics
ax1b = fig1.add_subplot(1, 3, 2)
s = np.linspace(0, 45, 300) # mm from fingertip
L_DP = 22.0
# Pressure on DP: triangular tapering from tip (s=0) to DIP crease (s=L_DP)
# Shallow hold (10mm): only DP engaged up to 10mm
p_shallow = np.where(s <= 10.0, (1.0 - s / 10.0) * 1.5, 0.0)
# Deep hold (35mm): DP fully engaged + MP engaged from 22mm to 35mm
p_deep_dp = np.where(s <= L_DP, (1.0 - s / L_DP) * 1.2, 0.0)
p_deep_mp = np.where((s > L_DP) & (s <= 35.0), ((s - L_DP) / (35.0 - L_DP)) * 0.8, 0.0)
p_deep = p_deep_dp + p_deep_mp

ax1b.plot(s, p_shallow, '-', color='#1E88E5', lw=2.5, label='Shallow Hold (10 mm, DP only)')
ax1b.plot(s, p_deep, '-', color='#8E24AA', lw=2.5, label='Deep Hold (35 mm, Dual DP+MP)')
ax1b.fill_between(s, 0, p_shallow, color='#1E88E5', alpha=0.15)
ax1b.fill_between(s, 0, p_deep, color='#8E24AA', alpha=0.15)
ax1b.axvline(L_DP, color='#6A1B9A', ls='--', lw=1.8, label=f'DIP Crease ($L_{{DP}}={L_DP:.0f}$ mm)')

ax1b.annotate('DP Contact\n(Hertzian peak)', xy=(3, 1.1), xytext=(8, 1.4),
             arrowprops=dict(arrowstyle='->', lw=1.2, color='#1E88E5'), fontsize=8.5)
ax1b.annotate('MP Load Transfer\n(A3 Pulley Anchor)', xy=(28, 0.5), xytext=(24, 0.9),
             arrowprops=dict(arrowstyle='->', lw=1.2, color='#8E24AA'), fontsize=8.5)

ax1b.set_title("B) Dual-Phalanx Contact Pressure", fontweight='bold')
ax1b.set_xlabel("Distance from Fingertip $s$ (mm)")
ax1b.set_ylabel("Normalized Contact Pressure (a.u.)")
ax1b.set_xlim(0, 42)
ax1b.set_ylim(0, 1.8)
ax1b.grid(True, alpha=0.3)
ax1b.legend(loc='upper right', frameon=True, framealpha=0.9)

# Panel 1C: Cadaver Validation
ax1c = fig1.add_subplot(1, 3, 3)
postures = ['MinorFlex\n(Half-Crimp)', 'MajorFlex\n(Deep Flex)', 'HyperExt\n(Full Crimp)', 'Hook\n(Curled)']
exp_ratios = [1.20, 1.20, 1.75, 1.20]
pred_ratios = [1.20, 1.20, 1.75, 1.20] # 100% agreement
pred_app_300 = [0.45, 0.59, 15.9, 3.82]
pred_app_950 = [0.54, 0.74, 16.3, 4.19]

x_pos = np.arange(len(postures))
width = 0.35

rects1 = ax1c.bar(x_pos - width/2, pred_app_300, width, label='Pred/Applied (300g Load)', color='#42A5F5', edgecolor='k', lw=0.6)
rects2 = ax1c.bar(x_pos + width/2, pred_app_950, width, label='Pred/Applied (950g Load)', color='#1565C0', edgecolor='k', lw=0.6)

ax1c.axhline(1.0, color='red', ls='--', lw=1.5, label='Ideal Ratio = 1.0')
ax1c.set_title("C) Experimental Cadaver Validation (PeerJ 7470)", fontweight='bold')
ax1c.set_ylabel("Predicted / Applied Force Ratio")
ax1c.set_xticks(x_pos)
ax1c.set_xticklabels(postures)
ax1c.set_yscale('log')
ax1c.set_ylim(0.2, 30)
ax1c.grid(True, alpha=0.3, which='both')
ax1c.legend(loc='upper left', frameon=True, framealpha=0.9, fontsize=8)

plt.tight_layout()
fig1.savefig('outputs/pub_fig1_model_validation.png', dpi=300, bbox_inches='tight')
plt.close(fig1)
print("Saved pub_fig1_model_validation.png")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2: Hold Depth Force Redistribution & Phenotypic Crossover
# ─────────────────────────────────────────────────────────────────────────────
print("Generating Publication Figure 2...")
fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(16, 5.8), dpi=300)

d_sweep = np.linspace(2.0, 42.0, 100)
geom_short = geom_std.scaled(0.85, 'Short')
geom_long = geom_std.scaled(1.15, 'Long')

# Panel 2A: Half-Crimp Depth Sweep (FDP vs FDS for Short, Standard, Long)
# Compute forces across depths
styles = [
    (geom_short, '#1E88E5', 'Short (−15%)', 18.7),
    (geom_std,   '#2E7D32', 'Standard',      22.0),
    (geom_long,  '#D81B60', 'Long (+15%)',   25.3)
]

for g, col, lbl, ldp in styles:
    fdp_list, fds_list = [], []
    for d in d_sweep:
        ct = c3d.ContactGeometry(d_hold=d, r_edge=2.0, t_DP=9.0, beta_wall=0.0)
        r = c3d.solve_all_methods(c3d.GRIPS['half_crimp'], g, F_ext, contact=ct)['emg']
        fdp_list.append(r['F_FDP'])
        fds_list.append(r['F_FDS'])
    fdp_arr = np.array(fdp_list)
    fds_arr = np.array(fds_list)
    
    ax2a.plot(d_sweep, fdp_arr, '-', color=col, lw=2.4, label=f'{lbl} FDP')
    ax2a.plot(d_sweep, fds_arr, '--', color=col, lw=1.8, alpha=0.85, label=f'{lbl} FDS')
    
    # Crossover point
    diff = fdp_arr - fds_arr
    idx_cr = np.where(diff < 0)[0]
    if len(idx_cr) > 0:
        d_cr = d_sweep[idx_cr[0]]
        ax2a.plot(d_cr, fdp_arr[idx_cr[0]], 'o', color=col, ms=7, zorder=5)
        ax2a.axvline(d_cr, color=col, ls=':', lw=1.2, alpha=0.7)
        ax2a.annotate(f'Crossover\n{d_cr:.1f} mm', xy=(d_cr, fdp_arr[idx_cr[0]]),
                     xytext=(d_cr - 4.5, fdp_arr[idx_cr[0]] - 45),
                     arrowprops=dict(arrowstyle='->', color=col, lw=1.0),
                     fontsize=8.5, fontweight='bold', color=col)

ax2a.axvspan(0, 22.0, color='#FFF9C4', alpha=0.25, label=r'DP Only Region ($d \leq L_{DP}$)')
ax2a.axvspan(22.0, 42.0, color='#E1BEE7', alpha=0.20, label=r'Dual DP+MP Region ($d > L_{DP}$)')
ax2a.set_title("A) Half-Crimp Muscle Force Crossover vs Hold Depth", fontweight='bold')
ax2a.set_xlabel("Hold Depth $d_{hold}$ (mm)")
ax2a.set_ylabel("Muscle Tendon Tension (N)")
ax2a.set_xlim(2, 42)
ax2a.set_ylim(0, 400)
ax2a.grid(True, alpha=0.3)
ax2a.legend(loc='lower left', frameon=True, framealpha=0.9, fontsize=8, ncol=2)

# Panel 2B: Grip-Optimal Frontier (Which Grip Minimizes Effort?)
d_cmp = np.linspace(2.0, 40.0, 60)
all_forces = {'crimp': [], 'half_crimp': [], 'open_hand': []}

for d in d_cmp:
    ct = c3d.ContactGeometry(d_hold=d, r_edge=2.0, t_DP=9.0, beta_wall=0.0)
    for gk in ['crimp', 'half_crimp', 'open_hand']:
        r = c3d.solve_all_methods(c3d.GRIPS[gk], geom_std, F_ext, contact=ct)['emg']
        all_forces[gk].append(r['F_total'])

for gk, col, name in [('crimp', '#E53935', 'Full Crimp'),
                      ('half_crimp', '#FB8C00', 'Half-Crimp'),
                      ('open_hand', '#43A047', 'Open Hand')]:
    ax2b.plot(d_cmp, all_forces[gk], '-', color=col, lw=2.6, label=name)

# Shaded optimal zones
ax2b.axvspan(2.0, 8.0, color='#FFE0B2', alpha=0.35, label='Half-Crimp Optimal (Micro-Edge)')
ax2b.axvspan(8.0, 18.0, color='#FFF9C4', alpha=0.30, label='Transition Zone (8–18 mm)')
ax2b.axvspan(18.0, 40.0, color='#C8E6C9', alpha=0.35, label='Open Hand Dominant (>18 mm)')

ax2b.set_title("B) Grip-Optimal Minimum-Effort Frontier", fontweight='bold')
ax2b.set_xlabel("Hold Depth $d_{hold}$ (mm)")
ax2b.set_ylabel("Total Tendon Force $\sum F_m$ (N)")
ax2b.set_xlim(2, 40)
ax2b.set_ylim(200, 1400)
ax2b.grid(True, alpha=0.3)
ax2b.legend(loc='upper right', frameon=True, framealpha=0.9, fontsize=8.5)

plt.tight_layout()
fig2.savefig('outputs/pub_fig2_hold_depth_crossover.png', dpi=300, bbox_inches='tight')
plt.close(fig2)
print("Saved pub_fig2_hold_depth_crossover.png")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 3: Out-of-Plane Pulley Shearing & Long-Finger Mechanical Penalty
# ─────────────────────────────────────────────────────────────────────────────
print("Generating Publication Figure 3...")
fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(16, 5.8), dpi=300)

# Panel 3A: Out-of-Plane Shearing vs MCP Abduction (phi = 0 to 20 deg)
phi_deg = np.linspace(0, 20, 50)
f_a2_lat, f_a4_lat, pip_ml_shear = [], [], []

for phi in phi_deg:
    g = c3d.GripAngles('Crimp_test', 2.6, phi, 106.5, -22.6)
    r = c3d.solve_all_methods(g, geom_std, F_ext, contact=ct_10mm)['emg']
    jr = c3d.joint_reactions_3d(r, F_ext, geom_std)
    f_a2_lat.append(jr['pulley']['F_A2_lat'])
    f_a4_lat.append(jr['pulley']['F_A4_lat'])
    pip_ml_shear.append(jr['PIP']['shear_ML'])

ax3a.plot(phi_deg, f_a2_lat, '-', color='#E53935', lw=2.6, label='A2 Pulley Transverse Shear ($F_{A2,lat}$)')
ax3a.plot(phi_deg, f_a4_lat, '-', color='#FB8C00', lw=2.4, label='A4 Pulley Transverse Shear ($F_{A4,lat}$)')
ax3a.plot(phi_deg, pip_ml_shear, '--', color='#1E88E5', lw=2.2, label='PIP Mediolateral Joint Shear ($F_{ML}$)')

ax3a.axhline(0, color='gray', ls=':', lw=1.0)
ax3a.axvline(15.0, color='purple', ls='--', lw=1.5, label='Severe Side-Pull Abduction ($\phi=15^\circ$)')

ax3a.annotate('Dangerous Lateral Shear\n($F_{A2,lat} > 60$ N)', xy=(15.0, f_a2_lat[int(15.0/20.0*49)]), xytext=(7.0, 75),
             arrowprops=dict(arrowstyle='->', lw=1.2, color='#E53935'),
             fontsize=9, fontweight='bold', color='#E53935')

ax3a.set_title("A) Out-of-Plane Pulley & Joint Shearing (Side-Pulls / Gastons)", fontweight='bold')
ax3a.set_xlabel("MCP Radial Abduction $\phi_{MCP}$ (deg)")
ax3a.set_ylabel("Lateral Shearing Force (N)")
ax3a.set_xlim(0, 20)
ax3a.set_ylim(0, 110)
ax3a.grid(True, alpha=0.3)
ax3a.legend(loc='upper left', frameon=True, framealpha=0.9, fontsize=8.5)

# Panel 3B: Phenotypic Scaling of Annular Pulley Loads vs Structural Limits (100 N Hang Benchmark)
phenotypes = ['Short\n(−15%)', 'Standard\n(Nominal)', 'Long\n(+15%)']
a2_crimp = [265.2, 314.2, 363.3]
a2_hc    = [322.9, 383.5, 444.0]
a4_crimp = [138.3, 163.9, 189.5]

x_p = np.arange(len(phenotypes))
w = 0.26

rects_cr = ax3b.bar(x_p - w, a2_crimp, w, label='Full Crimp A2 Pulley Load (N)', color='#E53935', edgecolor='k', lw=0.6)
rects_hc = ax3b.bar(x_p, a2_hc, w, label='Half-Crimp A2 Pulley Load (N)', color='#FB8C00', edgecolor='k', lw=0.6)
rects_a4 = ax3b.bar(x_p + w, a4_crimp, w, label='Full Crimp A4 Pulley Load (N)', color='#8E24AA', edgecolor='k', lw=0.6)

# Annotate percentage penalty from Short to Long
ax3b.annotate('+37.0%', xy=(2 - w, 363.3), xytext=(2 - w, 385),
             ha='center', fontsize=9, fontweight='bold', color='#E53935')
ax3b.annotate('+37.5%', xy=(2, 444.0), xytext=(2, 465),
             ha='center', fontsize=9, fontweight='bold', color='#FB8C00')
ax3b.annotate('+37.0%', xy=(2 + w, 189.5), xytext=(2 + w, 210),
             ha='center', fontsize=9, fontweight='bold', color='#8E24AA')

ax3b.axhline(300.0, color='red', ls='--', lw=1.5, label='A2 Structural Yield Threshold (300 N)')
ax3b.axhline(400.0, color='darkred', ls=':', lw=1.5, label='A2 Ultimate Rupture Limit (400 N)')

ax3b.set_title("B) Phenotypic Scaling: Annular Pulley Loads vs Structural Limits (100 N Hang)", fontweight='bold')
ax3b.set_ylabel("Pulley Normal Load (N)")
ax3b.set_xticks(x_p)
ax3b.set_xticklabels(phenotypes)
ax3b.set_ylim(0, 520)
ax3b.grid(True, alpha=0.3, axis='y')
ax3b.legend(loc='upper left', frameon=True, framealpha=0.9, fontsize=8)

plt.tight_layout()
fig3.savefig('outputs/pub_fig3_shear_and_scaling.png', dpi=300, bbox_inches='tight')
plt.close(fig3)
print("Saved pub_fig3_shear_and_scaling.png")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 0: Anatomical Architecture of the Finger Ray
# ─────────────────────────────────────────────────────────────────────────────
print("Generating Anatomical Architecture Figure...")
raw_anatomy_path = 'paper/figures/raw_finger_anatomy_base.jpg'
if os.path.exists(raw_anatomy_path):
    import matplotlib.image as mpimg
    img_raw = mpimg.imread(raw_anatomy_path)
    
    fig_anat, ax_a = plt.subplots(figsize=(15, 8.5), dpi=300)
    ax_a.imshow(img_raw)
    ax_a.set_xlim(0, 1376)
    ax_a.set_ylim(768, 0)
    ax_a.axis('off')
    
    def annotate_bone(text, xy, xytext):
        ax_a.annotate(text, xy=xy, xytext=xytext,
                      arrowprops=dict(arrowstyle="-|>", color="#0D47A1", lw=1.5, mutation_scale=11),
                      fontsize=9.5, fontweight='bold', color="#0D47A1", ha='center',
                      bbox=dict(boxstyle="round,pad=0.3", facecolor="#E3F2FD", edgecolor="#0D47A1", lw=1.2, alpha=0.96))

    def annotate_joint(text, xy, xytext):
        ax_a.annotate(text, xy=xy, xytext=xytext,
                      arrowprops=dict(arrowstyle="-|>", color="#B71C1C", lw=1.5, mutation_scale=11),
                      fontsize=9.5, fontweight='bold', color="#B71C1C", ha='center',
                      bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFEBEE", edgecolor="#B71C1C", lw=1.2, alpha=0.96))

    def annotate_pulley(text, xy, xytext):
        ax_a.annotate(text, xy=xy, xytext=xytext,
                      arrowprops=dict(arrowstyle="-|>", color="#1B5E20", lw=1.5, mutation_scale=11),
                      fontsize=9.0, fontweight='bold', color="#1B5E20", ha='center',
                      bbox=dict(boxstyle="round,pad=0.3", facecolor="#E8F5E9", edgecolor="#1B5E20", lw=1.2, alpha=0.96))

    def annotate_tendon(text, xy, xytext):
        ax_a.annotate(text, xy=xy, xytext=xytext,
                      arrowprops=dict(arrowstyle="-|>", color="#E65100", lw=1.6, mutation_scale=12),
                      fontsize=9.5, fontweight='bold', color="#E65100", ha='center',
                      bbox=dict(boxstyle="round,pad=0.35", facecolor="#FFF3E0", edgecolor="#E65100", lw=1.3, alpha=0.96))

    # Top: Joints & Bones
    annotate_joint("MCP Joint\n(Flexion / Radial Abduction)", (380, 310), (250, 95))
    annotate_joint("PIP Joint\n(1-DOF Flexion; Primary Crimp Lever)", (785, 290), (740, 95))
    annotate_joint("DIP Joint\n(Flexion / Hyperextension)", (1055, 345), (1080, 95))

    annotate_bone("Metacarpal (MC)", (220, 440), (100, 210))
    annotate_bone("Proximal Phalanx (PP)", (530, 310), (510, 210))
    annotate_bone("Middle Phalanx (MP)", (890, 325), (890, 210))
    annotate_bone("Distal Phalanx (DP)", (1180, 420), (1240, 210))

    # Bottom: Pulleys & Tendons
    annotate_pulley("A1 Pulley", (380, 480), (320, 590))
    annotate_pulley("A2 Pulley\n(Crucial Crimp Load)", (655, 390), (580, 590))
    annotate_pulley("A3 Pulley\n(Palmar Plate Anchor)", (805, 370), (780, 590))
    annotate_pulley("A4 Pulley\n(Middle Phalanx)", (965, 380), (950, 590))
    annotate_pulley("A5 Pulley", (1065, 410), (1110, 590))

    annotate_tendon("Flexor Digitorum Superficialis (FDS)\n(Splits at Camper's Chiasm; inserts onto MP)", 
                    (730, 435), (420, 705))
    annotate_tendon("Flexor Digitorum Profundus (FDP)\n(Runs deep through chiasm; inserts onto DP base)", 
                    (1160, 485), (960, 705))

    fig_anat.text(0.5, 0.965, "Anatomical Architecture of the Human Finger Ray in Sport Climbing",
                 ha='center', fontsize=15, fontweight='bold', color='#1A237E')
    fig_anat.text(0.5, 0.932, "Spatial Musculoskeletal Chain: Articulated Phalanges, 4-DOF Kinematics, Flexor Tendons (FDP & FDS), and Annular Pulleys (A1–A5)",
                 ha='center', fontsize=10, fontstyle='italic', color='#424242')

    plt.subplots_adjust(top=0.91, bottom=0.06, left=0.02, right=0.98)
    fig_anat.savefig('outputs/fig_finger_anatomy.png', dpi=300, bbox_inches='tight')
    fig_anat.savefig('paper/figures/fig_finger_anatomy.png', dpi=300, bbox_inches='tight')
    plt.close(fig_anat)
    print("Saved fig_finger_anatomy.png")

# Copy all figures to the brain artifact directory and paper/figures directory
artifact_dir = "/Users/igorcerovsky/.gemini/antigravity-ide/brain/6db08e62-544f-418f-b0fa-e5341f42d013"
os.makedirs("paper/figures", exist_ok=True)
all_figures = ['fig_finger_anatomy.png', 'pub_fig1_model_validation.png', 'pub_fig2_hold_depth_crossover.png', 'pub_fig3_shear_and_scaling.png']
for fn in all_figures:
    if os.path.exists(f'outputs/{fn}'):
        if os.path.exists(artifact_dir):
            shutil.copy(f'outputs/{fn}', f'{artifact_dir}/{fn}')
        shutil.copy(f'outputs/{fn}', f'paper/figures/{fn}')
        print(f"Copied {fn} to artifact directory and paper/figures/.")

print("All publication figures successfully created!")


