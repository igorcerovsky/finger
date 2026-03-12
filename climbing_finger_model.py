"""
============================================================
 CLIMBING FINGER BIOMECHANICS MODEL
============================================================
 Static 2D model of FDP/FDS tendon forces in rock climbing
 Focus: Disadvantage of long fingers on small holds (overhanging terrain)

 Theoretical framework:
   - Vigouroux et al. (2006) J Biomechanics 39:2583-2592
   - An et al. (1983) J Biomechanics — tendon moment arms
   - Schweizer (2001) — crimp grip biomechanics

 Anatomical defaults (middle finger, adult male):
   - Phalanx lengths: Özsoy et al. (Turkish pop.), avg male
     PP = 45 mm, MP = 28 mm, DP = 22 mm
   - Moment arms: An et al. 1983 polynomial fits
   - Grip angles: Vigouroux 2006 experimental data

 Model assumptions:
   - 2D sagittal plane (planar finger model)
   - Static equilibrium at each joint
   - Vertical fingertip force (overhang scenario)
   - Sequential joint solve: DIP → PIP → MCP
   - Passive DIP moment in crimp = 25% of external moment (Vigouroux 2006)
============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from dataclasses import dataclass, field
from typing import Optional
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
#  CONFIGURATION  (edit these values to change the simulation)
# ─────────────────────────────────────────────────────────────

class Config:
    # Load definition
    body_weight_kg   = 70.0        # Climber body weight
    bw_fraction      = 0.25        # Fraction of BW on one finger (e.g. 0.25 = 1/4 BW)
    custom_force_N   = None        # Set to a float to override BW-based load

    # Finger length comparison
    scale_short      = 0.85        # Short finger relative to standard (−15%)
    scale_long       = 1.15        # Long finger relative to standard (+15%)

    # Standard finger phalanx lengths (mm) — Literature defaults
    # Source: Özsoy et al. (Turkish pop.); broader survey average
    PP_mm = 45.0   # Proximal phalanx — Özsoy male avg: 41.7 mm; used 45 as broad average
    MP_mm = 28.0   # Middle phalanx
    DP_mm = 22.0   # Distal phalanx

    # Hold depth range for Figure 4 analysis (mm)
    hold_depth_min = 5.0
    hold_depth_max = 25.0

    # Passive DIP moment fraction for crimp (Vigouroux 2006: ~25% of external moment)
    passive_moment_fraction = 0.25

    # Output filenames
    save_figures = True
    output_prefix = "climbing_model"


# ─────────────────────────────────────────────────────────────
#  DATA CLASSES
# ─────────────────────────────────────────────────────────────

@dataclass
class FingerGeometry:
    """
    Parametric finger geometry.
    All lengths in mm. Defaults from literature (Özsoy et al.).
    """
    L1: float = Config.PP_mm   # Proximal phalanx length (mm)
    L2: float = Config.MP_mm   # Middle phalanx length (mm)
    L3: float = Config.DP_mm   # Distal phalanx length (mm)
    name: str  = "Standard"

    @property
    def total(self):
        return self.L1 + self.L2 + self.L3

    def scaled(self, factor: float, label: str = None) -> "FingerGeometry":
        tag = label or (f"Long (×{factor:.2f})" if factor > 1 else f"Short (×{factor:.2f})")
        return FingerGeometry(self.L1 * factor, self.L2 * factor, self.L3 * factor, tag)


@dataclass
class GripConfig:
    """
    Joint angles defining a grip type (degrees).
    Sign convention:
      +  = flexion  (curling toward palm)
      −  = extension / hyperextension
    Joints: MCP (proximal) | PIP (middle) | DIP (distal)

    Sources:
      Crimp / Slope: Vigouroux 2006 (experimental averages, expert climbers)
      Half-crimp:    PMC11048272 (mean PIP = 87° during climbing)
      Open hand:     Vigouroux 2006 slope grip angles
      Pinch:         estimated physiological values
    """
    name:      str
    theta_MCP: float    # degrees
    theta_PIP: float    # degrees
    theta_DIP: float    # degrees (negative = hyperextension)
    color:     str = "#555555"
    note:      str = ""


# ─────────────────────────────────────────────────────────────
#  GRIP LIBRARY
# ─────────────────────────────────────────────────────────────

GRIPS: dict[str, GripConfig] = {
    "crimp": GripConfig(
        "Crimp",  theta_MCP=2.6,  theta_PIP=106.5, theta_DIP=-22.6,
        color="#E53935",
        note="Vigouroux 2006: PIP 90–100°, DIP hyperextended. "
             "FDP:FDS ≈ 1.75 (ref)."
    ),
    "half_crimp": GripConfig(
        "Half-Crimp", theta_MCP=15.0, theta_PIP=90.0,  theta_DIP=10.0,
        color="#FB8C00",
        note="PMC11048272: mean PIP 87°, DIP partially flexed."
    ),
    "open_hand": GripConfig(
        "Open Hand", theta_MCP=21.0, theta_PIP=25.9,  theta_DIP=38.8,
        color="#43A047",
        note="Vigouroux 2006 slope: DIP 38.8°, PIP 25.9°. "
             "FDP:FDS ≈ 0.88 (ref)."
    ),
    "pinch": GripConfig(
        "Pinch", theta_MCP=25.0, theta_PIP=15.0, theta_DIP=10.0,
        color="#8E24AA",
        note="Low flexion; thumb adduction provides additional load."
    ),
}

GRIP_KEYS  = list(GRIPS.keys())
GRIP_LIST  = list(GRIPS.values())


# ─────────────────────────────────────────────────────────────
#  TENDON MOMENT ARM FUNCTIONS  (An et al. 1983)
# ─────────────────────────────────────────────────────────────
# Polynomial/linear fits to cadaveric data from An KN et al.
# J Biomechanics 1983; 16(8):639-651.
# Units: mm; input angle in degrees.

def ma_FDP_DIP(theta_deg: float) -> float:
    """FDP moment arm at DIP joint. An et al. 1983."""
    t = np.clip(theta_deg, -30.0, 90.0)
    return 6.0 + 0.045 * t           # ~6 mm at extension → ~10 mm at 90°

def ma_FDP_PIP(theta_deg: float) -> float:
    """FDP moment arm at PIP joint. An et al. 1983."""
    t = np.clip(theta_deg, 0.0, 120.0)
    return 9.0 + 0.033 * t           # ~9 mm at extension → ~13 mm at 120°

def ma_FDS_PIP(theta_deg: float) -> float:
    """FDS moment arm at PIP joint. An et al. 1983."""
    t = np.clip(theta_deg, 0.0, 120.0)
    return 7.5 + 0.020 * t           # ~7.5 mm → ~10 mm at 120°

def ma_FDP_MCP(theta_deg: float) -> float:
    """FDP moment arm at MCP. Relatively constant (An 1983)."""
    return 10.4                       # mm

def ma_FDS_MCP(theta_deg: float) -> float:
    """FDS moment arm at MCP. An 1983."""
    return 8.6                        # mm


# ─────────────────────────────────────────────────────────────
#  KINEMATICS  (2D sagittal plane)
# ─────────────────────────────────────────────────────────────

def finger_kinematics(grip: GripConfig, geom: FingerGeometry) -> dict:
    """
    Compute joint positions in 2D sagittal plane.

    Coordinate system:
      Origin = MCP joint
      x-axis = toward the wall / hold (horizontal)
      y-axis = dorsal (upward)

    Flexion angles make each segment rotate below horizontal
    (fingertip wraps down and forward around the hold).

    Cumulative angles:
      phi1 = MCP angle from horizontal (proximal phalanx)
      phi2 = phi1 + PIP angle (middle phalanx)
      phi3 = phi2 + DIP angle (distal phalanx)
    """
    phi1 = np.radians(grip.theta_MCP)
    phi2 = np.radians(grip.theta_MCP + grip.theta_PIP)
    phi3 = np.radians(grip.theta_MCP + grip.theta_PIP + grip.theta_DIP)

    MCP = np.array([0.0, 0.0])
    PIP = MCP + geom.L1 * np.array([ np.cos(phi1), -np.sin(phi1)])
    DIP = PIP + geom.L2 * np.array([ np.cos(phi2), -np.sin(phi2)])
    TIP = DIP + geom.L3 * np.array([ np.cos(phi3), -np.sin(phi3)])

    return dict(MCP=MCP, PIP=PIP, DIP=DIP, TIP=TIP,
                phi1=phi1, phi2=phi2, phi3=phi3)


def hold_depth(grip: GripConfig, geom: FingerGeometry) -> float:
    """
    Effective hold depth (mm): vertical drop of TIP below DIP level.
    A shallower hold forces a long finger into higher PIP flexion.
    """
    k = finger_kinematics(grip, geom)
    return k["DIP"][1] - k["TIP"][1]   # positive = tip below DIP


# ─────────────────────────────────────────────────────────────
#  STATIC EQUILIBRIUM SOLVER
# ─────────────────────────────────────────────────────────────

def solve_forces(grip: GripConfig, geom: FingerGeometry,
                 F_tip: float,
                 passive_frac: float = Config.passive_moment_fraction) -> dict:
    """
    Solve FDP and FDS tendon forces via sequential joint equilibrium.

    DIP equilibrium  → F_FDP
    PIP equilibrium  → F_FDS   (F_FDP already known)
    MCP equilibrium  → informational check

    Force model:
      F_tip acts HORIZONTALLY (+x, into the wall / hold).
      This matches the Vigouroux 2006 experimental setup (isometric
      grip on a device; also equivalent to pulling horizontally into
      a hold on a vertical wall face).

      Moment at joint J from horizontal tip force F_tip:
        M_J = F_tip × (J_y - TIP_y)   [vertical lever arm]
      Since TIP_y < J_y (tip hangs below joints), M_J > 0 (flexion).

    For crimp (DIP hyperextended), Vigouroux 2006 shows a passive
    moment of ~25% of the external DIP moment reduces the FDP demand.

    Returns:
        dict with F_FDP, F_FDS, moments, moment arms, kinematics
    """
    kin = finger_kinematics(grip, geom)
    MCP, PIP, DIP, TIP = kin["MCP"], kin["PIP"], kin["DIP"], kin["TIP"]

    # ── External moments at each joint (N·mm) ──────────────────
    # Force is horizontal (+x). Moment arm = vertical distance (y).
    # M_J = F_tip × (y_J - y_TIP)  [positive = flexion moment]
    M_DIP = F_tip * (DIP[1] - TIP[1])
    M_PIP = F_tip * (PIP[1] - TIP[1])
    M_MCP = F_tip * (MCP[1] - TIP[1])

    # ── Moment arms (mm) ───────────────────────────────────────
    dFDP_DIP = ma_FDP_DIP(grip.theta_DIP)
    dFDP_PIP = ma_FDP_PIP(grip.theta_PIP)
    dFDS_PIP = ma_FDS_PIP(grip.theta_PIP)
    dFDP_MCP = ma_FDP_MCP(grip.theta_MCP)
    dFDS_MCP = ma_FDS_MCP(grip.theta_MCP)

    # ── DIP equilibrium → F_FDP ───────────────────────────────
    is_crimp = (grip.theta_DIP < 0)
    if is_crimp:
        # Passive ligament / volar plate moment reduces FDP need
        net_DIP = M_DIP * (1.0 - passive_frac)
    else:
        net_DIP = M_DIP

    F_FDP = net_DIP / dFDP_DIP if dFDP_DIP > 0 else 0.0

    # ── PIP equilibrium → F_FDS ───────────────────────────────
    net_PIP = M_PIP - F_FDP * dFDP_PIP
    F_FDS = net_PIP / dFDS_PIP if dFDS_PIP > 0 else 0.0
    F_FDS = max(F_FDS, 0.0)   # cannot push

    # ── Pulley forces (simplified bow-string model) ────────────
    # A2 pulley (on proximal phalanx, resists bowstringing at PIP)
    F_A2 = 2.0 * (F_FDP + F_FDS) * np.sin(np.radians(grip.theta_PIP) / 2.0)
    # A4 pulley (on middle phalanx, resists bowstringing at DIP)
    dip_eff = max(grip.theta_DIP, 0.0)
    F_A4 = 2.0 * F_FDP * np.sin(np.radians(dip_eff) / 2.0)

    return dict(
        F_FDP=F_FDP, F_FDS=F_FDS,
        F_total=F_FDP + F_FDS,
        F_A2=F_A2, F_A4=F_A4,
        M_DIP=M_DIP, M_PIP=M_PIP, M_MCP=M_MCP,
        dFDP_DIP=dFDP_DIP, dFDP_PIP=dFDP_PIP,
        dFDS_PIP=dFDS_PIP,
        ratio=F_FDP / F_FDS if F_FDS > 0 else np.inf,
        is_crimp=is_crimp,
        kin=kin,
        grip=grip, geom=geom,
    )


# ─────────────────────────────────────────────────────────────
#  SWEEP: forces vs PIP angle
# ─────────────────────────────────────────────────────────────

def sweep_pip_angle(grip_key: str, geom: FingerGeometry,
                    F_tip: float,
                    pip_range=(5.0, 130.0), n=80) -> dict:
    """Sweep PIP angle and compute FDP/FDS forces."""
    base = GRIPS[grip_key]
    pips = np.linspace(*pip_range, n)
    f_fdp, f_fds, f_a2 = [], [], []

    for pip in pips:
        dip = -22.6 if grip_key == "crimp" else pip * 0.40
        g   = GripConfig(base.name, base.theta_MCP, pip, dip)
        r   = solve_forces(g, geom, F_tip)
        f_fdp.append(max(r["F_FDP"], 0))
        f_fds.append(max(r["F_FDS"], 0))
        f_a2.append(max(r["F_A2"],  0))

    return dict(pip=pips,
                F_FDP=np.array(f_fdp),
                F_FDS=np.array(f_fds),
                F_A2=np.array(f_a2))


# ─────────────────────────────────────────────────────────────
#  HOLD-DEPTH ANALYSIS
# ─────────────────────────────────────────────────────────────

def grip_forced_by_hold(depth_mm: float, geom: FingerGeometry) -> GripConfig:
    """
    Given a hold depth and finger geometry, determine what grip the
    finger is mechanically forced into.

    Physical rule:
      depth < 0.8 × L3  → crimp (DIP hyperextended to hook the edge)
      depth < 0.55 × L2 → half-crimp (PIP ~90°)
      else               → open hand with PIP angle proportional to depth
    """
    if depth_mm < 0.80 * geom.L3:
        return GRIPS["crimp"]
    elif depth_mm < 0.55 * geom.L2:
        pip = 90.0 + (1.0 - depth_mm / (0.55 * geom.L2)) * 25.0
        return GripConfig("Forced HC", 12.0, pip, pip * 0.30)
    else:
        pip = max(20.0, 100.0 - depth_mm * 3.2)
        return GripConfig("Forced OH", 18.0, pip, pip * 0.40)


def hold_depth_sweep(geom: FingerGeometry, F_tip: float,
                     depths: np.ndarray) -> dict:
    """Return FDP/FDS/A2 forces across a range of hold depths."""
    f_fdp, f_fds, f_a2 = [], [], []
    for d in depths:
        g = grip_forced_by_hold(d, geom)
        r = solve_forces(g, geom, F_tip)
        f_fdp.append(max(r["F_FDP"], 0))
        f_fds.append(max(r["F_FDS"], 0))
        f_a2.append(max(r["F_A2"],   0))
    return dict(F_FDP=np.array(f_fdp),
                F_FDS=np.array(f_fds),
                F_A2 =np.array(f_a2))


# ─────────────────────────────────────────────────────────────
#  DRAWING HELPERS
# ─────────────────────────────────────────────────────────────

_JOINT_COLOR = {"MCP": "#1565C0", "PIP": "#2E7D32", "DIP": "#E65100"}

def draw_finger(ax, kin: dict, geom: FingerGeometry, grip: GripConfig,
                F_tip: float, res: dict, show_tendons=True):
    """Render 2D finger schematic with force vectors."""
    MCP, PIP, DIP, TIP = kin["MCP"], kin["PIP"], kin["DIP"], kin["TIP"]
    pts = [MCP, PIP, DIP, TIP]
    xs  = [p[0] for p in pts]
    ys  = [p[1] for p in pts]

    # ── Bone segments ─────────────────────────────────────────
    ax.plot(xs, ys, "-", color="#5D4037", lw=10, solid_capstyle="round",
            solid_joinstyle="round", zorder=2, alpha=0.85)

    # ── Joint markers ─────────────────────────────────────────
    for pt, lab in zip([MCP, PIP, DIP], ["MCP", "PIP", "DIP"]):
        ax.plot(*pt, "o", color=_JOINT_COLOR[lab], ms=13, zorder=5)
        ax.annotate(lab, pt, fontsize=8, fontweight="bold",
                    color=_JOINT_COLOR[lab],
                    xytext=(0, 14), textcoords="offset points", ha="center")

    ax.plot(*TIP, "s", color="#B71C1C", ms=10, zorder=5)
    ax.annotate("TIP", TIP, fontsize=8, color="#B71C1C",
                xytext=(0, 14), textcoords="offset points", ha="center")

    # ── Fingertip force arrow (horizontal) ────────────────────
    arr_len = geom.total * 0.28
    ax.annotate("", xy=(TIP[0] + arr_len, TIP[1]), xytext=TIP,
                arrowprops=dict(arrowstyle="->", color="#B71C1C", lw=2.2))
    ax.text(TIP[0] + arr_len * 0.55, TIP[1] - 6,
            f"F={F_tip:.0f} N", color="#B71C1C", fontsize=8)

    # ── Tendons (schematic, offset to palmar side) ────────────
    if show_tendons:
        off = 5   # mm palmar offset
        # palmar unit vector perpendicular to each segment
        def palmar_offset(phi): return np.array([np.sin(phi), np.cos(phi)]) * off
        p1 = palmar_offset(kin["phi1"])
        p2 = palmar_offset(kin["phi2"])
        p3 = palmar_offset(kin["phi3"])

        fdp_x = [MCP[0]+p1[0], PIP[0]+p2[0], DIP[0]+p3[0], TIP[0]+p3[0]]
        fdp_y = [MCP[1]+p1[1], PIP[1]+p2[1], DIP[1]+p3[1], TIP[1]+p3[1]]
        ax.plot(fdp_x, fdp_y, "-", color="#1565C0", lw=2.2, alpha=0.75,
                label=f"FDP  {res['F_FDP']:.0f} N")

        fds_x = [MCP[0]+p1[0], PIP[0]+p2[0]]
        fds_y = [MCP[1]+p1[1], PIP[1]+p2[1]]
        ax.plot(fds_x, fds_y, "--", color="#6A1B9A", lw=2.2, alpha=0.75,
                label=f"FDS  {res['F_FDS']:.0f} N")

    # ── Moment arm dashes (vertical = lever arm for horizontal force) ─
    for jt, c in zip([DIP, PIP], ["#E65100", "#2E7D32"]):
        ax.plot([jt[0], jt[0]], [jt[1], TIP[1]], "--", color=c, lw=0.9, alpha=0.45)

    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)
    ax.set_xlabel("x (mm) toward wall", fontsize=8)
    ax.set_ylabel("y (mm) dorsal", fontsize=8)
    title_lines = [
        f"{grip.name}",
        f"FDP = {res['F_FDP']:.0f} N  |  FDS = {res['F_FDS']:.0f} N",
        f"Ratio FDP/FDS = {res['ratio']:.2f}",
    ]
    ax.set_title("\n".join(title_lines), fontsize=9, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")


# ─────────────────────────────────────────────────────────────
#  MAIN SIMULATION
# ─────────────────────────────────────────────────────────────

def run_simulation():
    # ── External load ─────────────────────────────────────────
    F_tip = (Config.custom_force_N
             if Config.custom_force_N
             else Config.body_weight_kg * 9.81 * Config.bw_fraction)

    # ── Finger geometries ─────────────────────────────────────
    geom_std   = FingerGeometry(Config.PP_mm, Config.MP_mm, Config.DP_mm,
                                f"Standard ({Config.PP_mm:.0f}/{Config.MP_mm:.0f}/"
                                f"{Config.DP_mm:.0f} mm)")
    geom_short = geom_std.scaled(Config.scale_short)
    geom_long  = geom_std.scaled(Config.scale_long)
    geoms  = [geom_short, geom_std, geom_long]
    gcols  = ["#2196F3", "#4CAF50", "#F44336"]
    glabls = [g.name for g in geoms]

    # ── Pre-compute all canonical results ─────────────────────
    #   all_res[grip_key][i] = results for geom i
    all_res = {k: [solve_forces(GRIPS[k], g, F_tip) for g in geoms]
               for k in GRIP_KEYS}

    # ════════════════════════════════════════════════════════════
    # FIG 1 — Bar chart: FDP & FDS across grip types × geometry
    # ════════════════════════════════════════════════════════════
    fig1, axes = plt.subplots(2, 4, figsize=(18, 9))
    fig1.suptitle(
        f"FDP & FDS Tendon Forces by Grip Type vs Finger Length\n"
        f"Load: {F_tip:.1f} N  ({Config.bw_fraction*100:.0f}% BW, {Config.body_weight_kg} kg)",
        fontsize=13, fontweight="bold")

    for col, (key, grip) in enumerate(GRIPS.items()):
        for row, (label, ykey) in enumerate([("FDP Force (N)", "F_FDP"),
                                              ("FDS Force (N)", "F_FDS")]):
            ax = axes[row, col]
            vals = [all_res[key][i][ykey] for i in range(3)]
            bars = ax.bar(range(3), vals, color=gcols, edgecolor="k", lw=0.6)
            ax.set_xticks(range(3))
            ax.set_xticklabels(["Short", "Std", "Long"], fontsize=8)
            ax.set_ylabel(label if col == 0 else "", fontsize=9)
            ax.set_title(f"{grip.name}\n{label}", fontsize=9, fontweight="bold",
                         color=grip.color)
            ymax = max(max(vals) * 1.35, 30)
            ax.set_ylim(0, ymax)
            ax.grid(axis="y", alpha=0.3)
            for bar, v in zip(bars, vals):
                if v > 1:
                    ax.text(bar.get_x() + bar.get_width()/2,
                            bar.get_height() + ymax * 0.02,
                            f"{v:.0f}", ha="center", va="bottom", fontsize=8)

    patches = [mpatches.Patch(color=c, label=l) for c, l in zip(gcols, glabls)]
    fig1.legend(handles=patches, loc="lower center", ncol=3, fontsize=9,
                bbox_to_anchor=(0.5, 0.00))
    plt.tight_layout(rect=[0, 0.06, 1, 1])

    # ════════════════════════════════════════════════════════════
    # FIG 2 — FDP & FDS vs PIP angle (sweeps)
    # ════════════════════════════════════════════════════════════
    fig2, axes2 = plt.subplots(2, 4, figsize=(18, 9))
    fig2.suptitle(
        f"Tendon Forces vs PIP Flexion Angle  |  Short / Standard / Long Finger\n"
        f"Load: {F_tip:.1f} N",
        fontsize=13, fontweight="bold")

    for col, key in enumerate(GRIP_KEYS):
        grip = GRIPS[key]
        for row, ykey in enumerate(["F_FDP", "F_FDS"]):
            ax = axes2[row, col]
            for geom, c in zip(geoms, gcols):
                sw = sweep_pip_angle(key, geom, F_tip)
                ax.plot(sw["pip"], sw[ykey], color=c, lw=2.0, label=geom.name)
            ax.axvline(grip.theta_PIP, color="k", ls=":", lw=1.2, alpha=0.7,
                       label=f"Canonical PIP={grip.theta_PIP:.0f}°")
            label = "FDP Force (N)" if ykey == "F_FDP" else "FDS Force (N)"
            ax.set_title(f"{grip.name}\n{label}", fontsize=9,
                         fontweight="bold", color=grip.color)
            ax.set_xlabel("PIP Flexion Angle (°)", fontsize=8)
            ax.set_ylabel(label if col == 0 else "", fontsize=8)
            ax.set_xlim(5, 130)
            ax.set_ylim(bottom=0)
            ax.grid(True, alpha=0.25)
            if col == 0 and row == 0:
                ax.legend(fontsize=7, loc="upper left")

    plt.tight_layout()

    # ════════════════════════════════════════════════════════════
    # FIG 3 — Finger schematics (standard geometry, all grips)
    # ════════════════════════════════════════════════════════════
    fig3, axes3 = plt.subplots(1, 4, figsize=(18, 7))
    fig3.suptitle(
        f"Finger Kinematics & Force Schematic — Standard Geometry  |  Load: {F_tip:.1f} N",
        fontsize=13, fontweight="bold")

    for col, (key, grip) in enumerate(GRIPS.items()):
        res = all_res[key][1]   # standard geometry
        draw_finger(axes3[col], res["kin"], geom_std, grip, F_tip, res)

    plt.tight_layout()

    # ════════════════════════════════════════════════════════════
    # FIG 4 — Hold depth analysis (long-finger disadvantage)
    # ════════════════════════════════════════════════════════════
    depths = np.linspace(Config.hold_depth_min, Config.hold_depth_max, 60)

    fig4, ax4s = plt.subplots(1, 3, figsize=(17, 6))
    fig4.suptitle(
        "Long Finger Disadvantage on Small Holds (Overhang)\n"
        "Tendon forces vs Hold Depth — short / standard / long finger",
        fontsize=13, fontweight="bold")

    sweep_data = {g.name: hold_depth_sweep(g, F_tip, depths) for g in geoms}
    hold_titles = ["FDP Force (N)", "FDS Force (N)", "A2 Pulley Force (N)"]
    hold_keys   = ["F_FDP", "F_FDS", "F_A2"]

    for i, (title, hkey) in enumerate(zip(hold_titles, hold_keys)):
        ax = ax4s[i]
        for geom, c in zip(geoms, gcols):
            ax.plot(depths, sweep_data[geom.name][hkey],
                    color=c, lw=2.2, label=geom.name)

        # Zone shading
        ax.axvspan(depths[0], 10, color="#FFCDD2", alpha=0.35, zorder=0)
        ax.axvspan(10, 16,        color="#FFE0B2", alpha=0.35, zorder=0)
        ax.axvspan(16, depths[-1],color="#C8E6C9", alpha=0.25, zorder=0)
        ymax = ax.get_ylim()[1]
        ax.text(7.5, ymax * 0.92, "Crimp\nzone", ha="center", fontsize=8,
                color="#C62828", fontweight="bold")
        ax.text(13,  ymax * 0.92, "Half-\ncrimp", ha="center", fontsize=8,
                color="#E65100")
        ax.text(20,  ymax * 0.92, "Open\nhand", ha="center", fontsize=8,
                color="#2E7D32")

        ax.set_xlabel("Hold Depth (mm)", fontsize=9)
        ax.set_ylabel(title, fontsize=9)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)

    plt.tight_layout()

    # ════════════════════════════════════════════════════════════
    # FIG 5 — Summary: FDP/FDS ratio, total force, % difference
    # ════════════════════════════════════════════════════════════
    fig5, ax5s = plt.subplots(1, 3, figsize=(17, 6))
    fig5.suptitle(
        "Biomechanical Summary — Long Finger Disadvantage",
        fontsize=13, fontweight="bold")

    xpos    = np.arange(len(GRIP_KEYS))
    width   = 0.25

    # 5a — FDP/FDS ratio
    ax = ax5s[0]
    for i, (geom, c) in enumerate(zip(geoms, gcols)):
        ratios = [all_res[k][i]["ratio"] for k in GRIP_KEYS]
        ax.bar(xpos + (i-1)*width, ratios, width, color=c, label=geom.name,
               edgecolor="k", lw=0.5)
    ax.axhline(1.75, color="#E53935", ls="--", lw=1.2, label="Vigouroux crimp ref (1.75)")
    ax.axhline(0.88, color="#43A047", ls="--", lw=1.2, label="Vigouroux slope ref (0.88)")
    ax.set_xticks(xpos)
    ax.set_xticklabels([g.name for g in GRIP_LIST], rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("FDP / FDS ratio")
    ax.set_title("FDP:FDS Force Ratio\n(higher → more A2 pulley stress)", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(axis="y", alpha=0.3)

    # 5b — Total tendon force
    ax = ax5s[1]
    for i, (geom, c) in enumerate(zip(geoms, gcols)):
        totals = [all_res[k][i]["F_total"] for k in GRIP_KEYS]
        ax.bar(xpos + (i-1)*width, totals, width, color=c, label=geom.name,
               edgecolor="k", lw=0.5)
    ax.set_xticks(xpos)
    ax.set_xticklabels([g.name for g in GRIP_LIST], rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Total Tendon Force (N)")
    ax.set_title("FDP + FDS Total Tendon Load\n(higher → greater injury risk)", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(axis="y", alpha=0.3)

    # 5c — % increase long vs short
    ax = ax5s[2]
    pct_changes = []
    for k in GRIP_KEYS:
        t_short = all_res[k][0]["F_total"]
        t_long  = all_res[k][2]["F_total"]
        pct = (t_long - t_short) / t_short * 100.0 if t_short > 0 else 0.0
        pct_changes.append(pct)
    bar_cols = ["#F44336" if v > 0 else "#4CAF50" for v in pct_changes]
    bars = ax.bar([g.name for g in GRIP_LIST], pct_changes,
                  color=bar_cols, edgecolor="k", lw=0.5)
    for bar, v in zip(bars, pct_changes):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + (1.5 if v >= 0 else -7),
                f"{v:+.1f}%", ha="center", fontsize=9, fontweight="bold")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticklabels([g.name for g in GRIP_LIST], rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("% Change in Total Tendon Force")
    ax.set_title(
        f"Long vs Short Finger\n"
        f"(×{Config.scale_long:.2f} vs ×{Config.scale_short:.2f} — same external load)",
        fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    return fig1, fig2, fig3, fig4, fig5, all_res, geoms, F_tip


# ─────────────────────────────────────────────────────────────
#  CONSOLE SUMMARY TABLE
# ─────────────────────────────────────────────────────────────

def print_summary(all_res, geoms, F_tip):
    W = 90
    print("\n" + "═"*W)
    print("  CLIMBING FINGER BIOMECHANICS — FORCE SUMMARY")
    print(f"  External load at fingertip: {F_tip:.1f} N")
    print("═"*W)
    hdr = f"{'Grip':<14} {'Geometry':<26} {'F_FDP':>9} {'F_FDS':>9} "
    hdr += f"{'Total':>9} {'Ratio':>8} {'A2 Pulley':>10}"
    print(hdr)
    print("─"*W)

    for key in GRIP_KEYS:
        grip = GRIPS[key]
        for i, geom in enumerate(geoms):
            r = all_res[key][i]
            ratio_str = f"{r['ratio']:.2f}" if not np.isinf(r['ratio']) else "inf"
            print(f"{grip.name:<14} {geom.name:<26} "
                  f"{r['F_FDP']:>9.1f} {r['F_FDS']:>9.1f} "
                  f"{r['F_total']:>9.1f} {ratio_str:>8} {r['F_A2']:>10.1f}")
        print()

    print("  Literature validation (Vigouroux 2006):")
    print("    Crimp grip  → FDP/FDS ratio ≈ 1.75 | FDS ~ 250 N at max effort")
    print("    Slope grip  → FDP/FDS ratio ≈ 0.88 | A2 pulley 36× lower than crimp")
    print("    A2 pulley failure threshold: ~300–400 N (Schweizer 2001)")
    print("═"*W)


# ─────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("═"*60)
    print("  CLIMBING FINGER BIOMECHANICS SIMULATOR")
    print("  Based on Vigouroux et al. 2006 J Biomechanics")
    print("═"*60)
    print(f"\nFingergeometry (standard, adult male middle finger):")
    print(f"  Proximal phalanx : {Config.PP_mm} mm  (Özsoy avg ≈ 41.7 mm)")
    print(f"  Middle phalanx   : {Config.MP_mm} mm")
    print(f"  Distal phalanx   : {Config.DP_mm} mm")
    print(f"\nLoad: {Config.body_weight_kg} kg × 9.81 × {Config.bw_fraction:.2f} "
          f"= {Config.body_weight_kg * 9.81 * Config.bw_fraction:.1f} N per finger\n")

    fig1, fig2, fig3, fig4, fig5, all_res, geoms, F_tip = run_simulation()

    print_summary(all_res, geoms, F_tip)

    if Config.save_figures:
        for i, fig in enumerate([fig1, fig2, fig3, fig4, fig5], 1):
            fname = f"/mnt/user-data/outputs/{Config.output_prefix}_fig{i}.png"
            fig.savefig(fname, dpi=150, bbox_inches="tight")
            print(f"  Saved → {fname}")

    plt.show()
    print("\nDone.")
