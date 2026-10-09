"""
============================================================
 3D CLIMBING FINGER BIOMECHANICS MODEL
============================================================
 Extension of the 2D model to full 3D static analysis.

 New capabilities vs 2D model:
   - MCP: 2 DOF (flexion θ + radial abduction φ)
   - Lumbrical (LU) muscle: MCP flexor / PIP+DIP extensor
   - 4th equilibrium equation: MCP radial abduction
   - 3 solution methods compared side-by-side (see below)
   - 3D A2 / A4 pulley force vectors with out-of-plane component
   - 6-DOF joint reaction wrenches (injury assessment)

 Why three methods?
   With 3 muscles (FDP, FDS, LU) and 3 primary flexion DOF the
   system is exactly determined. The 4th DOF (MCP abduction)
   adds a redundant row. The three methods differ in which
   equations they honour and how they weight residuals:

     Method 1 — Direct (3x3):
       Solve DIP/PIP/MCP-flex exactly. Abduction handled by
       moment arms passively. Zero assumptions. Fast.

     Method 2 — EMG-constrained:
       Fix F_FDP/F_FDS = ratio from Vigouroux 2006 intra-muscular
       EMG data. Reduce to 2x2 system, exact solve. Physiologically
       validated; grip-specific ratios.

     Method 3 — Lumbrical-minimising:
       Same EMG ratio, but set F_LU = 0 first (lumbricals are
       stabilisers, not primary flexors). Solve 2-muscle system.

 Coordinate system:
   x -> toward wall (grip force direction)
   y -> dorsal (upward in standard climbing posture)
   z -> radial (toward thumb)
   Origin = MCP joint centre

 References:
   Vigouroux et al. 2006 J Biomechanics 39:2583-2592
   An KN et al. 1983 J Biomechanics 16(8):639-651
   Brand & Hollister 1999 Clinical Mechanics of the Hand
============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D           # noqa
import matplotlib.patches as mpatches
from dataclasses import dataclass
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────────────────

class Config:
    body_weight_kg = 70.0
    bw_fraction    = 0.25
    custom_force_N = None
    F_lateral_N    = 0.0           # side-pull (+z, radial)

    # Middle finger, adult male (Ozsoy et al.)
    PP_mm = 45.0
    MP_mm = 28.0
    DP_mm = 22.0

    scale_short = 0.85
    scale_long  = 1.15

    # Passive DIP moment (crimp only) - Vigouroux 2006
    passive_frac = 0.25

    # EMG-derived FDP/FDS ratios (Vigouroux 2006)
    EMG_ratios = {
        'crimp':      1.75,
        'half_crimp': 1.20,
        'open_hand':  0.88,
    }

    wrist_pos     = np.array([-50.0, 0.0, 0.0])   # mm

    # ── Capstan Pulley Properties ─────────────────────────────
    use_capstan   = True
    mu_tendon     = 0.08   # tendon-sheath friction coefficient (Schweizer 2003)
    w_tendon_mm   = 4.0    # average flexor tendon width
    L_A2_mm       = 15.0   # A2 pulley band length
    L_A4_mm       = 8.0    # A4 pulley band length
    a2_pip_share  = 0.50   # proximal share of PIP bowstringing restrained by A2
    a4_pip_share  = 0.40   # distal share of PIP bowstringing restrained by A4

    # ── Instantaneous Centers of Rotation (ICR) ───────────────
    use_icr_shifting   = True
    icr_shift_max_PIP  = 2.0  # mm palmar translation at 90 deg flexion
    icr_shift_max_DIP  = 1.5  # mm palmar translation at 90 deg flexion

    # ── Contact geometry defaults ─────────────────────────────
    # DP palmar-dorsal thickness: Butz et al. 2012 (adult male middle finger)
    t_DP_mm       = 9.0    # mm
    
    # Non-linear Skin Pulp Compression (Serina et al. 1997)
    use_pulp_compression = True
    pulp_compress_k      = 1.15
    pulp_compress_F0     = 10.0
    pulp_compress_max    = 2.5
    
    # ── Anthropometric Scaling & Micro-Edge Contact ─────────
    # Sub-linear allometric moment arm scaling (Synek 2019, Schmidt & Krause 2011)
    # Internal moment arms scale with bone caliber/condyle radius (~ L^0.50) rather than 1:1 with shaft length
    allometric_ma_exponent: float = 0.50
    # Characteristic transition depth (mm) for fingertip pulp corner concentration on micro-edges
    d_pulp_corner_transition: float = 4.0

    # ── Chain-Wide Kinematic Posture Optimization (Point 2C) ─────
    # True : Optimize full 3-segment phalangeal chain (MCP elevation, PIP flexion, DIP hyperextension)
    # False: Legacy 2-DOF posture optimization (PIP/DIP with fixed MCP)
    optimize_chain_posture: bool = True
    weight_posture_pulley: float = 0.20   # Pulley bowstringing penalty weight in posture optimizer
    weight_posture_spatial: float = 0.15  # Hand-to-wall horizontal reach penalty weight

    # ── Extensor (EDC) Stiffness / Co-Contraction ─────────────
    use_edc_stiffness = True
    k_EDC_stiff       = 1.5     # N per exponential joint limit modifier
    theta_dip_max     = 25.0    # deg (baseline full ROM for hyperextension)

    # Skin-on-rock friction coefficient: Quaine et al. 2000 (0.4-0.8 range)
    mu_friction   = 0.5
    # Wall overhang angle from vertical: 0 = vertical wall, 90 = horizontal roof
    beta_wall_deg = 45.0
    # Hold edge radius: 0.5 = sharp crimp, 2-5 = rounded lip
    r_edge_mm     = 2.0
    # Hold depth (mm): how far the DP palmar surface is engaged from TIP toward DIP
    d_hold_mm     = 10.0

    # ── Climber COM Geometry ─────────────────────────────────────
    # Derives the 3D force vector from body geometry rather than a fixed wall
    # angle. The angle is determined by:
    #   h_below_hold_mm: vertical distance COM is BELOW the hold (mm)
    #     ~0: hold at same height as COM (arms at side)
    #     ~150: hold at shoulder level, COM near chin  (default, typical crimp)
    #     ~600: hold high overhead
    #   d_com_mm: perpendicular COm distance from wall surface (mm)
    # Model valid for beta_wall <= ~50 deg. On roofs, unmodelled body tension
    # (core/hip flexors) acts as a lower-bound estimate. See physics.md §3.1.
    use_com_vectoring  = True
    h_below_hold_mm    = 150.0   # COM vertical distance below hold (mm)
    d_com_mm           = 300.0   # COM perpendicular distance from wall (mm)

    # ── Interossei config ─────────────────────────────────────
    use_interossei     = True
    # ── Intrinsic & Extensor physiological capacities ──────────────
    # Physiological capacity bounds: F_max = PCSA * sigma_max (sigma_max ~ 35 N/cm^2, Lieber & Friden 2000)
    # Applied to bound intrinsic and antagonist recruitment in the multi-muscle solver.
    sigma_max_N_cm2    = 35.0    # Muscle specific isometric tension (N/cm²)
    PCSA_RI            = 2.8     # Radial Interosseous PCSA (cm²) -> ~98 N capacity
    PCSA_UI            = 2.2     # Ulnar Interosseous PCSA (cm²)  -> ~77 N capacity
    PCSA_LU            = 0.4     # Lumbrical PCSA (cm²)           -> ~14 N capacity
    PCSA_EDC           = 2.0     # EDC slip PCSA (cm²)            -> ~70 N capacity

    # ── Moment Arm Source ─────────────────────────────────────
    # 'an1983'   : An et al. 1983 literature averages (original)
    # 'peerj'    : PeerJ 7470 CT-calibrated path points (Iter 15)
    moment_arm_source = 'peerj'

    # ── Wrist Extension Coupling (Iteration 18) ───────────────────
    # Climbers extend the wrist during crimping (typically 20-35 deg),
    # which increases passive tension and effective flexor moment arm at MCP.
    use_wrist_coupling = True
    theta_wrist_deg    = 25.0    # Wrist extension angle (deg)
    k_wrist_ma         = 0.04    # Moment arm shift per degree of wrist extension (mm/deg)

    # ── Normal-Load Dependent Friction (Iteration 18) ────────────
    # Non-linear skin tribology (Fuss & Niegl 2008, Derler & Gerhardt 2012, Amca et al. 2012)
    # mu_eff(F_N) = mu_0 * (F_ref / max(F_N, 1.0))^(1-n)
    use_adaptive_friction = True
    friction_n            = 0.85   # Exponent (0.75 - 0.85 typical for skin)
    friction_F_ref        = 20.0   # Reference force where mu = mu_friction (N)

    save_figures  = True
    show_figures  = False
    output_prefix = "climbing_3d"

    # ── Anthropometric Moment Arm Scaling (Review Rec A1) ────────
    # True  : Isometric baseline where internal tendon moment arms scale proportionally
    #         with bone dimensions (ma ∝ f). Under isometric scaling on relative hold depths
    #         (d/L_DP = const), tendon force is scale-invariant.
    # False : Allometric / fixed moment arm mode (unscaled ma), which isolates lever penalty.
    scale_moment_arms_with_geometry = True

    # ── A2 Pulley MCP vs PIP Bowstringing Isolation (Review Rec A4) ──
    # False : Anatomical model where A2 sits on the proximal phalanx diaphysis and
    #         restrains PIP joint bowstringing (matching Schweizer 2001 & Vigouroux 2006).
    #         MCP deflection is restrained separately by the A1 / palmar aponeurosis pulley.
    # True  : Legacy mode adding wrist-to-MCP deflection vector directly into A2.
    a2_includes_mcp = False


# ─────────────────────────────────────────────────────────────
#  DATA CLASSES
# ─────────────────────────────────────────────────────────────

@dataclass
class FingerGeometry:
    L1: float = Config.PP_mm
    L2: float = Config.MP_mm
    L3: float = Config.DP_mm
    name: str = "Standard"
    scale_factor: float = 1.0

    @property
    def total(self): return self.L1 + self.L2 + self.L3

    def scaled(self, f, label=None):
        tag = label or (f"Long" if f > 1 else "Short")
        return FingerGeometry(self.L1*f, self.L2*f, self.L3*f, tag, scale_factor=f)


@dataclass
class GripAngles:
    name:       str
    theta_MCP:  float
    phi_MCP:    float   # radial abduction (deg)
    theta_PIP:  float
    theta_DIP:  float   # negative = hyperextension
    color:      str   = "#555555"
    emg_ratio:  float = 1.0


GRIPS = {
    "crimp":      GripAngles("Crimp",      2.6,  0.0, 106.5, -22.6, "#E53935", 1.75),
    "half_crimp": GripAngles("Half-Crimp", 15.0, 0.0,  90.0,  10.0, "#FB8C00", 1.20),
    "open_hand":  GripAngles("Open Hand",  20.0,  0.0, 30.0,  30.0, "#0277BD", 0.88),
}

@dataclass
class ContactGeometry:
    """
    Hold geometry and contact parameters.

    Physical model:
      The hold edge (radius r_edge) contacts the palmar surface of the DP
      at point C, located d_eff from the TIP along the DP palmar surface.

      d_eff = d_hold + r_edge * |e_DP · y_hat|   (rounded-edge depth correction)

      The correction term accounts for the rounded edge "wrapping" further
      onto the palmar surface: the larger the DP angle from horizontal,
      the more the edge bites in. For a sharp edge (r_edge=0): d_eff = d_hold.

    Force decomposition at C (in DP local frame via R_DIP^T @ F_ext):
      Local x (e_DP direction)  : along-DP friction = F_f_axial
      Local y (dorsal, +n_DP)   : normal force      = F_N
      Local z (lateral, b_DP)   : lateral friction   = F_f_lat
      Feasibility: sqrt(F_f_axial^2 + F_f_lat^2) <= mu * F_N

    Wall angle beta_wall (degrees from vertical):
      0  = vertical wall  -> F_ext = F_mag * x_hat  (Vigouroux setup)
      45 = moderate overhang
      90 = horizontal roof -> F_ext = -F_mag * y_hat  (pure gravity)
      F_ext = F_mag * (cos(beta)*x_hat - sin(beta)*y_hat)
    """
    d_hold:    float = Config.d_hold_mm
    r_edge:    float = Config.r_edge_mm
    t_DP:      float = Config.t_DP_mm
    mu:        float = Config.mu_friction
    beta_wall: float = Config.beta_wall_deg


MUSCLE_COLORS = {'FDP': '#1565C0', 'FDS': '#6A1B9A', 'LU': '#00838F'}
METHOD_LABELS  = {
    'direct':  'Direct (3×3 exact)',
    'emg':     'EMG ratio constrained',
    'lu_min':  'LU-minimising',
}
METHOD_COLORS = {'direct': '#E53935', 'emg': '#1565C0', 'lu_min': '#2E7D32'}
METHOD_STYLES = {'direct': '-', 'emg': '--', 'lu_min': '-.'}


# ─────────────────────────────────────────────────────────────
#  ROTATION MATRICES
# ─────────────────────────────────────────────────────────────

def R_flex(deg):
    """
    Rz(-theta): finger flexion toward palm (-y).
    Correct: R_flex(90deg) @ [1,0,0] = [0,-1,0]  (toward palm) -> verified
    """
    t = np.radians(deg); c, s = np.cos(t), np.sin(t)
    return np.array([[c, s, 0.], [-s, c, 0.], [0., 0., 1.]])

def R_abd(deg):
    """Ry(-phi): MCP radial abduction toward +z."""
    p = np.radians(deg); c, s = np.cos(p), np.sin(p)
    return np.array([[c,0.,-s],[0.,1.,0.],[s,0.,c]])


# ─────────────────────────────────────────────────────────────
#  3D KINEMATICS
# ─────────────────────────────────────────────────────────────

def kinematics_3d(grip, geom):
    """Forward kinematics. MCP at origin."""
    R_MCP = R_flex(grip.theta_MCP) @ R_abd(grip.phi_MCP)
    R_PIP = R_MCP @ R_flex(grip.theta_PIP)
    R_DIP = R_PIP @ R_flex(grip.theta_DIP)
    ex    = np.array([1.,0.,0.])

    # ── Instantaneous Centers of Rotation (ICR) Offsets ──
    # Map the palmar translation based on the flexion angle normalized to 90 degrees.
    shift_y_PIP = 0.0
    shift_y_DIP = 0.0
    if getattr(Config, 'use_icr_shifting', False):
        shift_y_PIP = -Config.icr_shift_max_PIP * (grip.theta_PIP / 90.0)
        shift_y_DIP = -Config.icr_shift_max_DIP * (grip.theta_DIP / 90.0)
        
    delta_PIP = np.array([0.0, shift_y_PIP, 0.0])
    delta_DIP = np.array([0.0, shift_y_DIP, 0.0])

    p_MCP = np.zeros(3)
    p_PIP = p_MCP + R_MCP @ (geom.L1 * ex + delta_PIP)
    p_DIP = p_PIP + R_PIP @ (geom.L2 * ex + delta_DIP)
    p_TIP = p_DIP + R_DIP @ (geom.L3 * ex)

    return dict(p_MCP=p_MCP, p_PIP=p_PIP, p_DIP=p_DIP, p_TIP=p_TIP,
                R_MCP=R_MCP, R_PIP=R_PIP, R_DIP=R_DIP,
                flex_axis=np.array([0.,0.,1.]),
                abd_axis=R_MCP @ np.array([0.,1.,0.]))


# ─────────────────────────────────────────────────────────────
#  MOMENT ARMS  (An et al. 1983 + Brand & Hollister 1999)
# ─────────────────────────────────────────────────────────────

def moment_arms(grip, geom=None, scale_ma=None):
    """Scalar moment arms (mm). Positive = flexion or radial abduction.
    
    Two calibration sources (selected by Config.moment_arm_source):
    
    'an1983' — An et al. 1983 Table 2 literature averages.
      Linear fits to tabulated moment arm data across joint angle ranges.
      These are population averages, not specimen-specific.
    
    'peerj' — Synek et al. 2019 (PeerJ 7470) CT-calibrated path points (Iterations 15-16).
      Linear fits to moment arms computed from cadaver-specific tendon
      path points (Geometry_Middle_Cal_Hum/) at 4 postures (R2>=0.99).
      Iteration 16 extends to ALL muscles (LU, EDC, not just FDP/FDS):
        - FDP: DIP uses An1983 (PeerJ 4.7mm incompatible with 3-DOF: absent RI/UI intrinsics)
          MCP 24% larger vs An1983
        - FDS: MCP 49% larger at theta=0 vs An1983
        - LU:  DIP/PIP via extensor mechanism fractions (RB=0.621, ES=0.379)
               MCP 9.02+0.111theta (+50% vs An1983 fixed 6.0)
        - EDC: all joints angle-dependent (was fixed at An1983 values)
    
    Note: FDS has zero moment arm at DIP — it inserts on the MP (middle
      phalanx), not the DP. The DIP row of the A matrix therefore has no FDS
      column entry (documented explicitly to avoid reconstruction errors).
    """
    tp, td, tm = grip.theta_PIP, grip.theta_DIP, grip.theta_MCP
    phi_val = getattr(grip, 'phi_MCP', 0.0)
    phi_rad = np.radians(phi_val)

    # For Digit III (middle finger), flexors align with digit midline at neutral (phi=0).
    # When abducted/adducted (phi != 0), minor angle-dependent tendon excursion occurs:
    ma_fdp_abd = -2.1 * np.sin(phi_rad) / np.sin(np.radians(15.0) + 1e-6) if abs(phi_val) > 1e-3 else 0.0
    ma_fds_abd = -1.5 * np.sin(phi_rad) / np.sin(np.radians(15.0) + 1e-6) if abs(phi_val) > 1e-3 else 0.0

    # Wrist extension coupling (Iteration 18): tenodesis increases effective MCP flexor moment arm
    wrist_boost = 0.0
    if getattr(Config, 'use_wrist_coupling', False):
        w_deg = getattr(Config, 'theta_wrist_deg', 0.0)
        wrist_boost = float(getattr(Config, 'k_wrist_ma', 0.04) * np.clip(w_deg, 0.0, 45.0))

    if Config.moment_arm_source == 'peerj':
        # Synek et al. 2019 (PeerJ 7470) CT-calibrated (Iterations 15-16)
        # Fits from Geometry_Middle_Cal_Hum path points at 4 postures (R2>=0.99)
        # Iteration 16: extends calibration to LU (extensor mechanism) and EDC
        # Iteration 17: Interossei (RI and UI) linear fits + extensor mechanism fractions
        edc_dip = min(-4.07 + 0.025*np.clip(td,-30,90), -1.5)
        edc_pip = min(-6.48 + 0.042*np.clip(tp,  0,120), -2.0)
        ma_dict = dict(
            FDP_DIP=max( 6.00 + 0.045*np.clip(td,-30,90),  2.0),  # An1983: PeerJ 4.7mm incompatible with 3-DOF solver (RI/UI intrinsics absent)
            FDP_PIP=max( 8.24 + 0.050*np.clip(tp,  0,120), 4.0),
            FDP_MCP=max( 9.89 + 0.087*np.clip(tm,-30,90),  6.0) + wrist_boost,
            FDS_PIP=max( 4.44 + 0.050*np.clip(tp,  0,120), 3.0),
            FDS_MCP=max(10.13 + 0.108*np.clip(tm,-30,90),  5.0) + wrist_boost,
            # LU via extensor mechanism (RB_frac=0.621, ES_frac=0.379)
            LU_DIP =min(-2.53 + 0.016*np.clip(td,-30,90), -0.5),  # extends DIP
            LU_PIP =min(-4.19 + 0.043*np.clip(tp,  0,120), -0.5), # extends PIP
            LU_MCP =max(min(9.02 + 0.111*np.clip(tm,-30,90), 12.0), 3.0), # flexes MCP (capped: LU assists, not dominates)
            FDP_abd=ma_fdp_abd,  # midline neutral on Digit III; scales with abduction
            FDS_abd=ma_fds_abd,
            LU_abd = 3.5,   # radial
            # Extensor mechanism (TE at DIP, ES at PIP, LE at MCP)
            EDC_DIP=edc_dip,  # extends DIP
            EDC_PIP=edc_pip,  # extends PIP
            EDC_MCP=min(-8.65 + 0.059*np.clip(tm,-30,90), -2.0),  # extends MCP
            EDC_abd=0.0,    # assumed neutral
            # Interossei MCP linear fits (Iteration 17 calibration)
            RI_MCP=max( 2.58 + 0.069*np.clip(tm,-30,90),  1.0),
            UI_MCP=max( 3.29 + 0.095*np.clip(tm,-30,90),  1.0),
            RI_abd=max( 9.78 - 0.024*np.clip(tm,-30,90),  3.0),
            UI_abd=min(-8.21 + 0.010*np.clip(tm,-30,90), -3.0),
            # Interossei extensor fractions (23.2% reaching DIP, 61.2% reaching PIP)
            RI_DIP=0.232 * edc_dip,
            UI_DIP=0.232 * edc_dip,
            RI_PIP=0.612 * edc_pip,
            UI_PIP=0.612 * edc_pip,
        )
    else:
        # An et al. 1983 literature averages (original)
        # Iteration 17: backward compatible interossei average values
        ma_dict = dict(
            FDP_DIP=max(6.0 + 0.045*np.clip(td,-30,90),  2.0),
            FDP_PIP=max(9.0 + 0.033*np.clip(tp,  0,120), 4.0),
            FDP_MCP=max(8.0 + 0.053*np.clip(tm,  0,90),  6.0) + wrist_boost,
            FDS_PIP=max(7.5 + 0.020*np.clip(tp,  0,120), 3.0),
            FDS_MCP=max(6.8 + 0.036*np.clip(tm,  0,90),  5.0) + wrist_boost,
            LU_DIP =-4.0,   # extends DIP
            LU_PIP =-5.0,   # extends PIP
            LU_MCP = 6.0,   # flexes MCP
            FDP_abd=ma_fdp_abd,  # midline neutral on Digit III; scales with abduction
            FDS_abd=ma_fds_abd,
            LU_abd = 3.5,   # radial side
            EDC_DIP=-4.0,   # extends DIP (FDS_DIP=0: FDS inserts on MP, not DP)
            EDC_PIP=-6.0,   # extends PIP
            EDC_MCP=-10.0,  # extends MCP
            EDC_abd=0.0,    # assumed neutral
            # Interossei literature fixed averages
            RI_MCP=3.0,
            UI_MCP=4.0,
            RI_abd=9.0,
            UI_abd=-8.0,
            RI_DIP=0.232 * -4.0,
            UI_DIP=0.232 * -4.0,
            RI_PIP=0.612 * -6.0,
            UI_PIP=0.612 * -6.0,
        )

    # Anthropometric scaling: internal moment arms scale sub-linearly with skeletal caliber (Point 2A, Synek 2019, Schmidt 2011)
    do_scale = scale_ma if scale_ma is not None else getattr(Config, 'scale_moment_arms_with_geometry', True)
    if do_scale and geom is not None:
        f = getattr(geom, 'scale_factor', 1.0)
        if f != 1.0:
            k_exp = getattr(Config, 'allometric_ma_exponent', 0.50)
            f_ma = float(f ** k_exp)
            ma_dict = {k: float(v * f_ma) for k, v in ma_dict.items()}

    return ma_dict


# ─────────────────────────────────────────────────────────────
#  CONTACT GEOMETRY & FORCE MODEL
# ─────────────────────────────────────────────────────────────

def compute_contact_point(grip, geom, contact: ContactGeometry, kin, F_mag: float) -> tuple:
    """
    Locate contact point(s) and distribute force between the Distal Phalanx (DP)
    and Middle Phalanx (MP) if the hold depth exceeds the length of the DP.

    Force Distribution Model — Triangular (Hertz-like):
      Skin contact pressure is non-uniform. Consistent with Hertz contact mechanics
      (Johnson 1985) and fingertip pad compliance measurements (Serina et al. 1997),
      the pressure peaks at the fingertip and tapers toward the DIP crease (DP portion),
      then rises again as the MP skin is compressed against the hold wall.

      DP (s from tip):  p(s) ∝ (1 - s/L3),  s in [0, L3]
        → area_DP  = L3/2
        → centroid at L3/3 from tip

      MP (s from DIP):  p(s) ∝ s/total,      s in [0, engaged_MP]
        → area_MP  = engaged_MP² / (2*total)
        → centroid at 2*engaged_MP/3 from DIP

    MP Centroid — Pulley-Weighted (A3):
      The A3 annular pulley (at ~15% of MP length from PIP end) acts as the primary
      skeletal anchor for skin traction over the MP. The effective force application
      point is a weighted average of the geometric centroid (40%) and the A3 position
      (60%), consistent with pulley anatomy (Doyle & Blythe 1984, Moutet 2003).

    NOTE (future improvement): d_hold is treated as a projected contact length
    (angle-independent). Future work should account for DIP/PIP angle to compute
    the true arc-length projection of the hold depth onto each phalanx.

    Returns:
      p_C_DP    : 3D contact centroid on DP
      p_C_MP    : 3D contact centroid on MP (or None if hold is shallow)
      F_DP      : Force magnitude on the DP
      F_MP      : Force magnitude on the MP
      d_eff     : effective engagement depth (mm)
      s_from_DIP: distance from DIP to DP centroid (mm)

    References:
      Johnson K.L. (1985) Contact Mechanics. Cambridge University Press.
      Serina E.R. et al. (1997) J Biomechanics 30(2):111-118.
      Doyle J.R. & Blythe W. (1984) Hand 16:419-426.
      Moutet F. (2003) Hand Clinics 19(2):168-175.
    """
    e_DP = kin['R_DIP'] @ np.array([1., 0., 0.])
    n_palm = kin['R_DIP'] @ np.array([0., -1., 0.])
    e_MP = kin['R_PIP'] @ np.array([1., 0., 0.])
    n_palm_MP = kin['R_PIP'] @ np.array([0., -1., 0.])

    # Hold normal derived from the expected external force direction
    # assuming vertical wall beta=0 maps to F_ext=x_hat, etc.
    b = np.radians(contact.beta_wall)
    n_hold = np.array([np.cos(b), -np.sin(b), 0.0])

    # Angle projection:
    cos_alpha3 = max(float(np.linalg.norm(np.cross(e_DP, n_hold))), 0.05)
    cos_alpha2 = max(float(np.linalg.norm(np.cross(e_MP, n_hold))), 0.05)

    correction = contact.r_edge * abs(float(np.dot(e_DP, n_hold)))
    proj_d_hold = contact.d_hold + correction

    max_d_hold_DP = geom.L3 * cos_alpha3

    # ── Non-linear Pulp Compression ────────────────────────────────────
    compression = 0.0
    if getattr(Config, 'use_pulp_compression', False):
        compression = Config.pulp_compress_k * np.log(1.0 + F_mag / Config.pulp_compress_F0)
        compression = min(compression, Config.pulp_compress_max)
        
    # Scale pulp thickness with allometric bone caliber (Point 2B)
    k_exp = getattr(Config, 'allometric_ma_exponent', 0.50)
    scale_geom = getattr(geom, 'scale_factor', 1.0)
    t_DP_actual = contact.t_DP * (scale_geom ** k_exp)

    # Minimum safe padding so the bone axis doesn't punch through the mathematical skin
    r_palmar = max((t_DP_actual / 2.0) - compression, 1.0)
    
    p_TIP_palmar = kin['p_TIP'] + r_palmar * n_palm

    if proj_d_hold <= max_d_hold_DP:
        # ── Shallow hold: all force on DP ──────────────────────────────────
        d_eff = proj_d_hold / cos_alpha3
        d_trans = getattr(Config, 'd_pulp_corner_transition', 4.0)
        # Hertzian corner concentration on micro-edges (Point 2B, Johnson 1985, Serina 1997):
        # As hold depth shrinks (d_eff < d_trans), pressure concentrates at the sharp edge corner,
        # shifting the centroid toward the lip and extending the unsupported bone cantilever (L3 - s_centroid).
        s_centroid = (d_eff / 3.0) * float(np.tanh(d_eff / d_trans))
        p_C_DP = p_TIP_palmar - s_centroid * e_DP
        return p_C_DP, None, F_mag, 0.0, d_eff, geom.L3 - s_centroid

    else:
        # ── Deep hold: force splits across DP (full) + MP (partial) ────────
        remain_d_hold = proj_d_hold - max_d_hold_DP
        engaged_MP = remain_d_hold / cos_alpha2
        
        # Cap MP engagement at MP length (minus small margin for numerical safety)
        engaged_MP = min(engaged_MP, geom.L2 - 0.5)
        
        d_eff = geom.L3 + engaged_MP
        total = geom.L3 + engaged_MP

        # Triangular pressure areas (proportional to integral of pressure profile)
        area_DP = geom.L3 / 2.0                           # ∫(1-s/L3)ds from 0→L3
        area_MP = (engaged_MP ** 2) / (2.0 * total)       # ∫(s/total)ds from 0→x
        total_area = area_DP + area_MP

        frac_DP = area_DP / total_area
        frac_MP = area_MP / total_area

        # DP centroid: at 1/3 of full DP length from tip (triangular load)
        p_C_DP = p_TIP_palmar - (geom.L3 / 3.0) * e_DP

        p_DIP_palmar = kin['p_DIP'] + r_palmar * n_palm_MP

        # Geometric centroid of MP contact: at 2/3 of engaged_MP from DIP
        geom_centroid_MP = p_DIP_palmar - (2.0 * engaged_MP / 3.0) * e_MP

        # A3 pulley position: ~15% of MP length from PIP (i.e. near the PIP-MP junction)
        # Force transfer to skeleton is dominated by A3 + volar plate (Moutet 2003).
        p_A3_MP = kin['p_PIP'] + kin['R_PIP'] @ (0.15 * geom.L2 * np.array([1., 0., 0.]))
        p_A3_MP_palmar = p_A3_MP + r_palmar * n_palm_MP

        # Pulley-weighted centroid: 40% geometric + 60% A3 skeletal anchor
        p_C_MP = 0.40 * geom_centroid_MP + 0.60 * p_A3_MP_palmar

        return p_C_DP, p_C_MP, F_mag * frac_DP, F_mag * frac_MP, d_eff, (geom.L3 / 3.0)


def contact_force_vector(F_mag: float, contact: ContactGeometry) -> np.ndarray:
    """
    Build the 3D external force vector from body geometry or wall angle.

    COM mode (use_com_vectoring=True):
        Direction derived from climber COM position relative to hold:
            dx = d_com_mm * cos(beta)   (wall-normal offset)
            dy = h_below_hold_mm        (vertical: COM below hold)
        Reaction force at hold:
            ux =  dx / norm  (into wall, positive x)
            uy = -dy / norm  (downward, negative y)
        On a vertical wall (beta=0) with default h=150mm, d=300mm:
            angle = atan(150/300) = 26.6 deg below horizontal.
        Model valid for beta_wall <= ~50 deg. On roofs, body tension
        (core/hip flexors) is unmodelled — force is a lower-bound estimate.

    Legacy mode (use_com_vectoring=False):
        beta_wall = 0  -> F_ext = F_mag * x_hat   (horizontal, Vigouroux)
        beta_wall = 45 -> combined
        beta_wall = 90 -> F_ext = -F_mag * y_hat  (vertical down, roof)

    Lateral component from Config.F_lateral_N is always added along z.
    """
    b = np.radians(contact.beta_wall)
    if getattr(Config, 'use_com_vectoring', False):
        dx = Config.d_com_mm * np.cos(b)      # wall-normal component
        dy = Config.h_below_hold_mm            # COM below hold (positive = below)
        norm = np.sqrt(dx**2 + dy**2)
        ux =  dx / norm   # into wall (positive x)
        uy = -dy / norm   # downward (negative y)
        return np.array([F_mag * ux, F_mag * uy, Config.F_lateral_N])
    else:
        # Legacy: static beta angle
        return np.array([F_mag * np.cos(b),
                         -F_mag * np.sin(b),
                         Config.F_lateral_N])


def check_friction_feasibility(F_ext: np.ndarray,
                                contact: ContactGeometry,
                                kin: dict) -> dict:
    """
    Decompose F_ext in the DP local frame (R_DIP coordinate axes):
      Local x (e_DP)    : axial friction component
      Local y (n_dorsal): normal component F_N  (must be > 0 for contact)
      Local z (b_lat)   : lateral friction component

    Friction feasibility: |F_friction| <= mu * F_N
    Returns analysis dict including friction utilisation ratio.
    """
    f_local   = kin['R_DIP'].T @ F_ext
    F_N       = float(f_local[1])                               # dorsal (normal)
    F_f_axial = float(f_local[0])                               # along DP
    F_f_lat   = float(f_local[2])                               # lateral
    F_f_total = float(np.sqrt(F_f_axial**2 + F_f_lat**2))      # total friction

    if F_N <= 0.01:
        return dict(feasible=False, F_N=F_N, F_f_total=F_f_total,
                    friction_ratio=np.inf, F_f_axial=F_f_axial, F_f_lat=F_f_lat,
                    mu_eff=contact.mu)

    # Adaptive non-linear skin friction (Iteration 18: Fuss et al. 2022)
    mu_base = contact.mu
    if getattr(Config, 'use_adaptive_friction', False):
        n_exp = getattr(Config, 'friction_n', 0.85)
        F_ref = getattr(Config, 'friction_F_ref', 20.0)
        mu_eff = mu_base * ((F_ref / max(F_N, 1.0)) ** (1.0 - n_exp))
        mu_eff = float(np.clip(mu_eff, 0.25, 0.85))
    else:
        mu_eff = mu_base

    ratio    = F_f_total / (mu_eff * F_N)
    feasible = ratio <= 1.0
    return dict(feasible=feasible, F_N=F_N, F_f_total=F_f_total,
                friction_ratio=ratio, F_f_axial=F_f_axial, F_f_lat=F_f_lat,
                mu_eff=mu_eff)


# ─────────────────────────────────────────────────────────────
#  EXTERNAL MOMENTS  (contact-aware)
# ─────────────────────────────────────────────────────────────

def external_moments(kin, F_ext_dir, grip, passive_frac=None,
                     p_contact_DP=None, p_contact_MP=None, F_mag_DP=0.0, F_mag_MP=0.0):
    """
    Compute external moments at each joint with distributed force support.
    F_ext_dir is the unit vector or direction geometry of the force.
    """
    if passive_frac is None:
        passive_frac = Config.passive_frac
    fa, aa = kin['flex_axis'], kin['abd_axis']

    # Normalized direction of the external force
    if np.linalg.norm(F_ext_dir) > 0:
        dir_vec = F_ext_dir / np.linalg.norm(F_ext_dir)
    else:
        dir_vec = np.zeros(3)

    # Force vectors
    F_vec_DP = F_mag_DP * dir_vec
    F_vec_MP = F_mag_MP * dir_vec

    # Default to TIP if missing
    p_app_DP = p_contact_DP if p_contact_DP is not None else kin['p_TIP']

    def moment_at(p_joint):
        # Contribution from Distal part
        M = np.cross(p_app_DP - p_joint, F_vec_DP)
        # Contribution from Middle part (if it exists AND joint is proximal to MP)
        if p_contact_MP is not None:
            # Check if this joint is PIP or MCP (MP forces don't affect DIP)
            if not np.allclose(p_joint, kin['p_DIP']):
                M += np.cross(p_contact_MP - p_joint, F_vec_MP)
        return M

    M_DIP = float(np.dot(moment_at(kin['p_DIP']), fa))
    M_PIP = float(np.dot(moment_at(kin['p_PIP']), fa))
    M_MCP = float(np.dot(moment_at(kin['p_MCP']), fa))
    M_abd = float(np.dot(moment_at(kin['p_MCP']), aa))

    if grip.theta_DIP < 0:
        M_DIP *= (1.0 - passive_frac)

    return dict(DIP=M_DIP, PIP=M_PIP, MCP=M_MCP, abd=M_abd)


def get_min_EDC_force(dip_deg: float) -> float:
    """
    Empirical antagonist stiffness. As the joint approaches extreme hyperextension,
    passive/active extensor structures exponentially stiffen to protect the capsule.
    """
    if not getattr(Config, 'use_edc_stiffness', False):
        return 0.0
    if dip_deg >= 0.0:
        return 0.0
    # Exponential rise based on how close we are to maximum ROM
    val = Config.k_EDC_stiff * np.exp(abs(dip_deg) / Config.theta_dip_max)
    return float(val)


def get_emg_ratio(r_base, frac_DP):
    """
    FDP:FDS ratio as an exact linear function of the DP force fraction.

    r_emg = r_base * (0.20 + 0.80 * frac_DP)

    Physical derivation:
      - frac_DP = 1.0 (all load on DP): DIP moment is fully intact → r_emg = r_base
      - frac_DP = 0.0 (all load on MP): DIP moment vanishes completely → r_emg = 0.20 * r_base
        (residual 20% accounts for passive FDP stiffness and lumbrical coupling)

    This is preferable to interpolating over raw d_hold because frac_DP is the
    *direct physical driver* of FDP recruitment (it governs the DIP moment magnitude),
    and it is already computed from the triangular Hertz pressure model in
    compute_contact_point(). The transition is therefore smooth, posture-aware, and
    phenotype-consistent (short DP fingers cross frac_DP < 0.5 at shallower holds).

    If frac_DP is None (tip-load mode, contact=None): returns r_base unchanged.
    """
    if frac_DP is None:
        return r_base
    frac_DP = float(np.clip(frac_DP, 0.0, 1.0))
    return float(r_base * (0.20 + 0.80 * frac_DP))


# ─────────────────────────────────────────────────────────────
#  EQUILIBRIUM POSTURE FINDER
# ─────────────────────────────────────────────────────────────

def find_equilibrium_posture(grip_base: GripAngles, geom: FingerGeometry,
                             F_ext: np.ndarray,
                             contact: 'ContactGeometry',
                             pip0: float = None, dip0: float = None,
                             mcp0: float = None,
                             optimize_chain: bool = None) -> GripAngles:
    """
    Find the finger posture that minimises total tendon force and pulley stress
    for a given hold depth and grip type.

    Parameters:
      grip_base     : Baseline GripAngles specifying nominal grip posture and style.
      geom          : FingerGeometry specifying phalanx lengths and allometric scale.
      F_ext         : Applied 3D external force vector at the hold (N).
      contact       : ContactGeometry specifying hold depth, edge radius, wall angle.
      pip0, dip0    : Optional warm-start angles for PIP and DIP (deg).
      mcp0          : Optional warm-start angle for MCP (deg).
      optimize_chain: If True, performs 3-DOF kinematic chain optimization across
                      (MCP elevation, PIP flexion, DIP hyperextension) [Point 2C].
                      If False, optimizes (PIP, DIP) with fixed MCP [2-DOF mode].
                      Defaults to Config.optimize_chain_posture.

    Biological & Mechanical Principles (Point 2C):
      1. Muscle Stress Minimisation: The motor control system coordinates multi-articular
         flexors and extensors to minimise metabolic cost (Uno et al. 1989, Latash 2012).
      2. Pulley Stress Protection: Extreme acute PIP flexion (>105°) spikes A2 bowstringing
         force non-linearly. Including a pulley stress penalty prevents unrealistic over-curling.
      3. Hand-to-Wall Kinematics: The climber maintains forearm/hand alignment with the wall
         to prevent peeling off shallow ledges. Total finger reach (x_reach = x_TIP - x_MCP)
         is constrained near nominal ergonomics.
      4. Grip Style Consistency: Optimisation operates within physiological envelopes
         specific to the grip (crimp, half-crimp, open-hand) to prevent unphysiological mode hopping.

    References:
      Vigouroux L. et al. (2011) J Biomechanics 44(8):1443-1449.
      Schweizer A. (2001) J Biomechanics 34(2):217-223.
      Uno Y. et al. (1989) Biological Cybernetics 61(2):89-101.
    """
    if optimize_chain is None:
        optimize_chain = getattr(Config, 'optimize_chain_posture', True)

    mcp_nom = grip_base.theta_MCP
    pip_nom = grip_base.theta_PIP
    dip_nom = grip_base.theta_DIP

    theta_MCP_0 = mcp0 if mcp0 is not None else mcp_nom
    theta_PIP_0 = pip0 if pip0 is not None else pip_nom
    theta_DIP_0 = dip0 if dip0 is not None else dip_nom

    F_mag = float(np.linalg.norm(F_ext[:2]))
    F_ext_dir = contact_force_vector(1.0, contact)

    kin_ref = kinematics_3d(grip_base, FingerGeometry())
    target_reach_x = float(kin_ref['p_TIP'][0] - kin_ref['p_MCP'][0])

    w_pulley  = getattr(Config, 'weight_posture_pulley', 0.20)
    w_spatial = getattr(Config, 'weight_posture_spatial', 0.15)

    name_lower = grip_base.name.lower()
    if 'crimp' in name_lower and 'half' not in name_lower:
        # Full crimp corridor: MCP elevated (0-25°), PIP flexed (95-118°), DIP hyperextended (-25 to 5°)
        bounds_3d = [(0.0, 25.0), (95.0, 118.0), (-25.0, 5.0)]
    elif 'half' in name_lower:
        # Half crimp corridor: MCP moderate (10-35°), PIP (75-105°), DIP neutral/flexed (0-25°)
        bounds_3d = [(10.0, 35.0), (75.0, 105.0), (0.0, 25.0)]
    elif 'open' in name_lower:
        # Open hand corridor: MCP flexed (15-45°), PIP extended (20-45°), DIP flexed (15-50°)
        bounds_3d = [(15.0, 45.0), (20.0, 45.0), (15.0, 50.0)]
    else:
        bounds_3d = [(max(0.0, mcp_nom - 20.0), min(60.0, mcp_nom + 20.0)),
                     (max(10.0, pip_nom - 25.0), min(120.0, pip_nom + 25.0)),
                     (max(-30.0, dip_nom - 25.0), min(80.0, dip_nom + 25.0))]

    def evaluate_posture_cost(mcp_val, pip_val, dip_val):
        g = GripAngles(grip_base.name, float(mcp_val), grip_base.phi_MCP,
                       float(pip_val), float(dip_val), grip_base.color, grip_base.emg_ratio)
        kin = kinematics_3d(g, geom)
        ma = moment_arms(g, geom=geom)

        p_C_DP, p_C_MP, F_DP, F_MP, d_eff, _ = compute_contact_point(g, geom, contact, kin, F_mag)
        ext = external_moments(kin, F_ext_dir, g,
                               p_contact_DP=p_C_DP, p_contact_MP=p_C_MP,
                               F_mag_DP=F_DP, F_mag_MP=F_MP)

        frac_DP = float(F_DP / (F_DP + F_MP + 1e-9))
        r_emg = get_emg_ratio(grip_base.emg_ratio, frac_DP)

        theta_A2, theta_A4, *_ = compute_pulley_angles(kin, geom)
        C_A2 = np.exp(Config.mu_tendon * theta_A2) if getattr(Config, 'use_capstan', True) else 1.0
        C_A4 = np.exp(Config.mu_tendon * theta_A4) if getattr(Config, 'use_capstan', True) else 1.0

        F_EDC_min = get_min_EDC_force(float(dip_val))

        from scipy.optimize import lsq_linear
        sigma_max = getattr(Config, 'sigma_max_N_cm2', 35.0)
        F_flex_ub = 1500.0
        F_LU_ub   = Config.PCSA_LU * sigma_max
        F_EDC_ub  = Config.PCSA_EDC * sigma_max
        F_RI_ub   = Config.PCSA_RI * sigma_max
        F_UI_ub   = Config.PCSA_UI * sigma_max

        if getattr(Config, 'use_interossei', False):
            A4e = np.array([
                [r_emg*ma['FDP_DIP']*C_A2*C_A4,                  ma['LU_DIP'],  ma['EDC_DIP'],  ma['RI_DIP'],  ma['UI_DIP']],
                [r_emg*ma['FDP_PIP']*C_A2 + ma['FDS_PIP']*C_A2,  ma['LU_PIP'],  ma['EDC_PIP'],  ma['RI_PIP'],  ma['UI_PIP']],
                [r_emg*ma['FDP_MCP'] + ma['FDS_MCP'],            ma['LU_MCP'],  ma['EDC_MCP'],  ma['RI_MCP'],  ma['UI_MCP']],
                [r_emg*ma['FDP_abd'] + ma['FDS_abd'],            ma['LU_abd'],  ma['EDC_abd'],  ma['RI_abd'],  ma['UI_abd']],
            ])
            b4e = np.array([ext['DIP'], ext['PIP'], ext['MCP'], ext['abd']])

            lam = 1e-4
            W = np.diag([np.sqrt(r_emg**2 + 1.0), 1.0, 1.0, 1.0, 1.0])
            A_stacked = np.vstack([A4e, lam * W])
            b_stacked = np.concatenate([b4e, np.zeros(5)])

            bounds = ([0.0, 0.0, F_EDC_min, 0.0, 0.0],
                      [F_flex_ub, F_LU_ub, F_EDC_ub, F_RI_ub, F_UI_ub])
            res = lsq_linear(A_stacked, b_stacked, bounds=bounds)
            x_clip = res.x

            F_FDS_clip = x_clip[0]
            F_LU_clip  = x_clip[1]
            F_EDC_clip = x_clip[2]
            F_RI_clip  = x_clip[3]
            F_UI_clip  = x_clip[4]
            F_FDP_clip = r_emg * F_FDS_clip

            raw_total = F_FDP_clip + F_FDS_clip + F_LU_clip + F_EDC_clip + F_RI_clip + F_UI_clip
            M_muscle = A4e.dot(x_clip)
            residual_error = float(np.linalg.norm(M_muscle - b4e))
        else:
            A3e = np.array([
                [r_emg*ma['FDP_DIP']*C_A2*C_A4,                  ma['LU_DIP'], ma['EDC_DIP']],
                [r_emg*ma['FDP_PIP']*C_A2 + ma['FDS_PIP']*C_A2,  ma['LU_PIP'], ma['EDC_PIP']],
                [r_emg*ma['FDP_MCP'] + ma['FDS_MCP'],            ma['LU_MCP'], ma['EDC_MCP']],
            ])
            b3e = np.array([ext['DIP'], ext['PIP'], ext['MCP']])
            bounds = ([0.0, 0.0, F_EDC_min], [F_flex_ub, F_LU_ub, F_EDC_ub])
            res = lsq_linear(A3e, b3e, bounds=bounds)
            x_clip = res.x

            F_FDS_clip = x_clip[0]
            F_LU_clip  = x_clip[1]
            F_EDC_clip = x_clip[2]
            F_FDP_clip = r_emg * F_FDS_clip

            raw_total = F_FDP_clip + F_FDS_clip + F_LU_clip + F_EDC_clip
            M_muscle = A3e.dot(x_clip)
            residual_error = float(np.linalg.norm(M_muscle - b3e))

        pf = pulley_forces_3d(F_FDP_clip, F_FDS_clip, kin, geom)
        F_A2 = pf['F_A2_mag']

        feas = check_friction_feasibility(F_ext_dir * F_mag, contact, kin)
        fr = float(feas.get('friction_ratio', 0.0))
        pen_fr = 0.0
        if fr > 0.8:
            k_fr = 200.0 if fr <= 1.0 else 2000.0
            pen_fr = k_fr * (fr - 0.8) ** 2

        reach_x = kin['p_TIP'][0] - kin['p_MCP'][0]
        pen_reach = w_spatial * (reach_x - target_reach_x) ** 2

        pen_posture = (
            0.06 * (mcp_val - mcp_nom) ** 2 +
            0.06 * (pip_val - pip_nom) ** 2 +
            0.06 * (dip_val - dip_nom) ** 2
        )

        return float(raw_total + w_pulley * F_A2 + 100.0 * residual_error + pen_fr + pen_reach + pen_posture)

    from scipy.optimize import minimize  # type: ignore

    if optimize_chain:
        x0 = [theta_MCP_0, theta_PIP_0, theta_DIP_0]
        bnds = bounds_3d
        def obj_chain(xy):
            try:
                return evaluate_posture_cost(xy[0], xy[1], xy[2])
            except Exception:
                return 1e9

        res = minimize(obj_chain, x0, method='L-BFGS-B', bounds=bnds,
                       options={'ftol': 1e-4, 'gtol': 1e-3, 'maxiter': 50})
        best_mcp, best_pip, best_dip = float(res.x[0]), float(res.x[1]), float(res.x[2])
    else:
        x0 = [theta_PIP_0, theta_DIP_0]
        bnds = [bounds_3d[1], bounds_3d[2]]
        def obj_2dof(xy):
            try:
                return evaluate_posture_cost(mcp_nom, xy[0], xy[1])
            except Exception:
                return 1e9

        res = minimize(obj_2dof, x0, method='L-BFGS-B', bounds=bnds,
                       options={'ftol': 1e-4, 'gtol': 1e-3, 'maxiter': 50})
        best_mcp = mcp_nom
        best_pip, best_dip = float(res.x[0]), float(res.x[1])

    return GripAngles(grip_base.name, best_mcp, grip_base.phi_MCP,
                      best_pip, best_dip, grip_base.color, grip_base.emg_ratio)


# ─────────────────────────────────────────────────────────────
#  THREE SOLUTION METHODS  (fast, no iterative optimisation)
# ─────────────────────────────────────────────────────────────

def solve_all_methods(grip: GripAngles,
                      geom: FingerGeometry,
                      F_ext: np.ndarray,
                      contact: ContactGeometry = None):
    """
    Returns dict keyed by method name, each with F_FDP, F_FDS, F_LU, etc.
    Three methods: direct, emg, lu_min.

    If contact is provided: force applied at contact point C on DP palmar surface.
    If contact is None:     force applied at TIP (original behaviour, backward compat).
    """
    kin = kinematics_3d(grip, geom)
    ma  = moment_arms(grip, geom=geom)

    # ── Determine application point and force vector ──────────
    if contact is not None:
        F_mag_total = float(np.linalg.norm(F_ext[:2]))   # sagittal magnitude
        F_ext_dir = contact_force_vector(1.0, contact) # unit direction essentially
        
        p_C_DP, p_C_MP, F_DP, F_MP, d_eff, s_from_DIP = compute_contact_point(grip, geom, contact, kin, F_mag_total)
        feas = check_friction_feasibility(F_ext_dir * F_mag_total, contact, kin) 
        c_info = {'frac_DP': float(F_DP / (F_DP + F_MP + 1e-9))}  # DP force fraction → EMG ratio
    else:
        F_ext_dir  = F_ext / (np.linalg.norm(F_ext)+1e-9)
        F_DP       = float(np.linalg.norm(F_ext))
        F_MP       = 0.0
        p_C_DP     = None      # will use TIP inside external_moments
        p_C_MP     = None
        d_eff      = None
        s_from_DIP = None
        feas       = None
        c_info     = {'frac_DP': None}

    ext = external_moments(kin, F_ext_dir, grip, p_contact_DP=p_C_DP, p_contact_MP=p_C_MP, F_mag_DP=F_DP, F_mag_MP=F_MP)

    # 3x3 matrix for DIP/PIP/MCP flexion
    A3 = np.array([
        [ma['FDP_DIP'],  0.0,          ma['LU_DIP']],
        [ma['FDP_PIP'],  ma['FDS_PIP'], ma['LU_PIP']],
        [ma['FDP_MCP'],  ma['FDS_MCP'], ma['LU_MCP']],
    ])
    b3 = np.array([ext['DIP'], ext['PIP'], ext['MCP']])

    F_EDC_min = get_min_EDC_force(grip.theta_DIP)

    if getattr(Config, 'use_interossei', False):
        from scipy.optimize import lsq_linear
        lam = 1e-4

        # Compute Capstan friction pulley multipliers
        theta_A2, theta_A4, *_ = compute_pulley_angles(kin, geom)
        C_A2 = np.exp(Config.mu_tendon * theta_A2) if getattr(Config, 'use_capstan', True) else 1.0
        C_A4 = np.exp(Config.mu_tendon * theta_A4) if getattr(Config, 'use_capstan', True) else 1.0

        # ─── Method 1: Direct 4x6 Bounded Least-Squares ───
        # Minimizes raw forces under 4 DOF constraints (DIP, PIP, MCP flexion + MCP abduction)
        # Incorporates Capstan friction for physical tendon amplification across DIP and PIP joints
        A4 = np.array([
            [ma['FDP_DIP']*C_A2*C_A4,                  0.0,          ma['LU_DIP'],  ma['EDC_DIP'],  ma['RI_DIP'],  ma['UI_DIP']],
            [ma['FDP_PIP']*C_A2,                       ma['FDS_PIP']*C_A2, ma['LU_PIP'],  ma['EDC_PIP'],  ma['RI_PIP'],  ma['UI_PIP']],
            [ma['FDP_MCP'],                            ma['FDS_MCP'], ma['LU_MCP'],  ma['EDC_MCP'],  ma['RI_MCP'],  ma['UI_MCP']],
            [ma['FDP_abd'],                            ma['FDS_abd'], ma['LU_abd'],  ma['EDC_abd'],  ma['RI_abd'],  ma['UI_abd']],
        ])
        b4 = np.array([ext['DIP'], ext['PIP'], ext['MCP'], ext['abd']])
        
        # Stacked L2 regularization to minimize muscle stress/effort
        A4_stacked = np.vstack([A4, lam * np.eye(6)])
        b4_stacked = np.concatenate([b4, np.zeros(6)])
        
        sigma_m   = getattr(Config, 'sigma_max_N_cm2', 35.0)
        F_flex_ub = 600.0  # Climber flexor hypertrophy capacity
        F_LU_ub   = float(getattr(Config, 'PCSA_LU', 0.4) * sigma_m)   # ~14 N
        F_RI_ub   = float(getattr(Config, 'PCSA_RI', 2.8) * sigma_m)   # ~98 N
        F_UI_ub   = float(getattr(Config, 'PCSA_UI', 2.2) * sigma_m)   # ~77 N
        F_EDC_ub  = F_EDC_min + 1e-6  # Antagonist stiffness floor (prevents unprompted antagonist co-contraction)

        bounds1 = ([0.0, 0.0, 0.0, F_EDC_min, 0.0, 0.0],
                   [F_flex_ub, F_flex_ub, F_LU_ub, F_EDC_ub, F_RI_ub, F_UI_ub])
        sol1 = lsq_linear(A4_stacked, b4_stacked, bounds=bounds1)
        f1 = sol1.x  # FDP, FDS, LU, EDC, RI, UI

        # ─── Method 2: EMG-constrained 4x5 Bounded Least-Squares ───
        r_emg = get_emg_ratio(grip.emg_ratio, c_info['frac_DP'] if contact else None)

        A4e = np.array([
            [r_emg*ma['FDP_DIP']*C_A2*C_A4,                  ma['LU_DIP'],  ma['EDC_DIP'],  ma['RI_DIP'],  ma['UI_DIP']],
            [r_emg*ma['FDP_PIP']*C_A2 + ma['FDS_PIP']*C_A2,  ma['LU_PIP'],  ma['EDC_PIP'],  ma['RI_PIP'],  ma['UI_PIP']],
            [r_emg*ma['FDP_MCP'] + ma['FDS_MCP'],            ma['LU_MCP'],  ma['EDC_MCP'],  ma['RI_MCP'],  ma['UI_MCP']],
            [r_emg*ma['FDP_abd'] + ma['FDS_abd'],            ma['LU_abd'],  ma['EDC_abd'],  ma['RI_abd'],  ma['UI_abd']],
        ])
        
        # Stacked L2 regularization to minimize muscle stress/effort
        W2 = np.diag([np.sqrt(r_emg**2 + 1.0), 1.0, 1.0, 1.0, 1.0])
        A4e_stacked = np.vstack([A4e, lam * W2])
        b4e_stacked = np.concatenate([b4, np.zeros(5)])
        
        bounds2 = ([0.0, 0.0, F_EDC_min, 0.0, 0.0],
                   [F_flex_ub, F_LU_ub, F_EDC_ub, F_RI_ub, F_UI_ub])
        sol2 = lsq_linear(A4e_stacked, b4e_stacked, bounds=bounds2)
        F_FDS2 = sol2.x[0]
        F_LU2  = sol2.x[1]
        F_EDC2 = sol2.x[2]
        F_RI2  = sol2.x[3]
        F_UI2  = sol2.x[4]
        F_FDP2 = r_emg * F_FDS2
        f2 = np.array([F_FDP2, F_FDS2, F_LU2, F_EDC2, F_RI2, F_UI2])

        # ─── Method 3: LU-minimising 4x4 Bounded Least-Squares ───
        A4_lu = np.array([
            [r_emg*ma['FDP_DIP']*C_A2*C_A4,                  ma['EDC_DIP'],  ma['RI_DIP'],  ma['UI_DIP']],
            [r_emg*ma['FDP_PIP']*C_A2 + ma['FDS_PIP']*C_A2,  ma['EDC_PIP'],  ma['RI_PIP'],  ma['UI_PIP']],
            [r_emg*ma['FDP_MCP'] + ma['FDS_MCP'],            ma['EDC_MCP'],  ma['RI_MCP'],  ma['UI_MCP']],
            [r_emg*ma['FDP_abd'] + ma['FDS_abd'],            ma['EDC_abd'],  ma['RI_abd'],  ma['UI_abd']],
        ])
        
        # Stacked L2 regularization to minimize muscle stress/effort
        W3 = np.diag([np.sqrt(r_emg**2 + 1.0), 1.0, 1.0, 1.0])
        A4_lu_stacked = np.vstack([A4_lu, lam * W3])
        b4_lu_stacked = np.concatenate([b4, np.zeros(4)])
        
        bounds3 = ([0.0, F_EDC_min, 0.0, 0.0],
                   [F_flex_ub, F_EDC_ub, F_RI_ub, F_UI_ub])
        sol3 = lsq_linear(A4_lu_stacked, b4_lu_stacked, bounds=bounds3)
        F_FDS3 = sol3.x[0]
        F_LU3  = 0.0
        F_EDC3 = sol3.x[1]
        F_RI3  = sol3.x[2]
        F_UI3  = sol3.x[3]
        F_FDP3 = r_emg * F_FDS3
        f3 = np.array([F_FDP3, F_FDS3, F_LU3, F_EDC3, F_RI3, F_UI3])

    else:
        # Legacy 3-muscle flexion systems
        A3 = np.array([
            [ma['FDP_DIP'],  0.0,          ma['LU_DIP']],
            [ma['FDP_PIP'],  ma['FDS_PIP'], ma['LU_PIP']],
            [ma['FDP_MCP'],  ma['FDS_MCP'], ma['LU_MCP']],
        ])
        b3 = np.array([ext['DIP'], ext['PIP'], ext['MCP']])

        # Method 1: Direct 3x3 solve
        b3_direct = b3 - F_EDC_min * np.array([ma['EDC_DIP'], ma['EDC_PIP'], ma['EDC_MCP']])
        try:
            f1 = np.linalg.solve(A3, b3_direct)
        except np.linalg.LinAlgError:
            f1 = np.linalg.lstsq(A3, b3_direct, rcond=None)[0]
        f1 = np.maximum(f1, 0.0)
        f1 = np.array([f1[0], f1[1], f1[2], F_EDC_min])

        # Method 2: EMG-constrained (Vigouroux 2006) + EDC
        from scipy.optimize import lsq_linear
        r_emg = get_emg_ratio(grip.emg_ratio, c_info['frac_DP'] if contact else None)
        theta_A2, theta_A4, *_ = compute_pulley_angles(kin, geom)
        C_A2 = np.exp(Config.mu_tendon * theta_A2) if getattr(Config, 'use_capstan', True) else 1.0
        C_A4 = np.exp(Config.mu_tendon * theta_A4) if getattr(Config, 'use_capstan', True) else 1.0

        A3e = np.array([
            [r_emg*ma['FDP_DIP']*C_A2*C_A4,                  ma['LU_DIP'], ma['EDC_DIP']],
            [r_emg*ma['FDP_PIP']*C_A2 + ma['FDS_PIP']*C_A2,  ma['LU_PIP'], ma['EDC_PIP']],
            [r_emg*ma['FDP_MCP'] + ma['FDS_MCP'],            ma['LU_MCP'], ma['EDC_MCP']],
        ])
        b3e = np.array([ext['DIP'], ext['PIP'], ext['MCP']])
        bounds2 = ([0.0, 0.0, F_EDC_min], [np.inf, np.inf, np.inf])
        sol2 = lsq_linear(A3e, b3e, bounds=bounds2)
        F_FDS2 = sol2.x[0]
        F_LU2  = sol2.x[1]
        F_EDC2 = sol2.x[2]
        F_FDP2 = r_emg * F_FDS2
        f2     = np.array([F_FDP2, F_FDS2, F_LU2, F_EDC2])

        # Method 3: LU-minimising
        A3_lu = np.array([
            [r_emg*ma['FDP_DIP']*C_A2*C_A4,                  ma['EDC_DIP']],
            [r_emg*ma['FDP_PIP']*C_A2 + ma['FDS_PIP']*C_A2,  ma['EDC_PIP']],
            [r_emg*ma['FDP_MCP'] + ma['FDS_MCP'],            ma['EDC_MCP']],
        ])
        bounds3 = ([0.0, F_EDC_min], [np.inf, np.inf])
        sol3 = lsq_linear(A3_lu, b3e, bounds=bounds3)
        F_FDS3 = sol3.x[0]
        F_LU3  = 0.0
        F_EDC3 = sol3.x[1]
        F_FDP3 = r_emg * F_FDS3
        f3 = np.array([F_FDP3, F_FDS3, F_LU3, F_EDC3])

    results = {}
    for mname, f in [('direct', f1), ('emg', f2), ('lu_min', f3)]:
        if getattr(Config, 'use_interossei', False):
            FDP, FDS, LU, EDC, RI, UI = f
            ratio = FDP/FDS if FDS > 0.1 else np.inf
            results[mname] = dict(
                F_FDP=float(FDP), F_FDS=float(FDS), F_LU=float(LU), F_EDC=float(EDC),
                F_RI=float(RI), F_UI=float(UI),
                F_total=float(FDP+FDS+LU+EDC+RI+UI),
                ratio=ratio,
                f_vec=f, kin=kin, ext=ext,
                p_C=p_C_DP, d_eff=d_eff, s_from_DIP=s_from_DIP,
                friction=feas,
            )
        else:
            FDP, FDS, LU, EDC = f
            ratio = FDP/FDS if FDS > 0.1 else np.inf
            results[mname] = dict(
                F_FDP=float(FDP), F_FDS=float(FDS), F_LU=float(LU), F_EDC=float(EDC),
                F_RI=0.0, F_UI=0.0,
                F_total=float(FDP+FDS+LU+EDC),
                ratio=ratio,
                f_vec=f, kin=kin, ext=ext,
                p_C=p_C_DP, d_eff=d_eff, s_from_DIP=s_from_DIP,
                friction=feas,
            )
    return results


# ─────────────────────────────────────────────────────────────
#  3D PULLEY FORCES  (Capstan model & joint limits)
# ─────────────────────────────────────────────────────────────

def compute_pulley_angles(kin, geom):
    ex    = np.array([1., 0., 0.])
    R_MCP = kin['R_MCP']
    R_PIP = kin['R_PIP']

    p_A2 = kin['p_MCP'] + R_MCP @ (0.40 * geom.L1 * ex)
    p_A4 = kin['p_PIP'] + R_PIP @ (0.20 * geom.L2 * ex)
    
    # Unit direction vectors along phalanges
    u_PP = R_MCP @ ex
    u_MP = R_PIP @ ex
    
    # Unit direction vector from wrist to MCP joint
    u_wrist = kin['p_MCP'] - Config.wrist_pos
    u_wrist /= np.linalg.norm(u_wrist)
    
    # Deflection angle at MCP (between wrist and PP)
    dot_MCP = np.clip(np.dot(u_wrist, u_PP), -1.0, 1.0)
    theta_MCP = np.arccos(dot_MCP)
    
    # Deflection angle at PIP (between PP and MP)
    dot_PIP = np.clip(np.dot(u_PP, u_MP), -1.0, 1.0)
    theta_PIP = np.arccos(dot_PIP)
    
    # Physiological bowstringing sharing across joints (Roloff et al. 2006, Vigouroux et al. 2006):
    # A2 restrains entry MCP deflection plus proximal share (~50%) of PIP bowstringing
    # A4 restrains distal share (~40%) of PIP bowstringing (A3/capsule absorbs ~10%)
    f_A2_pip = float(getattr(Config, 'a2_pip_share', 0.50))
    f_A4_pip = float(getattr(Config, 'a4_pip_share', 0.40))
    
    if getattr(Config, 'a2_includes_mcp', False):
        theta_A2 = theta_MCP + f_A2_pip * theta_PIP
    else:
        theta_A2 = f_A2_pip * theta_PIP
    theta_A4 = f_A4_pip * theta_PIP
    
    # Resultant bowstringing vector directions: delta_u = u_in - u_out
    delta_MCP = u_wrist - u_PP
    delta_PIP = u_PP - u_MP
    
    return theta_A2, theta_A4, delta_MCP, delta_PIP, u_PP, u_MP, p_A2, p_A4

def pulley_forces_3d(F_FDP, F_FDS, kin, geom):
    """
    Computes 3D pulley forces for A2 and A4 based on tendon deflection geometry
    and physiological bowstringing sharing across joints (Roloff et al. 2006, Vigouroux et al. 2006).
    """
    theta_A2, theta_A4, delta_MCP, delta_PIP, u_PP, u_MP, p_A2, p_A4 = compute_pulley_angles(kin, geom)
    
    f_A2_pip = float(getattr(Config, 'a2_pip_share', 0.50))
    f_A4_pip = float(getattr(Config, 'a4_pip_share', 0.40))
    
    # Both FDP and FDS pass under A2
    T_A2 = (F_FDP + F_FDS) * np.exp(Config.mu_tendon * theta_A2)
    # When a2_includes_mcp is True: legacy mode adding wrist-to-MCP deflection
    # When False (default): A2 restrains PIP bowstringing on PP shaft (Schweizer 2001, Vigouroux 2006)
    if getattr(Config, 'a2_includes_mcp', False):
        F_A2_vec = T_A2 * (delta_MCP + f_A2_pip * delta_PIP)
    else:
        F_A2_vec = T_A2 * (f_A2_pip * delta_PIP)
    F_A2_mag = float(np.linalg.norm(F_A2_vec))
    
    # Only FDP passes under A4 (FDS inserts onto MP)
    T_A4 = F_FDP * np.exp(Config.mu_tendon * (theta_A2 + theta_A4))
    # A4 restrains proximal PIP bowstringing
    F_A4_vec = T_A4 * (f_A4_pip * delta_PIP)
    F_A4_mag = float(np.linalg.norm(F_A4_vec))

    # A1 pulley restrains MCP tendon entry at the volar plate / metacarpal head
    u_PP_hat = kin['R_MCP'] @ np.array([1., 0., 0.])
    u_wrist_vec = kin['p_MCP'] - Config.wrist_pos
    u_wrist_hat = u_wrist_vec / (np.linalg.norm(u_wrist_vec) + 1e-9)
    theta_MCP_val = np.arccos(np.clip(np.dot(u_wrist_hat, u_PP_hat), -1.0, 1.0))
    T_A1 = (F_FDP + F_FDS) * np.exp(Config.mu_tendon * theta_MCP_val)
    F_A1_vec = T_A1 * delta_MCP
    F_A1_mag = float(np.linalg.norm(F_A1_vec))

    # Decompose into anatomical local coordinate frames:
    # A2 resides on Proximal Phalanx (frame R_MCP)
    F_A2_local = kin['R_MCP'].T @ F_A2_vec
    # A1 resides at MCP / Volar Plate entry (frame R_MCP)
    F_A1_local = kin['R_MCP'].T @ F_A1_vec
    # A4 resides on Middle Phalanx (frame R_PIP)
    F_A4_local = kin['R_PIP'].T @ F_A4_vec

    F_A2_lat_local = abs(float(F_A2_local[2]))
    F_A1_lat_local = abs(float(F_A1_local[2]))
    F_A4_lat_local = abs(float(F_A4_local[2]))

    # Global wall-plane z-projections (for wall-plane / shear reference)
    F_A2_lat_global = abs(float(F_A2_vec[2]))
    F_A1_lat_global = abs(float(F_A1_vec[2]))
    F_A4_lat_global = abs(float(F_A4_vec[2]))

    # Pressure distribution mappings
    P_A2_MPa = F_A2_mag / (Config.L_A2_mm * Config.w_tendon_mm)
    P_A4_MPa = F_A4_mag / (Config.L_A4_mm * Config.w_tendon_mm)
    P_A1_MPa = F_A1_mag / (Config.L_A2_mm * Config.w_tendon_mm)

    return dict(
        p_A2=p_A2, theta_A2=theta_A2, theta_A4=theta_A4,
        F_A2_vec=F_A2_vec, F_A2_mag=F_A2_mag, F_A2_lat=F_A2_lat_local,
        F_A2_lat_local=F_A2_lat_local, F_A2_lat_global=F_A2_lat_global,
        P_A2_MPa=P_A2_MPa,
        p_A4=p_A4, F_A4_vec=F_A4_vec, F_A4_mag=F_A4_mag, F_A4_lat=F_A4_lat_local,
        F_A4_lat_local=F_A4_lat_local, F_A4_lat_global=F_A4_lat_global,
        P_A4_MPa=P_A4_MPa,
        F_A1_vec=F_A1_vec, F_A1_mag=F_A1_mag, F_A1_lat=F_A1_lat_local,
        F_A1_lat_local=F_A1_lat_local, F_A1_lat_global=F_A1_lat_global,
        P_A1_MPa=P_A1_MPa,
    )


# ─────────────────────────────────────────────────────────────
#  6-DOF JOINT REACTIONS  (Newton-Euler, distal to proximal)
# ─────────────────────────────────────────────────────────────

def joint_reactions_3d(res, F_ext, geom):
    kin   = res['kin']
    F_FDP = res['F_FDP']; F_FDS = res['F_FDS']; F_LU = res['F_LU']
    ex    = np.array([1.,0.,0.])
    pulley = pulley_forces_3d(F_FDP, F_FDS, kin, geom)

    t_FDP_DP = -(kin['R_DIP'] @ ex)
    t_FDS_MP = -(kin['R_PIP'] @ ex)
    t_LU_PP  = -(kin['R_MCP'] @ ex)

    R_DIP_vec = -(F_ext + F_FDP*t_FDP_DP)
    R_PIP_vec = -(-R_DIP_vec + pulley['F_A4_vec'] + F_FDS*t_FDS_MP)
    R_MCP_vec = -(-R_PIP_vec + pulley['F_A2_vec'] + F_LU*t_LU_PP)

    def decompose(F_vec, R_local):
        ey = np.array([0.,1.,0.]); ez = np.array([0.,0.,1.])
        return dict(
            compressive = float(-np.dot(F_vec, R_local @ ex)),
            shear_AP    = abs(float(np.dot(F_vec, R_local @ ey))),
            shear_ML    = abs(float(np.dot(F_vec, R_local @ ez))),
            magnitude   = float(np.linalg.norm(F_vec)),
        )

    return dict(
        DIP=decompose(R_DIP_vec, kin['R_DIP']),
        PIP=decompose(R_PIP_vec, kin['R_PIP']),
        MCP=decompose(R_MCP_vec, kin['R_MCP']),
        pulley=pulley,
    )


# ─────────────────────────────────────────────────────────────
#  MAIN SIMULATION
# ─────────────────────────────────────────────────────────────

def run_simulation():
    F_tip  = (Config.custom_force_N if Config.custom_force_N
              else Config.body_weight_kg * 9.81 * Config.bw_fraction)
    F_ext  = np.array([F_tip, 0.0, Config.F_lateral_N])

    geom_std   = FingerGeometry(Config.PP_mm, Config.MP_mm, Config.DP_mm, "Standard")
    geom_short = geom_std.scaled(Config.scale_short, "Short")
    geom_long  = geom_std.scaled(Config.scale_long,  "Long")
    geoms  = [geom_short, geom_std, geom_long]
    gcols  = ['#2196F3', '#4CAF50', '#F44336']
    METHODS = list(METHOD_LABELS.keys())

    # Pre-compute all results with the configured default contact geometry (e.g. 10mm hold)
    # Crucially: use beta_wall=0.0 (vertical wall) for these static baseline evaluations!
    # If evaluated at beta=45 (overhang), static open hand / pinch grips mathematically 
    # fail equilibrium (negative MCP moments) and clip to 0 force. The equilibrium 
    # posture finder in Fig 7 & 8 handles overhangs by curling the finger into crimps.
    ct_base = ContactGeometry(d_hold=Config.d_hold_mm, r_edge=Config.r_edge_mm, 
                              t_DP=Config.t_DP_mm, beta_wall=0.0)
    all_res = {k: [solve_all_methods(GRIPS[k], g, F_ext, contact=ct_base) for g in geoms] for k in GRIPS}
    jreact  = {k: joint_reactions_3d(all_res[k][1]['emg'], F_ext, geom_std) for k in GRIPS}

    # ════════════════════════════════════════════════════════════
    # FIG 1 — 3D Finger Schematics
    # ════════════════════════════════════════════════════════════
    fig1 = plt.figure(figsize=(20, 6))
    fig1.suptitle(f"3D Finger Kinematics  |  Load: {F_tip:.1f} N", fontsize=13, fontweight='bold')
    for col, (key, grip) in enumerate(GRIPS.items()):
        ax  = fig1.add_subplot(1, 4, col+1, projection='3d')
        kin = all_res[key][1]['emg']['kin']
        pts = [kin['p_MCP'], kin['p_PIP'], kin['p_DIP'], kin['p_TIP']]
        xs  = [p[0] for p in pts]; ys = [p[1] for p in pts]; zs = [p[2] for p in pts]
        ax.plot(xs, zs, ys, '-', color='#5D4037', lw=6)
        for pt, lbl, jc in zip(pts, ['MCP','PIP','DIP','TIP'],
                                     ['#1565C0','#2E7D32','#E65100','#B71C1C']):
            ax.scatter(pt[0], pt[2], pt[1], color=jc, s=70, zorder=5)
            ax.text(pt[0], pt[2]+1, pt[1]+3, lbl, fontsize=7, color=jc, fontweight='bold')
        tip = kin['p_TIP']; fl = geom_std.total * 0.20
        ax.quiver(tip[0], tip[2], tip[1], fl, 0, 0, color='#B71C1C',
                  arrow_length_ratio=0.35, lw=2)
        pul = jreact[key]['pulley']
        for pp, pc, pl in [(pul['p_A2'],'#FF9800','A2'),(pul['p_A4'],'#FF5722','A4')]:
            ax.scatter(pp[0], pp[2], pp[1], color=pc, s=55, marker='D')
            ax.text(pp[0]-4, pp[2]+2, pp[1]-4, pl, fontsize=7, color=pc)
        ax.set_title(f"{grip.name}\nphiMCP={grip.phi_MCP:.0f}  "
                     f"PIP={grip.theta_PIP:.0f}  DIP={grip.theta_DIP:.0f}",
                     fontsize=9, fontweight='bold', color=grip.color)
        for lbl, fn in [('x',ax.set_xlabel),('z radial',ax.set_ylabel),('y dorsal',ax.set_zlabel)]:
            fn(f'{lbl} (mm)', fontsize=7)
        ax.tick_params(labelsize=6)
    plt.tight_layout()

    # ════════════════════════════════════════════════════════════
    # FIG 2 — Forces per grip × method × geometry
    # ════════════════════════════════════════════════════════════
    fig2, axes2 = plt.subplots(3, 3, figsize=(20, 12))
    fig2.suptitle(f"Tendon Forces — 3 Methods  |  Load: {F_tip:.1f} N\n"
                  "Standard (solid) vs Long (hatch)", fontsize=13, fontweight='bold')
    muscles = ['FDP','FDS','LU']
    mcols   = [MUSCLE_COLORS[m] for m in muscles]
    xpos    = np.arange(3)
    for col, (key, grip) in enumerate(GRIPS.items()):
        for row, mname in enumerate(METHODS):
            ax = axes2[row, col]
            vs = [all_res[key][1][mname][f'F_{m}'] for m in muscles]
            vl = [all_res[key][2][mname][f'F_{m}'] for m in muscles]
            ax.bar(xpos-0.2, vs, 0.35, color=mcols, edgecolor='k', lw=0.5, label='Std')
            ax.bar(xpos+0.2, vl, 0.35, color=mcols, edgecolor='k', lw=0.5,
                   hatch='//', alpha=0.55, label='Long')
            for i,(v,vv) in enumerate(zip(vs,vl)):
                if v>5:  ax.text(i-0.2, v+4,  f'{v:.0f}',  ha='center', fontsize=7)
                if vv>5: ax.text(i+0.2, vv+4, f'{vv:.0f}', ha='center', fontsize=7)
            ax.set_ylim(0, max(max(vs+vl)*1.45, 30))
            ax.set_xticks(xpos); ax.set_xticklabels(muscles, fontsize=9)
            ax.grid(axis='y', alpha=0.25)
            ax.set_title(f"{grip.name}\n{METHOD_LABELS[mname]}", fontsize=8,
                         fontweight='bold', color=grip.color)
            if col==0: ax.set_ylabel('Force (N)', fontsize=8)
            if col==2 and row==0: ax.legend(fontsize=7)
    plt.tight_layout()

    # ════════════════════════════════════════════════════════════
    # FIG 3 — FDP & FDS vs PIP angle
    # ════════════════════════════════════════════════════════════
    pip_range = np.linspace(10, 130, 25)
    fig3, axes3 = plt.subplots(2, 3, figsize=(20, 9))
    fig3.suptitle("FDP & FDS vs PIP Angle — 3 Methods & Finger Lengths", fontsize=13, fontweight='bold')
    for col, key in enumerate(GRIPS):
        grip = GRIPS[key]
        for row, muscle in enumerate(['FDP','FDS']):
            ax = axes3[row, col]
            for mname in METHODS:
                for geom, gc in zip(geoms, gcols):
                    vals = []
                    for pip in pip_range:
                        # Grip-specific DIP/PIP coupling (physiological)
                        # Crimp:     DIP hyperextended, fixed at -22.6°
                        # Half-crimp: DIP nearly straight, small coupling (~0.1)
                        # Open hand:  DIP follows PIP moderately (~0.5 ratio)
                        dip_map = {
                            'crimp':      -22.6,
                            'half_crimp': pip * 0.10,
                            'open_hand':  pip * 0.50,
                        }
                        dip = dip_map.get(key, pip * 0.40)
                        g = GripAngles(grip.name, grip.theta_MCP, grip.phi_MCP,
                                       pip, dip, grip.color, grip.emg_ratio)
                        r = solve_all_methods(g, geom, F_ext)
                        vals.append(max(r[mname][f'F_{muscle}'], 0.0))
                    ax.plot(pip_range, vals, ls=METHOD_STYLES[mname], color=gc, lw=1.6, alpha=0.85)
            ax.axvline(grip.theta_PIP, color='k', ls=':', lw=1.0, alpha=0.5)
            ax.set_title(f"{grip.name} — {muscle}", fontsize=9, fontweight='bold', color=grip.color)
            ax.set_xlabel('PIP Flexion (deg)', fontsize=8)
            if col==0: ax.set_ylabel(f'{muscle} Force (N)', fontsize=8)
            ax.set_xlim(10,130); ax.set_ylim(bottom=0); ax.grid(True, alpha=0.2)
    m_patches = [mpatches.Patch(color=METHOD_COLORS[m], label=METHOD_LABELS[m]) for m in METHODS]
    g_lines   = [plt.Line2D([0],[0], color=c, lw=2, label=g.name) for c,g in zip(gcols,geoms)]
    fig3.legend(handles=m_patches+g_lines, loc='lower center', ncol=6,
                fontsize=8, bbox_to_anchor=(0.5,0.0))
    plt.tight_layout(rect=[0, 0.05, 1, 1])

    # ════════════════════════════════════════════════════════════
    # FIG 4 — Pulley Forces vs MCP Abduction
    # ════════════════════════════════════════════════════════════
    phi_range = np.linspace(0, 20, 20)
    fig4, axes4 = plt.subplots(2, 3, figsize=(20, 9))
    fig4.suptitle("3D Pulley Force vs MCP Abduction — Out-of-plane Component\n"
                  "Dashed = lateral (z) force; unique to 3D model",
                  fontsize=13, fontweight='bold')
    for col, key in enumerate(GRIPS):
        grip = GRIPS[key]
        for row, (pname, pk, pkl) in enumerate([('A2','F_A2_mag','F_A2_lat'),
                                                  ('A4','F_A4_mag','F_A4_lat')]):
            ax = axes4[row, col]
            for geom, gc, gl in zip(geoms, gcols, ['Short','Std','Long']):
                mags, lats = [], []
                for phi in phi_range:
                    g  = GripAngles(grip.name, grip.theta_MCP, phi, grip.theta_PIP,
                                    grip.theta_DIP, grip.color, grip.emg_ratio)
                    r  = solve_all_methods(g, geom, F_ext)['emg']
                    pf = pulley_forces_3d(r['F_FDP'], r['F_FDS'], r['kin'], geom)
                    mags.append(pf[pk]); lats.append(pf[pkl])
                ax.plot(phi_range, mags, '-',  color=gc, lw=2.0, label=f'{gl} total')
                ax.plot(phi_range, lats, '--', color=gc, lw=1.4, alpha=0.7, label=f'{gl} lateral')
            ax.axvline(grip.phi_MCP, color='k', ls=':', lw=1.0, alpha=0.5)
            ax.set_title(f"{grip.name} — {pname}", fontsize=9, fontweight='bold', color=grip.color)
            ax.set_xlabel('MCP Abduction (deg)', fontsize=8)
            if col==0: ax.set_ylabel(f'{pname} Force (N)', fontsize=8)
            ax.set_ylim(bottom=0); ax.grid(True, alpha=0.2)
            if col==2 and row==0: ax.legend(fontsize=7)
    plt.tight_layout()

    # ════════════════════════════════════════════════════════════
    # FIG 5 — 6-DOF Joint Reactions
    # ════════════════════════════════════════════════════════════
    fig5, axes5 = plt.subplots(2, 3, figsize=(20, 10))
    fig5.suptitle("6-DOF Joint Reaction Forces — Injury Risk Assessment\n"
                  "Standard finger, EMG method  |  Shear_ML = mediolateral (3D only)",
                  fontsize=13, fontweight='bold')
    joints = ['DIP','PIP','MCP']
    jcols  = ['#E65100','#2E7D32','#1565C0']
    for col, (key, grip) in enumerate(GRIPS.items()):
        jr = jreact[key]
        for row, (ckey, clabel) in enumerate([('compressive','Compression (N)'),
                                               ('shear_ML','ML Shear (N) — 3D')]):
            ax = axes5[row, col]
            vals = [abs(jr[j][ckey]) for j in joints]
            bars = ax.bar(joints, vals, color=jcols, edgecolor='k', lw=0.5, alpha=0.85)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+2,
                        f'{v:.0f}', ha='center', va='bottom', fontsize=8)
            ax.set_title(f"{grip.name}\n{clabel}", fontsize=9, fontweight='bold', color=grip.color)
            ax.set_ylim(0, max(vals)*1.4+20); ax.grid(axis='y', alpha=0.25)
            if col==0: ax.set_ylabel('Force (N)', fontsize=8)
    plt.tight_layout()

    # ════════════════════════════════════════════════════════════
    # FIG 6 — Long Finger Disadvantage Summary
    # ════════════════════════════════════════════════════════════
    fig6, axes6 = plt.subplots(1, 3, figsize=(18, 6))
    fig6.suptitle("Long Finger Disadvantage — 3D Model Summary", fontsize=13, fontweight='bold')
    xpos = np.arange(len(GRIPS)); w = 0.28

    # 6a — FDP/FDS ratio
    ax = axes6[0]
    for i,(geom,gc,gl) in enumerate(zip(geoms,gcols,['Short','Std','Long'])):
        ratios = [all_res[k][i]['emg']['ratio'] for k in GRIPS]
        ratios = [r if not np.isinf(r) else 0.0 for r in ratios]
        ax.bar(xpos+(i-1)*w, ratios, w, color=gc, edgecolor='k', lw=0.4, label=gl)
    ax.axhline(1.75, color='#E53935', ls='--', lw=1.2, label='Vigouroux crimp 1.75')
    ax.axhline(0.88, color='#43A047', ls='--', lw=1.2, label='Vigouroux slope 0.88')
    ax.set_xticks(xpos); ax.set_xticklabels([GRIPS[k].name for k in GRIPS], rotation=20, ha='right', fontsize=9)
    ax.set_ylabel('FDP/FDS'); ax.set_title('FDP:FDS Ratio (EMG method)', fontsize=10)
    ax.legend(fontsize=7); ax.grid(axis='y', alpha=0.25)

    # 6b — % change long vs short
    ax = axes6[1]
    for i,(mname,mc) in enumerate(METHOD_COLORS.items()):
        pcts = [(all_res[k][2][mname]['F_total']-all_res[k][0][mname]['F_total'])
                /max(all_res[k][0][mname]['F_total'], 1.0)*100 for k in GRIPS]
        ax.bar(xpos+(i-1)*w, pcts, w, color=mc, edgecolor='k', lw=0.4,
               label=METHOD_LABELS[mname], alpha=0.85)
    ax.axhline(0, color='k', lw=0.8)
    ax.set_xticks(xpos); ax.set_xticklabels([GRIPS[k].name for k in GRIPS], rotation=20, ha='right', fontsize=9)
    ax.set_ylabel('% Change'); ax.set_title('Long vs Short\n% DTotal Tendon Force', fontsize=10)
    ax.legend(fontsize=7); ax.grid(axis='y', alpha=0.25)

    # 6c — A2 pulley force
    ax = axes6[2]
    for i,(geom,gc,gl) in enumerate(zip(geoms,gcols,['Short','Std','Long'])):
        a2_vals = []
        for key in GRIPS:
            r  = solve_all_methods(GRIPS[key], geom, F_ext, contact=ct_base)['emg']
            jr = joint_reactions_3d(r, F_ext, geom)
            a2_vals.append(jr['pulley']['F_A2_mag'])
        ax.bar(xpos+(i-1)*w, a2_vals, w, color=gc, edgecolor='k', lw=0.4, label=gl)
    ax.axhline(300, color='#B71C1C', ls='--', lw=1.5, label='A2 failure ~300N')
    ax.set_xticks(xpos); ax.set_xticklabels([GRIPS[k].name for k in GRIPS], rotation=20, ha='right', fontsize=9)
    ax.set_ylabel('A2 Pulley Force (N)'); ax.set_title('A2 Pulley Load (3D)', fontsize=10)
    ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.25)
    plt.tight_layout()

    # ════════════════════════════════════════════════════════════
    # FIG 7 — Hold Depth Analysis: contact model vs TIP model
    #         The core long-finger disadvantage on small holds
    # ════════════════════════════════════════════════════════════
    d_range   = np.linspace(2.0, 45.0, 50)  # extended: covers DP + MP engagement
    fig7, axes7 = plt.subplots(2, 3, figsize=(21, 11))
    fig7.suptitle(
        "Contact-Point Model: Forces vs Hold Depth (Static Baseline Grips)\n"
        f"Wall angle: 0° (Vertical benchmark)  |  "
        f"Edge radius: {Config.r_edge_mm:.1f} mm  |  "
        f"Friction coeff: {Config.mu_friction:.2f}  |  "
        f"Load: {F_tip:.1f} N\n"
        "Solid = FDP, Dashed = FDS | Vertical lines = max grippable depth per finger\n"
        "* >170% peak represents the point where short finger FULLY engages MP (minimizing FDP), while long finger is only partially engaged.",
        fontsize=11, fontweight='bold')

    contact_default = ContactGeometry()

    for col, key in enumerate(GRIPS):
        grip = GRIPS[key]

        # Row 0: FDP/FDS force vs hold depth (EMG method, 3 geometries)
        ax0 = axes7[0, col]
        # Row 1: % force increase (long vs short) + friction utilisation
        ax1 = axes7[1, col]

        pct_fdp, pct_fds = [], []
        fric_short, fric_std, fric_long = [], [], []

        for geom, gc, gl in zip(geoms, gcols, ['Short','Std','Long']):
            fdp_vals, fds_vals, feasible_d = [], [], []
            for d in d_range:
                # Crucial: Evaluate static baseline grips on vertical wall (beta=0)
                # Overhangs (beta=45) cause static unoptimized grips to fail equilibrium!
                ct = ContactGeometry(d_hold=d, r_edge=Config.r_edge_mm,
                                     t_DP=Config.t_DP_mm, mu=Config.mu_friction,
                                     beta_wall=0.0)
                r = solve_all_methods(grip, geom, F_ext, contact=ct)['emg']
                fdp_vals.append(r['F_FDP'])
                fds_vals.append(r['F_FDS'])
                if r['friction'] and r['friction']['feasible']:
                    feasible_d.append(d)

            # Max grippable depth (last feasible d)
            d_max = max(feasible_d) if feasible_d else 0.0

            ax0.plot(d_range, fdp_vals, '-',  color=gc, lw=2.2, label=f'{gl} FDP')
            ax0.plot(d_range, fds_vals, '--', color=gc, lw=1.6, alpha=0.75, label=f'{gl} FDS')
            ax0.axvline(d_max, color=gc, ls=':', lw=1.2, alpha=0.6)

            if gl == 'Short':
                fdp_s, fds_s = np.array(fdp_vals), np.array(fds_vals)
            elif gl == 'Std':
                fdp_m, fds_m = np.array(fdp_vals), np.array(fds_vals)
            elif gl == 'Long':
                fdp_l, fds_l = np.array(fdp_vals), np.array(fds_vals)

        # % increase long vs short
        with np.errstate(divide='ignore', invalid='ignore'):
            pct_fdp = np.where(fdp_s > 1, (fdp_l - fdp_s) / fdp_s * 100, 0)
            pct_fds = np.where(fds_s > 1, (fds_l - fds_s) / fds_s * 100, 0)

        ax1.plot(d_range, pct_fdp, '-',  color='#E53935', lw=2.2, label='FDP % inc')
        ax1.plot(d_range, pct_fds, '--', color='#1565C0', lw=2.2, label='FDS % inc')
        ax1.axhline(0, color='k', lw=0.8)
        ax1.fill_between(d_range, pct_fdp, 0,
                         where=(pct_fdp > 0), alpha=0.15, color='#E53935')

        # Annotate the >170% peak in the first column
        if col == 0 and len(pct_fdp) > 0:
            max_idx = np.argmax(pct_fdp)
            d_peak = d_range[max_idx]
            pct_peak = pct_fdp[max_idx]
            # Ensure we only annotate if there's actually a substantial peak
            if pct_peak > 100:
                ax1.annotate(
                    f'Peak Difference (d~{d_peak:.1f}mm):\nShort finger FULLY engages MP,\nbottoming out its FDP force.\nLong finger MP is only half-engaged,\nstill requiring significant FDP.',
                    xy=(d_peak, pct_peak), 
                    xytext=(d_peak - 18, pct_peak - 15),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5),
                    fontsize=7, 
                    bbox=dict(boxstyle="round,pad=0.4", fc="#FFF9C4", ec="gray", alpha=0.9),
                    zorder=10
                )

        ax0.set_title(f"{grip.name}\nFDP (solid) / FDS (dash)",
                      fontsize=9, fontweight='bold', color=grip.color)
        ax0.set_xlabel('Hold Depth (mm)', fontsize=8)
        if col == 0: ax0.set_ylabel('Tendon Force (N)', fontsize=8)
        ax0.set_xlim(d_range[0], d_range[-1])
        ax0.set_ylim(bottom=0)
        ax0.grid(True, alpha=0.2)

        # Shade hold zones: small / medium / large / deep(jug)
        for ax in [ax0, ax1]:
            ax.axvspan(d_range[0], 8,  color='#FFCDD2', alpha=0.20, zorder=0)
            ax.axvspan(8,          15, color='#FFE0B2', alpha=0.20, zorder=0)
            ax.axvspan(15,         geom_std.L3, color='#C8E6C9', alpha=0.15, zorder=0)
            ax.axvspan(geom_std.L3, d_range[-1], color='#E1BEE7', alpha=0.18, zorder=0)
            ax.axvline(geom_std.L3, color='#6A1B9A', ls='--', lw=1.2, alpha=0.7)

        ax1.set_title(f"{grip.name}\nLong vs Short: % ΔForce",
                      fontsize=9, fontweight='bold', color=grip.color)
        ax1.set_xlabel('Hold Depth (mm)', fontsize=8)
        if col == 0: ax1.set_ylabel('% Force Increase\nLong vs Short', fontsize=8)
        ax1.set_xlim(d_range[0], d_range[-1])
        ax1.grid(True, alpha=0.2)
        if col == 3: ax1.legend(fontsize=7)

    # Shared legend for Row 0
    g_lines = [plt.Line2D([0],[0], color=c, lw=2, label=gl)
               for c, gl in zip(gcols, ['Short','Std','Long'])]
    zone_p  = [mpatches.Patch(color='#FFCDD2', alpha=0.5, label='Small hold (<8mm)'),
               mpatches.Patch(color='#FFE0B2', alpha=0.5, label='Medium (8-15mm)'),
               mpatches.Patch(color='#C8E6C9', alpha=0.5, label='Large (15-22mm)'),
               mpatches.Patch(color='#E1BEE7', alpha=0.5, label='Deep/Jug (>22mm — MP engaged)')]
    axes7[0, 2].legend(handles=g_lines + zone_p, fontsize=7, loc='upper right')
    plt.tight_layout()

    # ════════════════════════════════════════════════════════════
    # FIG 8-10: grip depth sweeps — built by plot_grip_depth_sweep() below
    # ════════════════════════════════════════════════════════════

    # ════════════════════════════════════════════════════════════
    # FIG 8-10: Deep Hold Depth Sweep per grip type
    # ════════════════════════════════════════════════════════════

    def plot_grip_depth_sweep(grip_key: str, d_max: float, fig_label: str):
        """
        4-panel depth-sweep figure for one grip type:
          A) FDP vs FDS forces
          B) FDP:FDS ratio
          C) A2 pulley load
          D) Equilibrium PIP and DIP angles (posture panel)
        """
        base_grip = GRIPS[grip_key]
        d_sw      = np.linspace(2.0, d_max, 60)
        L_DP_ref  = geom_std.L3

        fig, axes = plt.subplots(4, 1, figsize=(14, 20), sharex=True)
        fig.suptitle(
            f"Deep Hold Biomechanics: Equilibrium Posture Sweep ({fig_label})\n"
            "EMG-constrained solver  |  frac_DP-based EMG ratio  |  "
            f"Load: {F_tip:.1f} N\n"
            "Shading: Short (\u221215%) / Std / Long (+15%) finger phenotype\n"
            "Refs: Crowninshield & Brand 1981; Vigouroux 2006; Schweizer 2001",
            fontsize=12, fontweight='bold')

        ax_f, ax_r, ax_p, ax_ang = axes

        for geom, gc, gl in zip(geoms, gcols, ['Short (\u221215%)', 'Standard', 'Long (+15%)']):
            fdp_eq, fds_eq, a2_eq, mcp_eq, pip_eq, dip_eq = [], [], [], [], [], []
            prev_mcp, prev_pip, prev_dip = None, None, None
            for d in d_sw:
                ct = ContactGeometry(d_hold=d, r_edge=Config.r_edge_mm,
                                     t_DP=Config.t_DP_mm, mu=Config.mu_friction,
                                     beta_wall=Config.beta_wall_deg)
                eq_g = find_equilibrium_posture(base_grip, geom, F_ext, ct,
                                                mcp0=prev_mcp, pip0=prev_pip, dip0=prev_dip)
                prev_mcp, prev_pip, prev_dip = eq_g.theta_MCP, eq_g.theta_PIP, eq_g.theta_DIP
                r  = solve_all_methods(eq_g, geom, F_ext, contact=ct)['emg']
                pf = pulley_forces_3d(r['F_FDP'], r['F_FDS'], r['kin'], geom)
                fdp_eq.append(r['F_FDP'])
                fds_eq.append(r['F_FDS'])
                a2_eq.append(pf['F_A2_mag'])
                mcp_eq.append(eq_g.theta_MCP)
                pip_eq.append(eq_g.theta_PIP)
                dip_eq.append(eq_g.theta_DIP)

            fdp_arr = np.array(fdp_eq)
            fds_arr = np.array(fds_eq)

            # Panel A: forces
            ax_f.plot(d_sw, fdp_arr, '-',  color=gc, lw=2.5, label=f'{gl} FDP')
            ax_f.plot(d_sw, fds_arr, '--', color=gc, lw=2.0, alpha=0.85, label=f'{gl} FDS')
            cross_idx = np.where(np.diff(np.sign(fdp_arr - fds_arr)))[0]
            for ci in cross_idx:
                d_cross = 0.5 * (d_sw[ci] + d_sw[ci+1])
                ax_f.axvline(d_cross, color=gc, ls=':', lw=1.0, alpha=0.55)
                ax_f.annotate(f'{gl[:3]}\n{d_cross:.1f}mm',
                               xy=(d_cross, 0.5*(fdp_arr[ci]+fds_arr[ci])),
                               fontsize=7, color=gc, ha='center')

            # Panel B: ratio
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio_arr = np.where(fds_arr > 1.0, fdp_arr / fds_arr, np.nan)
            ax_r.plot(d_sw, ratio_arr, '-', color=gc, lw=2.5, label=gl)

            # Panel C: A2
            ax_p.plot(d_sw, a2_eq, '-', color=gc, lw=2.5, label=gl)

            # Panel D: posture
            ax_ang.plot(d_sw, mcp_eq, ':',  color=gc, lw=1.8, alpha=0.85, label=f'{gl} MCP')
            ax_ang.plot(d_sw, pip_eq, '-',  color=gc, lw=2.5, label=f'{gl} PIP')
            ax_ang.plot(d_sw, dip_eq, '--', color=gc, lw=2.0, alpha=0.85, label=f'{gl} DIP')

        # Reference lines
        ax_r.axhline(1.75, color='#E53935', ls='--', lw=1.5, alpha=0.7, label='Crimp 1.75 (Vigouroux)')
        ax_r.axhline(1.20, color='#FB8C00', ls='--', lw=1.2, alpha=0.7, label='Half-crimp 1.20')
        ax_r.axhline(0.88, color='#43A047', ls='--', lw=1.5, alpha=0.7, label='Open-hand 0.88')
        ax_r.axhline(1.00, color='k',       ls=':',  lw=1.0, alpha=0.6, label='FDP = FDS')
        ax_p.axhline(300,  color='#B71C1C', ls='--', lw=2.0, label='A2 failure ~300N (Schweizer 2001)')
        ax_ang.axhline(0, color='k', ls=':', lw=0.8, alpha=0.4)

        # Depth zone shading — all 4 panels
        for ax in axes:
            if d_max > L_DP_ref:
                ax.axvspan(L_DP_ref, d_max, color='#E1BEE7', alpha=0.18, zorder=0,
                           label='MP engaged' if ax is ax_f else None)
                ax.axvline(L_DP_ref, color='#6A1B9A', ls='--', lw=1.8,
                           label=f'L_DP={L_DP_ref:.0f}mm' if ax is ax_f else None)
            ax.axvspan(d_sw[0],        min(8,       d_max), color='#FFCDD2', alpha=0.15, zorder=0)
            ax.axvspan(min(8, d_max),  min(15,      d_max), color='#FFE0B2', alpha=0.15, zorder=0)
            ax.axvspan(min(15, d_max), min(L_DP_ref,d_max), color='#C8E6C9', alpha=0.12, zorder=0)
            ax.grid(True, alpha=0.25)

        ax_f.set_ylabel('Tendon Force (N)', fontsize=11)
        ax_f.set_ylim(bottom=0)
        ax_f.set_title('A) FDP (solid) vs FDS (dashed) — Min-Effort Solver', fontsize=10, fontweight='bold')
        ax_f.legend(fontsize=8, ncol=2, loc='upper right')

        ax_r.set_ylabel('FDP / FDS Ratio', fontsize=11)
        ax_r.set_ylim(0, 3.0)
        ax_r.set_title('B) FDP:FDS Ratio — frac_DP-based EMG interpolation', fontsize=10, fontweight='bold')
        ax_r.legend(fontsize=8, ncol=2, loc='upper right')

        ax_p.set_ylabel('A2 Pulley Force (N)', fontsize=11)
        ax_p.set_ylim(bottom=0)
        ax_p.set_title('C) A2 Pulley Load — Injury Risk', fontsize=10, fontweight='bold')
        ax_p.legend(fontsize=8, ncol=2, loc='upper right')

        ax_ang.set_ylabel('Joint Angle (deg)', fontsize=11)
        ax_ang.set_title('D) Equilibrium Posture: PIP (solid), DIP (dashed) & MCP (dotted)', fontsize=10, fontweight='bold')
        ax_ang.set_xlabel('Hold Depth d_hold (mm)', fontsize=11)
        ax_ang.legend(fontsize=8, ncol=2, loc='upper right')

        grip_notes = {
            'open_hand': (
                'Short (\u221215%):\n'
                '  \u2192 Earlier FDS dominance\n'
                '  \u2192 Lower A2 stress at depth\n'
                'Long (+15%):\n'
                '  \u2192 Higher FDP demand\n'
                '  \u2192 Greater A2 risk'),
            'crimp': (
                'Short (\u221215%):\n'
                '  \u2192 frac_DP drops sooner\n'
                '  \u2192 Lower FDP at mid-depth\n'
                'Long (+15%):\n'
                '  \u2192 Higher FDP & A2 load\n'
                '  \u2192 Greater EDC cost'),
            'half_crimp': (
                'Short (\u221215%):\n'
                '  \u2192 Lower total load\n'
                '  \u2192 Earlier FDS balance\n'
                'Long (+15%):\n'
                '  \u2192 Higher FDP demand\n'
                '  \u2192 FDS crossover sooner'),
        }
        ax_r.text(0.72 * d_max, 2.9, grip_notes.get(grip_key, ''),
                  fontsize=8, verticalalignment='top',
                  bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFF9C4',
                            alpha=0.85, edgecolor='#F57F17'))
        plt.tight_layout()
        return fig

    fig8  = plot_grip_depth_sweep('open_hand',  d_max=45.0, fig_label='Open Hand')
    fig9  = plot_grip_depth_sweep('crimp',      d_max=30.0, fig_label='Full Crimp')
    fig10 = plot_grip_depth_sweep('half_crimp', d_max=35.0, fig_label='Half Crimp')

    # ════════════════════════════════════════════════════════════
    # FIG 11 — Grip-Optimal Comparison
    #   For each hold depth × phenotype: which grip minimises
    #   total tendon load? The "efficient grip frontier."
    # ════════════════════════════════════════════════════════════
    grip_keys  = ['crimp', 'half_crimp', 'open_hand']
    grip_label = {'crimp': 'Crimp', 'half_crimp': 'Half-Crimp', 'open_hand': 'Open Hand'}
    grip_color = {'crimp': '#E53935', 'half_crimp': '#FB8C00', 'open_hand': '#43A047'}
    d_cmp = np.linspace(2.0, 40.0, 50)

    fig11, axes11 = plt.subplots(len(geoms), 2, figsize=(16, 14), sharex=True)
    fig11.suptitle(
        "Grip-Optimal Comparison: Which Grip Minimises Total Tendon Load?\n"
        f"EMG-constrained min-effort solver  |  Load: {F_tip:.1f} N\n"
        "Left: total tendon force by grip  |  Right: optimal grip at each depth",
        fontsize=12, fontweight='bold')

    for row, (geom, gcol, glabel) in enumerate(zip(geoms, gcols,
                                                    ['Short (\u221215%)', 'Standard', 'Long (+15%)'])):
        ax_tot = axes11[row, 0]
        ax_opt = axes11[row, 1]
        ax_tot.set_title(f'{glabel} — Total Tendon Force by Grip', fontsize=10, fontweight='bold')
        ax_opt.set_title(f'{glabel} — Optimal Grip (min force)', fontsize=10, fontweight='bold')

        min_force_arr   = np.full(len(d_cmp), np.inf)
        opt_grip_arr    = [''] * len(d_cmp)
        all_forces      = {}

        for gk in grip_keys:
            bg = GRIPS[gk]
            forces, prev_mcp, prev_pip, prev_dip = [], None, None, None
            for d in d_cmp:
                ct = ContactGeometry(d_hold=d, r_edge=Config.r_edge_mm,
                                     t_DP=Config.t_DP_mm, mu=Config.mu_friction,
                                     beta_wall=Config.beta_wall_deg)
                eq_g = find_equilibrium_posture(bg, geom, F_ext, ct,
                                                mcp0=prev_mcp, pip0=prev_pip, dip0=prev_dip)
                prev_mcp, prev_pip, prev_dip = eq_g.theta_MCP, eq_g.theta_PIP, eq_g.theta_DIP
                r = solve_all_methods(eq_g, geom, F_ext, contact=ct)['emg']
                forces.append(r['F_total'])
            all_forces[gk] = np.array(forces)
            ax_tot.plot(d_cmp, all_forces[gk], '-', color=grip_color[gk],
                        lw=2.5, label=grip_label[gk])

        # Determine optimal grip at each depth
        for i, d in enumerate(d_cmp):
            best_gk = min(grip_keys, key=lambda gk: all_forces[gk][i])
            min_force_arr[i]  = all_forces[best_gk][i]
            opt_grip_arr[i]   = best_gk

        # Plot optimal region as colored background bands
        prev_gk, seg_start = opt_grip_arr[0], 0
        transition_depths = []   # (depth, from_grip, to_grip) for annotation
        for i in range(1, len(d_cmp) + 1):
            cur_gk = opt_grip_arr[i] if i < len(d_cmp) else None
            if cur_gk != prev_gk:
                d_seg_end = d_cmp[i-1]
                ax_opt.axvspan(d_cmp[seg_start], d_seg_end,
                               color=grip_color[prev_gk], alpha=0.30,
                               label=grip_label[prev_gk] if seg_start == 0 or
                               opt_grip_arr[seg_start-1] != prev_gk else '')
                # Record transition point
                if cur_gk is not None:
                    d_trans = 0.5 * (d_cmp[i-1] + d_cmp[i]) if i < len(d_cmp) else d_cmp[-1]
                    transition_depths.append((d_trans, prev_gk, cur_gk))
                seg_start = i
                prev_gk = cur_gk

        ax_opt.plot(d_cmp, min_force_arr, 'k-', lw=2.5, label='Min force (optimal grip)')
        for gk in grip_keys:
            ax_opt.plot(d_cmp, all_forces[gk], '--', color=grip_color[gk],
                        lw=1.2, alpha=0.55, label=grip_label[gk])

        # Annotate grip transitions with vertical lines + text
        for (d_tr, gk_from, gk_to) in transition_depths:
            ax_opt.axvline(d_tr, color='#333333', ls=':', lw=1.5, alpha=0.7)
            y_ann = ax_opt.get_ylim()[1] if ax_opt.get_ylim()[1] > 0 else 600
            ax_opt.annotate(
                f'{grip_label[gk_from][:1]}\u2192{grip_label[gk_to][:1]}\n{d_tr:.0f}mm',
                xy=(d_tr, min_force_arr[np.argmin(np.abs(d_cmp - d_tr))]),
                xytext=(d_tr + 0.8, min_force_arr[np.argmin(np.abs(d_cmp - d_tr))] * 0.85),
                fontsize=8, color='#333333', ha='left',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7,
                          edgecolor='#333333', lw=0.8))

        # Phenotype-specific L_DP line (NOT geom_std — each row gets its own geom)
        L_DP_this = geom.L3
        for ax in [ax_tot, ax_opt]:
            ax.set_ylabel('Total Tendon Force (N)', fontsize=10)
            ax.set_ylim(bottom=0)
            ax.grid(True, alpha=0.25)
            if d_cmp[-1] > L_DP_this:
                ax.axvline(L_DP_this, color='#6A1B9A', ls='--', lw=1.5,
                           label=f'L_DP={L_DP_this:.0f}mm ({glabel})')
            ax.legend(fontsize=8, loc='upper right')

    for col in range(2):
        axes11[-1, col].set_xlabel('Hold Depth d_hold (mm)', fontsize=11)
    plt.tight_layout()

    return (fig1, fig2, fig3, fig4, fig5, fig6, fig7,
            fig8, fig9, fig10, fig11), all_res, jreact, geoms, F_tip



# ─────────────────────────────────────────────────────────────
#  CONSOLE SUMMARY
# ─────────────────────────────────────────────────────────────

def print_summary(all_res, jreact, F_tip):
    W = 125 if getattr(Config, 'use_interossei', False) else 110
    print('\n' + '='*W)
    print('  3D CLIMBING FINGER BIOMECHANICS  (Standard geometry)')
    print(f'  Load: {F_tip:.1f} N')
    print('='*W)
    if getattr(Config, 'use_interossei', False):
        print(f"{'Grip':<13} {'Method':<25} {'F_FDP':>8} {'F_FDS':>8} {'F_LU':>7} {'F_EDC':>7} {'F_RI':>7} {'F_UI':>7} "
              f"{'Total':>8} {'Ratio':>7} {'A2(MPa)':>8}")
    else:
        print(f"{'Grip':<13} {'Method':<25} {'F_FDP':>8} {'F_FDS':>8} {'F_LU':>7} {'F_EDC':>7} "
              f"{'Total':>8} {'Ratio':>7} {'A2(MPa)':>8}")
    print('-'*W)
    for key in GRIPS:
        for mname in ['direct', 'emg', 'lu_min']:
            r   = all_res[key][1][mname]
            jr  = jreact[key]
            rat = f"{r['ratio']:.2f}" if not np.isinf(r['ratio']) else 'inf '
            note = ''
            if key == 'crimp' and mname == 'direct' and r['F_FDS'] < 50.0:
                note = '  ⚠ DIP hyperext — use EMG'
            if getattr(Config, 'use_interossei', False):
                print(f"{GRIPS[key].name:<13} {METHOD_LABELS[mname]:<25} "
                      f"{r['F_FDP']:>8.1f} {r['F_FDS']:>8.1f} {r['F_LU']:>7.1f} {r['F_EDC']:>7.1f} "
                      f"{r['F_RI']:>7.1f} {r['F_UI']:>7.1f} "
                      f"{r['F_total']:>8.1f} {rat:>7} {jr['pulley']['P_A2_MPa']:>8.1f}{note}")
            else:
                print(f"{GRIPS[key].name:<13} {METHOD_LABELS[mname]:<25} "
                      f"{r['F_FDP']:>8.1f} {r['F_FDS']:>8.1f} {r['F_LU']:>7.1f} {r['F_EDC']:>7.1f} "
                      f"{r['F_total']:>8.1f} {rat:>7} {jr['pulley']['P_A2_MPa']:>8.1f}{note}")
        print()
    print('  Notes: Direct (3×3) crimp: DIP hyperextension makes FDS near-zero (artifact).')
    print('         EMG method (Vigouroux 2006) is the physiological reference.')
    print('  References:')
    print('    Vigouroux et al. 2006 J Biomechanics 39:2583  | crimp FDP/FDS=1.75 | slope=0.88')
    print('    Schweizer 2001 J Biomechanics 34:217          | A2 pulley failure ~300-400 N')
    print('    An KN et al. 1983 J Biomechanics 16:639       | moment arm data')
    print('='*W)


# ─────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print('='*65)
    print('  3D CLIMBING FINGER BIOMECHANICS SIMULATOR')
    print('  Vigouroux 2006 | An 1983 | Brand & Hollister 1999')
    print('  Crowninshield & Brand 1981 | Schweizer 2001')
    print('='*65)
    F = Config.body_weight_kg * 9.81 * Config.bw_fraction
    print(f'\nGeometry: PP={Config.PP_mm}mm  MP={Config.MP_mm}mm  DP={Config.DP_mm}mm')
    print(f'Load: {Config.body_weight_kg}kg x 9.81 x {Config.bw_fraction} = {F:.1f} N\n')

    (figs), all_res, jreact, geoms, F_tip = run_simulation()
    print_summary(all_res, jreact, F_tip)

    if Config.save_figures:
        for i, fig in enumerate(figs, 1):
            fname = f'outputs/{Config.output_prefix}_fig{i}.png'
            fig.savefig(fname, dpi=150, bbox_inches='tight')
            print(f'  Saved -> {fname}')
    
    # Show figures only if configured and in an interactive terminal
    import sys
    if getattr(Config, 'show_figures', False) and sys.stdout.isatty():
        plt.show()
    else:
        print("  Figures successfully saved to disk (headless mode).")
    print('\nDone.')
