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
        'pinch':      1.50,
    }

    wrist_pos     = np.array([-50.0, 0.0, 0.0])   # mm

    # ── Contact geometry defaults ─────────────────────────────
    # DP palmar-dorsal thickness: Butz et al. 2012 (adult male middle finger)
    t_DP_mm       = 9.0    # mm
    # Skin-on-rock friction coefficient: Quaine et al. 2000 (0.4-0.8 range)
    mu_friction   = 0.5
    # Wall overhang angle from vertical: 0 = vertical wall, 90 = horizontal roof
    beta_wall_deg = 45.0
    # Hold edge radius: 0.5 = sharp crimp, 2-5 = rounded lip
    r_edge_mm     = 2.0
    # Hold depth (mm): how far the DP palmar surface is engaged from TIP toward DIP
    d_hold_mm     = 10.0

    save_figures  = True
    output_prefix = "climbing_3d"


# ─────────────────────────────────────────────────────────────
#  DATA CLASSES
# ─────────────────────────────────────────────────────────────

@dataclass
class FingerGeometry:
    L1: float = Config.PP_mm
    L2: float = Config.MP_mm
    L3: float = Config.DP_mm
    name: str = "Standard"

    @property
    def total(self): return self.L1 + self.L2 + self.L3

    def scaled(self, f, label=None):
        tag = label or (f"Long" if f > 1 else "Short")
        return FingerGeometry(self.L1*f, self.L2*f, self.L3*f, tag)


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
    "crimp":      GripAngles("Crimp",      2.6,  5.0, 106.5, -22.6, "#E53935", 1.75),
    "half_crimp": GripAngles("Half-Crimp", 15.0, 5.0,  90.0,  10.0, "#FB8C00", 1.20),
    "open_hand":  GripAngles("Open Hand",  21.0, 5.0,  25.9,  38.8, "#43A047", 0.88),
    "pinch":      GripAngles("Pinch",      25.0, 10.0, 15.0,  10.0, "#8E24AA", 1.50),
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
    'direct': 'Direct (3x3 exact)',
    'emg':    'EMG ratio constrained',
    'lu_min': 'LU-minimising',
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
    p_MCP = np.zeros(3)
    p_PIP = p_MCP + R_MCP @ (geom.L1 * ex)
    p_DIP = p_PIP + R_PIP @ (geom.L2 * ex)
    p_TIP = p_DIP + R_DIP @ (geom.L3 * ex)
    return dict(p_MCP=p_MCP, p_PIP=p_PIP, p_DIP=p_DIP, p_TIP=p_TIP,
                R_MCP=R_MCP, R_PIP=R_PIP, R_DIP=R_DIP,
                flex_axis=np.array([0.,0.,1.]),
                abd_axis=R_MCP @ np.array([0.,1.,0.]))


# ─────────────────────────────────────────────────────────────
#  MOMENT ARMS  (An et al. 1983 + Brand & Hollister 1999)
# ─────────────────────────────────────────────────────────────

def moment_arms(grip):
    """Scalar moment arms (mm). Positive = flexion or radial abduction."""
    tp, td = grip.theta_PIP, grip.theta_DIP
    return dict(
        FDP_DIP=max(6.0 + 0.045*np.clip(td,-30,90),  2.0),
        FDP_PIP=max(9.0 + 0.033*np.clip(tp,  0,120), 4.0),
        FDP_MCP=10.4,
        FDS_PIP=max(7.5 + 0.020*np.clip(tp,  0,120), 3.0),
        FDS_MCP=8.6,
        LU_DIP =-4.0,   # extends DIP
        LU_PIP =-5.0,   # extends PIP
        LU_MCP = 6.0,   # flexes MCP
        FDP_abd=-2.1,   # ulnar side
        FDS_abd=-1.5,
        LU_abd = 3.5,   # radial side
    )


# ─────────────────────────────────────────────────────────────
#  CONTACT GEOMETRY & FORCE MODEL
# ─────────────────────────────────────────────────────────────

def compute_contact_point(grip, geom, contact: ContactGeometry, kin) -> tuple:
    """
    Locate contact point C on the DP palmar surface given hold depth.

    The hold edge contacts the palmar surface at arc-length d_eff from TIP.

    Rounded-edge correction:
      The edge (radius r_edge) wraps further onto the DP as the DP
      tilts away from the wall. Correction proportional to the
      out-of-horizontal component of the DP:
        d_eff = d_hold + r_edge * |e_DP · y_hat|

    Returns:
      p_C       : 3D contact point (on palmar surface, mm)
      d_eff     : effective engagement depth (mm)
      s_from_DIP: arc-length from DIP to C (mm) — key for moment arms
    """
    e_DP   = kin['R_DIP'] @ np.array([1., 0., 0.])
    n_palm = kin['R_DIP'] @ np.array([0., -1., 0.])   # toward palm (away from dorsal)
    y_hat  = np.array([0., 1., 0.])

    # Rounded-edge depth correction: edge bites deeper as DP tilts vertically
    correction = contact.r_edge * abs(float(np.dot(e_DP, y_hat)))
    d_eff = float(np.clip(contact.d_hold + correction, 0.0, geom.L3 - 0.5))

    # s_from_DIP: how far C is from the DIP joint
    s_from_DIP = geom.L3 - d_eff

    # Palmar surface at TIP, then move d_eff toward DIP
    p_TIP_palmar = kin['p_TIP'] + (contact.t_DP / 2.0) * n_palm
    p_C = p_TIP_palmar - d_eff * e_DP

    return p_C, d_eff, s_from_DIP


def contact_force_vector(F_mag: float, contact: ContactGeometry) -> np.ndarray:
    """
    Build the 3D external force vector from wall angle and lateral load.

    beta_wall = 0  -> F_ext = F_mag * x_hat   (horizontal, Vigouroux)
    beta_wall = 45 -> combined
    beta_wall = 90 -> F_ext = -F_mag * y_hat  (vertical down, roof)

    Lateral component from Config.F_lateral_N is added along z.
    """
    b = np.radians(contact.beta_wall)
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
                    friction_ratio=np.inf, F_f_axial=F_f_axial, F_f_lat=F_f_lat)

    ratio    = F_f_total / (contact.mu * F_N)
    feasible = ratio <= 1.0
    return dict(feasible=feasible, F_N=F_N, F_f_total=F_f_total,
                friction_ratio=ratio, F_f_axial=F_f_axial, F_f_lat=F_f_lat)


# ─────────────────────────────────────────────────────────────
#  EXTERNAL MOMENTS  (contact-aware)
# ─────────────────────────────────────────────────────────────

def external_moments(kin, F_ext, grip, passive_frac=None,
                     p_contact=None):
    """
    Compute external moments at each joint.

    p_contact: if provided, use as the force application point (C).
               If None, falls back to p_TIP (original behaviour).

    The change from the original model:
      OLD: M_j = (p_TIP  - p_j) x F_ext   <- force always at TIP
      NEW: M_j = (p_C    - p_j) x F_ext   <- force at contact point

    For a LONG finger on the SAME SHALLOW hold:
      p_C is at the same absolute position from TIP, but the DIP is
      further away → longer lever arm at DIP → higher FDP force required.
    """
    if passive_frac is None:
        passive_frac = Config.passive_frac
    fa, aa = kin['flex_axis'], kin['abd_axis']

    # Application point: contact point C, or TIP for backward compatibility
    p_app = p_contact if p_contact is not None else kin['p_TIP']

    def moment_at(p_joint):
        return np.cross(p_app - p_joint, F_ext)

    M_DIP = float(np.dot(moment_at(kin['p_DIP']), fa))
    M_PIP = float(np.dot(moment_at(kin['p_PIP']), fa))
    M_MCP = float(np.dot(moment_at(kin['p_MCP']), fa))
    M_abd = float(np.dot(moment_at(kin['p_MCP']), aa))

    if grip.theta_DIP < 0:
        M_DIP *= (1.0 - passive_frac)

    return dict(DIP=M_DIP, PIP=M_PIP, MCP=M_MCP, abd=M_abd)


# ─────────────────────────────────────────────────────────────
#  THREE SOLUTION METHODS  (fast, no iterative optimisation)
# ─────────────────────────────────────────────────────────────

def solve_all_methods(grip, geom, F_ext, contact: ContactGeometry = None):
    """
    Returns dict keyed by method name, each with F_FDP, F_FDS, F_LU, etc.
    All three methods are direct linear-algebra solves. Fast.

    If contact is provided: force applied at contact point C on DP palmar surface.
    If contact is None:     force applied at TIP (original behaviour, backward compat).
    """
    kin = kinematics_3d(grip, geom)
    ma  = moment_arms(grip)

    # ── Determine application point and force vector ──────────
    if contact is not None:
        # Rebuild F_ext from wall angle (direction may differ from caller's F_ext)
        F_mag = float(np.linalg.norm(F_ext[:2]))   # sagittal magnitude
        F_ext_c = contact_force_vector(F_mag, contact)
        p_C, d_eff, s_from_DIP = compute_contact_point(grip, geom, contact, kin)
        feas = check_friction_feasibility(F_ext_c, contact, kin)
    else:
        F_ext_c    = F_ext
        p_C        = None      # will use TIP inside external_moments
        d_eff      = None
        s_from_DIP = None
        feas       = None

    ext = external_moments(kin, F_ext_c, grip, p_contact=p_C)

    # 3x3 matrix for DIP/PIP/MCP flexion
    A3 = np.array([
        [ma['FDP_DIP'],  0.0,          ma['LU_DIP']],
        [ma['FDP_PIP'],  ma['FDS_PIP'], ma['LU_PIP']],
        [ma['FDP_MCP'],  ma['FDS_MCP'], ma['LU_MCP']],
    ])
    b3 = np.array([ext['DIP'], ext['PIP'], ext['MCP']])

    # ── Method 1: Direct 3x3 solve ────────────────────────────
    try:
        f1 = np.linalg.solve(A3, b3)
    except np.linalg.LinAlgError:
        f1 = np.linalg.lstsq(A3, b3, rcond=None)[0]
    f1 = np.maximum(f1, 0.0)

    # ── Method 2: EMG-constrained ─────────────────────────────
    # Fix F_FDP = r * F_FDS; solve 2x2 for [FDS, LU] from PIP + MCP rows
    r = grip.emg_ratio
    A2e = np.array([
        [r*ma['FDP_PIP'] + ma['FDS_PIP'],  ma['LU_PIP']],
        [r*ma['FDP_MCP'] + ma['FDS_MCP'],  ma['LU_MCP']],
    ])
    b2e = np.array([ext['PIP'], ext['MCP']])
    try:
        sol = np.linalg.solve(A2e, b2e)
    except np.linalg.LinAlgError:
        sol = np.linalg.lstsq(A2e, b2e, rcond=None)[0]
    F_FDS2 = max(sol[0], 0.0)
    F_LU2  = max(sol[1], 0.0)
    F_FDP2 = r * F_FDS2
    f2     = np.array([F_FDP2, F_FDS2, F_LU2])

    # ── Method 3: LU-minimising ───────────────────────────────
    # Set F_LU = 0; solve for FDS from PIP row, get FDP from ratio
    denom_PIP = r*ma['FDP_PIP'] + ma['FDS_PIP']
    denom_MCP = r*ma['FDP_MCP'] + ma['FDS_MCP']
    if denom_PIP > 0.1 and denom_MCP > 0.1:
        F_FDS3 = max(0.5*(ext['PIP']/denom_PIP + ext['MCP']/denom_MCP), 0.0)
        F_FDP3 = r * F_FDS3
        # Any DIP residual absorbed by small LU correction
        dip_resid = ext['DIP'] - F_FDP3*ma['FDP_DIP']
        F_LU3 = max(-dip_resid / abs(ma['LU_DIP']), 0.0) if abs(ma['LU_DIP']) > 0.1 else 0.0
        f3 = np.array([F_FDP3, F_FDS3, F_LU3])
    else:
        f3 = f2.copy()

    results = {}
    for mname, f in [('direct',f1), ('emg',f2), ('lu_min',f3)]:
        FDP, FDS, LU = f
        ratio = FDP/FDS if FDS > 0.1 else np.inf
        results[mname] = dict(
            F_FDP=float(FDP), F_FDS=float(FDS), F_LU=float(LU),
            F_total=float(FDP+FDS+LU),
            ratio=ratio,
            f_vec=f, kin=kin, ext=ext,
            # Contact geometry info (None if no contact model used)
            p_C=p_C, d_eff=d_eff, s_from_DIP=s_from_DIP,
            friction=feas,
        )
    return results


# ─────────────────────────────────────────────────────────────
#  3D PULLEY FORCES  (bow-string model)
# ─────────────────────────────────────────────────────────────

def pulley_forces_3d(F_FDP, F_FDS, kin, geom):
    """
    F_pulley = T * (d_hat_in + d_hat_out).
    A2: FDP + FDS. A4: FDP only.
    Out-of-plane (z) component appears when phi_MCP != 0.
    """
    ex    = np.array([1.,0.,0.])
    R_MCP = kin['R_MCP']
    R_PIP = kin['R_PIP']

    p_A2     = kin['p_MCP'] + R_MCP @ (0.40*geom.L1*ex)
    d_in_A2  = Config.wrist_pos - p_A2
    d_in_A2 /= np.linalg.norm(d_in_A2)
    d_out_A2 = R_MCP @ ex
    F_A2_vec = (F_FDP+F_FDS) * (d_in_A2 + d_out_A2)

    p_A4     = kin['p_PIP'] + R_PIP @ (0.20*geom.L2*ex)
    d_in_A4  = R_MCP @ ex
    d_out_A4 = R_PIP @ ex
    F_A4_vec = F_FDP * (d_in_A4 + d_out_A4)

    return dict(
        p_A2=p_A2, F_A2_vec=F_A2_vec,
        F_A2_mag=float(np.linalg.norm(F_A2_vec)),
        F_A2_lat=abs(float(F_A2_vec[2])),
        p_A4=p_A4, F_A4_vec=F_A4_vec,
        F_A4_mag=float(np.linalg.norm(F_A4_vec)),
        F_A4_lat=abs(float(F_A4_vec[2])),
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

    # Pre-compute all results
    all_res = {k: [solve_all_methods(GRIPS[k], g, F_ext) for g in geoms] for k in GRIPS}
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
    fig2, axes2 = plt.subplots(3, 4, figsize=(20, 12))
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
            if col==3 and row==0: ax.legend(fontsize=7)
    plt.tight_layout()

    # ════════════════════════════════════════════════════════════
    # FIG 3 — FDP & FDS vs PIP angle
    # ════════════════════════════════════════════════════════════
    pip_range = np.linspace(10, 130, 25)
    fig3, axes3 = plt.subplots(2, 4, figsize=(20, 9))
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
                        # Pinch:      DIP nearly straight (~0.15 ratio)
                        dip_map = {
                            'crimp':      -22.6,
                            'half_crimp': pip * 0.10,
                            'open_hand':  pip * 0.50,
                            'pinch':      pip * 0.15,
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
    fig4, axes4 = plt.subplots(2, 4, figsize=(20, 9))
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
            if col==3 and row==0: ax.legend(fontsize=7)
    plt.tight_layout()

    # ════════════════════════════════════════════════════════════
    # FIG 5 — 6-DOF Joint Reactions
    # ════════════════════════════════════════════════════════════
    fig5, axes5 = plt.subplots(2, 4, figsize=(20, 10))
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
            r  = solve_all_methods(GRIPS[key], geom, F_ext)['emg']
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
    d_range   = np.linspace(2.0, 22.0, 30)
    fig7, axes7 = plt.subplots(2, 4, figsize=(21, 11))
    fig7.suptitle(
        "Contact-Point Model: Forces vs Hold Depth\n"
        f"Wall angle: {Config.beta_wall_deg:.0f}°  |  "
        f"Edge radius: {Config.r_edge_mm:.1f} mm  |  "
        f"Friction coeff: {Config.mu_friction:.2f}  |  "
        f"Load: {F_tip:.1f} N\n"
        "Solid = FDP, Dashed = FDS | Vertical lines = max grippable depth per finger",
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
                ct = ContactGeometry(d_hold=d, r_edge=Config.r_edge_mm,
                                     t_DP=Config.t_DP_mm, mu=Config.mu_friction,
                                     beta_wall=Config.beta_wall_deg)
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

        ax0.set_title(f"{grip.name}\nFDP (solid) / FDS (dash)",
                      fontsize=9, fontweight='bold', color=grip.color)
        ax0.set_xlabel('Hold Depth (mm)', fontsize=8)
        if col == 0: ax0.set_ylabel('Tendon Force (N)', fontsize=8)
        ax0.set_xlim(d_range[0], d_range[-1])
        ax0.set_ylim(bottom=0)
        ax0.grid(True, alpha=0.2)

        # Shade small/medium/large hold zones
        for ax in [ax0, ax1]:
            ax.axvspan(d_range[0], 8,  color='#FFCDD2', alpha=0.20, zorder=0)
            ax.axvspan(8,          15, color='#FFE0B2', alpha=0.20, zorder=0)
            ax.axvspan(15, d_range[-1],color='#C8E6C9', alpha=0.15, zorder=0)

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
               mpatches.Patch(color='#C8E6C9', alpha=0.5, label='Large (>15mm)')]
    axes7[0, 3].legend(handles=g_lines + zone_p, fontsize=7, loc='upper right')
    plt.tight_layout()

    return (fig1, fig2, fig3, fig4, fig5, fig6, fig7), all_res, jreact, geoms, F_tip


# ─────────────────────────────────────────────────────────────
#  CONSOLE SUMMARY
# ─────────────────────────────────────────────────────────────

def print_summary(all_res, jreact, F_tip):
    W = 108
    print('\n' + '='*W)
    print('  3D CLIMBING FINGER BIOMECHANICS  (Standard geometry)')
    print(f'  Load: {F_tip:.1f} N')
    print('='*W)
    print(f"{'Grip':<13} {'Method':<12} {'F_FDP':>8} {'F_FDS':>8} {'F_LU':>7} "
          f"{'Total':>8} {'Ratio':>7} {'A2 3D':>8}")
    print('-'*W)
    for key in GRIPS:
        for mname in ['direct','emg','lu_min']:
            r   = all_res[key][1][mname]
            jr  = jreact[key]
            rat = f"{r['ratio']:.2f}" if not np.isinf(r['ratio']) else 'inf'
            print(f"{GRIPS[key].name:<13} {mname:<12} "
                  f"{r['F_FDP']:>8.1f} {r['F_FDS']:>8.1f} {r['F_LU']:>7.1f} "
                  f"{r['F_total']:>8.1f} {rat:>7} {jr['pulley']['F_A2_mag']:>8.1f}")
        print()
    print('  Vigouroux 2006: crimp FDP/FDS=1.75 | slope FDP/FDS=0.88')
    print('  Schweizer 2001: A2 pulley failure ~300-400 N')
    print('='*W)


# ─────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print('='*65)
    print('  3D CLIMBING FINGER BIOMECHANICS SIMULATOR')
    print('  Vigouroux 2006  An 1983  Brand & Hollister 1999')
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
    plt.show()
    print('\nDone.')
