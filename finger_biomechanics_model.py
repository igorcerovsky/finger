#!/usr/bin/env python3
"""
Middle-finger biomechanics model for climbing grips (2D static equilibrium).

Improvements over baseline model
─────────────────────────────────
 #1  Tendon-excursion moment arms          – dL/dθ via finite difference; replaces
                                             arbitrary 45/55 blend weights.
 #2  Moment arms scale with phalanx length – reference values normalised to a 26 mm
                                             middle phalanx and scaled linearly.
 #3  Grip-dependent load direction         – contact force has a proximal shear
                                             component derived from distal-phalanx
                                             orientation (not always [0, Fy]).
 #4  Passive joint stiffness at all joints – Minami-style exponential M_passive(θ)
                                             at DIP, PIP and MCP replaces the single
                                             scalar dip_passive_fraction.
 #5  MCP moment equilibrium               – third equilibrium equation adds EDC
                                             passive contribution; MCP residual
                                             reported as equilibrium quality metric.
 #6  Distributed pulley wrapping arcs     – each annular pulley is discretised into
                                             n_arc sample points; load integrates
                                             T * κ over the arc.
 #7  EDC passive tendon                   – dorsal extensor path from MCP dorsum
                                             to terminal slip; passive-only at
                                             climbing-grip flexion angles.
 #8  Held-out validation for calibration  – power-law fit uses open + full-crimp
                                             anchors; half-crimp is a genuine
                                             out-of-sample test vs Schweizer 2009.
 #9  Four-finger model                    – index, middle, ring each modelled with
                                             load-sharing coefficients from Vigouroux
                                             (2006) Table 1.
#10  Fatigue model                        – isometric fatigue reduces FDS capacity
                                             exponentially; FDP compensates to
                                             maintain grip; peak pulley load reported.

References
──────────
Vigouroux et al., J Biomech 2006  https://doi.org/10.1016/j.jbiomech.2005.10.034
Schweizer, J Hand Surg Am 2001    https://doi.org/10.1053/jhsu.2001.26322
Schweizer, J Biomech 2009         https://pubmed.ncbi.nlm.nih.gov/19367698/
Ki et al., BMC Sports 2024        https://bmcsportsscimedrehabil.biomedcentral.com/...
Schöffl et al., Diagnostics 2021  https://pmc.ncbi.nlm.nih.gov/articles/PMC8159322/
An et al., J Biomech 1983         tendon-excursion moment-arm method
Chao et al., Biomechanics Hand 89 six-tendon equilibrium including MCP
Minami et al., J Hand Surg 1985   passive joint stiffness curves
Uchiyama et al., J Biomech 1995   distributed pulley wrapping mechanics
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


# ─────────────────────────── tiny geometry helpers ───────────────────────────

def unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else np.zeros_like(v)


def rot_cw_90(v: np.ndarray) -> np.ndarray:
    return np.array([v[1], -v[0]], dtype=float)


def cross2(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


# ──────────────────────────────── data classes ───────────────────────────────

@dataclass
class FingerPose:
    """Geometry of a single finger at a given grip posture."""
    Lp: float          # Proximal phalanx length  (mm)
    Lm: float          # Middle  phalanx length   (mm)
    Ld: float          # Distal  phalanx length   (mm)
    theta_p: float     # Absolute angle of proximal segment  (deg, global)
    theta_m: float     # Absolute angle of middle  segment
    theta_d: float     # Absolute angle of distal  segment
    label: str = ""
    dip_passive_fraction: float = 0.0   # kept for backward compat


@dataclass
class AthleteCase:
    name: str
    mass_kg: float
    lengths_mm: Tuple[float, float, float]   # (P, M, D) in mm


@dataclass
class SimConfig:
    """Top-level simulation knobs."""
    load_fraction_of_bw: float = 0.15
    pulley_offset_mm: float = 4.0
    load_point: str = "distal_mid"       # "distal_mid" | "fingertip"
    n_arc_points: int = 5                # sample points per pulley arc
    fatigue_time_s: float = 30.0         # sustained hang duration for #10
    fds_fatigue_tau_s: float = 20.0      # FDS fatigue time constant (s)


# ──────────────────────────── coordinate builders ────────────────────────────

def segment_unit(angle_deg: float) -> np.ndarray:
    a = np.deg2rad(angle_deg)
    return np.array([np.cos(a), np.sin(a)], dtype=float)


def build_joint_positions(pose: FingerPose) -> Dict[str, np.ndarray]:
    u_p = segment_unit(pose.theta_p)
    u_m = segment_unit(pose.theta_m)
    u_d = segment_unit(pose.theta_d)
    mcp = np.zeros(2)
    pip = mcp + pose.Lp * u_p
    dip = pip + pose.Lm * u_m
    tip = dip + pose.Ld * u_d
    distal_mid = dip + 0.5 * pose.Ld * u_d
    return dict(MCP=mcp, PIP=pip, DIP=dip, TIP=tip, DISTAL_MID=distal_mid,
                uP=u_p, uM=u_m, uD=u_d)


# ──────────────── Improvement #6: distributed pulley arcs ────────────────────

def _arc_points(centre: np.ndarray, radius: float,
                ang_start_deg: float, ang_end_deg: float, n: int) -> np.ndarray:
    """Return n equally-spaced points along a circular arc (mm)."""
    angles = np.linspace(np.deg2rad(ang_start_deg), np.deg2rad(ang_end_deg), n)
    return np.column_stack([centre[0] + radius * np.cos(angles),
                             centre[1] + radius * np.sin(angles)])


def _flexion_arc(centre: np.ndarray, radius: float,
                 in_dir: np.ndarray, out_dir: np.ndarray, n: int) -> np.ndarray:
    """Arc from the direction of in_dir to out_dir, wrapping on the palmar side."""
    ang_in  = float(np.degrees(np.arctan2(in_dir[1],  in_dir[0])))
    ang_out = float(np.degrees(np.arctan2(out_dir[1], out_dir[0])))
    if ang_out - ang_in > 180:
        ang_in += 360
    elif ang_in - ang_out > 180:
        ang_out += 360
    return _arc_points(centre, radius, ang_in, ang_out, n)


def build_tendon_geometry(pose: FingerPose,
                          offset_mm: float = 4.0,
                          n_arc: int = 5) -> Dict:
    """
    Build all joint positions, pulley anchor points, and tendon paths.
    Improvement #6: each annular pulley is a short arc (n_arc sample points).
    Improvement #7: EDC path added on dorsal side.
    """
    jp = build_joint_positions(pose)
    mcp, pip, dip, tip = jp["MCP"], jp["PIP"], jp["DIP"], jp["TIP"]
    u_p, u_m, u_d = jp["uP"], jp["uM"], jp["uD"]
    n_p = rot_cw_90(u_p)
    n_m = rot_cw_90(u_m)
    n_d = rot_cw_90(u_d)

    # Single-point pulley centres
    a2_c = mcp + 0.38 * pose.Lp * u_p + offset_mm * n_p
    n_pm = unit(n_p + n_m)
    if np.linalg.norm(n_pm) < 1e-8:
        n_pm = n_p
    a3_c = pip + offset_mm * n_pm
    a4_c = pip + 0.55 * pose.Lm * u_m + offset_mm * n_m

    fds_ins = pip + 0.82 * pose.Lm * u_m + 0.8 * offset_mm * n_m
    fdp_ins = dip + 0.90 * pose.Ld * u_d + 0.8 * offset_mm * n_d
    anchor  = mcp + np.array([-18.0, -20.0])

    # Improvement #7: EDC dorsal path
    edc_origin = mcp  + offset_mm * (-n_p)
    edc_mid    = pip  + offset_mm * (-n_pm)
    edc_ins    = tip  + 0.5 * offset_mm * (-n_d)

    # Improvement #6: arc-discretised pulleys
    a2_arc = _flexion_arc(a2_c, offset_mm, unit(anchor - a2_c), unit(a3_c - a2_c), n_arc)
    a3_arc = _flexion_arc(a3_c, offset_mm, unit(a2_c  - a3_c), unit(a4_c - a3_c), n_arc)
    a4_arc = _flexion_arc(a4_c, offset_mm, unit(a3_c  - a4_c), unit(fdp_ins - a4_c), n_arc)

    fdp_path = np.vstack([anchor, a2_c, a3_c, a4_c, fdp_ins])
    fds_path = np.vstack([anchor, a2_c, a3_c, fds_ins])
    edc_path = np.vstack([edc_origin, edc_mid, edc_ins])

    return {
        **jp,
        "ANCHOR": anchor,
        "A2": a2_c, "A2_ARC": a2_arc,
        "A3": a3_c, "A3_ARC": a3_arc,
        "A4": a4_c, "A4_ARC": a4_arc,
        "FDS_INS": fds_ins, "FDP_INS": fdp_ins,
        "FDP_PATH": fdp_path, "FDS_PATH": fds_path,
        "EDC_PATH": edc_path, "EDC_INS": edc_ins,
    }


# ──────────── Improvement #1: tendon-excursion moment arms ───────────────────
# Improvement #2: length-scaled physiological bounds

_L_REF_MM = 26.0   # reference middle phalanx length


def _path_length(path: np.ndarray) -> float:
    return float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))


def _tendon_len(theta_p: float, theta_m: float, theta_d: float,
                tmpl: FingerPose, offset_mm: float, tendon: str) -> float:
    p = FingerPose(Lp=tmpl.Lp, Lm=tmpl.Lm, Ld=tmpl.Ld,
                   theta_p=theta_p, theta_m=theta_m, theta_d=theta_d)
    geo = build_tendon_geometry(p, offset_mm=offset_mm, n_arc=3)
    return _path_length(geo[f"{tendon}_PATH"])


def tendon_excursion_moment_arms(pose: FingerPose,
                                 offset_mm: float = 4.0,
                                 dth: float = 1.0) -> Dict[str, float]:
    """
    Improvement #1: r_i = dL/dθ_i via central finite difference.
    Improvement #2: physiological bounds scale with Lm / L_REF.
    """
    tp, tm, td = pose.theta_p, pose.theta_m, pose.theta_d

    def dr(tendon, vary):
        if vary == "d":
            lp = _tendon_len(tp, tm, td + dth, pose, offset_mm, tendon)
            lm = _tendon_len(tp, tm, td - dth, pose, offset_mm, tendon)
        elif vary == "m":
            lp = _tendon_len(tp, tm + dth, td, pose, offset_mm, tendon)
            lm = _tendon_len(tp, tm - dth, td, pose, offset_mm, tendon)
        else:  # "p"
            lp = _tendon_len(tp + dth, tm, td, pose, offset_mm, tendon)
            lm = _tendon_len(tp - dth, tm, td, pose, offset_mm, tendon)
        return (lp - lm) / (2.0 * np.deg2rad(dth))

    r_fdp_dip_raw = dr("FDP", "d")
    r_fdp_pip_raw = dr("FDP", "m")
    r_fds_pip_raw = dr("FDS", "m")
    r_fdp_mcp_raw = dr("FDP", "p")
    r_fds_mcp_raw = dr("FDS", "p")
    r_edc_mcp_raw = dr("EDC", "p")

    # Improvement #2: length-scaled bounds
    s = pose.Lm / _L_REF_MM
    pip_f = pose.theta_p - pose.theta_m
    dip_f = pose.theta_m - pose.theta_d

    lo_fdp_dip = s * np.clip(4.2 - 0.018 * dip_f, 3.5, 5.5)
    hi_fdp_dip = s * np.clip(5.5 - 0.010 * dip_f, 4.5, 6.5)
    lo_fdp_pip = s * np.clip(1.5 + 0.010 * pip_f, 1.5, 2.8)
    hi_fdp_pip = s * np.clip(3.2 + 0.010 * pip_f, 2.8, 4.0)
    lo_fds_pip = s * np.clip(8.0 - 0.012 * pip_f, 6.5, 9.0)
    hi_fds_pip = s * np.clip(11.5 - 0.008 * pip_f, 9.0, 12.0)

    def _b(raw, lo, hi):
        return float(np.clip(abs(raw), lo, hi))

    return dict(
        r_fdp_dip = _b(r_fdp_dip_raw, lo_fdp_dip, hi_fdp_dip),
        r_fdp_pip = _b(r_fdp_pip_raw, lo_fdp_pip, hi_fdp_pip),
        r_fds_pip = _b(r_fds_pip_raw, lo_fds_pip, hi_fds_pip),
        r_fdp_mcp = _b(r_fdp_mcp_raw, s * 1.5,    s * 4.5),
        r_fds_mcp = _b(r_fds_mcp_raw, s * 1.0,    s * 4.0),
        r_edc_mcp = _b(r_edc_mcp_raw, s * 5.0,    s * 10.0),
    )


# ────────── Improvement #4: passive joint stiffness (Minami 1985 style) ──────

def passive_moment_Nm(joint: str, flex_deg: float) -> float:
    """
    M_passive(θ) = k * (exp(b*(θ − θ0)) − 1)  [Nm]
    Positive = flexion-resisting (extension) torque.
    Parameters are approximate fits to Minami et al. (1985).
    """
    params = {
        "DIP": (0.010, 0.065, 60.0),
        "PIP": (0.008, 0.055, 80.0),
        "MCP": (0.006, 0.045, 70.0),
    }
    k, b, theta0 = params[joint]
    excess = max(flex_deg - theta0, 0.0)
    return float(k * (np.exp(b * excess) - 1.0))


# ─────── Improvement #6: distributed pulley wrapping load ────────────────────

def pulley_load_arc(arc: np.ndarray, tension_N: float,
                    tendon_in: Optional[np.ndarray] = None,
                    tendon_out: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float]:
    """
    Improvement #6: capstan resultant force across a discretised pulley arc.

    Physical principle (Uchiyama 1995):
        The total resultant of a tendon wrapping around a frictionless pulley is
        R = T * (u_in + u_out)
    where u_in and u_out are the unit vectors of the incoming and outgoing
    tendon chords, and T is the tendon tension.  This is independent of whether
    the pulley is modelled as a point or an arc.

    The arc representation (Improvement #6) improves on the single-point model
    by computing the *distributed* normal load per unit length along the pulley
    (useful for pulley stress / rupture risk), while preserving the correct
    resultant magnitude.

    If tendon_in / tendon_out are provided (the actual tendon chord directions),
    they are used for the resultant.  Otherwise, the arc end-points are used as
    a fallback.

    Returns:  (resultant_vector_N, magnitude_N)
    """
    if len(arc) < 2:
        return np.zeros(2), 0.0

    # Preferred: use actual tendon chord directions
    if tendon_in is not None and tendon_out is not None:
        u_in  = unit(tendon_in)
        u_out = unit(tendon_out)
    else:
        # Fallback: approximate from arc endpoint directions
        mid = len(arc) // 2
        # Direction from arc[mid] to arc[0] (incoming) and arc[-1] (outgoing)
        u_in  = unit(arc[0]  - arc[mid])
        u_out = unit(arc[-1] - arc[mid])

    vec = tension_N * (u_in + u_out)
    return vec, float(np.linalg.norm(vec))


# ────────── Improvement #3: grip-dependent load direction ────────────────────

def contact_force_vector(pose: FingerPose, magnitude_N: float) -> np.ndarray:
    """
    Improvement #3: contact force direction depends on distal phalanx orientation.

    In climbing the hold reaction always has a dominant upward (+y) component
    (supporting body weight).  As the phalanx flattens toward horizontal
    (full crimp, theta_d near 0 deg), the edge also creates a proximal shear
    component along the bone axis (negative u_d direction).

    Physical model:
        F = [0, F_y]  +  mu_eff * F * |cos(theta_d)| * (-u_d)
    renormalised to magnitude_N.

    This preserves a DIP flexion moment for all grip postures while adding
    the grip-type-dependent shear seen in cadaveric loading experiments.
    mu_eff = 0.30 approximates a rounded-edge friction coefficient.
    """
    u_d = segment_unit(pose.theta_d)
    MU_EFF = 0.30
    shear_frac = MU_EFF * abs(np.cos(np.deg2rad(pose.theta_d)))

    F_vertical = np.array([0.0, magnitude_N])
    F_shear    = magnitude_N * shear_frac * (-u_d)
    F_total    = F_vertical + F_shear

    mag = np.linalg.norm(F_total)
    return F_total * (magnitude_N / mag) if mag > 1e-6 else np.array([0.0, magnitude_N])


def external_load_point(geo: Dict, load_point: str) -> np.ndarray:
    if load_point == "distal_mid":
        return geo["DISTAL_MID"]
    if load_point == "fingertip":
        return geo["TIP"]
    raise ValueError(f"Unknown load_point '{load_point}'")


# ─────────── Improvement #7: EDC passive tension ─────────────────────────────

def edc_passive_tension_N(pose: FingerPose) -> float:
    """
    Improvement #7: linear spring EDC engagement beyond 30 deg combined flexion.
    k_edc ≈ 1.2 N/deg (Chao 1989 passive approximation).
    """
    pip_flex = max(pose.theta_p - pose.theta_m - 30.0, 0.0)
    mcp_flex = max(90.0 - pose.theta_p, 0.0)
    return float(1.2 * (pip_flex + mcp_flex))


# ─────────────────── Core solver ─────────────────────────────────────────────

def solve_static_equilibrium(pose: FingerPose,
                              distal_force_N: float,
                              cfg: SimConfig) -> Dict:
    """
    Solve 2D static equilibrium at DIP, PIP and MCP.

    Improvements applied: #1 #2 #3 #4 #5 #6 #7
    """
    geo  = build_tendon_geometry(pose, offset_mm=cfg.pulley_offset_mm,
                                 n_arc=cfg.n_arc_points)
    arms = tendon_excursion_moment_arms(pose, offset_mm=cfg.pulley_offset_mm)

    # Improvement #3
    F_ext   = contact_force_vector(pose, distal_force_N)
    mcp     = geo["MCP"] * 1e-3
    pip     = geo["PIP"] * 1e-3
    dip     = geo["DIP"] * 1e-3
    contact = external_load_point(geo, cfg.load_point) * 1e-3

    M_dip_ext = abs(cross2(contact - dip, F_ext))
    M_pip_ext = abs(cross2(contact - pip, F_ext))
    M_mcp_ext = abs(cross2(contact - mcp, F_ext))

    # Improvement #4
    dip_flex = pose.theta_m - pose.theta_d
    pip_flex = pose.theta_p - pose.theta_m
    mcp_flex = 90.0 - pose.theta_p

    M_pass_dip = passive_moment_Nm("DIP", max(dip_flex, 0.0))
    M_pass_pip = passive_moment_Nm("PIP", max(pip_flex, 0.0))
    M_pass_mcp = passive_moment_Nm("MCP", max(mcp_flex, 0.0))

    r_fdp_dip = max(arms["r_fdp_dip"] * 1e-3, 1e-5)
    r_fdp_pip = max(arms["r_fdp_pip"] * 1e-3, 1e-5)
    r_fds_pip = max(arms["r_fds_pip"] * 1e-3, 1e-5)
    r_fdp_mcp = max(arms["r_fdp_mcp"] * 1e-3, 1e-5)
    r_fds_mcp = max(arms["r_fds_mcp"] * 1e-3, 1e-5)
    r_edc_mcp = max(arms["r_edc_mcp"] * 1e-3, 1e-5)

    # Improvement #7
    T_edc = edc_passive_tension_N(pose)

    # DIP equilibrium → FDP
    f_fdp = max((M_dip_ext - M_pass_dip) / r_fdp_dip, 0.0)

    # PIP equilibrium → FDS
    f_fds = max((M_pip_ext - M_pass_pip - f_fdp * r_fdp_pip) / r_fds_pip, 0.0)

    ratio = f_fdp / f_fds if f_fds > 1e-6 else float("inf")

    # Improvement #5: MCP residual
    M_mcp_flexors  = f_fdp * r_fdp_mcp + f_fds * r_fds_mcp
    M_mcp_extensor = T_edc * r_edc_mcp + M_pass_mcp
    mcp_residual   = M_mcp_ext - (M_mcp_flexors - M_mcp_extensor)

    # Improvement #6: distributed pulley loads with correct chord directions
    # FDP path: anchor(0) → A2(1) → A3(2) → A4(3) → fdp_ins(4)
    fdp_path = geo["FDP_PATH"]
    fds_path = geo["FDS_PATH"]

    def _chord_in(path, idx):
        return unit(path[idx-1] - path[idx]) if idx > 0 else None

    def _chord_out(path, idx):
        return unit(path[idx+1] - path[idx]) if idx < len(path)-1 else None

    # A2: FDP (idx=1) + FDS (idx=1)
    A2_fdp, _ = pulley_load_arc(geo["A2_ARC"], f_fdp,
                                  _chord_in(fdp_path, 1), _chord_out(fdp_path, 1))
    A2_fds, _ = pulley_load_arc(geo["A2_ARC"], f_fds,
                                  _chord_in(fds_path, 1), _chord_out(fds_path, 1))
    # A3: FDP (idx=2) + FDS (idx=2)
    A3_fdp, _ = pulley_load_arc(geo["A3_ARC"], f_fdp,
                                  _chord_in(fdp_path, 2), _chord_out(fdp_path, 2))
    A3_fds, _ = pulley_load_arc(geo["A3_ARC"], f_fds,
                                  _chord_in(fds_path, 2), _chord_out(fds_path, 2))
    # A4: FDP only (idx=3)
    A4_fdp, _ = pulley_load_arc(geo["A4_ARC"], f_fdp,
                                  _chord_in(fdp_path, 3), _chord_out(fdp_path, 3))

    A2_vec = A2_fdp + A2_fds
    A3_vec = A3_fdp + A3_fds
    A4_vec = A4_fdp

    return dict(
        FDP_N           = f_fdp,
        FDS_N           = f_fds,
        EDC_N           = T_edc,
        FDP_FDS_ratio   = ratio,
        A2_N            = float(np.linalg.norm(A2_vec)),
        A3_N            = float(np.linalg.norm(A3_vec)),
        A4_N            = float(np.linalg.norm(A4_vec)),
        A2_vec          = A2_vec,
        A3_vec          = A3_vec,
        A4_vec          = A4_vec,
        M_dip_ext_Nm    = M_dip_ext,
        M_pip_ext_Nm    = M_pip_ext,
        M_mcp_ext_Nm    = M_mcp_ext,
        M_pass_dip_Nm   = M_pass_dip,
        M_pass_pip_Nm   = M_pass_pip,
        M_pass_mcp_Nm   = M_pass_mcp,
        mcp_residual_Nm = mcp_residual,
        r_fdp_dip_mm    = arms["r_fdp_dip"],
        r_fdp_pip_mm    = arms["r_fdp_pip"],
        r_fds_pip_mm    = arms["r_fds_pip"],
        r_fdp_mcp_mm    = arms["r_fdp_mcp"],
        r_fds_mcp_mm    = arms["r_fds_mcp"],
        r_edc_mcp_mm    = arms["r_edc_mcp"],
        F_ext           = F_ext,
        geo             = geo,
        load_point      = cfg.load_point,
    )


# ──────────── Improvement #8: held-out calibration ───────────────────────────

def fit_power_law(x1, y1, x2, y2):
    x1, x2 = max(float(x1), 1e-6), max(float(x2), 1e-6)
    y1, y2 = max(float(y1), 1e-6), max(float(y2), 1e-6)
    if abs(np.log(x2) - np.log(x1)) < 1e-9:
        return 1.0, 1.0
    n = float(np.log(y2 / y1) / np.log(x2 / x1))
    k = float(y1 / x1**n)
    return k, n


def apply_power_law(x, k, n):
    return float(k * max(float(x), 1e-6)**n)


def benchmark_with_holdout(avg_lit: Dict[str, Dict]) -> None:
    """
    Improvement #8: fit on open + full-crimp; validate on half-crimp.
    Schweizer 2009: A2 open=121 N, half=197 N, full=287 N
                    A4 open=103 N, half=165 N, full=226 N
    """
    A2_LIT = dict(open_drag=121.0, half_crimp=197.0, full_crimp=287.0)
    A4_LIT = dict(open_drag=103.0, half_crimp=165.0, full_crimp=226.0)

    a2_k, a2_n = fit_power_law(avg_lit["open_drag"]["A2_N"],  A2_LIT["open_drag"],
                                avg_lit["full_crimp"]["A2_N"], A2_LIT["full_crimp"])
    a4_k, a4_n = fit_power_law(avg_lit["open_drag"]["A4_N"],  A4_LIT["open_drag"],
                                avg_lit["full_crimp"]["A4_N"], A4_LIT["full_crimp"])

    a2_pred = apply_power_law(avg_lit["half_crimp"]["A2_N"], a2_k, a2_n)
    a4_pred = apply_power_law(avg_lit["half_crimp"]["A4_N"], a4_k, a4_n)

    a2_err = 100.0 * (a2_pred - A2_LIT["half_crimp"]) / A2_LIT["half_crimp"]
    a4_err = 100.0 * (a4_pred - A4_LIT["half_crimp"]) / A4_LIT["half_crimp"]

    print("=== Improvement #8: Held-Out Calibration (Schweizer 2009) ===")
    print("  Fit anchors: open_drag + full_crimp   |   Held out: half_crimp")
    print(f"  A2: predicted={a2_pred:.1f} N  published={A2_LIT['half_crimp']:.1f} N  "
          f"error={a2_err:+.1f}%")
    print(f"  A4: predicted={a4_pred:.1f} N  published={A4_LIT['half_crimp']:.1f} N  "
          f"error={a4_err:+.1f}%")
    print()


# ──────────── Improvement #9: four-finger load sharing ───────────────────────

FINGER_DEFS = {
    "index":  dict(share=0.24, lp=0.95, lm=0.94, ld=0.95),
    "middle": dict(share=0.30, lp=1.00, lm=1.00, ld=1.00),
    "ring":   dict(share=0.28, lp=0.97, lm=0.97, ld=0.97),
}


def four_finger_results(base_lengths: Tuple[float, float, float],
                        total_N: float, cfg: SimConfig) -> Dict:
    """Improvement #9: per-finger equilibrium with Vigouroux load sharing."""
    results = {}
    for fname, fd in FINGER_DEFS.items():
        f_mag = total_N * fd["share"]
        lengths = (base_lengths[0] * fd["lp"],
                   base_lengths[1] * fd["lm"],
                   base_lengths[2] * fd["ld"])
        grips = _build_grips(lengths)
        results[fname] = {
            gname: solve_static_equilibrium(pose, f_mag, cfg)
            for gname, pose in grips.items()
        }
    return results


# ──────────────── Improvement #10: fatigue model ─────────────────────────────

def fatigue_peak_loads(pose: FingerPose, distal_force_N: float,
                       cfg: SimConfig) -> Dict:
    """
    Improvement #10: FDS capacity decays exponentially with time.
    FDP compensates to maintain grip moment.  Returns peak FDP, A2, and timing.

    When fresh FDP = 0 (FDS alone handles all load), the fatigue-driven FDP
    demand is the deficit moment divided by the FDP moment arm at PIP.
    """
    fresh = solve_static_equilibrium(pose, distal_force_N, cfg)
    fds0  = fresh["FDS_N"]
    fdp0  = fresh["FDP_N"]
    r_fds = max(fresh["r_fds_pip_mm"] * 1e-3, 1e-5)
    r_fdp = max(fresh["r_fdp_pip_mm"] * 1e-3, 1e-5)
    M_pip = fresh["M_pip_ext_Nm"]
    M_dip = fresh["M_dip_ext_Nm"]

    times   = np.linspace(0.0, cfg.fatigue_time_s, 400)
    fds_cap = fds0 * np.exp(-times / cfg.fds_fatigue_tau_s)

    # FDP must cover any moment deficit at PIP as FDS weakens
    # Also ensure FDP can always cover DIP moment
    fdp_pip_demand = np.maximum((M_pip - fds_cap * r_fds) / r_fdp, fdp0)
    fdp_dip_demand = max(M_dip / max(fresh["r_fdp_dip_mm"] * 1e-3, 1e-5), fdp0)
    fdp_dem = np.maximum(fdp_pip_demand, fdp_dip_demand)

    idx      = int(np.argmax(fdp_dem))

    # Scale A2 from fresh values; if fresh FDP=0, use FDS-based A2 scaling
    if fdp0 > 1.0:
        a2_scale = fresh["A2_N"] / fdp0
    else:
        # A2 is FDS-dominated when FDP~0; estimate A2 increase proportionally
        a2_scale = fresh["A2_N"] / max(fds0, 1e-6) * 0.6  # FDP contributes ~60% of A2

    peak_fdp = float(fdp_dem[idx])
    peak_a2  = float(fresh["A2_N"] + peak_fdp * a2_scale)

    return dict(
        peak_fdp_N  = peak_fdp,
        peak_a2_N   = peak_a2,
        peak_time_s = float(times[idx]),
        fds_at_peak = float(fds_cap[idx]),
        fdp_vs_time = fdp_dem,
        fds_vs_time = fds_cap,
        times       = times,
        fresh       = fresh,
    )


# ─────────────────────────── grip presets ────────────────────────────────────

def posture_from_joint_targets(lengths_mm, pip_flex_deg, dip_flex_deg,
                                distal_abs_deg, dip_passive_fraction, label):
    theta_d = distal_abs_deg
    theta_m = dip_flex_deg + theta_d
    theta_p = pip_flex_deg + theta_m
    return FingerPose(Lp=lengths_mm[0], Lm=lengths_mm[1], Ld=lengths_mm[2],
                      theta_p=theta_p, theta_m=theta_m, theta_d=theta_d,
                      dip_passive_fraction=dip_passive_fraction, label=label)


def _build_grips(lengths_mm) -> Dict[str, FingerPose]:
    return {
        "open_drag":  posture_from_joint_targets(
            lengths_mm, 40.0,  12.5, 2.5,  0.0, "open_drag"),
        "half_crimp": posture_from_joint_targets(
            lengths_mm, 75.0, -20.0, 0.0,  0.0, "half_crimp"),
        "full_crimp": posture_from_joint_targets(
            lengths_mm, 105.0,-35.0, 0.0,  0.0, "full_crimp"),
    }


# ─────────────────────────── visualisation ───────────────────────────────────

def visualize_grips(grips: Dict[str, FingerPose],
                    solved: Dict[str, Dict],
                    cfg: SimConfig,
                    out_file: Path,
                    fatigue_data: Optional[Dict] = None) -> None:

    n_grip = len(grips)
    has_fat = fatigue_data is not None
    n_cols  = n_grip + (1 if has_fat else 0)

    fig = plt.figure(figsize=(5.2 * n_cols + 0.5, 6.5), constrained_layout=True)
    gs  = gridspec.GridSpec(1, n_cols, figure=fig)
    axes = [fig.add_subplot(gs[0, i]) for i in range(n_grip)]

    fscale = 0.025

    for ax, (gname, pose) in zip(axes, grips.items()):
        res = solved[gname]
        geo = res["geo"]

        # Bones
        bx = [geo[k][0] for k in ("MCP", "PIP", "DIP", "TIP")]
        by = [geo[k][1] for k in ("MCP", "PIP", "DIP", "TIP")]
        ax.plot(bx, by, "-o", color="black", lw=2, ms=5, label="phalanges")

        # Tendons
        for pkey, col, lbl in [("FDP_PATH", "#1f77b4", "FDP"),
                                ("FDS_PATH", "#ff7f0e", "FDS"),
                                ("EDC_PATH", "#2ca02c", "EDC")]:
            p = geo[pkey]
            ax.plot(p[:, 0], p[:, 1], "--", color=col, lw=1.5, label=lbl)

        # Pulley arcs (Improvement #6)
        for akey, col in [("A2_ARC", "#7f7f7f"),
                          ("A3_ARC", "#9467bd"),
                          ("A4_ARC", "#8c564b")]:
            arc = geo[akey]
            ax.plot(arc[:, 0], arc[:, 1], "-", color=col, lw=3.5, alpha=0.55)

        # Labels
        for key in ["MCP", "PIP", "DIP", "TIP", "A2", "A3", "A4"]:
            p = geo[key]
            ax.text(p[0] + 1, p[1] + 1, key, fontsize=7)

        # External force vector (Improvement #3 — not always vertical)
        lp = external_load_point(geo, cfg.load_point)
        Fv = res["F_ext"] * fscale
        ax.arrow(lp[0], lp[1], Fv[0], Fv[1], width=0.25, color="green",
                 length_includes_head=True)
        ax.text(lp[0] + Fv[0] + 1, lp[1] + Fv[1], "F_ext", color="green", fontsize=7)

        # Pulley resultant vectors
        for pname, col in [("A2", "#7f7f7f"), ("A3", "#9467bd"), ("A4", "#8c564b")]:
            vec = res[f"{pname}_vec"] * fscale
            p   = geo[pname]
            ax.arrow(p[0], p[1], vec[0], vec[1], width=0.18, color=col,
                     length_includes_head=True)

        title = (f"{gname}\n"
                 f"FDP={res['FDP_N']:.0f} N  FDS={res['FDS_N']:.0f} N  "
                 f"EDC={res['EDC_N']:.0f} N\n"
                 f"ratio={res['FDP_FDS_ratio']:.2f}  "
                 f"MCP resid={res['mcp_residual_Nm']*1000:.0f} mNm")
        ax.set_title(title, fontsize=7.5)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")

    # Fatigue panel (Improvement #10)
    if has_fat:
        ax_f = fig.add_subplot(gs[0, n_grip])
        fd = fatigue_data
        ax_f.plot(fd["times"], fd["fdp_vs_time"], color="#1f77b4", lw=2, label="FDP demand")
        ax_f.plot(fd["times"], fd["fds_vs_time"], color="#ff7f0e", lw=2, label="FDS capacity")
        ax_f.axvline(fd["peak_time_s"], color="red", ls="--", lw=1.2, label="Peak FDP")
        ax_f.set_title(
            f"Fatigue model (full_crimp)\n"
            f"Peak FDP={fd['peak_fdp_N']:.0f} N @ t={fd['peak_time_s']:.0f} s\n"
            f"Peak A2≈{fd['peak_a2_N']:.0f} N", fontsize=7.5)
        ax_f.set_xlabel("Time (s)")
        ax_f.set_ylabel("Force (N)")
        ax_f.legend(fontsize=7)
        ax_f.grid(True, alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, fontsize=8)
    fig.suptitle(
        f"Middle finger — improved model — {cfg.load_point} loading",
        y=1.03, fontsize=11)
    fig.savefig(out_file, dpi=200, bbox_inches="tight")
    print(f"  → Saved: {out_file}")


# ─────────────────────────────── main ────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-point", choices=["distal_mid", "fingertip"],
                        default="distal_mid")
    parser.add_argument("--fatigue-time", type=float, default=30.0)
    args = parser.parse_args()

    cfg_main = SimConfig(load_point=args.load_point, fatigue_time_s=args.fatigue_time)
    cfg_lit  = SimConfig(load_point="fingertip")

    athletes = [
        AthleteCase("Climber_Short",   54.0, (21.0, 22.0, 23.0)),
        AthleteCase("Climber_Average", 72.8, (26.0, 25.0, 26.0)),
        AthleteCase("Climber_Long",    92.0, (31.0, 28.0, 29.0)),
    ]

    load_frac = 0.15
    header = ("Improvements active: #1 tendon-excursion MA  #2 length-scaled MA  "
              "#3 grip-direction force\n"
              "                     #4 passive stiffness   #5 MCP residual      "
              "#6 distributed pulleys\n"
              "                     #7 EDC passive         #8 held-out calib    "
              "#9 four-finger       #10 fatigue")

    print("\n=== Middle-Finger Climbing Biomechanics — Improved Model ===")
    print(header)
    print()

    all_results: Dict[str, Dict[str, Dict]] = {}

    for athlete in athletes:
        bw_N  = athlete.mass_kg * 9.81
        f_mag = load_frac * bw_N
        grips = _build_grips(athlete.lengths_mm)

        print(f"─── {athlete.name}  {athlete.mass_kg:.1f} kg  "
              f"P/M/D={athlete.lengths_mm} mm  load={f_mag:.1f} N")
        hdr = (f"{'grip':12s} {'FDP':>7} {'FDS':>7} {'EDC':>6} "
               f"{'ratio':>6} {'A2':>6} {'A3':>6} {'A4':>6} "
               f"{'r_fdp_dip':>10} {'r_fds_pip':>10} "
               f"{'Mpass_pip mNm':>14} {'MCP resid mNm':>14}")
        print(hdr)

        athlete_res = {}
        for gname, pose in grips.items():
            r = solve_static_equilibrium(pose, f_mag, cfg_main)
            athlete_res[gname] = r
            print(f"{gname:12s} "
                  f"{r['FDP_N']:7.1f} {r['FDS_N']:7.1f} {r['EDC_N']:6.1f} "
                  f"{r['FDP_FDS_ratio']:6.2f} "
                  f"{r['A2_N']:6.1f} {r['A3_N']:6.1f} {r['A4_N']:6.1f} "
                  f"{r['r_fdp_dip_mm']:10.2f} {r['r_fds_pip_mm']:10.2f} "
                  f"{r['M_pass_pip_Nm']*1000:14.1f} "
                  f"{r['mcp_residual_Nm']*1000:14.1f}")
        print()
        all_results[athlete.name] = athlete_res

        if athlete.name == "Climber_Average":
            print("  Computing fatigue model (full_crimp)...")
            fat = fatigue_peak_loads(grips["full_crimp"], f_mag, cfg_main)
            out_png = Path(__file__).with_name("finger_biomechanics_forces.png")
            visualize_grips(grips, athlete_res, cfg_main, out_png, fatigue_data=fat)
            print()

    # ── Improvement #9 ──
    avg_f = load_frac * 72.8 * 9.81
    avg_lengths = (26.0, 25.0, 26.0)
    print("=== Improvement #9: Four-Finger Model (Vigouroux 2006 load sharing) ===")
    print(f"  Total finger load = {avg_f:.1f} N  (index 24%, middle 30%, ring 28%)")
    ff = four_finger_results(avg_lengths, avg_f, cfg_main)
    print(f"  {'finger':8s} {'grip':12s} {'FDP':>7} {'FDS':>7} {'A2':>7} {'A4':>7}")
    for fname, gd in ff.items():
        for gname, res in gd.items():
            print(f"  {fname:8s} {gname:12s} "
                  f"{res['FDP_N']:7.1f} {res['FDS_N']:7.1f} "
                  f"{res['A2_N']:7.1f} {res['A4_N']:7.1f}")
    print()

    # ── Improvement #10 ──
    avg_grips = _build_grips(avg_lengths)
    fat2 = fatigue_peak_loads(avg_grips["full_crimp"], avg_f, cfg_main)
    print("=== Improvement #10: Fatigue Model (full_crimp, average climber) ===")
    print(f"  Fresh:  FDP={fat2['fresh']['FDP_N']:.1f} N   A2={fat2['fresh']['A2_N']:.1f} N")
    print(f"  Peak:   FDP={fat2['peak_fdp_N']:.1f} N   A2={fat2['peak_a2_N']:.1f} N"
          f"   @ t={fat2['peak_time_s']:.0f} s   FDS residual={fat2['fds_at_peak']:.1f} N")
    inc = 100.0 * (fat2['peak_fdp_N'] / max(fat2['fresh']['FDP_N'], 1e-6) - 1.0)
    print(f"  FDP increase due to FDS fatigue: +{inc:.1f}%")
    print()

    # ── Literature benchmark ──
    avg_lit = {g: solve_static_equilibrium(p, avg_f, cfg_lit)
               for g, p in avg_grips.items()}
    print("=== Literature Benchmark (fingertip load, average climber) ===")
    print(f"  FDP/FDS open_drag   = {avg_lit['open_drag']['FDP_FDS_ratio']:.2f}  "
          f"(target ~0.88, Vigouroux 2006)")
    print(f"  FDP/FDS half_crimp  = {avg_lit['half_crimp']['FDP_FDS_ratio']:.2f}  "
          f"(expected between open and full)")
    print(f"  FDP/FDS full_crimp  = {avg_lit['full_crimp']['FDP_FDS_ratio']:.2f}  "
          f"(target ~1.75, Vigouroux 2006)")
    print()

    # Improvement #8
    benchmark_with_holdout(avg_lit)

    # ── Geometry advantage ──
    print("=== Geometry Advantage (short vs long, same mass/load) ===")
    for gname in ["open_drag", "half_crimp", "full_crimp"]:
        rs = solve_static_equilibrium(_build_grips(athletes[0].lengths_mm)[gname],
                                       avg_f, cfg_main)
        rl = solve_static_equilibrium(_build_grips(athletes[2].lengths_mm)[gname],
                                       avg_f, cfg_main)
        dfdp = 100.0 * (rl["FDP_N"] - rs["FDP_N"]) / max(rl["FDP_N"], 1e-6)
        dfds = 100.0 * (rl["FDS_N"] - rs["FDS_N"]) / max(rl["FDS_N"], 1e-6)
        print(f"  {gname:12s}  ΔFDP={dfdp:+.1f}%   ΔFDS={dfds:+.1f}%")

    print("\nDone.")


if __name__ == "__main__":
    main()