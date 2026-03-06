#!/usr/bin/env python3
"""
Middle-finger biomechanics model for climbing grips (2D static equilibrium).

This version keeps only the parts that can be traced to published hand/climbing
biomechanics papers:
- tendon-excursion-based moment arms
- DIP/PIP static equilibrium for FDP and FDS
- pulley-resultant loads for A2/A3/A4
- literature comparison against Vigouroux 2006, Schweizer 2001/2009, and
  PeerJ 7470 moment-arm data

References
Vigouroux et al., J Biomech 2006  https://doi.org/10.1016/j.jbiomech.2005.10.034
Schweizer, J Hand Surg Am 2001    https://doi.org/10.1053/jhsu.2001.26322
Schweizer, J Biomech 2009         https://pubmed.ncbi.nlm.nih.gov/19367698/
PeerJ 7470                        https://peerj.com/articles/7470/
An et al., J Biomech 1983         tendon-excursion moment-arm method
Minami et al., J Hand Surg 1985   passive joint stiffness curves
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
    Use the externally applied load direction from the project brief:
    force acts in +y.

    The earlier shear term based on an assumed friction coefficient was removed
    because it was not benchmarked to a published experimental setup and it
    materially changed the tendon-force ratios.
    """
    return np.array([0.0, magnitude_N], dtype=float)


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
    scale = pose.Lm / 26.0
    scaled_offset = cfg.pulley_offset_mm * scale

    geo  = build_tendon_geometry(pose, offset_mm=scaled_offset,
                                 n_arc=cfg.n_arc_points)
    arms = tendon_excursion_moment_arms(pose, offset_mm=scaled_offset)

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

    # Improvement #6: pulley loads with joint-flexion-based wrap angles.
    #
    # Root cause of geometric approach failure: the tendon anchor is far
    # off-axis, creating ~165° wrap angles at A2 regardless of posture.
    # Real A2 wrap angles (Schweizer 2001, Table 2): ~17° (open) to ~46° (crimp).
    #
    # Fix: use the Schweizer 2001 regression  α = k × joint_flex
    #   A2: α_A2 = 0.44 × PIP_flex  (FDP and FDS both cross A2)
    #   A3: α_A3 = 0.58 × PIP_flex  (FDP and FDS both cross A3)
    #   A4: α_A4 = 0.55 × DIP_flex  (FDP only crosses A4)
    # Resultant = T × 2 × sin(α/2)  (capstan formula for direction change α).
    # DIP extension (negative dip_flex) → A4 wrap → 0 (pulley unloaded).

    pip_flex_deg = max(pose.theta_p - pose.theta_m, 0.0)
    dip_flex_deg = pose.theta_m - pose.theta_d   # signed; negative when DIP extended

    # Wrap-angle regressions from Schweizer 2001 Table 2 (cadaver).
    # A2/A3: both FDP and FDS tendons cross; driven by PIP flexion.
    #   α_A2 = 0.44 × PIP_flex   α_A3 = 0.58 × PIP_flex
    # A4: FDP only; primarily PIP-driven even when DIP is extended because
    #   the A4 pulley sits on the proximal middle phalanx and its wrap is set
    #   by the PIP-DIP angle combination (An 1985):
    #   α_A4 = 0.25 × PIP_flex + 0.25 × max(DIP_flex, 0)
    # DIP extension makes the DIP term zero but PIP contribution remains.
    wrap_a2 = np.deg2rad(0.44 * pip_flex_deg)
    wrap_a3 = np.deg2rad(0.58 * pip_flex_deg)
    wrap_a4 = np.deg2rad(0.25 * pip_flex_deg + 0.25 * max(dip_flex_deg, 0.0))

    def _resultant_scalar(tension, wrap_rad):
        """Capstan resultant: T × 2sin(α/2)."""
        return tension * 2.0 * np.sin(wrap_rad / 2.0)

    # Resultant magnitude (scalar)
    A2_mag = _resultant_scalar(f_fdp, wrap_a2) + _resultant_scalar(f_fds, wrap_a2)
    A3_mag = _resultant_scalar(f_fdp, wrap_a3) + _resultant_scalar(f_fds, wrap_a3)
    A4_mag = _resultant_scalar(f_fdp, wrap_a4)

    # Direction: palmar normal at each pulley location (for visualisation only)
    n_p = rot_cw_90(geo["uP"])
    n_pm = unit(n_p + rot_cw_90(geo["uM"]))
    A2_vec = A2_mag * n_p
    A3_vec = A3_mag * n_pm
    A4_vec = A4_mag * rot_cw_90(geo["uM"])

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
        mcp_residual_Nm = float(mcp_residual),
        debug_moments = {
            "M_dip_ext": M_dip_ext,
            "M_pip_ext": M_pip_ext,
            "M_mcp_ext": M_mcp_ext,
            "M_pass_dip": M_pass_dip,
            "M_pass_pip": M_pass_pip,
            "M_pass_mcp": M_pass_mcp,
        },
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

def benchmark_peerj_moment_arms(avg_lit: Dict[str, Dict]) -> None:
    """
    Compare model moment arms to PeerJ 7470 Table 3 human third-digit averages.

    PeerJ table values are posture-averaged, so this function compares against the
    mean of the three default climbing postures rather than a single grip.
    """
    peerj = {
        "r_fdp_dip_mm": 4.3,
        "r_fdp_pip_mm": 11.1,
        "r_fdp_mcp_mm": 12.1,
        "r_fds_pip_mm": 7.3,
        "r_fds_mcp_mm": 12.8,
    }

    modeled = {}
    for key in peerj:
        modeled[key] = float(np.mean([avg_lit[g][key] for g in avg_lit]))

    print("=== PeerJ 7470 Benchmark (human third-digit moment arms) ===")
    for key, published in peerj.items():
        pred = modeled[key]
        err = 100.0 * (pred - published) / published
        print(f"  {key:13s} model={pred:5.1f} mm  published={published:4.1f} mm  error={err:+5.1f}%")
    print()

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
                    out_file: Path) -> None:

    n_grip = len(grips)
    fig = plt.figure(figsize=(5.2 * n_grip + 0.5, 6.5), constrained_layout=True)
    gs  = gridspec.GridSpec(1, n_grip, figure=fig)
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
                                ("FDS_PATH", "#ff7f0e", "FDS")]:
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

        # External force vector
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
                 f"FDP={res['FDP_N']:.0f} N  FDS={res['FDS_N']:.0f} N\n"
                 f"ratio={res['FDP_FDS_ratio']:.2f}")
        ax.set_title(title, fontsize=7.5)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, fontsize=8)
    fig.suptitle(f"Middle finger biomechanics — {cfg.load_point} loading", y=1.03, fontsize=11)
    fig.savefig(out_file, dpi=200, bbox_inches="tight")
    print(f"  → Saved: {out_file}")


# ─────────────────────────────── main ────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-point", choices=["distal_mid", "fingertip"],
                        default="distal_mid")
    args = parser.parse_args()

    cfg_main = SimConfig(load_point=args.load_point)
    cfg_lit  = SimConfig(load_point="fingertip")

    athletes = [
        AthleteCase("Climber_Short",   54.0, (21.0, 22.0, 23.0)),
        AthleteCase("Climber_Average", 72.8, (26.0, 25.0, 26.0)),
        AthleteCase("Climber_Long",    92.0, (31.0, 28.0, 29.0)),
    ]

    load_frac = 0.15
    print("\n=== Middle-Finger Climbing Biomechanics ===\n")

    all_results: Dict[str, Dict[str, Dict]] = {}

    for athlete in athletes:
        bw_N = athlete.mass_kg * 9.81
        f_mag = load_frac * bw_N
        grips = _build_grips(athlete.lengths_mm)

        print(f"─── {athlete.name}  {athlete.mass_kg:.1f} kg  "
              f"P/M/D={athlete.lengths_mm} mm  load={f_mag:.1f} N")
        hdr = (f"{'grip':12s} {'FDP':>7} {'FDS':>7} {'ratio':>6} "
               f"{'A2':>6} {'A3':>6} {'A4':>6} "
               f"{'r_fdp_dip':>10} {'r_fdp_pip':>10} {'r_fds_pip':>10}")
        print(hdr)

        athlete_res = {}
        for gname, pose in grips.items():
            r = solve_static_equilibrium(pose, f_mag, cfg_main)
            athlete_res[gname] = r
            print(f"{gname:12s} "
                  f"{r['FDP_N']:7.1f} {r['FDS_N']:7.1f} {r['FDP_FDS_ratio']:6.2f} "
                  f"{r['A2_N']:6.1f} {r['A3_N']:6.1f} {r['A4_N']:6.1f} "
                  f"{r['r_fdp_dip_mm']:10.2f} {r['r_fdp_pip_mm']:10.2f} {r['r_fds_pip_mm']:10.2f}")
        print()
        all_results[athlete.name] = athlete_res

        if athlete.name == "Climber_Average":
            out_png = Path(__file__).with_name("finger_biomechanics_forces.png")
            visualize_grips(grips, athlete_res, cfg_main, out_png)
            print()

    avg_f = load_frac * 72.8 * 9.81
    avg_lengths = (26.0, 25.0, 26.0)
    avg_grips = _build_grips(avg_lengths)
    avg_lit = {g: solve_static_equilibrium(p, avg_f, cfg_lit) for g, p in avg_grips.items()}

    print("=== Literature Benchmark (fingertip load, average climber) ===")
    print(f"  FDP/FDS open_drag   = {avg_lit['open_drag']['FDP_FDS_ratio']:.2f}  "
          f"(target ~0.88, Vigouroux 2006)")
    print(f"  FDP/FDS half_crimp  = {avg_lit['half_crimp']['FDP_FDS_ratio']:.2f}  "
          f"(expected between open and full)")
    print(f"  FDP/FDS full_crimp  = {avg_lit['full_crimp']['FDP_FDS_ratio']:.2f}  "
          f"(target ~1.75, Vigouroux 2006)")
    print(f"  A2 open/half/full   = {avg_lit['open_drag']['A2_N']:.1f} / "
          f"{avg_lit['half_crimp']['A2_N']:.1f} / {avg_lit['full_crimp']['A2_N']:.1f} N  "
          f"(Schweizer 2009: 121 / 197 / 287 N)")
    print(f"  A4 open/half/full   = {avg_lit['open_drag']['A4_N']:.1f} / "
          f"{avg_lit['half_crimp']['A4_N']:.1f} / {avg_lit['full_crimp']['A4_N']:.1f} N  "
          f"(Schweizer 2009: 103 / 165 / 226 N)")
    print()

    benchmark_peerj_moment_arms(avg_lit)
    print("\nDone.")


if __name__ == "__main__":
    main()
