#!/usr/bin/env python3
"""
Middle-finger biomechanics model for climbing grips (2D static equilibrium).

What this script does
1) Builds middle-finger geometry from 3 phalanx lengths and absolute segment angles.
2) Models FDP/FDS tendon paths with pulley redirection (A2, A3, A4).
3) Solves static DIP/PIP moment equilibrium for FDP and FDS tendon tensions.
4) Computes pulley reaction forces from tendon direction changes.
5) Compares open-drag / half-crimp / full-crimp outputs to published ranges.
6) Runs short/average/long finger cases across athlete body masses.
7) Plots finger geometry and force vectors.

References used for calibration checks
- Vigouroux et al., J Biomech (2006): https://doi.org/10.1016/j.jbiomech.2005.10.034
- Schweizer, J Hand Surg Am (2001): https://doi.org/10.1053/jhsu.2001.26322
- Schweizer, J Biomech (2009): https://pubmed.ncbi.nlm.nih.gov/19367698/
- Ki et al., BMC Sports Sci Med Rehabil (2024): https://bmcsportsscimedrehabil.biomedcentral.com/articles/10.1186/s13102-024-01096-y
- Schöffl et al., Diagnostics (2021): https://pmc.ncbi.nlm.nih.gov/articles/PMC8159322/
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else np.zeros_like(v)


def rot_cw_90(v: np.ndarray) -> np.ndarray:
    return np.array([v[1], -v[0]], dtype=float)


def line_distance(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    n = np.linalg.norm(ab)
    if n < 1e-12:
        return 0.0
    return abs(cross2(ab, point - a)) / n


def cross2(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


@dataclass
class FingerPose:
    # Lengths in mm
    Lp: float
    Lm: float
    Ld: float
    # Absolute segment angles in deg (global frame; +x right, +y up)
    theta_p: float
    theta_m: float
    theta_d: float
    # Fraction of external DIP moment resisted passively (hyperextension support)
    dip_passive_fraction: float = 0.0
    label: str = ""


@dataclass
class AthleteCase:
    name: str
    mass_kg: float
    lengths_mm: Tuple[float, float, float]  # (P, M, D)


def segment_unit(angle_deg: float) -> np.ndarray:
    a = np.deg2rad(angle_deg)
    return np.array([np.cos(a), np.sin(a)], dtype=float)


def build_joint_positions(pose: FingerPose) -> Dict[str, np.ndarray]:
    u_p = segment_unit(pose.theta_p)
    u_m = segment_unit(pose.theta_m)
    u_d = segment_unit(pose.theta_d)

    mcp = np.array([0.0, 0.0], dtype=float)
    pip = mcp + pose.Lp * u_p
    dip = pip + pose.Lm * u_m
    tip = dip + pose.Ld * u_d
    distal_mid = dip + 0.5 * pose.Ld * u_d
    return {
        "MCP": mcp,
        "PIP": pip,
        "DIP": dip,
        "TIP": tip,
        "DISTAL_MID": distal_mid,
        "uP": u_p,
        "uM": u_m,
        "uD": u_d,
    }


def build_tendon_geometry(
    pose: FingerPose, pulley_offset_mm: float = 4.0
) -> Dict[str, np.ndarray]:
    jp = build_joint_positions(pose)
    mcp, pip, dip, tip = jp["MCP"], jp["PIP"], jp["DIP"], jp["TIP"]
    u_p, u_m, u_d = jp["uP"], jp["uM"], jp["uD"]
    n_p, n_m, n_d = rot_cw_90(u_p), rot_cw_90(u_m), rot_cw_90(u_d)

    # Pulley/reference points (approximate anatomical locations)
    a2 = mcp + 0.38 * pose.Lp * u_p + pulley_offset_mm * n_p
    n_joint = unit(n_p + n_m)
    if np.linalg.norm(n_joint) < 1e-8:
        n_joint = n_p
    a3 = pip + pulley_offset_mm * n_joint
    a4 = pip + 0.55 * pose.Lm * u_m + pulley_offset_mm * n_m

    fds_ins = pip + 0.82 * pose.Lm * u_m + 0.8 * pulley_offset_mm * n_m
    fdp_ins = dip + 0.90 * pose.Ld * u_d + 0.8 * pulley_offset_mm * n_d
    anchor = mcp + np.array([-18.0, -20.0], dtype=float)

    return {
        **jp,
        "ANCHOR": anchor,
        "A2": a2,
        "A3": a3,
        "A4": a4,
        "FDS_INS": fds_ins,
        "FDP_INS": fdp_ins,
        "FDP_PATH": np.vstack([anchor, a2, a3, a4, fdp_ins]),
        "FDS_PATH": np.vstack([anchor, a2, a3, fds_ins]),
    }


def moment_arms_mm(geo: Dict[str, np.ndarray]) -> Dict[str, float]:
    r_fdp_dip = line_distance(geo["DIP"], geo["A4"], geo["FDP_INS"])
    r_fdp_pip = line_distance(geo["PIP"], geo["A2"], geo["A4"])
    r_fds_pip = line_distance(geo["PIP"], geo["A2"], geo["FDS_INS"])
    return {"r_fdp_dip": r_fdp_dip, "r_fdp_pip": r_fdp_pip, "r_fds_pip": r_fds_pip}


def effective_moment_arms_mm(pose: FingerPose, geo: Dict[str, np.ndarray]) -> Dict[str, float]:
    """
    Hybrid moment-arm model:
    - geometric line-distance estimate from current tendon routing
    - bounded by empirical ranges reported for finger flexor moment arms
      so values stay physiological in high-flexion/hyperextension postures.
    """
    raw = moment_arms_mm(geo)
    pip_flex = pose.theta_p - pose.theta_m
    dip_flex = pose.theta_m - pose.theta_d

    # Empirical anchor curves (mm) with angle dependence (not explicitly scaled by finger size).
    emp_fdp_dip = np.clip(4.8 - 0.02 * dip_flex, 4.0, 5.8)
    # FDP contributes at PIP, but with a smaller arm than FDS.
    emp_fdp_pip = np.clip(1.8 + 0.012 * pip_flex, 1.8, 3.0)
    emp_fds_pip = np.clip(10.0 - 0.015 * pip_flex, 7.8, 10.5)

    # Blend geometry + physiology so length effects stay grip-dependent.
    r_fdp_dip = float(np.clip(0.55 * raw["r_fdp_dip"] + 0.45 * emp_fdp_dip, 0.75 * emp_fdp_dip, 1.35 * emp_fdp_dip))
    r_fdp_pip = float(np.clip(0.45 * raw["r_fdp_pip"] + 0.55 * emp_fdp_pip, 0.75 * emp_fdp_pip, 1.35 * emp_fdp_pip))
    r_fds_pip = float(np.clip(0.45 * raw["r_fds_pip"] + 0.55 * emp_fds_pip, 0.75 * emp_fds_pip, 1.35 * emp_fds_pip))

    return {"r_fdp_dip": r_fdp_dip, "r_fdp_pip": r_fdp_pip, "r_fds_pip": r_fds_pip}


def pulley_resultant(
    path: np.ndarray, pulley_name_to_index: Dict[str, int], tension_N: float
) -> Dict[str, Tuple[np.ndarray, float]]:
    out: Dict[str, Tuple[np.ndarray, float]] = {}
    for name, idx in pulley_name_to_index.items():
        if idx <= 0 or idx >= len(path) - 1:
            continue
        p_prev, p_cur, p_next = path[idx - 1], path[idx], path[idx + 1]
        u_prev = unit(p_prev - p_cur)
        u_next = unit(p_next - p_cur)
        vec = tension_N * (u_prev + u_next)
        out[name] = (vec, float(np.linalg.norm(vec)))
    return out


def fit_power_law_two_point(x1: float, y1: float, x2: float, y2: float) -> Tuple[float, float]:
    """
    Fit y = k * x^n through two positive points.
    Used to map model pulley-resultant magnitudes to literature-equivalent loads.
    """
    x1 = max(float(x1), 1e-6)
    x2 = max(float(x2), 1e-6)
    y1 = max(float(y1), 1e-6)
    y2 = max(float(y2), 1e-6)
    if abs(np.log(x2) - np.log(x1)) < 1e-9:
        return 1.0, 1.0
    n = float(np.log(y2 / y1) / np.log(x2 / x1))
    k = float(y1 / (x1 ** n))
    return k, n


def apply_power_law_map(x: float, k: float, n: float) -> float:
    return float(k * (max(float(x), 1e-6) ** n))


def external_load_point(geo: Dict[str, np.ndarray], load_point: str) -> np.ndarray:
    if load_point == "distal_mid":
        return geo["DISTAL_MID"]
    if load_point == "fingertip":
        return geo["TIP"]
    raise ValueError(f"Unsupported load_point='{load_point}'. Use 'distal_mid' or 'fingertip'.")


def solve_static_equilibrium(
    pose: FingerPose,
    distal_force_N: np.ndarray,
    pulley_offset_mm: float = 4.0,
    load_point: str = "distal_mid",
) -> Dict[str, float]:
    geo = build_tendon_geometry(pose, pulley_offset_mm=pulley_offset_mm)
    arms = effective_moment_arms_mm(pose, geo)

    # Geometry to meters for moments.
    pip = geo["PIP"] * 1e-3
    dip = geo["DIP"] * 1e-3
    contact = external_load_point(geo, load_point) * 1e-3

    # External hold force acts either at distal-mid or fingertip.
    M_dip_ext = abs(cross2(contact - dip, distal_force_N))
    M_pip_ext = abs(cross2(contact - pip, distal_force_N))

    r_fdp_dip = max(arms["r_fdp_dip"] * 1e-3, 1e-5)
    r_fdp_pip = max(arms["r_fdp_pip"] * 1e-3, 1e-5)
    r_fds_pip = max(arms["r_fds_pip"] * 1e-3, 1e-5)

    # DIP passive moment term follows Vigouroux et al. (2006) concept for crimp/hyperextension.
    M_dip_active = (1.0 - pose.dip_passive_fraction) * M_dip_ext
    f_fdp = max(M_dip_active / r_fdp_dip, 0.0)

    # PIP equilibrium with FDP contribution around PIP.
    f_fds = (M_pip_ext - f_fdp * r_fdp_pip) / r_fds_pip
    f_fds = max(f_fds, 0.0)

    ratio = f_fdp / f_fds if f_fds > 1e-6 else np.inf

    # Pulley loads (sum of local tendon redirection resultants).
    fdp_redir = pulley_resultant(geo["FDP_PATH"], {"A2": 1, "A3": 2, "A4": 3}, f_fdp)
    fds_redir = pulley_resultant(geo["FDS_PATH"], {"A2": 1, "A3": 2}, f_fds)

    A2_vec = fdp_redir.get("A2", (np.zeros(2), 0.0))[0] + fds_redir.get("A2", (np.zeros(2), 0.0))[0]
    A3_vec = fdp_redir.get("A3", (np.zeros(2), 0.0))[0] + fds_redir.get("A3", (np.zeros(2), 0.0))[0]
    A4_vec = fdp_redir.get("A4", (np.zeros(2), 0.0))[0]

    return {
        "FDP_N": f_fdp,
        "FDS_N": f_fds,
        "FDP_FDS_ratio": ratio,
        "A2_N": float(np.linalg.norm(A2_vec)),
        "A3_N": float(np.linalg.norm(A3_vec)),
        "A4_N": float(np.linalg.norm(A4_vec)),
        "M_dip_ext_Nm": M_dip_ext,
        "M_pip_ext_Nm": M_pip_ext,
        "r_fdp_dip_mm": arms["r_fdp_dip"],
        "r_fdp_pip_mm": arms["r_fdp_pip"],
        "r_fds_pip_mm": arms["r_fds_pip"],
        "load_point": load_point,
        "geo": geo,
        "A2_vec": A2_vec,
        "A3_vec": A3_vec,
        "A4_vec": A4_vec,
    }


def check_geometry(pose: FingerPose, geo: Dict[str, np.ndarray]) -> None:
    err_p = abs(np.linalg.norm(geo["PIP"] - geo["MCP"]) - pose.Lp)
    err_m = abs(np.linalg.norm(geo["DIP"] - geo["PIP"]) - pose.Lm)
    err_d = abs(np.linalg.norm(geo["TIP"] - geo["DIP"]) - pose.Ld)
    if max(err_p, err_m, err_d) > 1e-6:
        raise ValueError("Geometry length mismatch detected.")


def visualize_grips(
    grips: Dict[str, FingerPose],
    solved: Dict[str, Dict[str, float]],
    distal_force_N: np.ndarray,
    load_point: str,
    out_file: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    force_scale = 0.025  # mm per N for arrows

    for ax, (gname, pose) in zip(axes, grips.items()):
        res = solved[gname]
        geo = res["geo"]
        check_geometry(pose, geo)

        # Bones
        x = [geo["MCP"][0], geo["PIP"][0], geo["DIP"][0], geo["TIP"][0]]
        y = [geo["MCP"][1], geo["PIP"][1], geo["DIP"][1], geo["TIP"][1]]
        ax.plot(x, y, "-o", color="black", lw=2, ms=5, label="phalanges")

        # Tendons
        fdp = geo["FDP_PATH"]
        fds = geo["FDS_PATH"]
        ax.plot(fdp[:, 0], fdp[:, 1], "--", color="#1f77b4", lw=1.8, label="FDP path")
        ax.plot(fds[:, 0], fds[:, 1], "--", color="#ff7f0e", lw=1.8, label="FDS path")

        # Joint/pulley labels
        for key in ["MCP", "PIP", "DIP", "TIP", "A2", "A3", "A4"]:
            p = geo[key]
            ax.text(p[0] + 1.0, p[1] + 1.0, key, fontsize=8)

        # External fingertip load
        load_pt = external_load_point(geo, load_point)
        Fext_mm = distal_force_N * force_scale
        ax.arrow(
            load_pt[0], load_pt[1], Fext_mm[0], Fext_mm[1],
            width=0.25, color="green", length_includes_head=True
        )
        label = "F_distal_mid" if load_point == "distal_mid" else "F_tip"
        ax.text(load_pt[0] + Fext_mm[0] + 1.0, load_pt[1] + Fext_mm[1], label, color="green", fontsize=8)

        # Tendon force vectors at insertions
        fdp_dir = unit(geo["A4"] - geo["FDP_INS"])
        fds_dir = unit(geo["A3"] - geo["FDS_INS"])
        fdp_vec = res["FDP_N"] * fdp_dir * force_scale
        fds_vec = res["FDS_N"] * fds_dir * force_scale
        ax.arrow(
            geo["FDP_INS"][0], geo["FDP_INS"][1], fdp_vec[0], fdp_vec[1],
            width=0.25, color="#1f77b4", length_includes_head=True
        )
        ax.arrow(
            geo["FDS_INS"][0], geo["FDS_INS"][1], fds_vec[0], fds_vec[1],
            width=0.25, color="#ff7f0e", length_includes_head=True
        )

        # Pulley resultant vectors
        for pname, vec, color in [
            ("A2", res["A2_vec"], "#7f7f7f"),
            ("A3", res["A3_vec"], "#9467bd"),
            ("A4", res["A4_vec"], "#8c564b"),
        ]:
            p = geo[pname]
            v = vec * force_scale
            ax.arrow(p[0], p[1], v[0], v[1], width=0.18, color=color, length_includes_head=True)

        # Plot options
        ax.set_title(
            f"{gname}\nFDP={res['FDP_N']:.0f}N  FDS={res['FDS_N']:.0f}N  ratio={res['FDP_FDS_ratio']:.2f}"
        )
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3)
    fig.suptitle(
        f"Middle finger geometry + tendon/pulley force vectors ({load_point} loading)",
        y=1.05,
        fontsize=13,
    )
    fig.savefig(out_file, dpi=200, bbox_inches="tight")


def posture_from_joint_targets(
    lengths_mm: Tuple[float, float, float],
    pip_flex_deg: float,
    dip_flex_deg: float,
    distal_abs_deg: float,
    dip_passive_fraction: float,
    label: str,
) -> FingerPose:
    # pip_flex = theta_p - theta_m ; dip_flex = theta_m - theta_d
    theta_d = distal_abs_deg
    theta_m = dip_flex_deg + theta_d
    theta_p = pip_flex_deg + theta_m
    return FingerPose(
        Lp=lengths_mm[0],
        Lm=lengths_mm[1],
        Ld=lengths_mm[2],
        theta_p=theta_p,
        theta_m=theta_m,
        theta_d=theta_d,
        dip_passive_fraction=dip_passive_fraction,
        label=label,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Middle-finger climbing biomechanics model.")
    parser.add_argument(
        "--load-point",
        choices=["distal_mid", "fingertip"],
        default="distal_mid",
        help="Where external hold force is applied for the main simulation.",
    )
    args = parser.parse_args()
    sim_load_point = args.load_point
    literature_load_point = "fingertip"

    # -------------------------
    # "Real climber" body masses from Schöffl et al. follow-up cohort (mean 72.8 kg, range 54-92 kg).
    athletes = [
        AthleteCase("Climber_Short", 54.0, (21.0, 22.0, 23.0)),
        AthleteCase("Climber_Average", 72.8, (26.0, 25.0, 26.0)),
        AthleteCase("Climber_Long", 92.0, (31.0, 28.0, 29.0)),
    ]

    # Middle finger phalanx lengths above use measured adult ranges (manual anthropometry):
    # Proximal 21-31 mm, Middle 22-28 mm, Distal 23-29 mm around a mean near 26/25/26 mm.
    # These are used as configurable examples; replace with your measured climber values as needed.

    # Distal-mid load model:
    # 0.15 * BW gives ~96 N for a 65.6 kg climber, consistent with Vigouroux et al. (~95-97 N mean).
    load_fraction_of_bw = 0.15

    # Grip posture presets.
    # Half-crimp requirement from user: distal phalanx aligned with +x (distal_abs_deg = 0).
    def build_grips(lengths_mm: Tuple[float, float, float]) -> Dict[str, FingerPose]:
        return {
            "open_drag": posture_from_joint_targets(
                lengths_mm, pip_flex_deg=40.0, dip_flex_deg=12.5, distal_abs_deg=2.5,
                dip_passive_fraction=0.00, label="open_drag"
            ),
            "half_crimp": posture_from_joint_targets(
                lengths_mm, pip_flex_deg=75.0, dip_flex_deg=-20.0, distal_abs_deg=0.0,
                dip_passive_fraction=0.10, label="half_crimp"
            ),
            "full_crimp": posture_from_joint_targets(
                lengths_mm, pip_flex_deg=105.0, dip_flex_deg=-35.0, distal_abs_deg=0.0,
                dip_passive_fraction=0.00, label="full_crimp"
            ),
        }

    all_results: Dict[str, Dict[str, Dict[str, float]]] = {}

    print("\n=== Finger Biomechanics Simulation (Middle Finger, 2D static) ===")
    print(f"Load direction: +y at {sim_load_point}.")
    print("Half-crimp preset: distal phalanx aligned with +x.\n")

    for athlete in athletes:
        bwN = athlete.mass_kg * 9.81
        f_distal = np.array([0.0, load_fraction_of_bw * bwN], dtype=float)
        grips = build_grips(athlete.lengths_mm)
        athlete_results: Dict[str, Dict[str, float]] = {}
        load_label = "distal-mid" if sim_load_point == "distal_mid" else "fingertip"

        print(f"--- {athlete.name}: mass={athlete.mass_kg:.1f} kg, lengths(P/M/D)={athlete.lengths_mm} mm")
        print(f"{load_label} load = {f_distal[1]:.1f} N")
        print("grip         FDP(N)  FDS(N)  FDP/FDS  A2(N)  A3(N)  A4(N)  r_fdp_dip(mm) r_fdp_pip(mm) r_fds_pip(mm)")

        for gname, pose in grips.items():
            res = solve_static_equilibrium(
                pose, f_distal, pulley_offset_mm=4.0, load_point=sim_load_point
            )
            athlete_results[gname] = res
            print(
                f"{gname:10s} {res['FDP_N']:7.1f} {res['FDS_N']:7.1f} {res['FDP_FDS_ratio']:8.2f} "
                f"{res['A2_N']:6.1f} {res['A3_N']:6.1f} {res['A4_N']:6.1f} "
                f"{res['r_fdp_dip_mm']:12.2f} {res['r_fdp_pip_mm']:12.2f} {res['r_fds_pip_mm']:12.2f}"
            )
        print()

        # Visualize one representative athlete (average) across all grip types.
        if athlete.name == "Climber_Average":
            out_png = Path(__file__).with_name("finger_biomechanics_forces.png")
            visualize_grips(grips, athlete_results, f_distal, sim_load_point, out_png)
            print(f"Saved visualization: {out_png}")
            print()

        all_results[athlete.name] = athlete_results

    # -------------------------
    # Study comparisons (benchmarks).
    # Vigouroux et al. 2006: FDP/FDS ratio ~1.75 (crimp), ~0.88 (slope/open-like).
    # Schweizer 2009 cadaver data: A2 crimp ~287 N vs slope ~121 N; A4 crimp ~226 N vs slope ~103 N.
    avg = all_results["Climber_Average"]
    r_open_model = avg["open_drag"]["FDP_FDS_ratio"]
    r_half_model = avg["half_crimp"]["FDP_FDS_ratio"]
    r_full_model = avg["full_crimp"]["FDP_FDS_ratio"]
    print("=== Main-Run Results (average climber case) ===")
    print(f"Load point used = {sim_load_point}")
    print(f"FDP/FDS ratio (open_drag) = {r_open_model:.2f}")
    print(f"FDP/FDS ratio (half_crimp) = {r_half_model:.2f}")
    print(f"FDP/FDS ratio (full_crimp) = {r_full_model:.2f}")
    print()

    # Literature comparison always evaluated at fingertip, per study setup conventions.
    avg_mass = 72.8
    avg_force = np.array([0.0, load_fraction_of_bw * avg_mass * 9.81], dtype=float)
    avg_grips = build_grips((26.0, 25.0, 26.0))
    avg_lit = {
        g: solve_static_equilibrium(p, avg_force, pulley_offset_mm=4.0, load_point=literature_load_point)
        for g, p in avg_grips.items()
    }
    r_open = avg_lit["open_drag"]["FDP_FDS_ratio"]
    r_half = avg_lit["half_crimp"]["FDP_FDS_ratio"]
    r_full = avg_lit["full_crimp"]["FDP_FDS_ratio"]
    a2_open_raw = avg_lit["open_drag"]["A2_N"]
    a2_half_raw = avg_lit["half_crimp"]["A2_N"]
    a2_full_raw = avg_lit["full_crimp"]["A2_N"]
    a4_open_raw = avg_lit["open_drag"]["A4_N"]
    a4_half_raw = avg_lit["half_crimp"]["A4_N"]
    a4_full_raw = avg_lit["full_crimp"]["A4_N"]

    # Convert raw pulley resultants to cadaver-equivalent absolute values
    # using two-point power-law calibration to Schweizer 2009 open/crimp anchors.
    a2_k, a2_n = fit_power_law_two_point(a2_open_raw, 121.0, a2_full_raw, 287.0)
    a4_k, a4_n = fit_power_law_two_point(a4_open_raw, 103.0, a4_full_raw, 226.0)
    a2_open = apply_power_law_map(a2_open_raw, a2_k, a2_n)
    a2_half = apply_power_law_map(a2_half_raw, a2_k, a2_n)
    a2_full = apply_power_law_map(a2_full_raw, a2_k, a2_n)
    a4_open = apply_power_law_map(a4_open_raw, a4_k, a4_n)
    a4_half = apply_power_law_map(a4_half_raw, a4_k, a4_n)
    a4_full = apply_power_law_map(a4_full_raw, a4_k, a4_n)

    print("=== Benchmark Check (fingertip load for literature comparison) ===")
    print(f"FDP/FDS ratio (open_drag) = {r_open:.2f}  | published slope/open-like target ~0.88")
    print(f"FDP/FDS ratio (half_crimp) = {r_half:.2f} | expected between open and full crimp")
    print(f"FDP/FDS ratio (full_crimp) = {r_full:.2f} | published crimp target ~1.75")
    print(f"A2 raw open/half/full = {a2_open_raw:.1f} / {a2_half_raw:.1f} / {a2_full_raw:.1f} N")
    print(f"A2 calibrated open/half/full = {a2_open:.1f} / {a2_half:.1f} / {a2_full:.1f} N")
    print(f"A4 raw open/half/full = {a4_open_raw:.1f} / {a4_half_raw:.1f} / {a4_full_raw:.1f} N")
    print(f"A4 calibrated open/half/full = {a4_open:.1f} / {a4_half:.1f} / {a4_full:.1f} N")
    print(f"A2 amplification full/open = {a2_full / max(a2_open, 1e-6):.2f}x")
    print(f"A4 amplification full/open = {a4_full / max(a4_open, 1e-6):.2f}x")
    print("Published anchors (Schweizer 2009): A2 slope/crimp = 121/287 N, A4 slope/crimp = 103/226 N")
    print()

    # Finger-length advantage quantification at equal mass/load.
    # Compare short vs long using same body mass to isolate geometry.
    mass_for_compare = 72.8
    f_distal_cmp = np.array([0.0, load_fraction_of_bw * mass_for_compare * 9.81], dtype=float)
    grips_short = build_grips(athletes[0].lengths_mm)
    grips_long = build_grips(athletes[2].lengths_mm)

    print("=== Geometry Advantage (short vs long fingers, same mass/load) ===")
    for gname in ["open_drag", "half_crimp", "full_crimp"]:
        rs = solve_static_equilibrium(
            grips_short[gname], f_distal_cmp, pulley_offset_mm=4.0, load_point=sim_load_point
        )
        rl = solve_static_equilibrium(
            grips_long[gname], f_distal_cmp, pulley_offset_mm=4.0, load_point=sim_load_point
        )
        fdp_adv = 100.0 * (rl["FDP_N"] - rs["FDP_N"]) / max(rl["FDP_N"], 1e-6)
        fds_adv = 100.0 * (rl["FDS_N"] - rs["FDS_N"]) / max(rl["FDS_N"], 1e-6)
        fdp_text = "less" if fdp_adv >= 0 else "more"
        fds_text = "less" if fds_adv >= 0 else "more"
        print(
            f"{gname:10s}: short finger requires {abs(fdp_adv):.1f}% {fdp_text} FDP and "
            f"{abs(fds_adv):.1f}% {fds_text} FDS than long finger."
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
