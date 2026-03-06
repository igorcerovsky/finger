#!/usr/bin/env python3
"""
plot_length_vs_force_peerj.py
─────────────────────────────────────────────────────────────────────────────
Finger ray length  vs  tendon forces required to hold a fixed tip load.

Two models compared side-by-side:

  1. Standard (proportional) model
     Moment arms scale linearly with middle-phalanx length (Improvement #2 in
     finger_biomechanics_model.py).  Longer finger → proportionally larger
     moment arms → forces stay roughly constant.

  2. PeerJ 7470 'length-disadvantage' model
     Inspired by Chimpanzee / Bonobo comparison: bones are longer, but tendon
     moment arms at DIP grow only ~5 % per 22 % bone increase (≈ 0.23× the
     proportional rate).  Result: muscle forces must INCREASE with finger
     length to maintain the same tip load — a length disadvantage.

X-axis : total finger ray length  (Lp + Lm + Ld, mm)
         ranging from small (literature 5th-pct female) to long
         (literature 95th-pct male) for the middle finger.

Y-axis : required tendon force (N) for a fixed 100 N tip load,
         half-crimp posture (PIP 75°, DIP −20°).

Literature finger length ranges (middle finger, index-finger ray):
  Pheasant (1996): male P5–P95 ≈ 160–195 mm total finger length;
  Ruff & Walker (1993); An et al. (1983): P/M/D ratios ~40/33/27 % of total.
  Model uses Lp:Lm:Ld proportions 41 : 32 : 27, giving ray spans 60–91 mm.

References
──────────
  An KN et al., J Biomech 1983 — tendon geometry
  PeerJ 7470 (2019) — length disadvantage in apes
  Pheasant (1996) — anthropometric data
"""

from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

from finger_biomechanics_model import (
    FingerPose, SimConfig, posture_from_joint_targets,
    solve_static_equilibrium, tendon_excursion_moment_arms,
    _L_REF_MM,
)

# ── Literature-based finger length sweep ─────────────────────────────────────
# Middle-finger ray total length range: ~60 mm (small female) to ~91 mm (large male)
# Phalanx proportions from An et al. 1983: Lp≈41%, Lm≈32%, Ld≈27% of ray total.
# (Total ray = proximal + middle + distal phalanx, NOT including metacarpal.)

TOTAL_RAY_MM = np.linspace(60.0, 92.0, 28)   # mm

LP_FRAC = 0.41
LM_FRAC = 0.32
LD_FRAC = 0.27

def _lengths(total_mm):
    return (total_mm * LP_FRAC,
            total_mm * LM_FRAC,
            total_mm * LD_FRAC)


# ── Grip parameters ───────────────────────────────────────────────────────────
GRIP_PARAMS = {
    "open_drag":  dict(pip=40.0,  dip=12.5, d_abs=2.5),
    "half_crimp": dict(pip=75.0,  dip=-20.0, d_abs=0.0),
    "full_crimp": dict(pip=105.0, dip=-35.0, d_abs=0.0),
}

DISTAL_LOAD_N = 100.0        # fixed tip load (N) — literature standard

CFG = SimConfig(load_point="fingertip")


# ── Model A: Standard (proportional MA scaling) ───────────────────────────────
def run_standard_model(total_ray_mm_arr, grip="half_crimp"):
    """Use solve_static_equilibrium with default proportional MA clipping."""
    gp = GRIP_PARAMS[grip]
    FDP_arr, FDS_arr, A2_arr = [], [], []

    for total in total_ray_mm_arr:
        lengths = _lengths(total)
        pose = posture_from_joint_targets(
            lengths, gp["pip"], gp["dip"], gp["d_abs"], 0.0, grip)
        res = solve_static_equilibrium(pose, DISTAL_LOAD_N, CFG)
        FDP_arr.append(res["FDP_N"])
        FDS_arr.append(res["FDS_N"])
        A2_arr.append(res["A2_N"])

    return np.array(FDP_arr), np.array(FDS_arr), np.array(A2_arr)


# ── Model B: PeerJ length-disadvantage model ──────────────────────────────────
def run_peerj_model(total_ray_mm_arr, grip="half_crimp"):
    """
    Moment arms at DIP and PIP grow at only ~23 % of the proportional
    rate (4–7 % vs ~22 % bone increase, after PeerJ 7470).

    Implementation:
      We compute the 'standard' moment arms for an AVERAGE finger and then
      freeze them (no scaling), mimicking the bonobo finding where tendons
      insert at nearly the same absolute position regardless of bone length.
      This means that as the finger gets longer, the required FDP must rise
      to generate a larger moment arm deficit.

    Equilibrium equations (same as solver, done manually):
      DIP:  T_fdp * r_fdp_dip   = M_dip_ext − M_pass_dip
      PIP:  T_fds * r_fds_pip   = M_pip_ext − M_pass_pip − T_fdp * r_fdp_pip
    """
    gp = GRIP_PARAMS[grip]

    # Compute reference (average-length) moment arms once.
    avg_lengths = _lengths(77.0)   # ~average total ray ≈ 77 mm
    avg_pose = posture_from_joint_targets(
        avg_lengths, gp["pip"], gp["dip"], gp["d_abs"], 0.0, grip)
    avg_arms = tendon_excursion_moment_arms(avg_pose)

    # PeerJ scaling factor on moment arms: ~5 % increase per 22 % bone increase
    # → MA_scale_rate = 5/22 ≈ 0.23 of bone scale rate
    PEERJ_MA_FRAC = 0.23          # fraction of proportional scaling that applies

    avg_total = sum(avg_lengths)

    FDP_arr, FDS_arr, A2_arr = [], [], []

    for total in total_ray_mm_arr:
        lengths = _lengths(total)
        pose = posture_from_joint_targets(
            lengths, gp["pip"], gp["dip"], gp["d_abs"], 0.0, grip)

        # Bone scale relative to average
        bone_sf = total / avg_total

        # PeerJ MA scale: only PEERJ_MA_FRAC of the proportional scaling
        ma_sf = 1.0 + PEERJ_MA_FRAC * (bone_sf - 1.0)

        # Scaled moment arms (m)
        r_fdp_dip = max(avg_arms["r_fdp_dip"] * ma_sf * 1e-3, 1e-5)
        r_fdp_pip = max(avg_arms["r_fdp_pip"] * ma_sf * 1e-3, 1e-5)
        r_fds_pip = max(avg_arms["r_fds_pip"] * ma_sf * 1e-3, 1e-5)

        # External moments — these DO scale because the finger is longer
        from finger_biomechanics_model import (
            build_tendon_geometry, contact_force_vector,
            external_load_point, passive_moment_Nm, cross2
        )
        scaled_offset = CFG.pulley_offset_mm * (lengths[1] / _L_REF_MM)
        geo = build_tendon_geometry(pose, offset_mm=scaled_offset, n_arc=CFG.n_arc_points)
        F_ext = contact_force_vector(pose, DISTAL_LOAD_N)
        pip = geo["PIP"] * 1e-3
        dip = geo["DIP"] * 1e-3
        contact = external_load_point(geo, CFG.load_point) * 1e-3

        M_dip_ext = abs(cross2(contact - dip, F_ext))
        M_pip_ext = abs(cross2(contact - pip, F_ext))

        dip_flex = pose.theta_m - pose.theta_d
        pip_flex = pose.theta_p - pose.theta_m
        M_pass_dip = passive_moment_Nm("DIP", max(dip_flex, 0.0))
        M_pass_pip = passive_moment_Nm("PIP", max(pip_flex, 0.0))

        f_fdp = max((M_dip_ext - M_pass_dip) / r_fdp_dip, 0.0)
        f_fds = max((M_pip_ext - M_pass_pip - f_fdp * r_fdp_pip) / r_fds_pip, 0.0)

        # A2 pulley load (same capstan formula as main model)
        pip_flex_deg = max(pose.theta_p - pose.theta_m, 0.0)
        wrap_a2 = np.deg2rad(0.44 * pip_flex_deg)
        A2 = (f_fdp + f_fds) * 2.0 * np.sin(wrap_a2 / 2.0)

        FDP_arr.append(f_fdp)
        FDS_arr.append(f_fds)
        A2_arr.append(A2)

    return np.array(FDP_arr), np.array(FDS_arr), np.array(A2_arr)


# ── Plot ──────────────────────────────────────────────────────────────────────

def make_plot():
    grips_to_plot = ["open_drag", "half_crimp", "full_crimp"]
    grip_labels   = ["Open drag", "Half crimp", "Full crimp"]

    grip_colors = {
        "open_drag":  "#2ca02c",
        "half_crimp": "#1f77b4",
        "full_crimp": "#d62728",
    }
    grip_ls_std  = "-"       # solid for standard model
    grip_ls_peerj = "--"     # dashed for PeerJ model

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), sharey=False)
    fig.patch.set_facecolor("#0d1117")
    for ax in axes:
        ax.set_facecolor("#161b22")
        ax.spines[:].set_color("#30363d")
        ax.tick_params(colors="#c9d1d9")
        ax.xaxis.label.set_color("#c9d1d9")
        ax.yaxis.label.set_color("#c9d1d9")
        ax.title.set_color("#e6edf3")

    titles = ["FDP Force (N)", "FDS Force (N)", "A2 Pulley Load (N)"]

    for gi, grip in enumerate(grips_to_plot):
        fdp_std, fds_std, a2_std = run_standard_model(TOTAL_RAY_MM, grip)
        fdp_pj,  fds_pj,  a2_pj  = run_peerj_model(TOTAL_RAY_MM, grip)

        data_std = [fdp_std, fds_std, a2_std]
        data_pj  = [fdp_pj,  fds_pj,  a2_pj ]

        col = grip_colors[grip]

        for ai, ax in enumerate(axes):
            ax.plot(TOTAL_RAY_MM, data_std[ai],
                    color=col, ls=grip_ls_std, lw=2.2,
                    label=f"{grip_labels[gi]} – Standard")
            ax.plot(TOTAL_RAY_MM, data_pj[ai],
                    color=col, ls=grip_ls_peerj, lw=2.2,
                    label=f"{grip_labels[gi]} – PeerJ")

    # Vertical reference lines for known athlete sizes
    known = [
        (63.0, "Small\n(5th pct F)", "#858585"),
        (77.0, "Average\n(50th pct)", "#aaaaaa"),
        (91.0, "Large\n(95th pct M)", "#666666"),
    ]
    for ax in axes:
        for x, lbl, col in known:
            ax.axvline(x, color=col, lw=0.8, ls=":", alpha=0.7)
            ax.text(x + 0.3, ax.get_ylim()[1] * 0.98 if ax.get_ylim()[1] > 0 else 1,
                    lbl, color=col, fontsize=6.5, va="top", ha="left")

    for ai, (ax, ttl) in enumerate(zip(axes, titles)):
        ax.set_xlabel("Total finger ray length  Lp + Lm + Ld  (mm)", fontsize=9)
        ax.set_ylabel(ttl, fontsize=9)
        ax.set_title(ttl, fontsize=10, fontweight="bold", pad=8)
        ax.grid(True, color="#21262d", linewidth=0.7)
        ax.set_xlim(TOTAL_RAY_MM[0], TOTAL_RAY_MM[-1])
        # Refresh vertical labels after y-limits known
        for x, lbl, col in known:
            ylim = ax.get_ylim()
            ax.text(x + 0.3, ylim[1] - 0.04 * (ylim[1] - ylim[0]),
                    lbl, color=col, fontsize=6.5, va="top", ha="left",
                    clip_on=True)

    # Legend
    legend_elements = []
    for gi, grip in enumerate(grips_to_plot):
        col = grip_colors[grip]
        legend_elements.append(
            Line2D([0], [0], color=col, lw=2, ls="-",
                   label=f"{grip_labels[gi]}  – Proportional MA (standard)"))
        legend_elements.append(
            Line2D([0], [0], color=col, lw=2, ls="--",
                   label=f"{grip_labels[gi]}  – Fixed MA (PeerJ 7470)"))

    fig.legend(handles=legend_elements, loc="lower center",
               ncol=3, fontsize=7.5,
               facecolor="#21262d", edgecolor="#30363d",
               labelcolor="#c9d1d9",
               bbox_to_anchor=(0.5, -0.13))

    fig.suptitle(
        "Finger ray length  vs  required tendon forces\n"
        "100 N tip load · Standard (proportional MA) vs PeerJ 7470 (fixed MA)\n"
        "Middle finger · Literature size range",
        color="#e6edf3", fontsize=11, y=1.02)

    plt.tight_layout(rect=[0, 0.02, 1, 1])

    out = __file__.replace(".py", ".png")
    plt.savefig(out, dpi=200, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"Saved → {out}")
    plt.show()


if __name__ == "__main__":
    make_plot()
