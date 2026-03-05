"""
plot_ld_loads.py
================
Visualise how FDP, FDS and A2 pulley load change with distal phalanx
length (Ld) for a half-crimp at fixed joint angles and body weight.

Usage
-----
    python plot_ld_loads.py                    # defaults
    python plot_ld_loads.py --bw 70 --pip 80   # custom body weight / PIP angle
    python plot_ld_loads.py --out loads.png    # save instead of show

Design
------
Computation  → compute_ld_series()   pure function, no side-effects
Data model   → LdSeries dataclass     carries everything the plot needs
Visualisation→ plot_ld_series()       knows nothing about biomechanics
Entry point  → main()                 wires CLI → compute → plot
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# ── Import computation primitives from the main model ─────────────────────────
try:
    from finger_biomechanics_model_claude import (
        posture_from_joint_targets,
        build_tendon_geometry,
        contact_force_vector,
        tendon_excursion_moment_arms,
        passive_moment_Nm,
        cross2,
    )
except ImportError as exc:
    sys.exit(f"Cannot import finger_biomechanics_model: {exc}\n"
             "Run this script from the same directory as finger_biomechanics_model.py")

try:
    import matplotlib
    if "--out" in sys.argv:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    from matplotlib.lines import Line2D
except ImportError:
    sys.exit("matplotlib is required: pip install matplotlib")


# ═══════════════════════════════════════════════════════════════════════════════
# Data model
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class LdSeries:
    """One curve-set: loads vs Ld for a single hold depth."""
    hold_depth_mm: float
    ld_mm:  np.ndarray          # shape (N,)
    fdp_N:  np.ndarray          # shape (N,)
    fds_N:  np.ndarray          # shape (N,)
    a2_N:   np.ndarray          # shape (N,)


@dataclass
class SimParams:
    """All parameters that drive a computation run."""
    bw_kg:          float = 65.0
    grip_frac:      float = 0.15          # grip force as fraction of BW
    lp_mm:          float = 22.0          # proximal phalanx (fixed)
    lm_mm:          float = 21.0          # middle phalanx (fixed)
    pip_flex_deg:   float = 75.0
    dip_flex_deg:   float = -20.0
    grip_label:     str   = "half_crimp"
    ld_min_mm:      float = 14.0
    ld_max_mm:      float = 34.0
    ld_step_mm:     float = 0.1
    hold_depths_mm: List[float] = field(default_factory=lambda: [6, 10, 15, 20])
    a2_threshold_N: float = 400.0


# ═══════════════════════════════════════════════════════════════════════════════
# Computation layer  (pure: no plotting, no I/O)
# ═══════════════════════════════════════════════════════════════════════════════

def _forces_at(lp, lm, ld, pip, dip, grip_label, f_mag, hold_depth_mm) -> tuple[float, float, float]:
    """Return (FDP, FDS, A2) in Newtons for one (Ld, hold_depth) combination."""
    pose = posture_from_joint_targets(
        (lp, lm, ld), pip, dip, 0.0, 0.0, grip_label
    )
    geo    = build_tendon_geometry(pose)
    dip_pt = geo["DIP"] * 1e-3
    tip_pt = geo["TIP"] * 1e-3
    ld_act = float(np.linalg.norm(tip_pt - dip_pt) * 1000)          # mm

    contact_mm = max(ld_act - hold_depth_mm, 0.0)
    contact    = dip_pt + (contact_mm * 1e-3) * geo["uD"]

    F    = contact_force_vector(pose, f_mag)
    arms = tendon_excursion_moment_arms(pose)

    r_fdp_dip = arms["r_fdp_dip"] * 1e-3
    r_fdp_pip = arms["r_fdp_pip"] * 1e-3
    r_fds_pip = arms["r_fds_pip"] * 1e-3
    M_pass    = passive_moment_Nm("PIP", pose.theta_p - pose.theta_m)

    M_dip = abs(cross2(contact - dip_pt, F))
    M_pip = abs(cross2(contact - geo["PIP"] * 1e-3, F))

    fdp = M_dip / max(r_fdp_dip, 1e-6)
    fds = max((M_pip - fdp * r_fdp_pip - M_pass) / max(r_fds_pip, 1e-6), 0.0)

    pip_flex  = max(pose.theta_p - pose.theta_m, 0.0)
    dip_flex  = pose.theta_m - pose.theta_d
    wrap_a2   = np.deg2rad(0.44 * pip_flex)
    wrap_a4   = np.deg2rad(0.25 * pip_flex + 0.25 * max(dip_flex, 0.0))
    a2 = (fdp + fds) * 2.0 * np.sin(wrap_a2 / 2.0)

    return fdp, fds, a2


def compute_ld_series(params: SimParams) -> List[LdSeries]:
    """
    Compute FDP / FDS / A2 across the Ld range for every hold depth.

    Returns
    -------
    list of LdSeries, one per hold depth.
    """
    f_mag  = params.grip_frac * params.bw_kg * 9.81
    ld_arr = np.arange(params.ld_min_mm, params.ld_max_mm + 1e-9, params.ld_step_mm)

    series: List[LdSeries] = []
    for depth in params.hold_depths_mm:
        fdp_arr = np.empty_like(ld_arr)
        fds_arr = np.empty_like(ld_arr)
        a2_arr  = np.empty_like(ld_arr)
        for i, ld in enumerate(ld_arr):
            fdp_arr[i], fds_arr[i], a2_arr[i] = _forces_at(
                params.lp_mm, params.lm_mm, ld,
                params.pip_flex_deg, params.dip_flex_deg,
                params.grip_label, f_mag, depth,
            )
        series.append(LdSeries(
            hold_depth_mm=depth,
            ld_mm=ld_arr,
            fdp_N=fdp_arr,
            fds_N=fds_arr,
            a2_N=a2_arr,
        ))
    return series


# ═══════════════════════════════════════════════════════════════════════════════
# Visualisation layer  (pure: receives data, produces figure)
# ═══════════════════════════════════════════════════════════════════════════════

# Colour palette: one hue per hold depth, consistent across FDP/FDS/A2
_DEPTH_COLORS = {
    6:  "#e05a3a",
    10: "#f59e0b",
    15: "#4a9eff",
    20: "#22d3a0",
}
_LINESTYLES = {"fdp": "-", "fds": "--", "a2": ":"}
_LINEWIDTHS = {"fdp": 1.6, "fds": 1.4, "a2": 2.2}

# Canonical Ld reference marks
_REF_LDS = [(17, "short\n17 mm"), (24, "avg\n24 mm"), (29, "long\n29 mm")]


def plot_ld_series(
    series_list: List[LdSeries],
    params: SimParams,
    out_path: Optional[Path] = None,
) -> None:
    """
    Render the FDP / FDS / A2 vs Ld figure.

    Parameters
    ----------
    series_list : output of compute_ld_series()
    params      : SimParams used to produce the data (for annotations)
    out_path    : if given, save PNG/PDF there; otherwise plt.show()
    """
    fig, axs = plt.subplots(
        1, len(series_list),
        figsize=(4.2 * len(series_list), 5.2),
        sharey=True,
        squeeze=False,
    )
    axes = axs[0]

    # --- global y-range (all depths, all three curves) ----------------------
    all_vals = np.concatenate(
        [np.concatenate([s.fdp_N, s.fds_N, s.a2_N]) for s in series_list]
    )
    y_max = np.ceil(all_vals.max() / 100) * 100

    for ax, s in zip(axes, series_list):
        col = _DEPTH_COLORS.get(int(s.hold_depth_mm), "#aaaaaa")

        ax.plot(s.ld_mm, s.fdp_N, linestyle="-",  lw=1.6, color=col,
                alpha=0.9, label="FDP")
        ax.plot(s.ld_mm, s.fds_N, linestyle="--", lw=1.4, color=col,
                alpha=0.75, label="FDS")
        ax.plot(s.ld_mm, s.a2_N,  linestyle=":",  lw=2.2, color=col,
                alpha=1.0, label="A2")

        # A2 injury threshold
        ax.axhline(params.a2_threshold_N, color="#f59e0b", lw=1.0,
                   ls=(0, (6, 4)), alpha=0.7, zorder=1)
        ax.text(s.ld_mm[-1] - 0.2, params.a2_threshold_N + 8,
                "A2 threshold", ha="right", va="bottom",
                fontsize=7, color="#f59e0b", alpha=0.85)

        # Reference Ld verticals
        for ld_ref, lbl in _REF_LDS:
            ax.axvline(ld_ref, color="#555", lw=0.7, ls="--", alpha=0.5, zorder=0)
            ax.text(ld_ref, y_max * 0.97, lbl, ha="center", va="top",
                    fontsize=6.5, color="#888", linespacing=1.3)

        # Fill A2 region above threshold
        ax.fill_between(s.ld_mm, params.a2_threshold_N, s.a2_N,
                        where=(s.a2_N >= params.a2_threshold_N),
                        color="#f59e0b", alpha=0.10, zorder=0)

        ax.set_title(f"Hold depth  {s.hold_depth_mm:.0f} mm",
                     fontsize=9, fontweight="semibold", pad=6)
        ax.set_xlabel("Distal phalanx length  Ld (mm)", fontsize=8)
        ax.set_xlim(s.ld_mm[0], s.ld_mm[-1])
        ax.set_ylim(0, y_max * 1.05)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(100))
        ax.tick_params(axis="both", labelsize=7.5)
        ax.grid(axis="y", lw=0.4, alpha=0.4)
        ax.grid(axis="x", which="minor", lw=0.2, alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("Tendon / pulley load (N)", fontsize=8)

    # --- shared legend -------------------------------------------------------
    legend_elements = [
        Line2D([0], [0], ls="-",  lw=1.6, color="#aaa", label="FDP"),
        Line2D([0], [0], ls="--", lw=1.4, color="#aaa", label="FDS"),
        Line2D([0], [0], ls=":",  lw=2.2, color="#aaa", label="A2 pulley"),
    ] + [
        Line2D([0], [0], ls="-", lw=3,
               color=_DEPTH_COLORS.get(int(s.hold_depth_mm), "#aaa"),
               label=f"{s.hold_depth_mm:.0f} mm hold")
        for s in series_list
    ]
    fig.legend(handles=legend_elements, loc="lower center",
               ncol=len(legend_elements), fontsize=7.5,
               frameon=False, bbox_to_anchor=(0.5, -0.01))

    # --- suptitle ------------------------------------------------------------
    grip_n = params.grip_frac * params.bw_kg * 9.81
    fig.suptitle(
        f"FDP · FDS · A2 load vs distal phalanx length\n"
        f"Half-crimp  PIP={params.pip_flex_deg:.0f}°  DIP={params.dip_flex_deg:.0f}°  "
        f"BW={params.bw_kg:.0f} kg  grip={grip_n:.0f} N  "
        f"Lp={params.lp_mm:.0f} mm  Lm={params.lm_mm:.0f} mm",
        fontsize=8.5, y=1.01,
    )

    plt.tight_layout(rect=[0, 0.06, 1, 1])

    if out_path is not None:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {out_path}")
    else:
        plt.show()

    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot FDP/FDS/A2 vs distal phalanx length (Ld)."
    )
    p.add_argument("--bw",    type=float, default=65.0,
                   metavar="KG",  help="Body weight in kg (default 65)")
    p.add_argument("--pip",   type=float, default=75.0,
                   metavar="DEG", help="PIP flexion angle in degrees (default 75)")
    p.add_argument("--dip",   type=float, default=-20.0,
                   metavar="DEG", help="DIP flex angle relative to middle (default -20)")
    p.add_argument("--depths", type=float, nargs="+", default=[6, 10, 15, 20],
                   metavar="MM", help="Hold depths in mm (default: 6 10 15 20)")
    p.add_argument("--ld-min", type=float, default=14.0, metavar="MM")
    p.add_argument("--ld-max", type=float, default=34.0, metavar="MM")
    p.add_argument("--out",   type=Path,  default=None,
                   metavar="FILE", help="Save plot to file (PNG/PDF) instead of displaying")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    params = SimParams(
        bw_kg          = args.bw,
        pip_flex_deg   = args.pip,
        dip_flex_deg   = args.dip,
        hold_depths_mm = args.depths,
        ld_min_mm      = args.ld_min,
        ld_max_mm      = args.ld_max,
    )

    print(f"Computing loads for BW={params.bw_kg} kg, PIP={params.pip_flex_deg}°, "
          f"DIP={params.dip_flex_deg}°, "
          f"holds={params.hold_depths_mm} mm, "
          f"Ld {params.ld_min_mm}–{params.ld_max_mm} mm …")

    series = compute_ld_series(params)
    plot_ld_series(series, params, out_path=args.out)


if __name__ == "__main__":
    main()
