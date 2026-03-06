#!/usr/bin/env python3
"""
Plot the primary study outputs for finger-length dependent tendon-force demand.

Figure 1
- all three major grip types on one plot
- fixed external load
- fixed hold/contact point measured from the fingertip along the distal phalanx
- FDP and FDS shown together

Figure 2
- psychometric-like family of curves for contact-point distances
- default sweep: 0, 2, 4, 6, 8, 10, 12, 14, 18, 20 mm from the fingertip
- one selected grip posture
- FDP and FDS shown together

Figure 3
- heatmap view of the same selected grip
- x-axis: total finger length
- y-axis: hold/contact point from fingertip
- color: required tendon force

The script uses the current reduced solver from
`finger_biomechanics_model.py` directly.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(__file__))

from finger_biomechanics_model import SimConfig, posture_from_joint_targets, solve_static_equilibrium


TOTAL_RAY_MM = np.linspace(60.0, 92.0, 33)
LP_FRAC = 0.41
LM_FRAC = 0.32
LD_FRAC = 0.27
DISTAL_LOAD_N_DEFAULT = 100.0
CONTACT_SWEEP_MM = [0, 2, 4, 6, 8, 10, 12, 14, 18, 20]

GRIP_PARAMS = {
    "open_drag": {"pip": 40.0, "dip": 12.5, "d_abs": 2.5, "label": "Open drag", "color": "#2f7f5f"},
    "half_crimp": {"pip": 75.0, "dip": -20.0, "d_abs": 0.0, "label": "Half crimp", "color": "#1f4e79"},
    "full_crimp": {"pip": 105.0, "dip": -35.0, "d_abs": 0.0, "label": "Full crimp", "color": "#a6402b"},
}


def phalanx_lengths(total_mm: float) -> Tuple[float, float, float]:
    return (
        total_mm * LP_FRAC,
        total_mm * LM_FRAC,
        total_mm * LD_FRAC,
    )


def solve_case(total_mm: float, grip: str, distal_load_n: float, load_point_mm_from_tip: float) -> Dict[str, float]:
    gp = GRIP_PARAMS[grip]
    pose = posture_from_joint_targets(
        phalanx_lengths(total_mm),
        gp["pip"],
        gp["dip"],
        gp["d_abs"],
        0.0,
        grip,
    )
    cfg = SimConfig(load_point_mm_from_tip=load_point_mm_from_tip)
    return solve_static_equilibrium(pose, distal_load_n, cfg)


def sweep_lengths(
    totals_mm: Iterable[float],
    grip: str,
    distal_load_n: float,
    load_point_mm_from_tip: float,
) -> Dict[str, np.ndarray]:
    rows = [solve_case(total, grip, distal_load_n, load_point_mm_from_tip) for total in totals_mm]
    return {
        "FDP_N": np.array([r["FDP_N"] for r in rows]),
        "FDS_N": np.array([r["FDS_N"] for r in rows]),
        "FDP_FDS_ratio": np.array([r["FDP_FDS_ratio"] for r in rows]),
        "A2_N": np.array([r["A2_N"] for r in rows]),
        "A4_N": np.array([r["A4_N"] for r in rows]),
    }


def summarize_grip_comparison(distal_load_n: float, load_point_mm_from_tip: float) -> None:
    short_total = TOTAL_RAY_MM[0]
    avg_total = 77.0
    long_total = TOTAL_RAY_MM[-1]
    print("=== Figure 1 Summary: Grip Comparison ===")
    print(f"External load: {distal_load_n:.1f} N")
    print(f"Contact point: {load_point_mm_from_tip:.1f} mm from fingertip\n")
    for grip in GRIP_PARAMS:
        short_res = solve_case(short_total, grip, distal_load_n, load_point_mm_from_tip)
        avg_res = solve_case(avg_total, grip, distal_load_n, load_point_mm_from_tip)
        long_res = solve_case(long_total, grip, distal_load_n, load_point_mm_from_tip)
        print(
            f"{grip:10s}  "
            f"FDP short/avg/long = {short_res['FDP_N']:.1f} / {avg_res['FDP_N']:.1f} / {long_res['FDP_N']:.1f} N   "
            f"FDS short/avg/long = {short_res['FDS_N']:.1f} / {avg_res['FDS_N']:.1f} / {long_res['FDS_N']:.1f} N"
        )
    print()


def summarize_contact_sweep(grip: str, distal_load_n: float, distances_mm: List[float]) -> None:
    short_total = TOTAL_RAY_MM[0]
    long_total = TOTAL_RAY_MM[-1]
    print("=== Figure 2 Summary: Contact-Point Sweep ===")
    print(f"Grip: {grip}")
    print(f"External load: {distal_load_n:.1f} N\n")
    for distance in distances_mm:
        short_res = solve_case(short_total, grip, distal_load_n, distance)
        long_res = solve_case(long_total, grip, distal_load_n, distance)
        if short_res["FDP_N"] > 1e-6:
            fdp_change_text = f"{100.0 * (long_res['FDP_N'] - short_res['FDP_N']) / short_res['FDP_N']:+.1f}%"
        else:
            fdp_change_text = "n/a"
        print(
            f"contact {distance:4.1f} mm  "
            f"FDP short/long = {short_res['FDP_N']:.1f} / {long_res['FDP_N']:.1f} N   "
            f"FDS short/long = {short_res['FDS_N']:.1f} / {long_res['FDS_N']:.1f} N   "
            f"long-short FDP = {fdp_change_text}"
        )
    print()


def summarize_heatmap(grip: str, distal_load_n: float, distances_mm: List[float]) -> None:
    fdp_grid = []
    fds_grid = []
    for distance in distances_mm:
        data = sweep_lengths(TOTAL_RAY_MM, grip, distal_load_n, distance)
        fdp_grid.append(data["FDP_N"])
        fds_grid.append(data["FDS_N"])
    fdp_grid = np.array(fdp_grid)
    fds_grid = np.array(fds_grid)
    print("=== Figure 3 Summary: Heatmap Range ===")
    print(f"Grip: {grip}")
    print(f"External load: {distal_load_n:.1f} N")
    print(f"FDP range = {fdp_grid.min():.1f} to {fdp_grid.max():.1f} N")
    print(f"FDS range = {fds_grid.min():.1f} to {fds_grid.max():.1f} N")
    print()


def add_reference_lines(ax: plt.Axes) -> None:
    refs = [
        (63.0, "Small", "#8b949e"),
        (77.0, "Average", "#6e7681"),
        (91.0, "Large", "#484f58"),
    ]
    ymin, ymax = ax.get_ylim()
    for x, label, color in refs:
        ax.axvline(x, color=color, lw=0.8, ls=":", alpha=0.8)
        ax.text(x + 0.25, ymax - 0.05 * (ymax - ymin), label, color=color, fontsize=7, va="top")


def plot_figure1(ax: plt.Axes, distal_load_n: float, load_point_mm_from_tip: float) -> None:
    for grip, meta in GRIP_PARAMS.items():
        data = sweep_lengths(TOTAL_RAY_MM, grip, distal_load_n, load_point_mm_from_tip)
        ax.plot(TOTAL_RAY_MM, data["FDP_N"], color=meta["color"], lw=2.4)
        ax.plot(TOTAL_RAY_MM, data["FDS_N"], color=meta["color"], lw=2.0, ls="--")

    ax.set_title(
        f"Figure 1. Grip Comparison\nHold contact = {load_point_mm_from_tip:.1f} mm from fingertip",
        fontsize=11,
    )
    ax.set_xlabel("Total finger ray length Lp + Lm + Ld (mm)")
    ax.set_ylabel("Required tendon force (N)")
    ax.grid(True, alpha=0.25)
    ax.set_xlim(TOTAL_RAY_MM[0], TOTAL_RAY_MM[-1])
    add_reference_lines(ax)

    grip_handles = [
        Line2D([0], [0], color=meta["color"], lw=2.4, label=meta["label"])
        for meta in GRIP_PARAMS.values()
    ]
    tendon_handles = [
        Line2D([0], [0], color="black", lw=2.4, ls="-", label="FDP"),
        Line2D([0], [0], color="black", lw=2.0, ls="--", label="FDS"),
    ]
    leg1 = ax.legend(handles=grip_handles, loc="upper left", fontsize=8, title="Grip")
    ax.add_artist(leg1)
    ax.legend(handles=tendon_handles, loc="lower right", fontsize=8, title="Tendon")


def plot_figure2(ax: plt.Axes, grip: str, distal_load_n: float, distances_mm: List[float]) -> None:
    cmap = plt.get_cmap("viridis")
    colors = [cmap(i) for i in np.linspace(0.08, 0.95, len(distances_mm))]

    for distance, color in zip(distances_mm, colors):
        data = sweep_lengths(TOTAL_RAY_MM, grip, distal_load_n, distance)
        ax.plot(TOTAL_RAY_MM, data["FDP_N"], color=color, lw=2.1)
        ax.plot(TOTAL_RAY_MM, data["FDS_N"], color=color, lw=1.8, ls="--")

    ax.set_title(
        f"Figure 2. Contact-Point Family ({GRIP_PARAMS[grip]['label']})",
        fontsize=11,
    )
    ax.set_xlabel("Total finger ray length Lp + Lm + Ld (mm)")
    ax.set_ylabel("Required tendon force (N)")
    ax.grid(True, alpha=0.25)
    ax.set_xlim(TOTAL_RAY_MM[0], TOTAL_RAY_MM[-1])
    add_reference_lines(ax)

    contact_handles = [
        Line2D([0], [0], color=color, lw=2.1, label=f"{distance} mm")
        for distance, color in zip(distances_mm, colors)
    ]
    tendon_handles = [
        Line2D([0], [0], color="black", lw=2.1, ls="-", label="FDP"),
        Line2D([0], [0], color="black", lw=1.8, ls="--", label="FDS"),
    ]
    leg1 = ax.legend(handles=contact_handles, loc="upper left", fontsize=7, title="Contact point", ncol=2)
    ax.add_artist(leg1)
    ax.legend(handles=tendon_handles, loc="lower right", fontsize=8, title="Tendon")


def plot_heatmap(
    ax: plt.Axes,
    grip: str,
    distal_load_n: float,
    distances_mm: List[float],
    key: str,
    title: str,
    cmap: str,
) -> None:
    grid = np.array([
        sweep_lengths(TOTAL_RAY_MM, grip, distal_load_n, distance)[key]
        for distance in distances_mm
    ])
    im = ax.imshow(
        grid,
        aspect="auto",
        origin="lower",
        extent=[TOTAL_RAY_MM[0], TOTAL_RAY_MM[-1], distances_mm[0], distances_mm[-1]],
        cmap=cmap,
    )
    ax.set_title(title, fontsize=10.5)
    ax.set_xlabel("Total finger ray length Lp + Lm + Ld (mm)")
    ax.set_ylabel("Contact point from fingertip (mm)")
    ax.grid(False)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Force (N)")


def make_plot(distal_load_n: float, load_point_mm_from_tip: float, contact_sweep_grip: str, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), constrained_layout=True)
    plot_figure1(axes[0, 0], distal_load_n, load_point_mm_from_tip)
    plot_figure2(axes[0, 1], contact_sweep_grip, distal_load_n, CONTACT_SWEEP_MM)
    plot_heatmap(
        axes[1, 0],
        contact_sweep_grip,
        distal_load_n,
        CONTACT_SWEEP_MM,
        "FDP_N",
        f"Figure 3A. FDP Heatmap ({GRIP_PARAMS[contact_sweep_grip]['label']})",
        "YlOrRd",
    )
    plot_heatmap(
        axes[1, 1],
        contact_sweep_grip,
        distal_load_n,
        CONTACT_SWEEP_MM,
        "FDS_N",
        f"Figure 3B. FDS Heatmap ({GRIP_PARAMS[contact_sweep_grip]['label']})",
        "PuBuGn",
    )
    fig.suptitle(
        "Finger-Length Dependent Force Demand on a Given Hold",
        fontsize=13,
    )
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    print(f"Saved → {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--distal-load-n",
        type=float,
        default=DISTAL_LOAD_N_DEFAULT,
        help="External load magnitude in newtons.",
    )
    parser.add_argument(
        "--load-point-mm-from-tip",
        type=float,
        default=8.0,
        help="Hold contact point for Figure 1, measured proximally from the fingertip along the distal phalanx.",
    )
    parser.add_argument(
        "--contact-sweep-grip",
        "--grip",
        dest="contact_sweep_grip",
        choices=["open_drag", "half_crimp", "full_crimp"],
        default="half_crimp",
        help="Grip used for Figure 2 contact-point family.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output PNG path.",
    )
    return parser.parse_args()


def default_output_path(contact_sweep_grip: str, load_point_mm_from_tip: float) -> Path:
    stem = Path(__file__).stem
    contact_part = f"{load_point_mm_from_tip:.1f}mm".replace(".", "p")
    out_dir = Path(__file__).resolve().parent / "img"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{stem}_study_{contact_sweep_grip}_{contact_part}.png"


def main() -> None:
    args = parse_args()
    output = Path(args.output) if args.output else default_output_path(args.contact_sweep_grip, args.load_point_mm_from_tip)

    summarize_grip_comparison(args.distal_load_n, args.load_point_mm_from_tip)
    summarize_contact_sweep(args.contact_sweep_grip, args.distal_load_n, CONTACT_SWEEP_MM)
    summarize_heatmap(args.contact_sweep_grip, args.distal_load_n, CONTACT_SWEEP_MM)
    make_plot(args.distal_load_n, args.load_point_mm_from_tip, args.contact_sweep_grip, output)


if __name__ == "__main__":
    main()
