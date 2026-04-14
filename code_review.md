# Code Review: Iteration 8 — EMG Ratio Variable Fix + Grip Depth Sweeps

## Summary

Iteration 8 contains two deliverables:

1. **Bug Fix** — `get_emg_ratio()` receives `d_hold` (raw hold depth) instead of `d_eff`
   (arc-length centroid). This is a correctness fix with no physics model change.
2. **Feature** — Per-grip depth-sweep figures (Figs 9 and 10) for Full Crimp and Half Crimp,
   using a shared `plot_grip_depth_sweep()` helper refactored from the old Fig 8 code.

---

## Bug Fix: `get_emg_ratio` incorrect variable

### Root cause

`compute_contact_point` returns two quantities:

| Variable | Meaning |
|----------|---------|
| `d_hold` | Raw geometric hold depth (mm) — what the climber grips |
| `d_eff` | Arc-length centroid of pressure distribution along palmar surface |

In a **crimp posture** (DIP ≈ −25°, PIP ≈ 120°), the DP is nearly parallel to the wall
surface. The angle-projection formula amplifies `d_eff` relative to `d_hold`:

```
d_eff = proj_d_hold / cos_alpha3
      = (d_hold + r_edge·|dot(e_DP, n_hold)|) / cos_alpha3
```

At crimp with `cos_alpha3 ≈ 0.4` and `r_edge = 2 mm`:

```
d_hold = 11 mm → proj_d_hold ≈ 12 mm → max_d_hold_DP ≈ 22 × 0.4 = 8.8 mm
```

So at `d_hold = 11 mm` the deep-hold code path is triggered, setting
`d_eff = L_DP + engaged_MP = 22 + 6 = 28 mm`. The interpolation
`get_emg_ratio(r_base=1.75, d_eff=28, L_DP=22)` then returned **1.33** — a 24% error.

### Fix

```python
# Before (wrong):
r_emg = get_emg_ratio(grip_base.emg_ratio, d_eff, geom.L3, geom.L2)
c_info = {'d_eff': d_eff}
r_emg = get_emg_ratio(grip.emg_ratio, c_info['d_eff'] if contact else None, ...)

# After (correct):
r_emg = get_emg_ratio(grip_base.emg_ratio, contact.d_hold, geom.L3, geom.L2)
c_info = {'d_hold': contact.d_hold}
r_emg = get_emg_ratio(grip.emg_ratio, c_info['d_hold'] if contact else None, ...)
```

Also updated function signature: `get_emg_ratio(r_base, d_hold, ...)`.

### Verification

```
CRIMP ratio at d_hold=12mm:  was 1.327 → now 1.750  ✓
CRIMP ratio at d_hold=22mm:  was 1.108 → now 1.750  ✓  (DP-only regime)
CRIMP ratio at d_hold=28mm:  1.337                   ✓  (MP engaged, transition)
CRIMP ratio at d_hold=35mm:  0.856                   ✓  (FDS dominates)
OPEN HAND at d_hold=10mm:    0.880                   ✓  (no regression)
Main table (contact=None):   all ratios unchanged    ✓  (code path unaffected)
```

---

## Feature: Grip Depth Sweeps Figs 9 and 10

### Design

Old inline Fig 8 code (~120 lines) replaced with `plot_grip_depth_sweep()` helper:

```python
fig8  = plot_grip_depth_sweep('open_hand',  d_max=45.0, fig_label='Open Hand')
fig9  = plot_grip_depth_sweep('crimp',      d_max=22.0, fig_label='Full Crimp')
fig10 = plot_grip_depth_sweep('half_crimp', d_max=35.0, fig_label='Half Crimp')
```

Crimp capped at 22 mm (= L_DP): beyond this the crimp posture degrades into open-hand
geometry — a full-crimp hold deeper than the DP is anatomically inconsistent.

Each figure: 3 panels (force / FDP:FDS ratio / A2 pulley load), 3 phenotypes
(Short −15% / Standard / Long +15%), warm-start depth continuation, global grid fallback.

### Grip-mode DIP ceiling

A soft DIP ceiling of `grip_base.theta_DIP + 20°` prevents the optimizer from jumping
from open-hand into the crimp basin during a sweep. For crimp (nominal DIP = −25°),
the ceiling = −5°, which correctly blocks the optimizer from flipping to high-DIP postures
while sweeping crimp hold depths.

---

## Files Changed

- `climbing_finger_3d.py` — `get_emg_ratio()`, `find_equilibrium_posture()` inner solver,
  `solve_all_methods()`, Fig 8 refactored to helper, Figs 9–10 added
- `physics.md` — §3.5 added (EMG ratio variable derivation)
- `README.md` — Figure count updated (8→10), Discussion entries added for Iter 8

**Status: APPROVED — all 10 figures generated, EMG ratios verified**
