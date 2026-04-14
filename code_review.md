# Code Review: Iteration 11 — README §4.7 Correction + Fig 9 Extension

## Summary

Two targeted corrections based on the computed crossover depth data:

1. **README §4.7 entirely rewritten** — old crossover table was based on a pre-frac_DP model and contained incorrect values
2. **Fig 9 (Full Crimp) extended from 22 mm → 30 mm** — shows regime transition beyond L_DP

---

## 1. README §4.7 Correction

### What was wrong

The old table stated:

| Phenotype | Crossover |
|-----------|-----------|
| Short (−15%) | ~19 mm |
| Standard | ~22 mm (= L_DP) |
| Long (+15%) | ~25 mm |

These values came from an earlier model version using a `d_hold`-based interpolation with
fixed breakpoints at L_DP. They implied that **all three grips** (crimp, half-crimp, open-hand)
share the same crossover depth — which is physiologically incorrect.

### What the current model shows (Iteration 11 diagnostic)

**Open Hand** — NO crossover across 2–45 mm for any phenotype.  
Because r_base = 0.88 < 1.0, FDS already exceeds FDP from the shallowest hold.
The "crossover" concept does not apply to open-hand posture in this model.

**Half-Crimp** — crossover exists but at greater depths than previously stated:

| Phenotype | Crossover | L_DP |
|-----------|-----------|------|
| Short (−15%) | 26.3 mm | 18.7 mm |
| Standard | 31.9 mm | 22.0 mm |
| Long (+15%) | No crossover (≤35 mm) | 25.3 mm |

**Full Crimp** — No crossover within physiological range (FDP dominant throughout,
ratio never drops to 1.0 within ≤ L_DP hold depth).

### Why the frac_DP model changes this

With `r_emg = r_base * (0.20 + 0.80 * frac_DP)`, the FDP/FDS ratio reaches 1.0 when:
```
frac_DP = (1/r_base - 0.20) / 0.80
```
- Open hand (r=0.88): frac_DP = (1.136−0.20)/0.80 = 1.17 → impossible (>1) → no crossover
- Half-crimp (r=1.20): frac_DP = (0.833−0.20)/0.80 = 0.791 → crossover at ~79% DP load
- Crimp (r=1.75): frac_DP = (0.571−0.20)/0.80 = 0.464 → crossover at ~46% DP load (never in range)

This is physically correct and more nuanced than the previous table.

---

## 2. Fig 9 Extension (22 mm → 30 mm)

Changing `d_max=22.0` to `d_max=30.0` in the `plot_grip_depth_sweep('crimp', ...)` call.

**Rationale**: The previous cutoff at exactly L_DP = 22 mm hid the force transition that
occurs when the hold exceeds the DP length in crimp posture. Extending to 30 mm shows:
- The purple MP-engagement shading begins at L_DP
- The frac_DP starts declining, causing the ratio to drop below 1.75
- The posture panel (D) shows how PIP/DIP angles respond to this new load regime

The crimp grip-mode DIP ceiling constraint (DIP ceiling = −25° + 20° = −5°) prevents the
optimizer from jumping into open-hand basin even at 30 mm — so the plot remains a
valid "crimp grip at deep hold" simulation.

---

## Files Changed

- `climbing_finger_3d.py` — `fig9` call: `d_max=22.0 → d_max=30.0`
- `README.md` — §4.7 rewritten with correct crossover data and grip-specific analysis
- `code_review.md` — this file

**Status: APPROVED — all 11 figures regenerated, README corrected**
