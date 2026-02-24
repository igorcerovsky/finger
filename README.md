# Middle Finger Climbing Biomechanics Model

## Purpose
This project implements a 2D static biomechanics model of the **middle finger** for climbing grips, focused on:
- FDP and FDS tendon force estimation.
- A2/A3/A4 pulley force redirection.
- Grip-to-grip comparison (`open_drag`, `half_crimp`, `full_crimp`).
- Geometry advantage analysis (short vs long fingers).

The implementation is in:
- `/Users/igorcerovsky/Documents/finger/finger_biomechanics_model.py`

Generated visualization:
- `/Users/igorcerovsky/Documents/finger/finger_biomechanics_forces.png`

## Original Prompt Coverage
The model was structured directly from the original request.

1. **Middle finger model in climbing context**
- Implemented with climbing grip presets and climber mass/length cases.

2. **Use recent studies and compare to published values**
- Literature links embedded in code and documented below.
- Benchmark section reports model-vs-study checks.

3. **Visualize finger position and force vectors**
- Bone segments, tendon paths, external force, and pulley vectors are plotted.

4. **Inputs: phalanx lengths and phalanx angles**
- Core pose object uses P/M/D lengths and segment angles.
- Grip presets are configurable through `posture_from_joint_targets(...)`.

5. **Grips: open drag, half crimp, full crimp**
- All three are implemented as defaults.
- Half crimp keeps distal phalanx aligned with +x as requested.

6. **FDP/FDS + pulleys + static equilibrium**
- DIP/PIP moment equilibrium solved for tendon tensions.
- A2/A3/A4 pulley loads are derived from tendon redirection vectors.

7. **Compare with studies and quantify geometry advantage**
- Printed benchmark section (ratios and pulley loads).
- Printed short-vs-long force deltas at matched external load.

8. **Force orientation setup**
- Global frame uses +y up, with load vector in +y.
- Load-point switch supports `distal_mid` (realistic scenario) and `fingertip` (study comparison).

## Modeling Decisions and Reasoning
## 1) Kinematics and Coordinate System
- 2D planar model in global `(x, y)`.
- Finger points upward when straight.
- MCP fixed at origin.
- Absolute segment angles are used for proximal, middle, distal phalanges.

## 2) Tendons and Moment Arms
- FDP insertion near distal phalanx (`FDP_INS`).
- FDS insertion near middle phalanx (`FDS_INS`).
- Geometric moment arms are computed from perpendicular distance to tendon lines.
- A hybrid moment-arm model blends geometry with bounded empirical ranges to prevent non-physiological collapse in extreme joint configurations.

## 3) Static Equilibrium
External load is applied at selected contact point (`fingertip` or `distal_mid`):
- `M_DIP = |(r_contact - r_DIP) x F_ext|`
- `M_PIP = |(r_contact - r_PIP) x F_ext|`
- `FDP = M_DIP_active / r_fdp_dip`
- `FDS = (M_PIP - FDP * r_fdp_pip) / r_fds_pip`

Where:
- `M_DIP_active = (1 - dip_passive_fraction) * M_DIP`
- `dip_passive_fraction` lets crimp postures include passive DIP support behavior.

## 4) Pulley Loads
- Tendon direction change at each pulley gives a resultant vector.
- A2 and A3 include both FDP and FDS contributions.
- A4 includes FDP contribution.
- Magnitudes are reported in Newtons.

## 5) Load-Point Switch
CLI switch:
- `--load-point distal_mid` (default): main simulation at middle of distal phalanx.
- `--load-point fingertip`: main simulation at fingertip.

Important implementation rule:
- **Literature benchmark block is always evaluated with fingertip loading** so comparison is aligned with common experimental setups.

## 6) A2/A4 Absolute-Value Calibration for Literature Comparison
Raw model pulley magnitudes can diverge from cadaver absolute anchors even if trends are correct.  
To compare absolute levels, a two-point power-law mapping is applied in the benchmark section:
- `y = k * x^n`
- Fit with open/crimp anchors from Schweizer (2009):
  - A2: `121 N` (slope/open) and `287 N` (crimp)
  - A4: `103 N` (slope/open) and `226 N` (crimp)

The benchmark output prints:
- Raw A2/A4 (`open/half/full`)
- Calibrated A2/A4 (`open/half/full`)

This keeps physics output transparent while enabling direct absolute-value comparison.

## How To Run
From `/Users/igorcerovsky/Documents/finger`:

```bash
python3 finger_biomechanics_model.py
```

Main run with fingertip loading:

```bash
python3 finger_biomechanics_model.py --load-point fingertip
```

Main run with distal-mid loading (explicit):

```bash
python3 finger_biomechanics_model.py --load-point distal_mid
```

## Output Summary
The script prints:
- Per-athlete, per-grip: `FDP`, `FDS`, `FDP/FDS`, `A2`, `A3`, `A4`, moment arms.
- Main-run ratio summary.
- Fingertip-based benchmark comparison vs literature.
- Geometry advantage summary for short vs long finger cases at equal mass/load.

The script also saves:
- `/Users/igorcerovsky/Documents/finger/finger_biomechanics_forces.png`

## Work Done (Documented Revision History)
1. Implemented full 2D middle-finger static model with FDP/FDS and A2/A3/A4.
2. Added grip presets and corrected half-crimp orientation requirement (distal phalanx on +x).
3. Added robust geometry checks and force-vector plotting.
4. Added climber cases (short/average/long) and mass scaling.
5. Added literature references and printed benchmark checks.
6. Fixed issue where short-vs-long FDP difference was constant by improving moment-arm blending.
7. Corrected open-drag DIP flexion direction.
8. Moved external load application to distal-middle contact point for realistic scenario.
9. Added load-point switch (`distal_mid` / `fingertip`).
10. Forced benchmark comparison to fingertip load mode (study-aligned).
11. Added A2/A4 raw and calibrated outputs for absolute literature comparison.
12. Added two-point power-law A2/A4 calibration to match published anchors.

## Known Limits
- Planar 2D model (no out-of-plane mechanics).
- No explicit extensor/lumbrical/interossei force balance.
- Pulley geometry is simplified and parameterized.
- Absolute pulley loads depend on assumptions and calibration mapping.

## References
- Vigouroux et al., J Biomech (2006): [https://doi.org/10.1016/j.jbiomech.2005.10.034](https://doi.org/10.1016/j.jbiomech.2005.10.034)
- Schweizer, J Hand Surg Am (2001): [https://doi.org/10.1053/jhsu.2001.26322](https://doi.org/10.1053/jhsu.2001.26322)
- Schweizer, J Biomech (2009): [https://pubmed.ncbi.nlm.nih.gov/19367698/](https://pubmed.ncbi.nlm.nih.gov/19367698/)
- Ki et al., BMC Sports Sci Med Rehabil (2024): [https://bmcsportsscimedrehabil.biomedcentral.com/articles/10.1186/s13102-024-01096-y](https://bmcsportsscimedrehabil.biomedcentral.com/articles/10.1186/s13102-024-01096-y)
- Schöffl et al., Diagnostics (2021): [https://pmc.ncbi.nlm.nih.gov/articles/PMC8159322/](https://pmc.ncbi.nlm.nih.gov/articles/PMC8159322/)
- Requested reference mirror (Vigouroux context): [http://julienbruyer.free.fr/M2/Escalade/Articles/sdarticle(2).pdf](http://julienbruyer.free.fr/M2/Escalade/Articles/sdarticle(2).pdf)
