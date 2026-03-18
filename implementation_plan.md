# Deep Hold Biomechanics – Implementation Plan

Extend [climbing_finger_3d.py](file:///Users/igorcerovsky/Documents/finger/climbing_finger_3d.py) to simulate hold depths exceeding the Distal Phalanx (DP), distributing contact force across both DP and MP with physiologically realistic assumptions. The goal is a general model capable of answering questions about phenotype/genotype advantages for climbers (bone proportion, moment arm ratios).

## User Review Required

> [!IMPORTANT]
> **Minimum-effort solver** uses `scipy.optimize.minimize` (L-BFGS-B). Must be available in `.venv`.

> [!NOTE]
> **Projected contact length** is used for the DP/MP split (angle-independent). A TODO is placed in code for future angle-dependent projection.

---

## Proposed Changes

### Core biomechanics — [climbing_finger_3d.py](file:///Users/igorcerovsky/Documents/finger/climbing_finger_3d.py)

#### [MODIFY] [climbing_finger_3d.py](file:///Users/igorcerovsky/Documents/finger/climbing_finger_3d.py)

---

**A. Fix existing NameError (line 478)**
`p_C` → `p_C_DP` in [solve_all_methods](file:///Users/igorcerovsky/Documents/finger/climbing_finger_3d.py#392-482) return dict.

---

**B. Weighted (Hertzian-like) pressure distribution**

Replace uniform fraction with a **triangular distribution** peaking at the distal edge (fingertip) and tapering toward the DIP crease. Consistent with Hertz contact theory (Johnson 1985) and the higher deformability of the fingertip pad (Johansson & Flanagan 2009).

![Pressure distribution on DP and MP during deep hold contact](/Users/igorcerovsky/.gemini/antigravity/brain/42bbfbd3-85d8-4484-af6a-64f8b8d0cd02/pressure_distribution_diagram_1773821412463.png)

_Figure B: Triangular (Hertz-like) pressure profile p(s) along the contact. The resultant F_DP acts at L/3 from the tip; F_MP centroid is L_MP/3 from DIP._

**Mathematics:**
```
area_DP  = L3 / 2                                # ∫₀^L3 (1 - s/L3) ds
area_MP  = engaged_MP² / (2 × total_engaged)     # ∫₀^x (s/total) ds
frac_DP  = area_DP / (area_DP + area_MP)
frac_MP  = 1 - frac_DP

p_C_DP centroid = p_TIP_palmar − (L3/3) × e_DP         # ⅓ from tip
p_C_MP centroid = p_DIP_palmar − (engaged_MP/3) × e_MP  # ⅓ from DIP side
```

**References:**
- Johnson, K.L. (1985). *Contact Mechanics*. Cambridge University Press.
- Johansson R.S. & Flanagan J.R. (2009). Coding and use of tactile signals from the fingertips in object manipulation tasks. *Nature Reviews Neuroscience*, 10, 345–359.
- Serina E.R. et al. (1997). Rate effects on finger pad force–displacement during repetitive fingertip loading. *Journal of Biomechanics*, 30(2), 111–118.

---

**C. Pulley-aware MP contact centroid**

The A3 pulley (at ~15% of MP from PIP end, i.e. PIP-MP junction) is the primary skeletal anchor for skin traction on the proximal portion of the MP. The volar plate reinforces this. The effective force application point is a weighted mix of the geometric centroid and the A3 position.

![Annular pulley system and force transfer to skeleton via A3](/Users/igorcerovsky/.gemini/antigravity/brain/42bbfbd3-85d8-4484-af6a-64f8b8d0cd02/pulley_centroid_diagram_1773821433426.png)

_Figure C: A2 (over PP) and A3 (at PIP–MP junction, highlighted) are the key pulleys for force transfer when the hold engages the MP. The pulley-weighted centroid p\_C\_MP is placed closer to A3 than the pure geometric midpoint._

**Mathematics:**
```
p_A3         = p_PIP + R_PIP @ (0.15 × L2 × e_x)
geom_centroid = p_DIP_palmar − (engaged_MP / 2) × e_MP
p_C_MP       = 0.40 × geom_centroid + 0.60 × p_A3
```

> [!NOTE]
> Future: include A2 contribution for very deep holds engaging the PP. Also account for angle-dependent projected contact length.

**References:**
- Doyle J.R. & Blythe W. (1984). The finger flexor tendon sheath and pulleys: anatomy and reconstruction. *Hand*, 16, 419–426.
- Moutet F. (2003). Flexor tendon pulley system. *Hand Clinics*, 19(2), 168–175.
- Vigouroux L. et al. (2006). Estimation of finger muscle tendon forces during specific sport-climbing grip techniques. *Journal of Biomechanics*, 39(14), 2583–2599.

---

**D. Minimum-effort solver (new method: `'min_effort'`)**

Uses `scipy.optimize.minimize` (L-BFGS-B) to find **[F_FDP, F_FDS, F_LU]** minimising the quadratic cost (metabolic surrogate) subject to static equilibrium equality constraints.

```
Objective:   min f(x) = x[0]² + x[1]² + x[2]²
Constraints: A3 @ x == b3   (DIP, PIP, MCP equilibrium)
Bounds:      x[i] >= 0
```

This criterion is physiologically motivated: it reflects the nervous system's tendency to distribute effort across muscles to avoid fatigue (Crowninshield & Brand 1981). The quadratic cost is equivalent to minimising the sum of squared muscle stresses when cross-sectional areas are equal — a standard first-order approximation.

**References:**
- Seireg A. & Arvikar R. (1975). The prediction of muscular load sharing and joint forces in the lower extremities during walking. *Journal of Biomechanics*, 8(2), 89–102.
- Crowninshield R.D. & Brand R.A. (1981). A physiologically based criterion of muscle force prediction in locomotion. *Journal of Biomechanics*, 14(11), 793–801.
- An K.N. et al. (1984). Determination of muscle and joint forces: a new technique to solve the indeterminate problem. *Journal of Biomechanical Engineering*, 106(4), 364–367.

---

**E. Equilibrium posture finder**

For a given `d_hold`, the body adopts the posture that minimises total muscular effort (natural biomechanical optimisation). This is not a trivial point: the DIP and PIP angles that minimise the total tendon force depend strongly on hold depth, because deeper holds shift the external moment pattern across joints.

![Equilibrium posture search: minimum-effort posture at each hold depth](/Users/igorcerovsky/.gemini/antigravity/brain/42bbfbd3-85d8-4484-af6a-64f8b8d0cd02/equilibrium_posture_diagram_1773821446287.png)

_Figure E: At each hold depth, the posture finder sweeps the DIP/PIP search space and selects the configuration with minimum total tendon force. The objective is a smooth bowl-shaped function of joint angles; the minimum is found via grid search + local refinement._

**Algorithm:**
```python
find_equilibrium_posture(grip_base, geom, F_ext, contact) -> GripAngles:
  1. Grid search: PIP in [θ_PIP_base ± 20°], DIP in [θ_DIP_base ± 20°]  (9×9 grid)
  2. For each (PIP, DIP): call solve_min_effort → get F_total
  3. Best grid point as warm-start
  4. Local scipy.optimize.minimize over (PIP, DIP): objective = F_total
  5. Return GripAngles at optimum
```

> [!CAUTION]
> The posture search landscape may have local minima for extreme angles (crimp with hyperextended DIP). Grid + local refinement mitigates this. In biological reality, neural coactivation patterns constrain the feasible posture space further—this simplified model assumes passive mechanical optimisation only.

**References:**
- Vigouroux L. et al. (2011). Fingertip force and muscle force predictions during a pinching task using an EMG-constrained model. *Journal of Biomechanics*, 44(8), 1443–1449.
- Schweizer A. (2001). Biomechanical properties of the crimp grip position in rock climbers. *Journal of Biomechanics*, 34(2), 217–223.
- Byrd R.H. et al. (1995). A limited memory algorithm for bound constrained optimization. *SIAM Journal on Scientific Computing*, 16(5), 1190–1208. *(L-BFGS-B reference)*

---

**F. Extended Fig 7 + new Fig 8**

- **Fig 7**: Extend `d_range` to 0–45 mm; shade "Deep/Jug zone" (> 22 mm)
- **Fig 8** (new): `plot_deep_hold_sweep()`
  - 3 panels: FDP/FDS forces | FDP/FDS ratio | A2 pulley force
  - X-axis: `d_hold` 2→45 mm; equilibrium posture at each step
  - Short/Std/Long phenotypes overlaid → phenotype advantage visualised
  - Annotate `L_DP` boundary (~22 mm) and crossover point

---

## Verification Plan

### Automated (terminal)
```bash
cd /Users/igorcerovsky/Documents/finger && source .venv/bin/activate && python climbing_finger_3d.py
```

Expected:
1. No `NameError` or `p_C` errors
2. Crimp at d=10 mm: FDP > FDS all methods
3. Open hand at d=35 mm: FDS ≥ FDP (`min_effort` method)
4. `min_effort` total force ≤ `direct` total force for all grips
5. A2 < 400 N at 0.25 BW (70 kg)

### Visual (saved to `outputs/`)
1. Fig 7: deep zone shaded; curves extend and cross beyond 22 mm
2. Fig 8: FDP/FDS ratio crosses 1.0 near d ≈ 22 mm; short finger crossover at smaller d
3. Phenotype advantage visible: short DP → earlier FDS dominance on deep holds
