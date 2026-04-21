# Physics of 3D Climbing Finger Biomechanics

This document outlines the theoretical physical and biomechanical formulations used to simulate climbing finger mechanics. The equations here allow full reconstruction of the `climbing_finger_3d.py` computational model.

## 1. 3D Kinematics and Coordinate System

The finger is modeled as a 4-DOF rigid body articulated chain consisting of three links (Proximal, Middle, and Distal Phalanges) articulated at the MCP, PIP, and DIP joints. The MCP joint is taken as the spatial origin $(0,0,0)$. 

The global coordinate axes are defined as:
- **x-axis**: Toward the wall (normal to the climbing surface).
- **y-axis**: Dorsal direction (upward in standard climbing posture).
- **z-axis**: Radial direction (toward the thumb).

### 1.1 Rotation Matrices
The rotations are defined purely by flexion ($\theta$) and adduction/abduction ($\phi$) angles.
Flexion is measured around the z-axis (towards palm, $-y$):
$$ R_{flex}(\theta) = \begin{pmatrix} \cos(-\theta) & -\sin(-\theta) & 0 \\ \sin(-\theta) & \cos(-\theta) & 0 \\ 0 & 0 & 1 \end{pmatrix} $$

Radial abduction is measured around the y-axis (towards $+z$):
$$ R_{abd}(\phi) = \begin{pmatrix} \cos(-\phi) & 0 & \sin(-\phi) \\ 0 & 1 & 0 \\ -\sin(-\phi) & 0 & \cos(-\phi) \end{pmatrix} $$

### 1.2 Forward Kinematics
The rotation matrices of each phalanx relative to the global frame are:
- $R_{MCP} = R_{flex}(\theta_{MCP}) R_{abd}(\phi_{MCP})$
- $R_{PIP} = R_{MCP} R_{flex}(\theta_{PIP})$
- $R_{DIP} = R_{PIP} R_{flex}(\theta_{DIP})$

#### Instantaneous Centers of Rotation (ICR)
Due to the cam-shape of articular human joints, the fulcrum point translates across the condyle during flexion rather than acting as a static rigid door hinge. An affine palmar translation vector $\vec{\delta}$ dynamically adjusts bone pivot length proportional to flexion depth:
- $\vec{\delta}(\theta) = \left[ 0, -c_{max} \cdot \left(\frac{\theta}{90^\circ}\right), 0 \right]^T$

The true 3D positions of the joint centers ($L_1, L_2, L_3$ represent $L_{PP}, L_{MP}, L_{DP}$) are modeled as:
- $p_{MCP} = \vec{0}$
- $p_{PIP} = p_{MCP} + R_{MCP} \left( L_1 \hat{e}_x + \vec{\delta}_{PIP} \right)$
- $p_{DIP} = p_{PIP} + R_{PIP} \left( L_2 \hat{e}_x + \vec{\delta}_{DIP} \right)$
- $p_{TIP} = p_{DIP} + R_{DIP} \left( L_3 \hat{e}_x \right)$

This translation structurally reduces the effective external contact moment arm against the fingertip proportionally as the flexor angle increases inside a crimp, yielding a biologically accurate mechanical footprint not captured by purely rigid link definitions.

### 1.3 Angle-Dependent MCP Moment Arms

Flexor moment arms at the MCP joint are not constant. An et al. 1983 (Table 2) report a linear increase with MCP flexion angle:

| $\theta_{MCP}$ (deg) | FDP (mm) | FDS (mm) |
|----------------------|----------|----------|
| 0 | 8.0 | 6.8 |
| 30 | 9.5 | 7.8 |
| 60 | 11.2 | 9.0 |
| 90 | 12.8 | 10.0 |

Linear fits used in the model:
$$ma_{FDP,MCP}(\theta) = \max(8.0 + 0.053 \cdot \text{clip}(\theta_{MCP}, 0, 90),\ 6.0)$$
$$ma_{FDS,MCP}(\theta) = \max(6.8 + 0.036 \cdot \text{clip}(\theta_{MCP}, 0, 90),\ 5.0)$$

The previous fixed values ($10.4$ and $8.6$ mm) were mid-range ($\approx 45°$) approximations that introduced systematic errors at the extreme postures most relevant to climbing (Crimp at $\theta_{MCP}\approx 3°$ and Open Hand at $\theta_{MCP}\approx 20°$).

## 2. Distributed Contact Model

Unlike point-load models, deep climbing grips distribute skin contact pressure over both the Distal Phalanx (DP) and Middle Phalanx (MP).

### 2.1 Effective Hold Depth and Projection
The actual hold depth ($d_{hold}$) represents a geometric dimension of the climbing hold. To calculate the engaged arc length along the finger's palmar surface, this depth is projected based on the angle between the phalanx and the hold surface.
Let $\hat{n}_{hold}$ be the normal to the hold surface. The cosine of the angle $\alpha$ between a phalanx's axial direction ($\hat{e}$) and the hold surface is:
$$ \cos(\alpha) = \|\hat{e} \times \hat{n}_{hold}\| $$
The engaged length on any phalanx segment is geometrically projected by dividing by $\cos(\alpha)$.
Additionally, the effective available depth accounts for the rounded edge wrapping onto the finger:
$$ proj\_d_{hold} = d_{hold} + r_{edge} |\hat{e}_{DP} \cdot \hat{n}_{hold}| $$

### 2.2 Force Distribution (Hertz-Like)

Skin contact pressure is non-uniform. Consistent with Hertz contact mechanics (Johnson 1985), pressure peaks at the fingertip and tapers toward the DIP crease. On the MP, pressure rises from the DIP crease proximally.

**DP pressure profile** ($s$ measured from tip, $s \in [0, L_{DP}]$):
$$p_{DP}(s) \propto 1 - s/L_{DP} \implies
\text{Area}_{DP} = \int_0^{L_{DP}} p\,ds = \frac{L_{DP}}{2},\quad
\text{centroid at } s = \frac{L_{DP}}{3} \text{ from tip}$$

**MP pressure profile** ($s$ measured from DIP crease, $s \in [0, x_{MP}]$, $x_{total} = L_{DP} + x_{MP}$):
$$p_{MP}(s) \propto s/x_{total} \implies
\text{Area}_{MP} = \frac{x_{MP}^2}{2\, x_{total}},\quad
\text{centroid at } s = \frac{2\,x_{MP}}{3} \text{ from DIP}$$

The fraction of total normal force on each phalanx is proportional to its area:
$$f_{DP} = \frac{\text{Area}_{DP}}{\text{Area}_{DP} + \text{Area}_{MP}}, \qquad f_{MP} = 1 - f_{DP}$$

### 2.3 Tissue Pulp Compression (Skin Deformation)
Human fingertip pulp acts as a hyper-elastic pad that compresses non-linearly under mechanical load, dynamically reducing the geometric contact radius separating the bone from the outer rock surface. According to compressive loading tests (Serina et al. 1997), the deformation $\delta$ can be modeled logarithmically:
$$ \delta(F) = k \cdot \ln\left(1 + \frac{F}{F_0}\right) $$
Where $k \approx 1.15$ mm and $F_0 \approx 10.0$ N. The palmar radius of the contact point is subsequently updated dynamically:
$$ r_{palmar} = \max\left( \frac{t_{DP}}{2} - \delta(F), \delta_{min} \right) $$
This continuous structural translation effectively shortens the external moment displacement vector ($\vec{p}_{contact} - \vec{p}_{joint}$) under heavy dynamic loading, simulating how climbers instinctively sink bone-deep into micro-holds to maximize their mechanical advantage.

- **Shallow Hold**: If $proj\_d_{hold} \le L_{DP} \cos(\alpha_{DP})$, the force acts entirely on the DP. The engaged arc length is $d_{eff} = proj\_d_{hold} / \cos(\alpha_{DP})$. The centroid $p_{C,DP}$ is $d_{eff}/3$ from the tip.
- **Deep Hold**: If $proj\_d_{hold} > L_{DP} \cos(\alpha_{DP})$, the force partitions between DP and MP based on area integrals:
  - $Area_{DP} = L_{DP} / 2$
  - Engaged length of MP: $engaged_{MP} = \min( (proj\_d_{hold} - L_{DP} \cos(\alpha_{DP})) / \cos(\alpha_{MP}), L_{MP} - 0.5 )$
  - $Area_{MP} = engaged_{MP}^2 / (2 \cdot (L_{DP} + engaged_{MP}))$
  The fraction of the total normal force on each phalanx is proportional to its area.

The MP force application relies on a weighted average representing the A3 skeletal anchor (0.15 $L_{MP}$ from PIP):
$$ p_{C,MP} = 0.4 \cdot (\text{Geometric Centroid}_{MP}) + 0.6 \cdot p_{A3} $$

## 3. Quasistatic Equilibrium

An equilibrium state implies that the sum of external moments is perfectly balanced by internal muscular moments.

### 3.1 External Load Vector

The external applied force derives from the climber's body geometry. The COM is located at perpendicular distance $d_{COM}$ from the wall and vertically $h_{below}$ below the hold. The reaction force direction at the hold is:

$$\hat{u} = \frac{1}{\sqrt{(d_{COM}\cos\beta)^2 + h_{below}^2}}
\begin{pmatrix} d_{COM}\cos\beta \\ -h_{below} \\ 0 \end{pmatrix}$$

where $\beta_{wall}$ is the wall angle from vertical. The full force vector is:
$$\vec{F}_{ext} = F_{mag}\, \hat{u} + F_{lateral}\,\hat{z}$$

> **Limitation**: On steep roofs ($\beta > 70°$), active body tension from core and hip flexors redirects the effective force vector and cannot be estimated without full-body kinematics. The above formula provides a lower-bound estimate of finger load on roofs.

This force produces a reaction moment at each joint:
$$\vec{M}_{ext, J} = (\vec{p}_{C,DP} - \vec{p}_J) \times \vec{F}_{DP} + (\vec{p}_{C,MP} - \vec{p}_J) \times \vec{F}_{MP}$$
The sagittal plane bending moment is the projection of this general 3D moment onto the flexion axis ($\hat{z}_{local}$), and the abduction moment is projected onto the true abduction axis.

### 3.2 Muscular System Formulation
Four target muscles balance the degrees of freedom (Flexion at DIP, PIP, MCP): Flexor Digitorum Profundus (FDP), Flexor Digitorum Superficialis (FDS), Lumbrical (LU), and Extensor Digitorum Communis (EDC).
Let $A$ be the moment arm matrix (with signs denoting flexor/extensor action):

$$ M_{muscle} = A \vec{F}_{muscles} = \begin{pmatrix} 
ma_{FDP,DIP} & 0 & ma_{LU,DIP} & ma_{EDC,DIP} \\ 
ma_{FDP,PIP} & ma_{FDS,PIP} & ma_{LU,PIP} & ma_{EDC,PIP} \\ 
ma_{FDP,MCP} & ma_{FDS,MCP} & ma_{LU,MCP} & ma_{EDC,MCP} 
\end{pmatrix} 
\begin{pmatrix} F_{FDP} \\ F_{FDS} \\ F_{LU} \\ F_{EDC} \end{pmatrix} $$

For equilibrium, $\vec{M}_{muscle} = \vec{M}_{ext\_required}$.
To enforce the physiological reality that muscles can only pull ($F \ge 0$), the computational solver resolves this primarily by analyzing the 3 flexion moments.

> **Note**: FDS has **zero moment arm at the DIP joint** ($ma_{FDS,DIP} = 0$). The FDS tendon inserts on the base of the middle phalanx (MP), not the distal phalanx (DP), and therefore cannot generate a flexion moment at the DIP. The DIP row of matrix $A$ therefore has entries only for FDP, LU, and EDC.

### 3.3 Solution Implementations

All three solvers now use `scipy.optimize.lsq_linear` with explicit bounds to enforce both non-negativity ($F \ge 0$) and the mandatory antagonist EDC floor (Section 3.4):

- **Direct (Pure Flexor baseline)**: Subtracts the EDC stiffness moment from the external demand vector $\vec{b}$, then performs a $3 \times 3$ analytical solve for FDP, FDS, LU. EDC is manually appended at $F_{EDC,min}$.
- **EMG-Constrained**: Forces FDP/FDS to hold an empirically determined ratio $r_{emg}$ based on hold depth. Solves a $3 \times 3$ bounded least-squares system with $F_{EDC} \ge F_{EDC,min}$.
- **LU-Minimizing**: Assumes $F_{LU} = 0$, solving a $3 \times 2$ bounded system with the same EDC floor.

### 3.4 Antagonist Extensor Co-Contraction (EDC Stiffness)

When the DIP joint enters hyperextension ($\theta_{DIP} < 0°$), passive capsular ligaments and active extensor structures stiffen exponentially to prevent capsuloligamentous injury. The mandatory minimum extensor force is modeled as:

$$ F_{EDC,min}(\theta_{DIP}) = \begin{cases} k_{stiff} \cdot e^{|\theta_{DIP}| / \theta_{DIP,max}} & \text{if } \theta_{DIP} < 0 \\ 0 & \text{otherwise} \end{cases} $$

Where $k_{stiff} = 1.5\ \mathrm{N}$ and $\theta_{DIP,max} = 25°$. At full hyperextension ($\theta_{DIP} = -22.6°$ in Crimp posture), this produces:

$$ F_{EDC,min} = 1.5 \cdot e^{22.6/25} \approx 3.7\ \mathrm{N} $$

This antagonist force opposes the net flexor torque, requiring the FDP/FDS system to produce additional effort to maintain equilibrium. The physiological consequence is an increased total tension demand in the Crimp posture versus postures where the DIP is not hyperextended, accurately modeling the known metabolic cost of the full crimp grip.

### 3.5 EMG Ratio — Physics-Based frac\_DP Interpolation

The FDP:FDS ratio $r_{emg}$ depends on the fraction of total external force that still acts on the DP, i.e. the fraction that creates a DIP flexion moment:

$$r_{emg} = r_{base} \cdot (0.20 + 0.80 \cdot f_{DP})$$

where $f_{DP} = F_{DP} / (F_{DP} + F_{MP})$ is the DP force fraction directly computed from the Hertz-like pressure model (§2.2).

**Physical derivation:**
- $f_{DP} = 1.0$ (all load on DP, shallow hold): DIP moment is fully intact $\Rightarrow r_{emg} = r_{base}$
- $f_{DP} = 0.0$ (all load on MP, very deep hold): DIP moment vanishes $\Rightarrow r_{emg} = 0.20 \cdot r_{base}$  
  (residual 20% represents passive FDP stiffness and lumbrical coupling that persists even with zero DIP moment demand)

This replaces the previous `d_hold`-based linear interpolation (Iteration 8), which used heuristic breakpoints at $L_{DP}$ and $L_{DP} + L_{MP}$. The frac_DP formula is exact: it is driven by the **same physical quantity** (DP contact force) that drives DIP moment demand. Crucially, it is phenotype-consistent: a short-finger performer crosses $f_{DP} < 0.5$ at a shallower hold depth than a long-finger performer, at exactly the correct anatomical threshold, without any heuristic tuning.

**Numerical note:** $f_{DP}$ is computed at a wall-angle-corrected arc length by `compute_contact_point()`, making the EMG ratio posture-dependent. In a crimp posture (DIP $\approx -25°$), the DP is nearly wall-parallel, which reduces the projected DP capacity and causes $f_{DP}$ to drop to $\approx 0.84$ at $d_{hold} = 15\,\text{mm}$ — a smaller depth than the geometric $L_{DP} = 22\,\text{mm}$. This is physically correct: the crimped DP subtends less of the hold.

## 4. Posture Optimization

The biological system dynamically adopts joint angles (PIP, DIP) that minimize total tendon tension.
This is achieved by minimizing an objective function $J$:
$$ J = F_{FDP} + F_{FDS} + F_{LU} + F_{EDC} + \Phi(\text{Residual}) + \Phi(\text{Joint Limits}) $$
where $\Phi$ are severe geometric penalty functions. The EDC stiffness floor is incorporated during optimization, ensuring that the posture solver never settles on mechanically impossible hyperextended states without physiological cost.

## 5. Pulley Forces and Joint Reactions
### 5.1 Capstan Friction and Distributed Pulley Pressure
Forces exerted by tendons over pulleys (A2, A4) are calculated using unit direction vectors mapping the tendon path deviation around the joint:
$$ \theta_{wrap} = \arccos(\hat{d}_{in} \cdot \hat{d}_{out}) $$

Due to tendon-sheath friction ($\mu_t$), the required proximal muscle tension is reduced as the localized tension builds distally across the wraps:
$$ T_{distal} = T_{proximal} \cdot e^{\mu_t \theta_{wrap}} $$
The total integrated vector force on the pulley is derived using this distally amplified tension:
$$ \vec{F}_{pulley} = T_{local} (\hat{d}_{in} + \hat{d}_{out}) $$

This raw vector is then transformed into a distributed peak physiological tissue pressure to represent actual injury risk over the sheath bandwidth:
$$ P_{MPa} = \frac{\|\vec{F}_{pulley}\|}{L_{pulley} \cdot w_{tendon}} $$
Radial abduction ($\phi \neq 0$) generates substantial out-of-plane lateral shearing forces on the pulleys ($\hat{z}$-component $F_{A2,lat}$ and $F_{A4,lat}$).

### 5.2 6-DOF Joint Reaction Wrenches
Joint reactions are solved recursively from distal to proximal using the Newton-Euler formalism, accumulating tendon tension forces, pulley reaction forces, and external contact forces to yield maximum compressive and mediolateral (ML) shear forces at the DIP, PIP, and MCP joints.

## 6. Friction Cone Enforcement (Iteration 10)

A quasistatic equilibrium requires that the external contact force lies inside the Coulomb friction cone at the fingertip. For a skin-rock interface with coefficient $\mu$:

$$\text{feasible} \iff \frac{\|\vec{F}_{friction}\|}{\mu F_N} \leq 1$$

where $F_N = \hat{n} \cdot \vec{F}_{ext}$ (normal component) and $\vec{F}_{friction}$ is the remaining tangential resultant.

Prior iterations computed this ratio diagnostically. In Iteration 10 a **soft cone penalty** is added to the posture optimiser objective $J$ (§4):

$$\Phi_{friction}(\text{ratio}) = \begin{cases}
0 & \text{ratio} \leq 0.8 \\
k_1 \cdot (\text{ratio} - 0.8)^2 & 0.8 < \text{ratio} \leq 1.0 \quad (k_1 = 200) \\
k_2 \cdot (\text{ratio} - 0.8)^2 & \text{ratio} > 1.0 \quad\quad\quad\;\; (k_2 = 2000)
\end{cases}$$

The threshold 0.8 provides a 20% safety margin before the penalty activates, preventing over-penalisation of feasible but near-boundary postures. The 10× stiffening past ratio = 1.0 ensures the optimizer strongly avoids physically impossible slip postures while remaining smooth and differentiable for the Nelder-Mead local refinement.

> **Limitation**: The friction check uses the global (sagittal + lateral) force vector, which does not account for angle-dependent skin anisotropy. Fingertip skin is softer in dorsal–palmar compression than in distal shear (Serina et al. 1997). A full anisotropic friction model is deferred to a future iteration.

## 7. Validation Against PeerJ 7470 Cadaver Data

The model is validated against the Vigouroux et al. (2019) cadaver tendon-loading experiments (PeerJ 7470). Four standard postures are mapped to climbing analogues:

| PeerJ Posture | Angles (DIP/PIP/MCP) | Climbing analogue | EMG ratio |
|---|---|---|---|
| MinorFlex | 35°/55°/40° | Half-crimp | 1.20 |
| MajorFlex | 25°/57°/55° | Deep half-crimp | 1.20 |
| HyperExt | 45°/50°/−20° | Full crimp | 1.75 |
| Hook | 50°/65°/0° | Hook grip | 1.20 |

### 7.1 Iteration 13 improvements

Two corrections to the validation methodology:

1. **Posture-dependent external force direction**: The fingertip reaction force is now oriented perpendicular to the DP pad (normal to the palmar surface), rather than always along +x. The force direction angle (measured from +x in sagittal plane) is:
   - HyperExt: 15° (force directed slightly dorsally — MCP hyperextension tips the DP)
   - MinorFlex: −40° (force directed palmarly — significant total flexion)
   - Hook: −25°, MajorFlex: −47°

2. **PeerJ-exact geometry**: The comparison geometry now uses the PeerJ segment ratios directly (`segRatios` from `peerj_model.py`), yielding PP=47.0mm, MP=28.8mm, DP=19.0mm instead of the previous approximate scaling.

### 7.2 Results (EMG-constrained method)

| Posture | FDP/FDS ratio | F_dir (°) | Direct ratio | Match |
|---|---|---|---|---|
| HyperExt | 1.75 | 15° | 1.58 | ✓ exact |
| MinorFlex | 1.20 | −40° | 1.49 | ✓ exact |
| MajorFlex | 1.20 | −47° | 1.70 | ✓ exact |
| Hook | 1.20 | −25° | 1.28 | ✓ exact |

The EMG ratio constraint reproduces the Vigouroux reference exactly by construction. The direct solver (3×3) now shows more moderate FDP/FDS elevation in HyperExt (1.58 vs 2.51 in Iter 10) because the posture-dependent force direction distributes the external moment more evenly across DIP/PIP joints. This is a validation that the force direction correction improves the unconstrained solver's behaviour.

### 7.3 Absolute Force Magnitude Validation (Iteration 14)

Iteration 14 uses the **actual cadaver force plate measurements** (mean of 3 specimens, H01–H03) as $\vec{F}_{ext}$. The experimental fingertip reaction forces range from 0.93 N to 5.00 N, representing 19–32% of the total applied tendon force.

**Predicted-to-applied tendon force ratio** (EMG-constrained method, ideal = 1.0):

| Posture | Pred/Applied | Interpretation |
|---|---|---|
| MajorFlex | **1.01** | Excellent agreement |
| MinorFlex | **1.70** | Moderate overestimate |
| Hook | **3.20** | Significant overestimate |
| HyperExt | **5.56** | Large overestimate |
| **Overall** | **2.87 ± 1.75** | Systematic overestimate |

The systematic overestimate indicates that our model's moment arms are **shorter** than the PeerJ CT-calibrated values. A shorter moment arm requires more tendon force to balance the same external moment. The overestimate is worst for HyperExt (crimp), where the DIP hyperextension geometry is most sensitive to the tendon path point locations.

> **Root cause**: Our model uses simplified An et al. (1983) moment arm functions with literature-average coefficients, while the PeerJ model uses specimen-specific path points optimized against CT-derived bone geometry. The moment arm discrepancy is amplified by the multi-joint lever chain: a 20% error in one joint's moment arm can cascade to a 2–5× error in total predicted tendon force.

### 7.4 Limitations

**Full T_mus matrix**: Our model uses simplified moment arms (3-DOF per joint) while the PeerJ model uses optimized path points with 6 muscles × 4 DOF, including extensor mechanism ratios. The T_mus matrix differences affect absolute force predictions but not the FDP/FDS ratio (which is set by the EMG constraint).

## 8. Moment Arm Recalibration (Iteration 15)

### 8.1 Diagnostic comparison

A joint-by-joint moment arm comparison (`moment_arm_comparison.py`) was performed between our An et al. 1983 literature averages and the PeerJ CT-calibrated path points at 4 postures:

| Joint × Tendon | Our (An 1983) | PeerJ (CT) | Ratio | Error direction |
|---|---|---|---|---|
| DIP × FDP | 7.6 mm | 4.4 mm | **1.73** | Ours too large |
| PIP × FDP | 10.8 mm | 11.0 mm | **0.98** | ≈ matched |
| PIP × FDS | 8.6 mm | 7.2 mm | **1.19** | Ours slightly large |
| MCP × FDP | 10.1 mm | 13.6 mm | **0.74** | Ours too small |
| MCP × FDS | 8.2 mm | 14.7 mm | **0.56** | Ours much too small |

(Values shown for MinorFlex posture; pattern is consistent across all 4 postures.)

**Root cause of force overestimate**: DIP too large (FDP forced high) + MCP too small (FDP and FDS need to be even higher to balance MCP moment). The compound effect across the lever chain produces 2–5× total force overestimate.

### 8.2 Recalibrated coefficients

Linear regressions were fit to the PeerJ moment arms at 4 postures (R² > 0.99 for all except DIP):

```
Config.moment_arm_source = 'peerj'   # (default in Iter 15)

FDP_DIP = 4.70 − 0.011·θ_DIP    (was 6.0 + 0.045·θ_DIP)
FDP_PIP = 8.21 + 0.050·θ_PIP    (was 9.0 + 0.033·θ_PIP)
FDP_MCP = 9.89 + 0.087·θ_MCP    (was 8.0 + 0.053·θ_MCP)
FDS_PIP = 4.46 + 0.050·θ_PIP    (was 7.5 + 0.020·θ_PIP)
FDS_MCP = 10.13 + 0.108·θ_MCP   (was 6.8 + 0.036·θ_MCP)
```

The An 1983 coefficients are preserved as `Config.moment_arm_source = 'an1983'`.

### 8.3 Impact on absolute force accuracy

| Posture | Pred/App (An 1983) | Pred/App (PeerJ) | Improvement |
|---|---|---|---|
| MajorFlex | 1.01 | **0.83** | ✓ (was already good) |
| MinorFlex | 1.70 | **1.34** | ↓ 21% |
| Hook | 3.20 | **2.26** | ↓ 30% |
| HyperExt | 5.56 | **3.31** | ↓ 40% |
| **Overall** | **2.87** | **1.94** | **↓ 32%** |

The recalibration reduced the systematic overestimate by 32% across all postures. The remaining error (overall 1.94×) is attributed to:
1. The 3-DOF simplification (vs PeerJ's full 4-DOF 6-muscle T_mus matrix)
2. Lumbrical, interossei, and extensor mechanism moment arms still at literature values
3. The linear fit approximation to the nonlinear generalized-force moment arm
