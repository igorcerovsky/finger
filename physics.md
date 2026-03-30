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
Pressure distributes based on a triangular profile tapering towards the DIP crease, and a rising ramp profile on the MP.

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
The external applied force $F_{ext}$ depends on the wall angle $\beta_{wall}$ and lateral load $F_{lateral}$:
$$ \vec{F}_{ext} = F_{mag} \cos(\beta_{wall})\hat{x} - F_{mag} \sin(\beta_{wall})\hat{y} + F_{lateral}\hat{z} $$

This external force produces a reaction moment at each joint:
$$ \vec{M}_{ext, J} = (\vec{p}_{C,DP} - \vec{p}_J) \times \vec{F}_{DP} + (\vec{p}_{C,MP} - \vec{p}_J) \times \vec{F}_{MP} $$
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

For equilibrium, $ \vec{M}_{muscle} = \vec{M}_{ext\_required} $. 
To enforce the physiological reality that muscles can only pull ($F \ge 0$), the computational solver resolves this primarily by analyzing the 3 flexion moments.

### 3.3 Solution Implementations
- **Direct (Pure Flexor baseline)**: $3 \times 3$ analytical inversion using only FDP, FDS, LU. Yields theoretical bounds but fails physiologically on extreme overhangs.
- **EMG-Constrained**: Forces FDP/FDS to hold an empirically determined ratio $r_{emg}$ based on hold depth. Solves a $3 \times 3$ bounding system involving FDS, LU, and EDC using Non-Negative Least Squares (NNLS) to robustly enforce proper muscle boundary recruitment without matrix collapse.
- **LU-Minimizing**: Assumes $F_{LU} = 0$, solving a $3 \times 2$ NNLS limit boundary matching FDS to EDC.

## 4. Posture Optimization
The biological system dynamically adopts joint angles (PIP, DIP) that minimize total tendon tension. 
This is achieved by minimizing an objective function $J$:
$$ J = F_{FDP} + F_{FDS} + F_{LU} + F_{EDC} + \Phi(\text{Residual}) + \Phi(\text{Joint Limits}) $$
where $\Phi$ are severe geometric penalty functions. The inclusion of EDC permits natural physiological stabilization against un-holdable flexion-collapsing moments (such as on steep open-hand overhangs), removing artificially forced mathematical artifacts.

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
