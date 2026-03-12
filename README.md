# 3D Climbing Finger Biomechanics Model

![Finger Kinematics 3D Model](outputs/climbing_3d_fig1.png)

## 1. Abstract
The **3D Climbing Finger Biomechanics Model** is a full three-dimensional spatial analysis of human finger biomechanics during climbing. This repository provides a static equilibrium framework to evaluate deep flexor (FDP), superficial flexor (FDS), and lumbrical (LU) muscle forces across various climbing grips (crimp, half-crimp, open hand, pinch). Extending beyond traditional 2D planar models, this solver accounts for lateral wall friction, MCP radial abduction, out-of-plane pulley forces, and 6-DOF joint reactions to accurately assess off-axis injury risks.

## 2. Introduction & Theoretical Background
The biomechanical demands of rock climbing frequently exceed standard physiological limits, heavily loading the flexor tendon pulleys and collateral ligaments. Prior literature (e.g., Vigouroux et al. 2006, Schweizer 2001) primarily modelled these interactions in 2D. However, climbing holds inherently apply 3D forces—especially overhanging "side-pulls" or asymmetric slope grips.

This model incorporates:
*   **4 Degrees of Freedom (DOF):** Flexion at DIP, PIP, MCP, plus radial abduction at the MCP joint.
*   **3 Primary Flexors:** Flexor Digitorum Profundus (FDP), Flexor Digitorum Superficialis (FDS), and Lumbricals (LU).
*   **Empirical Physiological Constraints:** Utilizes *in vivo* intra-muscular EMG force-sharing ratios (Vigouroux 2006) to solve the exact flexor distribution, resolving the mathematical indeterminacy of the human hand without guessing optimization functions.

## 3. Methods

### 3.1 Kinematics
The forward kinematics define a local coordinate frame with the origin placed at the MCP joint:
- `x` -> toward wall (grip force direction)
- `y` -> dorsal (upward relative to standard climbing posture)
- `z` -> radial (toward the thumb)

Phalanx lengths (PP, MP, DP) are derived from published anthropometric data (Özsoy et al.), with default scaling parameters to analyze the universally observed **"long finger disadvantage"** on small holds.

### 3.2 Contact Geometry
Force application is not assumed purely at the anatomical tip. The model calculates the exact contact point ($C$) on the distal phalanx (DP) palmar surface based on:
1.  **Hold Depth ($d_{hold}$)**
2.  **Hold Edge Radius ($r_{edge}$)**
3.  **Wall Overhang Angle ($\beta$)**
4.  **Skin-Rock Friction Utilization Ratio ($\mu$)**

### 3.3 Solution Algorithms
To solve the 4-equation system (3 flexion moments + 1 abduction moment) with 3 muscles, three direct mathematical solvers are compared:
1.  **Direct (3x3 Exact):** Pure linear algebra solving DIP/PIP/MCP flexion directly.
2.  **EMG-Constrained (Method 2 - Primary):** Constrains the FDP/FDS ratio to Vigouroux 2006 biological data. Highly accurate and physiologically validated.
3.  **LU-minimizing (Method 3):** Solves a 2-muscle system assuming Lumbricals act purely as stabilizers.

## 4. Results

### 4.1 Tendon Forces & Recruitment (EMG vs Direct)
The required FDP and FDS forces dynamically scale based on the specific grip posture enforced by hold depth and edge radius.

![Tendon Forces](outputs/climbing_3d_fig2.png)

### 4.2 Kinematic Coupling & PIP Flexion
A continuous sweep of the PIP joint angle reveals how mechanical advantages shift. Crimp grips hyperextend the DIP, transferring extreme load to the A2 and A4 pulleys. 

![FDP & FDS vs PIP Angle](outputs/climbing_3d_fig3.png)

### 4.3 Out-of-Plane Pulley Load (3D Specificity)
Unlike 2D models, this 3D model identifies the severe lateral shearing force on the A2/A4 pulleys introduced by MCP radial abduction during side-pulls. 

![3D Pulley Forces](outputs/climbing_3d_fig4.png)

### 4.4 Mediolateral (ML) Joint Shear
The 6-DOF joint reaction solver outputs the mediolateral shear acting on the MCP and PIP collateral ligaments, which is critical for diagnosing lateral impingement injuries.

![6-DOF Joint Reactions](outputs/climbing_3d_fig5.png)

### 4.5 The Long Finger Disadvantage
Climbers with longer phalanges suffer a distinct biomechanical disadvantage on shallow holds. As hold depth is reduced, the lever arm artificially extends for a long finger, requiring exponentially higher internal tendon forces for the same external body weight.

![Long Finger Summary](outputs/climbing_3d_fig6.png)
![Hold Depth Analysis](outputs/climbing_3d_fig7.png)

## 5. Usage & Quick Start

### Installation
Ensure you have `numpy` and `matplotlib` installed:
```bash
pip install numpy matplotlib
```

### Running the Simulation
To run the full 3D simulation and generate updated output figures locally:
```bash
python3 climbing_finger_3d.py
```
*Note: Generated images will be dumped to `/outputs/` assuming configured paths or current working directory.*

### Configuration
You can edit the `Config` class directly inside `climbing_finger_3d.py` to change:
- `body_weight_kg` and `bw_fraction` (load per finger)
- `beta_wall_deg` (Wall steepness: 0 = vertical, 90 = full horizontal roof)
- `d_hold_mm` (Depth of the crimp edge)
- `F_lateral_N` (Applied side-pull force)

## 6. References
1.  **Vigouroux, Quaine, Labarre-Vila, Moutet.** *Estimation of finger muscle tendon tensions and pulley forces during specific sport-climbing grip techniques.* Journal of Biomechanics, 39:2583-2592 (2006).
2.  **An KN, Ueba Y, Chao EY, Cooney WP, Linscheid RL.** *Tendon excursion and moment arm of index finger muscles.* Journal of Biomechanics, 16(6):419-25 (1983).
3.  **Schweizer.** *Biomechanical properties of the crimp grip position in rock climbers.* Journal of Biomechanics, 34(2):217-23 (2001).
4.  **Brand PW, Hollister A.** *Clinical Mechanics of the Hand.* 3rd Edition, Mosby (1999).
