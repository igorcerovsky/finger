# Comparison Results: 3D Climbing Model vs PeerJ Human Model

## Overview
We integrated the native human biomechanics model provided in the **PeerJ 7470** article and tested it against our custom `climbing_finger_3d.py` framework under a controlled benchmark load:
- **Posture**: Half-crimp
- **Load Vector**: 100N directed primarily upwards toward the PIP/MCP fulcrum
- **Target metric**: FDP (Flexor Digitorum Profundus) and FDS (Flexor Digitorum Superficialis) tendon forces

## Quantitative Results

| Model Variation | FDP Force (N) | FDS Force (N) | Total Flexor Force |
| :--- | :--- | :--- | :--- |
| **Our 3D Model (Biological EMG Ratios)** | 211.7 N | 176.4 N | 388.1 N |
| **Our 3D Model (Direct Linear Algebra)** | 204.8 N | 185.1 N | 389.9 N |
| **PeerJ Model (Original PCSA-Stress Min)** | 137.5 N | 189.0 N | 326.5 N |
| **PeerJ Model (Theoretical PCAS-Free Min)** | 137.7 N | 189.0 N | 326.7 N |

### Modifying Force Application Point (Pad vs Fingertip)
A crucial difference between modeling climbing holds and medical biomechanics is *where* the force applies. In our default climbing model, we simulate a 10mm "crimp edge" where the force applies on the skin pad, creating a short lever arm at the DIP joint. The PeerJ model assumes force strictly at the **distal fingertip (0mm hold depth)**, creating a massive lever arm.

When we force both models to evaluate the exact same 100N **fingertip** load (ignoring pad friction and contact depth):

| Model Variation (Fingertip Load) | FDP Force (N) | FDS Force (N) |
| :--- | :--- | :--- |
| **Our 3D Model (EMG Ratios)** | 279.0 N | 232.5 N |
| **Our 3D Model (Direct linear algebra)** | 413.9 N | 63.3 N |
| **PeerJ Model (PCAS-Free Min)** | 332.3 N | 6.5 N |

## Scientific Analysis & Interpretation

The simulation results differ, primarily in how the mathematical engines deduce the **indeterminacy problem** of the human hand (more degrees of freedom than equations).

### 1. The Muscle Recruitment Paradigm
- **Our Model (EMG-Constrained)** uses explicitly measured biological co-contraction ratios (based on Vigouroux et al. 2006 EMG *in vivo* climbing data). In reality, climbers heavily recruit the deep flexor (FDP) even when mechanically disadvantageous to aggressively lock the DIP joint into hyperextension during a crimp.
- **The PeerJ Model** utilizes purely mathematical Static Optimization (specifically minimizing the sum of squared tendon stresses). Mathematically, the FDS tendon has a superior (larger) moment arm at the PIP and MCP joints. The SQP optimizer "sees" this and attempts to heavily rely on the FDS while offloading the FDP as much as mechanically possible, leading to a much lower FDP force (137.5N vs 211.7N). 

Upon stripping the **PCSA limits** entirely (treating the muscles as "theoretical" unbounded actuators with equal mathematical weighting), the prediction barely shifted (FDP moving from 137.5 to 137.7). This mathematically confirms that the discrepancy is not an artifact of muscle volume constraints, but a fundamental characteristic of the Extensor Hood's highly intertwined lever arms.

### 2. The Extensor Mechanism & Intrinsic Stabilizers
- **Our Model** distills the system down to the prime movers (FDP, FDS, Lumbricals) transferring load across the arc-routed pulleys to focus specifically on prime finger damage variables.
- **The PeerJ Model** employs a highly sophisticated routing algorithm modeling the entire "Extensor Hood" (ES, RB, UB, TE, etc.) including interosseous muscles via physiological cross-sectional area (PCSA) fractions. This implies the external load causes significant co-contraction within the stabilization web not accounted for natively in our simplified 3D prime-mover model. This co-contraction changes the net moment required by the major flexors, partially explaining why the PeerJ model's total absolute flexor force sum is lower (326.5N vs 388.1N).

## Conclusion
The initial discrepancy (137N vs 211N FDP) was almost entirely driven by the **external load application geometry**. 
When applying load to the extreme fingertip, the required DIP joint torque spikes. Since FDP is the *only* muscle crossing the DIP joint, the PeerJ mathematical optimizer is forced to recruit FDP heavily (332.3N). This immense FDP tension also contributes massively to PIP and MCP equilibrium, leaving almost no torque deficit for the FDS to cover (6.5N). 

Our EMG constrained model predicts a much more balanced 279N / 232.5N distribution at the fingertip because the nervous system actively co-contracts FDS to stabilize the PIP, rather than relying strictly on the minimum mathematical minimum threshold.
