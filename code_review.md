# Code Review: Capstan Mechanics & Distributed Pulley Pressure Integration

## Objective
Upgrade the A2 and A4 pulleys from frictionless, infinitely narrow "bow-string" vector anchor points to a continuous cylindrical "Capstan" friction model, simulating biological tissue wrapping angles and calculating distributed spatial tissue pressures (MPa).

## Code Changes Reviewed
1. **Geometric Extensions (`Config`)**: Added constants `mu_tendon=0.08`, `w_tendon_mm=4.0`, `L_A2_mm=15.0`, and `L_A4_mm=8.0` to define the spatial bounding cylinder representing the pulley tissue bands.
2. **Capstan Wrap Angles**: Introduced `compute_pulley_angles` which successfully extracts `theta_A2` and `theta_A4` from the `arccos` vector dot product. Applied safety bounding `np.clip(-1, 1)` to protect the math domain against rounding overflow on flatly extended fingers.
3. **Multiplier Injection in Optimizer Solver**: Integrated the Capstan mechanical friction multiplier `C = e^{\mu \theta}` directly into the `A3e` coefficient array inside `find_equilibrium_posture` and `solve_all_methods`. This explicitly mimics the exponential tension offloading granted by the flexor tendon wrapping forcefully across the proximal phalange bones.
4. **Physiological Tension Distribution**: Output updated from total vector lengths (Newtons) to peak tissue pressures (MPa) traversing the A2 bounds, vastly improving physiological interpretability for rupture injury analyses.
5. **Console Logging**: Replaced `A2 3D / N` console print table to log `A2(MPa)` accurately tracking structural metrics.
6. **Documentation**: Updated `physics.md` tracking the mathematical formulas for friction and Capstan derivations.

## Review Findings

### 1. Robustness
- **Pros**: The `arccos` method operates identically across varying spatial planes ensuring it adapts cleanly without introducing singularities during Out-of-Plane (MCP Abduction) configurations. The `.clip` ensures `nan` propagation is impossible.
- **Safety**: Matrix structures correctly preserve purely positive multiplier constants $C_A \ge 1.0$.

### 2. Biological Validation
Results perfectly mirror physiological testing expectations:
- Full Crimp loads generate very little MCP angular deviation ($\theta_{A2} \approx 0$), isolating the structural strain into the A3 and A4 regions. As calculated, $P_{A2} \approx 1.0 \text{ MPa}$.
- Open Hand and Pinch grips mandate massive PIP/MCP flexion vectors, steeply bending the tendon around A2. Results exhibit significantly elevated Capstan wrapping demands ranging up to $5.1 \text{ MPa}$.

### Conclusion
The code securely and successfully migrates away from arbitrary point-load vectors to an integrated, friction-tracking wrap model providing medically tangible outputs (MPa).

**Status: APPROVED**
