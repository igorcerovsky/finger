# Code Review: EDC Extensor Stabilizer Implementation

## Objective
Introduce the Extensor Digitorum Communis (EDC) to provide active physiological torque balance against extensor demands, which previously caused the pure-flexor optimizer matrices to fail via `residual_error`.

## Code Changes Reviewed
1. `climbing_finger_3d.py::moment_arms`: Added negative (extensor) moment arms for DIP, PIP, and MCP respectively (`-4.0`, `-6.0`, `-10.0`).
2. `climbing_finger_3d.py::find_equilibrium_posture`: Replaced unrestricted bounding pseudo-penalties (`lstsq` + `max(0, x)`) with structured bound constraints via `scipy.optimize.nnls`. The `A3e` coefficient array received the EDC vector.
3. `climbing_finger_3d.py::solve_all_methods`: Updated `emg` and `lu_min` systems identically using boundary constrained `nnls`. Retained `direct` solver as 3-parameter analytical historical baseline.
4. `physics.md`: Rewrote Sections 3.2, 3.3, and 4 to mathematically reflect the $3 \times 4$ solver properties enforcing biological constraints via non-negative active muscular loads.

## Review Findings

### 1. Robustness
- **Pros**: The `nnls` algorithm (Lawson and Hanson subset selection) strictly requires responses in the positive orthant. This resolves physiological violations instantaneously inside the mathematical bounds rather than patching objective functions post-calculation.
- **Safety**: Adding `EDC` ensures stability in dynamic edge-case geometries (like flat roofs or shallow slopes).

### 2. Analytical Precision Check
- The `b3e` vector maps external hold forces. Previous behavior clipped $F<0$, thus incorrectly tracking $M_{calc} \ne M_{true}$.
- Adding an independent bounded vector $(-, -, -)$ allows reaching previously inaccessible residual space vectors. The results match expectations: standard hanging ($M \ge 0$) triggers only flexors so $F_{EDC} = 0.0$. Negative moments (extreme overhang) recruit $F_{EDC}$.

### 3. Display Logic
- Added column tracking $F_{EDC}$ correctly aligns with formatting widths in standard terminal readouts.

## Conclusion
The EDC muscle inclusion works robustly utilizing `scipy` bound logic, successfully bypassing legacy artificial optimization fail constraints.

**Status: APPROVED**
