# Code Review: Iteration 6 — Antagonist EDC Co-Contraction

## Objective

Replace the `nnls` (non-negative least squares) solver — which only enforces $F \ge 0$ — with `scipy.optimize.lsq_linear` throughout all three solution methods, allowing an explicit lower-bound floor on the Extensor Digitorum Communis (EDC) force based on joint hyperextension state.

## Changes Verified

### 1. `Config` Class

Added:

- `use_edc_stiffness = True` — toggle for the feature
- `k_EDC_stiff = 1.5` N — exponential gain constant
- `theta_dip_max = 25.0` deg — normalisation ROM for the exponent

These are cleanly scoped inside the existing `Config` pattern, consistent with previous iterations.

### 2. `get_min_EDC_force(dip_deg)`

New module-level helper. Returns 0.0 whenever `dip_deg >= 0` (no hyperextension) or when the config flag is off. For hyperextension:

```python
F_EDC_min = k_EDC_stiff * exp(|dip_deg| / theta_dip_max)
```

Simple, stateless, and easily unit-testable. No side effects.

### 3. Solver Refactor: `nnls` → `lsq_linear`

All three places (posture optimizer, EMG method, LU-minimising method) now use `scipy.optimize.lsq_linear` with `bounds=([0, 0, F_EDC_min], [inf, inf, inf])`. This is a strict upgrade:

- `nnls` enforces $\ge 0$ only (equivalent to `bounds=([0,0,0],[inf,inf,inf])`)
- `lsq_linear` generalises this to arbitrary lower/upper bounds with the same computational cost

The Direct (3×3) method subtracts the EDC stiffness moment from the RHS vector before inversion, which is mathematically equivalent and avoids a 4-variable overdetermined system.

## Simulation Results (70 kg, 10 mm, 45° wall)

| Grip | Method | F_FDP | F_FDS | F_LU | **F_EDC** | Total |
|------|--------|-------|-------|------|-----------|-------|
| **Crimp** | EMG | 421.5 | 240.9 | 324.8 | **3.7** | 990.9 |
| Half-Crimp | EMG | 396.0 | 330.0 | 425.1 | **0.0** | 1151.2 |
| Open Hand | EMG | 373.0 | 423.8 | 325.9 | **0.0** | 1122.7 |

**Key finding**: EDC fires exclusively for the full Crimp, where DIP hyperextension is $-22.6°$. At $e^{22.6/25} \approx 2.46$, the floor is $1.5 \times 2.46 = 3.7$ N — exactly matching the printed output. The other postures (DIP > 0°) correctly remain at zero.

## Risks / Open Questions

- The $k_{stiff} = 1.5$ N coefficient is empirically calibrated rather than directly measured. Vigouroux (2011) reports co-contraction indices of 5–15% of maximal EDC — at the loads modelled here ($\approx 170$ N reaction), this is physiologically plausible but a sensitivity analysis against published EMG data would strengthen confidence.
- The `Direct` method now produces $F_{EDC} = F_{EDC,min}$ exactly (no freedom to increase). This is intentional — it shows the minimum cost; the EMG and LU-min methods may solve for higher EDC if warranted by the moment balance.

**Status: APPROVED — improves physiological fidelity, no regressions observed.**
