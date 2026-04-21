# Code Review: Iteration 12 — Fig 11 Annotations and Phenotype L_DP

## Summary

This iteration addresses two targeted visual and methodological improvements to Figure 11 (Grip-Optimal Comparison):

1. **Phenotype-Specific L_DP Lines**: The L_DP vertical indicator is now dynamically pulled from each row's specific finger geometry (`geom.L3`) instead of incorrectly using `geom_std.L3` for all phenotypes.
2. **Grip-Transition Annotations**: Explicit transition boundaries between optimal grips are now visually annotated with dashed vertical lines and text boxes in the right panel (e.g., "C→H 32mm").

---

## 1. Phenotype-Specific L_DP Lines

### The Problem
Previously, the `geom_std.L3` (22 mm) vertical line was drawn across all three rows (Short, Standard, Long) in Fig 11. This was misleading because the Short phenotype has an L_DP of 18.7 mm, and the Long phenotype has an L_DP of 25.3 mm.

### The Fix
Inside the row loop, the code now extracts the current phenotype's distal phalanx length:
```python
L_DP_this = geom.L3
# ...
ax.axvline(L_DP_this, color='#6A1B9A', ls='--', lw=1.5,
           label=f'L_DP={L_DP_this:.0f}mm ({glabel})')
```
Each row now accurately reflects its own morphological boundary, showing correctly when the hold depth exceeds the distal phalanx.

---

## 2. Grip-Transition Annotations

### The Problem
The optimal grip boundaries (the background shading changes in the right panels of Fig 11) were clear visually, but did not provide exact numeric depths without the user having to visually interpolate from the X-axis.

### The Fix
The code now tracks transitions during the background-span loop:
```python
if cur_gk is not None:
    d_trans = 0.5 * (d_cmp[i-1] + d_cmp[i]) if i < len(d_cmp) else d_cmp[-1]
    transition_depths.append((d_trans, prev_gk, cur_gk))
```

And applies annotations after plotting the curves:
```python
for (d_tr, gk_from, gk_to) in transition_depths:
    ax_opt.axvline(d_tr, color='#333333', ls=':', lw=1.5, alpha=0.7)
    ax_opt.annotate(
        f'{grip_label[gk_from][:1]}→{grip_label[gk_to][:1]}\n{d_tr:.0f}mm',
        xy=(d_tr, min_force), ...
```

This makes the exact crossover point immediately readable (e.g., Crimp to Half-Crimp transition). 

---

## Files Changed

- `climbing_finger_3d.py` — Refactored the `fig11` inner loop to correctly reference `geom.L3` and added transition boundary annotations.
- `code_review.md` — this file

**Status: APPROVED — Simulation completed successfully, Fig 11 regenerated and visually verified.**
