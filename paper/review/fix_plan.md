# Biomechanical Model & Manuscript Fix Plan
## Systematic Remediation Based on Literature Review (`finger_project_review.md`)

This plan coordinates the methodological, computational, and editorial corrections required to align the *finger* project with state-of-the-art finger biomechanics, climbing sports medicine, and robotics literature.

---

### Task Breakdown & Status

#### Phase 1: Planning & Tracking
- [x] **Task 1.1: Create Fix Plan**
  - Establish `paper/review/fix_plan.md` with status indicators.
  - Prioritize tasks from Essential (A1–A5) to Recommended (B6–B9).

#### Phase 2: Essential Methodological & Code Fixes (Review §10.A)
- [x] **Task 2.1: Implement Anthropometric Moment Arm Scaling (Issue 1 / Rec A1)**
  - *Completed:* Added `scale_factor` to `FingerGeometry` and `scale_moment_arms_with_geometry` config flag. `moment_arms(grip, geom=None)` scales $ma \propto f$ isometrically by default. Clarified isometric invariance vs unscaled lever arm penalty (+15.6% Long vs Std, +36.9% Long vs Short) across code, figures, and manuscript.
- [x] **Task 2.2: Recast the Crossover Mechanics & EMG Ratio Coupling (Issue 2 / Rec A2)**
  - *Completed:* Recast $r_{emg}(f_{DP})$ as a phenomenological load-partitioning model in code, docstrings, figures, and manuscript. Expressed crossover as a scale-invariant dimensionless threshold $\tilde{d}_{crossover} = d / L_{DP} \approx 1.52$.
- [x] **Task 2.3: Correct Validation Attribution & Forward Validation Reporting (Issue 3 / Rec A3)**
  - *Completed:* Corrected citation to Synek et al. (2019) across `paper_draft.md`, `compare_models.py`, `generate_publication_figures.py`, and references. Corrected "HyperExt" definition to MCP hyperextension. Transparently reported inverse solver discrepancies in extreme cadaveric postures and benchmarked against Schweizer (2001; 3:1 crimp A2 ratio) and Vigouroux et al. (2006).
- [x] **Task 2.4: Resolve A2 Pulley Inconsistencies & Wrist Angle Dynamics (Issue 4 / Rec A4)**
  - *Completed:* Decoupled A2 pulley deflection from MCP in `compute_pulley_angles` and `pulley_forces_3d` ($\vec{F}_{A2} = T_{A2} \cdot 0.50 \Delta\hat{u}_{PIP}$), reserving MCP deflection for A1. Harmonized Table 2 and Table 3 simulation values: Crimp produces the highest A2 load (483.6 N at 171.7 N load; 8.06 MPa); Open Hand reduces A2 load to <100 N (57.1 N at 171.7 N load; 0.95 MPa).
- [x] **Task 2.5: Re-evaluate Structural Thresholds & Rupture Limits (Issue 5 / Rec A5)**
  - *Completed:* Removed speculative 300 N yield threshold from manuscript, guide, and figures. Framed 400 N cautiously as an elderly cadaveric benchmark (Lin et al. 1990; Schöffl et al. 2009) and contextualized with in vivo remodeling capacity and dynamic slip loading.

#### Phase 3: Scientific Phrasing, Consistency & Guide Revision (Review §10.B)
- [x] **Task 3.1: Code ↔ Manuscript Parameter Harmonization (Issue 7 / Rec B9)**
  - *Completed:* Harmonized parameters across `climbing_finger_3d.py`, `paper_draft.md`, and `practical_training_guide.md`: $\mu_t = 0.08$, $c_{max,PIP} = 2.0$ mm, $c_{max,DIP} = 1.5$ mm, $\delta_{max} = 2.5$ mm.
- [x] **Task 3.2: Scientific Phrasing & Terminology Updates (Issue 7 & Rule Guidelines)**
  - *Completed:* Replaced "Hertzian triangular" with "linear triangular pressure formulation". Replaced "lateral shear" with "transverse (mediolateral) pulley load component". Strictly enforced non-dogmatic, scientific phrasing.
- [x] **Task 3.3: Overhaul Age-Specific Biomechanics in Paper & Training Guide (Issue 6 / Rec B8)**
  - *Completed:* Overhauled §6.9 of `paper_draft.md` and Section 3 of `practical_training_guide.md` to center on Primary Periphyseal Stress Injuries (PPSI Stages I–IV) of the middle phalanx base (Schöffl's classification). Clarified that acute pulley ruptures are rare in youth due to the cartilaginous physis being the mechanical weak link.
- [x] **Task 3.4: Reference Integrity & Bibliographic Fixes (Review §9)**
  - *Completed:* Added Lin et al. (1990), Marco et al. (1998), Moor et al. (2009), Roloff et al. (2006), Schöffl et al. (2009, 2023), Synek et al. (2019) to References; removed fictitious Vigouroux (2019) citation.

#### Phase 4: Regenerate Figures, Re-run Tests & Export
- [x] **Task 4.1: Update Figure Generation Scripts**
  - *Completed:* Updated `generate_publication_figures.py` with Linear triangular peak, Synek et al. (2019), dimensionless crossover $\tilde{d} \approx 1.52$, dynamically evaluated unscaled lever arm pulley loads, removed 300 N line, and regenerated publication figures.
- [x] **Task 4.2: Re-run Backward Compatibility Tests**
  - *Completed:* Re-ran `test_match_human_bonobo.py` (all tests passed) and `human_bonobo/compare_models.py` (clean output).
- [x] **Task 4.3: Export Updated Manuscripts to PDF**
  - *Completed:* Ran `export_papers_to_pdf.py` via virtual environment Python; successfully generated `paper_draft.pdf` (13 pages, 3 embedded figures) and `practical_training_guide.pdf` (11 pages).
- [x] **Task 4.4: Summary Presentation & User Review**
  - *Completed:* Presenting all changes, test results, and validation metrics to the user; strictly awaiting user confirmation prior to any git commit.
