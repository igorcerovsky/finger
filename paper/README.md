# Climbing Finger Biomechanics: Publications & Diagnostic Figures

This directory contains the publication-ready academic manuscript, companion athletic field guide, and high-resolution figures for the 3D musculoskeletal finger biomechanics project.

**Author:** Igor Cerovsky  
*Independent Climbing Physics Enthusiast*  
**Correspondence:** [igor.cerovsky@gmail.com](mailto:igor.cerovsky@gmail.com)  
**GitHub Repository:** [https://github.com/igorcerovsky/finger](https://github.com/igorcerovsky/finger)  

---

## Directory Overview

```
paper/
├── README.md                     # Overview and reproducibility guide (this file)
├── paper_draft.md                # Primary academic research manuscript
├── practical_training_guide.md   # Companion practical field guide for coaches & athletes
├── pdf/                          # Publication-ready vector PDFs (KaTeX math + 300 DPI figures)
│   ├── paper_draft.pdf           # 12-page academic manuscript PDF
│   └── practical_training_guide.pdf # 11-page field guide PDF
├── script/                       # Build and export tooling
│   └── export_papers_to_pdf.py   # Automated PDF export script (uses .venv python)
└── figures/                      # High-resolution (300 DPI) publication-grade figures
    ├── pub_fig1_model_validation.png
    ├── pub_fig2_hold_depth_crossover.png
    └── pub_fig3_shear_and_scaling.png
```

---

## 1. Publications

### Primary Academic Manuscript
- **Markdown Source:** [`paper_draft.md`](file:///Users/igorcerovsky/Documents/finger/paper/paper_draft.md)
- **Publication PDF:** [`paper_draft.pdf`](file:///Users/igorcerovsky/Documents/finger/paper/pdf/paper_draft.pdf) (12 pages, vector formulas & figures)
- **Title:** *A Three-Dimensional Musculoskeletal Model of the Climbing Finger: Dual-Phalanx Contact Mechanics, Anthropometric Phenotypic Scaling, and Out-of-Plane Annular Pulley Shearing*
- **Target Venues:** *Journal of Biomechanics*, *Frontiers in Bioengineering and Biotechnology*, or *Sports Biomechanics*.
- **Length:** 12 pages (standard academic formatting, ~4,800 words).
- **Core Scientific Contributions:**
  1. **4-DOF Spatial Kinematics:** Resolves non-sagittal joint actions (MCP flexion/abduction, PIP/DIP flexion) under spatial coordinate transforms.
  2. **Dual-Phalanx Hertzian Contact Mechanics:** Captures hold depth transitions ($s \le L_{DP}$ vs. $s > L_{DP}$) where the middle phalanx engages the hold edge and anchors the A3 pulley, eliminating external DIP moment arms.
  3. **Biomechanical Grip Crossovers:** Pinpoints the exact edge depth thresholds ($28.3\text{–}38.4\text{ mm}$) where FDS overtakes FDP as prime mover.
  4. **Annular Pulley Vector Shearing:** Formulates 3D Capstan deflection mechanics ($T(\hat{\mathbf{u}}_{in} - \hat{\mathbf{u}}_{out})$) and demonstrates that $15^\circ$ of MCP radial abduction produces $>60\text{ N}$ of transverse lateral shear on A2/A4.
  5. **Phenotypic Scaling Penalties:** Quantifies the $+37.5\%$ force penalty on long phalanges (+15%), explaining why long digits cross cadaveric rupture thresholds ($400\text{ N}$) on small edges under static bodyweight loads.
  6. **Cadaveric & In Vivo Validation:** Validated against cadaveric force-plate measurements across four standardized joint postures under 300 g and 950 g loads.

### Companion Practical Field Guide
- **Markdown Source:** [`practical_training_guide.md`](file:///Users/igorcerovsky/Documents/finger/paper/practical_training_guide.md)
- **Publication PDF:** [`practical_training_guide.pdf`](file:///Users/igorcerovsky/Documents/finger/paper/pdf/practical_training_guide.pdf) (11 pages, tables & training templates)
- **Title:** *Biomechanical Manual for Finger Training & Injury Prevention in Sport Climbing: A Practical Field Guide for Coaches, Clinicians, and Athletes*
- **Audience:** Climbing coaches, sports physical therapists, orthopedic clinicians, and dedicated athletes.
- **Core Practical Content:**
  1. **The 8 Core Biomechanical Recommendations:**
     - *Recommendation 1:* Grip load budgeting (80/20 Open Hand vs. Crimp ratio).
     - *Recommendation 2:* Forearm-hold co-linearity & Center of Gravity (CoG) pull vector optimization; avoiding the vanity peak-load trap on force gauges (Tindeq).
     - *Recommendation 3:* Long-finger periodization protocol (slower connective tissue adaptation, recommended 48–72h recovery).
     - *Recommendation 4:* Footwork as "pulley armor" (mitigating dynamic shock blowouts; climbing shoe rubber cleanliness).
     - *Recommendation 5:* Inter-digit asymmetry & ergonomic rung selection (Quadriga effect and middle finger hyper-flexion).
     - *Recommendation 6:* Joint capsule & collateral ligament protection (preventing dynamic slip trauma on the protruding long middle finger).
     - *Recommendation 7:* Edge normalization in athletic assessment ($d_{test} = 0.8 \times L_{DP}$).
     - *Recommendation 8:* Skin tribology, temperature & friction management.
  2. **Diagnostic Self-Assessment Table:** Clinical warning signs and actions for A2 morning stiffness, palmar PIP tenderness, collateral/capsular synovitis, and lumbrical shear.
  3. **Age-Specific Biomechanical Guidelines:** Connects phalanx lever scaling ($M_{ext} = r_{ext} \times F_{ext}$) and hold depth ratio ($d / L_{DP}$) across the lifespan: Kids (<12–13 yrs, growth plate shear & $d > L_{DP}$ inversion), Juniors (13–18 yrs, PHV lever-arm explosion & 12–24mo remodeling lag), Adults (18–45 yrs, +37.5% long-digit phenotypic penalty), and Masters (45+ yrs, cumulative lifetime contact stress & PIP osteoarthritis).
  4. **Phenotypic Micro-Cycle Template:** Side-by-side comparison of weekly training periodization for short (<85 mm) vs. long (>105 mm) finger phenotypes.

---

## 2. Publication Figures

| Figure | Filename | Description |
| :--- | :--- | :--- |
| **Figure 1** | [`pub_fig1_model_validation.png`](file:///Users/igorcerovsky/Documents/finger/paper/figures/pub_fig1_model_validation.png) | **3D Kinematics, Contact Pressure & Validation:** (A) 3D spatial bone segments in Full Crimp, Half-Crimp, and Open Hand. (B) Dual-phalanx contact pressure distributions ($p(s)$) for shallow ($10\text{ mm}$) and deep ($35\text{ mm}$) holds. (C) Predicted vs. measured force ratios across 4 cadaveric postures. |
| **Figure 2** | [`pub_fig2_hold_depth_crossover.png`](file:///Users/igorcerovsky/Documents/finger/paper/figures/pub_fig2_hold_depth_crossover.png) | **Hold Depth Redistribution & Grip Frontiers:** (A) FDP vs. FDS tendon forces across edge depths ($2\text{–}42\text{ mm}$) identifying crossover thresholds for Short, Nominal, and Long phenotypes. (B) Minimum-effort energetic grip selection zones (<8 mm Half-Crimp, 8–18 mm Transition, >18 mm Open Hand). |
| **Figure 3** | [`pub_fig3_shear_and_scaling.png`](file:///Users/igorcerovsky/Documents/finger/paper/figures/pub_fig3_shear_and_scaling.png) | **Transverse Shearing & Phenotypic Rupture Limits:** (A) Lateral pulley shear ($F_{A2,lat}$, $F_{A4,lat}$) and PIP joint shear as a function of MCP radial abduction ($0^\circ\text{–}20^\circ$). (B) Annular pulley loads for Short, Nominal, and Long phenotypes against the $300\text{ N}$ structural yield and $400\text{ N}$ ultimate rupture limits under 100 N ledge hang and 171.7 N dynamic slip conditions. |

---

## 3. PDF Export & Reproducibility

Both manuscripts can be compiled into publication-grade vector PDFs with embedded KaTeX mathematics and 300 DPI figures using the Python virtual environment:

```bash
# From repository root, execute the PDF export pipeline:
.venv/bin/python paper/script/export_papers_to_pdf.py

# Optional flags:
# Export only the scientific paper:
.venv/bin/python paper/script/export_papers_to_pdf.py --paper-only

# Export only the practical guide:
.venv/bin/python paper/script/export_papers_to_pdf.py --guide-only
```

### Full Simulation & Figure Reproduction
```bash
# Regenerate all high-resolution figures into paper/figures/ and outputs/:
.venv/bin/python generate_publication_figures.py

# Run model regression test suite:
.venv/bin/python test_match_human_bonobo.py
```
