# Critical Review of the *finger* Project
## A 3D Musculoskeletal Model of the Climbing Finger, Its Manuscript, and Its Practical Training Guide, Evaluated Against Current Literature

**Reviewed artefacts**
- Manuscript: [paper_draft.md](file:///Users/igorcerovsky/Documents/finger/paper/paper_draft.md)
- Field guide: [practical_training_guide.md](file:///Users/igorcerovsky/Documents/finger/paper/practical_training_guide.md)
- Simulation engine: [climbing_finger_3d.py](file:///Users/igorcerovsky/Documents/finger/climbing_finger_3d.py)
- Figures: [generate_publication_figures.py](file:///Users/igorcerovsky/Documents/finger/generate_publication_figures.py), `paper/figures/pub_fig1–3`
- Validation logs: `outputs/validation_peerj_iter*.txt`

**Review date:** 2026-10-07

**How the literature was gathered**
- I ran about 28 arXiv queries with the `literature-search-arxiv` skill. They covered climbing, finger tendon/pulley mechanics, fingertip contact and friction, tendon-driven robotic hands, tendon–sheath friction, musculoskeletal hand simulation, and the extensor mechanism. The arXiv papers used are listed in §11.1.
- arXiv carries very little climbing and hand-surgery research. For the core peer-reviewed literature (§11.2) I used web searches to confirm bibliographic details and key numbers. Please check §11.2 references against the original publications before citing them.

> [!NOTE]
> The review aims for a neutral, constructive peer-review tone. Where literature evidence is limited or contested, it says so, and it recommends rather than prescribes.

---

## 1. Executive Summary

The project is an ambitious, open-source static model of the middle finger during climbing grips. In its scope (3D kinematics, distributed contact over two phalanges, load-dependent skin friction, Capstan pulley friction, phenotype scaling) it goes beyond the classical planar formulations (An et al., 1983; Schweizer, 2001; Vigouroux et al., 2006). The code is reproducible, the parameters are mostly exposed in one `Config` class, and the manuscript has been moved toward cautious phrasing. These are real strengths.

The review does, however, find several methodological issues that affect the main conclusions. Most can be fixed.

| # | Issue | Affected claim(s) | Severity |
|---|---|---|---|
| 1 | Phenotype scaling multiplies segment lengths but keeps **tendon moment arms fixed**. The resulting "long-finger penalty" is mainly an artefact of this assumption. The headline "+37.5%" compares *long vs. short*. Long vs. standard is about **+15.6%**. | §4.3, §5.2, §6.3, Guide Rec. 3, §3, §4 | High |
| 2 | The FDP:FDS ratio is **imposed** through $r_{emg}(f_{DP})$. The FDP→FDS "crossover depths" therefore follow from the assumed coupling law, not from equilibrium mechanics. The reported "100% EMG ratio fidelity" is circular. | Abstract, §3, §4.1, Fig. 2A | High |
| 3 | The validation dataset (PeerJ 7470) is **Synek et al. (2019)**, not Vigouroux et al. Predicted/applied ratios range from 0.45 to 16.3, which does not support "excellent agreement". | Abstract, §3, Declarations | High |
| 4 | Predicted A2 loads are internally inconsistent (Table 2 vs. Table 3) and partly contradict the literature: half-crimp A2 > full-crimp A2, and open-hand A2 > full-crimp A2 in Table 2. Vigouroux et al. (2006) report A2 forces about 36× lower in slope than in crimp grip. | §4, §6.1, Guide Rec. 1 | High |
| 5 | Taken together, the predicted pulley loads and the chosen 300/400 N thresholds imply that A2 would fail during routine maximal hangs. This conflicts with everyday climbing experience, which points to a calibration problem. | §4.3, §6.4, Guide Rec. 3–4 | Medium–High |
| 6 | Several guide statements are not model outputs and lack direct literature support (physeal shear strength, +15–25% phalanx growth during PHV, >15 MPa joint contact stress, 12–24 month pulley remodelling, specific temperature/RH windows). | Paper §6.8–6.9, Guide §3 | Medium |
| 7 | Paper and code disagree on parameters, some references are misattributed or missing, and some terms are used loosely ("Hertzian triangular", "shear"). | Throughout | Medium |

Reframing issues 1–3 alone would substantially change the manuscript's claims. Without that reframing, the practical recommendations built on the long-finger penalty and the crossover depths do not yet appear to be supported by the model.

---

## 2. Summary of the Project's Approach

- **Kinematics:** a 4-DOF open chain (MCP flexion/abduction, PIP, DIP), with linear palmar ICR translation.
- **Contact:** a triangular pressure profile on the distal phalanx (DP); a second "ramp" on the middle phalanx (MP) when $d_{hold} > L_{DP}\cos\alpha$. The MP centroid is placed 60% toward the A3 pulley.
- **Soft tissue:** logarithmic pulp compression (from Serina et al., 1997) and a power-law friction coefficient $\mu \propto F_N^{n-1}$.
- **Muscles:** FDP, FDS, LU, EDC, RI, UI, with linear-in-angle moment arms fitted to Synek et al. (2019) path points, plus a wrist "tenodesis" offset.
- **Indeterminacy:** bounded least squares with $F_{FDP} = r_{emg}(f_{DP})\,F_{FDS}$ and an exponential EDC floor during DIP hyperextension.
- **Pulleys:** $\vec F_{A2} = T e^{\mu\theta}(\Delta\hat u_{MCP} + 0.5\,\Delta\hat u_{PIP})$ and $\vec F_{A4} = T e^{\mu\theta}\,0.4\,\Delta\hat u_{PIP}$.
- **Posture:** PIP/DIP angles chosen by minimising total tendon force with penalties, using continuation with Nelder–Mead.

---

## 3. State of the Art: Context for the Review

### 3.1 Finger musculoskeletal modelling
- **Planar and 3D static models** remain standard for climbing grips. Vigouroux et al. (2006) built a 3D static model of crimp and slope grips. They used EMG to quantify the passive DIP moment and report FDP:FDS ratios of **1.75 (crimp)** and **0.88 (slope)** as *model outputs*. Their A2 forces were about **36× lower** in slope than in crimp grip.
- **Moment arms and tendon paths:** An et al. (1983) remains the classical reference. Synek et al. (2019, *PeerJ* 7:e7470) identified human and bonobo index-finger models from CT-based tendon paths and compared them with in-vitro tendon-loading experiments. This is the dataset the project uses.
- **Force–length effects:** Goislard de Monsabert, …, Vigouroux (arXiv:2306.12842) found that wrist posture changes finger force capacity mostly through the force–length relationship of the extrinsic flexors, together with task constraints. This is a mechanistic alternative to the project's linear wrist moment-arm offset.
- **Extensor mechanism:** Dogadov, Valero-Cuevas et al. (arXiv:2507.15389) showed that the intercrossing fibre bundles of the extensor mechanism strongly change how muscle forces reach the middle and distal phalanges. The string-network idealisation can misallocate forces. This is relevant to the project's single "EDC" actuator and its exponential EDC floor.
- **Open simulation ecosystems:** MyoSuite (arXiv:2205.13600) and derived work (MS-MANO, arXiv:2404.10227; MUSIC, arXiv:2604.23886) provide validated, contact-capable musculoskeletal hand models. They offer an independent benchmark for cross-model comparison.

### 3.2 Climbing finger loads and injury
- **Pulley loading:** Schweizer (2001) estimated A2 load in the crimp grip at roughly **3×** the fingertip force. Marco et al. (1998) showed in cadavers that crimp-grip loading can rupture A2/A4. Their donors were elderly (mean ≈ 74 y), so the failure loads may underestimate those of trained climbers. Schöffl et al. (2009) reported that pulleys fail at **lower loads under eccentric** than concentric loading, with A2 the most frequent failure site.
- **Commonly cited A2 strength:** about **400 N** (Lin et al., 1990 and later citations). It is uncertain, specimen-dependent, and not established as a *yield* threshold. No literature source was found for a "300 N yield threshold".
- **Age-dependent injury patterns:** in adolescents the dominant overuse injury is the **epiphyseal (periphyseal) stress injury of the middle phalanx base** (Schöffl and colleagues; a 2025 five-grade PPSI classification). Pulley ruptures are comparatively **rare in youth**, most likely because the open physis is the weaker link.
- **Hold depth:** Amca et al. (2012) found that maximal vertical force rises with hold depth from 1 to 4 cm, with hand-level forces of about 350–576 N depending on grip and depth. They attributed the differences mainly to finger–hold contact rather than internal mechanics alone.

### 3.3 Climbing training evidence
- **Hangboard protocols:** López-Rivera & González-Badillo (2012, 2019) found that intermittent hangs favoured endurance and maximal hangs favoured strength. Later work (Devise et al., 2022; Hermans et al., 2022) and systematic reviews/meta-analyses (Stien et al., 2023; Langer et al., 2023) indicate that semi-specific finger training effectively improves finger strength and performance. Intensity, not grip morphology, is the main variable studied.
- **Edge standardisation:** many test protocols use fixed edges (e.g., 20 mm in IRCRA-style tests; 15–23 mm in intervention studies). Individualised edge depth is discussed by coaches, but no controlled studies using anthropometry-normalised edges were found. This leaves room for the project's "0.8·L_DP" proposal, but it remains a hypothesis.
- **Connective-tissue adaptation:** tendon and pulley tissue are widely held to adapt more slowly than muscle. Specific time constants for annular pulleys in climbers are not well established.

### 3.4 Robotics and soft-contact mechanics
- **Tendon–pulley design:** Khatik, Nishad & Saxena (arXiv:2010.02580; *J Biomech Eng* 2021) varied pulley and tendon attachment configurations to study tendon tension, bowstringing, and pulley stresses. This is an alternative to fixed load-sharing constants.
- **Tendon–sheath friction and hysteresis:** robotic tendon–sheath mechanisms (arXiv:1702.02063; arXiv:2605.16870) show direction-dependent friction and backlash hysteresis. Dermitzakis & Carbajal (arXiv:1401.5232) note a 9–12% difference between post-eccentric and post-concentric force in the human tendon–pulley system. MuJoCable (arXiv:2609.09612) implements *directional* Capstan propagation along surface-routed cables.
- **System identification:** Li et al. (arXiv:2408.13044) propose stepwise identification of the coupling matrix, joint viscoelasticity, and tendon friction for anthropomorphic tendon-driven fingers. This fits the project's passive DIP moment and Capstan parameters.
- **Anatomical testbeds:** the ACB Hand (arXiv:1909.07966) and MCR-Bionic Hand (arXiv:2606.13601) reproduce pulleys, volar plates, collateral ligaments, and the extensor hood. They could serve as instrumented physical surrogates. Tari et al. (arXiv:2609.05206) analyse conditioning of the actuation matrix and task Jacobian, which applies directly to the project's ill-conditioned "Direct 3×3" solutions.
- **Soft fingertip contact:** Xydas & Kao (1999) proposed the power-law soft-finger model $a \propto F_N^{\gamma}$ ($0 \le \gamma \le 1/3$, Hertz at 1/3). Jang et al. (arXiv:2310.04846) show that Coulomb validity and rotational stability depend on contact area and pressure. Friction-patch and limit-surface models (arXiv:1904.06677) treat combined shear and torsion. The PLATO Hand (arXiv:2602.05156) shows how nail, distal phalanx, and pulp divide fingertip deformation. FE fingertip studies (arXiv:1808.04252; arXiv:0909.3559; arXiv:1405.7848) show that subsurface pressure depends on surface curvature and skin thickness, and is not triangular in general. Persson-type contact mechanics (arXiv:2002.02226) link friction to adhesion and real contact area.
- **Climbing kinematics data:** the AscendMotion/ClimbingCap dataset (arXiv:2503.21268) provides 3D climbing motion that could constrain realistic forearm–hold orientations and MCP abduction ranges.

---

## 4. Methodological Review by Model Component

### 4.1 Kinematics and ICR
- The rotation composition and frame definitions are clear.
- The linear ICR shift is described as 1.0 mm in the paper. The code uses `icr_shift_max_PIP = 2.0` and `icr_shift_max_DIP = 1.5` ([L95–96](file:///Users/igorcerovsky/Documents/finger/climbing_finger_3d.py#L95-L96)). Reconciling the two is recommended.
- **Suggestion:** report a sensitivity of key outputs (A2 force, crossover depth) to ICR magnitude. Sub-millimetre changes in joint centre affect moment arms of 4–10 mm noticeably.

### 4.2 Contact model
1. **Terminology:** Hertz contact gives a semi-elliptical pressure distribution, not a triangular one. "Hertzian triangular" could be replaced with "assumed linear (triangular) pressure profile". Alternatively, a Hertz or Xydas–Kao power-law profile could be implemented.
2. **Peak location:** the model puts peak pressure at the distal tip ($s=0$), so the centroid sits at $d/3$ from the tip. On a square-cut edge, pulp indentation is plausibly greatest at the **edge lip**, at distance $d$ from the tip. That would move the centroid toward $2d/3$ and *reduce* the DIP external moment. The two assumptions bracket the result, and the outputs may be sensitive to this choice. Pressure-film or tactile-array measurements on instrumented edges are encouraged to settle it.
3. **"A3 anchoring":** the MP contact centroid is set to $0.4\,p_{geom} + 0.6\,p_{A3}$, described as "60% load transfer into the fibrous sheath". External contact load reaches the phalanx through skin and subcutaneous tissue. Annular pulleys restrain tendons and do not anchor external loads. The weighting should be presented as an empirical centroid correction, or replaced with a pressure-derived centroid.
4. **Area integrals:** the MP area expression $x^2/[2(L_3+x)]$ has no stated derivation. A short appendix deriving it from the assumed pressure field would aid reproducibility.

### 4.3 Soft tissue and friction
- Pulp compression: the paper gives $\delta_{max} = 2.5$ mm, the code uses `pulp_compress_max = 4.0` ([L106](file:///Users/igorcerovsky/Documents/finger/climbing_finger_3d.py#L106)). A 4 mm palmar shift on a 9 mm-thick DP is large and probably affects moment arms.
- The power-law friction form fits the skin-tribology literature (Derler & Gerhardt, 2012). Chalk, hold material, and hydration are not modelled. Statements on dry-firing, temperature, and RH (§6.8, Guide Rec. 8) are therefore literature-based or anecdotal, **not model outputs**, and would benefit from being labelled that way.
- **Side-pulls:** friction is resolved component-wise. Combined tangential force and torsion would be better captured with a soft-finger limit surface (Xydas & Kao, 1999; arXiv:1904.06677).

### 4.4 Moment arms and wrist coupling
- Synek et al. (2019) modelled the **index** finger. The project applies the fitted moment arms to a **middle**-finger geometry (Ozsoy-based lengths). This could be stated as a limitation.
- $\Delta ma_{wrist} = k_{wrist}\theta_{wrist}$ is openly flagged as phenomenological, which is good practice. Wrist extension changes extrinsic flexor *capacity* mainly through force–length behaviour (arXiv:2306.12842). It is not established that it shifts the MCP *moment arm* by about 1 mm. Modelling the effect through force–length capacity constraints may be more defensible.

### 4.5 Muscle redundancy
- Explicit FDP:FDS coupling is a reasonable way to avoid FDP collapsing to zero under pure optimisation. But the ratios **1.75 and 0.88 are outputs** of Vigouroux et al. (2006), not measured EMG ratios. **No literature source was found for the 1.20 half-crimp value.**
- **The coupling law decides the crossover.** With $F_{FDP} = r_{base}(0.2 + 0.8 f_{DP})F_{FDS}$, FDP and FDS are equal exactly when $f_{DP} = (1/r_{base} - 0.2)/0.8$:
  - half-crimp: $f_{DP} \approx 0.79$;
  - full crimp: $f_{DP} \approx 0.46$;
  - open hand: never, because $r_{base} < 1$.

  The crossover depth is therefore the depth at which the *contact partition* reaches a pre-set fraction. Neither the $0.2 + 0.8 f_{DP}$ form nor the "100% EMG fidelity" is an independent prediction.
- **Similarity check:** the reported crossovers divided by $L_{DP}$ are 28.3/18.7 = 1.51, 33.5/22.0 = 1.52 and 38.4/25.3 = 1.52. The crossover is thus a constant $d/L_{DP} \approx 1.52$, as expected from geometric similarity. "Long digits fail to reach crossover" only reflects the fixed absolute depth window (≤ 35 mm). Reporting results against $d/L_{DP}$ is recommended.
- The "Direct (3×3)" open-hand solution (11,506 N total, Table 2) indicates an ill-conditioned system. Reporting the condition number of the moment-arm matrix (cf. arXiv:2609.05206) and leaving non-physiological solutions out of the main table would improve clarity.
- **Alternatives:** EMG-informed optimisation with measured climbing EMG; or a min-max / sum-of-squared-activations criterion with force–length–PCSA capacity bounds, with the results compared against the imposed-ratio approach as a sensitivity analysis.

### 4.6 Pulley forces
- The load-sharing constants (A2 takes 50% of PIP deflection, A4 40%, "A3/capsule 10%") come from no cited quantitative source. They could be derived from a tendon-path model (e.g., arXiv:2010.02580) or varied in a sensitivity analysis.
- **Capstan direction:** in a quasi-static hang, the direction of tendon–sheath friction depends on loading history (concentric vs. eccentric) (Schöffl et al., 2009; arXiv:1401.5232; arXiv:1702.02063). A one-directional multiplier $e^{+\mu\theta}$ is one limiting case. Reporting both bounds ($e^{\pm\mu\theta}$) is suggested. The coefficient also differs: paper $\mu_t = 0.09$, code `mu_tendon = 0.08` ([L88](file:///Users/igorcerovsky/Documents/finger/climbing_finger_3d.py#L88)).
- **Wrist direction:** the MCP deflection uses a fixed wrist position `[-50, 0, 0]` mm. A2 load is therefore sensitive to an assumed forearm direction that does not vary with grip. This may explain why half-crimp (MCP 15°) exceeds full crimp (MCP 2.6°) in A2 load, against the literature. At the MCP level the tendon is also redirected largely by A1 and the palmar aponeurosis pulley, not only by A2.
- **"Lateral shear":** $F_{A2,lat}$ is the transverse *component* of the resultant pulley load vector, not shear stress within the pulley material. Calling it "transverse (mediolateral) pulley load component" avoids implying a material failure mode. The ">60 N dangerous" label (Fig. 3A) has no tissue-strength reference.

---

## 5. Validation Review

- **Attribution:** *PeerJ* 7:e7470 is **Synek, Lu, Vereecke, Nauwelaerts, Kivell & Pahr (2019)**, *"Musculoskeletal models of a human and bonobo finger: parameter identification and comparison to in vitro experiments."* The manuscript credits it to "Vigouroux, Domalain & Berton (2019)" with a different title. This should be corrected in the text, the references, and the Declarations.
- **Inverse vs. forward comparison:** in the experiment, *known tendon loads* produced *measured fingertip forces*. The project instead feeds the measured fingertip force into its inverse solver and compares the predicted tendon forces with the applied loads. The reported Pred/App ratios of **0.45–0.74 (flexed)**, **3.8–4.2 (Hook)** and **15.9–16.3 (HyperExt)** amount to errors of −55% to +1530%. The current wording ("excellent order-of-magnitude agreement") is not supported by these numbers.
  - The more natural test is a **forward simulation**: apply the experimental tendon loads, predict fingertip force magnitude and direction, and compare with the measured values (Synek et al. did this for their own models).
- **HyperExt explanation:** attributing the 16× discrepancy to "cadaveric jig damping" is speculative. A second possible reason is that the model's crimp-specific terms (EDC floor, RI/UI recruitment, imposed ratio 1.75) are not the right representation for that posture. The validation logs show RI/UI reaching 16 N in HyperExt at 300 g, where the applied tendon load is only 4.9 N.
- **No climbing-specific validation:** none of the four postures involves contact on an edge. The contact model, depth dependence, and pulley-load predictions, which carry the main claims, are therefore not validated. Possible validation targets:
  - Schweizer (2001) A2/fingertip ratio (~3 in crimp);
  - Vigouroux et al. (2006) crimp/slope A2 and A4 ratios;
  - Amca et al. (2012) depth–force trends.

---

## 6. Review of Main Results

| Claim (manuscript) | What the model actually supports | Literature consistency | Assessment |
|---|---|---|---|
| FDP→FDS crossover at 28–35 mm | Depth at which $f_{DP}$ reaches a value fixed by the imposed coupling law; constant at $d/L_{DP} \approx 1.52$ | No direct literature for depth-dependent FDP:FDS crossover | Hypothesis; depends on assumptions |
| Long-finger penalty "+37.5%" | +15.6% long vs. standard; +37% long vs. short; both with **unscaled tendon moment arms** | Moment arms broadly scale with bone size; under isometric scaling tendon forces at equal relative depth are length-invariant | Mostly a modelling artefact; reframing recommended |
| Long digits exceed A2 rupture limit in static hangs | Follows from the item above plus the thresholds chosen | Climbers routinely sustain about 100–170 N per middle finger (estimated from Amca et al. 2012 hand forces) without rupture | Points to calibration error; reinterpretation recommended |
| Open hand reduces A2 >70% | Table 3 has no open-hand data; Table 2 shows open-hand A2 (2.0 MPa) > full crimp (0.8 MPa) | Vigouroux et al. (2006): slope A2 ≈ 36× lower than crimp | Internally inconsistent; the direction agrees with literature, not with the model's own Table 2 |
| Half-crimp A2 > full-crimp A2 (Table 3) | Model output, likely driven by the fixed wrist direction and MCP angle | Literature: full crimp places the highest load on A2 | Conflicts with literature; investigation recommended |
| Half-crimp optimal below 8 mm (Fig. 2B) | Fig. 2B shows half-crimp total force **above** full crimp and open hand at every depth | — | Not supported by the figure itself; feasibility (friction/geometry) constraints not shown |
| 15° abduction → 61 N A2 transverse load | Plausible direction; magnitude depends on the fixed wrist vector and the A2 sharing constant | Side-pull injury mechanism is clinically plausible; no quantitative literature comparison | Qualitative insight; magnitude uncertain |
| A4 rarely tears in isolation | Model gives moderate A4 load | Isolated A4 ruptures are reported clinically; Marco et al. (1998) observed A4 failure in cadavers | Overstated; softening recommended |
| Dynamic slip → 536–621 N A2 | Static model scaled to 171.7 N; no dynamics | Eccentric failure occurs at lower loads (Schöffl et al., 2009) | Direction plausible; numbers are static extrapolations |

**Internal consistency of A2 values.** Table 2 (171.7 N load) gives crimp A2 = 0.8 MPa. Using the code's pressure area ($15 \times 4$ mm²) that is about 48 N. Table 3 gives 536.8 N for the same load (dynamic column). The two tables seem to come from different configurations (wall angle, COM vectoring, hold depth). The paper does not explain this, and the difference is about 11×.

---

## 7. Review of the Practical Training Guide

The guide is well organised and clearly written. Some recommendations agree with established practice (open-hand-dominant volume, careful footwork, release on foot slip, conservative youth loading). However, its stated *justifications* rely on the model results questioned above.

| Recommendation | Evidence basis | Comment |
|---|---|---|
| Rec. 1: 80/20 open-hand bias | Literature (Schweizer, 2001; Vigouroux et al., 2006) supports lower A2 load in open grips | The 80/20 split is a heuristic; the model's own Table 2 contradicts the "<100 N" claim |
| Rec. 2: Forearm–hold alignment | Mechanically plausible; quantitative magnitudes uncertain | The "vanity load" remarks are opinion; neutral phrasing recommended |
| Rec. 3: Long-finger protocol | Depends on the moment-arm scaling artefact | Withholding until scaling is corrected is recommended |
| Rec. 4: Footwork / release on slip | Consistent with eccentric failure data (Schöffl et al., 2009) | Sound; numerical values to be updated |
| Rec. 5: Curved rungs / pocket guidance | Lumbrical injuries in pockets are documented clinically | "Quadriga effect" applies mainly to FDP of digits III–V; reasonable advice, not model-derived |
| Rec. 6: Escaping middle finger | Mechanistic hypothesis; not modelled (no dynamics, no multi-digit) | The "100% momentum on one finger for 20–60 ms" figure has no source; labelling as hypothesis recommended |
| Rec. 7: Edge = 0.8·L_DP | Consistent with the model's similarity scaling | A testable, potentially valuable proposal; empirical validation encouraged |
| Rec. 8: Tribology / climate | Literature and practitioner knowledge; not modelled | Specific °C/RH windows have no cited source |
| §3.1 Kids: physeal shear 2–4 MPa; 30–40 N failure | No model or cited source | Recommend removal or citation. Literature emphasises adolescents (around PHV), not mainly pre-pubertal children |
| §3.2 Juniors: "+15–25% phalanx growth in 12–18 mo", "pulley rupture epidemic" | Not supported | Literature: PPSI dominates and pulley ruptures are rare in youth. Revision strongly recommended |
| §3.4 Masters: ">15–20 MPa", accelerated OA from finger length | No model output (no cartilage contact model) | Recommend labelling as speculative or removing |
| §4 Micro-cycle table by phenotype | Not supported by intervention studies | Recommend presenting as illustrative only |

Note also that guide Rec. 3 states "+37.5% force penalty" for +15% length relative to *standard*, while the model data give about +15.6% for that comparison.

---

## 8. Code ↔ Manuscript Consistency

| Parameter | Manuscript | Code (`Config`) |
|---|---|---|
| Tendon–sheath friction $\mu_t$ | 0.09 | 0.08 |
| ICR shift $c_{max}$ | 1.0 mm (all joints) | 2.0 mm PIP, 1.5 mm DIP |
| Pulp compression $\delta_{max}$ | 2.5 mm | 4.0 mm |
| Half-crimp FDP:FDS | "Vigouroux et al. 2006" | 1.20 (not in Vigouroux et al. 2006) |
| Validation source | Vigouroux et al. 2019 | `human_bonobo/` = Synek et al. 2019 data |

Exporting a machine-generated parameter table from `Config` straight into the manuscript would prevent this kind of drift.

---

## 9. Reference Integrity

- **Misattributed:** "Vigouroux, Domalain & Berton (2019) PeerJ 7:e7470" should be **Synek et al. (2019)**.
- **Cited but missing from the list:** Lin et al. (1990), Moor et al. (2009).
- **Could not be confirmed during this review; please verify:** Bourne et al. (2011, *Sports Biomechanics*); Lutter et al. (2021, OJSM, "wrist kinematics"); Lutter, Schweizer & Schöffl (2020, *Sportverletz Sportschaden*) details; King & Lien (2018) page range.
- **Questionable relevance:** Johansson & Flanagan (2009) is a tactile-neuroscience review and does not support a "Hertzian/elastomeric pressure" profile. Serina et al. (1997) supports pulp force–displacement, not the specific constants used.
- **Missing key literature:** Marco et al. (1998); Schöffl et al. (2009, eccentric loading); Synek et al. (2019); Xydas & Kao (1999); hangboard intervention studies and meta-analyses; PPSI literature.

---

## 10. Prioritised Recommendations

**A. Essential before submission**
1. **Scale moment arms with segment size** (isometric baseline), and treat allometric deviation as a separate, literature-informed sensitivity case. Report long vs. *standard* comparisons explicitly.
2. **Recast the crossover analysis.** Either remove the $f_{DP}$-dependent ratio and let an optimisation criterion with capacity bounds decide recruitment, or present the crossover openly as a consequence of the assumed coupling. Report results against $d/L_{DP}$.
3. **Redo validation as a forward simulation** against Synek et al. (2019), with corrected attribution. Add literature benchmarks for crimp vs. slope (Schweizer, 2001; Vigouroux et al., 2006).
4. **Resolve the A2 inconsistencies** (Tables 2 vs. 3; half-crimp vs. full crimp; open-hand claims) and document the configuration behind each table.
5. **Reconsider the thresholds.** Drop the unsupported "300 N yield". Present 400 N as an uncertain cadaveric reference from elderly donors, and discuss why predicted routine loads approach it.

**B. Strongly recommended**
6. Run a global sensitivity analysis (e.g., Sobol or Morris) over ICR, pulp compression, pulley-sharing constants, $\mu_t$, wrist vector, contact-peak location, and FDP:FDS assumptions.
7. Label every practical statement as **model-derived**, **literature-derived**, or **hypothesis**, and move non-model content (physis, OA, climate) to a separately referenced discussion.
8. Revise the age-specific section to match the PPSI literature.
9. Harmonise code and manuscript parameters through an auto-generated table.

**C. Robotics-inspired extensions**
10. **Directional Capstan with hysteresis** (arXiv:2609.09612; arXiv:1702.02063; arXiv:1401.5232) to bound eccentric vs. concentric pulley loads.
11. **Tendon-path-derived pulley loads** (arXiv:2010.02580) in place of fixed 0.5/0.4 sharing; optionally a fibre-network extensor model (arXiv:2507.15389).
12. **Soft-finger contact** using the Xydas–Kao power law and limit-surface friction for side-pulls (arXiv:1904.06677; arXiv:2310.04846), calibrated against FE fingertip data (arXiv:1808.04252).
13. **Cross-model benchmarking** against MyoSuite/MyoHand (arXiv:2205.13600) for identical postures and loads.
14. **Physical surrogate testing** on an anatomically faithful tendon-driven finger (arXiv:1909.07966; arXiv:2606.13601) with load cells at A2/A4 and an instrumented edge, using the identification protocol of arXiv:2408.13044.
15. **Realistic loading envelopes** from climbing motion data (arXiv:2503.21268) to set plausible ranges for MCP abduction and forearm direction.

**D. Proposed empirical studies (where the project could contribute novel data)**
- An instrumented edge with a pressure-film or tactile array at 6–40 mm depth, to settle the location of the pressure peak and the onset of DP/MP load sharing as a function of $d/L_{DP}$.
- A cross-sectional study relating anthropometry ($L_{DP}$, total digit length, hand size) to maximal force on fixed vs. $0.8\,L_{DP}$-normalised edges, testing whether normalisation reduces between-subject variance.
- Ultrasound of A2 tendon–bone distance under graded loads in crimp vs. half-crimp vs. open grips, as an in-vivo benchmark for predicted pulley-load trends.

---

## 11. References

### 11.1 arXiv papers retrieved and used (via `literature-search-arxiv`)

| arXiv ID | Title (short) | URL |
|---|---|---|
| 2010.02580 | Comprehending finger flexor tendon pulley system using a computational analysis (Khatik et al.; J Biomech Eng 2021) | https://arxiv.org/abs/2010.02580 |
| 2507.15389 | Intercrossing fibers of the extensor mechanism influence muscle force transmission (Dogadov, Valero-Cuevas et al.) | https://arxiv.org/abs/2507.15389 |
| 2306.12842 | Force–length relationship and task constraints on finger force capacity (Goislard de Monsabert, …, Vigouroux) | https://arxiv.org/abs/2306.12842 |
| 2205.13600 | MyoSuite: contact-rich musculoskeletal simulation suite | https://arxiv.org/abs/2205.13600 |
| 2404.10227 | MS-MANO: hand pose tracking with biomechanical constraints | https://arxiv.org/abs/2404.10227 |
| 2604.23886 | MUSIC: learning muscle-driven dexterous hand control | https://arxiv.org/abs/2604.23886 |
| 2408.13044 | Identification and validation of a tendon-driven anthropomorphic finger dynamic model | https://arxiv.org/abs/2408.13044 |
| 2601.20682 | Tendon-based modelling, estimation and control for a high-DoF anthropomorphic hand | https://arxiv.org/abs/2601.20682 |
| 1909.07966 | Design of the Anatomically Correct, Biomechatronic (ACB) Hand | https://arxiv.org/abs/1909.07966 |
| 2606.13601 | MCR-Bionic Hand: anatomical structural priors for dexterous manipulation | https://arxiv.org/abs/2606.13601 |
| 2609.05206 | Morphology and actuation as inductive biases in robotic hand manipulation | https://arxiv.org/abs/2609.05206 |
| 2609.09612 | MuJoCable: surface-routed cable transmission with directional Capstan | https://arxiv.org/abs/2609.09612 |
| 1702.02063 | Tendon-driven surgical robots with friction and hysteresis | https://arxiv.org/abs/1702.02063 |
| 2605.16870 | SSTL: hysteresis modelling in tendon–sheath mechanisms | https://arxiv.org/abs/2605.16870 |
| 1401.5232 | Bio-inspired friction switches: adaptive pulley systems | https://arxiv.org/abs/1401.5232 |
| 2310.04846 | Soft finger rotational stability for precision grasps | https://arxiv.org/abs/2310.04846 |
| 1904.06677 | Quasi-static planar sliding using friction patches | https://arxiv.org/abs/1904.06677 |
| 2602.05156 | PLATO Hand: fingernail-shaped contact behaviour | https://arxiv.org/abs/2602.05156 |
| 1808.04252 | Structural FE model of fingertip response to tactile stimuli | https://arxiv.org/abs/1808.04252 |
| 0909.3559 | Compliance and hysteresis of synthetic vs. human fingertip skin | https://arxiv.org/abs/0909.3559 |
| 1405.7848 | Artificial skin thickness and subsurface pressure profiles | https://arxiv.org/abs/1405.7848 |
| 2002.02226 | Sphere and cylinder contact mechanics during slip (Persson group) | https://arxiv.org/abs/2002.02226 |
| 2109.11504 | Distributed contact force measurement for slip detection | https://arxiv.org/abs/2109.11504 |
| 2503.21268 | ClimbingCap / AscendMotion climbing motion dataset | https://arxiv.org/abs/2503.21268 |

### 11.2 Peer-reviewed literature (confirmed via web search; please verify against originals before citation)
- Amca, A.M., Vigouroux, L., Aritan, S., Berton, E. (2012). Effect of hold depth and grip technique on maximal finger forces in rock climbing. *J Sports Sci* 30(7):669–677.
- An, K.N. et al. (1983). Tendon excursion and moment arm of index finger muscles. *J Biomech* 16(6):419–425.
- Derler, S., Gerhardt, L.C. (2012). Tribology of skin. *Tribol Lett* 45:1–27.
- López-Rivera, E., González-Badillo, J.J. (2019). Comparison of the effects of three hangboard strength and endurance training programs on grip endurance in sport climbers. *J Hum Kinet*.
- Marco, R.A., Sharkey, N.A., Smith, T.S., Zissimos, A.G. (1998). Pathomechanics of closed rupture of the flexor tendon pulleys in rock climbers. *J Bone Joint Surg Am* 80(7):1012–1019.
- Schöffl, I., Einwag, F., Strecker, W., Hennig, F., Schöffl, V. (2009). Impact of eccentric and concentric loading on pulley rupture (cadaver study). *J Appl Biomech*.
- Schweizer, A. (2001). Biomechanical properties of the crimp grip position in rock climbers. *J Biomech* 34(2):217–223.
- Stien, N. et al. (2023). Effects of climbing-specific strength and endurance training: systematic review and meta-analysis.
- Synek, A., Lu, S.-C., Vereecke, E.E., Nauwelaerts, S., Kivell, T.L., Pahr, D.H. (2019). Musculoskeletal models of a human and bonobo finger: parameter identification and comparison to in vitro experiments. *PeerJ* 7:e7470.
- Vigouroux, L., Quaine, F., Labarre-Vila, A., Moutet, F. (2006). Estimation of finger muscle tendon tensions and pulley forces during specific sport-climbing grip techniques. *J Biomech* 39(14):2583–2592.
- Xydas, N., Kao, I. (1999). Modeling of contact mechanics and friction limit surfaces for soft fingers in robotics. *Int J Robot Res* 18(9):941–950.
- Schöffl, V. and colleagues: epiphyseal/periphyseal stress injuries of the middle phalanx in adolescent climbers, including a 2025 five-grade PPSI classification (*Frontiers*). Exact citation to be confirmed.

> [!IMPORTANT]
> Please check the licence of each retrieved arXiv paper before reusing figures or text, and consult the arXiv API terms of use: https://info.arxiv.org/help/api/index.html
