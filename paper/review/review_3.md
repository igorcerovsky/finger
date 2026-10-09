# Independent Review of the *finger* Project (Round 3)
## Verification of the Remediations Claimed in Review 2, and Remaining Problems

**Reviewed artefacts (state on 2026-10-07)**
- Manuscript: [short_vs_long_finger_advantage.md](file:///Users/igorcerovsky/Documents/finger/paper/short_vs_long_finger_advantage.md)
- Field guide: [practical_training_guide.md](file:///Users/igorcerovsky/Documents/finger/paper/practical_training_guide.md)
- Prior reviews: [review_1.md](file:///Users/igorcerovsky/Documents/finger/paper/review/review_1.md), [review_2.md](file:///Users/igorcerovsky/Documents/finger/paper/review/review_2.md), [fix_plan.md](file:///Users/igorcerovsky/Documents/finger/paper/review/fix_plan.md)
- Code: [climbing_finger_3d.py](file:///Users/igorcerovsky/Documents/finger/climbing_finger_3d.py), [generate_publication_figures.py](file:///Users/igorcerovsky/Documents/finger/generate_publication_figures.py), [global_sensitivity_analysis.py](file:///Users/igorcerovsky/Documents/finger/benchmarks/global_sensitivity_analysis.py), [benchmark_myosuite.py](file:///Users/igorcerovsky/Documents/finger/benchmarks/benchmark_myosuite.py)
- Figures: `paper/figures/pub_fig1–3`

**How this review was done**
1. Each "PROPERLY FIXED" verdict in Review 2 was checked against the manuscript text, the guide, and the code. Review 2 was not taken at face value.
2. The current code was run with the same calls as the figure script (10 mm edge, vertical wall, `F_ext = [F, 0, 0]`). The aim was to check whether the published numbers can be reproduced. The verification script is kept as a scratch artefact (`verify_claims.py`, conversation scratch folder).
3. Key bibliographic details and quantitative literature values were checked by web search (§9). Please confirm against the originals before citing.

> [!NOTE]
> This review aims to be neutral and constructive. "Not fixed" means the problem is still present in the current artefacts. It does not imply that no effort was made. Several fixes are genuine improvements.

---

## 1. Executive Summary

The project has made real progress since Review 1:
- Isometric moment-arm scaling exists in the code and is the default.
- The coupling law is openly described as phenomenological.
- A2 deflection no longer includes the MCP angle.
- The 300 N "yield" threshold is gone.
- The code parameters for μ_t, ICR shift, and pulp compression now match the manuscript.

However, this independent check finds that **Review 2 overstates the remediation status**. Of the nine Category A/B items that Review 2 marked "PROPERLY FIXED", this review finds **2 fixed, 4 partially fixed, and 3 not fixed**. Several new problems also came up, some of which bear directly on the headline claims.

### 1.1 Most consequential findings (new or unresolved)

| # | Finding | Evidence | Impact |
|---|---|---|---|
| N1 | **The "transverse" A2/A4 pulley load under MCP abduction is a coordinate-frame artefact.** Expressed in the proximal-phalanx (pulley) frame, the A2 force has *exactly zero* mediolateral component at every abduction angle. The reported value is the projection onto the *global wall* z-axis. | `F_A2_lat = abs(F_A2_vec[2])` ([L1222](file:///Users/igorcerovsky/Documents/finger/climbing_finger_3d.py#L1222)); verification: PP-local z = 0.0000 N at φ = 0°, 5°, 15° | Abstract, §4.2, §5.1, §6.2, Guide Rec 2, diagnostic table "F_lat > 60 N" |
| N2 | **The PIP "mediolateral shear" equals F·sin φ exactly** (44.4 N = 171.7 N × sin 15°). It is an input assumption (the external force stays wall-normal while the finger rotates), not an emergent result. | verification output | §4.2, §6.2, Guide Rec 2/6 |
| N3 | **The published tables are not reproduced by the current code.** Example (crimp, 171.7 N): A2 = 521.7 N (paper 483.6 N), EDC = 186.9 N (paper 3.7 N), RI = 135.6 N (paper 286.5 N). Open hand: A2 = 106 N (paper 57.1 N; "below 100 N" no longer holds). At 100 N, crimp A2 = 303.9 N (Table 3: 281.6 N). | verification output vs. Tables 2–3 | All quantitative claims |
| N4 | **Figures disagree with the text and partly use hard-coded values.** Fig. 3A shows F_A2,lat ≈ 110 N at 15°; text says 61.0 N. Fig. 3B shows about 304 N standard crimp A2; Table 3 says 281.6 N. Fig. 1C bars are hard-coded old ratios (0.45/0.59/15.9/3.82); Table 1 lists 0.43–0.73/24.9–25.3/5.1–5.5. | [generate_publication_figures.py L99–102, L243](file:///Users/igorcerovsky/Documents/finger/generate_publication_figures.py#L99-L102) | Reproducibility |
| N5 | **The validation reference is still wrong, now with invented co-authors.** The manuscript lists "Synek, A., Cegoñino, J., Ramakrishna, A.S., Pérez del Palomar, A. (2019). *A subject-specific musculoskeletal model of the index finger…*". The actual paper is Synek, Lu, Vereecke, Nauwelaerts, Kivell & Pahr (2019), *Musculoskeletal models of a human and bonobo finger: parameter identification and comparison to in vitro experiments*. Five further references have wrong journals, volumes, or titles (§9). | References #24, #3, #11, #16, #18, #20 | Integrity |
| N6 | **The intrinsic muscles appear to carry physiologically implausible forces.** RI is recruited at 79–286 N in sagittal grips. This is driven by constant ulnar abduction moment arms for FDP/FDS (−2.1/−1.5 mm), an undisclosed default 5° radial abduction in crimp and half-crimp, and a solver that is explicitly "agnostic to PCSA" (no capacity bounds). | [L141–142](file:///Users/igorcerovsky/Documents/finger/climbing_finger_3d.py#L141-L142), [L214–215](file:///Users/igorcerovsky/Documents/finger/climbing_finger_3d.py#L214-L215), [L367–368](file:///Users/igorcerovsky/Documents/finger/climbing_finger_3d.py#L367-L368) | Total force, A2 load |
| N7 | **The sensitivity analysis does not support the "high numerical stability" claim.** It is one-at-a-time (not global). `a2_share` is applied as a post-hoc multiplier, so ±20% in gives ±20% out by construction. `r_base_hc` is passed in but never applied, so the outputs are identical at r_base = 1.0 and 1.5. | [global_sensitivity_analysis.py L29–48](file:///Users/igorcerovsky/Documents/finger/benchmarks/global_sensitivity_analysis.py#L29-L48); verification output | B6 verdict |
| N8 | **The headline claims in the Abstract and Conclusions were not updated.** They still state "up to 37.5% higher total tendon tension", "experimentally validated", ">60 N shearing", and "approach structural pulley failure limits at significantly lower body-weight percentages". | Abstract L15–19; §6.9; Guide §3.3, §3.5 | Primary claims |

### 1.2 Independent verdict on Review 2

| Item | Review 2 verdict | Independent verdict | Key reason |
|---|---|---|---|
| A1 Moment-arm scaling | Properly fixed | **Partially fixed** | Code correct. Abstract, §6.3, §6.9, Guide Rec 3 and §3 still build on the non-default *unscaled* case or cite 37.5%. |
| A2 Crossover / coupling | Properly fixed | **Partially fixed** | Framing improved, but the f_DP threshold is stated wrongly (0.46 vs. 0.79 for half-crimp). "EMG ratio fidelity" is still listed as validation. Abstract still presents it as a finding. Review 2 cites a formula (`f_DP^1.5`) that is not in the code. |
| A3 Attribution / forward validation | Properly fixed | **Not fixed** | The reference is mis-authored (N5). No forward simulation was done. The "forward validation" compares an A2/fingertip ratio, not a crimp/slope ratio. The HyperExt definition differs between manuscript and Review 2. |
| A4 A2 inconsistencies | Properly fixed | **Partially fixed** | MCP decoupling done. Half-crimp A2 ≥ full-crimp A2 persists (Table 2: 8.58 vs. 8.06 MPa; Table 3: 299.7 vs. 281.6 N). Current code gives open-hand A2 > 100 N. The crimp/open ratio (~5×) differs from Vigouroux et al. (2006; ~36×). |
| A5 Thresholds | Properly fixed | **Partially fixed** | "300 N" removed. But 171.7 N per middle finger is relabelled "dynamic shock" although it lies within the range of maximal static hangs. The model therefore still implies A2 rupture during controlled maximal hangs (§4.5). |
| B6 Sensitivity analysis | Properly fixed | **Not fixed** | See N7. Not reported in the manuscript. |
| B7 Evidence tiers | Properly fixed | **Partially fixed** | Tags exist in Guide §1 only. Paper §6, Guide §2–4 are untagged. Several tags do not match their content (§7). |
| B8 PPSI section | Properly fixed | **Partially fixed** | PPSI is now central. But "Stages I–IV" and "Salter-Harris II/III" do not match the 2025 five-grade classification. "+15–25% phalanx growth in 12–18 months" remains in the guide. |
| B9 Parameter harmonisation | Properly fixed | **Fixed** (for the four listed parameters) | μ_t, c_max, δ_max now agree. Other drift remains (N3, N4); no auto-generated parameter table. |
| C13 MyoSuite benchmark | Operational | **Methodologically limited** | See §5. Not reported in the manuscript. |

> [!WARNING]
> **Reliability of Review 2.** Review 2 cited `paper/paper_draft.md` (which was renamed to `paper/short_vs_long_finger_advantage.md` / companion to `paper/practical_training_guide.md`). It also misstates the coupling formula, lists posture angles that do not match the code (e.g., crimp "MCP = 30°, DIP = −15°" vs. code 2.6°/−22.6° with 5° abduction), and calls numbers "harmonised" that the current code no longer reproduces. It is therefore recommended to treat Review 2 as a change log rather than an independent audit.

---

## 2. State-of-the-Art Context (Updated and Verified)

This section adds to Review 1 §3, with values confirmed during this review.

### 2.1 Pulley mechanics
- **Schweizer (2001)** measured bowstringing force *in vivo* in 16 fingers of 4 subjects. A2 load was about 3× fingertip force in the crimp grip, with A2 loads up to about **116 N**. The 3:1 ratio was therefore observed at fingertip forces well below 100 N. Applying it at 100–172 N per finger is an extrapolation.
- **Vigouroux et al. (2006)** used a 3D static model with EMG-informed constraints. They estimated A2 force about **36×** and A4 force about **4×** higher in crimp than in slope grip. FDP:FDS ratios were 1.75 (crimp) and 0.88 (slope). The DIP passive moment was taken as about ¼ of the external moment. These are *model estimates*, not EMG ratios. The study did not use ultrasound.
- **Roloff et al. (2006)**, *J Biomech* 39(5):915–923, extended Hume's pulley model with pulley stiffness and two tendons. Their sensitivity analysis found **pulley position relative to the PIP centre of rotation** to be the most influential parameter. The present model has no such parameter; it uses fixed sharing constants (0.50/0.40).
- **Lin et al. (1990)** is widely cited for an A2 breaking strength of about **407 N**. It came from elderly cadaveric specimens.
- **Schöffl I. et al. (2009)**, *J Biomech* 42(13):2124–2128, tested 39 cadaver fingers. Pulleys failed at **lower forces under eccentric** than concentric loading, and A2 rupture dominated (59%) in the eccentric condition.
- **Moor et al. (2009)**, *Clin Biomech* 24(1):20–25, found that tendon–pulley friction is roughly linear with load and peaks near 90° PIP flexion. This is a directly relevant source for μ_t that the manuscript cites with incorrect bibliographic data.

### 2.2 Adolescent injury
- **Primary periphyseal stress injury (PPSI)** of the middle-phalanx base is the most common overuse injury in adolescent climbers. In 2025, Schöffl and colleagues proposed a **five-grade classification with a/b subtypes** (sclerosis on CT), partly *because* Salter-Harris was considered unsuitable for these chronic injuries. The manuscript and guide use "Stages I–IV" and "Salter-Harris type II or III", which do not match this classification.

### 2.3 Climbing training evidence
- Systematic reviews and meta-analyses (e.g., Stien et al., 2023; Langer et al., 2023) and intervention studies (López-Rivera & González-Badillo, 2012, 2019; Devise et al., 2022) suggest that hangboard and climbing-specific strength training improves finger strength. The manipulated variables were load, intensity, and work:rest structure. **No controlled study was found that prescribes training by finger length or phenotype.** The phenotype micro-cycles (Guide §4) therefore remain untested hypotheses.
- Tendon and pulley tissue are generally reported to adapt more slowly than muscle. Specific time constants for annular pulleys (e.g., the "12–24 months" in §6.3 and Guide Rec 3) were not found in the literature.

### 2.4 Robotics and tendon-driven mechanisms (relevant to N1)
- In tendon-driven robotic fingers, guide (pulley) reaction forces are usually computed **in the frame of the guide**. Out-of-plane guide loading arises only when the tendon path deviates from the plane of the guide, e.g., misalignment between successive routing points (cf. Khatik et al., arXiv:2010.02580; MuJoCable directional Capstan, arXiv:2609.09612). The current model rotates the whole flexion plane with the MCP abduction, so the tendon never leaves the plane of A2/A4. This explains the zero local transverse load (N1).

---

## 3. Detailed Audit of Category A Items

### A1 — Moment-arm scaling: *Partially fixed*
**Fixed:**
- `moment_arms()` scales all moment arms by `geom.scale_factor` when `scale_moment_arms_with_geometry=True` (default) ([L416–421](file:///Users/igorcerovsky/Documents/finger/climbing_finger_3d.py#L416-L421)).
- §4.3 and Table 3 now separate the isometric and unscaled cases, and give +15.6% (long vs. standard) for the unscaled case.

**Remaining:**
1. **Abstract** still states "up to 37.5% higher total tendon tension" and that long-fingered climbers "approach structural pulley failure limits at significantly lower body-weight percentages". Neither statement holds under the isometric baseline the paper now adopts (+0.9% to +2.3%).
2. §5.2, §6.3, and Guide Rec 3 / §3.3 / §4 build practical advice (fewer sessions, 48–72 h recovery, 12–16-week ramps, 75% open hand) **only on the unscaled case**. The paper gives no evidence that tendon moment arms fail to scale with phalanx length. Literature on hand anthropometry generally suggests that bone dimensions, and with them tendon–joint distances, covary with segment length. Presenting the unscaled case as one bound of a sensitivity range, not as the basis for recommendations, would be more consistent with the paper's own baseline.
3. Percentages disagree: 37.5% (Abstract, §6.9, Guide Rec 3 / §3.3 / §3.5), 37.2–38.1% (§4.3), 36.9% (Fig. 3B, Review 2).
4. Guide §3.0 says "internal tendon tension and pulley normal forces scale directly with skeletal phalanx lever lengths" and that the FDP moment arm "expands only minimally across ontogeny" (h_FDP ≈ 3.5–5.5 mm). This contradicts the isometric baseline. The quoted h_FDP is also smaller than the model's own PIP/MCP moment arms (8–13 mm), and no source is given.
5. §5.2 says the penalty "helps explain why long-fingered climbers experience a higher incidence of chronic A2 pulley tenosynovitis". No epidemiological source is cited, and none was found.

### A2 — Crossover and coupling law: *Partially fixed*
**Fixed:** §2.4 calls r_emg a "phenomenological load-partitioning function". §4.1 says the collapse at d/L_DP ≈ 1.52 is "an exact mathematical consequence of the load-partitioning model" and a "mechanistic hypothesis". This is a clear improvement.

**Remaining:**
1. **Wrong threshold.** §4.1 says the crossover happens "at f_DP ≈ 0.46". For half-crimp (r_base = 1.20), r_emg = 1 at f_DP = (1/1.2 − 0.2)/0.8 = **0.79**. The value 0.46 belongs to full crimp (r_base = 1.75).
2. **Circular validation remains.** §3, "Validation Analysis 1 — EMG Ratio Fidelity", still lists exact recovery of the imposed ratios as validation. Fig. 1C plots the imposed ratios under the label `exp_ratios` ([L99–100](file:///Users/igorcerovsky/Documents/finger/generate_publication_figures.py#L99-L100), comment "# 100% agreement").
3. **Abstract** still describes the crossover as a result ("triggering a pronounced shift in prime-mover recruitment"), with phenotype-specific depths and "long digits fail to reach crossover". §4.1 itself says the crossover sits at a constant d/L_DP. The "long digits fail" statement only reflects an arbitrary ≤ 35 mm window, and it conflicts with the 38.3 mm crossover reported two bullets earlier.
4. **Source of r_base.** §2.4 says r_base is "derived from in vivo surface EMG data (Vigouroux et al., 2006)". Vigouroux et al. report 1.75 and 0.88 as model-estimated tendon-force ratios. FDP and FDS are deep forearm muscles and are not separable by surface EMG. **The 1.20 half-crimp value still has no source.**
5. **Coupling law vs. passive DIP moment.** Vigouroux et al. handled DIP hyperextension through a passive moment (~¼ of the external moment), not through a fixed ratio. Offering that approach as an alternative would let FDP:FDS emerge from equilibrium and provide a non-circular comparison.
6. Review 2 describes the original formula as `r_base · f_DP^1.5`. The code and manuscript use `r_base · (0.20 + 0.80 f_DP)` ([L693](file:///Users/igorcerovsky/Documents/finger/climbing_finger_3d.py#L693)). Review 2 also says the manuscript calls the crossover "emergent", which contradicts the manuscript's (correct) "mathematical consequence" framing.

### A3 — Attribution and forward validation: *Not fixed*
1. **Reference #24** has fabricated co-authors and title (N5). Review 2's statement that the citation was "corrected … across all … reference lists" does not hold for the manuscript.
2. **Digit identity.** Review 1 described Synek et al. as an *index*-finger model. The dataset paths used in the code (`Geometry_Middle_Cal_Hum`) suggest a *middle*-finger geometry. Confirming the digit from the original paper is recommended, as it affects the "index vs. middle" limitation in Review 1 §4.4.
3. **No forward simulation.** Review 1 recommended applying the experimental tendon loads and predicting fingertip force magnitude and direction. This was not done. The inverse comparison remains, and its ratios have *worsened*: HyperExt 24.9–25.3 (was 15.9–16.3), Hook 5.1–5.5 (was 3.8–4.2).
4. **The "forward validation" in §3 item 4 is not a forward validation.** It compares the model's A2/fingertip ratio (~2.8–3.0) with Schweizer's in-vivo ratio. That is useful, but:
   - Schweizer's ratio was measured at A2 loads up to about 116 N, i.e., fingertip forces around 40 N, not 100 N.
   - Review 2 calls it a "crimp-to-slope ratio of 2.8:1 (282 N vs 100 N)". That mixes up A2/fingertip with crimp/slope. The model's crimp/open A2 ratio is ~8.5× (Table 2) or ~4.9× (current code), compared with ~36× in Vigouroux et al. (2006).
   - "Closely matching experimental in vivo ultrasound and EMG observations (Vigouroux et al., 2006)": Vigouroux et al. is a modelling study without ultrasound.
5. **Posture definitions differ.** HyperExt is "45° DIP / 50° PIP / −20° MCP" in the manuscript but "15° MCP hyperextension with neutral IP joints" in Review 2. MinorFlex is 35/55/40 in the manuscript and benchmark but 18/15/25 in Review 2.
6. **"Pred/App ratio" is not defined.** For MinorFlex 300 g, (FDP + FDS)/applied = 1.9/4.91 = 0.39, not the listed 0.43. Stating what is summed would make the column interpretable. It would also help to state which tendon(s) were loaded in each Synek trial, as comparing a summed prediction against a single applied tendon load may not be like-for-like.
7. The **Abstract** still says "experimentally validated against cadaveric force-plate measurements". Given errors from −57% to +2400% and no edge-contact validation, a more cautious phrasing would better match the evidence, e.g., "compared against …; agreement was limited to flexed postures".

### A4 — A2 consistency: *Partially fixed*
**Fixed:** A2 deflection now uses only the PIP angle when `a2_includes_mcp=False` (default; [L1168–1171](file:///Users/igorcerovsky/Documents/finger/climbing_finger_3d.py#L1168-L1171), [L1194–1197](file:///Users/igorcerovsky/Documents/finger/climbing_finger_3d.py#L1194-L1197)). Tables 2 and 3 are internally consistent (8.06 MPa × 60 mm² = 483.6 N).

**Remaining:**
1. **Half-crimp ≥ full crimp in A2 load** persists in the manuscript (Table 2: 8.58 vs. 8.06 MPa; Table 3: 299.7 vs. 281.6 N) and in the current code (303.0 vs. 303.9 N, practically equal). Review 2 says "Full Crimp produces the highest A2 normal force", and §6.1 recommends rationing the full crimp on A2 grounds. The tables do not support either statement. Literature generally reports greater pulley loading with more PIP flexion and DIP hyperextension.
2. **Open-hand A2.** Current code: 61.7 N at 100 N and 106.0 N at 171.7 N. That contradicts "below 100 N across typical athletic loads" (§6.1, Guide Rec 1) and the 33.2/57.1 N values in the text.
3. **Crimp/open ratio** (~5–8.5×) is much lower than the ~36× in Vigouroux et al. (2006). This is the most direct literature benchmark for the pulley model and is not discussed. One possible reason is that A2 load depends only on PIP angle through a fixed 0.50 factor, while Roloff et al. (2006) identified pulley position relative to the PIP centre as the dominant parameter.
4. **Sharing constants still unsourced.** The code comment attributes 0.50/0.40/0.10 to "Roloff et al. 2006, Vigouroux et al. 2006", but neither paper reports these fractions.
5. **"A2 stress (MPa)"** is total pulley force divided by a nominal 15 × 4 mm² tendon–pulley interface. It is a mean contact pressure, not the hoop or tensile stress in the pulley. The manuscript also calls it "fibrocartilage stress" and "hoop stress". Consistent naming is recommended.

### A5 — Thresholds and the "dynamic shock" reframing: *Partially fixed*
1. The 300 N line is removed. The 400 N value is attributed to elderly cadaveric data. Both changes are improvements.
2. **The reframing hides the calibration issue.** 171.7 N on one middle finger is now called a "dynamic catch / foot slip" load. For a 70 kg climber, assuming the middle finger carries roughly a quarter to a third of hand force, 171.7 N is about a **one-arm hang or a heavily weighted two-arm hang**. These are static loads that advanced climbers reach in controlled hangboard testing. The model predicts 484–522 N A2 load in that condition, above the ~400 N reference. The model therefore still implies that A2 rupture would be common in controlled maximal hangs. This was Review 1's Issue 5, and it remains.
3. "Acute pulley ruptures occur predominantly during unexpected dynamic foot slips rather than during controlled static hangs" is presented as explained by the model. The model is static, so this is a hypothesis consistent with Schöffl I. et al. (2009) and clinical reports, not a model result.
4. "Schöffl, Heid, Küpper (2009), *Sportverletz Sportschaden*" is cited for the cadaveric limit. The relevant cadaveric study is Schöffl I. et al. (2009), *J Biomech* 42(13):2124–2128 (§9).
5. §5.3 states that dynamic loads exceed static values "by 150–200% (Schweizer, 2001; Schöffl et al., 2003)". Neither source was found to report this figure.

---

## 4. Detailed Audit of Category B Items

### B6 — Sensitivity analysis: *Not fixed*
| Issue | Detail |
|---|---|
| Not global | Called "Morris Elementary Effects & OAT"; implemented as one-at-a-time low/high only. No interactions, no Morris trajectories, no Sobol indices. |
| Tautological `a2_share` | `f_a2_cr = F_A2 * (a2_share / 0.50)` is applied *after* the solve ([L48](file:///Users/igorcerovsky/Documents/finger/benchmarks/global_sensitivity_analysis.py#L48)). The reported "±20% → ±20%" follows by construction. |
| Inert `r_base_hc` | Accepted but never used. Outputs are identical at 1.0 and 1.5 (verified). The parameter that sets the crossover therefore appears to have zero influence. |
| Trivial `phi_deg` | Changes only F_A2,lat, which is a frame projection (N1). |
| Missing parameters | A4 share; MP-centroid weighting (0.60); contact-peak location; `wrist_pos`; FDP/FDS abduction moment arms; k_EDC; functional form of r_emg; moment-arm uncertainty (±1–2 mm); pulp constant k. Review 1 explicitly requested wrist vector, contact-peak location, and FDP:FDS assumptions. |
| Not reported | Neither the manuscript nor the guide mentions the analysis. |

The finding that μ_t and c_max change total force by < 3.5% is plausible and worth reporting. It shows low sensitivity to *those two* parameters, not overall model robustness.

### B7 — Evidence tiers: *Partially fixed*
- Tags exist for Guide Recs 1–8 only. Paper §6, Guide §2 (diagnostic table), §3 (age cohorts), and §4 (micro-cycles) have none, although these contain the least-supported claims.
- Some tags do not match their content:
  - Rec 3 is tagged "Model-Derived" but rests on the non-default unscaled case.
  - Rec 4 is tagged "Literature-Supported" but its numbers are model outputs, and the 400 N comparison is the calibration problem in A5.
  - Rec 6 is tagged "Clinical Observation", yet no clinical source supports "100% of falling body momentum … 20–60 ms".
- The guide still contains prescriptive and non-neutral wording: "must be suspected immediately", "hazardous cheating", "high-octane racing fuel", "Physics Meets the Climbing Wall". The paper §6.9 uses "mandates". The workspace guidelines recommend avoiding such language.

### B8 — PPSI and age section: *Partially fixed*
- **Improved:** PPSI is now the central adolescent concern, and pulley ruptures are described as rare in youth.
- **Remaining:**
  - "Stages I–IV" (paper §6.9, Guide §3.2, §3.5) does not match the 2025 five-grade (a/b) classification.
  - "Salter-Harris type II or III" (Guide §3.1) uses the scheme the PPSI authors considered unsuitable.
  - "+15% to +25% within a single 12–18 month period" (Guide §3.2, §3.5) is still present without a source. Review 1 flagged it, and Review 2 called the section fixed.
  - PPSI is placed in the "Kids < 12–13" cohort as a main failure mode. The literature associates it mainly with the pubertal growth spurt.
  - Masters OA claims (paper §6.9, Guide §3.4) are not model outputs (the model has no cartilage contact) and are untagged.
  - Reference #21 ("Schöffl, Lutter, Popp 2023, *WEM* 34(2):198–205") could not be confirmed in this review; please verify. The 2025 classification paper is not cited.

### B9 — Parameter harmonisation: *Fixed (scope-limited)*
- μ_t = 0.08, c_max = 2.0/1.5 mm, δ_max = 2.5 mm now agree between code ([L88–106](file:///Users/igorcerovsky/Documents/finger/climbing_finger_3d.py#L88-L106)) and manuscript.
- **Still drifting:**
  - The μ_t source is "Roloff et al. 2006" (manuscript) vs. "Schweizer 2003" (code comment).
  - The default crimp and half-crimp grips include **5° radial abduction** ([L214–215](file:///Users/igorcerovsky/Documents/finger/climbing_finger_3d.py#L214-L215)). The manuscript does not mention this, and §4.2 implies φ = 0° gives zero lateral load.
  - `FDP_DIP` is described as CT-calibrated (R² ≥ 0.99), but the code comment says the CT value (4.7 mm) was replaced by An et al. (1983) values because it was "incompatible with 3-DOF solver" ([L358](file:///Users/igorcerovsky/Documents/finger/climbing_finger_3d.py#L358)). That deviation is worth reporting, since it changes the DIP equilibrium and the FDP force directly.
  - Generating Tables 1–3 and the parameter table automatically from code is still recommended, given N3/N4.

---

## 5. Review of the MyoSuite / MyoHand Benchmark (C13)

The benchmark is a valuable start. In its current form it supports only the moment-arm comparison, and that only with caveats.

1. **Moment-arm signs are discarded.** All MyoHand values pass through `abs()` ([L135–137](file:///Users/igorcerovsky/Documents/finger/benchmarks/benchmark_myosuite.py#L135-L137)). EDC therefore appears as a "+9.4 mm" extensor at the MCP. Joint sign conventions and zero positions between the two models were not checked. Comparing signed values after mapping conventions is recommended.
2. **The muscle-force row is not a MyoSuite result.** "MyoHand FDP = 585 N, FDS = 0 N" comes from a hand-written sequential solve: FDP from DIP equilibrium, FDS from the residual PIP moment, MCP equilibrium ignored, no intrinsics ([L169–173](file:///Users/igorcerovsky/Documents/finger/benchmarks/benchmark_myosuite.py#L169-L173)). It is compared with the project's EMG-constrained solve, which includes RI and EDC. The two solutions differ in solver, constraints, and DOF set, so the 0.69 ratio is not interpretable as model disagreement. A like-for-like comparison would use the same solver and muscle set with each model's moment-arm matrix, or MyoSuite's own static optimisation.
3. **"MyoHand's DIP moment arm is roughly half the biological value."** The project's FDP_DIP is *not* CT-derived (§4, B9). By the code's own note, the CT-derived value (~4.7 mm) lies between MyoHand (1.6–3.8 mm) and the adopted 6.0 mm. Presenting the comparison as a range of plausible values, not as a deficiency of MyoHand, would be more defensible.
4. **"MyoSuite cannot compute pulley loads … confirming the unique scientific contribution."** MyoHand routes tendons through sites and wrapping geometry. Reaction forces at those routing points can in principle be computed from tendon tension and path direction, even though this is not a built-in output. Dedicated pulley-load models also exist (Schweizer, 2001; Roloff et al., 2006; Vigouroux et al., 2006; Khatik et al., 2021). Describing the contribution as "integration of edge-contact mechanics with pulley-load estimation" would be more accurate than "unique".
5. **Posture mismatch.** Review 2's posture table differs from the angles the benchmark actually passes (`c3d.GRIPS`, including 5° abduction).
6. **Not in the manuscript.** If kept, a short methods-and-results paragraph with signed moment arms is recommended.

---

## 6. New Methodological Issues (Not Raised in Reviews 1–2)

### 6.1 Out-of-plane pulley load is a frame artefact (N1, N2)
- `delta_PIP = u_PP − u_MP = R_MCP (e_x − R_flex(θ_PIP) e_x)` lies entirely in the local flexion plane of the proximal phalanx. Its component along the PP's own mediolateral axis is identically zero, whatever φ is.
- `F_A2_lat = |F_A2_vec[2]|` takes the **global** z-component, i.e., the component along the wall's horizontal axis. That reflects the finger's orientation relative to the wall, not loading across the pulley's width.
- Verification: global z = 0 / 36.4 / 109.5 N at φ = 0° / 5° / 15°, while PP-local z = 0.0000 N in all cases.
- The PIP "ML shear" is computed in the local frame, but it equals F·sin φ exactly. That is because the external force is assumed to stay wall-normal while the finger abducts. On real side-pulls the hold reaction tends to align with the pull direction, within the friction cone. This lateral load is therefore set by an assumed boundary condition.
- **Consequences:**
  - The "15° abduction → 61 N transverse shear" claim (Abstract, §4.2, §5.1, §6.2) is not supported. The figure shows ~110 N, which is equally an artefact.
  - So are the mechanistic explanation of side-pull pulley injury (§5.1(2)), the diagnostic threshold "F_lat > 60 N", and Guide Rec 2's physics paragraph. Rec 2's advice (align forearm and hold; avoid twisting) may still be reasonable on clinical and practical grounds, but it is not a model output.
- **Recommendation (robotics-inspired):** Model out-of-plane pulley loading through **tendon-path misalignment**. For example, route the flexor tendons from a forearm/carpal-tunnel point through A1 with the finger abducted relative to the tendon line of action, and resolve the reaction at A1/proximal A2 in the pulley's local frame. The legacy `a2_includes_mcp=True` path partly captured this and was removed under A4. Reporting loads in anatomical (local) frames throughout is recommended.

### 6.2 Intrinsic-muscle forces without capacity bounds (N6)
- RI is recruited at 79–135.6 N (current code; 100–171.7 N load) and 286.5 N (Table 2), and reaches 235 N at φ = 15°. Even at φ = 0° (crimp), RI = 85.5 N.
- The driver is the constant FDP/FDS abduction moment arms (−2.1/−1.5 mm, labelled "An 1983") combined with RI's 9.8 mm abduction moment arm. RI's sagittal extensor fractions then raise flexor demand.
- With the code's own PCSA_RI = 2.8 cm² and commonly used specific tensions of roughly 25–50 N/cm², RI capacity would be on the order of 70–140 N. The solver has no such bound, by design ("agnostic to PCSA").
- **Consequence:** total tendon force and, through the extensor fractions, flexor and A2 loads may be inflated by an unmodelled abduction balance. In-plane grips with a symmetric hold contact may not need large abduction moments at all.
- **Recommendation:** add force-capacity bounds (PCSA × specific tension, with a sensitivity range). Treat FDP/FDS abduction moment arms as posture-dependent and near zero in neutral abduction, or include them in the sensitivity analysis. Report the A-matrix condition number (Review 1 §4.5).

### 6.3 Unexplained extensor co-contraction
- The current code recruits **EDC at 109–214 N** in crimp and half-crimp. In half-crimp, DIP = +10°, so the exponential EDC floor is zero. The manuscript describes EDC as a ~3.7 N floor and reports 3.7/0.0 N in Table 2.
- Explaining what drives EDC recruitment would help. One possibility is interaction with RI/abduction equilibrium. Another is the least-squares objective. Large antagonist forces raise flexor and A2 loads directly.

### 6.4 Contact model items carried over from Review 1 (not addressed)
- "Hertzian contact pressure" remains in the objectives (§1, L37), and Johansson & Flanagan (2009) is still cited for the pressure profile.
- The MP centroid is still described as "60% load transfer into the fibrous sheath" via A3 anchoring. External loads reach the phalanx through skin and soft tissue. Annular pulleys restrain tendons and do not anchor external contact loads.
- The MP area expression x²/[2(L₃ + x)] still lacks a derivation.
- The peak-pressure location (tip vs. edge lip) was not tested. Together with the coupling law it governs f_DP, and with it the crossover.

### 6.5 Figure 2B zones contradict the plotted curves
- The "Half-Crimp Optimal (< 8 mm)" and "Transition Zone" shading is hard-coded ([L200–202](file:///Users/igorcerovsky/Documents/finger/generate_publication_figures.py#L200-L202)).
- The plotted totals show half-crimp **above** full crimp and open hand at every depth, and open hand lowest everywhere.
- §4.4's grip-transition boundaries are therefore not supported by the figure. Review 1 raised this, and it was not addressed. Geometric or friction feasibility constraints (e.g., whether an open hand can hold a 4 mm edge) would be needed to support a "half-crimp optimal" zone, and they are not modelled.

---

## 7. Remaining Issues in the Practical Training Guide

| Section | Issue | Suggested handling |
|---|---|---|
| Exec. summary | "frequently ignore the underlying laws of 3D musculoskeletal mechanics"; "reveals … fundamentally governed by" | Neutral phrasing; the evidence is model-based |
| Rec 1 | 33/57 N open-hand A2 and "below 100 N" not reproduced by current code (61.7/106 N); 80/20 split heuristic | Update numbers; label 80/20 as heuristic |
| Rec 2 | Physics rests on the frame artefact (§6.1); "hazardous cheating" | Re-tag as practitioner heuristic; neutral tone |
| Rec 3 | Built on the unscaled case; 37.5% figure; "12–24 months" unsourced | Recommend withholding or recasting as a hypothesis to test |
| Rec 4 | 171.7 N described as "dynamic", although it is within maximal static hang range | Discuss the calibration gap (A5) |
| Rec 5 | "Digit III forced into > 105° PIP flexion" on flat rungs; "Quadriga" for lumbrical injury | Mechanistically plausible; cite clinical lumbrical-injury literature |
| Rec 6 | "100% of falling body momentum … 20–60 ms" unsourced; therapeutic claims (contrast hydrotherapy "accelerates metabolic clearance") not evidenced | Label as hypothesis; remove or cite therapy claims |
| Rec 7 | "0.8·L_DP" is a reasonable, testable proposal | Keep as hypothesis; propose a validation study |
| Rec 8 | Temperature/RH windows (8–15 °C, 40–60% RH), μ drop 0.65 → 0.35 unsourced | Cite skin-friction literature or label as practitioner experience |
| §2 diagnostic table | Called "evidence-based clinical checklist"; actions ("reduce 50%", "10 days") and "F_lat > 60 N" unsourced | Re-title as illustrative; advise clinical assessment |
| §3 age cohorts | PPSI staging and Salter-Harris (B8); +15–25% growth; h_FDP claims; OA claims | Align with the 2025 PPSI classification; remove unsupported numbers |
| §4 micro-cycles | No intervention evidence for phenotype-specific programming | Mark as illustrative, not prescriptive |
| §5 primer | Calls the A2 equation "Capstan redirection"; it is a deflection-sharing heuristic with a Capstan multiplier | Clarify |

---

## 8. Prioritised List of Remaining Problems

**A. Essential before submission**
1. **Reproducibility (N3, N4).** Regenerate every table and figure from the current code with one script and one documented configuration (load, wall angle, abduction, contact). Remove hard-coded values (Fig. 1C, Fig. 2B zones, Fig. 3A annotation). Report the configuration behind each number.
2. **Out-of-plane loading (N1, N2).** Withdraw or re-derive the transverse pulley-load claims in local anatomical frames, with a tendon-path misalignment mechanism. Update Abstract, §4.2, §5.1, §6.2, Guide Rec 2, and §2.
3. **References (N5, §9).** Correct Synek et al. (2019) and the other mis-cited entries. Verify every remaining reference against the original.
4. **Abstract and Conclusions (N8).** Align them with the isometric baseline, the hypothesis status of the crossover, and the limited validation.
5. **Intrinsic and extensor forces (N6, §6.3).** Add muscle capacity bounds, revisit constant FDP/FDS abduction moment arms, disclose the default 5° abduction, and explain EDC recruitment.
6. **A2 benchmarks (A4, A5).** Compare quantitatively with Vigouroux et al. (2006; ~36× crimp/slope, ~4× A4) and Schweizer (2001; in-vivo range up to ~116 N). Discuss why predicted A2 loads exceed ~400 N at loads reached in controlled maximal hangs. Consider a PIP-centre-relative pulley-position parameter (Roloff et al., 2006).
7. **Validation (A3).** Run the forward simulation recommended in Review 1. Remove "EMG ratio fidelity" from validation. Define the Pred/App metric and the HyperExt posture consistently.

**B. Strongly recommended**
8. Replace the sensitivity script with a genuine global analysis (Morris, then Sobol on the influential subset) in which every parameter acts *inside* the solve, and report it in the manuscript (B6).
9. Fix the crossover threshold (f_DP ≈ 0.79 for half-crimp). Offer a passive-DIP-moment variant (Vigouroux et al., 2006) as a non-circular alternative (A2).
10. Apply evidence tiers to all practical content in both documents, and correct mismatched tags (B7).
11. Update the PPSI content to the 2025 five-grade classification. Remove the unsourced growth, physis-strength, and OA figures (B8).
12. Report the FDP_DIP deviation from the CT data and its effect on FDP force (B9).
13. Rework the MyoSuite benchmark as a like-for-like comparison with signed moment arms, or omit the force comparison (§5).

**C. Suggested empirical and robotics-informed extensions**
- An instrumented edge (pressure film or tactile array) to settle the peak location and f_DP(d/L_DP). This is the key input to the crossover.
- An anatomically faithful tendon-driven finger with load cells at A1/A2/A4, tested under controlled abduction and tendon-line misalignment, to measure real out-of-plane pulley loads.
- A cross-sectional study of finger length vs. force on fixed vs. 0.8·L_DP-normalised edges, to test Rec 7.

---

## 9. Reference Verification

| # in manuscript | As cited | Verified details | Status |
|---|---|---|---|
| 24 | Synek, Cegoñino, Ramakrishna, Pérez del Palomar (2019). *A subject-specific musculoskeletal model of the index finger…* PeerJ 7:e7470 | **Synek A, Lu S-C, Vereecke EE, Nauwelaerts S, Kivell TL, Pahr DH (2019).** *Musculoskeletal models of a human and bonobo finger: parameter identification and comparison to in vitro experiments.* PeerJ 7:e7470 | **Incorrect authors and title** |
| 18 | Roloff et al. (2006). *Biomechanical model for tendon friction in the flexor pulleys…* J Biomech 39(14):2683–2692 | **Roloff I, Schöffl VR, Vigouroux L, Quaine F (2006).** *Biomechanical model for the determination of the forces acting on the finger pulley system.* J Biomech 39(5):915–923 | **Incorrect title, issue, pages** |
| 16 | Moor et al. (2009). *Friction between tendon and pulleys…* J Hand Surg 34(7):1279–1284 | **Moor BK, Nagy L, Snedeker JG, Schweizer A (2009).** *Friction between finger flexor tendons and the pulley system in the crimp grip position.* Clin Biomech 24(1):20–25 | **Incorrect journal, title, pages** |
| 11 | King & Lien (2018). Hand Clin 34(3):329–335 | **King EA, Lien JR (2017).** *Flexor tendon pulley injuries in rock climbers.* Hand Clin 33(1):141–148 | **Incorrect year, volume, pages** |
| 3 | Bourne et al. (2011). *The effect of hold depth on finger forces and movement time…* Sports Biomech 10(2):114–124 | Bourne R, et al. (2011). *Measuring lifting forces in rock climbing: effect of hold size and fingertip structure.* J Appl Biomech | **Incorrect journal and title** (verify authors) |
| 20 | Schöffl, Heid, Küpper (2009). Sportverletz Sportschaden 23(1):38–45 (cited for cadaveric limits) | The cadaveric eccentric-loading study is **Schöffl I, Oppelt K, Jüngert J, Schweizer A, Bayer T, Neuhuber W, Schöffl V (2009).** *The influence of concentric and eccentric loading on the finger pulley system.* J Biomech 42(13):2124–2128 | **Likely wrong source** for this claim |
| 12 | Lin et al. (1990). J Hand Surg 15(3):429–434 | Lin GT, Cooney WP, Amadio PC, An KN (1990). *Mechanical properties of human pulleys.* J Hand Surg Br 15(4):429–434 | Minor (issue, volume series) |
| 21 | Schöffl, Lutter, Popp (2023). WEM 34(2):198–205 | Not confirmed; 2025 five-grade PPSI classification (Schöffl et al.) not cited | **Verify; add 2025 source** |
| 14 | Lutter et al. (2021). OJSM, "wrist kinematics" | Not confirmed (also flagged in Review 1) | **Verify** |
| 22 | Schweizer (2001). J Biomech 34(2):217–223 | Confirmed; in vivo, 16 fingers / 4 subjects, A2 ≈ 3× fingertip force, A2 up to ~116 N | OK; describe measurement range |
| 25 | Vigouroux et al. (2006). J Biomech 39(14):2583–2592 | Confirmed; model study, FDP:FDS 1.75/0.88, A2 ~36× and A4 ~4× crimp vs. slope | OK; correct "EMG/ultrasound" description |
| 19 | Schöffl et al. (2003). WEM 14(2):94–100 | Consistent with known record | OK |
| — | Missing | Schöffl I et al. (2009) J Biomech; Stien et al. (2023); López-Rivera & González-Badillo (2012/2019); 2025 PPSI classification; Xydas & Kao (1999) | Recommend adding |

> [!IMPORTANT]
> Bibliographic details above came from web searches during this review. Checking each one against the publisher record (DOI) before submission is recommended.

---

## 10. Concluding Assessment

The project is an open, well-structured static finger model. It brings together edge-contact geometry, pulley-load estimation, and phenotype scaling, and the first remediation round improved its transparency. However, several headline outputs either cannot be reproduced from the current code or arise from modelling artefacts: the out-of-plane pulley load and the PIP shear (frame and boundary-condition effects), the long-finger penalty (an unscaled-lever assumption), and the crossover depth (the imposed coupling law). Literature benchmarks also indicate that predicted A2 loads are too high relative to loads climbers sustain, while the crimp/slope contrast is too small. Addressing items A1–A7 in §8 would put the manuscript in a position where its quantitative claims are traceable to the code and consistent with the current finger-biomechanics and climbing-training literature. The practical guide would then benefit from being rebuilt on those corrected outputs, with each statement clearly tiered by its evidence base.
