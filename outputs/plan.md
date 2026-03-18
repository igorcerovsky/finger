The landmark EMG climbing studies (including Vigouroux et al.) specifically compared the Crimp Grip (where all force is on the distal phalanx tip) to the Slope Grip (where the finger is open and force is distributed across the full skin padded length of the finger, similar to a jug).

Here are their exact findings which perfectly align with our math:

Crimp Grip Ratio: As we know, when force is on the DP tip, the FDP muscle has to work the hardest to prevent the DIP joint from collapsing. The measured EMG ratio is 1.20 : 1 (FDP pulls 20% harder than FDS).
Slope/Deep Grip Ratio: When the hold is deep and force distributes down onto the middle/proximal phalanges, the moment requirement at the DIP vanishes for the proximal load. The measured EMG ratio directly flips to 0.88 : 1.
This means that in biological reality, the FDS overtakes the FDP in force production on deep open-handed holds, exactly as our implementation plan predicts will happen when we offload the Distal Phalanx!

Deep Hold Biomechanics implementation Plan
Goal Description
The user wants to investigate what happens when a climbing hold is extremely deep (i.e. > 22mm, which is longer than the Distal Phalanx, DP). In this scenario, the skin contact area bridges across the DIP joint onto the Middle Phalanx (MP). We need to answer:

How to mathematically simulate a distributed hold?
How do the FDP / FDS tendon forces change?
How does the body balance this contact area?
Biomechanical Theory
When the hold depth d_hold exceeds L_DP (e.g. 35mm), the external contact force $\vec{F}_{ext}$ is no longer a single point load on the DP. Instead, the force is distributed across the skin of both the DP and the MP. Because the MP is proximal to the DIP joint, any external force applied directly to the MP does not generate a flexion moment at the DIP joint.

Since the only muscle capable of generating a flexion moment at the DIP joint is the FDP (Flexor Digitorum Profundus), reducing the external force on the DP linearly reduces the required FDP force. Simultaneously, the force on the MP still generates moments at the PIP and MCP joints. The FDS (Flexor Digitorum Superficialis) crosses the PIP and MCP but NOT the DIP. Thus, a deep hold allows the climber to heavily recruit the FDS while "offloading" the easily-injured FDP tendon.

Proposed Changes
climbing_finger_3d.py
[MODIFY] climbing_finger_3d.py
Remove the DP-clamping constraint: Currently, d_eff = np.clip(d_hold + ..., 0.0, geom.L3 - 0.5). We must allow d_hold to exceed L3.
Implement Distributive Contact Modeling:
If d_eff <= geom.L3: 100% of the force is applied to the DP linearly.
If d_eff > geom.L3: The force splits into $F_{DP}$ and $F_{MP}$ based on the contact area.
Assuming generic uniform pressure, the fraction of force on DP vs MP can be roughly proportional to the engaged length on each phalanx:
Engaged MP length = d_eff - geom.L3
Fraction on DP: L_DP_engaged / total_engaged
Fraction on MP: L_MP_engaged / total_engaged
Calculate Multiple External Moments:
p_C_DP: centroid of contact on the DP.
p_C_MP: centroid of contact on the MP.
Compute M_DIP = (p_C_DP - p_DIP) x F_DP (NOTICE: $F_{MP}$ is excluded because it's proximal to DIP)
Compute M_PIP = (p_C_DP - p_PIP) x F_DP + (p_C_MP - p_PIP) x F_MP
Compute M_MCP = (p_C_DP - p_MCP) x F_DP + (p_C_MP - p_MCP) x F_MP
Verification Plan
Create a script or extend the plotting logic to sweep d_hold from 5mm (micro-crimp) up to 40mm (deep open hand/jug).
Observe the predicted FDP vs FDS crossing point. As d_hold exceeds ~22mm, FDP should plummet and FDS should spike.
Compare the total flexor force required to see the gross mechanical advantage of deep holds.
