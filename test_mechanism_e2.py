import numpy as np

# Athletes definition
athletes = [
    ("Climber_Short",   54.0, (21.0, 22.0, 23.0)),
    ("Climber_Average", 72.8, (26.0, 25.0, 26.0)),
    ("Climber_Long",    92.0, (31.0, 28.0, 29.0)),
]

def analyze_dip_fulcrum():
    # If the hold is large enough that the DIP joint sits perfectly on the hold edge
    # (or behind it), the FDP tendon no longer needs to generate torque to maintain 
    # the distal phalanx against the external force!
    # Instead, the DIP joint itself can act as a fulcrum directly supported by the hold,
    # and the PIP joint (and FDS tendon) takes over the primary supporting role.
    # We call this the transition to the "Open Drag" or a fully supported DIP.
    
    # What is the minimum hold depth required for the DIP joint to clear the edge
    # and rest directly on the hold surface?
    # It depends on the horizontal length of the distal phalanx from the wall.
    # In a half crimp (distal horizontal), horizontal length = Ld + r_pad_tip (approx).
    # Let's say the fleshy tip is pressed against the wall.
    print(f"{'Athlete':>15} {'Ld (Bone)':>10} {'Flesh +':>10} {'Req. Depth (DIP Support)':>25}")
    
    for name, mass, lengths in athletes:
        Ld = lengths[2]
        
        # Approximate flesh extension past the bone tip
        # We assume the pad radius at the tip is ~6mm for the short climber, scaling linearly.
        flesh_extension = 6.0 * (lengths[1] / 22.0)
        
        # If the finger is slightly angled (e.g. open drag), the calculation changes,
        # but for a standard horizontal crimp approach:
        req_depth = Ld + flesh_extension
        
        print(f"{name:>15} {Ld:10.1f} {flesh_extension:10.1f} {req_depth:25.1f} mm")

analyze_dip_fulcrum()
