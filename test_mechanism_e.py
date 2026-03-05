import numpy as np

athletes = [
    ("Climber_Short",   54.0, (21.0, 22.0, 23.0)),
    ("Climber_Average", 72.8, (26.0, 25.0, 26.0)),
    ("Climber_Long",    92.0, (31.0, 28.0, 29.0)),
]

def analyze_20mm():
    # The user hypothesizes that around 20mm, the hold edge sits at the DIP joint,
    # causing a biomechanical shift. 
    # Let's calculate the horizontal length of the distal phalanx from the wall.
    # In a half crimp, the distal phalanx is roughly at 0 degrees (horizontal) 
    # or slightly hyperextended.
    # Assuming the fingertip pad touches the back wall.
    
    print(f"{'Athlete':>15} {'Bone Ld':>10} {'Est Pad Rad':>12} {'Total Tip-to-DIP':>18}")
    for name, mass, lengths in athletes:
        Ld = lengths[2]
        
        # Estimate the soft tissue padding at the very tip (extends past the bone).
        # We know pad radius scales with Lm (from previous mechanisms).
        # Let's say the fleshy tip extends roughly the same as the pad radius ~ 6mm
        r_pad = 6.0 * (lengths[1] / 22.0)
        
        # The horizontal distance from the back wall (where the tip touches) 
        # to the DIP joint center is approximately Ld + soft_tissue.
        # It might be slightly less if the flex angle is >0, but near half crimp it's ~0.
        total_dist = Ld + r_pad / 2.0  # Just a rough estimate of tip flesh vs bone end
        
        print(f"{name:>15} {Ld:10.1f} {r_pad:12.1f} {total_dist:18.1f}")
        
analyze_20mm()
