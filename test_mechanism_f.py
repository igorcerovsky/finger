import numpy as np

def _friction_cone_analysis():
    # If the climber pulls with purely vertical force V = 100N.
    # To stay on a sloper, they MUST pull inward with horizontal force H.
    # To not slip, H must satisfy: H * (cos(th) + mu * sin(th)) >= V * (sin(th) - mu * cos(th))
    # Minimum H to prevent slip:
    
    mu = 0.5
    angles_deg = np.linspace(0, 45, 10)
    V = 100.0
    
    for th_deg in angles_deg:
        th = np.deg2rad(th_deg)
        
        # Min H required
        num = V * (np.sin(th) - mu * np.cos(th))
        den = np.cos(th) + mu * np.sin(th)
        min_H = max(num / den, 0.0)
        
        # Required Force Magnitude F
        F_req = np.sqrt(V**2 + min_H**2)
        
        # Now, how does the finger generate this force?
        # The finger applies force vector [-min_H, -V].
        print(f"Angle: {th_deg:4.1f} deg | Req F_in (H): {min_H:5.1f} N | Total F: {F_req:5.1f} N")

_friction_cone_analysis()
