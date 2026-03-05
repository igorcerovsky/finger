import numpy as np

def run_test():
    holds = [15, 10, 8, 6, 4]
    
    # Scale based on body length L (1.0 vs 1.15)
    # R scales linearly with finger length/girth
    R_short = 6.0  # mm
    R_long  = 8.0  # mm
    
    # Internal FDP moment arms
    r_fdp_short = 4.2
    r_fdp_long  = 5.3
    
    print("Micro-crimp Wall-Clearance Penalty")
    print(f"{'Hold(mm)':<10} {'Short DIP_MA':<15} {'Long DIP_MA':<15} {'Short FDP/F':<15} {'Long FDP/F':<15} {'Penalty_Factor':<15}")
    
    for D in holds:
        # Minimum DIP x-coordinate to clear the wall
        dip_x_short = R_short
        dip_x_long  = R_long
        
        # Center of pressure x-coordinate. It cannot exceed the hold depth D.
        # But if the finger is placed optimally, it's at min(dip_x, D). 
        # Actually, for maximum advantage, you put COP as close to DIP_x as possible.
        # But COP must be <= D. So COP_x = min(dip_x, D) -> wait, if hold is large, you can place COP exactly at dip_x.
        # If hold is smaller than dip_x, COP is forced to be at D.
        cop_short = min(dip_x_short, D)
        cop_long  = min(dip_x_long, D)
        
        ma_short = dip_x_short - cop_short
        ma_long  = dip_x_long - cop_long
        
        # Add a baseline minimum MA because bones aren't perfectly vertical 
        # (say 2mm baseline horizontal offset even in full crimp)
        ma_short += 2.0
        ma_long  += 2.5 # scales slightly
        
        fdp_short = ma_short / r_fdp_short
        fdp_long  = ma_long / r_fdp_long
        
        ratio = fdp_long / fdp_short
        
        print(f"{D:<10} {ma_short:<15.1f} {ma_long:<15.1f} {fdp_short:<15.2f} {fdp_long:<15.2f} {ratio:<15.2f}")

run_test()
