import sys
import os
import numpy as np

# Add parent directory to path to import our 3D model
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from climbing_finger_3d import Config as OurConfig, FingerGeometry, GripAngles, solve_all_methods, ContactGeometry

# Add current directory to path to import PeerJ model
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from Fingermodel import fullModel
from Fingermodel import extLoad

def cart2pol2D(cartArray):   
    mag = np.linalg.norm(cartArray)
    ang = np.arctan2(cartArray[1],cartArray[0])
    return np.array([ang,mag])

def compare_models():
    # Define a standard benchmark case: 100N horizontal force, Half-Crimp posture
    F_ext_mag = 100.0
    
    print("======================================================")
    print("   MODEL COMPARISON: 3D Climbing vs PeerJ Human       ")
    print("======================================================\n")
    
    # --------------------------------------------------------
    # 1. Run Our 3D Model
    # --------------------------------------------------------
    print("--- 1. Running Our 3D Model ('climbing_finger_3d.py') ---")
    
    # We use our standard geometry scaling (which incorporates long finger disadvantage scaling)
    # The default lengths: PP=45, MP=28, DP=22
    our_geom = FingerGeometry(L1=45.0, L2=28.0, L3=22.0, name="Standard")
    
    # Half-crimp: MCP: 15, PIP: 90, DIP: 10
    our_grip = GripAngles("Half-Crimp", theta_MCP=15.0, phi_MCP=0.0, theta_PIP=90.0, theta_DIP=10.0, emg_ratio=1.20)
    
    # Contact is at d_hold=10mm, horizontal wall
    contact = ContactGeometry(d_hold=10.0, beta_wall=0.0)
    F_ext_vector_our = np.array([F_ext_mag, 0.0, 0.0]) # Pure horizontal pull towards wall
    
    # Solve using EMG-constrained method (our most realistic) and the direct method
    our_res_all = solve_all_methods(our_grip, our_geom, F_ext_vector_our, contact=None)
    our_res_emg = our_res_all['emg']
    our_res_direct = our_res_all['direct']
    
    print(f"-> Our Model (EMG Constrained) - FDP: {our_res_emg['F_FDP']:.1f} N, FDS: {our_res_emg['F_FDS']:.1f} N")
    print(f"-> Our Model (Direct 3x3)      - FDP: {our_res_direct['F_FDP']:.1f} N, FDS: {our_res_direct['F_FDS']:.1f} N\n")
    
    # --------------------------------------------------------
    # 2. Run PeerJ Model
    # --------------------------------------------------------
    print("--- 2. Running PeerJ Model ('Fingermodel.py') ---")
    O2O3 = 0.02363 # Scaling param
    geomPath = os.path.join(os.path.dirname(__file__), 'Geometry_Middle_Cal_Hum/')
    
    DIPPoints = np.loadtxt(geomPath+'DIP_path.csv',delimiter=',',skiprows=1)
    PIPPoints = np.loadtxt(geomPath+'PIP_path.csv',delimiter=',',skiprows=1)
    MCPPoints = np.loadtxt(geomPath+'MCP_path.csv',delimiter=',',skiprows=1)
    
    segRatios = np.array([0.015/O2O3, 0.17 , 0.22, 1.62, 0.37])
    
    # PCSA Data
    PCSAEDC = 1.7 
    PCSAFDS = 4.2 
    PCSAFDP = 4.1 
    PCSALU = 0.2
    PCSAUI = 2.2 
    PCSARI = 2.8 
    specTension = 45.0
    
    EMFractions = np.loadtxt(geomPath+'EM_CS_fractions.csv',delimiter=',',skiprows=1)
    RI_PP, UI_PP = EMFractions[4], EMFractions[5]
    RI_ES, LU_ES = EMFractions[0], EMFractions[1]
    UI_ES, LE_ES = EMFractions[2], EMFractions[3]
    
    ES_Ratios = np.zeros(10)
    ES_Ratios[[0,1,2,5]] = np.array([RI_PP*RI_ES,LU_ES,UI_PP*UI_ES,LE_ES])
    RB_Ratios = np.zeros(10)
    RB_Ratios[[0,1,5]] = np.array([(RI_PP*(1-RI_ES)),(1-LU_ES),(1-LE_ES)/2.0])
    UB_Ratios = np.zeros(10)
    UB_Ratios[[2,5]] = np.array([(UI_PP*(1-UI_ES)),(1-LE_ES)/2.0])
    TE_Ratios = np.zeros(10)
    TE_Ratios[[8,9]] = np.array([1.0,1.0])    
    
    fingerModel = fullModel(O2O3)
    fingerModel.setTendonPaths(MCPPoints,PIPPoints,DIPPoints)
    fingerModel.setKinScalingPars(segRatios)
    fingerModel.setEERatios(ES_Ratios,RB_Ratios,UB_Ratios,TE_Ratios)
    fingerModel.setPCSA(PCSAEDC,PCSAFDS,PCSAFDP,PCSALU,PCSARI,PCSAUI)
    fingerModel.setSpecTension(specTension)
    
    p_ext = np.array([-(segRatios[0]*0.5+segRatios[1])*O2O3, 0, 0])
    
    angles = np.array([10.0, 90.0, 15.0, 0.0]) # Half-crimp posture map
    
    extLoad_tmp = extLoad()
    
    # Setting an inward force matching the experimental validation mapping 
    # the normal force acts upwards and inwards depending on posture, using -x here
    F_ext_peerj = np.array([-F_ext_mag, 0, 0]) 
    extLoad_tmp.addDPforce(F_ext_peerj, p_ext)
    
    # Use SLSQP static optimization with 'landsmeer' modeling
    F_mus_peerj = fingerModel.computeMuscleForces(
        angles[0], angles[1], angles[2], angles[3], 
        extLoad_tmp, landsmeer=True
    )
    peerj_FDP = F_mus_peerj[3]
    peerj_FDS = F_mus_peerj[4]
    
    print(f"-> PeerJ Model (Min Stress SQP) - FDP: {peerj_FDP:.1f} N, FDS: {peerj_FDS:.1f} N\n")
    
    print("------------------------------------------------------")
    print("   ANALYSIS OF DIFFERENCES                            ")
    print("------------------------------------------------------")
    print("1. Muscle Recruitment Paradigm:")
    print("   - Our 3D Model: Uses exact biological EMG sharing ratios (Vigouroux 2006).")
    print("   - PeerJ Model: Uses Static Optimization (minimizing squared muscle stress).")
    print("     This leads the PeerJ model to favor FDS strongly because its moment arms ")
    print("     at PIP and MCP are superior, mathematically avoiding FDP when possible ")
    print("     even though climbers intrinsically recruit FDP to stabilize the DIP joint.")
    print("\n2. Geometry & Extensor Mechanism:")
    print("   - Our 3D Model: Simplifies extensor mechanism to focus on primary flexors ")
    print("     with lateral friction and exact crimp contact point shifting.")
    print("   - PeerJ Model: A full hand model including intrinsic muscles (Lumbricals, ")
    print("     Interossei) and the entire Extensor Hood mechanism via routing fractions. ")
    print("     The external force causes co-contraction in these stabilizers, ")
    print("     altering the net flexor demand.")

if __name__ == '__main__':
    compare_models()
