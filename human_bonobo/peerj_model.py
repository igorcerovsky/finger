# -*- coding: utf-8 -*-
"""

SCRIPT FOR COMPARISON OF MODEL PREDICTIONS WITH EXPERIMENTAL DATA
IN COMBINED LOADING SCENARIO

"""

#==============================================================================
# IMPORT
#==============================================================================

from __future__ import division

import numpy
import matplotlib.pyplot as plt

from Fingermodel import fullModel
from Fingermodel import extLoad

plt.close('all')

#==============================================================================
# FUNC
#==============================================================================

def cart2pol2D(cartArray):
    # script to convert 2D cartesian to polar coordinates    
    mag = numpy.linalg.norm(cartArray,axis=1)
    ang = numpy.arctan2(cartArray[:,1],cartArray[:,0])
        
    return numpy.column_stack((ang,mag))

def pol2cart2D(polArray):
    # script to convert 2D polar to cartesian coordinates
    x = numpy.cos(polArray[:,0]) * polArray[:,1]
    y = numpy.sin(polArray[:,0]) * polArray[:,1]
    
    return numpy.column_stack((x,y))

def angleMeanStd(angleArray):
    # function that computes mean and std of angles
    # - takes array where each row represents a dataset
    # - general idea: convert angles to unit vectors and compute mean (angle)
    # - for std, we use circular standard deviation as defined by:
    #   std = sqrt(ln(1/norm(vec)^2)) = sqrt(-ln(norm(vec^2)))
    # https://en.wikipedia.org/wiki/Directional_statistics#The_fundamental_difference_between_linear_and_circular_statistics

    avrgAll = numpy.zeros(len(angleArray))
    stdAll  = numpy.zeros(len(angleArray))
    for  rowIdx,row in enumerate(angleArray):
        sinSum = numpy.sum(numpy.sin(row))
        cosSum = numpy.sum(numpy.cos(row))
        sinAvrg = sinSum / len(row)
        cosAvrg = cosSum / len(row)
    
        avrg = numpy.arctan2(sinAvrg,cosAvrg)
        std  = numpy.sqrt(-numpy.log(sinAvrg*sinAvrg+cosAvrg*cosAvrg))        
        
        avrgAll[rowIdx] = avrg
        stdAll[rowIdx]  = std

    return numpy.column_stack((avrgAll,stdAll))
        
def rotZ(vec,theta):
    # function that rotates 2D vector (x-y) around z-axis
    theta = theta/180.*numpy.pi
    R = numpy.array( [ [numpy.cos(theta),-numpy.sin(theta)],
                       [numpy.sin(theta), numpy.cos(theta)]   ] )
    return numpy.dot(R,vec)                

def rotZ3D(vec,theta):
    # function that rotates 3D vector (x-y) around z-axis
    theta = theta/180.*numpy.pi
    R = numpy.array( [ [numpy.cos(theta),-numpy.sin(theta),0],
                       [numpy.sin(theta), numpy.cos(theta),0],
                       [               0,                0,1]] )
    return numpy.dot(R,vec)  

#==============================================================================
# I/O
#==============================================================================

# **********************************
# EXPERIMENTAL DATA I/O
# **********************************

# set up paths to experimental data
basePathFtip = 'Experiments/Fingertip_forces/'
basePathFmc  = 'Experiments/Mc_forces/'

filesPathFtip = ['H01_fingertip_combined.csv','H02_fingertip_combined.csv','H03_fingertip_combined.csv']
filesPathFmc  = ['H01_mcforce_combined.csv','H02_mcforce_combined.csv','H03_mcforce_combined.csv']

tendonloadLabels = ['FDP=300g + FDS=300g','FDP=950g + FDS=950g']
numLoadcases = len(tendonloadLabels)

# define postures (i.e. joint angles) and corresponding names
# DIP - PIP - MCP_flex - MCP_abd
jointAngles = {}
jointAngles['MinorFlex'] = numpy.array([35,55,40,0])
jointAngles['MajorFlex'] = numpy.array([25,57,55,0])
jointAngles['HyperExt'] = numpy.array([45,50,-20,0])
jointAngles['Hook'] = numpy.array([50,65,0,0])
jointAnglesArray = numpy.array([  jointAngles['MinorFlex'],
                                jointAngles['MajorFlex'],
                                jointAngles['HyperExt'], 
                                jointAngles['Hook'] ])
                       
# set pulley efficiency (default: 100% = 1.000)
pulley_efficiency = 0.835

# set used stds for visualization in polar plots
forceStd = False
forceMagStd = 0.10 # force a standard deviation as the fraction X of the force magnitude for visualization
forceAngStd = 10.0 # force a standard deviation of X degrees of the force angle for visualization

# **********************************
# MODEL I/O AND PARAMETERS
# **********************************

# overall scaling paramater for model geometry
O2O3 = 0.02363

# set path to optimized model parameters
geomPath = 'Geometry_Middle_Cal_Hum/'

# load optimized tendonpathpoints 
DIPPoints = numpy.loadtxt(geomPath+'DIP_path.csv',delimiter=',',skiprows=1)
PIPPoints = numpy.loadtxt(geomPath+'PIP_path.csv',delimiter=',',skiprows=1)
MCPPoints = numpy.loadtxt(geomPath+'MCP_path.csv',delimiter=',',skiprows=1)

# set flag for landsmeer type 1 model for extensors
landsmeer_flag = True

# segment ratios - order: O0O1, O1O2, O3O4, O4O5, O5O6
segRatios = numpy.array([0.015/O2O3, 0.17 , 0.22, 1.62, 0.37])

# muscle PCSA in cm2 from Chao et al. 1989
PCSAEDC = 1.7 # i.e. EDC3
PCSAFDS = 4.2 # i.e. FDS3
PCSAFDP = 4.1 # i.e. FDP3
PCSALU = 0.2  # i.e. LU2
PCSAUI = 2.2  # i.e. DI3
PCSARI = 2.8  # i.e. DI2

# set specific muscle tension in N/cm2 from Holzbaur et al. 2005
specTension = 45.0

# load optimized extensor mechanism parameters
# for each part of the EM ( ES - RB - UB - TE), ratios are ordered:
#        RI - LU - UI - FDP - FDS - LE - ES - TE - RB - UB

EMFractions = numpy.loadtxt(geomPath+'EM_CS_fractions.csv',delimiter=',',skiprows=1)

# extensor mechanism ratios
RI_PP = EMFractions[4]
UI_PP = EMFractions[5]

RI_ES = EMFractions[0] 
LU_ES = EMFractions[1]
UI_ES = EMFractions[2] 
LE_ES = EMFractions[3] 

ES_Ratios = numpy.zeros(10)
ES_Ratios[[0,1,2,5]] = numpy.array([RI_PP*RI_ES,LU_ES,UI_PP*UI_ES,LE_ES])
RB_Ratios = numpy.zeros(10)
RB_Ratios[[0,1,5]] = numpy.array([(RI_PP*(1-RI_ES)),(1-LU_ES),(1-LE_ES)/2.0])
UB_Ratios = numpy.zeros(10)
UB_Ratios[[2,5]] = numpy.array([(UI_PP*(1-UI_ES)),(1-LE_ES)/2.0])
TE_Ratios = numpy.zeros(10)
TE_Ratios[[8,9]] = numpy.array([1.0,1.0])    

# set STL filenames
MC_filename = geomPath+'3mc_transformed.stl'
PP_filename = geomPath+'3proxph_transformed.stl'
MP_filename = geomPath+'3midph_transformed.stl'
DP_filename = geomPath+'3distph_transformed.stl'


# segment and point of load application
p_ext = numpy.array([ -(segRatios[0]*0.5+segRatios[1])*O2O3, 0, 0 ]) 

loadBone = 'DP'

#==============================================================================
# CALC
#==============================================================================

# **********************************
# EXPERIMENTAL DATA - FINGERTIP FORCES
# **********************************   

# read individual loads
tendonForces = {}
ftipForcesExp3D  = {}
ftipForcesExp2DPolar = {}
specNames   = []

for filePath in filesPathFtip:

    # create specimen name
    specName = filePath[:3]
    specNames.append(specName)
    
    # read fingertip forces
    alldata = numpy.loadtxt(basePathFtip+filePath,dtype=numpy.str,delimiter=',',skiprows=1)
    postureLables = alldata[:,0][0::2]
    tendonForces[specName] = alldata[:,1:7].astype(numpy.float) * 9.81 / 1000.0 * pulley_efficiency
    ftipForcesExp3D[specName]  = alldata[:,7:].astype(numpy.float)   
    
    # convert to polar (2D)
    ftipForcesExp2DPolar[specName] = cart2pol2D(ftipForcesExp3D[specName][:,:2])
    
# polar averages
ftipForcesExp2DPolarMagMat = numpy.empty([len(ftipForcesExp2DPolar[specNames[0]]),0])
ftipForcesExp2DPolarAngMat = numpy.empty([len(ftipForcesExp2DPolar[specNames[0]]),0])

# stack result columns (one for each specimen)
for specName in specNames:
    ftipForcesExp2DPolarMagMat = numpy.column_stack((ftipForcesExp2DPolarMagMat,
                                                    ftipForcesExp2DPolar[specName][:,1]))
    ftipForcesExp2DPolarAngMat = numpy.column_stack((ftipForcesExp2DPolarAngMat,
                                                    ftipForcesExp2DPolar[specName][:,0]))                                                        

# mean and std of angle (circular statistics)
ftipForcesExp2DPolarCircMeanAngle = angleMeanStd(ftipForcesExp2DPolarAngMat)[:,0]
ftipForcesExp2DPolarCircStdAngle  = angleMeanStd(ftipForcesExp2DPolarAngMat)[:,1]

ftipForcesExp2DPolarCircMeanMag   = ftipForcesExp2DPolarMagMat.mean(axis=1)
ftipForcesExp2DPolarCircStdMag    = ftipForcesExp2DPolarMagMat.std(axis=1)

# compute mean vectors
ftipForcesExp3DArithMean = numpy.zeros([len(ftipForcesExp3D[specNames[0]]),3])
for specName in specNames:
    ftipForcesExp3DArithMean = ftipForcesExp3DArithMean + ftipForcesExp3D[specName][:,:3]
ftipForcesExp3DArithMean = ftipForcesExp3DArithMean / len(specNames)
ftipForcesExp2DArithMean = ftipForcesExp3DArithMean[:,:2]
ftipForcesExp2DPolarArithMean = cart2pol2D(ftipForcesExp2DArithMean)

# ===> assign arithmetic means as target vectors
ftipForcesExpFinal2DPolar = ftipForcesExp2DPolarArithMean
ftipForcesExpFinal2DCart  = ftipForcesExp2DArithMean
ftipForcesExpFinal3DCart  = ftipForcesExp3DArithMean

# re-assign std deviations for visualization if desired
if forceStd == True:
    ftipForcesExp2DPolarCircStdAngle = numpy.ones(len(ftipForcesExp2DPolarCircStdAngle)) * numpy.deg2rad(forceAngStd)
    ftipForcesExp2DPolarCircStdMag   = ftipForcesExpFinal2DPolar[:,1] * forceMagStd

# **********************************
# EXPERIMENTAL DATA - MC BONE FORCES
# ********************************** 

# read individual loads
fmcForcesExp3D  = {}
fmcForcesExp2DPolar = {}
specNames   = []

for filePath in filesPathFmc:

    # create specimen name
    specName = filePath[:3]
    specNames.append(specName)
    
    # read fingertip forces
    alldata = numpy.loadtxt(basePathFmc+filePath,dtype=numpy.str,delimiter=',',skiprows=1)
    postureLables = alldata[:,0][0::2]
    fmcForcesExp3D[specName]  = - alldata[:,7:].astype(numpy.float)   
    
    # convert to polar (2D)
    fmcForcesExp2DPolar[specName] = cart2pol2D(fmcForcesExp3D[specName][:,:2])
    
# polar averages
fmcForcesExp2DPolarMagMat = numpy.empty([len(fmcForcesExp2DPolar[specNames[0]]),0])
fmcForcesExp2DPolarAngMat = numpy.empty([len(fmcForcesExp2DPolar[specNames[0]]),0])

# stack result columns (one for each specimen)
for specName in specNames:
    fmcForcesExp2DPolarMagMat = numpy.column_stack((fmcForcesExp2DPolarMagMat,
                                                    fmcForcesExp2DPolar[specName][:,1]))
    fmcForcesExp2DPolarAngMat = numpy.column_stack((fmcForcesExp2DPolarAngMat,
                                                    fmcForcesExp2DPolar[specName][:,0]))                                                        

# mean and std of angle (circular statistics)
fmcForcesExp2DPolarCircMeanAngle = angleMeanStd(fmcForcesExp2DPolarAngMat)[:,0]
fmcForcesExp2DPolarCircStdAngle  = angleMeanStd(fmcForcesExp2DPolarAngMat)[:,1]

fmcForcesExp2DPolarCircMeanMag   = fmcForcesExp2DPolarMagMat.mean(axis=1)
fmcForcesExp2DPolarCircStdMag    = fmcForcesExp2DPolarMagMat.std(axis=1)

# compute mean vectors
fmcForcesExp3DArithMean = numpy.zeros([len(fmcForcesExp3D[specNames[0]]),3])
for specName in specNames:
    fmcForcesExp3DArithMean = fmcForcesExp3DArithMean + fmcForcesExp3D[specName][:,:3]
fmcForcesExp3DArithMean = fmcForcesExp3DArithMean / len(specNames)
fmcForcesExp2DArithMean = fmcForcesExp3DArithMean[:,:2]
fmcForcesExp2DPolarArithMean = cart2pol2D(fmcForcesExp2DArithMean)

# ===> assign arithmetic means as target vectors
fmcForcesExpFinal2DPolar = fmcForcesExp2DPolarArithMean
fmcForcesExpFinal2DCart  = fmcForcesExp2DArithMean
fmcForcesExpFinal3DCart  = fmcForcesExp3DArithMean

# re-assign std deviations for visualization if desired
if forceStd == True:
    fmcForcesExp2DPolarCircStdAngle = numpy.ones(len(ftipForcesExp2DPolarCircStdAngle)) * numpy.deg2rad(forceAngStd)
    fmcForcesExp2DPolarCircStdMag   = fmcForcesExpFinal2DPolar[:,1] * forceMagStd

# **********************************
# GENERATE MODEL
# **********************************   
    
# full model
fingerModel = fullModel(O2O3)
fingerModel.setTendonPaths(MCPPoints,PIPPoints,DIPPoints)
fingerModel.setKinScalingPars(segRatios)
fingerModel.setEERatios(ES_Ratios,RB_Ratios,UB_Ratios,TE_Ratios)
fingerModel.setPCSA(PCSAEDC,PCSAFDS,PCSAFDP,PCSALU,PCSARI,PCSAUI)
fingerModel.setSpecTension(specTension)
fingerModel.setSTLFilename(MC_filename,PP_filename,MP_filename,DP_filename)

# **********************************
# COMPUTE FTIP AND Bone-to-bone FORCES
# ********************************** 

# init fingertip force vectors
ftipForcesPred4DInGlobal  = numpy.empty((0,4))
ftipForcesPred3DCartInDP = numpy.empty((0,3))
ftipForcesPred2DCartInDP = numpy.empty((0,2))
fjointmcpForcesPred3D = numpy.empty((0,3))
fjointallForcesPred3D = numpy.empty((3,3,0))
ftipForcesPred3DReactionInGlobal = numpy.empty((0,3))

modelExtloadAll = numpy.array([])
tendonForcesArray = tendonForces[specNames[0]]

for angleIdx, angles in enumerate(jointAnglesArray):

    for loadIdx,loadLabel in enumerate(tendonloadLabels): 

        # compute current index
        curIdx = angleIdx*numLoadcases+loadIdx
    
        # assign tendon loads of current trial
        tendonForces_tmp = tendonForcesArray[curIdx,:]
                    
        # compute transmission matrix T_mus and Jacobian J
        T_mus = fingerModel.computeForceTransmissionMatrix(angles[0],angles[1],angles[2],angles[3],
                                                              landsmeer=landsmeer_flag)
                                                            
        J     = fingerModel.computeJacobian(angles[0],angles[1],angles[2],angles[3],p_ext,loadBone)
                                             
        # extend Jacobian to account for z-axis moments at DP
        J_square_flex = numpy.row_stack((J,numpy.array([1.,1.,1.,0.])))
        
        # compute REACTIO fingertip force vector
        ftipForcesPred4DInGlobal_tmp = numpy.dot( numpy.linalg.inv(J_square_flex.T) , numpy.dot(T_mus,tendonForces_tmp) )
           
        # generate rotated vectors in sagittal plane (global to DP CS)
        sum_angles = numpy.sum(angles)
        ftipForcesPred3DCartInDP_tmp  = rotZ3D(ftipForcesPred4DInGlobal_tmp[:3],-sum_angles)        
        
        # compute REACTIO fingertip force vector for JRF
        ftipForcesPred3DReactionInGlobal_tmp = -ftipForcesPred4DInGlobal_tmp[:3]
        
        # compute joint reaction forces
        extLoad_tmp = extLoad()
        extLoad_tmp.addDPforce(ftipForcesPred3DReactionInGlobal_tmp,p_ext)
        fjointallForcesPred3D_tmp = fingerModel.computeJointReaction(angles[0],angles[1],angles[2],angles[3],extLoad_tmp,
                                                        use_F_mus=tendonForces_tmp,inGlobal=True,landsmeer=landsmeer_flag)
        fjointmcpForcesPred3D_tmp = fjointallForcesPred3D_tmp[2,:]
        
        # stack vectors    
        ftipForcesPred4DInGlobal = numpy.row_stack((ftipForcesPred4DInGlobal,ftipForcesPred4DInGlobal_tmp))
        ftipForcesPred3DCartInDP = numpy.row_stack((ftipForcesPred3DCartInDP,ftipForcesPred3DCartInDP_tmp))
        fjointmcpForcesPred3D = numpy.row_stack((fjointmcpForcesPred3D,fjointmcpForcesPred3D_tmp))
        fjointallForcesPred3D = numpy.concatenate((fjointallForcesPred3D,numpy.atleast_3d(fjointallForcesPred3D_tmp)),axis=2)
        modelExtloadAll = numpy.append(modelExtloadAll,extLoad_tmp)
        ftipForcesPred3DReactionInGlobal = numpy.row_stack((ftipForcesPred3DReactionInGlobal,ftipForcesPred3DReactionInGlobal_tmp))
  
ftipForcesPred2DPolarInDP   = cart2pol2D(ftipForcesPred3DCartInDP[:,:2])
fjointmcpForcesPred2DPolar  = cart2pol2D(fjointmcpForcesPred3D[:,:2])
 
# ===> assign final vectors for comparison
ftipForcesPredFinal2DPolar = ftipForcesPred2DPolarInDP
ftipForcesPredFinal2DCart  = ftipForcesPred3DCartInDP[:,:2]
ftipForcesPredFinal3DCart  = ftipForcesPred3DCartInDP
 
fjointmcpForcesPredFinal3DCart  = fjointmcpForcesPred3D
fjointmcpForcesPredFinal2DPolar = fjointmcpForcesPred2DPolar

# **********************************
# COMPUTE MC REACTION FORCES
# ********************************** 

fjointmcpForcesPred3D_check = numpy.empty((0,3))
fmcForcesPred3D = numpy.empty((0,3))
fmcmusdistForces3D = numpy.empty((0,3))
fmcmusproxForces3D = numpy.empty((0,3))

for angleIdx,angles in enumerate(jointAnglesArray):
    
    for loadIdx,loadLabel in enumerate(tendonloadLabels): 
    
        # compute current index
        curIdx = angleIdx*numLoadcases+loadIdx

        # assign tendon loads of current trial
        f_exp = tendonForcesArray[curIdx,:]  
        
        # set angles
        MCP_flex = angles[2]
        MCP_abd  = angles[3]

        # check if landsmeer assumption is applicable
        if landsmeer_flag == True:
            if MCP_flex <= 0:
                landsmeer_tmp = True
            else:
                landsmeer_tmp = False
                
        else:
            landsmeer_tmp = False
         
        # assumption:
        # proximal to the MCP joint, the tendon runs parallel to long axis of MC bone       
        F_MUS_MC_prox_FDP = numpy.array([1,0,0]) * f_exp[3]
        F_MUS_MC_prox_FDS = numpy.array([1,0,0]) * f_exp[4]
        
        # compute force vectors at MCP joint
        MCPPointsProx = fingerModel.MCPPathPoints[:,3:6]*fingerModel.O2O3
        MCPPointsDist = fingerModel.MCPPathPoints[:,0:3]*fingerModel.O2O3
        F_FDP_inO5 = fingerModel.computeFVec(MCPPointsProx[0,:],MCPPointsDist[0,:],numpy.array([fingerModel.O5O6,0,0]),\
                                                    numpy.array([0,fingerModel.deg2rad(MCP_abd),fingerModel.deg2rad(MCP_flex)]),f_exp[3],
                                                    landsmeer=landsmeer_tmp)
        F_FDS_inO5 = fingerModel.computeFVec(MCPPointsProx[1,:],MCPPointsDist[1,:],numpy.array([fingerModel.O5O6,0,0]),\
                                    numpy.array([0,fingerModel.deg2rad(MCP_abd),fingerModel.deg2rad(MCP_flex)]),f_exp[4],
                                    landsmeer=landsmeer_tmp) 
        R_O5_O6 = fingerModel.computeRotation([0,fingerModel.deg2rad(MCP_abd),fingerModel.deg2rad(MCP_flex)])
        F_FDP_inO6 = numpy.dot(R_O5_O6,F_FDP_inO5)
        F_FDS_inO6 = numpy.dot(R_O5_O6,F_FDS_inO5)
        
        F_MUS_MC_dist = -F_FDP_inO6 - F_FDS_inO6
        F_MUS_MC_prox = F_MUS_MC_prox_FDP + F_MUS_MC_prox_FDS
        
        F_EXT = ftipForcesPred3DReactionInGlobal[curIdx,:]
        fjointmcpForcesPred3D_check_tmp  = - (F_FDP_inO6 + F_FDS_inO6 + F_EXT)
        
        fmcForcesPred3D_tmp  = - (-fjointmcpForcesPred3D_check_tmp + F_MUS_MC_dist + F_MUS_MC_prox)
    
        # stack vectors
        fjointmcpForcesPred3D_check = numpy.row_stack((fjointmcpForcesPred3D_check,fjointmcpForcesPred3D_check_tmp))
        fmcForcesPred3D  = numpy.row_stack((fmcForcesPred3D,fmcForcesPred3D_tmp))
        fmcmusdistForces3D   = numpy.row_stack((fmcmusdistForces3D,F_MUS_MC_dist))
        fmcmusproxForces3D   = numpy.row_stack((fmcmusproxForces3D,F_MUS_MC_prox))
    
fmcForcesPred2DPolar = cart2pol2D(fmcForcesPred3D[:,:2])

# ===> assign final vectors
fmcForcesPredFinal3DCart  = fmcForcesPred3D
fmcForcesPredFinal2DCart  = fmcForcesPred3D[:,:2]
fmcForcesPredFinal2DPolar = fmcForcesPred2DPolar

# **********************************
# COMPARE MODEL TO EXPERIMENTS
# ********************************** 

# compute joint load and muscle force per fingertip force ratios in 3D
fmcForcesPredFinal3DMag = numpy.linalg.norm(fmcForcesPredFinal3DCart,axis=1)
fjointmcpForcesPredFinal3DMag = numpy.linalg.norm(fjointmcpForcesPredFinal3DCart,axis=1)
fmcForcesExpFinal3DMag  = numpy.linalg.norm(fmcForcesExpFinal3DCart, axis=1)
ftipForcesPredFinal3DMag = numpy.linalg.norm(ftipForcesPredFinal3DCart,axis=1)
ftipForcesExpFinal3DMag  = numpy.linalg.norm(ftipForcesExpFinal3DCart, axis=1)

fmc_ratio_exp_3D     = fmcForcesExpFinal3DMag / ftipForcesExpFinal3DMag
fmc_ratio_pred_3D    = fmcForcesPredFinal3DMag / ftipForcesPredFinal3DMag
fjoint_ratio_pred_3D = fjointmcpForcesPredFinal3DMag / ftipForcesPredFinal3DMag
fmus_ratio_exp_3D  = tendonForcesArray.sum(axis=1) / ftipForcesExpFinal3DMag
fmus_ratio_pred_3D = tendonForcesArray.sum(axis=1) / ftipForcesPredFinal3DMag

# convert directions from range [-180;+180[ to [0;360[
fmcForcesPredFinal2DAngleDeg = numpy.rad2deg(numpy.mod(fmcForcesPredFinal2DPolar[:,0] + numpy.pi * 2,numpy.pi*2) - numpy.pi)
fjointmcpForcesPredFinal2DAngleDeg = numpy.rad2deg(numpy.mod(fjointmcpForcesPredFinal2DPolar[:,0] + numpy.pi * 2,numpy.pi*2) - numpy.pi)
ftipForcesPredFinal2DAngleDeg = numpy.rad2deg(numpy.mod(ftipForcesPredFinal2DPolar[:,0] + numpy.pi * 2,numpy.pi*2) - numpy.pi)

fmcForcesExpFinal2DAngleDeg = numpy.rad2deg(numpy.mod(fmcForcesExpFinal2DPolar[:,0] + numpy.pi * 2,numpy.pi*2) - numpy.pi)
ftipForcesExpFinal2DAngleDeg = numpy.rad2deg(numpy.mod(ftipForcesExpFinal2DPolar[:,0] + numpy.pi * 2,numpy.pi*2) - numpy.pi)

# 2D absolute errors in terms of angles and magnitudes
ftip_ang_error_2D = numpy.abs(ftipForcesPredFinal2DAngleDeg - ftipForcesExpFinal2DAngleDeg)
ftip_mag_error_2D = numpy.abs(ftipForcesPredFinal2DPolar[:,1] - ftipForcesExpFinal2DPolar[:,1])
ftip_mag_error_2D_rel = ftip_mag_error_2D / ftipForcesExpFinal2DPolar[:,1]*100

fmc_ang_error_2D = numpy.abs(fmcForcesPredFinal2DAngleDeg - fmcForcesExpFinal2DAngleDeg)
fmc_mag_error_2D = numpy.abs(fmcForcesPredFinal2DPolar[:,1] - fmcForcesExpFinal2DPolar[:,1])
fmc_mag_error_2D_rel = fmc_mag_error_2D / fmcForcesExpFinal2DPolar[:,1]*100


#==============================================================================
# PRINT
#==============================================================================

print ' **************************************'
print ' ** RESULTS: ACCURACY'
print ' **************************************'

print '\n ** fingertip forces errors: ang in deg / mag in N (mag in %):'
for postureIdx,postureLabel in enumerate(postureLables):
    for loadIdx,loadLabel in enumerate(tendonloadLabels):
        curIdx = postureIdx*numLoadcases+loadIdx
        print '    -> %s, %s: %.2f / %.2f (%.2f)'%(postureLabel,loadLabel,
                                            ftip_ang_error_2D[curIdx],
                                            ftip_mag_error_2D[curIdx],ftip_mag_error_2D_rel[curIdx])
print '\n => overall mean: %.2f / %.2f (%.2f)\n'%(ftip_ang_error_2D.mean(),
                                            ftip_mag_error_2D.mean(),ftip_mag_error_2D_rel.mean())                                            
print ' ** mc bone forces errors: ang in deg / mag in N (mag in %):'
for postureIdx,postureLabel in enumerate(postureLables):
    for loadIdx,loadLabel in enumerate(tendonloadLabels):
        curIdx = postureIdx*numLoadcases+loadIdx
        print '    -> %s, %s: %.2f / %.2f (%.2f)'%(postureLabel,loadLabel,
                                            fmc_ang_error_2D[curIdx],
                                            fmc_mag_error_2D[curIdx],fmc_mag_error_2D_rel[curIdx])
print '\n => overall mean: %.2f / %.2f (%.2f)\n'%(fmc_ang_error_2D.mean(),
                                            fmc_mag_error_2D.mean(),fmc_mag_error_2D_rel.mean())

print ' **************************************'
print ' ** RESULTS: RELATIVE LOAD MAGNITUDES'
print ' **************************************'

print '\n ** fmus/ftip, fmc/ftip predicted:'
for postureIdx,postureLabel in enumerate(postureLables):
    for loadIdx,loadLabel in enumerate(tendonloadLabels):
        curIdx = postureIdx*numLoadcases+loadIdx
        print '    -> %s, %s: %.2f / %.2f '%(postureLabel,loadLabel,
                                            fmus_ratio_pred_3D[curIdx],fmc_ratio_pred_3D[curIdx])
print '\n => overall mean: %.2f / %.2f'%(fmus_ratio_pred_3D.mean(),fmc_ratio_pred_3D.mean())


#==============================================================================
# PLOT
#==============================================================================

# plot settings
cmap = plt.cm.Accent
cmaplist = [cmap(i) for i in numpy.linspace(0,1,numLoadcases)]

markers    = ['o','s','^','*']
linestyles = ['-.','--',':','-']

arr_width = 1.0
arr_hwidth = 6.0
arr_frac = 0.002
       
# **********************************
# PLOT OF COMBINED LOADING FINGERTIP FORCES
# ********************************** 
       
plt.figure(figsize=[15,15])
plt.suptitle('Fingertip forces of combined loading',fontsize=14,fontweight='bold')

maxMag = numpy.max(numpy.concatenate((ftipForcesPredFinal2DPolar[:,1],ftipForcesExpFinal2DPolar[:,1])))
ylim   = maxMag * 1.2

for postureIdx, posture in enumerate(postureLables):
    ax = plt.subplot(2,2,postureIdx+1,projection='polar')
    ax.set_title(posture)
    
    for loadIdx,loadLabel in enumerate(tendonloadLabels):    
    
        curIdx = postureIdx*numLoadcases+loadIdx
        
        # exp mean +- 1 std
        ax.bar( ftipForcesExpFinal2DPolar[curIdx,0]-ftipForcesExp2DPolarCircStdAngle[curIdx],
                ftipForcesExp2DPolarCircStdMag[curIdx]*2,
                width=ftipForcesExp2DPolarCircStdAngle[curIdx]*2,
                bottom=(ftipForcesExpFinal2DPolar[curIdx,1]-ftipForcesExp2DPolarCircStdMag[curIdx]),
                fill=True,fc=cmaplist[loadIdx],ec=cmaplist[loadIdx],linestyle='-',alpha=0.6) 
        
        # predicted mean
        plt.plot(ftipForcesPredFinal2DPolar[curIdx,0],ftipForcesPredFinal2DPolar[curIdx,1],color=cmaplist[loadIdx],
                         marker='o',markeredgecolor='None',markersize=5,zorder=3,label=loadLabel)
        plt.plot([0,ftipForcesPredFinal2DPolar[curIdx,0]],[0,ftipForcesPredFinal2DPolar[curIdx,1]],
                         linestyle='-',color=cmaplist[loadIdx])
                         
    plt.ylim([0,ylim])          
    plt.legend(loc='best')

       
# **********************************
# PLOT OF COMBINED LOADING MC BONE FORCES
# ********************************** 
       
plt.figure(figsize=[15,15])
plt.suptitle('MC bone forces of combined loading',fontsize=14,fontweight='bold')

maxMag = numpy.max(numpy.concatenate((fmcForcesPredFinal2DPolar[:,1],fmcForcesExpFinal2DPolar[:,1])))
ylim   = maxMag * 1.2

for postureIdx, posture in enumerate(postureLables):
    ax = plt.subplot(2,2,postureIdx+1,projection='polar')
    ax.set_title(posture)
    
    for loadIdx,loadLabel in enumerate(tendonloadLabels):    
    
        curIdx = postureIdx*numLoadcases+loadIdx
                
        # exp mean +- 1 std
        ax.bar( fmcForcesExpFinal2DPolar[curIdx,0]-fmcForcesExp2DPolarCircStdAngle[curIdx],
                fmcForcesExp2DPolarCircStdMag[curIdx]*2,
                width=fmcForcesExp2DPolarCircStdAngle[curIdx]*2,
                bottom=(fmcForcesExpFinal2DPolar[curIdx,1]-fmcForcesExp2DPolarCircStdMag[curIdx]),
                fill=True,fc=cmaplist[loadIdx],ec=cmaplist[loadIdx],linestyle='-',alpha=0.6)         
        
        # predicted mean JOINT forces
        plt.plot(fjointmcpForcesPredFinal2DPolar[curIdx,0],fjointmcpForcesPredFinal2DPolar[curIdx,1],color=cmaplist[loadIdx],
                         marker='x',markerfacecolor='None',markeredgewidth=1,
                         markeredgecolor=cmaplist[loadIdx],markersize=5,zorder=3,label=loadLabel)
        plt.plot([0,fjointmcpForcesPredFinal2DPolar[curIdx,0]],[0,fjointmcpForcesPredFinal2DPolar[curIdx,1]],
                         linestyle='--',color=cmaplist[loadIdx])
                         
        # predicted mean MC BONE forces
        plt.plot(fmcForcesPredFinal2DPolar[curIdx,0],fmcForcesPredFinal2DPolar[curIdx,1],color=cmaplist[loadIdx],
                         marker='o',markerfacecolor=cmaplist[loadIdx],markeredgecolor='None',
                         markersize=5,zorder=3,label=loadLabel)
        plt.plot([0,fmcForcesPredFinal2DPolar[curIdx,0]],[0,fmcForcesPredFinal2DPolar[curIdx,1]],
                         linestyle='-',color=cmaplist[loadIdx])                         
    
    plt.ylim([0,ylim])        
    plt.legend(loc='best')
      
    