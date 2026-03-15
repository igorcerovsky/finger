# -*- coding: utf-8 -*-
"""

SCRIPT FOR PARAMTER IDENTIFICATION OF THE HUMAN MODEL

"""

#==============================================================================
# IMPORT
#==============================================================================

from __future__ import division

import numpy
import matplotlib.pyplot as plt
import copy
from scipy import optimize

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
    
def computeFtipStar(fingerModel,jointAngle,f_exp,p_ext,loadBone,landsmeer=False):
    # function to compute F_tip*, i.e. spatial force and moment vector  
    
    # compute transmission matrix T_mus and Jacobian J
    T_mus = fingerModel.computeForceTransmissionMatrix(jointAngle[0],jointAngle[1],jointAngle[2],jointAngle[3],
                                                          landsmeer=landsmeer)
                                                        
    J     = fingerModel.computeJacobian(jointAngle[0],jointAngle[1],jointAngle[2],jointAngle[3],p_ext,loadBone)
                                         
    # extend Jacobian to account for z-axis moments at DP (abduction angle=0)
    J_square_flex = numpy.row_stack((J,numpy.array([1.,1.,1.,0.])))
    
    # compute fingertip force vector
    ftip_star_tmp = numpy.dot( numpy.linalg.inv(J_square_flex.T) , numpy.dot(T_mus,f_exp) )
    
    return ftip_star_tmp

def cost_FDS(x,y,fingerModel,jointAngles,f_exp,p_ext,loadBone,landsmeer,x0,w=numpy.array([1,0]),returnEstimate=False):
    # cost function for optimization of FDS tenton path optimization
    
    # update via points of fingerModel
    fingerModel.PIPPathPoints[3,[0,1,3,4]] = x[:4]
    fingerModel.MCPPathPoints[1,[0,1,3,4]] = x[4:]
    
    # compute ftip_star vector with updated model
    ftip_star_tmp_sag_rot_all = numpy.zeros([jointAngles.shape[0],2])
    for postureIdx, jointAngle in enumerate(jointAngles):
        # compute fingertip force vector
        ftip_star_tmp = computeFtipStar(fingerModel,jointAngle,f_exp,p_ext,loadBone,landsmeer=landsmeer)            
                                              
        # generate rotated vectors in sagittal plane (global to DP CS)
        sum_angles = numpy.sum(jointAngle)
        ftip_star_tmp_sag_rot = rotZ(ftip_star_tmp[:2],-sum_angles)    
        
        # save only rotated x-y ftip forces
        ftip_star_tmp_sag_rot_all[postureIdx,:] = ftip_star_tmp_sag_rot[:2]
    
    # flatten all predicted ftip forces to obtain 8x1 vector
    y_est = ftip_star_tmp_sag_rot_all.flatten()
    
    # compute SSE of forces
    cost_forces = numpy.sum((y - y_est)**2)
        
    # compute penalty of via-point shift for each joint
    cost_shift_PIP = numpy.sum((x0[:4]-x[:4])**2)
    cost_shift_MCP = numpy.sum((x0[4:]-x[4:])**2)
    
    # assamble weighted cost function
    cost = cost_forces * w[0] + cost_shift_PIP * w[2] + cost_shift_MCP * w[3]    
    
    if returnEstimate==True:
        return cost, y_est
    else:
        return cost
            
        
def cost_FDP(x,y,fingerModel,jointAngles,f_exp,p_ext,loadBone,landsmeer,x0,w=numpy.array([1,0]),returnEstimate=False):
    # cost function for optimization of FDP tenton path optimization    
    
    # update via points of fingerModel
    fingerModel.DIPPathPoints[1,[0,1,3,4]] = x[:4]
    fingerModel.PIPPathPoints[0,[0,1,3,4]] = x[4:8]
    fingerModel.MCPPathPoints[0,[0,1,3,4]] = x[8:]
    
    # compute ftip_star vector with updated model
    ftip_star_tmp_sag_rot_all = numpy.zeros([jointAngles.shape[0],2])
    for postureIdx, jointAngle in enumerate(jointAngles):
        # compute
        ftip_star_tmp = computeFtipStar(fingerModel,jointAngle,f_exp,p_ext,loadBone,landsmeer=landsmeer)            
                                              
        # generate rotated vectors in sagittal plane (global to DP CS)
        sum_angles = numpy.sum(jointAngle)
        ftip_star_tmp_sag_rot = rotZ(ftip_star_tmp[:2],-sum_angles)    
        
        # save only rotated x-y ftip forces
        ftip_star_tmp_sag_rot_all[postureIdx,:] = ftip_star_tmp_sag_rot[:2]
    
    # flatten all predicted ftip forces to obtain 12x1 vector
    y_est = ftip_star_tmp_sag_rot_all.flatten()
    
    # compute MSE of forces
    cost_forces = numpy.sum((y-y_est)**2)
    
    # compute MSE of via-point shift for each joint
    cost_shift_DIP = numpy.sum((x0[:4]-x[:4])**2)
    cost_shift_PIP = numpy.sum((x0[4:8]-x[4:8])**2)
    cost_shift_MCP = numpy.sum((x0[8:12]-x[8:12])**2)
    
    # assamble weighted cost function
    cost = cost_forces * w[0] + cost_shift_DIP * w[1] + cost_shift_PIP * w[2] + cost_shift_MCP * w[3] 
        
    if returnEstimate==True:
        return cost, y_est
    else:
        return cost                

def cost_EM(x,y,fingerModel,jointAngles,f_exp_all,p_ext,loadBone,landsmeer,x0,
            w=numpy.array([1,0,0,0]),em_mcp_fracs=numpy.array([1,1]),returnEstimate=False):
                
    # cost function for optimization of tendon paths involving the extensor mechanism
        
    # update via points of fingerModel
    fingerModel.DIPPathPoints[0,[0,1,3,4]] = x[:4]

    fingerModel.PIPPathPoints[4,[0,1,3,4]] = x[4:8]
    fingerModel.PIPPathPoints[1,[0,1,3,4]] = x[8:12]
    fingerModel.PIPPathPoints[2,[0,1,3,4]] = x[12:16]
    
    fingerModel.MCPPathPoints[2,[0,1,3,4]] = x[16:20]
    fingerModel.MCPPathPoints[3,[0,1,3,4]] = x[20:24]
    fingerModel.MCPPathPoints[4,[0,1,3,4]] = x[24:28]
    fingerModel.MCPPathPoints[5,[0,1,3,4]] = x[28:32]
    
    # update EM force transmission fractions    
    RI_ES = x[32]  
    LU_ES = x[33] 
    UI_ES = x[34]   
    LE_ES = x[35]  
    
    RI_PP = x[36]
    UI_PP = x[37]
    
    ES_Ratios = numpy.zeros(10)
    ES_Ratios[[0,1,2,5]] = numpy.array([RI_PP*RI_ES,LU_ES,UI_PP*UI_ES,LE_ES])
    RB_Ratios = numpy.zeros(10)
    RB_Ratios[[0,1,5]] = numpy.array([(RI_PP*(1-RI_ES)),(1-LU_ES),(1-LE_ES)/2.0])
    UB_Ratios = numpy.zeros(10)
    UB_Ratios[[2,5]] = numpy.array([(UI_PP*(1-UI_ES)),(1-LE_ES)/2.0])
    TE_Ratios = numpy.zeros(10)
    TE_Ratios[[8,9]] = numpy.array([1.0,1.0])  
    
    fingerModel.setEERatios(ES_Ratios,RB_Ratios,UB_Ratios,TE_Ratios)
    
    # compute ftip_star vector with updated model
    numMuscles  = f_exp_all.shape[0]
    numPostures = jointAngles.shape[0]
    ftip_star_tmp_sag_rot_all = numpy.zeros([numMuscles*numPostures,2])
    for muscleIdx, f_exp in enumerate(f_exp_all):
        
        for postureIdx, jointAngle in enumerate(jointAngles):
            
            # current index
            # curIdx = muscleIdx * numMuscles + postureIdx
            curIdx = muscleIdx * numPostures + postureIdx 
            
            # compute
            ftip_star_tmp = computeFtipStar(fingerModel,jointAngle,f_exp,p_ext,loadBone,landsmeer=landsmeer)            
                                                  
            # generate rotated vectors in sagittal plane (global to DP CS)
            sum_angles = numpy.sum(jointAngle)
            ftip_star_tmp_sag_rot = rotZ(ftip_star_tmp[:2],-sum_angles)    
            
            # save only rotated x-y ftip forces
            ftip_star_tmp_sag_rot_all[curIdx,:] = ftip_star_tmp_sag_rot[:2]
       
    # flatten all predicted ftip forces to obtain 32x1 vector
    y_est = ftip_star_tmp_sag_rot_all.flatten()
    
    # compute MSE of forces
    cost_forces = numpy.sum((y-y_est)**2)

    # compute MSE of via-point shift for each joint
    cost_shift_DIP = numpy.sum((x0[:4]-x[:4])**2)
    cost_shift_PIP = numpy.sum((x0[4:16]-x[4:16])**2)
    cost_shift_MCP = numpy.sum((x0[16:32]-x[16:32])**2)
    
    # assamble weighted cost function
    cost = cost_forces * w[0] + cost_shift_DIP * w[1] + cost_shift_PIP * w[2] + cost_shift_MCP * w[3]     
        
    if returnEstimate==True:
        return cost, y_est
    else:
        return cost     

#==============================================================================
# I/O
#==============================================================================

# **********************************
# SAVEDATA I/O
# **********************************

# directory to save optimized parameters
outdir = 'Geometry_Middle_Cal_Hum/'

# flag to save parameters
save = True

# **********************************
# EXPERIMENTAL DATA I/O
# **********************************

# set up paths to experimental data
basePath  = 'Experiments/Fingertip_forces/'
filesPath = ['H01_fingertip_summed.csv','H02_fingertip_summed.csv','H03_fingertip_summed.csv']

# define postures (i.e. joint angles) and corresponding names
# DIP - PIP - MCP_flex - MCP_abd
jointAngles = {}
jointAngles['MinorFlex'] = numpy.array([35,55,40,0])
jointAngles['MajorFlex'] = numpy.array([25,57,55,0])
jointAngles['HyperExt']  = numpy.array([45,50,-20,0])
jointAngles['Hook']      = numpy.array([50,65,0,0])

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
O2O3 = 0.02363 # 23.63 mm on average

# set path to initial model parameters
geomPath = 'Geometry_Middle_Uncal_Hum/'

# load tendonpathpoints [An et al. 1979]
DIPPoints = numpy.loadtxt(geomPath+'DIP_path.csv',delimiter=',',skiprows=1)
PIPPoints = numpy.loadtxt(geomPath+'PIP_path.csv',delimiter=',',skiprows=1)
MCPPoints = numpy.loadtxt(geomPath+'MCP_path.csv',delimiter=',',skiprows=1)

# adjustments / corrections
MCPPoints[4,4] =  -0.264   # UI Y-Prox original:  0.264 - error confirmed by Dr. Kai-nan An via email

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

# for each part of the EM ( ES - RB - UB - TE), ratios are ordered:
#        RI - LU - UI - FDP - FDS - LE - ES - TE - RB - UB

# wEM model extensor tendion ratios
RI_PP = 1.0
UI_PP = 1.0

RI_ES = RI_PP * 0.5   # range: [0,tPPRI]
LU_ES = 0.5           # range: [0,1.0]
UI_ES = UI_PP * 0.5   # range: [0,tPPUI]
LE_ES = 0.5           # range: [0,1.0]

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
# Note: point of load application is irrelevant for parameter identification!
p_ext = numpy.array([ -(segRatios[0]*0.5+segRatios[1])*O2O3+0.5, 0, 0 ]) 
loadBone = 'DP'

# set boundaries for optimization (max shift in mm)
DIPTol_x = 10
DIPTol_y = 10

PIPTol_x = 10
PIPTol_y = 10

MCPTol_x = 20
MCPTol_y = 20

# RI_ES, LU_ES, UI_ES, LE_ES, RI_PP, UI_PP
fractions_lb = numpy.array([0.0, 0.0, 0.0, 0.0, 0.5, 1.0])
fractions_ub = numpy.array([1.0, 1.0, 1.0, 1.0, 0.5, 1.0])

# weights:
#  w1 = for Squared Error (SE) of force
#  w2 = for SE of DIP points
#  w3 = for SE of PIP points
#  w4 = for SE of MCP points
weights_FDS = numpy.array([1,10,10,1])
weights_FDP = numpy.array([1,10,10,1])
weights_EM  = numpy.array([1,10,10,1])

#==============================================================================
# CALC
#==============================================================================

# **********************************
# EXPERIMENTAL DATA
# **********************************

# read individual loads
tendonLoads = {}
ftipForces  = {}
ftipForcesPolar = {}
specNames   = []

for filePath in filesPath:

    # create specimen name
    specName = filePath[:3]
    specNames.append(specName)
    
    # read fingertip forces
    alldata = numpy.loadtxt(basePath+filePath,dtype=numpy.str,delimiter=',',skiprows=1)
    postureLables = alldata[:,0][0::6]
    muscleLables  = alldata[:,1][:6]
    tendonLoads[specName] = alldata[:,2].astype(numpy.float) * 9.81 / 1000.0 * pulley_efficiency
    ftipForces[specName]  = alldata[:,3:].astype(numpy.float)
    
    # convert to polar (2D)
    ftipForcesPolar[specName] = cart2pol2D(ftipForces[specName][:,:2])
    
# polar averages
ftipForcesPolarAvrgMagMat = numpy.empty([len(ftipForcesPolar[specNames[0]]),0])
ftipForcesPolarAvrgAngMat = numpy.empty([len(ftipForcesPolar[specNames[0]]),0])
# stack result columns (one for each specimen)
for specName in specNames:
    ftipForcesPolarAvrgMagMat = numpy.column_stack((ftipForcesPolarAvrgMagMat,
                                                    ftipForcesPolar[specName][:,1]))
    ftipForcesPolarAvrgAngMat = numpy.column_stack((ftipForcesPolarAvrgAngMat,
                                                    ftipForcesPolar[specName][:,0]))                                                        

# mean and std of magnitude (simple mean)
ftipForcesPolarAvrgMag = numpy.mean(ftipForcesPolarAvrgMagMat,axis=1)
ftipForcesPolarStdMag = numpy.std(ftipForcesPolarAvrgMagMat,axis=1)

# mean and std of angle 
ftipForcesPolarAvrgAng = angleMeanStd(ftipForcesPolarAvrgAngMat)[:,0]
ftipForcesPolarStdAng  = angleMeanStd(ftipForcesPolarAvrgAngMat)[:,1]

# compute mean vectors
ftipForcesAvrg = numpy.zeros([len(ftipForces[specNames[0]]),2])
totmag = numpy.zeros(len(ftipForces[specNames[0]]))
for specName in specNames:
    ftipForcesAvrg = ftipForcesAvrg + ftipForces[specName][:,:2]   
ftipForcesAvrg = ftipForcesAvrg / len(specNames)

# assign arithmetic mean as target vectors
ftipForcesExpAvrg = ftipForcesAvrg
ftipForcesExpAvrgPolar = cart2pol2D(ftipForcesAvrg)

if forceStd == True:
    ftipForcesPolarStdAng = numpy.ones(len(ftipForcesPolarStdAng)) * numpy.deg2rad(forceAngStd)
    ftipForcesPolarStdMag = ftipForcesExpAvrgPolar[:,1] * forceMagStd    
    
# **********************************
# MODEL INITIALISATION
# **********************************

# number of finger postures
numLC = len(jointAnglesArray)

# compute upper muscle force bounds
F_mus_limit = numpy.array([PCSARI,PCSALU,PCSAUI,PCSAFDP,PCSAFDS,PCSAEDC])*specTension

# init model
fingerModel = fullModel(O2O3)
fingerModel.setTendonPaths(MCPPoints,PIPPoints,DIPPoints)
fingerModel.setKinScalingPars(segRatios)
fingerModel.setEERatios(ES_Ratios,RB_Ratios,UB_Ratios,TE_Ratios)
fingerModel.setPCSA(PCSAEDC,PCSAFDS,PCSAFDP,PCSALU,PCSARI,PCSAUI)
fingerModel.setSpecTension(specTension)
fingerModel.setSTLFilename(MC_filename,PP_filename,MP_filename,DP_filename)
    
# **********************************
# OPTIMIZATION
# **********************************   

print '\n************************************************'
print '*** BEGIN OPTIMIZATION: FDS'
print '************************************************'

# create copy of finger model to be modified in optimization
fingerModelTmp = copy.deepcopy(fingerModel)

# set up paraemter vector:
# x_0, x_1 = FDS_x, FDS_y @ PIP_dist
# x_2, x_3 = FDS_x, FDS_y @ PIP_prox
# x_4, x_5 = FDS_x, FDS_y @ MCP_dist
# x_6, x_7 = FDS_x, FDS_y @ MCP_prox

x0 = numpy.array([ PIPPoints[3,[0,1,3,4]] , MCPPoints[1,[0,1,3,4]] ]).flatten()

# set bounds (rel to O2O3)
dx = numpy.concatenate((numpy.tile([PIPTol_x,PIPTol_y],2),
                       numpy.tile([MCPTol_x,MCPTol_y],2)))/ 1000.0 / O2O3
x_lb = x0-dx
x_ub = x0+dx
x_b  = numpy.column_stack((x_lb,x_ub))
bounds = tuple(map(tuple,x_b))

# set up target fingertip force vector
# y_0, y_1 = F_x, F_y of FDS in Minor Flex
# y_2, y_3 = F_x, F_y of FDS in Major Flex
# y_4, y_5 = F_x, F_y of FDS in Hyper Extension
# y_6, y_7 = F_x, F_y of FDS in Hook
y0 = ftipForcesExpAvrg[0::6].flatten()

# set up experimental tendon load
f_exp = numpy.zeros(6)
f_exp[4] = tendonLoads[specNames[0]][0]

# set up weights for cost function (force / viapoint shift)
weights = weights_FDS

# set up args
args = (y0,fingerModelTmp, jointAnglesArray, f_exp, p_ext, loadBone, landsmeer_flag, x0, weights )

# init and run optimization
cost0,y_est0 = cost_FDS(x0,y0,fingerModelTmp, jointAnglesArray, f_exp, p_ext, loadBone, landsmeer_flag, x0, weights, returnEstimate=True)
xOpt = optimize.fmin_slsqp(cost_FDS,x0,args=args,bounds=bounds,iprint=2,acc=1e-8)

# evaluate new model output
costOpt,y_estOpt = cost_FDS(xOpt,y0,fingerModelTmp, jointAnglesArray, f_exp, p_ext, loadBone, landsmeer_flag, x0, weights,returnEstimate=True)
print '-> Cost before opt: ', cost0
print '-> Cost after opt:  ', costOpt

# assign FDS data
ftipForcesPred0_FDS   = y_est0.reshape(numLC,2)
ftipForcesPredOpt_FDS = y_estOpt.reshape(numLC,2)
x0_FDS   = copy.deepcopy(x0)
xOpt_FDS = copy.deepcopy(xOpt)
cost0_FDS   = copy.deepcopy(cost0)
costOpt_FDS = copy.deepcopy(costOpt)
x_b_FDS     = copy.deepcopy(x_b)
SE0_FDS   = (y0-y_est0)**2 
SEOpt_FDS = (y0-y_estOpt)**2 

ftipForcesPred0_FDS_pol   = cart2pol2D(ftipForcesPred0_FDS)
ftipForcesPredOpt_FDS_pol = cart2pol2D(ftipForcesPredOpt_FDS)

print '\n************************************************'
print '*** BEGIN OPTIMIZATION: FDP'
print '************************************************'

# create copy of finger model to be modified in optimization
fingerModelTmp = copy.deepcopy(fingerModel)

# set up paraemter vector:
# x_0, x_1  = FDS_x, FDS_y @ DIP_dist
# x_2, x_3  = FDS_x, FDS_y @ DIP_prox
# x_4, x_5  = FDS_x, FDS_y @ PIP_dist
# x_6, x_7  = FDS_x, FDS_y @ PIP_prox
# x_7, x_8  = FDS_y, FDS_y @ MCP_dist
# x_9, x_10 = FDS_x, FDS_y @ MCP_prox

x0 = numpy.array([ DIPPoints[1,[0,1,3,4]], PIPPoints[0,[0,1,3,4]] , MCPPoints[0,[0,1,3,4]] ]).flatten()

# set bounds (rel to O2O3)
dx = numpy.concatenate((numpy.tile([DIPTol_x,DIPTol_y],2),numpy.tile([PIPTol_x,PIPTol_y],2),
                        numpy.tile([MCPTol_x,MCPTol_y],2))) / 1000.0 / O2O3
x_lb = x0-dx
x_ub = x0+dx
x_b  = numpy.column_stack((x_lb,x_ub))
bounds = tuple(map(tuple,x_b))

# set up target fingertip force vector
# y_0, y_1 = F_x, F_y of FDP in Minor Flex
# y_2, y_3 = F_x, F_y of FDP in Major Flex
# y_4, y_5 = F_x, F_y of FDP in Hyper Extension
# y_6, y_7 = F_x, F_y of FDP in Hook
y0 = ftipForcesExpAvrg[1::6].flatten()

# set up experimental tendon load
f_exp = numpy.zeros(6)
f_exp[3] = tendonLoads[specNames[0]][1]

# set up weights for cost function (force / viapoint shift)
weights = weights_FDP

# set up args
args = (y0,fingerModelTmp, jointAnglesArray, f_exp, p_ext, loadBone, landsmeer_flag, x0, weights)

# init and run optimization
cost0,y_est0 = cost_FDP(x0,y0,fingerModelTmp, jointAnglesArray, f_exp, p_ext, loadBone, landsmeer_flag,x0, weights, returnEstimate=True)
xOpt = optimize.fmin_slsqp(cost_FDP,x0,args=args,bounds=bounds,iprint=2,acc=1e-8)

# evaluate new model output
costOpt,y_estOpt = cost_FDP(xOpt,y0,fingerModelTmp, jointAnglesArray, f_exp, p_ext, loadBone, landsmeer_flag,x0, weights,returnEstimate=True)
print '-> Cost before opt: ', cost0
print '-> Cost after opt:  ', costOpt

# assign FDP data
ftipForcesPred0_FDP   = y_est0.reshape(numLC,2)
ftipForcesPredOpt_FDP = y_estOpt.reshape(numLC,2)
x0_FDP    = copy.deepcopy(x0)
xOpt_FDP  = copy.deepcopy(xOpt)
cost0_FDP   = copy.deepcopy(cost0)
costOpt_FDP = copy.deepcopy(costOpt)
x_b_FDP     = copy.deepcopy(x_b)
SE0_FDP   = (y0-y_est0)**2 
SEOpt_FDP = (y0-y_estOpt)**2 

ftipForcesPred0_FDP_pol = cart2pol2D(ftipForcesPred0_FDP)
ftipForcesPredOpt_FDP_pol = cart2pol2D(ftipForcesPredOpt_FDP)

print '\n************************************************'
print '*** BEGIN OPTIMIZATION: EM (RI, UI, LU, LE)'
print '************************************************'

# create copy of finger model to be modified in optimization
fingerModelTmp = copy.deepcopy(fingerModel)

# set up paraemter vector:
# (base on rule: distal to proximal; and muscle order  RI - LU - UI - FDP - FDS - LE - ES - TE - RB - UB)
# x_0,  x_1   = TE_x, TE_y @ DIP_dist
# x_2,  x_3   = TE_x, TE_y @ DIP_prox

# x_4,  x_5   = ES_x, ES_y @ PIP_dist
# x_6,  x_7   = ES_x, ES_y @ PIP_prox
# x_8,  x_9   = RB_x, RB_y @ PIP_dist
# x_10, x_11  = RB_x, RB_y @ PIP_prox
# x_12, x_13  = UB_x, UB_y @ PIP_dist
# x_14, x_15  = UB_x, UB_y @ PIP_prox

# x_16, x_17  = RI_x, RI_y @ MCP_dist
# x_18, x_19  = RI_x, RI_y @ MCP_prox
# x_20, x_21  = LU_x, LU_y @ MCP_dist
# x_22, x_23  = LU_x, LU_y @ MCP_prox
# x_24, x_25  = UI_x, UI_y @ MCP_dist
# x_26, x_27  = UI_x, UI_y @ MCP_prox
# x_28, x_29  = LE_x, LE_y @ MCP_dist
# x_30, x_31  = LE_x, LE_y @ MCP_prox

# x_32, x_33, x_34, x_35 = RI_CS, LU_CS, UI_CS, EDC_CS
x0 = numpy.concatenate(( DIPPoints[0,[0,1,3,4]], # TE @ DIP
                   PIPPoints[4,[0,1,3,4]], # ES @ PIP
                   PIPPoints[1,[0,1,3,4]], # RB @ PIP
                   PIPPoints[2,[0,1,3,4]], # UB @ PIP
                   MCPPoints[2,[0,1,3,4]], # RI @ MCP
                   MCPPoints[3,[0,1,3,4]], # LU @ MCP
                   MCPPoints[4,[0,1,3,4]], # UI @ MCP
                   MCPPoints[5,[0,1,3,4]], # LE @ MCP
                   numpy.array([RI_ES,LU_ES,UI_ES,LE_ES,RI_PP,UI_PP]) ))  # extensor mechanism ES ratios

# set bounds (rel to O2O3)
dx = numpy.concatenate((numpy.tile([DIPTol_x,DIPTol_y],2),
                        numpy.tile([PIPTol_x,PIPTol_y],6),
                        numpy.tile([MCPTol_x,MCPTol_y],8),
                        numpy.zeros(6))) / 1000.0 / O2O3  # tol in mm


x_lb = x0-dx
x_ub = x0+dx

x_lb[-6:] = fractions_lb
x_ub[-6:] = fractions_ub

x_b  = numpy.column_stack((x_lb,x_ub))
bounds = tuple(map(tuple,x_b))


# set up target fingertip force vector
# y_0,  y_1  = F_x, F_y of RI in Minor Flex
# y_2,  y_3  = F_x, F_y of RI in Major Flex
# y_4,  y_5  = F_x, F_y of RI in Hyper Extension
# y_6,  y_7  = F_x, F_y of RI in Hook

# y_8,  y_9  = F_x, F_y of LU in Minor Flex
# y_10, y_11 = F_x, F_y of LU in Major Flex
# y_12, y_13 = F_x, F_y of LU in Hyper Extension
# y_14, y_15 = F_x, F_y of LU in Hook

# y_16, y_17 = F_x, F_y of UI in Minor Flex
# y_18, y_19 = F_x, F_y of UI in Major Flex
# y_20, y_21 = F_x, F_y of UI in Hyper Extension
# y_22, y_23 = F_x, F_y of UI in Hook

# y_24, y_25 = F_x, F_y of LE in Minor Flex
# y_26, y_27 = F_x, F_y of LE in Major Flex
# y_28, y_29 = F_x, F_y of LE in Hyper Extension
# y_30, y_31 = F_x, F_y of LE in Hook

y0 = numpy.array([ ftipForcesExpAvrg[4::6], # RI forces
                   ftipForcesExpAvrg[3::6], # LU forces
                   ftipForcesExpAvrg[5::6], # UI
                   ftipForcesExpAvrg[2::6]]).flatten() # LE

# set up experimental tendon load
# each row represents a single muscle experiment (RI - LU - UI - LE)
# muscle vector order: RI - LU - UI - FDP - FDS - LE
f_exp = numpy.zeros([4,6])
f_exp[0,0] = tendonLoads[specNames[0]][4]
f_exp[1,1] = tendonLoads[specNames[0]][3]
f_exp[2,2] = tendonLoads[specNames[0]][5]
f_exp[3,5] = tendonLoads[specNames[0]][2]

# set up weights for cost function (force / viapoint shift)
weights = weights_EM

# set up args
args = (y0,fingerModelTmp, jointAnglesArray, f_exp, p_ext, loadBone, landsmeer_flag, x0, weights)

# init and run optimization
cost0,y_est0 = cost_EM(x0,y0,fingerModelTmp, jointAnglesArray, f_exp, p_ext, loadBone, landsmeer_flag, x0, weights,returnEstimate=True)
xOpt = optimize.fmin_slsqp(cost_EM,x0,args=args,bounds=bounds,iprint=2,acc=1e-8)

# evaluate new model output
costOpt,y_estOpt = cost_EM(xOpt,y0,fingerModelTmp, jointAnglesArray, f_exp, p_ext, loadBone, landsmeer_flag, x0, weights,returnEstimate=True)
print '-> Cost before opt: ', cost0
print '-> Cost after opt:  ', costOpt

# assign FDP data
ftipForcesPred0_EM   = y_est0.reshape(numLC*4,2)
ftipForcesPredOpt_EM = y_estOpt.reshape(numLC*4,2)
x0_EM    = copy.deepcopy(x0)
xOpt_EM  = copy.deepcopy(xOpt)
cost0_EM   = copy.deepcopy(cost0)
costOpt_EM = copy.deepcopy(costOpt)
x_b_EM     = copy.deepcopy(x_b)
SE0_EM   = (y0-y_est0)**2 
SEOpt_EM = (y0-y_estOpt)**2 

ftipForcesPred0_EM_pol   = cart2pol2D(ftipForcesPred0_EM)
ftipForcesPredOpt_EM_pol = cart2pol2D(ftipForcesPredOpt_EM)

# **********************************
# SET UP OPTIMIZED MODEL
# **********************************

fingerModelOpt = copy.deepcopy(fingerModel)

# FDS coordinates update
fingerModelOpt.PIPPathPoints[3,[0,1,3,4]] = xOpt_FDS[:4]
fingerModelOpt.MCPPathPoints[1,[0,1,3,4]] = xOpt_FDS[4:]

# FDP coordinates update
fingerModelOpt.DIPPathPoints[1,[0,1,3,4]] = xOpt_FDP[:4]
fingerModelOpt.PIPPathPoints[0,[0,1,3,4]] = xOpt_FDP[4:8]
fingerModelOpt.MCPPathPoints[0,[0,1,3,4]] = xOpt_FDP[8:]

# EM coordinates update
fingerModelOpt.DIPPathPoints[0,[0,1,3,4]] = xOpt_EM[:4]

fingerModelOpt.PIPPathPoints[4,[0,1,3,4]] = xOpt_EM[4:8]
fingerModelOpt.PIPPathPoints[1,[0,1,3,4]] = xOpt_EM[8:12]
fingerModelOpt.PIPPathPoints[2,[0,1,3,4]] = xOpt_EM[12:16]

fingerModelOpt.MCPPathPoints[2,[0,1,3,4]] = xOpt_EM[16:20]
fingerModelOpt.MCPPathPoints[3,[0,1,3,4]] = xOpt_EM[20:24]
fingerModelOpt.MCPPathPoints[4,[0,1,3,4]] = xOpt_EM[24:28]
fingerModelOpt.MCPPathPoints[5,[0,1,3,4]] = xOpt_EM[28:32]

# EM transmission fractions update
RI_ES = xOpt_EM[32]  
LU_ES = xOpt_EM[33] 
UI_ES = xOpt_EM[34]   
LE_ES = xOpt_EM[35]  

RI_PP = xOpt_EM[36]
UI_PP = xOpt_EM[37]

ES_Ratios = numpy.zeros(10)
ES_Ratios[[0,1,2,5]] = numpy.array([RI_PP*RI_ES,LU_ES,UI_PP*UI_ES,LE_ES])
RB_Ratios = numpy.zeros(10)
RB_Ratios[[0,1,5]] = numpy.array([(RI_PP*(1-RI_ES)),(1-LU_ES),(1-LE_ES)/2.0])
UB_Ratios = numpy.zeros(10)
UB_Ratios[[2,5]] = numpy.array([(UI_PP*(1-UI_ES)),(1-LE_ES)/2.0])
TE_Ratios = numpy.zeros(10)
TE_Ratios[[8,9]] = numpy.array([1.0,1.0])  

fingerModelOpt.setEERatios(ES_Ratios,RB_Ratios,UB_Ratios,TE_Ratios)

# **********************************
# SET UP FORCE PREDICTION ARRAY
# **********************************

# num loadcases
numLC = len(jointAnglesArray)

ftipForcesPredOpt_all = numpy.zeros([numLC*6,2])

ftipForcesPredOpt_all[::6,:]  = ftipForcesPredOpt_FDS
ftipForcesPredOpt_all[1::6,:] = ftipForcesPredOpt_FDP
ftipForcesPredOpt_all[2::6,:] = ftipForcesPredOpt_EM[numLC*3:numLC*4,:] # EDC
ftipForcesPredOpt_all[3::6,:] = ftipForcesPredOpt_EM[numLC:numLC*2,:]   # LU
ftipForcesPredOpt_all[4::6,:] = ftipForcesPredOpt_EM[0:numLC,:]   # RI
ftipForcesPredOpt_all[5::6,:] = ftipForcesPredOpt_EM[numLC*2:numLC*3,:]  # RI

ftipForcesPred0_all = numpy.zeros([numLC*6,2])

ftipForcesPred0_all[::6,:]  = ftipForcesPred0_FDS
ftipForcesPred0_all[1::6,:] = ftipForcesPred0_FDP
ftipForcesPred0_all[2::6,:] = ftipForcesPred0_EM[numLC*3:numLC*4,:]  # EDC
ftipForcesPred0_all[3::6,:] = ftipForcesPred0_EM[numLC:numLC*2,:]    # LU
ftipForcesPred0_all[4::6,:] = ftipForcesPred0_EM[0:numLC,:]    # RI
ftipForcesPred0_all[5::6,:] = ftipForcesPred0_EM[numLC*2:numLC*3,:]   # RI

ftipForcesPredOpt_all_pol = cart2pol2D(ftipForcesPredOpt_all)
ftipForcesPred0_all_pol   = cart2pol2D(ftipForcesPred0_all)

# **********************************
# GENERAL COMPUTATIONS
# **********************************

# concatenate fingertip force vectors for optimized model
ftipForcesPredOpt_all = numpy.zeros([numLC*6,2])

ftipForcesPredOpt_all[::6,:]  = ftipForcesPredOpt_FDS
ftipForcesPredOpt_all[1::6,:] = ftipForcesPredOpt_FDP
ftipForcesPredOpt_all[2::6,:] = ftipForcesPredOpt_EM[numLC*3:numLC*4,:] # EDC
ftipForcesPredOpt_all[3::6,:] = ftipForcesPredOpt_EM[numLC:numLC*2,:]   # LU
ftipForcesPredOpt_all[4::6,:] = ftipForcesPredOpt_EM[0:numLC,:]   # RI
ftipForcesPredOpt_all[5::6,:] = ftipForcesPredOpt_EM[numLC*2:numLC*3,:]  # RI

# concatenate fingertip force vectors for initial model
ftipForcesPred0_all = numpy.zeros([numLC*6,2])

ftipForcesPred0_all[::6,:]  = ftipForcesPred0_FDS
ftipForcesPred0_all[1::6,:] = ftipForcesPred0_FDP
ftipForcesPred0_all[2::6,:] = ftipForcesPred0_EM[numLC*3:numLC*4,:]  # EDC
ftipForcesPred0_all[3::6,:] = ftipForcesPred0_EM[numLC:numLC*2,:]    # LU
ftipForcesPred0_all[4::6,:] = ftipForcesPred0_EM[0:numLC,:]    # RI
ftipForcesPred0_all[5::6,:] = ftipForcesPred0_EM[numLC*2:numLC*3,:]   # RI

ftipForcesPredOpt_all_pol = cart2pol2D(ftipForcesPredOpt_all)
ftipForcesPred0_all_pol   = cart2pol2D(ftipForcesPred0_all)

# **********************************
# GENERAL COMPUTATIONS
# **********************************

# fingertip force RMSEs
RMSE0_FDS   = numpy.sqrt(numpy.mean(SE0_FDS)) 
RMSEOpt_FDS = numpy.sqrt(numpy.mean(SEOpt_FDS)) 
RMSE0_FDS_rel = RMSE0_FDS/numpy.mean(ftipForcesExpAvrgPolar[::6,1])
RMSEOpt_FDS_rel = RMSEOpt_FDS/numpy.mean(ftipForcesExpAvrgPolar[::6,1])

RMSE0_FDP   = numpy.sqrt(numpy.mean(SE0_FDP)) 
RMSEOpt_FDP = numpy.sqrt(numpy.mean(SEOpt_FDP)) 
RMSE0_FDP_rel = RMSE0_FDP/numpy.mean(ftipForcesExpAvrgPolar[1::6,1])
RMSEOpt_FDP_rel = RMSEOpt_FDP/numpy.mean(ftipForcesExpAvrgPolar[1::6,1])

RMSE0_RI    = numpy.sqrt(numpy.mean(SE0_EM[:numLC*2]))
RMSEOpt_RI  = numpy.sqrt(numpy.mean(SEOpt_EM[:numLC*2]))
RMSE0_RI_rel = RMSE0_RI/numpy.mean(ftipForcesExpAvrgPolar[4::6,1])
RMSEOpt_RI_rel = RMSEOpt_RI/numpy.mean(ftipForcesExpAvrgPolar[4::6,1])

RMSE0_LU    = numpy.sqrt(numpy.mean(SE0_EM[numLC*2:numLC*4]))
RMSEOpt_LU  = numpy.sqrt(numpy.mean(SEOpt_EM[numLC*2:numLC*4]))
RMSE0_LU_rel = RMSE0_LU/numpy.mean(ftipForcesExpAvrgPolar[3::6,1])
RMSEOpt_LU_rel = RMSEOpt_LU/numpy.mean(ftipForcesExpAvrgPolar[3::6,1])

RMSE0_UI    = numpy.sqrt(numpy.mean(SE0_EM[numLC*4:numLC*6]))
RMSEOpt_UI  = numpy.sqrt(numpy.mean(SEOpt_EM[numLC*4:numLC*6]))
RMSE0_UI_rel = RMSE0_UI/numpy.mean(ftipForcesExpAvrgPolar[5::6,1])
RMSEOpt_UI_rel = RMSEOpt_UI/numpy.mean(ftipForcesExpAvrgPolar[5::6,1])

RMSE0_LE    = numpy.sqrt(numpy.mean(SE0_EM[numLC*6:numLC*8]))
RMSEOpt_LE  = numpy.sqrt(numpy.mean(SEOpt_EM[numLC*6:numLC*8]))
RMSE0_LE_rel = RMSE0_LE/numpy.mean(ftipForcesExpAvrgPolar[2::6,1])
RMSEOpt_LE_rel = RMSEOpt_LE/numpy.mean(ftipForcesExpAvrgPolar[2::6,1])

RMSE0_all   = numpy.sqrt(numpy.mean(numpy.concatenate((SE0_FDS,SE0_FDP,SE0_EM))))
RMSEOpt_all = numpy.sqrt(numpy.mean(numpy.concatenate((SEOpt_FDS,SEOpt_FDP,SEOpt_EM))))
RMSE0_all_rel = RMSE0_all/numpy.mean(ftipForcesExpAvrgPolar[:,1])
RMSEOpt_all_rel = RMSEOpt_all/numpy.mean(ftipForcesExpAvrgPolar[:,1])

# evaulate via-point shifts
RMSE_FDS_points = numpy.sqrt(numpy.mean(((xOpt_FDS-x0_FDS)*O2O3)**2))
RMSE_FDP_points = numpy.sqrt(numpy.mean(((xOpt_FDP-x0_FDP)*O2O3)**2))
RMSE_EM_points  = numpy.sqrt(numpy.mean(((xOpt_EM[:32]-x0_EM[:32])*O2O3)**2))
RMSE_all_points = numpy.sqrt(numpy.mean(numpy.concatenate((
                              ((xOpt_FDS-x0_FDS)*O2O3)**2,
                              ((xOpt_FDP-x0_FDP)*O2O3)**2,
                              ((xOpt_EM[:32]-x0_EM[:32])*O2O3)**2))))

MAE_FDS_points = numpy.mean(numpy.abs(((xOpt_FDS-x0_FDS)*O2O3)))
MAE_FDP_points = numpy.mean(numpy.abs(((xOpt_FDP-x0_FDP)*O2O3)))
MAE_EM_points  = numpy.mean(numpy.abs(((xOpt_EM[:32]-x0_EM[:32])*O2O3)))
MAE_all_points = numpy.mean(numpy.abs(numpy.concatenate((
                              ((xOpt_FDS-x0_FDS)*O2O3),
                              ((xOpt_FDP-x0_FDP)*O2O3),
                              ((xOpt_EM[:32]-x0_EM[:32])*O2O3)))))

shift_all = numpy.abs(numpy.concatenate((
                              ((xOpt_FDS-x0_FDS)*O2O3),
                              ((xOpt_FDP-x0_FDP)*O2O3),
                              ((xOpt_EM[:32]-x0_EM[:32])*O2O3))))

#==============================================================================
# PLOT
#==============================================================================

# set up colors for plots
cmap = plt.cm.Accent
cmaplist = [cmap(i) for i in numpy.linspace(0,1,6)]
cmaplist[2] = (0.99477124214172363, 0.83529412746429454, 0.55032682418823242)

# **********************************
# MODEL ACCURACY PLOT - CARTESIAN
# ********************************** 

plt.figure(figsize=[15,10])

plt.subplot(121)
plt.title('Ftip X (N)')
plt.plot([-5,5],[-5,5],'--k')
for muscleIdx, muscle in enumerate(muscleLables):
    plt.plot(ftipForcesExpAvrg[muscleIdx::6,0],
             ftipForcesPred0_all[muscleIdx::6,0],'o',label=muscle+'_init',
             markeredgecolor=cmaplist[muscleIdx],markeredgewidth=1,markerfacecolor='none')
    plt.plot(ftipForcesExpAvrg[muscleIdx::6,0],
             ftipForcesPredOpt_all[muscleIdx::6,0],'o',label=muscle+'_opt',
             color=cmaplist[muscleIdx],markeredgewidth=0)      

plt.legend(loc='lower right',fontsize='x-small')
plt.xlabel('Experiment')
plt.ylabel('Prediction')
plt.axis('square')
plt.xlim([-2.5,1])
plt.ylim([-2.5,1])
plt.grid(b=True)


plt.subplot(122)
plt.title('Ftip Y (N)')
plt.plot([-5,5],[-5,5],'--k')
for muscleIdx, muscle in enumerate(muscleLables):
    plt.plot(ftipForcesExpAvrg[muscleIdx::6,1],
             ftipForcesPred0_all[muscleIdx::6,1],'o',label=muscle+'_init',
             markeredgecolor=cmaplist[muscleIdx],markeredgewidth=1,markerfacecolor='none')
    plt.plot(ftipForcesExpAvrg[muscleIdx::6,1],
             ftipForcesPredOpt_all[muscleIdx::6,1],'o',label=muscle+'_opt',
             color=cmaplist[muscleIdx],markeredgewidth=0)      

plt.legend(loc='lower right',fontsize='x-small')
plt.xlabel('Experiment')
plt.ylabel('Prediction')
plt.axis('square')
plt.xlim([-4,2])
plt.ylim([-4,2])
plt.grid(b=True)

if save == True:
    plt.savefig(outdir+'CHECK_accuracy_cartesian.pdf')

#==============================================================================
# PRINT
#==============================================================================

print '\n************************************************'
print '*** OPTIMIZATION RESULT OVERVIEW'
print '************************************************'

print '\n* TOLERANCES'
print '  -> MCP x/y: %.1f / %.1f'%(MCPTol_x,MCPTol_y)
print '  -> PIP x/y: %.1f / %.1f'%(PIPTol_x,DIPTol_y)
print '  -> DIP x/y: %.1f / %.1f'%(DIPTol_x,DIPTol_y)
print '  -> EM CS-frac LB for RI/LU/UI/LE: %.1f / %.1f / %.1f / %.1f'%(fractions_lb[0],fractions_lb[1],fractions_lb[2],fractions_lb[3])
print '  -> EM CS-frac UB for RI/LU/UI/LE: %.1f / %.1f / %.1f / %.1f'%(fractions_ub[0],fractions_ub[1],fractions_ub[2],fractions_ub[3])

print '\n* COST FUNCTIONS'
print '  -> FDS init/opt: %.2f / %.2f'%(cost0_FDS,costOpt_FDS)
print '  -> FDP init/opt: %.2f / %.2f'%(cost0_FDP,costOpt_FDP)
print '  -> EM init/opt:  %.2f / %.2f'%(cost0_EM,costOpt_EM)

print '\n* MEAN VIA-POINT SHIFT (ABS)'
print '  -> FDS: %.2f mm'%(MAE_FDS_points*1000)
print '  -> FDP: %.2f mm'%(MAE_FDP_points*1000)
print '  -> EM:  %.2f mm'%(MAE_EM_points*1000)

print '\n* RMSE FOR EACH MUSCLE (post processing)'
print '  -> FDS init/opt: %.2f / %.2f'%(RMSE0_FDS,RMSEOpt_FDS)
print '  -> FDP init/opt: %.2f / %.2f'%(RMSE0_FDP,RMSEOpt_FDP)
print '  -> RI init/opt:  %.2f / %.2f'%(RMSE0_RI,RMSEOpt_RI)
print '  -> LU init/opt:  %.2f / %.2f'%(RMSE0_LU,RMSEOpt_LU)
print '  -> UI init/opt:  %.2f / %.2f'%(RMSE0_UI,RMSEOpt_UI)
print '  -> LE init/opt:  %.2f / %.2f'%(RMSE0_LE,RMSEOpt_LE)

print '\n* RMSE REL (TO MEAN FEXP_MAG) FOR EACH MUSCLE'
print '  -> FDS init/opt: %.2f / %.2f'%(RMSE0_FDS_rel*100,RMSEOpt_FDS_rel*100)
print '  -> FDP init/opt: %.2f / %.2f'%(RMSE0_FDP_rel*100,RMSEOpt_FDP_rel*100)
print '  -> RI init/opt:  %.2f / %.2f'%(RMSE0_RI_rel*100,RMSEOpt_RI_rel*100)
print '  -> LU init/opt:  %.2f / %.2f'%(RMSE0_LU_rel*100,RMSEOpt_LU_rel*100)
print '  -> UI init/opt:  %.2f / %.2f'%(RMSE0_UI_rel*100,RMSEOpt_UI_rel*100)
print '  -> LE init/opt:  %.2f / %.2f'%(RMSE0_LE_rel*100,RMSEOpt_LE_rel*100)

print '\n* RMSE FOR WHOLE MODEL'
print '  -> FTIP TOT init/opt :      %.2f / %.2f N'%(RMSE0_all,RMSEOpt_all)
print '  -> FTIP TOT REL init/opt :  %.2f / %.2f %%'%(RMSE0_all_rel*100,RMSEOpt_all_rel*100)
print '  -> VIA-POINT SHIFT mean/max:    %.2f / %.2f mm'%(MAE_all_points*1000,numpy.max(shift_all)*1000)



#==============================================================================
# VTK PLOT        
#==============================================================================
     
# set up finctional load for model (just for visualization)
ExtLoad_tmp = extLoad()
ExtLoad_tmp.addDPforce(numpy.zeros(3),p_ext)        

if save == True:
    vtkSnap = outdir + 'CHECK_VTK_model_RU_view.png'
else:
    vtkSnap = False

# show initial model
fingerModel.vtkVisualize(  0,0,0,0, ExtLoad_tmp,
                        plot_muscle = True,
                        plot_stl = True,
                        plot_RF = False,
                        stl_opacity = 0.5,
                        scale_muscle_activation = False,
                        scale_muscle_pcsa = False,
                        view_roll=0,
                        plot_colBar=False,
                        use_F_mus=numpy.zeros(10))         

# show optimized model      
fingerModelOpt.vtkVisualize(  0,0,0,0, ExtLoad_tmp,
                        plot_muscle = True,
                        plot_stl = True,
                        plot_RF = False,
                        stl_opacity = 0.5,
                        scale_muscle_activation = False,
                        scale_muscle_pcsa = False,
                        view_roll=0,
                        plot_colBar=False,
                        use_F_mus=numpy.zeros(10),
                        snapshot=vtkSnap)

#==============================================================================
# SAVE     
#==============================================================================

if save == True:
    # save path points
    header = 'D_X,D_Y,D_Z,P_X,P_Y,P_Z'
    numpy.savetxt(outdir+'MCP_path.csv',fingerModelOpt.MCPPathPoints,header=header,delimiter=',')
    numpy.savetxt(outdir+'PIP_path.csv',fingerModelOpt.PIPPathPoints,header=header,delimiter=',')
    numpy.savetxt(outdir+'DIP_path.csv',fingerModelOpt.DIPPathPoints,header=header,delimiter=',')
    
    # save EM force transmission fractions
    header = 'RI_ES, LU_ES, UI_ES, LE_ES'
    numpy.savetxt(outdir+'EM_CS_fractions.csv',numpy.array([xOpt_EM[32:]]),header=header,delimiter=',')
    
    # save optimization results
    header = 'RMSE0 [N], RMSE0_rel [%], RMSEOpt [N], RMSEOpt_rel [%], Shift_mean [mm], Shift_max [mm]'
    outArray = numpy.array([RMSE0_all,RMSE0_all_rel*100,RMSEOpt_all,RMSEOpt_all_rel*100,
                                  MAE_all_points*1000,numpy.max(shift_all)*1000])
    numpy.savetxt(outdir+'CHECK_calibration_results.csv',numpy.atleast_2d(outArray),header=header,delimiter=',')

