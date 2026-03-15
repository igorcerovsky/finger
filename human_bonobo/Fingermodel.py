# -*- coding: utf-8 -*-
"""

GENERIC MUSCULOSKELETAL FINGER MODEL

"""

import numpy
import vtk
import scipy.optimize as opt

class extLoad:
    """
    Class to collect external finger loading
    ****************************************************
    - Forces are expressed in global coordinate system
    - Points of force application are expressed in local bone coordinate system
    
    DP: Distal phalanx; MP: Middle phalanx; PP: Proximal phalanx
    """
    def __init__(self):
        self.DP_Forces = numpy.empty((0,3))
        self.DP_Points = numpy.empty((0,3))
        self.MP_Forces = numpy.empty((0,3))
        self.MP_Points = numpy.empty((0,3))
        self.PP_Forces = numpy.empty((0,3))
        self.PP_Points = numpy.empty((0,3))
        pass
    
    def addDPforce(self,F_ext,p_ext):
        self.DP_Forces = numpy.row_stack([self.DP_Forces,F_ext])
        self.DP_Points = numpy.row_stack([self.DP_Points,p_ext])
        
    def addMPforce(self,F_ext,p_ext):
        self.MP_Forces = numpy.row_stack([self.MP_Forces,F_ext])
        self.MP_Points = numpy.row_stack([self.MP_Points,p_ext])

    def addPPforce(self,F_ext,p_ext):
        self.PP_Forces = numpy.row_stack([self.PP_Forces,F_ext])
        self.PP_Points = numpy.row_stack([self.PP_Points,p_ext])        
        

class fullModel:
    """
    Class for generic finger model
    ****************************************************
    - Allows computing posture-specific moment arms
    - Compute joint torques to counteract external loading
    - Compute optimal muscle forces to balance external load (static optimization)
    - Compute joint loads
    
    The model largely follows descriptions provided by An et al. 1979
    """    
    
    # CONSTRUCTOR
    def __init__(self,O2O3In):
        # O2O3 is the main model scaling parameter 
        self.O2O3 = O2O3In
        
    # SET FUNCTIONS
    def setScalingPar(self,O2O3New):
        # define new scale factor
        self.O2O3 = O2O3New 
    
    def setTendonPaths(self,MCPPathPointsIn,PIPPathPointsIn,DIPPathPointsIn):
        # set tendon path points, relative to O2O3
        # DoF and muscles ordered consistent with An et al. 1979
        self.MCPPathPoints = MCPPathPointsIn
        self.PIPPathPoints = PIPPathPointsIn
        self.DIPPathPoints = DIPPathPointsIn
       
    def setEERatios(self,ES_RatiosIn,RB_RatiosIn,UB_RatiosIn,TE_RatiosIn):
        # extensor mechanism ratios
        # must be ordered as follows:
        #   [0]*RI + [1]*LU + [2]*UI + [3]*FDP + [4]*FDS + 
        #   [5]*LE + [6]*ES + [7]*TE + [8]*RB  + [9]*UB
        self.TE_Ratios = TE_RatiosIn
        self.ES_Ratios = ES_RatiosIn
        self.RB_Ratios = RB_RatiosIn
        self.UB_Ratios = UB_RatiosIn
              
    def setKinScalingPars(self,segRatios):
        # takes segment rations as input, computes physical segment lengths
        self.O0O1 = segRatios[0] * self.O2O3
        self.O1O2 = segRatios[1] * self.O2O3
        self.O3O4 = segRatios[2] * self.O2O3
        self.O4O5 = segRatios[3] * self.O2O3
        self.O5O6 = segRatios[4] * self.O2O3
        
    def setPCSA(self,PCSAEDCIn,PCSAFDSIn,PCSAFDPIn,PCSALUIn,PCSARIIn,PCSAUIIn):
        # set PCSA data of the model
        # upper bounds of extensor mechanism are already set accounting for
        # force transmission fractions
        self.PCSALE  = PCSAEDCIn
        self.PCSAFDP = PCSAFDPIn
        self.PCSAFDS = PCSAFDSIn
        self.PCSARI  = PCSARIIn
        self.PCSALU  = PCSALUIn
        self.PCSAUI  = PCSAUIIn
        self.PCSAES  = self.ES_Ratios[0]*PCSARIIn  + self.ES_Ratios[2]*PCSAUIIn + \
                       self.ES_Ratios[5]*PCSAEDCIn + self.ES_Ratios[1]*PCSALUIn
        self.PCSARB  = self.RB_Ratios[0]*PCSARIIn + self.RB_Ratios[5]*PCSAEDCIn + \
                       self.RB_Ratios[1]*PCSALUIn
        self.PCSAUB  = self.UB_Ratios[2]*PCSAUIIn + self.UB_Ratios[5]*PCSAEDCIn
        self.PCSATE  = self.TE_Ratios[8]*self.PCSARB + self.TE_Ratios[9]*self.PCSAUB + \
                       self.TE_Ratios[5]*PCSAEDCIn
            
    def setSpecTension(self,specTensionIn):
        # muscle specific tension used for maximum muscle force computation
        self.specTension = specTensionIn
    
    def setSTLFilename(self,MC_filename_In,PP_filename_In,MP_filename_In,DP_filename_In):
        # setting the paths to stl files (just for visualization)
        self.MC_filename = MC_filename_In
        self.PP_filename = PP_filename_In
        self.MP_filename = MP_filename_In
        self.DP_filename = DP_filename_In    
    
    # UTILITY FUNCTIONS
    def deg2rad(self,deg):
        # compute degrees from radians
        return (deg/180.*numpy.pi)    
    
    def computeRotation(self,theta):
        # compute rotation matrix given 3 angles (theta)
        tx,ty,tz = theta
        Rx = numpy.array([[1,0,0],
                          [0, numpy.cos(tx), -numpy.sin(tx)],
                          [0, numpy.sin(tx), numpy.cos(tx)]])
                          
        Ry = numpy.array([[numpy.cos(ty), 0, numpy.sin(ty)],
                          [0, 1, 0],
                          [-numpy.sin(ty), 0, numpy.cos(ty)]])
                          
        Rz = numpy.array([[numpy.cos(tz), -numpy.sin(tz), 0],
                          [numpy.sin(tz), numpy.cos(tz), 0],
                          [0,0,1]])
        return numpy.dot(Rx, numpy.dot(Ry, Rz))

    def computeMA(self,r_p,r_d,offset,axis,theta):
        # compute a moment arm using the generalized force method,
        # i.e. compute moment around specific axis and divide with muscle tension,
        # in a posture specific manner
        
        # compute posture specific rotation matrix
        R_mat = numpy.linalg.inv(self.computeRotation(theta))
        
        # transform into distal coordinate system
        r_p_ind = numpy.dot(R_mat,r_p) + offset
        axis_ind = numpy.dot(R_mat,axis) 
         
        r_d_ind = r_d    
        r_pd_ind = (r_p_ind-r_d_ind)
        r_pd_norm_ind = r_pd_ind/numpy.linalg.norm(r_pd_ind)  
        
        # compute moment from unit tension in direction r_pd, around axis origin
        mom_ind = numpy.cross(r_p_ind-offset,r_pd_norm_ind)   
        
        # project moment around origin onto axis
        mom_axis_ind = numpy.dot(mom_ind,axis_ind)
              
        # as a unit tension was applied, this moment is equivalent to the moment arm
        ma = mom_axis_ind
        
        return ma 
    
    def computeFVec(self,r_p,r_d,offset,theta,F_scalar,landsmeer=False):
        # compute a force vector for specific joint posture
        if landsmeer==False:
            R_mat = numpy.linalg.inv(self.computeRotation(theta))
        else:
            R_mat = numpy.diag(numpy.ones(3))
    
        r_p_ind = numpy.dot(R_mat,r_p) + offset
        r_d_ind = r_d    
        r_pd_ind = (r_p_ind-r_d_ind)
        r_pd_norm_ind = r_pd_ind/numpy.linalg.norm(r_pd_ind)  
                  
        F_vec = F_scalar * r_pd_norm_ind
        
        return F_vec        
    
    # COMPUTATION OF MOMENTARMS AND MOMENT ARM MATRIX
    def computeAllMA(self,DIP_flex,PIP_flex,MCP_flex,MCP_abd,landsmeer=False,return_Rmus=False):
        # compute all moment arms for specific posture
        # landsmeer flag = False means pure bowstringing
        #                = True means "wrapping"/landsmeer model 1 is enabled (see main manuscript) 
        
        # compute physical coordinates of tendon path points
        DIPPointsProx = self.DIPPathPoints[:,3:6]*self.O2O3 
        DIPPointsDist = self.DIPPathPoints[:,0:3]*self.O2O3 
        PIPPointsProx = self.PIPPathPoints[:,3:6]*self.O2O3
        PIPPointsDist = self.PIPPathPoints[:,0:3]*self.O2O3
        MCPPointsProx = self.MCPPathPoints[:,3:6]*self.O2O3
        MCPPointsDist = self.MCPPathPoints[:,0:3]*self.O2O3
        
        # MCP Flexion/Extension moment arms
        
        # rotate flexion axis depending on MCP abduction angle
        MCP_flex_axis = numpy.dot(self.computeRotation([0,self.deg2rad(MCP_abd),0]),numpy.array([0,0,1]))  
                                                   
        self.MCP_flex_MA_RI = self.computeMA(MCPPointsProx[2,:],MCPPointsDist[2,:],numpy.array([self.O5O6,0,0]),\
                                        MCP_flex_axis,numpy.array([0,self.deg2rad(MCP_abd),self.deg2rad(MCP_flex)])) 
        self.MCP_flex_MA_LU = self.computeMA(MCPPointsProx[3,:],MCPPointsDist[3,:],numpy.array([self.O5O6,0,0]),\
                                        MCP_flex_axis,numpy.array([0,self.deg2rad(MCP_abd),self.deg2rad(MCP_flex)])) 
        self.MCP_flex_MA_UI = self.computeMA(MCPPointsProx[4,:],MCPPointsDist[4,:],numpy.array([self.O5O6,0,0]),\
                                        MCP_flex_axis,numpy.array([0,self.deg2rad(MCP_abd),self.deg2rad(MCP_flex)])) 
                                        
        # pure bowstringing                                
        if landsmeer == False:        
            self.MCP_flex_MA_LE = self.computeMA(MCPPointsProx[5,:],MCPPointsDist[5,:],numpy.array([self.O5O6,0,0]),\
                                            MCP_flex_axis,numpy.array([0,self.deg2rad(MCP_abd),self.deg2rad(MCP_flex)]))
            self.MCP_flex_MA_FDP = self.computeMA(MCPPointsProx[0,:],MCPPointsDist[0,:],numpy.array([self.O5O6,0,0]),\
                                            MCP_flex_axis,numpy.array([0,self.deg2rad(MCP_abd),self.deg2rad(MCP_flex)]))
            self.MCP_flex_MA_FDS = self.computeMA(MCPPointsProx[1,:],MCPPointsDist[1,:],numpy.array([self.O5O6,0,0]),\
                                            MCP_flex_axis,numpy.array([0,self.deg2rad(MCP_abd),self.deg2rad(MCP_flex)]))                                            
        
        # if landsmeer model 1 is used, "wrapping" of muscles is considered
        elif landsmeer == True:
            if MCP_flex >= 0:
                # if angle >= 0: LE is wrapping, FDP and FDS are bowstringing
                self.MCP_flex_MA_LE = self.computeMA(MCPPointsProx[5,:],MCPPointsDist[5,:],numpy.array([self.O5O6,0,0]),\
                                            MCP_flex_axis,numpy.array([0,self.deg2rad(MCP_abd),self.deg2rad(0.0)])) 
                self.MCP_flex_MA_FDP = self.computeMA(MCPPointsProx[0,:],MCPPointsDist[0,:],numpy.array([self.O5O6,0,0]),\
                                            MCP_flex_axis,numpy.array([0,self.deg2rad(MCP_abd),self.deg2rad(MCP_flex)]))
                self.MCP_flex_MA_FDS = self.computeMA(MCPPointsProx[1,:],MCPPointsDist[1,:],numpy.array([self.O5O6,0,0]),\
                                            MCP_flex_axis,numpy.array([0,self.deg2rad(MCP_abd),self.deg2rad(MCP_flex)]))                                            

            else:
                # if angle < 0: FDP and FDS are wrapping, LE is bowstringing
                self.MCP_flex_MA_LE = self.computeMA(MCPPointsProx[5,:],MCPPointsDist[5,:],numpy.array([self.O5O6,0,0]),\
                                            MCP_flex_axis,numpy.array([0,self.deg2rad(MCP_abd),self.deg2rad(MCP_flex)]))
                self.MCP_flex_MA_FDP = self.computeMA(MCPPointsProx[0,:],MCPPointsDist[0,:],numpy.array([self.O5O6,0,0]),\
                                            MCP_flex_axis,numpy.array([0,self.deg2rad(MCP_abd),self.deg2rad(0.0)]))
                self.MCP_flex_MA_FDS = self.computeMA(MCPPointsProx[1,:],MCPPointsDist[1,:],numpy.array([self.O5O6,0,0]),\
                                            MCP_flex_axis,numpy.array([0,self.deg2rad(MCP_abd),self.deg2rad(0.0)]))

        # MCP Adduction/Abduction moment arms
                                        
        self.MCP_abd_MA_FDP = self.computeMA(MCPPointsProx[0,:],MCPPointsDist[0,:],numpy.array([self.O5O6,0,0]),\
                                        numpy.array([0,1,0]),numpy.array([0,self.deg2rad(MCP_abd),self.deg2rad(MCP_flex)]))
        self.MCP_abd_MA_FDS = self.computeMA(MCPPointsProx[1,:],MCPPointsDist[1,:],numpy.array([self.O5O6,0,0]),\
                                        numpy.array([0,1,0]),numpy.array([0,self.deg2rad(MCP_abd),self.deg2rad(MCP_flex)]))                                        
        self.MCP_abd_MA_RI = self.computeMA(MCPPointsProx[2,:],MCPPointsDist[2,:],numpy.array([self.O5O6,0,0]),\
                                        numpy.array([0,1,0]),numpy.array([0,self.deg2rad(MCP_abd),self.deg2rad(MCP_flex)]))    
        self.MCP_abd_MA_LU = self.computeMA(MCPPointsProx[3,:],MCPPointsDist[3,:],numpy.array([self.O5O6,0,0]),\
                                        numpy.array([0,1,0]),numpy.array([0,self.deg2rad(MCP_abd),self.deg2rad(MCP_flex)]))    
        self.MCP_abd_MA_UI = self.computeMA(MCPPointsProx[4,:],MCPPointsDist[4,:],numpy.array([self.O5O6,0,0]),\
                                        numpy.array([0,1,0]),numpy.array([0,self.deg2rad(MCP_abd),self.deg2rad(MCP_flex)]))           
        self.MCP_abd_MA_LE = self.computeMA(MCPPointsProx[5,:],MCPPointsDist[5,:],numpy.array([self.O5O6,0,0]),\
                                        numpy.array([0,1,0]),numpy.array([0,self.deg2rad(MCP_abd),self.deg2rad(MCP_flex)])) 
                                                                          
        # PIP Flexion/Extension moment arms
                           
        self.PIP_flex_MA_RB = self.computeMA(PIPPointsProx[1,:],PIPPointsDist[1,:],numpy.array([self.O3O4,0,0]),\
                                        numpy.array([0,0,1]),numpy.array([0,0,self.deg2rad(PIP_flex)]))                                    
        self.PIP_flex_MA_UB = self.computeMA(PIPPointsProx[2,:],PIPPointsDist[2,:],numpy.array([self.O3O4,0,0]),\
                                        numpy.array([0,0,1]),numpy.array([0,0,self.deg2rad(PIP_flex)]))                                    
        if landsmeer == False: 
            self.PIP_flex_MA_ES = self.computeMA(PIPPointsProx[4,:],PIPPointsDist[4,:],numpy.array([self.O3O4,0,0]),\
                                            numpy.array([0,0,1]),numpy.array([0,0,self.deg2rad(PIP_flex)]))     
            self.PIP_flex_MA_FDP = self.computeMA(PIPPointsProx[0,:],PIPPointsDist[0,:],numpy.array([self.O3O4,0,0]),\
                                            numpy.array([0,0,1]),numpy.array([0,0,self.deg2rad(PIP_flex)]))
            self.PIP_flex_MA_FDS = self.computeMA(PIPPointsProx[3,:],PIPPointsDist[3,:],numpy.array([self.O3O4,0,0]),\
                                            numpy.array([0,0,1]),numpy.array([0,0,self.deg2rad(PIP_flex)]))                                            
                                            
        elif landsmeer == True: 
            if PIP_flex >=0:
                self.PIP_flex_MA_ES = self.computeMA(PIPPointsProx[4,:],PIPPointsDist[4,:],numpy.array([self.O3O4,0,0]),\
                                                numpy.array([0,0,1]),numpy.array([0,0,self.deg2rad(0.0)]))                                      
                self.PIP_flex_MA_FDP = self.computeMA(PIPPointsProx[0,:],PIPPointsDist[0,:],numpy.array([self.O3O4,0,0]),\
                                                            numpy.array([0,0,1]),numpy.array([0,0,self.deg2rad(PIP_flex)]))
                self.PIP_flex_MA_FDS = self.computeMA(PIPPointsProx[3,:],PIPPointsDist[3,:],numpy.array([self.O3O4,0,0]),\
                                                numpy.array([0,0,1]),numpy.array([0,0,self.deg2rad(PIP_flex)]))   
            else:
                self.PIP_flex_MA_ES = self.computeMA(PIPPointsProx[4,:],PIPPointsDist[4,:],numpy.array([self.O3O4,0,0]),\
                                                numpy.array([0,0,1]),numpy.array([0,0,self.deg2rad(PIP_flex)])) 
                self.PIP_flex_MA_FDP = self.computeMA(PIPPointsProx[0,:],PIPPointsDist[0,:],numpy.array([self.O3O4,0,0]),\
                                                numpy.array([0,0,1]),numpy.array([0,0,self.deg2rad(0.0)]))
                self.PIP_flex_MA_FDS = self.computeMA(PIPPointsProx[3,:],PIPPointsDist[3,:],numpy.array([self.O3O4,0,0]),\
                                                numpy.array([0,0,1]),numpy.array([0,0,self.deg2rad(0.0)])) 
                                            
        # DIP Flexion/Extension moment arms
          
        if landsmeer == False:                                        
            self.DIP_flex_MA_TE = self.computeMA(DIPPointsProx[0,:],DIPPointsDist[0,:],numpy.array([self.O1O2,0,0]),\
                                            numpy.array([0,0,1]),numpy.array([0,0,self.deg2rad(DIP_flex)]))
            self.DIP_flex_MA_FDP = self.computeMA(DIPPointsProx[1,:],DIPPointsDist[1,:],numpy.array([self.O1O2,0,0]),\
                                            numpy.array([0,0,1]),numpy.array([0,0,self.deg2rad(DIP_flex)])) 
        elif landsmeer == True:
            if DIP_flex >= 0:
                self.DIP_flex_MA_TE = self.computeMA(DIPPointsProx[0,:],DIPPointsDist[0,:],numpy.array([self.O1O2,0,0]),\
                                                numpy.array([0,0,1]),numpy.array([0,0,self.deg2rad(0.0)])) 
                self.DIP_flex_MA_FDP = self.computeMA(DIPPointsProx[1,:],DIPPointsDist[1,:],numpy.array([self.O1O2,0,0]),\
                                                numpy.array([0,0,1]),numpy.array([0,0,self.deg2rad(DIP_flex)]))
            else:
                self.DIP_flex_MA_TE = self.computeMA(DIPPointsProx[0,:],DIPPointsDist[0,:],numpy.array([self.O1O2,0,0]),\
                                                numpy.array([0,0,1]),numpy.array([0,0,self.deg2rad(DIP_flex)]))
                self.DIP_flex_MA_FDP = self.computeMA(DIPPointsProx[1,:],DIPPointsDist[1,:],numpy.array([self.O1O2,0,0]),\
                                                numpy.array([0,0,1]),numpy.array([0,0,self.deg2rad(0.0)]))
                                                
    def computeForceTransmissionMatrix(self,DIP_flex,PIP_flex,MCP_flex,MCP_abd,landsmeer=False):
        
        # method to set up the 4x6 effective moment arm matrix:
        #               RI  LU  UI  FDP FDS  LE
        #   DIP_flex
        #   PIP_flex
        #   MCP_flex
        #   MCP_abd
        
        self.computeAllMA(DIP_flex,PIP_flex,MCP_flex,MCP_abd,landsmeer=landsmeer)
                                                                                      
        T_mus = numpy.zeros((4,6))
        # DIP_flex (DoF 1)
        T_mus[0,0] = self.DIP_flex_MA_TE * self.TE_Ratios[8] * self.RB_Ratios[0]   # RI
        T_mus[0,1] = self.DIP_flex_MA_TE * self.TE_Ratios[8] * self.RB_Ratios[1]   # LU
        T_mus[0,2] = self.DIP_flex_MA_TE * self.TE_Ratios[9] * self.UB_Ratios[2]   # UI
        T_mus[0,3] = self.DIP_flex_MA_FDP                                          # FDP
        T_mus[0,4] = 0.0                                                           # FDS
        T_mus[0,5] = self.DIP_flex_MA_TE * (self.TE_Ratios[8] * self.RB_Ratios[5] + \
                                            self.TE_Ratios[9] * self.UB_Ratios[5] + \
                                            self.TE_Ratios[5]) # LE                                              
                                            
        # PIP_flex (DoF 2)
        T_mus[1,0] = self.PIP_flex_MA_RB * self.RB_Ratios[0] + \
                     self.PIP_flex_MA_ES * self.ES_Ratios[0]                       # RI
        T_mus[1,1] = self.PIP_flex_MA_RB * self.RB_Ratios[1] + \
                     self.PIP_flex_MA_ES * self.ES_Ratios[1]                       # LU
        T_mus[1,2] = self.PIP_flex_MA_UB * self.UB_Ratios[2] + \
                     self.PIP_flex_MA_ES * self.ES_Ratios[2]                       # UI
        T_mus[1,3] = self.PIP_flex_MA_FDP                                          # FDP
        T_mus[1,4] = self.PIP_flex_MA_FDS                                          # FDS
        T_mus[1,5] = self.PIP_flex_MA_RB * self.RB_Ratios[5] + \
                     self.PIP_flex_MA_UB * self.UB_Ratios[5] + \
                     self.PIP_flex_MA_ES * self.ES_Ratios[5]                       # LE                     
                         
        # MCP_flex (DoF 3)
        T_mus[2,0] = self.MCP_flex_MA_RI                                           # RI
        T_mus[2,1] = self.MCP_flex_MA_LU                                           # LU
        T_mus[2,2] = self.MCP_flex_MA_UI                                           # UI
        T_mus[2,3] = self.MCP_flex_MA_FDP                                          # FDP
        T_mus[2,4] = self.MCP_flex_MA_FDS                                          # FDS
        T_mus[2,5] = self.MCP_flex_MA_LE                                           # LE
        
        # MCP_abd (DoF 4)
        T_mus[3,0] = self.MCP_abd_MA_RI                                            # RI
        T_mus[3,1] = self.MCP_abd_MA_LU                                            # LU
        T_mus[3,2] = self.MCP_abd_MA_UI                                            # UI
        T_mus[3,3] = self.MCP_abd_MA_FDP                                           # FDP
        T_mus[3,4] = self.MCP_abd_MA_FDS                                           # FDS
        T_mus[3,5] = self.MCP_abd_MA_LE                                            # LE
                    
        return T_mus 

    # COMPUTATION OF JOINT TORQUES FROM EXTERNAL LOADING

    def computeJacobian(self,DIP_flex,PIP_flex,MCP_flex,MCP_abd,p_ext,body):
        # method to compute jacobian for specific posture and point of force application
        # p_ext is a vector to the point of force application
        # body is either DP, MP or PP
    
        # define physical bone lengths
        l1 = self.O0O1 + self.O1O2
        l2 = self.O2O3 + self.O3O4
        l3 = self.O4O5 + self.O5O6
        
        a_MCP_abd_0 = numpy.array([[0],[1],[0]])
        a_MCP_flex_0 = numpy.array([[0],[0],[1]])
        
        r_PIP_0 = numpy.array([[-l3],[0],[0]])                 
        a_PIP_0 = numpy.array([[0],[0],[1]])
        
        r_DIP_0 = numpy.array([[-l2],[0],[0]])                 
        a_DIP_0 = numpy.array([[0],[0],[1]])
                       
        r_ext_0 = numpy.reshape(p_ext,[3,1])
        
        # transformed points and axes
        a_MCP_abd_1 = a_MCP_abd_0
        a_MCP_flex_1 = numpy.dot(self.computeRotation([0,self.deg2rad(MCP_abd),0]),a_MCP_flex_0)
        
        r_PIP_1 = numpy.dot(self.computeRotation([0,self.deg2rad(MCP_abd),self.deg2rad(MCP_flex)]),r_PIP_0)
        a_PIP_1 = numpy.dot(self.computeRotation([0,self.deg2rad(MCP_abd),self.deg2rad(MCP_flex)]),a_PIP_0)
        
        r_DIP_1 = numpy.dot(self.computeRotation([0,self.deg2rad(MCP_abd),self.deg2rad(MCP_flex+PIP_flex)]),r_DIP_0)
        a_DIP_1 = numpy.dot(self.computeRotation([0,self.deg2rad(MCP_abd),self.deg2rad(MCP_flex+PIP_flex)]),a_DIP_0)
        
        if body=='DP':         
            r_ext_1 = numpy.dot(self.computeRotation([0,self.deg2rad(MCP_abd),self.deg2rad(MCP_flex+PIP_flex+DIP_flex)]),r_ext_0)
            
            J = numpy.column_stack( [ numpy.cross(a_DIP_1,r_ext_1,axis=0),
                                      numpy.cross(a_PIP_1,(r_DIP_1+r_ext_1),axis=0),
                                      numpy.cross(a_MCP_flex_1,(r_PIP_1+r_DIP_1+r_ext_1),axis=0),
                                      numpy.cross(a_MCP_abd_1,(r_PIP_1+r_DIP_1+r_ext_1),axis=0)])
        elif body=='MP':
            r_ext_1 = numpy.dot(self.computeRotation([0,self.deg2rad(MCP_abd),self.deg2rad(MCP_flex+PIP_flex)]),r_ext_0)
            
            J = numpy.column_stack( [ numpy.zeros(3),
                                      numpy.cross(a_PIP_1,(r_ext_1),axis=0),
                                      numpy.cross(a_MCP_flex_1,(r_PIP_1+r_ext_1),axis=0),
                                      numpy.cross(a_MCP_abd_1,(r_PIP_1+r_ext_1),axis=0)])
        elif body=='PP':
            r_ext_1 = numpy.dot(self.computeRotation([0,self.deg2rad(MCP_abd),self.deg2rad(MCP_flex)]),r_ext_0)
            
            J = numpy.column_stack( [ numpy.zeros(3),
                                      numpy.zeros(3),
                                      numpy.cross(a_MCP_flex_1,(r_ext_1),axis=0),
                                      numpy.cross(a_MCP_abd_1,(r_ext_1),axis=0)])                                      
        return J                                      

    def computeQExt(self,DIP_flex,PIP_flex,MCP_flex,MCP_abd,F_ext,p_ext,body):                                  
        # method  to compute torques from external loading, i.e. "inverse dynamics"       
        
        # compute Jacobian
        J = self.computeJacobian(DIP_flex,PIP_flex,MCP_flex,MCP_abd,p_ext,body)

        # external forces, mapped to generalized coordinates
        Q_ext = numpy.dot(J.T,F_ext)
        return Q_ext

    def computeInverseDyn(self,DIP_flex,PIP_flex,MCP_flex,MCP_abd,extLoads):
        # function to compute total torques from ALL external loading contained
        # in extLoads class
    
        Q_ext_sum = numpy.zeros(4)
        for i in range(len(extLoads.DP_Forces)):
            Q_ext_sum = Q_ext_sum + self.computeQExt(   DIP_flex,PIP_flex,MCP_flex,MCP_abd, \
                                            extLoads.DP_Forces[i,:],extLoads.DP_Points[i,:],'DP')
        
        for i in range(len(extLoads.MP_Forces)):
            Q_ext_sum = Q_ext_sum + self.computeQExt(   DIP_flex,PIP_flex,MCP_flex,MCP_abd, \
                                            extLoads.MP_Forces[i,:],extLoads.MP_Points[i,:],'MP')
        
        for i in range(len(extLoads.PP_Forces)):
            Q_ext_sum = Q_ext_sum + self.computeQExt(   DIP_flex,PIP_flex,MCP_flex,MCP_abd, \
                                            extLoads.PP_Forces[i,:],extLoads.PP_Points[i,:],'PP')
        
        return Q_ext_sum        


    # COMPUTATION OF MUSCLE FORCES / STATIC OPTIMIZATION  
           
    def func_stress_quadratic(self,x):          
        # cost function: squared muscle stress (Modified to IGNORE PCSA limits)
        # Originally: F_norm = x[0:self.optIdx]/self.F_mus_limit[0:self.optIdx] * self.specTension
        # We now simply minimize the raw sum of squared forces to treat all muscles 
        # as "theoretical" generic actuators without volume-specific advantages.
        F_norm = x[0:self.optIdx] 
        J = numpy.sum(F_norm**2) * self.critScale
        return J    
    
    def func_stress_quadratic_prime(self,x):
        # Modified derivative corresponding to the raw force minimization
        J_prime = 2.0 * x[0:self.optIdx] * self.critScale
        return J_prime    
    
    # equality constraints based on transmission matrix
    def eq_T_j1(self,x):
        return numpy.dot(self.T_mus_tmp[0,:],x) + self.Q_ext_tmp[0]

    def eq_T_j2(self,x):
        return numpy.dot(self.T_mus_tmp[1,:],x) + self.Q_ext_tmp[1]
        
    def eq_T_j3(self,x):
        return numpy.dot(self.T_mus_tmp[2,:],x) + self.Q_ext_tmp[2]
    
    def eq_T_j4(self,x):
        return numpy.dot(self.T_mus_tmp[3,:],x) + self.Q_ext_tmp[3]     

    # analytical gradients of equality constraints
    def eq_T_j1_prime(self,x):
        return self.T_mus_tmp[0,:].flatten()
    
    def eq_T_j2_prime(self,x):
        return self.T_mus_tmp[1,:].flatten()
    
    def eq_T_j3_prime(self,x):
        return self.T_mus_tmp[2,:].flatten()
        
    def eq_T_j4_prime(self,x):
        return self.T_mus_tmp[3,:].flatten()    
    
    def computeMuscleForces(self,DIP_flex,PIP_flex,MCP_flex,MCP_abd,extLoads,landsmeer=False, \
                                F_mus_0_scalar=0,critScaleIn=1,tol=1e-10, \
                                maxIterations=100,returnExitMode=False,returnFuncValue=False, \
                                printWarnings=False, silent=False, \
                                useReserve=False,reservePCSA=numpy.ones(4)):
        
        # set a scaling factor for optimization criterion
        # (this might be needed to avoid numerical problems)
        self.critScale = critScaleIn
                     
        # compute external torque requirements and save as member variable
        self.Q_ext_tmp = self.computeInverseDyn(DIP_flex,PIP_flex,MCP_flex,MCP_abd,extLoads)
        
        # set upper bound for muscle force in order RI - LU - UI - FDP - FDS - LE - ES - TE - RB - UB                                     
        # UNIFORM LIMIT to ignore muscle size differences and act as an unbounded theoretical muscle
        tmp_vals = numpy.array([self.PCSARI,self.PCSALU,self.PCSAUI,self.PCSAFDP,self.PCSAFDS,\
                            self.PCSALE,self.PCSAES,self.PCSATE,self.PCSARB,self.PCSAUB]) * self.specTension
        self.F_mus_limit = numpy.ones(10) * max(numpy.max(tmp_vals), 5000.0)
                                               
        # set up force transmission matrix                                            
        self.T_mus_tmp = self.computeForceTransmissionMatrix(DIP_flex,PIP_flex,MCP_flex,MCP_abd,landsmeer=landsmeer)
        
        # set list of equality constraints
        eqcons_tmp=[self.eq_T_j1,self.eq_T_j2,self.eq_T_j3,self.eq_T_j4]                                    
        eqcons_prime_tmp = [self.eq_T_j1_prime,self.eq_T_j2_prime,self.eq_T_j3_prime,self.eq_T_j4_prime] 
        
        # check if reserve actuator should be used or not
        if useReserve == True:
            # reserve actuators are "muscles" that act with a lever arm of 1meters,
            # can both pull and push, and are limited by a force +/- reserveScale*specTension
            self.T_mus_tmp = numpy.column_stack((self.T_mus_tmp,numpy.diag(numpy.ones(4))))
            self.optIdx = 10
            self.F_mus_limit[6:] = reservePCSA * self.specTension
            
            # new bounds: reserve actuators may produce tensile AND compressive forces
            bounds_tmp =[(0.0,self.F_mus_limit[0]), (0.0,self.F_mus_limit[1]),\
                    (0.0,self.F_mus_limit[2]),(0.0,self.F_mus_limit[3]),\
                    (0.0,self.F_mus_limit[4]),(0.0,self.F_mus_limit[5]),\
                    (-numpy.inf,numpy.inf),(-numpy.inf,numpy.inf),\
                    (-numpy.inf,numpy.inf),(-numpy.inf,numpy.inf)]
            
            F_mus_0 = self.F_mus_limit[:10]*F_mus_0_scalar
            opt_result = opt.fmin_slsqp(self.func_stress_quadratic,F_mus_0,eqcons=eqcons_tmp,bounds=bounds_tmp,
                                    fprime = self.func_stress_quadratic_prime,
                                    fprime_eqcons = eqcons_prime_tmp,
                                    acc=tol,disp=0,iter=maxIterations,full_output=True)
            
        else:
            self.optIdx = 6
            bounds_tmp =[(0.0,self.F_mus_limit[0]), (0.0,self.F_mus_limit[1]),\
                    (0.0,self.F_mus_limit[2]),(0.0,self.F_mus_limit[3]),\
                    (0.0,self.F_mus_limit[4]),(0.0,self.F_mus_limit[5])]
                    
            F_mus_0 = self.F_mus_limit[:6]*F_mus_0_scalar
            opt_result = opt.fmin_slsqp(self.func_stress_quadratic,F_mus_0,eqcons=eqcons_tmp,bounds=bounds_tmp,
                                    fprime = self.func_stress_quadratic_prime,
                                    fprime_eqcons = eqcons_prime_tmp,
                                    acc=tol,disp=0,iter=maxIterations,full_output=True)          
        
        # assign optimization results
        F_mus = opt_result[0][0:10]
        exitMode = opt_result[3]
        funcValue = opt_result[1] 
        
        # double check for equality constraints
        res = numpy.array([ self.eq_T_j1(F_mus), self.eq_T_j2(F_mus), self.eq_T_j3(F_mus), self.eq_T_j4(F_mus)])
        res = numpy.abs(res)
        
        if (exitMode==0 and numpy.any((res>tol)==True)):
            # special exit mode, if slsqp ignores the equality constraints
            exitMode=99
            
        tolDigits = numpy.log(funcValue)-numpy.log(tol)
        
        if printWarnings==True:
            if (tolDigits)<8:
                print('  WARNING: log(funcValue)-numpy.log(tol) = %.1f < 8'%tolDigits)
                print('           Decrease tolerace or double-check results.'%tolDigits)
         

        if exitMode!=0 and silent==False:
            print('  ERROR: Static Optimization Failed. Exit mode %i'%exitMode)

                
        # return values as requested
        if returnExitMode:          
            if returnFuncValue:
                return F_mus, exitMode,funcValue
            return F_mus, exitMode 
        else:
            if returnFuncValue:
                return F_mus, funcValue
            return F_mus         
        
    # COMPUTATION OF JOINT LOADS    
        
    def computeJointReaction(self,DIP_flex,PIP_flex,MCP_flex,MCP_abd,extLoads,\
                                use_F_mus=[],inGlobal=False,landsmeer=False):
        
        # set up tendon path points in physical coordinates
        DIPPointsProx = self.DIPPathPoints[:,3:6]*self.O2O3 
        DIPPointsDist = self.DIPPathPoints[:,0:3]*self.O2O3 
        PIPPointsProx = self.PIPPathPoints[:,3:6]*self.O2O3
        PIPPointsDist = self.PIPPathPoints[:,0:3]*self.O2O3
        MCPPointsProx = self.MCPPathPoints[:,3:6]*self.O2O3
        MCPPointsDist = self.MCPPathPoints[:,0:3]*self.O2O3
       
        # allows either using a predifined list of muscles forces, or, if not given,
        # muscle forces will be computed for respective posture
        if use_F_mus==[]:
            F_mus = self.computeMuscleForces(DIP_flex,PIP_flex,MCP_flex,MCP_abd,extLoads)
        else:
            if use_F_mus.shape[0]==6:
                # muscle force order RI - LU - UI - FDP - FDS - LE - ES - TE - RB - UB
                F_mus_new = numpy.zeros(10)
                F_mus_new[:6] = use_F_mus
                F_mus_new[6]  = self.ES_Ratios[0]*F_mus_new[0]  + self.ES_Ratios[2]*F_mus_new[2] + \
                                self.ES_Ratios[5]*F_mus_new[5] + self.ES_Ratios[1]*F_mus_new[1]
                F_mus_new[8]  = self.RB_Ratios[0]*F_mus_new[0] + self.RB_Ratios[5]*F_mus_new[5] + \
                                self.RB_Ratios[1]*F_mus_new[1]
                F_mus_new[9]  = self.UB_Ratios[2]*F_mus_new[2] + self.UB_Ratios[5]*F_mus_new[5]
                F_mus_new[7]  = self.TE_Ratios[8]*F_mus_new[8] + self.TE_Ratios[9]*F_mus_new[9] + \
                                self.TE_Ratios[5]*F_mus_new[5]
                F_mus = F_mus_new
            elif use_F_mus.shape[0]==10:
                F_mus = use_F_mus            
       
        # computes joint reaction force on proximal bone, in proximal coordinate system
       
        # DIP
        R_ref_O1 = numpy.linalg.inv(self.computeRotation([0,self.deg2rad(MCP_abd),self.deg2rad(MCP_flex+PIP_flex+DIP_flex)]))
        F_ext_inO1 = numpy.zeros(3)        
        for F_ext_DP in extLoads.DP_Forces:        
            F_ext_inO1 = F_ext_inO1 + numpy.dot(R_ref_O1,F_ext_DP)
        
        if landsmeer==False:
            F_FDP_inO1 = self.computeFVec(DIPPointsProx[1,:],DIPPointsDist[1,:],numpy.array([self.O1O2,0,0]),\
                                            numpy.array([0,0,self.deg2rad(DIP_flex)]),F_mus[3]) 
            F_TE_inO1 = self.computeFVec(DIPPointsProx[0,:],DIPPointsDist[0,:],numpy.array([self.O1O2,0,0]),\
                                            numpy.array([0,0,self.deg2rad(DIP_flex)]),F_mus[7]) 
        else:
            if DIP_flex>=0:
                F_FDP_inO1 = self.computeFVec(DIPPointsProx[1,:],DIPPointsDist[1,:],numpy.array([self.O1O2,0,0]),\
                                            numpy.array([0,0,self.deg2rad(DIP_flex)]),F_mus[3],landsmeer=False) 
                F_TE_inO1 = self.computeFVec(DIPPointsProx[0,:],DIPPointsDist[0,:],numpy.array([self.O1O2,0,0]),\
                                            numpy.array([0,0,self.deg2rad(DIP_flex)]),F_mus[7],landsmeer=True) 
            else:
                F_FDP_inO1 = self.computeFVec(DIPPointsProx[1,:],DIPPointsDist[1,:],numpy.array([self.O1O2,0,0]),\
                                            numpy.array([0,0,self.deg2rad(DIP_flex)]),F_mus[3],landsmeer=True) 
                F_TE_inO1 = self.computeFVec(DIPPointsProx[0,:],DIPPointsDist[0,:],numpy.array([self.O1O2,0,0]),\
                                            numpy.array([0,0,self.deg2rad(DIP_flex)]),F_mus[7],landsmeer=False) 
        
        F_DIP_inO1 = -F_ext_inO1 - F_FDP_inO1 - F_TE_inO1
                
        R_O1_O2 = self.computeRotation([0,0,self.deg2rad(DIP_flex)])
        F_DIP_inO2 = -numpy.dot(R_O1_O2,F_DIP_inO1)
        
        # PIP
        R_ref_O3 = numpy.linalg.inv(self.computeRotation([0,self.deg2rad(MCP_abd),self.deg2rad(MCP_flex+PIP_flex)]))
        F_ext_inO3 = numpy.zeros(3)        
        for F_ext_DP in extLoads.DP_Forces:        
            F_ext_inO3 = F_ext_inO3 + numpy.dot(R_ref_O3,F_ext_DP)
        for F_ext_MP in extLoads.MP_Forces:        
            F_ext_inO3 = F_ext_inO3 + numpy.dot(R_ref_O3,F_ext_MP)        
                 
        F_RB_inO3 = self.computeFVec(PIPPointsProx[1,:],PIPPointsDist[1,:],numpy.array([self.O3O4,0,0]),\
                                        numpy.array([0,0,self.deg2rad(PIP_flex)]),F_mus[8]) 
        F_UB_inO3 = self.computeFVec(PIPPointsProx[2,:],PIPPointsDist[2,:],numpy.array([self.O3O4,0,0]),\
                                        numpy.array([0,0,self.deg2rad(PIP_flex)]),F_mus[9])                                         
        
        if landsmeer==False:
            F_FDP_inO3 = self.computeFVec(PIPPointsProx[0,:],PIPPointsDist[0,:],numpy.array([self.O3O4,0,0]),\
                                        numpy.array([0,0,self.deg2rad(PIP_flex)]),F_mus[3]) 
            F_FDS_inO3 = self.computeFVec(PIPPointsProx[3,:],PIPPointsDist[3,:],numpy.array([self.O3O4,0,0]),\
                                    numpy.array([0,0,self.deg2rad(PIP_flex)]),F_mus[4])                                         
            F_ES_inO3 = self.computeFVec(PIPPointsProx[4,:],PIPPointsDist[4,:],numpy.array([self.O3O4,0,0]),\
                                            numpy.array([0,0,self.deg2rad(PIP_flex)]),F_mus[6])
        else:
            if PIP_flex>=0:
                F_FDP_inO3 = self.computeFVec(PIPPointsProx[0,:],PIPPointsDist[0,:],numpy.array([self.O3O4,0,0]),\
                                        numpy.array([0,0,self.deg2rad(PIP_flex)]),F_mus[3],landsmeer=False) 
                F_FDS_inO3 = self.computeFVec(PIPPointsProx[3,:],PIPPointsDist[3,:],numpy.array([self.O3O4,0,0]),\
                                        numpy.array([0,0,self.deg2rad(PIP_flex)]),F_mus[4],landsmeer=False)                                         
                F_ES_inO3 = self.computeFVec(PIPPointsProx[4,:],PIPPointsDist[4,:],numpy.array([self.O3O4,0,0]),\
                                        numpy.array([0,0,self.deg2rad(PIP_flex)]),F_mus[6],landsmeer=True)
            else:
                F_FDP_inO3 = self.computeFVec(PIPPointsProx[0,:],PIPPointsDist[0,:],numpy.array([self.O3O4,0,0]),\
                                        numpy.array([0,0,self.deg2rad(PIP_flex)]),F_mus[3],landsmeer=True) 
                F_FDS_inO3 = self.computeFVec(PIPPointsProx[3,:],PIPPointsDist[3,:],numpy.array([self.O3O4,0,0]),\
                                        numpy.array([0,0,self.deg2rad(PIP_flex)]),F_mus[4],landsmeer=True)                                         
                F_ES_inO3 = self.computeFVec(PIPPointsProx[4,:],PIPPointsDist[4,:],numpy.array([self.O3O4,0,0]),\
                                        numpy.array([0,0,self.deg2rad(PIP_flex)]),F_mus[6],landsmeer=False)
        
        F_PIP_inO3 = -F_ext_inO3 - F_FDP_inO3 - F_FDS_inO3 - F_ES_inO3 - F_RB_inO3 - F_UB_inO3
        R_O3_O4 = self.computeRotation([0,0,self.deg2rad(PIP_flex)])
        F_PIP_inO4 = -numpy.dot(R_O3_O4,F_PIP_inO3)
        
        # MCP
        R_ref_O5 = numpy.linalg.inv(self.computeRotation([0,self.deg2rad(MCP_abd),self.deg2rad(MCP_flex)]))
        F_ext_inO5 = numpy.zeros(3)        
        for F_ext_DP in extLoads.DP_Forces:        
            F_ext_inO5 = F_ext_inO5 + numpy.dot(R_ref_O5,F_ext_DP)
        for F_ext_MP in extLoads.MP_Forces:        
            F_ext_inO5 = F_ext_inO5 + numpy.dot(R_ref_O5,F_ext_MP)       
        for F_ext_PP in extLoads.PP_Forces:        
            F_ext_inO5 = F_ext_inO5 + numpy.dot(R_ref_O5,F_ext_PP)  
        
        F_RI_inO5 = self.computeFVec(MCPPointsProx[2,:],MCPPointsDist[2,:],numpy.array([self.O5O6,0,0]),\
                                            numpy.array([0,self.deg2rad(MCP_abd),self.deg2rad(MCP_flex)]),F_mus[0])
        F_LU_inO5 = self.computeFVec(MCPPointsProx[3,:],MCPPointsDist[3,:],numpy.array([self.O5O6,0,0]),\
                                            numpy.array([0,self.deg2rad(MCP_abd),self.deg2rad(MCP_flex)]),F_mus[1])
        F_UI_inO5 = self.computeFVec(MCPPointsProx[4,:],MCPPointsDist[4,:],numpy.array([self.O5O6,0,0]),\
                                            numpy.array([0,self.deg2rad(MCP_abd),self.deg2rad(MCP_flex)]),F_mus[2])
 
        if landsmeer==False:
            F_FDP_inO5 = self.computeFVec(MCPPointsProx[0,:],MCPPointsDist[0,:],numpy.array([self.O5O6,0,0]),\
                                                numpy.array([0,self.deg2rad(MCP_abd),self.deg2rad(MCP_flex)]),F_mus[3])
            F_FDS_inO5 = self.computeFVec(MCPPointsProx[1,:],MCPPointsDist[1,:],numpy.array([self.O5O6,0,0]),\
                                        numpy.array([0,self.deg2rad(MCP_abd),self.deg2rad(MCP_flex)]),F_mus[4])                                            
            F_LE_inO5 = self.computeFVec(MCPPointsProx[5,:],MCPPointsDist[5,:],numpy.array([self.O5O6,0,0]),\
                                                numpy.array([0,self.deg2rad(MCP_abd),self.deg2rad(MCP_flex)]),F_mus[5])
        else:
            if MCP_flex>=0:
                F_FDP_inO5 = self.computeFVec(MCPPointsProx[0,:],MCPPointsDist[0,:],numpy.array([self.O5O6,0,0]),\
                                                numpy.array([0,self.deg2rad(MCP_abd),self.deg2rad(MCP_flex)]),F_mus[3],landsmeer=False)
                F_FDS_inO5 = self.computeFVec(MCPPointsProx[1,:],MCPPointsDist[1,:],numpy.array([self.O5O6,0,0]),\
                                        numpy.array([0,self.deg2rad(MCP_abd),self.deg2rad(MCP_flex)]),F_mus[4],landsmeer=False)                                            
                F_LE_inO5 = self.computeFVec(MCPPointsProx[5,:],MCPPointsDist[5,:],numpy.array([self.O5O6,0,0]),\
                                                numpy.array([0,self.deg2rad(MCP_abd),self.deg2rad(MCP_flex)]),F_mus[5],landsmeer=True)
            else:
                F_FDP_inO5 = self.computeFVec(MCPPointsProx[0,:],MCPPointsDist[0,:],numpy.array([self.O5O6,0,0]),\
                                                numpy.array([0,self.deg2rad(MCP_abd),self.deg2rad(MCP_flex)]),F_mus[3],landsmeer=True)
                F_FDS_inO5 = self.computeFVec(MCPPointsProx[1,:],MCPPointsDist[1,:],numpy.array([self.O5O6,0,0]),\
                                        numpy.array([0,self.deg2rad(MCP_abd),self.deg2rad(MCP_flex)]),F_mus[4],landsmeer=True)                                            
                F_LE_inO5 = self.computeFVec(MCPPointsProx[5,:],MCPPointsDist[5,:],numpy.array([self.O5O6,0,0]),\
                                                numpy.array([0,self.deg2rad(MCP_abd),self.deg2rad(MCP_flex)]),F_mus[5],landsmeer=False)
 
 
        F_MCP_inO5 = -F_ext_inO5 - F_FDP_inO5 - F_FDS_inO5 - F_LE_inO5 - F_RI_inO5 - F_LU_inO5 - F_UI_inO5
        R_O5_O6 = self.computeRotation([0,self.deg2rad(MCP_abd),self.deg2rad(MCP_flex)])
        F_MCP_inO6 = -numpy.dot(R_O5_O6,F_MCP_inO5)
        
        # combine all joint reaction forces on prox bone / in prox CS. optional: global CS
        # rowwise: DIP, PIP, MCP joint load in 3D
        if inGlobal:
            F_JR_All = numpy.row_stack([ numpy.dot(numpy.linalg.inv(R_ref_O1),F_DIP_inO1),\
                                         numpy.dot(numpy.linalg.inv(R_ref_O3),F_PIP_inO3),\
                                         numpy.dot(numpy.linalg.inv(R_ref_O5),F_MCP_inO5)])
        else:
            F_JR_All = numpy.row_stack([F_DIP_inO2,F_PIP_inO4,F_MCP_inO6])  
        
        return F_JR_All
     
     
    # VISUALIZATION WITH VTK
     
    def vtkTransMatrix(self,rot,t):
        matrix = vtk.vtkMatrix4x4()
        matrix.Identity()
        for i in range(3):
            matrix.SetElement(i, 0, rot[i,0])
            matrix.SetElement(i, 1, rot[i,1])
            matrix.SetElement(i, 2, rot[i,2])    
        
        transform = vtk.vtkTransform()
        transform.Translate(t)
        transform.Concatenate(matrix)
        
        return transform.GetMatrix()
        
    def readSTL(self,filename):
        reader = vtk.vtkSTLReader()
        reader.SetFileName(filename)
        reader.Update()
        
        polydata = vtk.vtkPolyData()
        polydata.DeepCopy(reader.GetOutput())
        
        normalgenerator = vtk.vtkPolyDataNormals()
        normalgenerator.SetInputData(polydata)
        normalgenerator.ComputePointNormalsOn()
        normalgenerator.ComputeCellNormalsOn()
        normalgenerator.Update()
        
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(normalgenerator.GetOutput())
        
        actor = vtk.vtkActor()    
        actor.SetMapper(mapper)
        
        return actor  
        
    def polyTube(self,points,radius,color):
        # Method to generate polyline-tube
        
        # generate vtk Points    
        vtkPoints = vtk.vtkPoints()
        for i in range(points.shape[0]):
            vtkPoints.InsertNextPoint(points[i,:])
        
        # generate vtk Lines
        vtkLines = vtk.vtkCellArray()
        for i in range(points.shape[0]-1):
            line = vtk.vtkLine()
            line.GetPointIds().SetId(0,i)
            line.GetPointIds().SetId(1,i+1)
            vtkLines.InsertNextCell(line)
    
        # generate polydata
        linesPolyData = vtk.vtkPolyData()
        linesPolyData.SetPoints(vtkPoints)
        linesPolyData.SetLines(vtkLines)
    
        # generate tube
        tubeFilter = vtk.vtkTubeFilter() 
        tubeFilter.SetInputData(linesPolyData)
        tubeFilter.SetRadius(radius)
        tubeFilter.SetNumberOfSides(50)
        tubeFilter.CappingOn()
        tubeFilter.Update()
        
        tubeMapper = vtk.vtkPolyDataMapper()
        tubeMapper.SetInputConnection(tubeFilter.GetOutputPort())
        tubeActor = vtk.vtkActor()
        tubeActor.SetMapper(tubeMapper)
        tubeActor.GetProperty().SetColor(color)
            
        return tubeActor    

    def polySpheres(self,points,radius,color):
        # Method to generate multiple spheres at given points
        appendFilter = vtk.vtkAppendPolyData()    
        for i in range(points.shape[0]):  
            sphere = vtk.vtkSphereSource()
            sphere.SetCenter(points[i,:])
            sphere.SetRadius(radius)
            sphere.SetPhiResolution(50)
            sphere.SetThetaResolution(50)
            
            appendFilter.AddInputConnection(sphere.GetOutputPort())
            appendFilter.Update()
            
        
        spheresMapper = vtk.vtkPolyDataMapper()
        spheresMapper.SetInputConnection(appendFilter.GetOutputPort())
        
        spheresActor = vtk.vtkActor()    
        spheresActor.SetMapper(spheresMapper)
        spheresActor.GetProperty().SetColor(color)
        
        return spheresActor

    def arrow(self,point1,point2,color):
        # Method to generate an arrow from point 1 to point 2
        
        arrowSource = vtk.vtkArrowSource()
        arrowSource.SetTipResolution(50)
        arrowSource.SetShaftResolution(50)
    
        length = numpy.linalg.norm(point2-point1)
        X = point2-point1
        if X[2]==0:
            Y = numpy.array([0,0,1])
        else:
            Y = numpy.array([1,1,-(X[0]+X[1])/X[2]])
        Z = numpy.cross(X,Y)
        
        normalizedX = X/numpy.linalg.norm(X)
        normalizedY = Y/numpy.linalg.norm(Y)
        normalizedZ = Z/numpy.linalg.norm(Z)
            
        # Create the direction cosine matrix
        matrix = vtk.vtkMatrix4x4()
        matrix.Identity()
        for i in range(3):
            matrix.SetElement(i, 0, normalizedX[i])
            matrix.SetElement(i, 1, normalizedY[i])
            matrix.SetElement(i, 2, normalizedZ[i])    
        
        transform = vtk.vtkTransform()
        transform.Translate(point1)
        transform.Concatenate(matrix)
        transform.Scale(length, length, length)    
        
        arrowMapper = vtk.vtkPolyDataMapper()
        arrowMapper.SetInputConnection(arrowSource.GetOutputPort())
        
        arrowActor = vtk.vtkActor()
        arrowActor.SetUserMatrix(transform.GetMatrix())
        arrowActor.SetMapper(arrowMapper)
        
        arrowActor.GetProperty().SetColor(color)
        
        return arrowActor

    def transform(self,vec,theta,t):
        return numpy.dot(self.computeRotation(theta),vec)+t
         
         
    def vtkVisualize(self,DIP_flex,PIP_flex,MCP_flex,MCP_abd,extLoads,  \
                            plot_muscle = True,
                            plot_stl = True,
                            plot_colBar = True,
                            plot_RF = False,
                            plot_axes = True,
                            stl_opacity = 1.0,
                            scale_muscle_activation = False,
                            scale_muscle_pcsa = False,
                            snapshot=False,
                            view_roll=0.,
                            forceScale = 0.002,
                            use_F_mus=[],
                            use_F_JR=[],
                            windowInteractor=True,
                            returnRenderWindowInt=False,
                            customActor=[]):

        # Pathpoints
        DIPPointsProx = self.DIPPathPoints[:,3:6]*self.O2O3 
        DIPPointsDist = self.DIPPathPoints[:,0:3]*self.O2O3 
        PIPPointsProx = self.PIPPathPoints[:,3:6]*self.O2O3
        PIPPointsDist = self.PIPPathPoints[:,0:3]*self.O2O3
        MCPPointsProx = self.MCPPathPoints[:,3:6]*self.O2O3
        MCPPointsDist = self.MCPPathPoints[:,0:3]*self.O2O3        
                
        # compute muscle force limits (based on PCSA)
        F_mus_limit = numpy.array([self.PCSARI,self.PCSALU,self.PCSAUI,self.PCSAFDP,self.PCSAFDS,\
                                self.PCSALE,self.PCSAES,self.PCSATE,self.PCSARB,self.PCSAUB]) * self.specTension       
        
        # compute muscle forces
        if use_F_mus==[]:
            F_mus = self.computeMuscleForces(DIP_flex,PIP_flex,MCP_flex,MCP_abd,extLoads)
        else:
            if use_F_mus.shape[0]==6:
                # muscle force order RI - LU - UI - FDP - FDS - LE - ES - TE - RB - UB
                F_mus_new = numpy.zeros(10)
                F_mus_new[:6] = use_F_mus
                F_mus_new[6]  = self.ES_Ratios[0]*F_mus_new[0]  + self.ES_Ratios[2]*F_mus_new[2] + \
                                self.ES_Ratios[5]*F_mus_new[5] + self.ES_Ratios[1]*F_mus_new[1]
                F_mus_new[8]  = self.RB_Ratios[0]*F_mus_new[0] + self.RB_Ratios[5]*F_mus_new[5] + \
                                self.RB_Ratios[1]*F_mus_new[1]
                F_mus_new[9]  = self.UB_Ratios[2]*F_mus_new[2] + self.UB_Ratios[5]*F_mus_new[5]
                F_mus_new[7]  = self.TE_Ratios[8]*F_mus_new[8] + self.TE_Ratios[9]*F_mus_new[9] + \
                                self.TE_Ratios[5]*F_mus_new[5]
                F_mus = F_mus_new
            elif use_F_mus.shape[0]==10:
                F_mus = use_F_mus  
            
        
        # compute joint reaction loads
        if use_F_JR==[]:
            F_JR = self.computeJointReaction(DIP_flex,PIP_flex,MCP_flex,MCP_abd,extLoads,inGlobal=True,
                                             use_F_mus=F_mus)
        else:
            F_JR = use_F_JR
                        
        # compute muscle activation
        if scale_muscle_activation:
            a_mus = F_mus/F_mus_limit
        else:
            a_mus = numpy.zeros(F_mus_limit.shape)
        
        if scale_muscle_pcsa:
            F_mus_limit_rel = (F_mus_limit/numpy.max(F_mus_limit))**(0.5)
        else:
            F_mus_limit_rel = numpy.ones(F_mus_limit.shape)*0.3
                
        # kinematic chain
        MCPLoc = numpy.array([0.,0.,0.])
        PIPLoc = numpy.array([-(self.O4O5+self.O5O6),0.,0.])
        DIPLoc = numpy.array([-(self.O2O3+self.O3O4),0.,0.])
        FLoc   = numpy.array([-(self.O0O1+self.O1O2),0.,0.])
        
        kinPoints = numpy.zeros([4,3])
        kinPoints[0,:] = MCPLoc
        kinPoints[1,:] = self.transform(PIPLoc,\
                        numpy.array([0,self.deg2rad(MCP_abd),self.deg2rad(MCP_flex)]),
                        numpy.array([0,0,0]))
        kinPoints[2,:] = self.transform(DIPLoc,\
                        numpy.array([0,self.deg2rad(MCP_abd),self.deg2rad(MCP_flex+PIP_flex)]),
                        kinPoints[1,:])        
        kinPoints[3,:] = self.transform(FLoc,\
                        numpy.array([0,self.deg2rad(MCP_abd),self.deg2rad(MCP_flex+PIP_flex+DIP_flex)]),
                        kinPoints[2,:])                  
        
        kinTubeActor = self.polyTube(kinPoints,0.0005,[0,0,0])
        kinSphereActor = self.polySpheres(kinPoints,0.001,[0,0,0])
        
        # force vectors
        forceArrowActors = []
        for i in range(len(extLoads.DP_Forces)):
            F_ext = extLoads.DP_Forces[i,:]
            forceEnd   = self.transform(extLoads.DP_Points[i,:],numpy.array([0,self.deg2rad(MCP_abd),self.deg2rad(MCP_flex+PIP_flex+DIP_flex)]),
                        kinPoints[2,:])
            forceStart = forceEnd-F_ext * forceScale
            forceArrowActors.append(self.arrow(forceStart,forceEnd,[1,0,0]))
        
        for i in range(len(extLoads.MP_Forces)):
            F_ext = extLoads.MP_Forces[i,:]
            forceEnd   = self.transform(extLoads.MP_Points[i,:],numpy.array([0,self.deg2rad(MCP_abd),self.deg2rad(MCP_flex+PIP_flex)]),
                        kinPoints[1,:])
            forceStart = forceEnd-F_ext * forceScale
            forceArrowActors.append(self.arrow(forceStart,forceEnd,[1,0,0]))
            
        for i in range(len(extLoads.PP_Forces)):
            F_ext = extLoads.PP_Forces[i,:]
            forceEnd   = self.transform(extLoads.PP_Points[i,:],numpy.array([0,self.deg2rad(MCP_abd),self.deg2rad(MCP_flex)]),
                        kinPoints[0,:])
            forceStart = forceEnd-F_ext * forceScale
            forceArrowActors.append(self.arrow(forceStart,forceEnd,[1,0,0]))
        
        # colorbar legend
        luTable = vtk.vtkLookupTable()
        luTable.SetTableRange(0,1)
        luTable.SetNumberOfTableValues(255)
        luTable.SetHueRange (1,1);
        luTable.SetSaturationRange (0.1,1);
        luTable.SetValueRange (1, 1);
        luTable.Build()
        
        colBar = vtk.vtkScalarBarActor()
        colBar.SetTitle('Activation')
        colBar.SetNumberOfLabels(4)
        colBar.SetLookupTable(luTable)
        colBar.GetTitleTextProperty().SetColor(0,0,0)
        colBar.GetLabelTextProperty().SetColor(0,0,0)
        colBar.GetLabelTextProperty().SetFontSize(1)
        colBar.SetWidth(0.1)
        colBar.SetHeight(0.3)
        

        # RI
        RI_Points = numpy.zeros([2,3])
        RI_MCPProxLoc = MCPPointsProx[2,:]
        RI_MCPDistLoc = MCPPointsDist[2,:]-numpy.array([self.O5O6,0,0])
        
        
        RI_Points[0,:] = RI_MCPProxLoc
        RI_Points[1,:] = self.transform(RI_MCPDistLoc,\
                                   numpy.array([0,self.deg2rad(MCP_abd),self.deg2rad(MCP_flex)]),\
                                   numpy.array([0,0,0]))
        
        RI_TubeActor = self.polyTube(RI_Points,F_mus_limit_rel[0]*0.002,luTable.GetTableValue(int(a_mus[0]*255))[0:3])
        RI_SphereActor = self.polySpheres(RI_Points,F_mus_limit_rel[0]*0.0025,luTable.GetTableValue(int(a_mus[0]*255))[0:3])
        
        # LU
        LU_Points = numpy.zeros([2,3])
        LU_MCPProxLoc = MCPPointsProx[3,:]
        LU_MCPDistLoc = MCPPointsDist[3,:]-numpy.array([self.O5O6,0,0])
        
        
        LU_Points[0,:] = LU_MCPProxLoc
        LU_Points[1,:] = self.transform(LU_MCPDistLoc,\
                                   numpy.array([0,self.deg2rad(MCP_abd),self.deg2rad(MCP_flex)]),\
                                   numpy.array([0,0,0]))
        
        LU_TubeActor = self.polyTube(LU_Points,F_mus_limit_rel[1]*0.002,luTable.GetTableValue(int(a_mus[1]*255))[0:3])
        LU_SphereActor = self.polySpheres(LU_Points,F_mus_limit_rel[1]*0.0025,luTable.GetTableValue(int(a_mus[1]*255))[0:3])
        
        
        # UI
        UI_Points = numpy.zeros([2,3])
        UI_MCPProxLoc = MCPPointsProx[4,:]
        UI_MCPDistLoc = MCPPointsDist[4,:]-numpy.array([self.O5O6,0,0])
        
        
        UI_Points[0,:] = UI_MCPProxLoc
        UI_Points[1,:] = self.transform(UI_MCPDistLoc,\
                                   numpy.array([0,self.deg2rad(MCP_abd),self.deg2rad(MCP_flex)]),\
                                   numpy.array([0,0,0]))
                 
        UI_TubeActor = self.polyTube(UI_Points,F_mus_limit_rel[2]*0.002,luTable.GetTableValue(int(a_mus[2]*255))[0:3])
        UI_SphereActor = self.polySpheres(UI_Points,F_mus_limit_rel[2]*0.0025,luTable.GetTableValue(int(a_mus[2]*255))[0:3])
             

        # FDS
        FDS_Points = numpy.zeros([4,3])
        FDS_MCPProxLoc = MCPPointsProx[1,:]
        FDS_MCPDistLoc = MCPPointsDist[1,:]-numpy.array([self.O5O6,0,0])
        FDS_PIPProxLoc = PIPPointsProx[3,:]-numpy.array([self.O4O5+self.O5O6,0,0])
        FDS_PIPDistLoc = PIPPointsDist[3,:]-numpy.array([self.O3O4,0,0])
        
        FDS_Points[0,:] = FDS_MCPProxLoc
        FDS_Points[1,:] = self.transform(FDS_MCPDistLoc,\
                                   numpy.array([0,self.deg2rad(MCP_abd),self.deg2rad(MCP_flex)]),\
                                   numpy.array([0,0,0]))      
        FDS_Points[2,:] = self.transform(FDS_PIPProxLoc,\
                                   numpy.array([0,self.deg2rad(MCP_abd),self.deg2rad(MCP_flex)]),\
                                   numpy.array([0,0,0])) 
        FDS_Points[3,:] = self.transform(FDS_PIPDistLoc,\
                                   numpy.array([0,self.deg2rad(MCP_abd),self.deg2rad(MCP_flex+PIP_flex)]),\
                                   kinPoints[1,:])                               
                                   
        FDS_TubeActor = self.polyTube(FDS_Points,F_mus_limit_rel[3]*0.002,luTable.GetTableValue(int(a_mus[4]*255))[0:3])
        FDS_SphereActor = self.polySpheres(FDS_Points,F_mus_limit_rel[3]*0.0025,luTable.GetTableValue(int(a_mus[4]*255))[0:3])
             
        # FDP
        FDP_Points = numpy.zeros([6,3])
        FDP_MCPProxLoc = MCPPointsProx[0,:]
        FDP_MCPDistLoc = MCPPointsDist[0,:]-numpy.array([self.O5O6,0,0])
        FDP_PIPProxLoc = PIPPointsProx[0,:]-numpy.array([self.O4O5+self.O5O6,0,0])
        FDP_PIPDistLoc = PIPPointsDist[0,:]-numpy.array([self.O3O4,0,0])
        FDP_DIPProxLoc = DIPPointsProx[1,:]-numpy.array([self.O2O3+self.O3O4,0,0])
        FDP_DIPDistLoc = DIPPointsDist[1,:]-numpy.array([self.O1O2,0,0])
        
        FDP_Points[0,:] = FDP_MCPProxLoc
        FDP_Points[1,:] = self.transform(FDP_MCPDistLoc,\
                                   numpy.array([0,self.deg2rad(MCP_abd),self.deg2rad(MCP_flex)]),\
                                   numpy.array([0,0,0]))      
        FDP_Points[2,:] = self.transform(FDP_PIPProxLoc,\
                                   numpy.array([0,self.deg2rad(MCP_abd),self.deg2rad(MCP_flex)]),\
                                   numpy.array([0,0,0])) 
        FDP_Points[3,:] = self.transform(FDP_PIPDistLoc,\
                                   numpy.array([0,self.deg2rad(MCP_abd),self.deg2rad(MCP_flex+PIP_flex)]),\
                                   kinPoints[1,:])    
        FDP_Points[4,:] = self.transform(FDP_DIPProxLoc,\
                                   numpy.array([0,self.deg2rad(MCP_abd),self.deg2rad(MCP_flex+PIP_flex)]),\
                                   kinPoints[1,:])     
        FDP_Points[5,:] = self.transform(FDP_DIPDistLoc,\
                                   numpy.array([0,self.deg2rad(MCP_abd),self.deg2rad(MCP_flex+PIP_flex+DIP_flex)]),\
                                   kinPoints[2,:])                             
                                   
        FDP_TubeActor = self.polyTube(FDP_Points,F_mus_limit_rel[3]*0.002,luTable.GetTableValue(int(a_mus[3]*255))[0:3])
        FDP_SphereActor = self.polySpheres(FDP_Points,F_mus_limit_rel[3]*0.0025,luTable.GetTableValue(int(a_mus[3]*255))[0:3])
        
    
        # LE
        LE_Points = numpy.zeros([2,3])
        LE_MCPProxLoc = MCPPointsProx[5,:]
        LE_MCPDistLoc = MCPPointsDist[5,:]-numpy.array([self.O5O6,0,0])
        
        
        LE_Points[0,:] = LE_MCPProxLoc
        LE_Points[1,:] = self.transform(LE_MCPDistLoc,\
                                   numpy.array([0,self.deg2rad(MCP_abd),self.deg2rad(MCP_flex)]),\
                                   numpy.array([0,0,0]))      
                                   
        LE_TubeActor = self.polyTube(LE_Points,F_mus_limit_rel[5]*0.002,luTable.GetTableValue(int(a_mus[5]*255))[0:3])
        LE_SphereActor = self.polySpheres(LE_Points,F_mus_limit_rel[5]*0.0025,luTable.GetTableValue(int(a_mus[5]*255))[0:3])

        # ES
        ES_Points = numpy.zeros([2,3])
        ES_PIPProxLoc = PIPPointsProx[4,:]-numpy.array([self.O4O5+self.O5O6,0,0])
        ES_PIPDistLoc = PIPPointsDist[4,:]-numpy.array([self.O3O4,0,0])
        
        ES_Points[0,:] = self.transform(ES_PIPProxLoc,\
                                   numpy.array([0,self.deg2rad(MCP_abd),self.deg2rad(MCP_flex)]),\
                                   numpy.array([0,0,0])) 
        ES_Points[1,:] = self.transform(ES_PIPDistLoc,\
                                   numpy.array([0,self.deg2rad(MCP_abd),self.deg2rad(MCP_flex+PIP_flex)]),\
                                   kinPoints[1,:])    
        ES_TubeActor = self.polyTube(ES_Points,F_mus_limit_rel[6]*0.002,luTable.GetTableValue(int(a_mus[6]*255))[0:3])
        ES_SphereActor = self.polySpheres(ES_Points,F_mus_limit_rel[6]*0.0025,luTable.GetTableValue(int(a_mus[6]*255))[0:3])


        # TE
        TE_Points = numpy.zeros([2,3])
        TE_DIPProxLoc = DIPPointsProx[0,:]-numpy.array([self.O2O3+self.O3O4,0,0])
        TE_DIPDistLoc = DIPPointsDist[0,:]-numpy.array([self.O1O2,0,0])
         
        TE_Points[0,:] = self.transform(TE_DIPProxLoc,\
                                   numpy.array([0,self.deg2rad(MCP_abd),self.deg2rad(MCP_flex+PIP_flex)]),\
                                   kinPoints[1,:])     
        TE_Points[1,:] = self.transform(TE_DIPDistLoc,\
                                   numpy.array([0,self.deg2rad(MCP_abd),self.deg2rad(MCP_flex+PIP_flex+DIP_flex)]),\
                                   kinPoints[2,:])                             
                                   
        TE_TubeActor = self.polyTube(TE_Points,F_mus_limit_rel[7]*0.002,luTable.GetTableValue(int(a_mus[7]*255))[0:3])
        TE_SphereActor = self.polySpheres(TE_Points,F_mus_limit_rel[7]*0.0025,luTable.GetTableValue(int(a_mus[7]*255))[0:3])
        
        
        # RB
        RB_Points = numpy.zeros([2,3])
        RB_PIPProxLoc = PIPPointsProx[1,:]-numpy.array([self.O4O5+self.O5O6,0,0])
        RB_PIPDistLoc = PIPPointsDist[1,:]-numpy.array([self.O3O4,0,0])
          
        RB_Points[0,:] = self.transform(RB_PIPProxLoc,\
                                   numpy.array([0,self.deg2rad(MCP_abd),self.deg2rad(MCP_flex)]),\
                                   numpy.array([0,0,0])) 
        RB_Points[1,:] = self.transform(RB_PIPDistLoc,\
                                   numpy.array([0,self.deg2rad(MCP_abd),self.deg2rad(MCP_flex+PIP_flex)]),\
                                   kinPoints[1,:])                            
                                   
        RB_TubeActor = self.polyTube(RB_Points,F_mus_limit_rel[8]*0.002,luTable.GetTableValue(int(a_mus[8]*255))[0:3])
        RB_SphereActor = self.polySpheres(RB_Points,F_mus_limit_rel[8]*0.0025,luTable.GetTableValue(int(a_mus[8]*255))[0:3])
        
        # UB
        UB_Points = numpy.zeros([2,3])
        UB_PIPProxLoc = PIPPointsProx[2,:]-numpy.array([self.O4O5+self.O5O6,0,0])
        UB_PIPDistLoc = PIPPointsDist[2,:]-numpy.array([self.O3O4,0,0])
          
        UB_Points[0,:] = self.transform(UB_PIPProxLoc,\
                                   numpy.array([0,self.deg2rad(MCP_abd),self.deg2rad(MCP_flex)]),\
                                   numpy.array([0,0,0])) 
        UB_Points[1,:] = self.transform(UB_PIPDistLoc,\
                                   numpy.array([0,self.deg2rad(MCP_abd),self.deg2rad(MCP_flex+PIP_flex)]),\
                                   kinPoints[1,:])                            
                                   
        UB_TubeActor = self.polyTube(UB_Points,F_mus_limit_rel[9]*0.002,luTable.GetTableValue(int(a_mus[9]*255))[0:3])
        UB_SphereActor = self.polySpheres(UB_Points,F_mus_limit_rel[9]*0.0025,luTable.GetTableValue(int(a_mus[9]*255))[0:3])
        
        # Extensor expansion
        EE_TE_Points = numpy.zeros([3,3])
        EE_TE_Points[0,:] = UB_Points[1,:]
        EE_TE_Points[1,:] = TE_Points[0,:]      
        EE_TE_Points[2,:] = RB_Points[1,:]                                     
        EE_TE_TubeActor = self.polyTube(EE_TE_Points,0.0005,[0,0,0])
        
        EE_RB_Points = numpy.zeros([5,3])
        EE_RB_Points[0,:] = LU_Points[1,:]
        EE_RB_Points[1,:] = RB_Points[0,:]      
        EE_RB_Points[2,:] = RI_Points[1,:]      
        EE_RB_Points[3,:] = RB_Points[0,:]    
        EE_RB_Points[4,:] = LE_Points[1,:]                           
        EE_RB_TubeActor = self.polyTube(EE_RB_Points,0.0005,[0,0,0])
        
        EE_UB_Points = numpy.zeros([3,3])
        EE_UB_Points[0,:] = UI_Points[1,:]
        EE_UB_Points[1,:] = UB_Points[0,:]      
        EE_UB_Points[2,:] = LE_Points[1,:]                                     
        EE_UB_TubeActor = self.polyTube(EE_UB_Points,0.0005,[0,0,0])
        
        EE_ES_Points = numpy.zeros([7,3])
        EE_ES_Points[0,:] = LU_Points[1,:]
        EE_ES_Points[1,:] = ES_Points[0,:]      
        EE_ES_Points[2,:] = RI_Points[1,:]
        EE_ES_Points[3,:] = ES_Points[0,:]  
        EE_ES_Points[4,:] = LE_Points[1,:]
        EE_ES_Points[5,:] = ES_Points[0,:]  
        EE_ES_Points[6,:] = UI_Points[1,:]                                     
        EE_ES_TubeActor = self.polyTube(EE_ES_Points,0.0005,[0,0,0])
        
        # generate axes
        axes = vtk.vtkAxesActor()
        axes.SetShaftTypeToCylinder()
        axes.AxisLabelsOff()
        axes.SetTotalLength(0.01,0.01,0.01)
        
        # plot joint loads
        F_MCP_inRef = F_JR[2,:]
        F_MCP_inRef_Start = kinPoints[0,:]+F_MCP_inRef * forceScale
        F_MCP_inRef_End = kinPoints[0,:]
        F_MCP_Actor = self.arrow(F_MCP_inRef_Start,F_MCP_inRef_End,[0,0,1])
        
        F_PIP_inRef = F_JR[1,:]
        F_PIP_inRef_Start = kinPoints[1,:]+F_PIP_inRef * forceScale
        F_PIP_inRef_End = kinPoints[1,:]
        F_PIP_Actor = self.arrow(F_PIP_inRef_Start,F_PIP_inRef_End,[0,0,1])
        
        F_DIP_inRef = F_JR[0,:]
        F_DIP_inRef_Start = kinPoints[2,:]+F_DIP_inRef * forceScale
        F_DIP_inRef_End = kinPoints[2,:]
        F_DIP_Actor = self.arrow(F_DIP_inRef_Start,F_DIP_inRef_End,[0,0,1])
        
        # read and transform stl files
        MC_stlActor = self.readSTL(self.MC_filename)
        MC_stlActor.GetProperty().SetOpacity(stl_opacity)
        
        PP_stlActor = self.readSTL(self.PP_filename)
        vtkMatrix = self.vtkTransMatrix(self.computeRotation(numpy.array([0,self.deg2rad(MCP_abd),self.deg2rad(MCP_flex)])),\
                                    kinPoints[1,:])
        PP_stlActor.SetUserMatrix(vtkMatrix)
        PP_stlActor.GetProperty().SetOpacity(stl_opacity)
        
        MP_stlActor = self.readSTL(self.MP_filename)
        vtkMatrix = self.vtkTransMatrix(self.computeRotation(numpy.array([0,self.deg2rad(MCP_abd),self.deg2rad(MCP_flex+PIP_flex)])),\
                                   kinPoints[2,:])
        MP_stlActor.SetUserMatrix(vtkMatrix)
        MP_stlActor.GetProperty().SetOpacity(stl_opacity)
        
        DP_stlActor = self.readSTL(self.DP_filename)
        vtkMatrix = self.vtkTransMatrix(self.computeRotation(numpy.array([0,self.deg2rad(MCP_abd),self.deg2rad(MCP_flex+PIP_flex+DIP_flex)])),\
                                   kinPoints[3,:])
        DP_stlActor.SetUserMatrix(vtkMatrix)
        DP_stlActor.GetProperty().SetOpacity(stl_opacity)
        
        # init renderer
        renderer = vtk.vtkRenderer()
        
        cam = renderer.GetActiveCamera()
        cam.Zoom(2.0)
        cam.Roll(view_roll)
        
        renderWindow = vtk.vtkRenderWindow()
        renderWindow.AddRenderer(renderer)
        renderWindow.SetSize(1000,1000)
        
        renderWindowInteractor = vtk.vtkRenderWindowInteractor()
        renderWindowInteractor.SetRenderWindow(renderWindow)
        style = vtk.vtkInteractorStyleTrackballCamera()
        renderWindowInteractor.SetInteractorStyle(style)
        
        renderer.SetBackground(1, 1, 1)
              
        # add actors
        if plot_axes:
            renderer.AddActor(axes)
        
        renderer.AddActor(kinTubeActor)
        renderer.AddActor(kinSphereActor)
        
        for forceArrowActor in forceArrowActors:
            renderer.AddActor(forceArrowActor)
        
        if plot_muscle:
            renderer.AddActor(RI_TubeActor)
            renderer.AddActor(RI_SphereActor)
            
            renderer.AddActor(LU_TubeActor)
            renderer.AddActor(LU_SphereActor)            
                
            renderer.AddActor(UI_TubeActor)
            renderer.AddActor(UI_SphereActor)
            
            renderer.AddActor(LE_TubeActor)
            renderer.AddActor(LE_SphereActor)
            
            renderer.AddActor(FDP_TubeActor)
            renderer.AddActor(FDP_SphereActor)
            
            renderer.AddActor(FDS_TubeActor)
            renderer.AddActor(FDS_SphereActor)
            
            renderer.AddActor(TE_TubeActor)
            renderer.AddActor(TE_SphereActor)
            
            renderer.AddActor(ES_TubeActor)
            renderer.AddActor(ES_SphereActor)
            
            renderer.AddActor(RB_TubeActor)
            renderer.AddActor(RB_SphereActor)
            
            renderer.AddActor(UB_TubeActor)
            renderer.AddActor(UB_SphereActor)

            renderer.AddActor(EE_TE_TubeActor)
            renderer.AddActor(EE_ES_TubeActor)
            renderer.AddActor(EE_RB_TubeActor)
            renderer.AddActor(EE_UB_TubeActor)
        
        if plot_stl:
            renderer.AddActor(MC_stlActor)
            renderer.AddActor(PP_stlActor)
            renderer.AddActor(MP_stlActor)
            renderer.AddActor(DP_stlActor)
        
        if plot_RF:
            renderer.AddActor(F_MCP_Actor)
            renderer.AddActor(F_PIP_Actor)
            renderer.AddActor(F_DIP_Actor)
            
        if customActor!=[]:
            renderer.AddActor(customActor)            
            
        if plot_colBar:
            renderer.AddActor2D(colBar)
            
        # start renderer
        renderWindow.Render()
        
        if windowInteractor:
            renderWindowInteractor.Start()
       
        
        if snapshot!=False:
            w2if = vtk.vtkWindowToImageFilter()
            w2if.SetInput(renderWindow)
            w2if.Update()
             
            writer = vtk.vtkPNGWriter()
            writer.SetFileName(snapshot)
            writer.SetInputData(w2if.GetOutput()) 
            
            writer.Write()
        
        if returnRenderWindowInt:
            return renderWindow,renderWindowInteractor
            