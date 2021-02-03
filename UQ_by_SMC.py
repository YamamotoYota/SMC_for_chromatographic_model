# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 11:33:53 2021

@author: Yota Yamamoto
"""
#Import the required libraries
import os #use to create a folder to save the results
import numpy as np #Numeric computation libraries (multi-dimensional arrays, statistical functions, etc.)
from scipy.integrate import odeint #Used to solve ordinary differential equations.
import matplotlib.pyplot as plt #used to draw graphs
import datetime #Used to get the time.
from tqdm import tqdm #Used to visualize progress.
import pandas as pd #Used to read and process data
from joblib import Parallel, delayed #Used to implement parallel computing
import pymc as pm #Used for MAP estimation and MCMC calculation
import numba #Used for JIT compilation
from scipy.stats import norm,uniform #Used to calculate probability density function

#Create a folder to automatically save the obtained data.
dirname = ".\\UQ_by_SMC" #Name of the folder (directory) you want to create
os.makedirs(dirname,exist_ok=False) #Create a folder

#Folder to save the process of SMC
dirnameHa = dirname+"\\Ha"
os.makedirs(dirnameHa,exist_ok=False)
dirnameHb = dirname+"\\Hb"
os.makedirs(dirnameHb,exist_ok=False)
dirnameKa = dirname+"\\Ka"
os.makedirs(dirnameKa,exist_ok=False)
dirnameKb = dirname+"\\Kb"
os.makedirs(dirnameKb,exist_ok=False)
dirnameba = dirname+"\\ba"
os.makedirs(dirnameba,exist_ok=False)
dirnamebb = dirname+"\\bb"
os.makedirs(dirnamebb,exist_ok=False)
dirnameObsSigma = dirname+"\\ObsSigma"
os.makedirs(dirnameObsSigma,exist_ok=False)

#Load data from Experiment A
SampleFile1 = pd.read_csv('Pseudo_A.csv').dropna(how="all")
tData1 = SampleFile1["t[s]"].values
CaOutData1 = SampleFile1["C[g/L]"].values

#Load data from Experiment B
SampleFile2 = pd.read_csv("Pseudo_B.csv").dropna(how="all")
tData2 = SampleFile2["t[s]"].values
CbOutData2 = SampleFile2["C[g/L]"].values

#Load data from Experiment C
SampleFile3 = pd.read_csv("Pseudo_C.csv").dropna(how="all")
tData3 = SampleFile3["t[s]"].values
CTotalOutData3 = SampleFile3["C[g/L]"].values

#Load data for validation of experiment C
SampleFile4 = pd.read_csv("Pseudo_C_deconvoluted.csv").dropna(how="all")
tData4 = SampleFile4["t[s]"].values
CaOutData4 = SampleFile4["Ca[g/L]"].values
CbOutData4 = SampleFile4["Cb[g/L]"].values

#Number of particles
NParticle = 10000
#Number of cores used in parallel computing
core = 14

#Calling and calculating constants needed for calculations
Ca01,Ca02,Ca03,Ca04 = SampleFile1["CA0[g/L]"].values[0], SampleFile2["CA0[g/L]"].values[0], SampleFile3["CA0[g/L]"].values[0], SampleFile4["CA0[g/L]"].values[0] #concentration of component a of feed [g/L]
Cb01,Cb02,Cb03,Cb04 = SampleFile1["CB0[g/L]"].values[0], SampleFile2["CB0[g/L]"].values[0], SampleFile3["CB0[g/L]"].values[0], SampleFile4["CB0[g/L]"].values[0] #concentration of component b of feed [g/L]
Vinject1,Vinject2,Vinject3,Vinject4 = SampleFile1["Vinject[uL]"].values[0]*1e-9, SampleFile2["Vinject[uL]"].values[0]*1e-9, SampleFile3["Vinject[uL]"].values[0]*1e-9, SampleFile4["Vinject[uL]"].values[0]*1e-9 #injection volume of feed [m^3]
Q1,Q2,Q3,Q4 = SampleFile1["Q[mL/min]"].values[0]*(1e-6/60), SampleFile2["Q[mL/min]"].values[0]*(1e-6/60), SampleFile3["Q[mL/min]"].values[0]*(1e-6/60), SampleFile4["Q[mL/min]"].values[0]*(1e-6/60) #flow of mobile phase [m^3/s]
N = 100 #number of discretizations in spatial direction
M1,M2,M3,M4 = len(tData1),len(tData2),len(tData3),len(tData4) #Number of data points
Z1,Z2,Z3,Z4 = SampleFile1["Z[m]"].values[0], SampleFile2["Z[m]"].values[0], SampleFile3["Z[m]"].values[0], SampleFile4["Z[m]"].values[0] #column length [m]
dz1,dz2,dz3,dz4 = Z1/N,Z2/N,Z3/N,Z4/N #length of differential column [m]
d1,d2,d3,d4 = SampleFile1["d[m]"].values[0], SampleFile2["d[m]"].values[0], SampleFile3["d[m]"].values[0], SampleFile4["d[m]"].values[0] #inner diameter of column [m]
S1,S2,S3,S4 = np.pi*d1*d1/4, np.pi*d2*d2/4, np.pi*d3*d3/4, np.pi*d4*d4/4 #cross-section area of column [m^2]
u1,u2,u3,u4 = Q1/S1,Q2/S2,Q3/S3,Q4/S4 #superficial velocity in column [m/s]
Vcolumn1,Vcolumn2,Vcolumn3,Vcolumn4 = S1*Z1,S2*Z2,S3*Z3,S4*Z4 #volume of column [m^3]
e1,e2,e3,e4 = SampleFile1["e[-]"].values[0], SampleFile2["e[-]"].values[0], SampleFile3["e[-]"].values[0], SampleFile4["e[-]"].values[0] #overall bed porosity　of column [-]
tf1,tf2,tf3,tf4 = Vinject1/Q1,Vinject2/Q2,Vinject3/Q3,Vinject4/Q4 #injection time [s]
CaExpMax1,CbExpMax2,CTotalExpMax3,CaExpMax4,CbExpMax4 = max(CaOutData1),max(CbOutData2),max(CTotalOutData3),max(CaOutData4),max(CbOutData4) #Maximum value of concentration in each data [g/L]

#Define the data set to be used for estimation
DATA123 = CaOutData1[0:M1]/CaExpMax1,CbOutData2[0:M2]/CbExpMax2,CTotalOutData3[0:M3]/CTotalExpMax3

#Approximate Henry's constant and mass transfer coefficient values calculated from moments.
Ha_int = SampleFile1["Hint[-]"].values[0]
Hb_int = SampleFile2["Hint[-]"].values[0]
Ka_int = SampleFile1["Kint[1/s]"].values[0]
Kb_int = SampleFile2["Kint[1/s]"].values[0]

#Upper and lower bounds on the uniform prior distribution used for MAP estimation
Ha_LOW = (0.5)*Ha_int
Ha_UP = (1.5)*Ha_int
Hb_LOW = (0.5)*Hb_int
Hb_UP = (1.5)*Hb_int
Ka_LOW = (0.1)*Ka_int
Ka_UP = 10*Ka_int
Kb_LOW = (0.1)*Kb_int
Kb_UP = 10*Kb_int
ba_LOW = 0
ba_UP = 0.001
bb_LOW = 0
bb_UP = 0.001
ObsSigma_LOW = 0.0001
ObsSigma_UP = 0.1

#initial conditions of the LDF model at all of x
f0 = np.zeros(4*N+1)

#JIT compile to speed up
@numba.jit("f8[:](f8[:],f8,f8[:],f8,f8,f8,f8,f8,f8)",nopython=True)
def func(x,t,p,tf,Ca0,Cb0,e,u,dz): #Calculating the Time Derivatives of C and q in the LDF Model
    Ha,Hb,Ka,Kb,ba,bb = p #Summarize the parameters as an array
    if x[-1]<=tf: #Conditional branching for injected concentration
        Cain,Cbin = Ca0,Cb0 #feed
    else:
        Cain,Cbin = 0,0 #solvent

    ret = np.zeros(4*N+1) #Prepare the return value as an array first
    dCadt = ret[:N]
    dCbdt = ret[N:2*N]
    dqadt = ret[2*N:3*N]
    dqbdt = ret[3*N:4*N]
    ret[4*N]=1.0

    Caeq = Hb*x[2*N:3*N]/(Ha*Hb+Ha*bb*x[3*N:4*N]+Hb*ba*x[2*N:3*N]) #anti-Langmuir isotherm for component a
    Cbeq = Ha*x[3*N:4*N]/(Ha*Hb+Ha*bb*x[3*N:4*N]+Hb*ba*x[2*N:3*N]) #anti-Langmuir isotherm for component b

    dqadt[:] = (Ka/(1-e))*(x[0:N]-Caeq) #Solid-phase mass balance of component a (liquid-phase basis)
    dqbdt[:] = (Kb/(1-e))*(x[N:2*N]-Cbeq) #Solid-phase mass balance of component b (liquid-phase basis)

    (udive,dz_mul_by2,E) = (u/e,dz*2,(1-e)/e) #Calculate the fixed values first.

    #Liquid-phase mass balance of component a
    dCadt[0] = -udive*(x[1]-Cain)/dz_mul_by2-E*dqadt[0] #Mass balance at the entrance of the column
    dCadt[1:N-1] =  -udive*(x[2:N]-x[0:N-2])/dz_mul_by2-E*dqadt[1:N-1] #mass balance in column (central differential)
    dCadt[N-1] = -udive*(x[N-1]-x[N-2])/dz - E*dqadt[N-1] #Mass balance at column exit (backward differential)

    #Liquid-phase mass balance of component b
    dCbdt[0] = -udive*(x[N+1]-Cbin)/dz_mul_by2 - E*dqbdt[0]
    dCbdt[1:N-1] = -udive*(x[N+2:2*N]-x[N:2*N-2])/dz_mul_by2-E*dqbdt[1:N-1]
    dCbdt[N-1] = -udive*(x[2*N-1]-x[2*N-2])/dz-E*dqbdt[N-1]

    return ret

#Functions to compute the LDF model
def PROCESSMODEL(Ha,Hb,Ka,Kb,ba,bb,f0,tspan,tf,Ca0,Cb0,e,u,dz):
    p=np.array([Ha,Hb,Ka,Kb,ba,bb])
    sol = odeint(func, f0, tspan, args=(p,tf,Ca0,Cb0,e,u,dz)) #Numerical computation of differential equations
    return sol

#Equivalent functions separately to calculate for each data
@numba.jit("f8[:](f8[:],f8,f8,f8,f8,f8,f8,f8)",nopython=True)
def func1(x,t,Ha,Hb,Ka,Kb,ba,bb):
    if x[-1]<=tf1:
        Cain,Cbin = Ca01,Cb01
    else:
        Cain,Cbin = 0,0
    ret = np.zeros(4*N+1)
    dCadt = ret[:N]
    dCbdt = ret[N:2*N]
    dqadt = ret[2*N:3*N]
    dqbdt = ret[3*N:4*N]
    ret[4*N]=1.0
    Caeq = Hb*x[2*N:3*N]/(Ha*Hb+Ha*bb*x[3*N:4*N]+Hb*ba*x[2*N:3*N])
    Cbeq = Ha*x[3*N:4*N]/(Ha*Hb+Ha*bb*x[3*N:4*N]+Hb*ba*x[2*N:3*N])
    dqadt[:] = (Ka/(1-e1))*(x[0:N]-Caeq)
    dqbdt[:] = (Kb/(1-e1))*(x[N:2*N]-Cbeq)
    (udive,dz_mul_by2,E) = (u1/e1,dz1*2,(1-e1)/e1)
    dCadt[0] = -udive*(x[1]-Cain)/dz_mul_by2-E*dqadt[0]
    dCadt[1:N-1] =  -udive*(x[2:N]-x[0:N-2])/dz_mul_by2-E*dqadt[1:N-1]
    dCadt[N-1] = -udive*(x[N-1]-x[N-2])/dz1 - E*dqadt[N-1]
    dCbdt[0] = -udive*(x[N+1]-Cbin)/dz_mul_by2 - E*dqbdt[0]
    dCbdt[1:N-1] = -udive*(x[N+2:2*N]-x[N:2*N-2])/dz_mul_by2-E*dqbdt[1:N-1]
    dCbdt[N-1] = -udive*(x[2*N-1]-x[2*N-2])/dz1-E*dqbdt[N-1]
    return ret

@numba.jit("f8[:](f8[:],f8,f8,f8,f8,f8,f8,f8)",nopython=True)
def func2(x,t,Ha,Hb,Ka,Kb,ba,bb):
    if x[-1]<=tf2:
        Cain,Cbin = Ca02,Cb02
    else:
        Cain,Cbin = 0,0
    ret = np.zeros(4*N+1)
    dCadt = ret[:N]
    dCbdt = ret[N:2*N]
    dqadt = ret[2*N:3*N]
    dqbdt = ret[3*N:4*N]
    ret[4*N]=1.0
    Caeq = Hb*x[2*N:3*N]/(Ha*Hb+Ha*bb*x[3*N:4*N]+Hb*ba*x[2*N:3*N])
    Cbeq = Ha*x[3*N:4*N]/(Ha*Hb+Ha*bb*x[3*N:4*N]+Hb*ba*x[2*N:3*N])
    dqadt[:] = (Ka/(1-e2))*(x[0:N]-Caeq)
    dqbdt[:] = (Kb/(1-e2))*(x[N:2*N]-Cbeq)
    (udive,dz_mul_by2,E) = (u2/e2,dz2*2,(1-e2)/e2)
    dCadt[0] = -udive*(x[1]-Cain)/dz_mul_by2-E*dqadt[0]
    dCadt[1:N-1] =  -udive*(x[2:N]-x[0:N-2])/dz_mul_by2-E*dqadt[1:N-1]
    dCadt[N-1] = -udive*(x[N-1]-x[N-2])/dz2 - E*dqadt[N-1]
    dCbdt[0] = -udive*(x[N+1]-Cbin)/dz_mul_by2 - E*dqbdt[0]
    dCbdt[1:N-1] = -udive*(x[N+2:2*N]-x[N:2*N-2])/dz_mul_by2-E*dqbdt[1:N-1]
    dCbdt[N-1] = -udive*(x[2*N-1]-x[2*N-2])/dz2-E*dqbdt[N-1]
    return ret

@numba.jit("f8[:](f8[:],f8,f8,f8,f8,f8,f8,f8)",nopython=True)
def func3(x,t,Ha,Hb,Ka,Kb,ba,bb):
    if x[-1]<=tf3:
        Cain,Cbin = Ca03,Cb03
    else:
        Cain,Cbin = 0,0
    ret = np.zeros(4*N+1)
    dCadt = ret[:N]
    dCbdt = ret[N:2*N]
    dqadt = ret[2*N:3*N]
    dqbdt = ret[3*N:4*N]
    ret[4*N]=1.0
    Caeq = Hb*x[2*N:3*N]/(Ha*Hb+Ha*bb*x[3*N:4*N]+Hb*ba*x[2*N:3*N])
    Cbeq = Ha*x[3*N:4*N]/(Ha*Hb+Ha*bb*x[3*N:4*N]+Hb*ba*x[2*N:3*N])
    dqadt[:] = (Ka/(1-e3))*(x[0:N]-Caeq)
    dqbdt[:] = (Kb/(1-e3))*(x[N:2*N]-Cbeq)
    (udive,dz_mul_by2,E) = (u3/e3,dz3*2,(1-e3)/e3)
    dCadt[0] = -udive*(x[1]-Cain)/dz_mul_by2-E*dqadt[0]
    dCadt[1:N-1] =  -udive*(x[2:N]-x[0:N-2])/dz_mul_by2-E*dqadt[1:N-1]
    dCadt[N-1] = -udive*(x[N-1]-x[N-2])/dz3 - E*dqadt[N-1]
    dCbdt[0] = -udive*(x[N+1]-Cbin)/dz_mul_by2 - E*dqbdt[0]
    dCbdt[1:N-1] = -udive*(x[N+2:2*N]-x[N:2*N-2])/dz_mul_by2-E*dqbdt[1:N-1]
    dCbdt[N-1] = -udive*(x[2*N-1]-x[2*N-2])/dz3-E*dqbdt[N-1]
    return ret

#Return the sum of squares of the normalized error between the simulation data and experimantal observed data for experiment A.
def PROCESSMODEL1(Ha,Hb,Ka,Kb,ba,bb):
    sol = odeint(func1, f0, tData1, args=(Ha,Hb,Ka,Kb,ba,bb))
    ERROR1 = np.zeros(M1)
    ERROR1[0:M1] = (CaOutData1[0:M1] - sol[0:M1,N-1])/CaExpMax1
    PoweredERROR1 = np.power(ERROR1,2)
    return np.sum(PoweredERROR1)

#Return the sum of squares of the normalized error between the simulation data and experimantal observed data for experiment B.
def PROCESSMODEL2(Ha,Hb,Ka,Kb,ba,bb):
    sol = odeint(func2, f0, tData2, args=(Ha,Hb,Ka,Kb,ba,bb))
    ERROR2 = np.zeros(M2)
    ERROR2[0:M2] = (CbOutData2[0:M2] - sol[0:M2,2*N-1])/CbExpMax2
    PoweredERROR2 = np.power(ERROR2,2)
    return np.sum(PoweredERROR2)

#Return the sum of squares of the normalized error between the simulation data and experimantal observed data for experiment C.
def PROCESSMODEL3(Ha,Hb,Ka,Kb,ba,bb):
    sol = odeint(func3, f0, tData3, args=(Ha,Hb,Ka,Kb,ba,bb))
    ERROR3 = np.zeros(M3)
    ERROR3[0:M3] = (CTotalOutData3[0:M3] - (sol[0:M3,N-1] + sol[0:M3,2*N-1]))/CTotalExpMax3
    PoweredERROR3 = np.power(ERROR3,2)
    return np.sum(PoweredERROR3)

#Record and display the time when the UQ was started.
StartDateTime = datetime.datetime.now()
print("UQ is started at {}".format(StartDateTime))

#prior distributions for MAP estimation
Ham = pm.Uniform("Ha",lower=Ha_LOW,upper=Ha_UP)
Hbm = pm.Uniform("Hb",lower=Hb_LOW,upper=Hb_UP)
Kam = pm.Uniform("Ka",lower=Ka_LOW,upper=Ka_UP)
Kbm = pm.Uniform("Kb",lower=Kb_LOW,upper=Kb_UP)
bam = pm.Uniform("ba",lower=ba_LOW,upper=ba_UP)
bbm = pm.Uniform("bb",lower=bb_LOW,upper=bb_UP)
ObsSigmam = pm.Uniform("ObsSigma",lower=ObsSigma_LOW,upper=ObsSigma_UP)

#Calculate the mean value of the likelihood function assumed to be normally distributed
@pm.deterministic
def mu(Ham=Ham,Hbm=Hbm,Kam=Kam,Kbm=Kbm,bam=bam,bbm=bbm):
    Calc1 = PROCESSMODEL(Ham,Hbm,Kam,Kbm,bam,bbm,f0,tData1,tf1,Ca01,Cb01,e1,u1,dz1)
    Calc2 = PROCESSMODEL(Ham,Hbm,Kam,Kbm,bam,bbm,f0,tData2,tf2,Ca02,Cb02,e2,u2,dz2)
    Calc3 = PROCESSMODEL(Ham,Hbm,Kam,Kbm,bam,bbm,f0,tData3,tf3,Ca03,Cb03,e3,u3,dz3)
    z1 = Calc1[0:M1,N-1]
    z2 = Calc2[0:M2,2*N-1]
    z3 = Calc3[0:M3,N-1] + Calc3[0:M3,2*N-1]
    z123 = z1/CaExpMax1,z2/CbExpMax2,z3/CTotalExpMax3
    return z123

A = pm.Normal("A", mu=mu, tau=1/(ObsSigmam**2), value=DATA123, observed=True) #define likelihood function
model = pm.Model([Ham,Hbm,Kam,Kbm,bam,bbm,ObsSigmam]) #MCMC model definition
map_ = pm.MAP(model) #Define the model for MAP estimation
map_.fit() #Perform MAP estimation
HaMAP,HbMAP,KaMAP,KbMAP,baMAP,bbMAP,ObsSigmaMAP = 1*Ham.value,1*Hbm.value,1*Kam.value,1*Kbm.value,1*bam.value,1*bbm.value,abs(1*ObsSigmam.value)
print(HaMAP,HbMAP,KaMAP,KbMAP,baMAP,bbMAP,ObsSigmaMAP)

#Value of how many times the standard deviation of the MAP solution should be when using the normal distribution for the prior distribution of UQ.
cHa,cHb,cKa,cKb,cba,cbb,cObsSigma = 0.05,0.05,0.05,0.05,0.1,0.1,0.1
#prior distribution for UQ
#Multivariate probability density function for calculating the value of prior probability at any parameter value.
def P0(Parameters):
    p0Ha = norm.pdf(Parameters[0,:], HaMAP, cHa*HaMAP)
    p0Hb = norm.pdf(Parameters[1,:], HbMAP, cHb*HbMAP)
    p0Ka = norm.pdf(Parameters[2,:], KaMAP, cKa*KaMAP)
    p0Kb = norm.pdf(Parameters[3,:], KbMAP, cKb*KbMAP)
    p0ba = uniform.pdf(Parameters[4,:], ba_LOW, ba_UP-ba_LOW)
    p0bb = uniform.pdf(Parameters[5,:], bb_LOW, bb_UP-bb_LOW)
    p0ObsSigma = norm.pdf(Parameters[6,:], ObsSigmaMAP, cObsSigma*ObsSigmaMAP)
    return p0Ha*p0Hb*p0Ka*p0Kb*p0ba*p0bb*p0ObsSigma
#Produce initial particles that follow prior distributions
HaDist = np.random.normal(loc=HaMAP,scale=cHa*HaMAP,size=NParticle)
HbDist = np.random.normal(loc=HbMAP,scale=cHb*HbMAP,size=NParticle)
KaDist = np.random.normal(loc=KaMAP,scale=cKa*KaMAP,size=NParticle)
KbDist = np.random.normal(loc=KbMAP,scale=cKb*KbMAP,size=NParticle)
baDist = np.random.uniform(low=ba_LOW,high=ba_UP,size=NParticle)
bbDist = np.random.uniform(low=bb_LOW,high=bb_UP,size=NParticle)
ObsSigmaDist = np.random.normal(loc=ObsSigmaMAP,scale=cObsSigma*ObsSigmaMAP,size=NParticle)

#Empty arrays, etc. required for calculations
ParameterDist = np.vstack((HaDist,HbDist,KaDist,KbDist,baDist,bbDist,ObsSigmaDist))
ParameterDistProp = 1*ParameterDist
Total_ERROR2 = np.zeros(NParticle)
Weight = np.empty_like(HaDist)
WeightProp = np.empty_like(HaDist)
l_Weight = np.empty_like(HaDist)
l_WeightProp = np.empty_like(HaDist)
psi = np.empty_like(HaDist)
Ratio = np.empty_like(HaDist)

#Value of γ at each step of likelihood tempering
Gamma = np.array([0,0.001,0.004,0.01,0.03,0.06,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0])

#SMC loops
for t in tqdm(range(1,len(Gamma))):
    print(t)
    print(Gamma[t])
    Parameters = 1*ParameterDist
    #Numeric sequence for each parameter
    EstimatedHaDist = Parameters[0,:]
    EstimatedHbDist = Parameters[1,:]
    EstimatedKaDist = Parameters[2,:]
    EstimatedKbDist = Parameters[3,:]
    EstimatedbaDist = Parameters[4,:]
    EstimatedbbDist = Parameters[5,:]
    EstimatedObsSigmaDist = Parameters[6,:]

    plt.rcParams["font.family"] ="Arial"
    plt.rcParams["font.size"] = 18
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"
    plt.rcParams["figure.figsize"] = (7,7)

    #Draw a histogram to show the distribution trend.
    NbinHa = np.linspace(np.percentile(EstimatedHaDist, 0), np.percentile(EstimatedHaDist, 100), num = 30) #Create an array to set the range and number of bins for the histogram
    BinWidthHa = NbinHa[1] - NbinHa[0] #Calculate the width of the bin
    fig,ax=plt.subplots() #Prepare a canvas for drawing graphs
    freqHa, binnHa, _Ha = ax.hist(EstimatedHaDist, bins=NbinHa, range=None, density=True,color="0.2") #draw a histogram
    HaMode = NbinHa[np.argmax(freqHa)] + BinWidthHa/2 #Calculate the mode of the parameter
    plt.xlabel("$\mathrm{{\it H_{Glu}}\ [-]}$",size=25) #axis label
    plt.ylabel("$\mathrm{{\it P(H_{Glu})}\ [-]}$",size=25) #axis label
    ax.ticklabel_format(axis="both", style="sci",scilimits=(0,0),useMathText=True) #format of axis label
    plt.savefig(dirnameHa + "\\Ha_gamma={:.4f}.png".format(Gamma[t-1]), bbox_inches="tight") #Save the graph to a folder
    plt.show() #Show graph
    #same for each parameter below
    NbinHb = np.linspace(np.percentile(EstimatedHbDist, 0), np.percentile(EstimatedHbDist, 100), num = 30)
    BinWidthHb = NbinHb[1] - NbinHb[0]
    fig,ax=plt.subplots()
    freqHb, binnHb, _Hb = ax.hist(EstimatedHbDist, bins=NbinHb, range=None, density=True,color="0.2")
    HbMode = NbinHb[np.argmax(freqHb)] + BinWidthHb/2
    plt.xlabel("$\mathrm{{\it H_{Fru}}\ [-]}$",size=25)
    plt.ylabel("$\mathrm{{\it P(H_{Fru})}\ [-]}$",size=25)
    ax.ticklabel_format(axis="both", style="sci",scilimits=(0,0),useMathText=True)
    plt.savefig(dirnameHb + "\\Hb_gamma={:.4f}.png".format(Gamma[t-1]), bbox_inches="tight")
    plt.show()
    NbinKa = np.linspace(np.percentile(EstimatedKaDist, 0), np.percentile(EstimatedKaDist, 100), num = 30)
    BinWidthKa = NbinKa[1] - NbinKa[0]
    fig,ax=plt.subplots()
    freqKa, binnKa, _Ka = ax.hist(EstimatedKaDist, bins=NbinKa, range=None, density=True,color="0.2")
    KaMode = NbinKa[np.argmax(freqKa)] + BinWidthKa/2
    plt.xlabel("$\mathrm{{\it K_{Glu}}\ [s^{-1}]}$",size=25)
    plt.ylabel("$\mathrm{{\it P(K_{Glu})}\ [s]}$",size=25)
    ax.ticklabel_format(axis="both", style="sci",scilimits=(0,0),useMathText=True)
    plt.savefig(dirnameKa + "\\Ka_gamma={:.4f}.png".format(Gamma[t-1]), bbox_inches="tight")
    plt.show()
    NbinKb = np.linspace(np.percentile(EstimatedKbDist, 0), np.percentile(EstimatedKbDist, 100), num = 30)
    BinWidthKb = NbinKb[1] - NbinKb[0]
    fig,ax=plt.subplots()
    freqKb, binnKb, _Kb = ax.hist(EstimatedKbDist, bins=NbinKb, range=None, density=True,color="0.2")
    KbMode = NbinKb[np.argmax(freqKb)] + BinWidthKb/2
    plt.xlabel("$\mathrm{{\it K_{Fru}}\ [s^{-1}]}$",size=25)
    plt.ylabel("$\mathrm{{\it P(K_{Fru})}\ [s]}$",size=25)
    ax.ticklabel_format(axis="both", style="sci",scilimits=(0,0),useMathText=True)
    plt.savefig(dirnameKb + "\\kb_gamma={:.4f}.png".format(Gamma[t-1]), bbox_inches="tight")
    plt.show()
    Nbinba = np.linspace(np.percentile(EstimatedbaDist, 0), np.percentile(EstimatedbaDist, 100), num = 30)
    BinWidthba = Nbinba[1] - Nbinba[0]
    fig,ax=plt.subplots()
    freqba, binnba, _ba = ax.hist(EstimatedbaDist, bins=Nbinba, range=None, density=True,color="0.2")
    baMode = Nbinba[np.argmax(freqba)] + BinWidthba/2
    plt.xlabel("$\mathrm{{\it b_{Glu}}\ [L/g]}$",size=25)
    plt.ylabel("$\mathrm{{\it P(b_{Glu})}\ [g/L]}$",size=25)
    ax.ticklabel_format(axis="both", style="sci",scilimits=(0,0),useMathText=True)
    plt.savefig(dirnameba + "\\ba_gamma={:.4f}.png".format(Gamma[t-1]), bbox_inches="tight")
    plt.show()
    Nbinbb = np.linspace(np.percentile(EstimatedbbDist, 0), np.percentile(EstimatedbbDist, 100), num = 30)
    BinWidthbb = Nbinbb[1] - Nbinbb[0]
    fig,ax=plt.subplots()
    freqbb, binnbb, _bb = ax.hist(EstimatedbbDist, bins=Nbinbb, range=None, density=True,color="0.2")
    bbMode = Nbinbb[np.argmax(freqbb)] + BinWidthbb/2
    plt.xlabel("$\mathrm{{\it b_{Fru}}\ [L/g]}$",size=25)
    plt.ylabel("$\mathrm{{\it P(b_{Fru})}\ [g/L]}$",size=25)
    ax.ticklabel_format(axis="both", style="sci",scilimits=(0,0),useMathText=True)
    plt.savefig(dirnamebb + "\\bb_gamma={:.4f}.png".format(Gamma[t-1]), bbox_inches="tight")
    plt.show()
    NbinObsSigma = np.linspace(np.percentile(EstimatedObsSigmaDist, 0), np.percentile(EstimatedObsSigmaDist, 100), num = 30)
    BinWidthObsSigma = NbinObsSigma[1] - NbinObsSigma[0]
    fig,ax=plt.subplots()
    freqObsSigma, binnObsSigma, _ObsSigma = ax.hist(EstimatedObsSigmaDist, bins=NbinObsSigma, range=None, density=True,color="0.2")
    ObsSigmaMode = NbinObsSigma[np.argmax(freqObsSigma)] + BinWidthObsSigma/2
    plt.xlabel("$\mathrm{{\it \sigma}\ [-]}$",size=25)
    plt.ylabel("$\mathrm{{\it P(\sigma)}\ [-]}$",size=25)
    ax.ticklabel_format(axis="both", style="sci",scilimits=(0,0),useMathText=True)
    plt.savefig(dirnameObsSigma + "\\ObsSigma_gamma={:.4f}.png".format(Gamma[t-1]), bbox_inches="tight")
    plt.show()

    ParameterDistProp = 1*ParameterDist
    #Skip just the first time of the loop.
    if t==1:
        pass
    else:
        VHa = np.random.normal(loc=0,scale=0.5*np.std(EstimatedHaDist),size=NParticle) #Determine the steps of a Markov chain using random numbers.
        VHb = np.random.normal(loc=0,scale=0.5*np.std(EstimatedHbDist),size=NParticle)
        VKa = np.random.normal(loc=0,scale=0.5*np.std(EstimatedKaDist),size=NParticle)
        VKb = np.random.normal(loc=0,scale=0.5*np.std(EstimatedKbDist),size=NParticle)
        Vba = np.random.normal(loc=0,scale=0.5*np.std(EstimatedbaDist),size=NParticle)
        Vbb = np.random.normal(loc=0,scale=0.5*np.std(EstimatedbbDist),size=NParticle)
        VObsSigma = np.random.normal(loc=0,scale=0.5*np.std(EstimatedObsSigmaDist),size=NParticle)

        #Compute candidate parameters.
        ParameterDistProp[0,0:NParticle] += VHa[0:NParticle]
        ParameterDistProp[1,0:NParticle] += VHb[0:NParticle]
        ParameterDistProp[2,0:NParticle] += VKa[0:NParticle]
        ParameterDistProp[3,0:NParticle] += VKb[0:NParticle]
        ParameterDistProp[4,0:NParticle] += Vba[0:NParticle]
        ParameterDistProp[5,0:NParticle] += Vbb[0:NParticle]
        ParameterDistProp[6,0:NParticle] += VObsSigma[0:NParticle]

    #arrays to store the sum of the squares of the errors of the observed data and the simulation data by each particle.
    Ha = ParameterDistProp[0,:]
    Hb = ParameterDistProp[1,:]
    Ka = ParameterDistProp[2,:]
    Kb = ParameterDistProp[3,:]
    ba = ParameterDistProp[4,:]
    bb = ParameterDistProp[5,:]
    ObsSigma = ParameterDistProp[6,:]
    #calculate sum of squares of the normalized error between the simulation data and experimantal observed data.
    POWEREDERROR1 = Parallel(n_jobs=core)( [delayed(PROCESSMODEL1)(Ha[P],Hb[P],Ka[P],Kb[P],ba[P],bb[P]) for P in range(NParticle)] )
    POWEREDERROR2 = Parallel(n_jobs=core)( [delayed(PROCESSMODEL2)(Ha[P],Hb[P],Ka[P],Kb[P],ba[P],bb[P]) for P in range(NParticle)] )
    POWEREDERROR3 = Parallel(n_jobs=core)( [delayed(PROCESSMODEL3)(Ha[P],Hb[P],Ka[P],Kb[P],ba[P],bb[P]) for P in range(NParticle)] )
    #joblib returns a Python list type, so convert it to ndarray to make it easier to handle.
    POWEREDERROR1 = np.array(POWEREDERROR1)
    POWEREDERROR2 = np.array(POWEREDERROR2)
    POWEREDERROR3 = np.array(POWEREDERROR3)
    #Lumping together the sum of the squares of the errors in each data set.
    Total_ERROR2[0:NParticle] = POWEREDERROR1[0:NParticle] + POWEREDERROR2[0:NParticle] + POWEREDERROR3[0:NParticle]
    #Calculate the log likelihood for each particle
    l_WeightProp[0:NParticle] = -((M1+M2+M3)/2)*np.log(2*np.pi*ObsSigma[0:NParticle]**2) - Total_ERROR2[0:NParticle]/(2*ObsSigma[0:NParticle]**2) #各粒子の対数尤度そのものを計算
    #Metropolis algorithm
    if t == 1: #Skip only the first time of the loop
        l_Weight = 1*l_WeightProp
        ParameterDist = 1*ParameterDistProp
    else:
        Ratio = ((np.exp(l_WeightProp - l_Weight))**Gamma[t-1])*(P0(ParameterDistProp)/P0(ParameterDist)) #Calculate the adoption rate (ratio of the posterior probability of the parameter candidate to the posterior probability of the current parameter)
        RandomNumbers = np.random.rand(NParticle) #Since we have to decide whether to adopt or reject the new parameter according to the Ratio, we generate a uniform random sequence of 0-1 and adopt it if each element is below the Ratio.
        for i in range (NParticle):
            if RandomNumbers[i] <= Ratio[i]: #If the random number value is less than the adoption rate, then
                l_Weight[i] = 1*l_WeightProp[i] #Accept the value of the proposal distribution (update the log likelihood)
                ParameterDist[:,i] = 1*ParameterDistProp[:,i] #Accept the value of the proposed distribution (update the parameters)
            else: #When the random number value is greater than the adoption rate, the
                pass #reject the value of the proposal distribution and use the original value

    l_max = l_Weight.max() #Calculate the maximum log likelihood
    psi[0:NParticle] = np.exp(l_Weight[0:NParticle] - l_max) #Calculate the value psi, which is the exp of the difference between the log likelihood of each particle and the maximum of the log likelihood
    PSI = np.sum(psi) #Calculate the sum of each element (= each particle) of #psi
    Weight[0:NParticle] = psi[0:NParticle]/PSI #Calculate the normalized weight of each particle again as Weight
    WeightGamma = np.power(Weight, Gamma[t] - Gamma[t-1]) #For likelihood tempering, each element of Weight, the likelihood of each particle, is multiplied by (γt-γ-1).
    ParameterPrediction = 1*ParameterDist
    l_WeightPrediction = 1*l_Weight
    RelativeWeight = WeightGamma[0:NParticle]/np.sum(WeightGamma) #normalize the likelihood of each particle by (0~1)
    ParticleWeight = NParticle*RelativeWeight[0:NParticle] #Calculate how many particles each one is worth.
    dWeight , Nsample  = np.modf(ParticleWeight) #Decompose the elements of the above array into the integer part Nsample and the fractional part dWeight
    Ntmp = NParticle - np.sum(Nsample) #Tentatively use Ntmp to store the number of decimals that cannot be resampled.
    Wrand = np.random.uniform(0,1/NParticle,1) #Generate one uniform random number in the range of 0~1/NParticle.
    Sum = 0 #Set the height of the experience distribution function to start at 0
    dWeight = dWeight/NParticle #Normalize the fractional part

    #Repeat for number of particles
    for i in range (NParticle):
        Sum += dWeight[i] #Increase the height of the empirical distribution function by a fractional part of the likelihood x number of particles
        if Sum > Wrand: #If the height of the empirical distribution function exceeds the value of the uniform random number
            Nsample[i] += 1 #Sample one i-th particle.
            Ntmp -= 1 #Reduce by one the number of remaining numbers that need to be sampled.
            Wrand += 1/NParticle #Step up the value of the uniform random number to the next height

    Nsum = 0 #Nsum=0 as the initial value because the resampled particles are relocated to the Nsumth of ParameterDist.
    #Repeat for number of particles
    for i in range (NParticle):
        for j in range (int(Nsample[i])): # by the number of the i-th box.
            ParameterDist[:,Nsum] = 1*ParameterPrediction[:,i] #Relocate the particles
            l_Weight[Nsum] = 1*l_WeightPrediction[i] #Relocate the particles
            Nsum += 1 #Nsum to the next one.
    Weight[0:NParticle] = Weight[0:NParticle]/np.sum(Weight[0:NParticle]) #Normalize the weights so that they sum to 1

#After the SMC annealing is complete, only the Metropolis method is repeated several times to make sure that a stationary distribution is obtained.
for k in tqdm(range(5)):
    Parameters = 1*ParameterDist
    EstimatedHaDist = Parameters[0,:]
    EstimatedHbDist = Parameters[1,:]
    EstimatedKaDist = Parameters[2,:]
    EstimatedKbDist = Parameters[3,:]
    EstimatedbaDist = Parameters[4,:]
    EstimatedbbDist = Parameters[5,:]
    EstimatedObsSigmaDist = Parameters[6,:]

    plt.rcParams["font.family"] ="Arial"
    plt.rcParams["font.size"] = 18
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"
    plt.rcParams["figure.figsize"] = (7,7)

    NbinHa = np.linspace(np.percentile(EstimatedHaDist, 0), np.percentile(EstimatedHaDist, 100), num = 30)
    BinWidthHa = NbinHa[1] - NbinHa[0]
    fig,ax=plt.subplots()
    freqHa, binnHa, _Ha = ax.hist(EstimatedHaDist, bins=NbinHa, range=None, density=True,color="0.2")
    HaMode = NbinHa[np.argmax(freqHa)] + BinWidthHa/2
    plt.xlabel("$\mathrm{{\it H_{Glu}}\ [-]}$",size=25)
    plt.ylabel("$\mathrm{{\it P(H_{Glu})}\ [-]}$",size=25)
    ax.ticklabel_format(axis="both", style="sci",scilimits=(0,0),useMathText=True)
    plt.savefig(dirnameHa + "\\Ha_k={}.png".format(k), bbox_inches="tight")
    plt.show()
    NbinHb = np.linspace(np.percentile(EstimatedHbDist, 0), np.percentile(EstimatedHbDist, 100), num = 30)
    BinWidthHb = NbinHb[1] - NbinHb[0]
    fig,ax=plt.subplots()
    freqHb, binnHb, _Hb = ax.hist(EstimatedHbDist, bins=NbinHb, range=None, density=True,color="0.2")
    HbMode = NbinHb[np.argmax(freqHb)] + BinWidthHb/2
    plt.xlabel("$\mathrm{{\it H_{Fru}}\ [-]}$",size=25)
    plt.ylabel("$\mathrm{{\it P(H_{Fru})}\ [-]}$",size=25)
    ax.ticklabel_format(axis="both", style="sci",scilimits=(0,0),useMathText=True)
    plt.savefig(dirnameHb + "\\Hb_k={}.png".format(k), bbox_inches="tight")
    plt.show()
    NbinKa = np.linspace(np.percentile(EstimatedKaDist, 0), np.percentile(EstimatedKaDist, 100), num = 30)
    BinWidthKa = NbinKa[1] - NbinKa[0]
    fig,ax=plt.subplots()
    freqKa, binnKa, _Ka = ax.hist(EstimatedKaDist, bins=NbinKa, range=None, density=True,color="0.2")
    KaMode = NbinKa[np.argmax(freqKa)] + BinWidthKa/2
    plt.xlabel("$\mathrm{{\it K_{Glu}}\ [s^{-1}]}$",size=25)
    plt.ylabel("$\mathrm{{\it P(K_{Glu})}\ [s]}$",size=25)
    ax.ticklabel_format(axis="both", style="sci",scilimits=(0,0),useMathText=True)
    plt.savefig(dirnameKa + "\\Ka_k={}.png".format(k), bbox_inches="tight")
    plt.show()
    NbinKb = np.linspace(np.percentile(EstimatedKbDist, 0), np.percentile(EstimatedKbDist, 100), num = 30)
    BinWidthKb = NbinKb[1] - NbinKb[0]
    fig,ax=plt.subplots()
    freqKb, binnKb, _Kb = ax.hist(EstimatedKbDist, bins=NbinKb, range=None, density=True,color="0.2")
    KbMode = NbinKb[np.argmax(freqKb)] + BinWidthKb/2
    plt.xlabel("$\mathrm{{\it K_{Fru}}\ [s^{-1}]}$",size=25)
    plt.ylabel("$\mathrm{{\it P(K_{Fru})}\ [s]}$",size=25)
    ax.ticklabel_format(axis="both", style="sci",scilimits=(0,0),useMathText=True)
    plt.savefig(dirnameKb + "\\kb_k={}.png".format(k), bbox_inches="tight")
    plt.show()
    Nbinba = np.linspace(np.percentile(EstimatedbaDist, 0), np.percentile(EstimatedbaDist, 100), num = 30)
    BinWidthba = Nbinba[1] - Nbinba[0]
    fig,ax=plt.subplots()
    freqba, binnba, _ba = ax.hist(EstimatedbaDist, bins=Nbinba, range=None, density=True,color="0.2")
    baMode = Nbinba[np.argmax(freqba)] + BinWidthba/2
    plt.xlabel("$\mathrm{{\it b_{Glu}}\ [L/g]}$",size=25)
    plt.ylabel("$\mathrm{{\it P(b_{Glu})}\ [g/L]}$",size=25)
    ax.ticklabel_format(axis="both", style="sci",scilimits=(0,0),useMathText=True)
    plt.savefig(dirnameba + "\\ba_k={}.png".format(k), bbox_inches="tight")
    plt.show()
    Nbinbb = np.linspace(np.percentile(EstimatedbbDist, 0), np.percentile(EstimatedbbDist, 100), num = 30)
    BinWidthbb = Nbinbb[1] - Nbinbb[0]
    fig,ax=plt.subplots()
    freqbb, binnbb, _bb = ax.hist(EstimatedbbDist, bins=Nbinbb, range=None, density=True,color="0.2")
    bbMode = Nbinbb[np.argmax(freqbb)] + BinWidthbb/2
    plt.xlabel("$\mathrm{{\it b_{Fru}}\ [L/g]}$",size=25)
    plt.ylabel("$\mathrm{{\it P(b_{Fru})}\ [g/L]}$",size=25)
    ax.ticklabel_format(axis="both", style="sci",scilimits=(0,0),useMathText=True)
    plt.savefig(dirnamebb + "\\bb_k={}.png".format(k), bbox_inches="tight")
    plt.show()
    NbinObsSigma = np.linspace(np.percentile(EstimatedObsSigmaDist, 0), np.percentile(EstimatedObsSigmaDist, 100), num = 30)
    BinWidthObsSigma = NbinObsSigma[1] - NbinObsSigma[0]
    fig,ax=plt.subplots()
    freqObsSigma, binnObsSigma, _ObsSigma = ax.hist(EstimatedObsSigmaDist, bins=NbinObsSigma, range=None, density=True,color="0.2")
    ObsSigmaMode = NbinObsSigma[np.argmax(freqObsSigma)] + BinWidthObsSigma/2
    plt.xlabel("$\mathrm{{\it \sigma}\ [-]}$",size=25)
    plt.ylabel("$\mathrm{{\it P(\sigma)}\ [-]}$",size=25)
    ax.ticklabel_format(axis="both", style="sci",scilimits=(0,0),useMathText=True)
    plt.savefig(dirnameObsSigma + "\\ObsSigma_k={}.png".format(k), bbox_inches="tight")
    plt.show()

    ParameterDistProp = 1*ParameterDist
    if t==1:
        pass
    else:
        VHa = np.random.normal(loc=0,scale=0.5*np.std(EstimatedHaDist),size=NParticle)
        VHb = np.random.normal(loc=0,scale=0.5*np.std(EstimatedHbDist),size=NParticle)
        VKa = np.random.normal(loc=0,scale=0.5*np.std(EstimatedKaDist),size=NParticle)
        VKb = np.random.normal(loc=0,scale=0.5*np.std(EstimatedKbDist),size=NParticle)
        Vba = np.random.normal(loc=0,scale=0.5*np.std(EstimatedbaDist),size=NParticle)
        Vbb = np.random.normal(loc=0,scale=0.5*np.std(EstimatedbbDist),size=NParticle)
        VObsSigma = np.random.normal(loc=0,scale=0.5*np.std(EstimatedObsSigmaDist),size=NParticle)

        ParameterDistProp[0,0:NParticle] += VHa[0:NParticle]
        ParameterDistProp[1,0:NParticle] += VHb[0:NParticle]
        ParameterDistProp[2,0:NParticle] += VKa[0:NParticle]
        ParameterDistProp[3,0:NParticle] += VKb[0:NParticle]
        ParameterDistProp[4,0:NParticle] += Vba[0:NParticle]
        ParameterDistProp[5,0:NParticle] += Vbb[0:NParticle]
        ParameterDistProp[6,0:NParticle] += VObsSigma[0:NParticle]

    Ha = ParameterDistProp[0,:]
    Hb = ParameterDistProp[1,:]
    Ka = ParameterDistProp[2,:]
    Kb = ParameterDistProp[3,:]
    ba = ParameterDistProp[4,:]
    bb = ParameterDistProp[5,:]
    ObsSigma = ParameterDistProp[6,:]

    POWEREDERROR1 = Parallel(n_jobs=core)( [delayed(PROCESSMODEL1)(Ha[P],Hb[P],Ka[P],Kb[P],ba[P],bb[P]) for P in range(NParticle)] )
    POWEREDERROR2 = Parallel(n_jobs=core)( [delayed(PROCESSMODEL2)(Ha[P],Hb[P],Ka[P],Kb[P],ba[P],bb[P]) for P in range(NParticle)] )
    POWEREDERROR3 = Parallel(n_jobs=core)( [delayed(PROCESSMODEL3)(Ha[P],Hb[P],Ka[P],Kb[P],ba[P],bb[P]) for P in range(NParticle)] )
    POWEREDERROR1 = np.array(POWEREDERROR1)
    POWEREDERROR2 = np.array(POWEREDERROR2)
    POWEREDERROR3 = np.array(POWEREDERROR3)
    Total_ERROR2[0:NParticle] = POWEREDERROR1[0:NParticle] + POWEREDERROR2[0:NParticle] + POWEREDERROR3[0:NParticle]
    l_WeightProp[0:NParticle] = -((M1+M2+M3)/2)*np.log(2*np.pi*ObsSigma[0:NParticle]**2) - Total_ERROR2[0:NParticle]/(2*ObsSigma[0:NParticle]**2)
    Ratio = ((np.exp(l_WeightProp - l_Weight))**Gamma[t-1])*(P0(ParameterDistProp)/P0(ParameterDist))
    RandomNumbers = np.random.rand(NParticle)
    for i in range (NParticle):
        if RandomNumbers[i] <= Ratio[i]:
            l_Weight[i] = 1*l_WeightProp[i]
            ParameterDist[:,i] = 1*ParameterDistProp[:,i]
        else:
            pass


#Record and display the time when UQ is finished.
FinishDateTime = datetime.datetime.now()
print("UQ is finished at {}".format(FinishDateTime))
#Show the time it took to perform UQ.
print("Computationtime is {}".format(FinishDateTime-StartDateTime))


Parameters = 1*ParameterDist
EstimatedHaDist = Parameters[0,:]
EstimatedHbDist = Parameters[1,:]
EstimatedKaDist = Parameters[2,:]
EstimatedKbDist = Parameters[3,:]
EstimatedbaDist = Parameters[4,:]
EstimatedbbDist = Parameters[5,:]
EstimatedObsSigmaDist = Parameters[6,:]

plt.rcParams["font.family"] ="Arial"
plt.rcParams["font.size"] = 18
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["figure.figsize"] = (7,7)

#Create a folder to store the final results of UQ.
dirnameHISTGRAMWB = dirname+"\\Results of UQ"
os.makedirs(dirnameHISTGRAMWB,exist_ok=False)

fig,ax=plt.subplots()
freqHa, binnHa, _Ha = ax.hist(EstimatedHaDist, bins=NbinHa, range=None, density=True,color="0.2")
plt.xlabel("$\mathrm{{\it H_{Glu}}\ [-]}$",size=25)
plt.ylabel("$\mathrm{{\it P(H_{Glu})}\ [-]}$",size=25)
ax.ticklabel_format(axis="both", style="sci",scilimits=(0,0),useMathText=True)
plt.savefig(dirnameHISTGRAMWB + "/Ha_Dist.png", bbox_inches="tight")
plt.show()

fig,ax=plt.subplots()
freqHb, binnHb, _Hb = ax.hist(EstimatedHbDist, bins=NbinHb, range=None, density=True,color="0.2")
plt.xlabel("$\mathrm{{\it H_{Fru}}\ [-]}$",size=25)
plt.ylabel("$\mathrm{{\it P(H_{Fru})}\ [-]}$",size=25)
ax.ticklabel_format(axis="both", style="sci",scilimits=(0,0),useMathText=True)
plt.savefig(dirnameHISTGRAMWB + "/Hb_Dist.png", bbox_inches="tight")
plt.show()

fig,ax=plt.subplots()
freqKa, binnKa, _Ka = ax.hist(EstimatedKaDist, bins=NbinKa, range=None, density=True,color="0.2")
plt.xlabel("$\mathrm{{\it K_{Glu}}\ [s^{-1}]}$",size=25)
plt.ylabel("$\mathrm{{\it P(K_{Glu})}\ [s]}$",size=25)
ax.ticklabel_format(axis="both", style="sci",scilimits=(0,0),useMathText=True)
plt.savefig(dirnameHISTGRAMWB + "/Ka_Dist.png", bbox_inches="tight")
plt.show()

fig,ax=plt.subplots()
freqKb, binnKb, _Kb = ax.hist(EstimatedKbDist, bins=NbinKb, range=None, density=True,color="0.2")
plt.xlabel("$\mathrm{{\it K_{Fru}}\ [s^{-1}]}$",size=25)
plt.ylabel("$\mathrm{{\it P(K_{Fru})}\ [s]}$",size=25)
ax.ticklabel_format(axis="both", style="sci",scilimits=(0,0),useMathText=True)
plt.savefig(dirnameHISTGRAMWB + "/Kb_Dist.png", bbox_inches="tight")
plt.show()

fig,ax=plt.subplots()
freqba, binnba, _ba = ax.hist(EstimatedbaDist, bins=Nbinba, range=None, density=True,color="0.2")
plt.xlabel("$\mathrm{{\it b_{Glu}}\ [L/g]}$",size=25)
plt.ylabel("$\mathrm{{\it P(b_{Glu})}\ [g/L]}$",size=25)
ax.ticklabel_format(axis="both", style="sci",scilimits=(0,0),useMathText=True)
plt.savefig(dirnameHISTGRAMWB + "/ba_Dist.png", bbox_inches="tight")
plt.show()

fig,ax=plt.subplots()
freqbb, binnbb, _bb = ax.hist(EstimatedbbDist, bins=Nbinbb, range=None, density=True,color="0.2")
plt.xlabel("$\mathrm{{\it b_{Fru}}\ [L/g]}$",size=25)
plt.ylabel("$\mathrm{{\it P(b_{Fru})}\ [g/L]}$",size=25)
ax.ticklabel_format(axis="both", style="sci",scilimits=(0,0),useMathText=True)
plt.savefig(dirnameHISTGRAMWB + "/bb_Dist.png", bbox_inches="tight")
plt.show()

fig,ax=plt.subplots()
freqObsSigma, binnObsSigma, _ObsSigma = ax.hist(EstimatedObsSigmaDist, bins=NbinObsSigma, range=None, density=True,color="0.2")
plt.xlabel("$\mathrm{{\it \sigma}\ [-]}$",size=25)
plt.ylabel("$\mathrm{{\it P(\sigma)}\ [-]}$",size=25)
ax.ticklabel_format(axis="both", style="sci",scilimits=(0,0),useMathText=True)
plt.savefig(dirnameHISTGRAMWB + "/ObsSigma_Dist.png", bbox_inches="tight")
plt.show()

#Show representative summary statistics
print("HaMode = {}".format(HaMode))
HaAverage = np.mean(EstimatedHaDist)
HaStdev = np.std(EstimatedHaDist)
Ha95PercentileLow, Ha95PercentileUp = np.percentile(EstimatedHaDist, [2.5, 97.5])
print("HaStdev={}".format(HaStdev))
print("Ha95Percentile={}, {}".format(Ha95PercentileLow,Ha95PercentileUp))

print("HbMode = {}".format(HbMode))
HbAverage = np.mean(EstimatedHbDist)
HbStdev = np.std(EstimatedHbDist)
Hb95PercentileLow, Hb95PercentileUp = np.percentile(EstimatedHbDist, [2.5, 97.5])
print("HbStdev={}".format(HbStdev))
print("Hb95Percentile={}, {}".format(Hb95PercentileLow,Hb95PercentileUp))

print("KaMode = {}".format(KaMode))
KaAverage = np.mean(EstimatedKaDist)
KaStdev = np.std(EstimatedKaDist)
Ka95PercentileLow, Ka95PercentileUp = np.percentile(EstimatedKaDist, [2.5, 97.5])
print("KaStdev={}".format(KaStdev))
print("Ka95Percentile={}, {}".format(Ka95PercentileLow,Ka95PercentileUp))

print("KbMode = {}".format(KbMode))
KbAverage = np.mean(EstimatedKbDist)
KbStdev = np.std(EstimatedKbDist)
Kb95PercentileLow, Kb95PercentileUp = np.percentile(EstimatedKbDist, [2.5, 97.5])
print("KbStdev={}".format(KbStdev))
print("Kb95Percentile={}, {}".format(Kb95PercentileLow,Kb95PercentileUp))

print("baMode = {}".format(baMode))
baAverage = np.mean(EstimatedbaDist)
baStdev = np.std(EstimatedbaDist)
ba95PercentileLow, ba95PercentileUp = np.percentile(EstimatedbaDist, [2.5, 97.5])
print("baStdev={}".format(baStdev))
print("ba95Percentile={}, {}".format(ba95PercentileLow,ba95PercentileUp))

print("bbMode = {}".format(bbMode))
bbAverage = np.mean(EstimatedbbDist)
bbStdev = np.std(EstimatedbbDist)
bb95PercentileLow, bb95PercentileUp = np.percentile(EstimatedbbDist, [2.5, 97.5])
print("bbStdev={}".format(bbStdev))
print("bb95Percentile={}, {}".format(bb95PercentileLow,bb95PercentileUp))

print("ObsSigmaMode = {}".format(ObsSigmaMode))
ObsSigmaAverage = np.mean(EstimatedObsSigmaDist)
ObsSigmaStdev = np.std(EstimatedObsSigmaDist)
ObsSigma95PercentileLow, ObsSigma95PercentileUp = np.percentile(EstimatedObsSigmaDist, [2.5, 97.5])
print("ObsSigmaStdev={}".format(ObsSigmaStdev))
print("ObsSigma95Percentile={}, {}".format(ObsSigma95PercentileLow,ObsSigma95PercentileUp))

#Run the simulation using the most frequent values of the estimated parameters.
SimulationData1 = PROCESSMODEL(HaMode,HbMode,KaMode,KbMode,baMode,bbMode,f0,tData1,tf1,Ca01,Cb01,e1,u1,dz1)
SimulationData2 = PROCESSMODEL(HaMode,HbMode,KaMode,KbMode,baMode,bbMode,f0,tData2,tf2,Ca02,Cb02,e2,u2,dz2)
SimulationData3 = PROCESSMODEL(HaMode,HbMode,KaMode,KbMode,baMode,bbMode,f0,tData3,tf3,Ca03,Cb03,e3,u3,dz3)
SimulationData4 = PROCESSMODEL(HaMode,HbMode,KaMode,KbMode,baMode,bbMode,f0,tData4,tf4,Ca04,Cb04,e4,u4,dz4)
CaOutSim1 = SimulationData1[0:M1,N-1]
CbOutSim2 = SimulationData2[0:M2,2*N-1]
CaOutSim3 = SimulationData3[0:M3,N-1]
CbOutSim3 = SimulationData3[0:M3,2*N-1]
CTotalOutSim3 = CaOutSim3 + CbOutSim3
CaOutSim4 = SimulationData4[0:M4,N-1]
CbOutSim4 = SimulationData4[0:M4,2*N-1]

plt.rcParams["font.family"] ="Arial"
plt.rcParams["font.size"] = 18
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"

#Run the simulation multiple times with parameters sampled from the posterior distribution to show the credit interval graphically.
dirnameSAMPLINGSIM = dirname+"\\Fittings and Predictions"
os.makedirs(dirnameSAMPLINGSIM,exist_ok=False)

plt.scatter(tData1,CaOutData1,label="$\mathrm{Glu_{exp}}$", marker="1", s=10,color="0.1")
plt.plot(tData1,CaOutSim1,label="$\mathrm{Glu_{SMC}}$",linewidth=2,color="0.4")
for k in range(1000):
    i = np.random.randint(NParticle)
    SimulationData1 = PROCESSMODEL(Parameters[0,i],Parameters[1,i],Parameters[2,i],Parameters[3,i],Parameters[4,i],Parameters[5,i],f0,tData1,tf1,Ca01,Cb01,e1,u1,dz1)
    CaOutSim1 = SimulationData1[0:M1,N-1]
    plt.plot(tData1,CaOutSim1,linewidth=2,color="0.4",alpha=0.8)
plt.xlim(0,max(tData1))
plt.ylim(0,1.2*max(CaOutData1))
plt.xlabel("Time [s]")
plt.ylabel("Concentration [g/L]")
plt.legend(bbox_to_anchor=(0.99, 0.99), loc='upper right', borderaxespad=0,fontsize=18)
plt.tight_layout()
plt.savefig(dirnameSAMPLINGSIM + "\\Fitting of A.png", bbox_inches="tight")
plt.show()

plt.scatter(tData2,CbOutData2,label="$\mathrm{Fru_{exp}}$", marker="+", s=10,color="0.1")
plt.plot(tData2,CbOutSim2,label="$\mathrm{Fru_{SMC}}$",linewidth=2,color="0.4")
for k in range(1000):
    i = np.random.randint(NParticle)
    SimulationData2 = PROCESSMODEL(Parameters[0,i],Parameters[1,i],Parameters[2,i],Parameters[3,i],Parameters[4,i],Parameters[5,i],f0,tData2,tf2,Ca02,Cb02,e2,u2,dz2)
    CbOutSim2 = SimulationData2[0:M2,2*N-1]
    plt.plot(tData2,CbOutSim2,linewidth=2,color="0.4",alpha=0.8)
plt.xlim(0,max(tData2))
plt.ylim(0,1.2*max(CbOutData2))
plt.xlabel("Time [s]")
plt.ylabel("Concentration [g/L]")
plt.legend(bbox_to_anchor=(0.99, 0.99), loc='upper right', borderaxespad=0,fontsize=18)
plt.tight_layout()
plt.savefig(dirnameSAMPLINGSIM + "\\Fitting of B.png", bbox_inches="tight")
plt.show()

plt.scatter(tData3,CTotalOutData3,label="$\mathrm{Total_{exp}}$", marker="v", s=10,color="0.1")
plt.plot(tData3,CTotalOutSim3,label="$\mathrm{Total_{SMC}}$",linewidth=2,color="0.4")
for k in range(1000):
    i = np.random.randint(NParticle)
    SimulationData3 = PROCESSMODEL(Parameters[0,i],Parameters[1,i],Parameters[2,i],Parameters[3,i],Parameters[4,i],Parameters[5,i],f0,tData3,tf3,Ca03,Cb03,e3,u3,dz3)
    CaOutSim3 = SimulationData3[0:M3,N-1]
    CbOutSim3 = SimulationData3[0:M3,2*N-1]
    CTotalOutSim3 = CaOutSim3 + CbOutSim3
    plt.plot(tData3,CTotalOutSim3,linewidth=2,color="0.4",alpha=0.8)
plt.xlim(0,max(tData3))
plt.ylim(0,1.2*max(CTotalOutData3))
plt.xlabel("Time [s]")
plt.ylabel("Concentration [g/L]")
plt.legend(bbox_to_anchor=(0.01, 0.99), loc='upper left', borderaxespad=0,fontsize=18)
plt.tight_layout()
plt.savefig(dirnameSAMPLINGSIM + "\\Fitting of C.png", bbox_inches="tight")
plt.show()

plt.scatter(tData4,CaOutData4,label="$\mathrm{Glu_{exp}}$", marker="1", s=10,color="0.1")
plt.scatter(tData4,CbOutData4,label="$\mathrm{Fru_{exp}}$", marker="+", s=10,color="0.1")
plt.plot(tData4,CaOutSim4,label="$\mathrm{Glu_{SMC}}$",linewidth=2,color="0.4")
plt.plot(tData4,CbOutSim4,label="$\mathrm{Fru_{SMC}}$",linewidth=2,color="0.4",ls="--")
for k in range(1000):
    i = np.random.randint(NParticle)
    SimulationData4 = PROCESSMODEL(Parameters[0,i],Parameters[1,i],Parameters[2,i],Parameters[3,i],Parameters[4,i],Parameters[5,i],f0,tData4,tf4,Ca04,Cb04,e4,u4,dz4)
    CaOutSim4 = SimulationData4[0:M4,N-1]
    CbOutSim4 = SimulationData4[0:M4,2*N-1]
    plt.plot(tData4,CaOutSim4,linewidth=2,color="0.4",alpha=0.8)
    plt.plot(tData4,CbOutSim4,linewidth=2,color="0.4",alpha=0.8,ls="--")
plt.xlim(0,max(tData4))
plt.ylim(0,1.2*max(CaOutData4))
plt.xlabel("Time [s]")
plt.ylabel("Concentration [g/L]")
plt.legend(bbox_to_anchor=(0.01, 0.99), loc='upper left', borderaxespad=0,fontsize=18)
plt.tight_layout()
plt.savefig(dirnameSAMPLINGSIM + "\\Validation using C_deconvoluted.png", bbox_inches="tight")
plt.show()
