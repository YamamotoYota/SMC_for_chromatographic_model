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
import pandas as pd #Used to read and process data
import pymc as pm #Used for MAP estimation and MCMC calculation
import numba #Used for JIT compilation

#Create a folder to automatically save the obtained data.
dirname = ".\\UQ_by_MCMC" #Name of the folder (directory) you want to create
os.makedirs(dirname,exist_ok=False) #Create a folder

#Folder to save the process of MCMC
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
Ham = pm.Normal("Ha",mu=HaMAP,tau=1/((cHa*HaMAP)**2),value=HaMAP)
Hbm = pm.Normal("Hb",mu=HbMAP,tau=1/((cHb*HbMAP)**2),value=HbMAP)
Kam = pm.Normal("Ka",mu=KaMAP,tau=1/((cKa*KaMAP)**2),value=KaMAP)
Kbm = pm.Normal("Kb",mu=KbMAP,tau=1/((cKb*KbMAP)**2),value=KbMAP)
bam = pm.Uniform("ba",lower=ba_LOW,upper=ba_UP)
bbm = pm.Uniform("bb",lower=bb_LOW,upper=bb_UP)
ObsSigmam = pm.Normal("ObsSigma",mu=ObsSigmaMAP,tau=1/((cObsSigma*ObsSigmaMAP)**2))


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

sampler = pm.MCMC(model)
sampler.sample(iter=50000,burn=30000, thin=2) #Set sampling frequency, burn-in, and rejection width
pm.Matplot.plot(sampler,format='png',path=dirname)

#Record and display the time when UQ is finished.
FinishDateTime = datetime.datetime.now()
print("UQ is finished at {}".format(FinishDateTime))
#Show the time it took to perform UQ.
print("Computationtime is {}".format(FinishDateTime-StartDateTime))

EstimatedHaDist = Ham.trace()
EstimatedHbDist = Hbm.trace()
EstimatedKaDist = Kam.trace()
EstimatedKbDist = Kbm.trace()
EstimatedbaDist = bam.trace()
EstimatedbbDist = bbm.trace()
EstimatedObsSigmaDist = ObsSigmam.trace()
Parameters = np.vstack((EstimatedHaDist,EstimatedHbDist,EstimatedKaDist,EstimatedKbDist,EstimatedbaDist,EstimatedbbDist,EstimatedObsSigmaDist))

plt.rcParams["font.family"] ="Arial"
plt.rcParams["font.size"] = 18
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["figure.figsize"] = (7,7)

#Create a folder to store the final results of UQ.
dirnameHISTGRAMWB = dirname+"\\Results of UQ"
os.makedirs(dirnameHISTGRAMWB,exist_ok=False)

NbinHa = np.linspace(np.percentile(EstimatedHaDist, 0), np.percentile(EstimatedHaDist, 100), num = 30)
BinWidthHa = NbinHa[1] - NbinHa[0]
fig,ax=plt.subplots()
freqHa, binnHa, _Ha = ax.hist(EstimatedHaDist, bins=NbinHa, range=None, density=True,color="0.2")
HaMode = NbinHa[np.argmax(freqHa)] + BinWidthHa/2
plt.xlabel("$\mathrm{{\it H_{Glu}}\ [-]}$",size=25)
plt.ylabel("$\mathrm{{\it P(H_{Glu})}\ [-]}$",size=25)
ax.ticklabel_format(axis="both", style="sci",scilimits=(0,0),useMathText=True)
plt.savefig(dirnameHISTGRAMWB + "\\Ha.png", bbox_inches="tight")
plt.show()
NbinHb = np.linspace(np.percentile(EstimatedHbDist, 0), np.percentile(EstimatedHbDist, 100), num = 30)
BinWidthHb = NbinHb[1] - NbinHb[0]
fig,ax=plt.subplots()
freqHb, binnHb, _Hb = ax.hist(EstimatedHbDist, bins=NbinHb, range=None, density=True,color="0.2")
HbMode = NbinHb[np.argmax(freqHb)] + BinWidthHb/2
plt.xlabel("$\mathrm{{\it H_{Fru}}\ [-]}$",size=25)
plt.ylabel("$\mathrm{{\it P(H_{Fru})}\ [-]}$",size=25)
ax.ticklabel_format(axis="both", style="sci",scilimits=(0,0),useMathText=True)
plt.savefig(dirnameHISTGRAMWB + "\\Hb.png", bbox_inches="tight")
plt.show()
NbinKa = np.linspace(np.percentile(EstimatedKaDist, 0), np.percentile(EstimatedKaDist, 100), num = 30)
BinWidthKa = NbinKa[1] - NbinKa[0]
fig,ax=plt.subplots()
freqKa, binnKa, _Ka = ax.hist(EstimatedKaDist, bins=NbinKa, range=None, density=True,color="0.2")
KaMode = NbinKa[np.argmax(freqKa)] + BinWidthKa/2
plt.xlabel("$\mathrm{{\it K_{Glu}}\ [s^{-1}]}$",size=25)
plt.ylabel("$\mathrm{{\it P(K_{Glu})}\ [s]}$",size=25)
ax.ticklabel_format(axis="both", style="sci",scilimits=(0,0),useMathText=True)
plt.savefig(dirnameHISTGRAMWB + "\\Ka.png", bbox_inches="tight")
plt.show()
NbinKb = np.linspace(np.percentile(EstimatedKbDist, 0), np.percentile(EstimatedKbDist, 100), num = 30)
BinWidthKb = NbinKb[1] - NbinKb[0]
fig,ax=plt.subplots()
freqKb, binnKb, _Kb = ax.hist(EstimatedKbDist, bins=NbinKb, range=None, density=True,color="0.2")
KbMode = NbinKb[np.argmax(freqKb)] + BinWidthKb/2
plt.xlabel("$\mathrm{{\it K_{Fru}}\ [s^{-1}]}$",size=25)
plt.ylabel("$\mathrm{{\it P(K_{Fru})}\ [s]}$",size=25)
ax.ticklabel_format(axis="both", style="sci",scilimits=(0,0),useMathText=True)
plt.savefig(dirnameHISTGRAMWB + "\\Kb.png", bbox_inches="tight")
plt.show()
Nbinba = np.linspace(np.percentile(EstimatedbaDist, 0), np.percentile(EstimatedbaDist, 100), num = 30)
BinWidthba = Nbinba[1] - Nbinba[0]
fig,ax=plt.subplots()
freqba, binnba, _ba = ax.hist(EstimatedbaDist, bins=Nbinba, range=None, density=True,color="0.2")
baMode = Nbinba[np.argmax(freqba)] + BinWidthba/2
plt.xlabel("$\mathrm{{\it b_{Glu}}\ [L/g]}$",size=25)
plt.ylabel("$\mathrm{{\it P(b_{Glu})}\ [g/L]}$",size=25)
ax.ticklabel_format(axis="both", style="sci",scilimits=(0,0),useMathText=True)
plt.savefig(dirnameHISTGRAMWB + "\\ba.png", bbox_inches="tight")
plt.show()
Nbinbb = np.linspace(np.percentile(EstimatedbbDist, 0), np.percentile(EstimatedbbDist, 100), num = 30)
BinWidthbb = Nbinbb[1] - Nbinbb[0]
fig,ax=plt.subplots()
freqbb, binnbb, _bb = ax.hist(EstimatedbbDist, bins=Nbinbb, range=None, density=True,color="0.2")
bbMode = Nbinbb[np.argmax(freqbb)] + BinWidthbb/2
plt.xlabel("$\mathrm{{\it b_{Fru}}\ [L/g]}$",size=25)
plt.ylabel("$\mathrm{{\it P(b_{Fru})}\ [g/L]}$",size=25)
ax.ticklabel_format(axis="both", style="sci",scilimits=(0,0),useMathText=True)
plt.savefig(dirnameHISTGRAMWB + "\\bb.png", bbox_inches="tight")
plt.show()
NbinObsSigma = np.linspace(np.percentile(EstimatedObsSigmaDist, 0), np.percentile(EstimatedObsSigmaDist, 100), num = 30)
BinWidthObsSigma = NbinObsSigma[1] - NbinObsSigma[0]
fig,ax=plt.subplots()
freqObsSigma, binnObsSigma, _ObsSigma = ax.hist(EstimatedObsSigmaDist, bins=NbinObsSigma, range=None, density=True,color="0.2")
ObsSigmaMode = NbinObsSigma[np.argmax(freqObsSigma)] + BinWidthObsSigma/2
plt.xlabel("$\mathrm{{\it \sigma}\ [-]}$",size=25)
plt.ylabel("$\mathrm{{\it P(\sigma)}\ [-]}$",size=25)
ax.ticklabel_format(axis="both", style="sci",scilimits=(0,0),useMathText=True)
plt.savefig(dirnameHISTGRAMWB + "\\Sigma.png", bbox_inches="tight")
plt.show()

NParticle = len(EstimatedHaDist)

#Show representative summary statistics
print("HaMode = {}".format(HaMode))
HaAverage = np.mean(EstimatedHaDist)
HaStdev = np.std(EstimatedHaDist)
Ha95PercentileLow, Ha95PercentileUp = np.percentile(EstimatedHaDist, [5, 95])
print("HaStdev={}".format(HaStdev))
print("Ha95Percentile={}, {}".format(Ha95PercentileLow,Ha95PercentileUp))

print("HbMode = {}".format(HbMode))
HbAverage = np.mean(EstimatedHbDist)
HbStdev = np.std(EstimatedHbDist)
Hb95PercentileLow, Hb95PercentileUp = np.percentile(EstimatedHbDist, [5, 95])
print("HbStdev={}".format(HbStdev))
print("Hb95Percentile={}, {}".format(Hb95PercentileLow,Hb95PercentileUp))

print("KaMode = {}".format(KaMode))
KaAverage = np.mean(EstimatedKaDist)
KaStdev = np.std(EstimatedKaDist)
Ka95PercentileLow, Ka95PercentileUp = np.percentile(EstimatedKaDist, [5, 95])
print("KaStdev={}".format(KaStdev))
print("Ka95Percentile={}, {}".format(Ka95PercentileLow,Ka95PercentileUp))

print("KbMode = {}".format(KbMode))
KbAverage = np.mean(EstimatedKbDist)
KbStdev = np.std(EstimatedKbDist)
Kb95PercentileLow, Kb95PercentileUp = np.percentile(EstimatedKbDist, [5, 95])
print("KbStdev={}".format(KbStdev))
print("Kb95Percentile={}, {}".format(Kb95PercentileLow,Kb95PercentileUp))

print("baMode = {}".format(baMode))
baAverage = np.mean(EstimatedbaDist)
baStdev = np.std(EstimatedbaDist)
ba95PercentileLow, ba95PercentileUp = np.percentile(EstimatedbaDist, [5, 95])
print("baStdev={}".format(baStdev))
print("ba95Percentile={}, {}".format(ba95PercentileLow,ba95PercentileUp))

print("bbMode = {}".format(bbMode))
bbAverage = np.mean(EstimatedbbDist)
bbStdev = np.std(EstimatedbbDist)
bb95PercentileLow, bb95PercentileUp = np.percentile(EstimatedbbDist, [5, 95])
print("bbStdev={}".format(bbStdev))
print("bb95Percentile={}, {}".format(bb95PercentileLow,bb95PercentileUp))

print("ObsSigmaMode = {}".format(ObsSigmaMode))
ObsSigmaAverage = np.mean(EstimatedObsSigmaDist)
ObsSigmaStdev = np.std(EstimatedObsSigmaDist)
ObsSigma95PercentileLow, ObsSigma95PercentileUp = np.percentile(EstimatedObsSigmaDist, [5, 95])
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
plt.plot(tData1,CaOutSim1,label="$\mathrm{Glu_{MCMC}}$",linewidth=2,color="0.4")
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
plt.plot(tData2,CbOutSim2,label="$\mathrm{Fru_{MCMC}}$",linewidth=2,color="0.4")
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
plt.plot(tData3,CTotalOutSim3,label="$\mathrm{Total_{MCMC}}$",linewidth=2,color="0.4")
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
plt.plot(tData4,CaOutSim4,label="$\mathrm{Glu_{MCMC}}$",linewidth=2,color="0.4")
plt.plot(tData4,CbOutSim4,label="$\mathrm{Fru_{MCMC}}$",linewidth=2,color="0.4",ls="--")
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
