# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 11:33:53 2021

@author: Yota Yamamoto
"""

"""
Pseudo-experimental data generation program for glucose (component A)-fructose (component B) system
References

Multi-column chromatographic process development using simulated
moving bed superstructure and simultaneous optimization – Model
correction framework

Balamurali Sreedhar, Yoshiaki Kawajiri

Chemical Engineering Science 116 (2014) 428–441

Ha,Hb,ka,kb = 0.301,0.531,0.0047,0.0083

Synergistic effects in competitive adsorption of
carbohydrates on an ion-exchange resin

J. Nowak, K. Gedicke, D. Antos,
W. Piatkowski, A. Seidel-Morgenstern

Journal of Chromatography A, 1164 (2007) 224–234

ba,bb = 6.34e-4, 2.48e-4
"""

#Import the required libraries
import numpy as np #Numeric computation libraries (multi-dimensional arrays, statistical functions, etc.)
from scipy.integrate import odeint #Used to solve ordinary differential equations.
import matplotlib.pyplot as plt #used to draw graphs
import pandas as pd #Used to read and process data
import numba #Used for JIT compilation

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

tEnd = 2500.0
tData = np.arange(0.0,tEnd,tEnd/100)

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

    return ret #関数funcの返り値

#Functions to compute the LDF model
def PROCESSMODEL(Ha,Hb,Ka,Kb,ba,bb,f0,tspan,tf,Ca0,Cb0,e,u,dz):
    p=np.array([Ha,Hb,Ka,Kb,ba,bb])
    sol = odeint(func, f0, tspan, args=(p,tf,Ca0,Cb0,e,u,dz)) #Numerical computation of differential equations
    return sol


HaMode,HbMode,kaMode,kbMode,baMode,bbMode = 0.301,0.531,0.0047,0.0083,6.34e-4, 2.48e-4

#Generate pseudo-experimental data
SimulationData1 = PROCESSMODEL(HaMode,HbMode,kaMode,kbMode,baMode,bbMode,f0,tData,tf1,Ca01,Cb01,e1,u1,dz1)
SimulationData2 = PROCESSMODEL(HaMode,HbMode,kaMode,kbMode,baMode,bbMode,f0,tData,tf2,Ca02,Cb02,e2,u2,dz2)
SimulationData3 = PROCESSMODEL(HaMode,HbMode,kaMode,kbMode,baMode,bbMode,f0,tData,tf3,Ca03,Cb03,e3,u3,dz3)
SimulationData4 = PROCESSMODEL(HaMode,HbMode,kaMode,kbMode,baMode,bbMode,f0,tData,tf4,Ca04,Cb04,e4,u4,dz4)
CaOutSim1 = SimulationData1[0:M1,N-1] + np.random.normal(loc=0,scale=0.005*max(SimulationData1[0:M1,N-1]),size=M1)
CbOutSim2 = SimulationData2[0:M2,2*N-1] + np.random.normal(loc=0,scale=0.005*max(SimulationData2[0:M2,2*N-1]),size=M2)
SimulationTotalData3 = SimulationData3[0:M3,N-1] + SimulationData3[0:M3,2*N-1]
CTotalOutSim3 = SimulationTotalData3 + np.random.normal(loc=0,scale=0.005*max(SimulationTotalData3),size=M3)
CaOutSim4 = SimulationData4[0:M4,N-1] + np.random.normal(loc=0,scale=0.005*max(SimulationData4[0:M4,N-1]),size=M4)
CbOutSim4 = SimulationData4[0:M4,2*N-1] + np.random.normal(loc=0,scale=0.005*max(SimulationData4[0:M4,2*N-1]),size=M4)


plt.figure
plt.rcParams['font.family'] ='Arial'
plt.rcParams["font.size"] = 20
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

plt.figure
plt.plot(tData,CaOutSim1,label="$\mathrm{a_{SimData}}$",linewidth=2,color="blue")
plt.xlim(0,max(tData))
plt.ylim(0,1.2*max(CaOutSim1))
plt.xlabel("Time [s]")
plt.ylabel("Concentration [g/L]")
plt.legend(bbox_to_anchor=(1.01, 0.99), loc='upper left', borderaxespad=0,fontsize=18)
plt.tight_layout()
plt.show()

plt.figure
plt.plot(tData,CbOutSim2,label="$\mathrm{b_{SimData}}$",linewidth=2,color="green")
plt.xlim(0,max(tData))
plt.ylim(0,1.2*max(CbOutSim2))
plt.xlabel("Time [s]")
plt.ylabel("Concentration [g/L]")
plt.legend(bbox_to_anchor=(1.01, 0.99), loc='upper left', borderaxespad=0,fontsize=18)
plt.tight_layout()
plt.show()

plt.figure
plt.plot(tData,CTotalOutSim3,label="$\mathrm{Total_{SimData}}$",linewidth=2,color="orange")
plt.xlim(0,max(tData))
plt.ylim(0,1.2*max(CTotalOutSim3))
plt.xlabel("Time [s]")
plt.ylabel("Concentration [g/L]")
plt.legend(bbox_to_anchor=(1.01, 0.99), loc='upper left', borderaxespad=0,fontsize=18)
plt.tight_layout()
plt.show()

plt.figure
plt.plot(tData,CaOutSim4,label="$\mathrm{a_{SimData}}$",linewidth=2,color="blue")
plt.plot(tData,CbOutSim4,label="$\mathrm{b_{SimData}}$",linewidth=2,color="green")
plt.xlim(0,max(tData))
plt.ylim(0,1.2*max(CaOutSim4,CbOutSim4))
plt.xlabel("Time [s]")
plt.ylabel("Concentration [g/L]")
plt.legend(bbox_to_anchor=(1.01, 0.99), loc='upper left', borderaxespad=0,fontsize=18)
plt.tight_layout()
plt.show()