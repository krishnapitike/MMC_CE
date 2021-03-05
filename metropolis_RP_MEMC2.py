#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/python
#This program runs Metropolis Monte Carlo simulations for high entropy ceramics
#Methodology is described in [Pitike et.al., Chem. Mater. 32, 7507 (2020)]
#Trail moves: Atomic swaps
#Energy model: cluster expansion as implemented in atat - parameters from DFT mixing enthalpies
#See atat manual for information: https://www.brown.edu/Departments/Engineering/Labs/avdw/atat/manual/manual.html
#Following input files are required for running memc2
# clusters.out
# eci.out
# gs_str.out
# lat.in
# str.out
# control.in
# output files written by memc2
# mc.out         : energy is written on second row 23rd column?
# mcheader.out   : read more information
# mcsnapshot.out : structure is written

import numpy as np
from math import exp
import random
import csv
import subprocess
import timeit
from os import path, remove, mkdir

# Atomic simulation environment for NN-list etc.
# see https://wiki.fysik.dtu.dk/ase/
# to install ASE through anaconda: https://anaconda.org/conda-forge/ase

import ase.io
import ase.neighborlist
from ase import atom

np.set_printoptions(precision=3)
kB          = 0.00008617      #Boltzmann constant in [ eV/K ]
searchDist  = 7.1      #angstrom
nn1D        = [3.6,4.0]
nn2D        = [5.3,5.7]
nn3D        = [6.7,7.1]
nn1coord    = 4  #4 first nearest neighbors (in plane)
nn2coord    = 4  #4 second nearest neighbors (in plane)
nn3coord    = 8  #4 third nearest neighbors (out of plane)
topdir      = './'
inputPFN    = topdir + 'strInit.out'
#system      = ase.io.read(inputPFN, format='vasp')
#nBsite      = system.get_number_of_atoms()
#nBsite      = 1280
#annealAfter = nBsite * (nBsite - 1) / 2

writeRejected   = 0  # 1 for true and 0 for false
writeAccepted   = 1  # 1 for true and 0 for false

#inputFiles      = [ 'FEbin.txt']
nSpecies        = 5  # Co Cu Mg Ni Zn
nPhases         = 1  # 0.Ruddlesden Popper

initSkipSteps   = 10000 #40000
allSkipSteps    = 6000 #10000
collectSteps    = 4001 #20001
iCheck          = 1000  #500

Tmin            = 300
Tmax            = 4000
Tstep           = 100
pO2             = 1

topdir          = './'
tempdirname     = topdir + 'tempDir/'
accepteddirname = tempdirname + 'accepted/'
rejecteddirname = tempdirname + 'rejected/'
outFormat       = 'cif'  # or cif, vasp, etc. for more formats see:


# In[2]:


# Safely creating directories
def createDirectory(tempdir):
    if not path.exists(tempdir):
        mkdir(tempdir)
        print ('Directory ', tempdir, ' Created ')
    else:
        print ('Directory ', tempdir, ' already exists')


# converting chemical symbols to unique integers
def chem2Index(chemSeq):
    speciesList = []
    for i in chemSeq:
        if i == 'Co':
            speciesList.append(0)
        elif i == 'Cu':
            speciesList.append(1)
        elif i == 'Mg':
            speciesList.append(2)
        elif i == 'Ni':
            speciesList.append(3)
        elif i == 'Zn':
            speciesList.append(4)
        #elif i == 'Ti':
        #    speciesList.append(5)
        #elif i == 'Zr':
        #    speciesList.append(6)
    return speciesList

def randomSwap2(chemSeq):
    idx = range(len(chemSeq))
    while (1) :                                          #This while loops until differenct species are selected
        i1, i2 = random.sample(idx, 2)
        if (chemSeq[i1]!=chemSeq[i2]):
            break                                        #Breaks the while loop when differenct species are selected
    chemSeq[i1], chemSeq[i2] = chemSeq[i2], chemSeq[i1]  #atoms are swapped
    return(chemSeq)                                      #swapped list is returned

def readStrInit():
    f = open("strInit.out", "r")
    strInit=[]
    chemSeq=[]
    it=0
    for line in f:
        line=line.split()
        strInit.append(line[:3])
        if (it>5):
            chemSeq.append(line[3])
        it += 1
    return(chemSeq,strInit)

def writeTrialStr(trialConfig):
    f = open("trial.out", "w")
    for it,val in enumerate(strInit):
        if (it<=5): line="  {:s}  {:s}  {:s}\n".format(val[0],val[1],val[2])
        else      : line="  {:s}  {:s}  {:s}  {:s}\n".format(val[0],val[1],val[2],trialConfig[it-6])
        f.write(line)
    return()

def getEnergyMEMC2(strfname, chemseq):
    
    #running memc2
    subprocess.run("memc2 -is=%s -n=0 -eq=0 -g2c -q -sigdig=10"%strfname, shell=True, check=True)

    #checking if the memc2 outputfile exists
    while True: 
        if path.exists('mc.out'): break

    #reading energy from mc.out and removing it
    f = open("mc.out", "r")
    f.readline()
    totalEnergy= float(f.readline().split()[23]) * nBsite
    f.close()
    if path.exists('mc.out'): remove('mc.out')

    #converting trial.out to ase system for getting short range order
    system      = ase.Atoms( chemseq, scaled_positions=coords, cell=lat, pbc=[1, 1, 1] )
    firstatom   = ase.neighborlist.neighbor_list('i', system, searchDist)  # array of first atom indices 0,1,2,3,4.....nBsite
    secondatom  = ase.neighborlist.neighbor_list('j', system, searchDist)  # array of second atom indices 0,1,2,3,4.....nBsite
    distance    = ase.neighborlist.neighbor_list('d', system, searchDist)  # distance
    speciesList = chem2Index(system.get_chemical_symbols())  # array of chemical species in whole numbers 0,1,2,3,4 -> Co, Cu, Mg, Ni, Zn
    #localEnergy = np.zeros(nPhases - 1)

    ## bifurcating the atoms list to in-plane and out-of-plane lists
    NN1 = []
    NN2 = []
    NN3 = []
    for tempI,tempD in enumerate(distance):
        if   tempD > nn1D[0] and tempD < nn1D[1]: NN1.append( [ firstatom[tempI], secondatom[tempI], tempD ] )
        elif tempD > nn2D[0] and tempD < nn2D[1]: NN2.append( [ firstatom[tempI], secondatom[tempI], tempD ] )
        elif tempD > nn3D[0] and tempD < nn3D[1]: NN3.append( [ firstatom[tempI], secondatom[tempI], tempD ] )
    ## NN1
    sopNN1 = np.zeros((nSpecies, nSpecies))  # short range order parameter
    if nBsite * 4 == len(NN1): #checking if all atoms have six first nearest neighbors
        for i,tempNN in enumerate(NN1):
            sopNN1[speciesList[tempNN[0]],speciesList[tempNN[1]]] += 1  # short range order parameter
    else: quit('Simulation ended due to bad trial move: problem with first in-plane NNs.')
    ## NN2
    sopNN2 = np.zeros((nSpecies, nSpecies))  # short range order parameter
    if nBsite * 4 == len(NN2): #checking if all atoms have six first nearest neighbors
        for i,tempNN in enumerate(NN2):
            sopNN2[speciesList[tempNN[0]],speciesList[tempNN[1]]] += 1  # short range order parameter
    else: quit('Simulation ended due to bad trial move: problem with first in-plane NNs.')
    ## NN3
    sopNN3 = np.zeros((nSpecies, nSpecies))  # short range order parameter
    if nBsite * 8 == len(NN3): #checking if all atoms have six first nearest neighbors
        for i,tempNN in enumerate(NN3):
            sopNN3[speciesList[tempNN[0]],speciesList[tempNN[1]]] += 1  # short range order parameter
    else: quit('Simulation ended due to bad trial move: problem with first in-plane NNs.')

    sopNN1      = sopNN1 * 5  / (nBsite * nn1coord)
    sopNN2      = sopNN2 * 5  / (nBsite * nn2coord)
    sopNN3      = sopNN3 * 5  / (nBsite * nn3coord)
    return( totalEnergy, sopNN1, sopNN2, sopNN3 )


# In[3]:


if __name__ == "__main__":

    #initializing the structure
    chemSeqInit, strInit = readStrInit()
    nBsite               = len(chemSeqInit)
    specIndex            = np.unique(chem2Index(chemSeqInit))
    
    # creating directories to store configurations
    for it in [tempdirname, accepteddirname, rejecteddirname]:
        createDirectory(it)

    # beginning of the simulation
    for T in range(Tmin, Tmax + Tstep, Tstep):

        # BEGIN BLOCK to setup filenames and write variables
        statsFN       = topdir + '%05d.stats.csv' % T  # file writing energies
        statsFile     = open(statsFN, 'w')
        statsWrite    = csv.writer(statsFile)
        short1FN      = topdir + '%05d.short1.csv' % T  # file writing short range order parameter
        short1File    = open(short1FN, 'w')
        short1Write   = csv.writer(short1File)
        short2FN      = topdir + '%05d.short2.csv' % T  # file writing short range order parameter
        short2File    = open(short2FN, 'w')
        short2Write   = csv.writer(short2File)
        short3FN      = topdir + '%05d.short3.csv' % T  # file writing short range order parameter
        short3File    = open(short3FN, 'w')
        short3Write   = csv.writer(short3File)
        ratioFN       = topdir + '%05d.ratio.csv' % T  # file writing acceptance ratio
        ratioFile     = open(ratioFN, 'w')
        ratioWrite    = csv.writer(ratioFile)
        xvalsFN       = topdir + '%05d.xvals.csv' % T  # file writing experimental conditions
        xvalsFile     = open(xvalsFN, 'w')
        xvalsWrite    = csv.writer(xvalsFile)

        # temperorily writing energy values
        energyFN      = topdir + '%05d.energy.csv' % T
        energyFile    = open(energyFN, 'w')
        energyWrite   = csv.writer(energyFile)

        #setup physical parameters
        beta          = 1 / (kB * T)
        totEn         = 0
        
        #initializing
        accRate       = 0
        tot           = 0
        
        # block to setup number of steps and thermalization steps for each temperature
        # For the first temperature simulation ; we run a lot of thermaization (skip) steps
        if T == Tmin:
            skipSteps = initSkipSteps
            maxSteps  = initSkipSteps + collectSteps
        else:
            skipSteps = allSkipSteps
            maxSteps  = allSkipSteps + collectSteps

        # converting lattice constants from str.out format to three vectors in angstroms
        lat0   = np.asarray(strInit[0:3],dtype=float)
        mult   = np.asarray(strInit[3:6],dtype=float)
        lat    = np.matmul( lat0, mult )
        coords = np.asarray(strInit[6:],dtype=float)
        # converting coorinates from str.out format to fractional
        for it in range(3): coords[:,it]=coords[:,it]/mult[it,it]

        #Monte Carlo steps begins here
        ( apE, sop1, sop2, sop3 ) = getEnergyMEMC2('strInit.out',chemSeqInit) #getEnergy(system)  #first step is always accepted
        accRate                  += 1
        #acceptedPhaseIndex = phaseUpdate()  # #Checking the phase composition

        #update accepted state
        acceptedConfig = chemSeqInit[:]

        #update stats
        stats    = apE
        short1   = sop1
        short2   = sop2
        short3   = sop3
        tot     += 1    #this is not done here for some reason
        
        #Monte Carlo steps after the initial step
        for i in range(maxSteps):
            print(i)
            acceptedConfigCopy = acceptedConfig[:]        #making a copy of configuration before trail move
            trialConfig = randomSwap2(acceptedConfigCopy) #making trial move: swapping two atoms
            writeTrialStr(trialConfig)                    #writing the trial structure for memc2
            ( tpE, tsop1, tsop2, tsop3 ) = getEnergyMEMC2('trial.out',trialConfig)  # #getting energy of the trial configuration
            if exp(-(tpE - apE) * beta) > random.random() or tpE < apE:
                apE, sop1, sop2, sop3 = tpE, tsop1, tsop2, tsop3
                acceptedConfig        = trialConfig
                accRate              += 1
            
            # temperorily writing energies
            energyWrite.writerow('{:3.4e}'.format(x) for x in [apE, tpE])
            
            #updating and writing stats
            if i >= skipSteps:
                stats  += apE
                short1 += sop1
                short2 += sop2
                short3 += sop3
                tot    += 1
                xvals   = [T, tot]

                if tot % iCheck == 0:
                    
                    #cast short range order parameter in a 1x25 array for easy csv printing
                    sop1List = []
                    sop2List = []
                    sop3List = []
                    for ix in specIndex:
                        for jx in specIndex:
                            sop1List.append(short1[ix, jx] / (tot) )
                            sop2List.append(short2[ix, jx] / (tot) )
                            sop3List.append(short3[ix, jx] / (tot) )
                    short1Write.writerow('{:3.4e}'.format(x) for x in sop1List)
                    short2Write.writerow('{:3.4e}'.format(x) for x in sop2List)
                    short3Write.writerow('{:3.4e}'.format(x) for x in sop3List)

                    #writing energies
                    statsList = [stats / (tot), stats ** 2 / (tot) ]
                    statsWrite.writerow('{:3.4e}'.format(x) for x in statsList)

                    #writing phases, acceptance rate and xvals

                    ratioWrite.writerow('{:3.4e}'.format(x) for x in [ accRate / (tot+skipSteps) ] )
                    xvalsWrite.writerow('{:3.4e}'.format(x) for x in xvals)

                    #flushing all files
                    short1File.flush()
                    short2File.flush()
                    short3File.flush()
                    statsFile.flush()
                    ratioFile.flush()
                    xvalsFile.flush()
                    energyFile.flush()

            #preiodically writing system in cif file
            if i == 0 or tot % iCheck == 0:
                system = ase.Atoms( acceptedConfig, scaled_positions=coords, cell=lat, pbc=[1, 1, 1] )
                system.write(accepteddirname + '%05d.stats.out'%T + '%010d.'%tot + outFormat, format=outFormat)

        #closing all outfiles
        short1File.close()
        short2File.close()
        short3File.close()
        statsFile.close()
        ratioFile.close()
        xvalsFile.close()
        energyFile.close()
        print (T, i)


# In[ ]:




