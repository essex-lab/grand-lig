"""
Script to perform Monte Carlo resampling of an ensemble with AMOEBA

Marley Samways
"""

import argparse
import numpy as np
from copy import deepcopy
from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *


parser = argparse.ArgumentParser()
parser.add_argument("-p", "--pdbs", default=["frame1.pdb"], nargs="+",
                    help="List of PDB files describing the original ensemble")
parser.add_argument("-n", "--nsamples", default=1000, type=int,
                    help="Number of samples to create for the new ensemble")
parser.add_argument("-T", "--temperature", default=300.0, type=float,
                    help="Temperature in Kelvin")
parser.add_argument("--restart", default=False, action='store_true',
                    help="Restart from a previous run")
parser.add_argument("-o", "--output", default='resampl',
                    help="Output file stem.")
args = parser.parse_args()

# Calculate kT
kT = BOLTZMANN_CONSTANT_kB * AVOGADRO_CONSTANT_NA * args.temperature * kelvin

if args.restart:
    # Reload a previous run
    with open('{}_ensemble.dat'.format(args.output), 'r') as f:
        ensemble = []
        for line in f.readlines():
            ensemble.append(line.strip())
    # Load the last frame
    filename = ensemble[-1]
    pdb = PDBFile(filename)
else:
    # Load a random PDB file
    filename = args.pdbs[np.random.randint(len(args.pdbs))]
    pdb = PDBFile(filename)
    ensemble = [filename]  # Include first frame
    with open('{}_ensemble.dat'.format(args.output), 'w') as f:
        f.write("{}\n".format(ensemble[0]))

# Initialise the two ForceField 1 - fixed charge, 2 - AMOEBA
ff_fixed = ForceField('amber14-all.xml', "amber14/tip3p.xml")
ff_amoeba = ForceField("amoeba2013.xml")

#top_file = GromacsTopFile('processed.top')
#top_file.topology.setPeriodicBoxVectors(pdb.topology.getPeriodicBoxVectors())
#system_fixed = top_file.createSystem(nonbondedMethod=PME,
#                                     nonbondedCutoff=12.0*angstroms,
#                                     constraints=HBonds)


# Create the two systems
system_fixed = ff_fixed.createSystem(pdb.topology, nonbondedMethod=PME, nonbondedCutoff=10.0*angstroms,
                                     constraints=HBonds)
system_amoeba = ff_amoeba.createSystem(pdb.topology, nonbondedMethod=PME, nonbondedCutoff=7*angstroms,
                                       vdwCutoff=9*angstroms, rigidWater=False, polarization='mutual',
                                       mutualInducedTargetEpsilon=0.00001)

# Create fixed-charge simulation
simulation_fixed = Simulation(pdb.topology, system_fixed,
                              LangevinIntegrator(300*kelvin, 1.0/picosecond, 0.002*picoseconds),
                              Platform.getPlatformByName('CUDA'))
simulation_fixed.context.setPositions(pdb.positions)
simulation_fixed.context.setVelocitiesToTemperature(300*kelvin)
simulation_fixed.context.setPeriodicBoxVectors(*pdb.topology.getPeriodicBoxVectors())

# Create AMOEBA simulation
simulation_amoeba = Simulation(pdb.topology, system_amoeba,
                               LangevinIntegrator(300*kelvin, 1.0/picosecond, 0.002*picoseconds),
                               Platform.getPlatformByName('CUDA'))
simulation_amoeba.context.setPositions(pdb.positions)
simulation_amoeba.context.setVelocitiesToTemperature(300*kelvin)
simulation_amoeba.context.setPeriodicBoxVectors(*pdb.topology.getPeriodicBoxVectors())

# Create DCD reporter to save new trajectory
if args.restart:
    dcd_reporter = DCDReporter("{}_trj.dcd".format(args.output), 0, append=True)
else:
    dcd_reporter = DCDReporter("{}_trj.dcd".format(args.output), 0)

# Create variables to store statistics
n_accepted = 0

# Get energies and positions of the current state, for efficiency
state_fixed = simulation_fixed.context.getState(getEnergy=True)
state_amoeba = simulation_amoeba.context.getState(getEnergy=True, getPositions=True)
filename_old = filename
pdb_old = deepcopy(pdb)
energy_fixed_old = state_fixed.getPotentialEnergy()
energy_amoeba_old = state_amoeba.getPotentialEnergy()

if not args.restart:
    # Write out frame
    dcd_reporter.report(simulation_amoeba, state_amoeba)

# Carry out moves
for i in range(args.nsamples - len(ensemble)):
    #print(filename_old)
    #print('\tE_old (AMBER) = {}'.format(energy_fixed_old))
    #print('\tE_old (AMOEBA) = {}'.format(energy_amoeba_old))
    # Load a random PDB file
    filename_new = args.pdbs[np.random.randint(len(args.pdbs))]
    pdb_new = PDBFile(filename_new)

    # Update simulation Contexts with new positions
    simulation_fixed.context.setPositions(pdb_new.positions)
    simulation_fixed.context.setPeriodicBoxVectors(*pdb_new.topology.getPeriodicBoxVectors())
    simulation_amoeba.context.setPositions(pdb_new.positions)
    simulation_amoeba.context.setPeriodicBoxVectors(*pdb_new.topology.getPeriodicBoxVectors())

    # Get energies of the new state
    state_fixed = simulation_fixed.context.getState(getEnergy=True)
    state_amoeba = simulation_amoeba.context.getState(getEnergy=True, getPositions=True)
    energy_fixed_new = state_fixed.getPotentialEnergy()
    energy_amoeba_new = state_amoeba.getPotentialEnergy()

    #print(filename_new)
    #print('\tE_new (AMBER) = {}'.format(energy_fixed_new))
    #print('\tE_new (AMOEBA) = {}'.format(energy_amoeba_new))

    # Calculate acceptance probability
    dU_fixed = energy_fixed_new - energy_fixed_old
    dU_amoeba = energy_amoeba_new - energy_amoeba_old
    ddU = dU_amoeba - dU_fixed
    acc_prob = np.exp(-ddU/kT)

    #print('dU (AMBER) = {}'.format(dU_fixed))
    #print('dU (AMOEBA) = {}'.format(dU_amoeba))
    #print('ddU = {}'.format(ddU))
    #print('P = {}\n'.format(acc_prob))

    # Check whether to accept change
    if acc_prob > np.random.rand():
        # Move is accepted
        n_accepted += 1
        ensemble.append(filename_new)
        # Update variables
        filename_old = filename_new
        pdb_old = deepcopy(pdb_new)
        energy_fixed_old = deepcopy(energy_fixed_new)
        energy_amoeba_old = deepcopy(energy_amoeba_new)
    else:
        # Move is rejected
        ensemble.append(filename_old)
        # Update Contexts to previous state
        simulation_fixed.context.setPositions(pdb_old.positions)
        simulation_fixed.context.setPeriodicBoxVectors(*pdb_old.topology.getPeriodicBoxVectors())
        simulation_amoeba.context.setPositions(pdb_old.positions)
        simulation_amoeba.context.setPeriodicBoxVectors(*pdb_old.topology.getPeriodicBoxVectors())
        state_amoeba = simulation_amoeba.context.getState(getEnergy=True, getPositions=True)

    # Write out frame
    dcd_reporter.report(simulation_amoeba, state_amoeba)
    with open('{}_ensemble.dat'.format(args.output), 'a') as f:
        f.write("{}\n".format(ensemble[-1]))

print("{}/{} moves accepted".format(n_accepted, args.nsamples-1))

