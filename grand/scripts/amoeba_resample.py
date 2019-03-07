# -*- coding: utf-8 -*-

"""
amoeba_resample.py
Marley Samways

Description
-----------
Script to perform Monte Carlo resampling of an ensemble with AMOEBA. This will generate a new ensemble
with the states distributed by their probabilities according to the AMOEBA force field.
"""

import argparse
import numpy as np
from copy import deepcopy
from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *


def create_fixed_charge_simulation(pdb, force_field, simulation_dict):
    """
    Create the necessary fixed charge simulation object for a system size which has not yet been seen

    Parameters
    ----------
    pdb :

    force_field :

    simulation_dict : dict

    """
    # Create system
    system = force_field.createSystem(pdb.topology, nonbondedMethod=PME, nonbondedCutoff=10.0 * angstroms,
                                      constraints=HBonds)

    # Create simulation
    simulation = Simulation(pdb.topology, system,
                            LangevinIntegrator(300 * kelvin, 1.0 / picosecond, 0.002 * picoseconds),
                            Platform.getPlatformByName('CUDA'))
    simulation.context.setPositions(pdb.positions)
    simulation.context.setVelocitiesToTemperature(300 * kelvin)
    simulation.context.setPeriodicBoxVectors(*pdb.topology.getPeriodicBoxVectors())

    # Add the simulation to the dictionary
    simulation_dict[len(pdb.positions)] = simulation
    return None


def create_amoeba_simulation(pdb, force_field, simulation_dict):
    """
    Create the necessary AMOEBA simulation object for a system size which has not yet been seen

    Parameters
    ----------
    pdb :

    force_field :

    simulation_dict : dict

    """
    # Create system
    system = force_field.createSystem(pdb.topology, nonbondedMethod=PME, nonbondedCutoff=7 * angstroms,
                                      vdwCutoff=9 * angstroms, rigidWater=False, polarization='mutual',
                                      mutualInducedTargetEpsilon=0.00001)

    # Create simulation
    simulation = Simulation(pdb.topology, system,
                                   LangevinIntegrator(300 * kelvin, 1.0 / picosecond, 0.002 * picoseconds),
                                   Platform.getPlatformByName('CUDA'))
    simulation.context.setPositions(pdb.positions)
    simulation.context.setVelocitiesToTemperature(300 * kelvin)
    simulation.context.setPeriodicBoxVectors(*pdb.topology.getPeriodicBoxVectors())

    # Add the simulation to the dictionary
    simulation_dict[len(pdb.positions)] = simulation
    return None


parser = argparse.ArgumentParser()
parser.add_argument("-p", "--pdbs", default=["frame1.pdb"], nargs="+",
                    help="List of PDB files describing the original ensemble")
parser.add_argument("-n", "--nsamples", default=1000, type=int,
                    help="Number of samples to create for the new ensemble")
parser.add_argument("-T", "--temperature", default=300.0, type=float,
                    help="Temperature in Kelvin")
parser.add_argument("-dN", "--deltaN", default=1, type=int,
                    help="Maximum difference in number of water molecules")
parser.add_argument("--restart", default=False, action='store_true',
                    help="Restart from a previous run")
parser.add_argument("-o", "--output", default='amoeba',
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

# Need to check the length of the PDB to separate ut the frames with different numbers of water molecules
pdb_len = len(pdb.positions)
position_lens = [pdb_len]

# Initialise the two ForceField 1 - fixed charge, 2 - AMOEBA
#ff_fixed = ForceField('amber14-all.xml', "amber14/tip3p.xml")
ff_fixed = ForceField('amber10.xml', "tip3p.xml")
ff_amoeba = ForceField("amoeba2013.xml")

# Create Simulation objects
simulations_fixed = {}
create_fixed_charge_simulation(pdb, ff_fixed, simulations_fixed)
simulations_amoeba = {}
create_fixed_charge_simulation(pdb, ff_fixed, simulations_amoeba)

# Create variables to store statistics
n_accepted = 0

# Get energies and positions of the current state, for efficiency
state_fixed = simulations_fixed[pdb_len].context.getState(getEnergy=True)
state_amoeba = simulations_amoeba[pdb_len].context.getState(getEnergy=True)
filename_old = filename
pdb_old = deepcopy(pdb)
energy_fixed_old = state_fixed.getPotentialEnergy()
energy_amoeba_old = state_amoeba.getPotentialEnergy()

# Carry out moves
for i in range(args.nsamples - len(ensemble)):
    # Load a random PDB file
    filename_new = args.pdbs[np.random.randint(len(args.pdbs))]
    pdb_new = PDBFile(filename_new)

    # Check the length of the PDB file, and if it's new, then create a new Simulation object to handle this
    pdb_len = len(pdb.positions)
    if pdb_len not in position_lens:
        # Create new simulation objects
        create_fixed_charge_simulation(pdb, ff_fixed, simulations_fixed)
        create_fixed_charge_simulation(pdb, ff_fixed, simulations_amoeba)
        # Update list of lengths
        position_lens.append(pdb_len)
    else:
        # Update the positions if an object already exists
        simulations_fixed[pdb_len].context.setPositions(pdb_new.positions)
        simulations_fixed[pdb_len].context.setPeriodicBoxVectors(*pdb_new.topology.getPeriodicBoxVectors())
        simulations_amoeba[pdb_len].context.setPositions(pdb_new.positions)
        simulations_amoeba[pdb_len].context.setPeriodicBoxVectors(*pdb_new.topology.getPeriodicBoxVectors())

    # Get energies of the new state
    state_fixed = simulations_fixed[pdb_len].context.getState(getEnergy=True)
    state_amoeba = simulations_amoeba[pdb_len].context.getState(getEnergy=True)
    energy_fixed_new = state_fixed.getPotentialEnergy()
    energy_amoeba_new = state_amoeba.getPotentialEnergy()

    # Calculate acceptance probability
    dU_fixed = energy_fixed_new - energy_fixed_old
    dU_amoeba = energy_amoeba_new - energy_amoeba_old
    ddU = dU_amoeba - dU_fixed
    acc_prob = np.exp(-ddU/kT)

    # Check whether to accept change
    if acc_prob > np.random.rand():
        # Move is accepted
        n_accepted += 1
        ensemble.append(filename_new)
        # Update variables
        filename_old = filename_new
        energy_fixed_old = deepcopy(energy_fixed_new)
        energy_amoeba_old = deepcopy(energy_amoeba_new)
    else:
        # Move is rejected
        ensemble.append(filename_old)

    # Write out frame
    with open('{}_ensemble.dat'.format(args.output), 'a') as f:
        f.write("{}\n".format(ensemble[-1]))

print("{}/{} moves accepted".format(n_accepted, args.nsamples-1))
