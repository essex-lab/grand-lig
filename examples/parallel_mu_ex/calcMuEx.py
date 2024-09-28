from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
from sys import stdout
import numpy as np
import openmmtools
import pymbar

import grandlig as grand

import argparse


parser = argparse.ArgumentParser()
parser.add_argument('-p', '--pdb', type=str)
parser.add_argument('-x', '--xml', type=str)
parser.add_argument("-lam", "--lam", help="Input the actual lambda value were at between 0-29", default=0, type=int)
args = parser.parse_args()

def calc_mu_ex_independant(system, topology, positions, resname, resid, box_vectors, temperature, n_lambdas, selected_lambda,
                           n_samples, n_equil, log_file, pressure=None):
    """
    Calculate the excess chemical potential of a molecule in a given system,
    as the hydration free energy, using MBAR

    Parameters
    ----------
    system : simtk.openmm.System
        System of interest
    topology : simtk.openmm.app.Topology
        Topology of the system
    positions : simtk.unit.Quantity
        Initial positions for the simulation
    resname : str
        Resname of the molecule to couple
    resid : int
        Resid of the residue to couple
    box_vectors : simtk.unit.Quantity
        Periodic box vectors for the system
    temperature : simtk.unit.Quantity
        Temperature of the simulation
    n_lambdas : int
        Number of lambda values
    n_samples : int
        Number of energy samples to collect at each lambda value
    n_equil : int
        Number of MD steps to run between each sample
    log_file : str
        Name of the log file to write out
    pressure : simtk.unit.Quantity
        Pressure of the simulation, will default to NVT

    Returns
    -------
    dG : simtk.unit.Quantity
        Calculated free energy value
    """
    # Use the BAOAB integrator to sample the equilibrium distribution
    integrator = openmmtools.integrators.BAOABIntegrator(temperature, 1.0/picosecond, 0.002*picoseconds)

    # Name the log file, if not already done
    if log_file is None:
        'dG.log'

    # Define a GCMC sampler object, just to allow easy switching of a water - won't use this to sample
    gcmc_mover = grand.samplers.BaseGrandCanonicalMonteCarloSampler(system=system, topology=topology,
                                                                    temperature=temperature, resname=resname,
                                                                    log=log_file,
                                                                    ghostFile='calc_mu-ghosts.txt',
                                                                    overwrite=True)
    # Remove unneeded ghost file
    os.remove('calc_mu-ghosts.txt')


    # Define the platform, first try CUDA, then OpenCL, then CPU

    platform = Platform.getPlatformByName('CUDA')
    platform.setPropertyDefaultValue('Precision', 'mixed')
    print(f'Running on {platform.getName()} platform.')

    # Create a simulation object
    simulation = Simulation(topology, system, integrator, platform)
    simulation.context.setPositions(positions)
    simulation.context.setVelocitiesToTemperature(temperature)
    original_box_vectors = box_vectors
    simulation.context.setPeriodicBoxVectors(*original_box_vectors)
    print('Simulation created')

    # Make sure the GCMC sampler has access to the Context
    gcmc_mover.context = simulation.context

    lambdas = np.linspace(0.0, 1.0, n_lambdas)  # Lambda values to use

    print(f"Lambdas = {lambdas}")

    i = selected_lambda
    U = np.zeros((n_lambdas, n_lambdas, n_samples))  # Energy values calculated
    simulation.reporters.append(StateDataReporter(stdout, 1000, step=True,
        time=True, potentialEnergy=True, temperature=True, density=True, volume=True))

    # Simulate the system at specified lambda window
    # Set lambda values
    print('Simulating at lambda = {:.4f}'.format(np.round(lambdas[i], 4)))
    gcmc_mover.logger.info('Simulating at lambda = {:.4f}'.format(np.round(lambdas[i], 4)))
    gcmc_mover.adjustSpecificMolecule(resid, lambdas[i])

    # Minimise
    simulation.minimizeEnergy()

    # NVT equil
    simulation.step(50000)

    # NPT equil
    system.addForce(MonteCarloBarostat(pressure, temperature, 25))
    simulation.context.reinitialize(preserveState=True)
    simulation.step(100000)
    simulation.step(500000)


    print('Equil Done.. Simulation now')
    if selected_lambda == 0 or selected_lambda == 19:
        simulation.reporters.append(DCDReporter(f"lambda_{selected_lambda}_{lambdas[i]}.dcd", 1000))

    for k in range(n_samples):
        # Run production MD
        simulation.step(n_equil)
        box_vectors = simulation.context.getState(getPositions=True).getPeriodicBoxVectors()
        volume = box_vectors[0][0] * box_vectors[1][1] * box_vectors[2][2]
        # Calculate energy at each lambda value
        for j in range(n_lambdas):
            # Set lambda value
            gcmc_mover.adjustSpecificMolecule(resid, lambdas[j])
            # Calculate energy
            U[i, j, k] = simulation.context.getState(getEnergy=True).getPotentialEnergy() / gcmc_mover.kT
        # Reset lambda value
        gcmc_mover.adjustSpecificMolecule(resid, lambdas[i])

    final_positions = simulation.context.getState(getPositions=True).getPositions()  # Get the initial positions

    PDBFile.writeFile(simulation.topology, final_positions, open("./FinalFrame.pdb", 'w'))
    # with open(f"./FinalFrame.pdb", 'w') as f:  # Write out the initial structure in OpenMM coords - useful
    #     gro.writeFile(positions=final_positions, topology=simulation.topology, file=f)

    # Save the numpy matrix (for now)
    np.save(f'U_matrix_{i}.npy', U)

    return None


# Load in PDB file
pdb = PDBFile(args.pdb)
# Create system
ff = ForceField('amber14/tip3p.xml', args.xml)


list_of_resis = []  # Get a list of resids so we can choose a random one to decouple
resname = "L02"

for residue in pdb.topology.residues():
    if residue.name == resname:
        list_of_resis.append(residue.index)

resid = np.random.choice(list_of_resis)


# Create system

system = ff.createSystem(pdb.topology,
                         nonbondedMethod=PME,
                         nonbondedCutoff=12.0*angstroms,
                         switchDistance=10.0*angstroms,
                         constraints=HBonds)


# Run free energy calculation using grand
calc_mu_ex_independant(system=system,
                       topology=pdb.topology,
                       positions=pdb.positions,
                       box_vectors=pdb.topology.getPeriodicBoxVectors(),
                       temperature=298.15*kelvin,
                       pressure=1*bar,
                       resname=resname,
                       resid=resid,
                       n_lambdas=20,
                       selected_lambda=args.lam,
                       n_samples=10000,
                       n_equil=200,
                       log_file="dG.txt")

