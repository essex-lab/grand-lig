# -*- coding: utf-8 -*-

"""
Description
-----------
Set of functions to calibrate the excess chemical potential and standard state volume of water

Marley Samways
"""

import numpy as np
import pymbar
import openmmtools
from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
from sys import stdout
import grand
import os
import collections
import copy

def get_lambda_values(lambda_in):
    """
    Calculate the lambda_sterics and lambda_electrostatics values for a given lambda.
    Electrostatics are decoupled from lambda=1 to 0.5, and sterics are decoupled from
    lambda=0.5 to 0.

    Parameters
    ----------
    lambda_in : float
        Input lambda value

    Returns
    -------
    lambda_vdw : float
        Lambda value for steric interactions
    lambda_ele : float
        Lambda value for electrostatic interactions
    """
    if lambda_in > 1.0:
        # Set both values to 1.0 if lambda > 1
        lambda_vdw = 1.0
        lambda_ele = 1.0
    elif lambda_in < 0.0:
        # Set both values to 0.0 if lambda < 0
        lambda_vdw = 0.0
        lambda_ele = 0.0
    else:
        # Scale values between 0 and 1
        lambda_vdw = min([1.0, 2.0*lambda_in])
        lambda_ele = max([0.0, 2.0*(lambda_in-0.5)])
    return lambda_vdw, lambda_ele


def calc_mu_ex(system, topology, positions, resname, resid, box_vectors, temperature, n_lambdas, n_samples, n_equil,
               log_file, pressure=None, turnOff=False):
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

    # Add barostat, if needed
    if pressure is not None:
        system.addForce(MonteCarloBarostat(pressure, temperature, 25))

    # Define the platform, first try CUDA, then OpenCL, then CPU
    try:
        platform = Platform.getPlatformByName('CUDA')
        platform.setPropertyDefaultValue('Precision', 'mixed')
    except:
        try:
            platform = Platform.getPlatformByName('OpenCL')
            #platform.setPropertyDefaultValue('Precision', 'mixed')
        except:
            platform = Platform.getPlatformByName('CPU')

    # Create a simulation object
    simulation = Simulation(topology, system, integrator, platform)
    simulation.context.setPositions(positions)
    simulation.context.setVelocitiesToTemperature(temperature)
    simulation.context.setPeriodicBoxVectors(*box_vectors)
    print('Simulation created')
    force_labels = {}
    for i, force in enumerate(system.getForces()):
        force_labels[force] = i
    # Separate all forces into separate force groups.
    for force_index, force in enumerate(system.getForces()):
        force.setForceGroup(force_index)
    new_sys = copy.deepcopy(system)
    inte_1 = VerletIntegrator(1.0 * femtosecond)
    contex_1 = Context(new_sys, inte_1)
    contex_1.setPositions(positions)
    energy_components = collections.OrderedDict()
    for force_label, force_index in force_labels.items():
        energy_components[force_label] = contex_1.getState(getEnergy=True,
                                                          groups=2 ** force_index).getPotentialEnergy()
    for key in energy_components.keys():
        print(f'{key} - {energy_components[key]}')

    print(f'Total energy = {simulation.context.getState(getEnergy=True).getPotentialEnergy()}')
    quit()

    # Make sure the GCMC sampler has access to the Context
    gcmc_mover.context = simulation.context

    lambdas = np.linspace(0.0, 1.0, n_lambdas)  # Lambda values to use
    if turnOff: # If turning off (removing) reverse the lambdas
        lambdas = np.linspace(1.0, 0.0, n_lambdas)
    U = np.zeros((n_lambdas, n_lambdas, n_samples))  # Energy values calculated
    simulation.reporters.append(StateDataReporter(stdout, 500, step=True,
        time=True, potentialEnergy=True, temperature=True, density=True, volume=True))

    # Simulate the system at each lambda window
    for i in range(n_lambdas):
        # Set lambda values
        print('Simulating at lambda = {:.4f}'.format(np.round(lambdas[i], 4)))
        gcmc_mover.logger.info('Simulating at lambda = {:.4f}'.format(np.round(lambdas[i], 4)))
        gcmc_mover.adjustSpecificMolecule(resid, lambdas[i])
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

    # Save the numpy matrix (for now)
    np.save('U_matrix.npy', U)

    # Calculate equilibration & number of uncorrelated samples
    N_k = np.zeros(n_lambdas, np.int32)
    for i in range(n_lambdas):
        n_equil, g, neff_max = pymbar.timeseries.detectEquilibration(U[i, i, :])
        indices = pymbar.timeseries.subsampleCorrelatedData(U[i, i, :], g=g)
        N_k[i] = len(indices)
        U[i, :, 0:N_k[i]] = U[i, :, indices].T

    # Calculate free energy differences
    mbar = pymbar.MBAR(U, N_k)
    results = mbar.getFreeEnergyDifferences()
    deltaG_ij = results[0]
    ddeltaG_ij = results[1]

    # Extract overall free energy change
    dG = deltaG_ij[0, -1]

    # Write out intermediate free energies
    for i in range(n_lambdas):
        dG_i = (deltaG_ij[0, i] * gcmc_mover.kT).in_units_of(kilocalorie_per_mole)
        gcmc_mover.logger.info('Free energy ({:.3f} -> {:.3f}) = {}'.format(lambdas[0], lambdas[i], dG_i))

    # Convert free energy to kcal/mol
    dG = (dG * gcmc_mover.kT).in_units_of(kilocalorie_per_mole)
    dG_err = (ddeltaG_ij[0, -1] * gcmc_mover.kT).in_units_of(kilocalorie_per_mole)

    gcmc_mover.logger.info('Excess chemical potential = {}'.format(dG))
    gcmc_mover.logger.info('Estimated error = {}'.format(dG_err))

    return dG


def calc_avg_volume(system, topology, positions, box_vectors, temperature, n_samples, n_equil):
    """
    Calculate the average volume of each species in a given system and parameters, this is the volume
    per molecule & will also return concentration

    Parameters
    ----------
    system : simtk.openmm.System
        System of interest
    topology : simtk.openmm.app.Topology
        Topology of the system
    positions : simtk.unit.Quantity
        Initial positions for the simulation
    box_vectors : simtk.unit.Quantity
        Periodic box vectors for the system
    temperature : simtk.unit.Quantity
        Temperature of the simulation
    n_samples : int
        Number of volume samples to collect
    n_equil : int
        Number of MD steps to run between each sample

    Returns
    -------
    avg_vol_dict : dict
        Dictionary storing the average volume per molecule for each species
    conc_dict : dict
        Dictionary storing the average concentration for each species (directly related to average volume, but
        returning both for convenience)
    """
    # Use the BAOAB integrator to sample the equilibrium distribution
    integrator = openmmtools.integrators.BAOABIntegrator(temperature, 1.0 / picosecond, 0.002 * picoseconds)

    # Define the platform, first try CUDA, then OpenCL, then CPU
    try:
        platform = Platform.getPlatformByName('CUDA')
        platform.setPropertyDefaultValue('Precision', 'mixed')
    except:
        try:
            platform = Platform.getPlatformByName('OpenCL')
            #platform.setPropertyDefaultValue('Precision', 'mixed')
        except:
            platform = Platform.getPlatformByName('CPU')

    # Create a simulation object
    simulation = Simulation(topology, system, integrator, platform)
    simulation.context.setPositions(positions)
    simulation.context.setVelocitiesToTemperature(temperature)
    simulation.context.setPeriodicBoxVectors(*box_vectors)

    # Count number of residues
    n_dict = {}  # Store the number of molecules for each species
    for residue in topology.residues():
        resname = residue.name
        # Create a dictionary if there is not already one
        if resname not in n_dict.keys():
            n_dict[resname] = 0

        # Add to the count
        n_dict[resname] += 1

    # Collect volume samples
    volume_list = []
    for i in range(n_samples):
        # Run a short amount of MD
        simulation.step(n_equil)
        # Calculate volume & then volume per molecule
        state = simulation.context.getState(getPositions=True)
        box_vectors = state.getPeriodicBoxVectors(asNumpy=True)
        volume = box_vectors[0, 0] * box_vectors[1, 1] * box_vectors[2, 2]
        volume_list.append(volume)

    # Calculate mean volume
    mean_volume = sum(volume_list) / len(volume_list)

    # Calculate average volume for each molecule of each species
    avg_vol_dict = {}
    for resname in n_dict.keys():
        avg_vol = mean_volume / n_dict[resname]
        avg_vol_dict[resname] = avg_vol

    # Calculate average concentration for each species
    conc_dict = {}
    for resname in n_dict.keys():
        n_moles = n_dict[resname] / AVOGADRO_CONSTANT_NA
        conc = n_moles / mean_volume
        conc_dict[resname] = conc

    return avg_vol_dict, conc_dict


def calc_mu_ex_independant(system, topology, positions, resname, resid, box_vectors, temperature, n_lambdas, n_samples, n_equil,
               log_file, pressure=None, turnOff=False):
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
    # for p in range(system.getNumParticles()):
    #     system.setParticleMass(p, 0.0)

    # Add barostat, if needed
    if pressure is not None:
        system.addForce(MonteCarloBarostat(pressure, temperature, 25))

    # Define the platform, first try CUDA, then OpenCL, then CPU
    try:
        platform = Platform.getPlatformByName('CUDA')
        platform.setPropertyDefaultValue('Precision', 'mixed')
    except:
        try:
            platform = Platform.getPlatformByName('OpenCL')
            #platform.setPropertyDefaultValue('Precision', 'mixed')
        except:
            platform = Platform.getPlatformByName('CPU')

    # Create a simulation object
    simulation = Simulation(topology, system, integrator, platform)
    simulation.context.setPositions(positions)
    simulation.context.setVelocitiesToTemperature(temperature)
    original_box_vectors = box_vectors
    simulation.context.setPeriodicBoxVectors(*original_box_vectors)
    print('Simulation created')
    '''
    force_labels = {}
    for i, force in enumerate(system.getForces()):
        force_labels[force] = i
    # Separate all forces into separate force groups.
    for force_index, force in enumerate(system.getForces()):
        force.setForceGroup(force_index)
    new_sys = copy.deepcopy(system)
    inte_1 = VerletIntegrator(1.0 * femtosecond)
    contex_1 = Context(new_sys, inte_1)
    contex_1.setPositions(positions)
    energy_components = collections.OrderedDict()
    for force_label, force_index in force_labels.items():
        energy_components[force_label] = contex_1.getState(getEnergy=True,
                                                          groups=2 ** force_index).getPotentialEnergy()
    for key in energy_components.keys():
        print(f'{key} - {energy_components[key]}')

    print(f'Total energy = {simulation.context.getState(getEnergy=True).getPotentialEnergy()}')
'''
    # Make sure the GCMC sampler has access to the Context
    gcmc_mover.context = simulation.context

    lambdas = np.linspace(0.0, 1.0, n_lambdas)  # Lambda values to use
    if turnOff: # If turning off (removing) reverse the lambdas
        lambdas = np.linspace(1.0, 0.0, n_lambdas)
    U = np.zeros((n_lambdas, n_lambdas, n_samples))  # Energy values calculated
    simulation.reporters.append(StateDataReporter(stdout, 1000, step=True,
        time=True, potentialEnergy=True, temperature=True, density=True, volume=True))

    simulation.saveState('Original_state.xml')
    # Simulate the system at each lambda window
    for i in range(n_lambdas):
        # Setting the positions, random velocities and box vectors so each lambda starts from the same
        simulation.loadState('Original_state.xml')
        print(simulation.context.getState(getPositions=True, enforcePeriodicBox=True).getPositions()[0])
        # Set lambda values
        print('Simulating at lambda = {:.4f}'.format(np.round(lambdas[i], 4)))
        gcmc_mover.logger.info('Simulating at lambda = {:.4f}'.format(np.round(lambdas[i], 4)))
        gcmc_mover.adjustSpecificMolecule(resid, lambdas[i])
        # Minimise
        simulation.minimizeEnergy()
        # Equilibrate
        print('Equilibrating at lambda = {}'.format(np.round(lambdas[i], 4)))
        n_steps = (2 * nanosecond) / (0.002 * picosecond)
        simulation.step(int(n_steps))
        print('Equil Done.. Simulation now')
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

    # Save the numpy matrix (for now)
    np.save('U_matrix.npy', U)

    # Calculate equilibration & number of uncorrelated samples
    N_k = np.zeros(n_lambdas, np.int32)
    for i in range(n_lambdas):
        n_equil, g, neff_max = pymbar.timeseries.detectEquilibration(U[i, i, :])
        indices = pymbar.timeseries.subsampleCorrelatedData(U[i, i, :], g=g)
        N_k[i] = len(indices)
        U[i, :, 0:N_k[i]] = U[i, :, indices].T

    # Calculate free energy differences
    mbar = pymbar.MBAR(U, N_k)
    results = mbar.getFreeEnergyDifferences()
    deltaG_ij = results[0]
    ddeltaG_ij = results[1]

    # Extract overall free energy change
    dG = deltaG_ij[0, -1]

    # Write out intermediate free energies
    for i in range(n_lambdas):
        dG_i = (deltaG_ij[0, i] * gcmc_mover.kT).in_units_of(kilocalorie_per_mole)
        gcmc_mover.logger.info('Free energy ({:.3f} -> {:.3f}) = {}'.format(lambdas[0], lambdas[i], dG_i))

    # Convert free energy to kcal/mol
    dG = (dG * gcmc_mover.kT).in_units_of(kilocalorie_per_mole)
    dG_err = (ddeltaG_ij[0, -1] * gcmc_mover.kT).in_units_of(kilocalorie_per_mole)

    gcmc_mover.logger.info('Excess chemical potential = {}'.format(dG))
    gcmc_mover.logger.info('Estimated error = {}'.format(dG_err))

    return dG