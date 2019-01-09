# -*- coding: utf-8 -*-

"""
potential.py
Marley Samways

Description
-----------
Set of functions to calculate the excess chemical potential for water by calculating the hydration free energy of
a specific system setup
"""

import numpy as np
import pymbar
import openmmtools
from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *


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
    lambda_sterics : float
        Lambda value for steric interactions
    lambda_ele : float
        Lambda value for electrostatic interactions
    """
    lambda_sterics = min([1.0, 2.0*lambda_in])
    lambda_ele = max([0.0, 2.0*(lambda_in-0.5)])
    return lambda_sterics, lambda_ele


def calc_mu(model, box_len, cutoff, switch_dist, nb_method=PME, temperature=300*kelvin, sample_time=50*picoseconds,
            n_lambdas=16, n_samples=50, equil_time=1*nanosecond, dt=2*femtoseconds, integrator=None, platform=None,
            args={}):
    """
    Calculate the hydration free energy of water for a particular set of parameters

    Parameters
    ----------
    model : str
        Name of the water model to be simulated
    box_len : simtk.unit.Quantity
        Length of the water box
    cutoff : simtk.unit.Quantity
        Cutoff to be used
    switch_dist : simtk.unit.Quantity
        Distance at which to start switching Lennard-Jones interactions
    nb_method : (any suitable long-range nonbonded method)
        Method to use for long range interactions, default is PME
    temperature : simtk.unit.Quantity
        Temperature to run the simulation at, if None, will run at 300 K
    sample_time : simtk.unit.Quantity
        Amount of time to simulate for each sample of each lambda window. Each window
        will be simulated for n_samples*lambda_time
    n_lambdas : int
        Number of lambda values to simulate
    n_samples : int
        Number of samples to collect for each lambda value
    equil_time : simtk.unit.Quantity
        Time to equilibrate the system at each lambda window
    dt : simtk.unit.Quantity
        Timestep value
    integrator : (valid OpenMM integrator)
        Any valid OpenMM Integrator to use for the simulation. If None, will use the
        plain LangevinIntegrator
    platform : simtk.openmm.Platform
        OpenMM Platform object to use for the simulation, if None, will use CPU
    args : dict
        Any additional arguments to pass into creating the WaterBox object

    Returns
    -------

    """
    kT = AVOGADRO_CONSTANT_NA * BOLTZMANN_CONSTANT_kB * temperature

    print('Building water box...')
    # Load water box from openmmtools.testsystems
    water_box = openmmtools.testsystems.WaterBox(box_edge=box_len, cutoff=cutoff, model=model,
                                                 switch_width=cutoff-switch_dist, nonbondedMethod=nb_method,
                                                 dispersion_correction=False, **args)
    # Add a barostat to the system
    water_box.system.addForce(MonteCarloBarostat(1*bar, temperature, 25))

    # Get the atom IDs of one water residue
    alchemical_ids = []
    for water in water_box.topology.residues():
        for atom in water.atoms():
            alchemical_ids.append(atom.index)
        break
    print('Alchemical atoms: {}'.format(alchemical_ids))

    # Create alchemical system
    alchemical_region = openmmtools.alchemy.AlchemicalRegion(alchemical_atoms=alchemical_ids)
    factory = openmmtools.alchemy.AbsoluteAlchemicalFactory()
    alchemical_system = factory.create_alchemical_system(water_box.system, alchemical_region)
    alchemical_state = openmmtools.alchemy.AlchemicalState.from_system(alchemical_system)

    # Check integrator and platform, then create Simulation object
    if integrator is None:
        integrator = LangevinIntegrator(temperature, 1.0/picosecond, dt)
    if platform is None:
        platform = Platform.getPlatformByName('CPU')
    #simulation = Simulation(water_box.topology, alchemical_system, integrator, platform)
    context = Context(alchemical_system, integrator)

    # Update context
    #simulation.context.setPositions(water_box.positions)
    #simulation.context.setVelocitiesToTemperature(temperature)
    #simulation.context.setPeriodicBoxVectors(box_vectors)
    context.setPositions(water_box.positions)
    context.setVelocitiesToTemperature(temperature)

    # Minimise system
    print("Minimising system...")
    LocalEnergyMinimizer.minimize(context)

    # Get all variables for the free energy calculation ready
    equil_steps = int(round(equil_time / dt))  # Number of equilibration steps
    sample_steps = int(round(sample_time / dt))  # Number of sampling MD steps
    lambdas = np.linspace(1.0, 0.0, n_lambdas)  # Lambda values to use
    U = np.zeros((n_lambdas, n_lambdas, n_samples))  # Energy values calculated

    # Simulate the system at each lambda window
    for i in range(n_lambdas):
        # Set lambda values
        print("Simulating lambda = {}".format(np.round(lambdas[i], 4)))
        alchemical_state.lambda_sterics, alchemical_state.lambda_electrostatics = get_lambda_values(lambdas[i])
        alchemical_state.apply_to_context(context)
        print('vdW = {:.3f},  Ele = {:.3f}'.format(alchemical_state.lambda_sterics,
                                                   alchemical_state.lambda_electrostatics))
        # Equilibrate system at this window
        integrator.step(equil_steps)
        for k in range(n_samples):
            # Run production MD
            integrator.step(sample_steps)
            # Calculate energy at each lambda value
            for j in range(n_lambdas):
                # Set lambda value
                alchemical_state.lambda_sterics = get_lambda_values(lambdas[j])[0]
                alchemical_state.lambda_electrostatics = get_lambda_values(lambdas[j])[1]
                alchemical_state.apply_to_context(context)
                # Calculate energy
                U[i, j, k] = context.getState(getEnergy=True).getPotentialEnergy() / kT
            # Reset lambda value
            alchemical_state.lambda_sterics = get_lambda_values(lambdas[i])[0]
            alchemical_state.lambda_electrostatics = get_lambda_values(lambdas[i])[1]
            #print('vdW = {}'.format(alchemical_state.lambda_sterics))
            #print('Ele = {}'.format(alchemical_state.lambda_electrostatics))
            alchemical_state.apply_to_context(context)
    print('U = {}'.format(U))

    # Calculate equilibration & number of uncorrelated samples
    N_k = np.zeros(n_lambdas, np.int32)
    for i in range(n_lambdas):
        n_equil, g, neff_max = pymbar.timeseries.detectEquilibration(U[i, i, :])
        indices = pymbar.timeseries.subsampleCorrelatedData(U[i, i, :], g=g)
        N_k[i] = len(indices)
        U[i, :, 0:N_k[i]] = U[i, :, indices].T

    print('N = {}'.format(N_k))

    # Calculate free energy differences
    mbar = pymbar.MBAR(U, N_k)
    [deltaG_ij, ddeltaG_ij, theta_ij] = mbar.getFreeEnergyDifferences()
    dG = -deltaG_ij[0, -1]

    # Convert free energy to kcal/mol
    dG = (dG * kT).in_units_of(kilocalorie_per_mole)

    return dG
