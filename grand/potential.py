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
from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
from openmmtools import alchemy


def calc_mu(model, topology, positions, box_vectors, system_args, temperature=300*kelvin, sample_time=50*picoseconds,
            n_lambdas=16, n_samples=50, equil_time=1*nanosecond, dt=2*femtoseconds, integrator=None, platform=None):
    """
    Calculate the hydration free energy of water for a particular set of parameters

    Parameters
    ----------
    model : str
        Water model to use. Must be a model for which OpenMM has an XML file
    topology : simtk.openmm.app.Topology
        Topology of the water box
    positions : simtk.unit.Quantity
        Positions of the atoms in the system
    box_vectors :
        Periodic box vectors for the simulation
    system_args : dict
        Arguments (beyond topology) to use to create the System object
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

    Returns
    -------

    """
    kT = AVOGADRO_CONSTANT_NA * BOLTZMANN_CONSTANT_kB * temperature

    # Load force field and create System
    ff = ForceField('{}.xml'.format(model))
    system = ff.createSystem(topology, **system_args)
    system.addForce(MonteCarloBarostat(1*bar, temperature, 25))

    # Get the atom IDs of one water residue
    alchemical_ids = []
    for water in topology.residues():
        for atom in water.atoms():
            alchemical_ids.append(atom.index)

    # Create alchemical system
    alchemical_region = alchemy.AlchemicalRegion(alchemical_atoms=alchemical_ids)
    factory = alchemy.AbsoluteAlchemicalFactory()
    alchemical_system = factory.create_alchemical_system(system, alchemical_region)

    # Check integrator and platform, then create Simulation object
    if integrator is None:
        integrator = LangevinIntegrator(temperature, 1.0/picosecond, dt)
    if platform is None:
        platform = Platform.getPlatformByName('CPU')
    simulation = Simulation(topology, system, integrator, platform)

    # Update context
    simulation.context.setPositions(positions)
    simulation.context.setVelocitiesToTemperature(temperature)
    simulation.context.setPeriodicBoxVectors(box_vectors)

    # Minimise system
    simulation.minimizeEnergy(0.01*kilojoule_per_mole, 10000)

    # Get all variables for the free energy calculation ready
    equil_steps = int(round(equil_time / dt))  # Number of equilibration steps
    sample_steps = int(round(sample_time / dt))  # Number of sampling MD steps
    lambdas = np.linspace(1.0, 0.0, n_lambdas)  # Lambda values to use
    U = np.zeros((n_lambdas, n_lambdas, n_samples))  # Energy values calculated

    # Simulate the system at each lambda window
    for i in range(n_lambdas):
        # Set lambda value
        simulation.context.setParameter('lambda', lambdas[i])
        # Equilibrate system at this window
        simulation.step(equil_steps)
        for k in range(n_samples):
            # Run production MD
            simulation.step(sample_steps)
            # Calculate energy at each lambda value
            for j in range(n_lambdas):
                simulation.context.setParameter('lambda', lambdas[j])
                U[i, j, k] = simulation.context.getState(getEnergy=True).getPotentialEnergy() / kT
            # Reset lambda value
            simulation.context.setParameter('lambda', lambdas[i])

    # Calculate equilibration & number of uncorrelated samples
    N_k = np.zeros(n_lambdas)
    for i in range(n_lambdas):
        n_equil, g, neff_max = pymbar.timeseries.detectEquilibration(U[i, i, :])
        indices = pymbar.timeseries.subsampleCorrelatedData(U[i, i, :], g=g)
        N_k[i] = len(indices)
        U[i, :, 0:N_k[i]] = U[i, :, indices].T

    # Calculate free energy differences
    mbar = pymbar.MBAR(U, N_k)
    [deltaG_ij, ddeltaG_ij, theta_ij] = mbar.getFreeEnergyDifferences()

    # Return the matrix for now, need to extract the total free energy
    return deltaG_ij
