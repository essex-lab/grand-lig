import numpy as np
from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
from sys import stdout

import mdtraj
from parmed.openmm.reporters import RestartReporter

import gcmc

pdb = PDBFile('bpti-equil.pdb')
pdb.topology, pdb.positions, ghosts = gcmc.utils.flood_system(pdb.topology, pdb.positions, n=25, pdb='bpti-gcmc.pdb')

ff = ForceField('amber10.xml', 'tip3p.xml')
system = ff.createSystem(pdb.topology, nonbondedMethod=PME, nonbondedCutoff=10.0*angstroms,
         constraints=HBonds)

ref_atoms = [['CA', 'TYR', '10'], ['C', 'ASN', '43']]
gcmc_mover = gcmc.sampler.GrandCanonicalMonteCarloSampler(system=system, topology=pdb.topology, temperature=300*kelvin,
                                                          referenceAtoms=ref_atoms, sphereRadius=4*angstroms)

# Langevin integrator
integrator = LangevinIntegrator(300*kelvin, 1.0/picosecond, 0.002*picoseconds)

platform = Platform.getPlatformByName('CUDA')
simulation = Simulation(pdb.topology, system, integrator, platform)
simulation.context.setPositions(pdb.positions)
simulation.context.setVelocitiesToTemperature(300*kelvin)
simulation.context.setPeriodicBoxVectors(*pdb.topology.getPeriodicBoxVectors())

# reporters
dcd = mdtraj.reporters.DCDReporter('bpti-gcmc.dcd', 0)
rst7 = RestartReporter('bpti-gcmc.rst7', 0)

# Switch off ghost waters and in sphere
simulation.context = gcmc_mover.prepareGCMCSphere(simulation.context, ghosts)
simulation.context = gcmc_mover.deleteWatersInGCMCSphere(simulation.context)

# Equilibrate water distribution
print("GCMC equilibration...")
for i in range(75):
    simulation.context = gcmc_mover.move(simulation.context, 200)  # 200 GCMC moves
    simulation.step(50)  # 100 fs propagation between moves
    print("\t{} GCMC moves completed. N = {}".format(gcmc_mover.n_moves, gcmc_mover.N))

print("{}/{} moves accepted".format(gcmc_mover.n_accepted, gcmc_mover.n_moves))
simulation.reporters.append(StateDataReporter(stdout, 1000, step=True,
        potentialEnergy=True, temperature=True, volume=True))
gcmc_mover.reset()

# Run simulation
print("\n\nGCMC production")
for i in range(50):
    # Carry out 100 GCMC moves per 1 ps of MD
    simulation.step(500)
    simulation.context = gcmc_mover.move(simulation.context, 100)
    # Write data out
    state = simulation.context.getState(getPositions=True, getVelocities=True)
    dcd.report(simulation, state)
    rst7.report(simulation, state)
    gcmc_mover.report()
    print("\t{} GCMC moves completed. N = {}".format(gcmc_mover.n_moves, gcmc_mover.N))
print("{}/{} moves accepted".format(gcmc_mover.n_accepted, gcmc_mover.n_moves))

gcmc.utils.remove_trajectory_ghosts('bpti-gcmc.pdb', 'bpti-gcmc.dcd', 'gcmc-ghost-wats.txt')

