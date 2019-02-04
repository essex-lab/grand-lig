from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
from sys import stdout
import os

import mdtraj
import numpy as np
from parmed.openmm.reporters import RestartReporter

import grand

prmtop = AmberPrmtopFile('scytalone.prmtop')
inpcrd = AmberInpcrdFile('scytalone-equil.inpcrd')
prmtop.topology.setPeriodicBoxVectors(inpcrd.boxVectors)

prmtop.topology, inpcrd.positions, ghosts = grand.utils.add_ghosts(prmtop.topology, inpcrd.positions, n=10,
                                                                   pdb='sd-ghosts.pdb')

new_prmtop, new_inpcrd = grand.utils.write_amber_input(pdb='sd-ghosts.pdb', prepi='mq1.prepi',
                                                       frcmod='mq1.frcmod')

prmtop2 = AmberPrmtopFile(new_prmtop)
inpcrd2 = AmberInpcrdFile(new_inpcrd)

system = prmtop2.createSystem(nonbondedMethod=PME, nonbondedCutoff=12.0*angstroms, constraints=HBonds,
                              switchDistance=10*angstroms)

ref_atoms = [['OH', 'TYR', '23'], ['OH', 'TYR', '43']]
gcmc_mover = grand.samplers.StandardGCMCSampler(system=system, topology=prmtop2.topology, temperature=300*kelvin,
                                                referenceAtoms=ref_atoms, sphereRadius=4*angstroms,
                                                dcd='sd-raw.dcd', rst7='sd-gcmc.rst7')

# Langevin integrator
integrator = LangevinIntegrator(300*kelvin, 1.0/picosecond, 0.002*picoseconds)

platform = Platform.getPlatformByName('OpenCL')
simulation = Simulation(prmtop2.topology, system, integrator, platform)
simulation.context.setPositions(inpcrd2.positions)
simulation.context.setVelocitiesToTemperature(300*kelvin)
simulation.context.setPeriodicBoxVectors(*inpcrd2.boxVectors)

# Switch off ghost waters and in sphere
gcmc_mover.prepareGCMCSphere(simulation.context, ghosts)
gcmc_mover.deleteWatersInGCMCSphere()

# Equilibrate water distribution
print("GCMC equilibration...")
for i in range(75):
    gcmc_mover.move(simulation.context, 200)  # 200 GCMC moves
    simulation.step(50)  # 100 fs propagation between moves
print("{}/{} equilibration GCMC moves accepted. N = {}".format(gcmc_mover.n_accepted, gcmc_mover.n_moves, gcmc_mover.N))

# Add StateDataReporter for production
simulation.reporters.append(StateDataReporter(stdout, 1000, step=True, potentialEnergy=True, temperature=True,
                                              volume=True))
# Reset GCMC statistics
gcmc_mover.reset()

# Run simulation
print("\n\nGCMC production")
for i in range(50):
    # Carry out 100 GCMC moves per 1 ps of MD
    simulation.step(500)
    gcmc_mover.move(simulation.context, 100)
    # Write data out
    gcmc_mover.report(simulation)

# Format the trajectory
trj = grand.utils.shift_ghost_waters(ghost_file='gcmc-ghost-wats.txt', topology='sd-ghosts.pdb',
                                     trajectory='sd-raw.dcd')
trj = grand.utils.recentre_traj(t=trj, resname='TYR', resid=23)
grand.utils.align_traj(t=trj, output='scytalone-gcmc.dcd')

# Write out a trajectory of the GCMC sphere
grand.utils.write_sphere_traj(ref_atoms=ref_atoms, radius=4.0, topology='sd-ghosts.pdb', trajectory='sd-gcmc.dcd',
                              output='gcmc_sphere.pdb', initial_frame=True)

