"""
scytalone.py
Marley Samways

Description
-----------
Example script of how to run GCMC in OpenMM for a scytalone dehydratase (SD) system
"""

from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
from sys import stdout

from openmmtools.integrators import BAOABIntegrator

import grand

# Load in AMBER files
prmtop = AmberPrmtopFile('scytalone.prmtop')
inpcrd = AmberInpcrdFile('scytalone-equil.inpcrd')
prmtop.topology.setPeriodicBoxVectors(inpcrd.boxVectors)

# Add water ghosts, which can be inserted
prmtop.topology, inpcrd.positions, ghosts = grand.utils.add_ghosts(prmtop.topology,
                                                                   inpcrd.positions,
                                                                   n=5,
                                                                   pdb='sd-ghosts.pdb')

# Write out AMBER files for the new PDB file (needed to retain the FF info)
new_prmtop, new_inpcrd = grand.utils.write_amber_input(pdb='sd-ghosts.pdb',
                                                       prepi='mq1.prepi',
                                                       frcmod='mq1.frcmod')

# Load in new AMBER files (containing ghosts)
prmtop2 = AmberPrmtopFile(new_prmtop)
inpcrd2 = AmberInpcrdFile(new_inpcrd)

# Create system
system = prmtop2.createSystem(nonbondedMethod=PME,
                              nonbondedCutoff=12.0*angstroms,
                              switchDistance=10.0*angstroms,
                              constraints=HBonds)

# Define reference atoms around which the GCMC sphere is based
ref_atoms = [{'name': 'OH', 'resname': 'TYR', 'resid': '23'},
             {'name': 'OH', 'resname': 'TYR', 'resid': '43'}]

# Create GCMC Sampler object
gcmc_mover = grand.samplers.StandardGCMCSampler(system=system,
                                                topology=prmtop2.topology,
                                                temperature=300*kelvin,
                                                referenceAtoms=ref_atoms,
                                                sphereRadius=4*angstroms,
                                                log='sd-gcmc.log',
                                                dcd='sd-raw.dcd',
                                                rst7='sd-gcmc.rst7',
                                                overwrite=False)

# BAOAB Langevin integrator (important)
integrator = BAOABIntegrator(300*kelvin, 1.0/picosecond, 0.002*picoseconds)

platform = Platform.getPlatformByName('CUDA')
platform.setPropertyDefaultValue('Precision', 'mixed')

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
print("{}/{} equilibration GCMC moves accepted. N = {}".format(gcmc_mover.n_accepted,
                                                               gcmc_mover.n_moves,
                                                               gcmc_mover.N))

# Add StateDataReporter for production
simulation.reporters.append(StateDataReporter(stdout,
                                              1000,
                                              step=True,
                                              potentialEnergy=True,
                                              temperature=True,
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

# Move ghost waters out of the simulation cell
trj = grand.utils.shift_ghost_waters(ghost_file='gcmc-ghost-wats.txt',
                                     topology='sd-ghosts.pdb',
                                     trajectory='sd-raw.dcd')

# Recentre the trajectory on a particular residue
trj = grand.utils.recentre_traj(t=trj, resname='TYR', resid=23)

# Align the trajectory to the protein
grand.utils.align_traj(t=trj, output='sd-gcmc.dcd')

# Write out a trajectory of the GCMC sphere
grand.utils.write_sphere_traj(radius=4.0,
                              ref_atoms=ref_atoms,
                              topology='sd-ghosts.pdb',
                              trajectory='sd-gcmc.dcd',
                              output='gcmc_sphere.pdb',
                              initial_frame=True)

