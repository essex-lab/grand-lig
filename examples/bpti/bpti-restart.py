"""
bpti-restart.py
Marley Samways

Description
-----------
Example script of how to run GCMC in OpenMM for a BPTI system, restarting from a previous simulation
"""

from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
from sys import stdout

from openmmtools.integrators import BAOABIntegrator

import grand

# Load in PDB file
pdb = PDBFile('bpti-rst.pdb')

# Shouldn't need to add ghosts as these can just be read in from before (all frames contained below)
ghosts = grand.utils.read_ghosts_from_file('gcmc-ghost-wats.txt')

# Create system
ff = ForceField('amber14-all.xml', 'amber14/tip3p.xml')
system = ff.createSystem(pdb.topology,
                         nonbondedMethod=PME,
                         nonbondedCutoff=12.0*angstroms,
                         switchDistance=10.0*angstroms,
                         constraints=HBonds)

# Define atoms around which the GCMC sphere is based
ref_atoms = [{'name': 'CA', 'resname': 'TYR', 'resid': '10'},
             {'name': 'CA', 'resname': 'ASN', 'resid': '43'}]

gcmc_mover = grand.samplers.StandardGCMCSphereSampler(system=system,
                                                      topology=pdb.topology,
                                                      temperature=300*kelvin,
                                                      ghostFile='gcmc-ghost-wats2.txt',
                                                      referenceAtoms=ref_atoms,
                                                      sphereRadius=4*angstroms,
                                                      log='bpti-gcmc2.log',
                                                      dcd='bpti-raw2.dcd',
                                                      rst='bpti-rst2.pdb',
                                                      overwrite=False)

# BAOAB Langevin integrator
integrator = BAOABIntegrator(300*kelvin, 1.0/picosecond, 0.002*picoseconds)

platform = Platform.getPlatformByName('CUDA')
platform.setPropertyDefaultValue('Precision', 'mixed')

simulation = Simulation(pdb.topology, system, integrator, platform)
simulation.context.setPositions(pdb.positions)
simulation.context.setVelocitiesToTemperature(300*kelvin)
simulation.context.setPeriodicBoxVectors(*pdb.topology.getPeriodicBoxVectors())

# Make sure the variables are all ready to run & switch of the ghosts from the final frame of the previous run
gcmc_mover.initialise(simulation.context, ghosts[-1])

# Add StateDataReporter for production
simulation.reporters.append(StateDataReporter(stdout,
                                              1000,
                                              step=True,
                                              potentialEnergy=True,
                                              temperature=True,
                                              volume=True))

# Run simulation - jump straight into production, assuming we're restarting from an equilibration run
print("\n\nGCMC production")
for i in range(50):
    # Carry out 100 GCMC moves per 1 ps of MD
    simulation.step(500)
    gcmc_mover.move(simulation.context, 100)
    # Write data out
    gcmc_mover.report(simulation)

# Shift ghost waters outside the simulation cell
trj = grand.utils.shift_ghost_waters(ghost_file='gcmc-ghost-wats2.txt',
                                     topology='bpti-rst.pdb',
                                     trajectory='bpti-raw2.dcd')

# Centre the trajectory on a particular residue
trj = grand.utils.recentre_traj(t=trj, resname='TYR', resid=10)

# Align the trajectory to the protein
grand.utils.align_traj(t=trj, output='bpti-gcmc2.dcd')

# Write out a PDB trajectory of the GCMC sphere
grand.utils.write_sphere_traj(radius=4.0,
                              ref_atoms=ref_atoms,
                              topology='bpti-rst.pdb',
                              trajectory='bpti-gcmc2.dcd',
                              output='gcmc_sphere2.pdb',
                              initial_frame=True)
