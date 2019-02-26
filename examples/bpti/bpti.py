"""
bpti.py
Marley Samways

Description
-----------
Example script of how to run GCMC in OpenMM for a BPTI system
"""

from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
from sys import stdout

import grand

pdb = PDBFile('bpti-equil.pdb')
pdb.topology, pdb.positions, ghosts = grand.utils.add_ghosts(pdb.topology, pdb.positions, n=5, pdb='bpti-gcmc.pdb')

ff = ForceField('amber10.xml', 'tip3p.xml')
system = ff.createSystem(pdb.topology, nonbondedMethod=PME, nonbondedCutoff=10.0*angstroms,
                         constraints=HBonds)

ref_atoms = [['CA', 'TYR', '10'], ['C', 'ASN', '43']]
gcmc_mover = grand.samplers.StandardGCMCSampler(system=system, topology=pdb.topology, temperature=300*kelvin,
                                                referenceAtoms=ref_atoms, sphereRadius=4*angstroms,
                                                dcd='bpti-raw.dcd', rst7='bpti-gcmc.rst7')

# Langevin integrator
integrator = LangevinIntegrator(300*kelvin, 1.0/picosecond, 0.002*picoseconds)

platform = Platform.getPlatformByName('OpenCL')
simulation = Simulation(pdb.topology, system, integrator, platform)
simulation.context.setPositions(pdb.positions)
simulation.context.setVelocitiesToTemperature(300*kelvin)
simulation.context.setPeriodicBoxVectors(*pdb.topology.getPeriodicBoxVectors())

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
trj = grand.utils.shift_ghost_waters(ghost_file='gcmc-ghost-wats.txt', topology='bpti-gcmc.pdb',
                                     trajectory='bpti-raw.dcd')
trj = grand.utils.recentre_traj(t=trj, resname='TYR', resid=10)
grand.utils.align_traj(t=trj, output='bpti-gcmc.dcd')

# Write out a trajectory of the GCMC sphere
grand.utils.write_sphere_traj(radius=4.0, ref_atoms=ref_atoms, topology='bpti-gcmc.pdb', trajectory='bpti-gcmc.dcd',
                              output='gcmc_sphere.pdb', initial_frame=True)
