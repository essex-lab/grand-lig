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
from openmmtools.testsystems import WaterBox

import grand

water_box = WaterBox(box_edge=2.5*nanometer, model='tip3p')

water_box.topology, water_box.positions, ghosts = grand.utils.add_ghosts(water_box.topology, water_box.positions,
                                                                         n=5, pdb='water-ghosts.pdb')

ff = ForceField('tip3p.xml')
system = ff.createSystem(water_box.topology, nonbondedMethod=PME, nonbondedCutoff=10.0*angstroms,
                         constraints=HBonds)

gcmc_mover = grand.samplers.StandardGCMCSampler(system=system, topology=water_box.topology, temperature=300*kelvin,
                                                sphereRadius=2*angstroms, sphereCentre=[12.5, 12.5, 12.5]*angstroms,
                                                dcd='water-raw.dcd', rst7='water-gcmc.rst7')

# Langevin integrator
integrator = LangevinIntegrator(300*kelvin, 1.0/picosecond, 0.002*picoseconds)

platform = Platform.getPlatformByName('OpenCL')
simulation = Simulation(water_box.topology, system, integrator, platform)
simulation.context.setPositions(water_box.positions)
simulation.context.setVelocitiesToTemperature(300*kelvin)
simulation.context.setPeriodicBoxVectors(*water_box.topology.getPeriodicBoxVectors())

# Switch off ghost waters and in sphere
gcmc_mover.prepareGCMCSphere(simulation.context, ghosts)
gcmc_mover.deleteWatersInGCMCSphere()

# Equilibrate water distribution
print("GCMC equilibration...")
#for i in range(50):
#    gcmc_mover.move(simulation.context, 200)  # 200 GCMC moves
#    simulation.step(50)  # 100 fs propagation between moves
#print("{}/{} equilibration GCMC moves accepted. N = {}".format(gcmc_mover.n_accepted, gcmc_mover.n_moves, gcmc_mover.N))

# Add StateDataReporter for production
simulation.reporters.append(StateDataReporter(stdout, 1000, step=True, potentialEnergy=True, temperature=True,
                                              volume=True))
# Reset GCMC statistics
gcmc_mover.reset()

# Run simulation
print("\n\nGCMC production")
for i in range(50):
    # Carry out 100 GCMC moves per 1 ps of MD
    simulation.step(5)
    gcmc_mover.move(simulation.context, 100)
    # Write data out
    gcmc_mover.report(simulation)

# Format the trajectory
grand.utils.shift_ghost_waters(ghost_file='gcmc-ghost-wats.txt', topology='water-ghosts.pdb',
                               trajectory='water-raw.dcd', output='water-gcmc.dcd')

# Write out a trajectory of the GCMC sphere
grand.utils.write_sphere_traj(radius=2.0, sphere_centre=[12.5, 12.5, 12.5]*angstroms, topology='water-ghosts.pdb',
                              trajectory='water-gcmc.dcd', output='gcmc_sphere.pdb', initial_frame=True)
