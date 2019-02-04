# gcmc-openmm

## Introduction

This private repository stores my attempts to run GCMC sampling of water molecules in the OpenMM simulation engine.
We currently have a basic framework to do this, but certain points need to be added/tweaked, and only basic testing has been performed thus far.
The idea is to make this framework as generally transferable as possible, so that it can be easily integrated with other work based in OpenMM.

It should be noted that the current version of the code is not optimised for efficiency, so may not be as fast as possible.
This will come at a later date - when I am sure that the code works and have built up some test cases, I will then start optimising.

## Usage

The aim is for this code to be very simply incorporated into OpenMM scripts, without entailing extensive knowledge of the underlying fundamentals behind GCMC simulations.

An example of how this code is shown in `examples/bpti.py`, where much of the code is as would be used in a normal OpenMM simulation script (though some parts are rearranged slightly, such as carrying out the MD portions in batches).
However, there are a number of extra lines introduced, and those are discussed here.

The first extra line we see is the use of the `grand.utils.add_ghosts()` function:
```python
pdb.topology, pdb.positions, ghosts = grand.utils.add_ghosts(pdb.topology, pdb.positions, n=5, pdb='bpti-gcmc.pdb')
```
This function is used to add 'ghost' water molecules to the topology of the system (these are switched on/off with insertions & deletions, as this is *much* easier than actually adding and removing particles from the system on the fly), and should be called after loading the PDB data.
This returns the list of residue IDs corresponding to ghost water molecules, which should be retained as these will need to be swtiched off before the simulation begins.
Additionally, the modified topology is written to a .pdb file, which may be useful in visualising the simulation data.

Next, we have the following section:
```python
ref_atoms = [['CA', 'TYR', '10'], ['C', 'ASN', '43']]
gcmc_mover = grand.samplers.StandardGCMCSampler(system=system, topology=pdb.topology, temperature=300*kelvin,
                                                referenceAtoms=ref_atoms, sphereRadius=4*angstroms,
                                                dcd='bpti-raw.dcd', rst7='bpti-gcmc.rst7')
```
Where we first define the reference atoms for the GCMC sphere, which will take its centre from the centre of geometry of these atoms.
The reference atoms are defined by their atom name, residue name and, optionally, a residue number (this is important if there are more than one of the residue of interest).
We are then able to initialise the `StandardGCMCSampler` object, which is used to carry out the GCMC moves of the simulation, making sure to define the radius of the GCMC sphere.
In this example, we have minimally specified the system, topology, temperature, reference atoms, GCMC sphere radius for the simulation (along with the DCD and Restart7 output file names), though there are many other options allowing greater control (I'll add further documentation for this later, though the docstring for the `__init__()` method provides some extra description).

The next GCMC-specific portion includes the lines:
```python
simulation.context = gcmc_mover.prepareGCMCSphere(simulation.context, ghosts)
simulation.context = gcmc_mover.deleteWatersInGCMCSphere(simulation.context)
```
These lines update the `Context` by deleting ghost water molecules of interest.
The first of these is very important, as the ghost water molecules are liking clashing with the environment, and must be deleted.
The second line is not essential, and deletes the water molecules currently present in the GCMC region, and can be used to start from a clean slate in the region, if desired (this *should* prevent the observed solvation being biased by the starting configuration).

Given that we have 'dried' the GCMC region, here we want to equilibrate the system, to generate a suitable degree of hydration:
```python
print("GCMC equilibration...")
for i in range(75):
    gcmc_mover.move(simulation.context, 200)  # 200 GCMC moves
    simulation.step(50)  # 100 fs propagation between moves
```
The first line within the loop carries out a batch of 200 GCMC moves (the code is more efficient if the moves are carried out in batches), and the second executes 100 fs of molecular dynamics, in order to sample the system's other degrees of freedom between GCMC moves.

The simulation section below is similar to the previous section, but with a little more. Between these sections we also use the `gcmc_mover.reset()` method to reset the counting of accepted and total moves.
```python
for i in range(50):
    # Carry out 100 GCMC moves per 1 ps of MD
    simulation.step(500)
    simulation.context = gcmc_mover.move(simulation.context, 100)
    # Write data out
    gcmc_mover.report(simulation)
print("{}/{} moves accepted".format(gcmc_mover.n_accepted, gcmc_mover.n_moves))
```
This includes the reporting of the current state to .dcd and .rst7 files, with the use of the `gcmc_mover.report()` method.
This method writes out a list of ghost residue IDs to file, to aid in visualisation, as discussed below.

Finally, we adjust the trajectory so that it looks nicer when visualising.
These lines will do the following: 
1) Move all non-interacting waters away from the system, so as not to interfere with visualisation.
2) Recentre the trajectory on a particular residue.
3) Align the trajectory on the protein.
```python
trj = grand.utils.shift_ghost_waters(ghost_file='gcmc-ghost-wats.txt', topology='bpti-gcmc.pdb',
                                     trajectory='bpti-raw.dcd')
trj = grand.utils.recentre_traj(t=trj, resname='TYR', resid=10)
grand.utils.align_traj(t=trj, output='bpti-gcmc.dcd')
```
This line creates a new trajectory file, which should be much easier to visualise.

## To Do
- Make sure all parameters are correct (need to calculate hydration free energies for each model)
- Start generating some test results on previously used systems
- Optimise code as far as possible
