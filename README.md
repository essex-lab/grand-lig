# gcmc-openmm

NOTE: This README is seriously out of date, so don't read it at the moment. It's on my to-do list to update this.

## Introduction

This private repository stores my attempts to run GCMC sampling of water molecules in the OpenMM simulation engine.
We currently have a basic framework to do this, but certain points need to be added/tweaked, and only very basic testing has been performed thus far.
The idea is to make this framework as generally transferable as possible, so that it can be easily integrated with other work based in OpenMM.

It should be noted that the current version of the code is not optimised for efficiency, so may not be as fast as possible.
This will come at a later date - when I am sure that the code works and have built up some test cases, I will then start optimising.

## Usage

The aim is for this code to be very simply incorporated into OpenMM scripts, without entailing an extensive knowledge of the underlying fundamentals behind GCMC simulations.

An example of how this code is shown in examples/bpti.py, where much of the code is as would be used in a normal OpenMM simulation script (though some parts are rearranged slightly, such as carrying out the MD portions in batches).
However, there are a number of extra lines introduced, and those are discussed here.

The first extra line we see is the use of the `add_ghosts()` function:
```python
pdb.topology, pdb.positions, ghosts = gcmc.utils.flood_system(pdb.topology, pdb.positions, n=25, pdb='bpti-gcmc.pdb')
```
This function is used to add 'ghost' water molecules to the topology of the system (these are switched on/off with insertions & deletions, as this is much easier than actually adding and removing particles from the system on the fly), and should be called after loading the PDB data.
This returns the list of residue IDs corresponding to ghost water molecules, which should be retained as these will need to be swtiched off before the simulation begins.
Additionally, the modified topology is written to a .pdb file, which may be useful in visualising the simulation data.

Next, we have the following section:
```python
ref_atoms = [['CA', 'TYR', '10'], ['C', 'ASN', '43']]
gcmc_mover = gcmc.sampler.GrandCanonicalMonteCarloSampler(system=system, topology=pdb.topology, temperature=300*kelvin,
                                                          referenceAtoms=ref_atoms, sphereRadius=4*angstroms)
```
Where we first define the reference atoms for the GCMC sphere, which will take its centre from the centre of geometry of these atoms.
The reference atoms are defined by their atom name, residue name and, optionally, a residue number (this is important if there are more than one of the residue of interest).
We are then able to initialise the `GrandCanonicalMonteCarloSampler` object, which is used to carry out the GCMC moves of the simulation, making sure to define the radius of the GCMC sphere.
In this example, we have minimally specified the system, topology, temperature, reference atoms and GCMC sphere radius for the simulation, though there are many other options allowing greater control (I'll add further documentation for this later, though the docstring for the `__init__()` method provides some extra description).

The next GCMC-specific portion includes the lines:
```python
simulation.context = gcmc_mover.prepareGCMCSphere(simulation.context, ghosts)
simulation.context = gcmc_mover.deleteWatersInGCMCSphere(simulation.context)
```
These lines update the simulation context by deleting ghost water molecules of interest.
The first of these is very important, as the ghost water molecules are liking clashing with the environment, and must be deleted, additionally, the `gcmc_mover.prepareGCMCSphere()` method sets up the harmonic restraints to keep water molecules in/out of the GCMC sphere, as appropriate.
The second line is not essential, and deletes the water molecules currently present in the GCMC region, and can be used to start from a clean slate in the region, if desired (this should prevent the observed solvation being biased by the starting configuration).

Given that we have 'dried' the GCMC region, here we want to equilibrate the system, to generate a suitable degree of hydration:
```python
for i in range(75):
    simulation.context = gcmc_mover.move(simulation.context, 200)  # 200 GCMC moves
    simulation.step(50)  # 100 fs propagation between moves
    print("\t{} GCMC moves completed. N = {}".format(gcmc_mover.n_moves, gcmc_mover.N))
```
The first line within the loop carries out a batch of 200 GCMC moves (the code is more efficient if the moves are carried out in batches), and the second executes 100 fs of molecular dynamics, in order to sample the system's other degrees of freedom between GCMC moves.
The print statement is used to monitor the occupancy of the GCMC region.

The simulation section below is similar to the previous section, but with a little more. Between these sections we also use the `gcmc_mover.reset()` method to reset the counting of accepted and total moves.
```python
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
```
Primarily, this includes the reporting of the current state to .dcd and .rst7 files, and importantly the use of the `gcmc_mover.report()` method.
This method writes out a list of ghost residue IDs to file, to aid in visualisation, as discussed below.

Finally, we use the line:
```python
gcmc.utils.remove_trajectory_ghosts('bpti-gcmc.pdb', 'bpti-gcmc.dcd', 'gcmc-ghost-wats.txt')
```
This line creates a new trajectory file, with the ghost water molecules displaced several unit cells away from their original positions.
This allows the user to visualise the water sampling in the region of interest without being hindered by the presence of ghost/dummy water molecules.

## To Do (Development)
- Make sure all parameters are correct (need to calculate hydration free energies for each model)
- Start generating some test results on previously used systems
- Optimise code as far as possible

