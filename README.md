# gcmc-openmm

## Intro

This private repo repo stores my attempts to run GCMC sampling of water molecules in the OpenMM simulation engine.
Currently have a basic framework to do this, but certain points need to be added/tweaked.

The idea is to make this as generally transferable as possible, so that it can be easily integrated with other work based in OpenMM.

The current version of the code is not optimised for efficiency, so may not be as fast as possible.
This will come at a later date - when I am sure that the code works and have built up some test cases, I will then start optimising.

## Usage

The aim is for this code to be very simply incorporated into OpenMM scripts, with minimal extra knowledge/effort necessary.

An example of how this code is shown in example.py, where much of the code is as would be used in a normal OpenMM simulation script.
However, there are a number of extra lines introduced, and those are discussed here.

The first line we see is the use of the flood\_system() function:
```python
pdb.topology, pdb.positions, ghosts = gcmc.utils.flood_system(pdb.topology, pdb.positions, n=25, pdb='bpti-gcmc.pdb')
```
This function is used to add 'ghost' water molecules to the topology of the system, and should be called after loading the PDB data.
This returns the list of residue IDs corresponding to ghost water molecules, whcih should be retained as these will need to be swtiched off before the simulation begins.
Additionally, the modified topology is written to a .pdb file, which may be useful in visualising the simulation data.

Next, we have the following section:
```python
ref_atoms = [['CA', 'TYR', '10'], ['C', 'ASN', '43']]
gcmc_box = np.array([7.0, 7.0, 7.0])*angstroms
gcmc_mover = gcmc.sampler.GrandCanonicalMonteCarloSampler(system=system, topology=pdb.topology, temperature=300*kelvin,
                                                          boxSize=gcmc_box, boxAtoms=ref_atoms)
```
Where we first define the reference atoms for the GCMC box, which will take its centre from the centre of geometry of these atoms, and the size of the cubic GCMC region (it is important that units are specified).
The reference atoms are defined by their atom name, residue name and, optionally, a residue number (this is important if there are more than one of the residue of interest), and it is important to include the units in the definition of the box size.
We are then able to initialise the GrandCanonicalMonteCarloSampler object, which is used to carry out the GCMC portions of the simulation.
In this example, we have minimally specified the system, topology, temperature, box size and reference atoms for the simulation, though there are many other options allowing finer control.

The next GCMC-specific portion includes the lines:
```python
simulation.context = gcmc_mover.deleteGhostWaters(simulation.context, ghosts)
simulation.context = gcmc_mover.deleteWatersInGCMCBox(simulation.context)
```
These lines update the simulation context by deleting water molecules of interest.
The first of these is very important, as the ghost water molecules are liking clashing with the environment, and must be deleted.
However, the second line deletes the water molecules currently present in the GCMC region, and can be used to start from a clean slate in the region, if desired.

Given that we have 'emptied' the GCMC region, here we want to equilibrate the system:
```python
for i in range(75):
    simulation.context = gcmc_mover.move(simulation.context, 200)  # 200 GCMC moves
    simulation.step(50)  # 100 fs propagation between moves
    print("\t{} GCMC moves completed. N = {}".format(gcmc_mover.n_moves, gcmc_mover.N))
```
The first line within the loop carries out a batch of 200 GCMC insertion/deletion moves, and the second executes 100 fs of molecular dynamics, in order to sample the system's other degrees of freedom between GCMC moves.
The print statement is used to monitor the occupancy of the GCMC region.

The simulation section below is similar to the previous section, but with a little more.
```python
for i in range(50):
    # Carry out 100 GCMC moves per 1 ps of MD
    simulation.step(500)
    simulation.context = gcmc_mover.move(simulation.context, 100)
    # Write data out
    state = simulation.context.getState(getPositions=True, getVelocities=True)
    dcd.report(simulation, state)
    rst7.report(simulation, state)
    gcmc_mover.writeFrame()
    print("\t{} GCMC moves completed. N = {}".format(gcmc_mover.n_moves, gcmc_mover.N))
print("{}/{} moves accepted".format(gcmc_mover.n_accepted, gcmc_mover.n_moves))
```
Primarily, this includes the reporting of the current state to .dcd and .rst7 files, and importantly the use of the gcmc\_mover.writeFrame() method.
This method writes out a list of ghost residues to file and also the coordinates of the GCMC box, to aid in visualisation, as discussed below.

Finally, we use the line:
```python
gcmc.utils.remove_trajectory_ghosts('bpti-gcmc.pdb', 'bpti-gcmc.dcd', 'gcmc-ghost-wats.txt')
```
This line creates a new trajectory file, with the ghost water molecules displaced several unit cells away from their original positions.
This allows the user to visualise the water smapling in the region of interest without being hindered by the present of ghost/dummy water molecules.

## To Do (Development)
- Make sure all parameters are correct (need to calculate hydration free energies for each model)
- Add option to prevent water leaks when not simulating at equilibrium
- Add support for spherical GCMC regions
- Start generating some test results on previously used systems
- Figure out a way to use AMBER input easily
- Optimise code as far as possible

