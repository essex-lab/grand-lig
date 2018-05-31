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
pdb.topology, pdb.positions, ghosts = gcmcutils.flood_system(pdb.topology, pdb.positions, n=25)
```

## To Do (Development)
- Make sure all parameters are correct (need to calculate hydration free energies for each model)
- Add option to prevent water leaks when not simulating at equilibrium
- Add support for spherical GCMC regions
- Start generating some test results on previously used systems
- Figure out a way to use AMBER input easily

