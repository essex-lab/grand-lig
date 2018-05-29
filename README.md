# gcmc-openmm

## Intro

This private repo repo stores my attempts to run GCMC sampling of water molecules in the OpenMM simulation engine.
Currently have a basic framework to do this, but certain points need to be added/tweaked.

The idea is to make this as generally transferable as possible, so that it can be easily integrated with other work based in OpenMM.

The current version of the code is not optimised for efficiency, so may not be as fast as possible.
This will come at a later date - when I am sure that the code works and have built up some test cases, I will then start optimising.

## Usage

The aim is for this code to be very simply incorporated into OpenMM scripts, with minimal extra knowledge/effort necessary.

*Need to write a mini-tutorial on how this fits in with OpenMM here...*

## To Do (Development)
- Make sure all parameters are correct (need to calculate hydration free energies for each model)
- Add support for spherical GCMC regions
- Start generating some test results on previously used systems

