# gcmc-openmm

This private repo repo stores my attempts to run GCMC sampling of water molecules in the OpenMM simulation engine.
Currently have a basic framework to do this, but certain points need to be added/tweaked.

The idea is to make this as generally transferable as possible, so that it can be easily integrated with other work based in OpenMM.

### To Do
- Make sure all parameters are correct
- Allow the use of multiple atoms to define the centre of the GCMC box
- Allow more intuitive definition of reference atoms
- Add periodic boundary considerations
- Make sure there are no singularities involving ghost waters
- Write out which waters are ghosts
- Add support for other water models
- Add random rotation upon particle insertion
- Enforce units in all relevant parameters

Also need to do some rigorous testing at some point to make sure that everything has been implemented correctly.

