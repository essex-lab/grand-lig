# gcmc-openmm

This private repo repo stores my attempts to run GCMC sampling of water molecules in the OpenMM simulation engine.
Currently have a basic framework to do this, but certain points need to be added/tweaked.

### To Do
- Make sure all parameters are correct
- Allow the use of multiple atoms to define the centre of the GCMC box
- Add periodic boundary considerations
- Make sure there are no singularities involving ghost waters
- Write out which waters are ghosts
- Add support for other water models
- Add random rotation upon particle insertion

Also need to do some rigorous testing at some point to make sure that everything has been implemented correctly.

