# Helpful scripts

Folder of helpful scripts which do not form part of the main package but may be useful for running certain type of simulations. Always manually check the outputs of these scripts.


`Whole_Protein_Sphere.py` will take a PDB and optionally a trajectory as input and provide recommendations of reference protein atoms to use as your sphere anchors as well as radius. 

`Getting_good_atoms_from_holo.py` will take a PDB file with a bound ligand and find two protein atoms where the center of geometry of the two protein atoms is closes to the center of the ligand. Note: the atom numbers that are printed will be in the numbering scheme of the supplied PDB. Your simulation ready structure may have different numbering. 

`SphereSetup.py` in this script you can enter the results of either the above two scripts and run it to get a static PDB file with a carbon atom placed at the coordinates you have chosen for visualisation before simulation. 