"""
Functions to provide support for the use of GCMC in OpenMM
These functions are not used during the simulation, but will
be relevant in setting up and processing results
"""

import numpy as np
from simtk.openmm import app


def flood_system(topology, positions, ff='tip3p', n=100, pdb='extrawats.pdb'):
    """
    Function to add water molecules to a topology, as extras for GCMC
    This is to avoid changing the number of particles throughout a simulation
    Instead, we can just switch between 'ghost' and 'real' waters...

    Notes
    -----
    Could do with a more elegant way to add many waters. Adding them one by one
    results in them all being added to different chains. This won't affect
    correctness, but doesn't look nice.

    Parameters
    ----------
    topology : simtk.openmm.app.Topology
        Topology of the initial system
    positions : simtk.unit.Quantity
        Atomic coordinates of the initial system
    ff : str
        Water forcefield to use. Currently the only option is 'tip3p'
        Should be the same as used for the solvent
    n : int
        Number of waters to add to the system
    pdb : str
        Name of the PDB file to write containing the updated system
        This will be useful for visualising the results obtained.

    Returns
    -------
    modeller.topology : simtk.openmm.app.Topology
        Topology of the system after modification
    modeller.positions : simtk.unit.Quantity
        Atomic positions of the system after modification
    ghosts : list
        List of the residue numbers (counting from 0) of the ghost
        waters added to the system.
    """
    # Create a Modeller instance of the system
    modeller = app.Modeller(topology=topology, positions=positions)
    # Load topology of water model - only TIP3P for now
    assert ff == 'tip3p', "Only TIP3P is currently supported!"
    water = app.PDBFile(file='tip3p.pdb')
    # Add multiple copies of the same water, then write out a pdb (for visualisation)
    ghosts = []
    for i in range(n):
        # Need a slightly more elegant way than this as each water is written to a different chain...
        modeller.add(addTopology=water.topology, addPositions=np.random.rand(3)*water.positions)
        ghosts.append(modeller.topology._numResidues - 1)
    pdbfile = open(pdb, 'w')
    water.writeFile(topology=modeller.topology, positions=modeller.positions, file=pdbfile)
    pdbfile.close()
    return modeller.topology, modeller.positions, ghosts

