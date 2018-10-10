# -*- coding: utf-8 -*-

"""
utils.py
Marley Samways

Description
-----------
Functions to provide support for the use of GCMC in OpenMM.
These functions are not used during the simulation, but will be relevant in setting up
simulations and processing results

Notes
-----
Need to think of what else will need to be added here (think what's useful)
"""

import os
import numpy as np
import mdtraj
import openmoltools
from simtk import unit
from simtk.openmm import app
from copy import deepcopy


def get_file(filename):
    """
    Get the absolute path of one of the data files included in the package

    Parameters
    ----------
    filename : str
        Name of the file

    Returns
    -------
    filepath : str
        Name of the file including the path
    """
    filepath = os.path.join(os.path.dirname(__file__), "data", filename)
    if os.path.isfile(filepath):
        return filepath
    else:
        raise Exception("{} does not exist. You may need to reinstall the code.".format(filepath))


def add_ghosts(topology, positions, ff='tip3p', n=100, pdb='gcmc-extra-wats.pdb'):
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
        Water forcefield to use. Currently the only options
        are 'tip3p', 'spce' or 'tip4pew'. Should be the same
        as used for the solvent
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
    # Read in simulation box size
    box_vectors = topology.getPeriodicBoxVectors()
    box_size = np.array([box_vectors[0][0]._value,
                         box_vectors[1][1]._value,
                         box_vectors[2][2]._value]) * unit.nanometer

    # Load topology of water model 
    assert ff.lower() in ['spce', 'tip3p', 'tip4pew'], "Water model must be SPCE, TIP3P or TIP4Pew!"
    water = app.PDBFile(file=get_file("{}.pdb".format(ff.lower())))

    # Add multiple copies of the same water, then write out a pdb (for visualisation)
    ghosts = []
    for i in range(n):
        # Need a slightly more elegant way than this as each water is written to a different chain...
        # Read in template water positions
        positions = water.positions
        # Need to translate the water to a random point in the simulation box
        new_centre = np.random.rand(3) * box_size
        new_positions = deepcopy(water.positions)
        for i in range(len(positions)):
            new_positions[i] = positions[i] + new_centre - positions[0]
        # Add the water to the model and include the resid in a list
        modeller.add(addTopology=water.topology, addPositions=new_positions)
        ghosts.append(modeller.topology._numResidues - 1)

    # Write the new topology and positions to a PDB file
    if pdb is not None:
        pdbfile = open(pdb, 'w')
        water.writeFile(topology=modeller.topology, positions=modeller.positions, file=pdbfile)
        pdbfile.close()

    return modeller.topology, modeller.positions, ghosts


def write_ligand_xml(mol2, frcmod, gaff, xml="ligand.xml"):
    """
    Create a .xml file for the ligand forcefield using openmoltools, from the .mol2,
    .frcmod and gaff.dat files supplied

    Parameters
    ----------
    mol2 : str
        Name of the ligand .mol2 file (in AMBER format, with GAFF atom types)
    frcmod : str
        Name of the .frcmod file for the ligand
    gaff : str
        Name of the gaff.dat file (including path)
    xml : str
        Name of the .xml file to write. Default is 'ligand.xml'
    """
    # Read in parameters
    ligand_parser = openmoltools.amber_parser.AmberParser()
    ligand_parser.parse_filenames([gaff, mol2, frcmod])

    # Generate .xml output and write to file
    stream = ligand_parser.generate_xml()
    with open(xml, 'w') as f:
        f.write(stream.read())

    return None


def remove_trajectory_ghosts(topology, trajectory, ghost_file, output="gcmc-traj.dcd"):
    """
    Translate all ghost waters in a trajectory out of the simulation box, to make
    visualisation clearer

    Parameters
    ----------
    topology : str
        Name of the topology/connectivity file (e.g. PDB, GRO, etc.)
    trajectory : str
        Name of the trajectory file (e.g. DCD, XTC, etc.)
    ghost_file : str
        Name of the file containing the ghost water residue IDs at each frame
    output : str
        Name of the file to which the new trajectory is written
    """
    # Read in residue IDs for the ghost waters in each frame
    ghost_resids = []
    with open(ghost_file, 'r') as f:
        for line in f.readlines():
            ghost_resids.append([int(resid) for resid in line.split(",")])

    # Read in trajectory data
    t = mdtraj.load(trajectory, top=topology, discard_overlapping_frames=False)

    # Identify which atoms need to be moved out of sight
    ghost_atom_ids = []
    for frame in range(len(ghost_resids)):
        atom_ids = []
        for i, residue in enumerate(t.topology.residues):
            if i in ghost_resids[frame]:
                for atom in residue.atoms:
                    atom_ids.append(atom.index)
        ghost_atom_ids.append(atom_ids)

    # Shift coordinates of ghost atoms by several unit cells and write out trajectory
    for frame, atom_ids in enumerate(ghost_atom_ids):
        for index in atom_ids:
            t.xyz[frame, index, :] += 3 * t.unitcell_lengths[frame, :]
    t.save(output)

    return None

