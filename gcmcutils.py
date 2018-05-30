# -*- coding: utf-8 -*-

"""
gcmcutils.py
Marley Samways

Description
-----------
Functions to provide support for the use of GCMC in OpenMM.
These functions are not used during the simulation, but will be relevant in setting up
simulations and processing results

Notes
-----
Need to add trajectory processing functions (to aid visualisation).
Also need to think of what else will need to be added here (think what's useful)
"""

import numpy as np
import mdtraj
import parmed
from simtk import unit
from simtk.openmm import app
from copy import deepcopy


def flood_system(topology, positions, ff='tip3p', n=100, pdb='gcmc-extra-wats.pdb'):
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
    water = app.PDBFile(file='{}.pdb'.format(ff.lower()))
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


def flood_system_parmed(prmtop, inpcrd, n=100, out='gcmc-extra-wats'):
    """
    Add a series of ghost waters to the structure of interest.
    This function was written because the previous approach does not work
    well when using AMBER simulation files, so this function uses ParmEd
    to achieve the same goal.
    
    Parameters
    ----------
    prmtop : str
        Name of the AMBER .prmtop file
    inpcrd : str
        Name of the AMBER coordinate file (in .rst7 format)
    n : int
        Number of water molecules to add to the system
    out : str
        Stem for the output file names
    
    Returns
    -------
    ghost_resids : list
        List of the reisdue IDs for the ghost waters added
    """
    # Read in structure
    struct = parmed.load_file(prmtop, xyz=inpcrd, structure=True)
    # Read in a template residue - to get water resname and number of atoms
    for residue in struct.residues:
        if residue.name in ["WAT", "HOH"]:
            water = deepcopy(residue)
            water_name = water.name
            water_natoms = len(water.atoms)
            break
    # Simulation box size
    box_size = struct.box[:3]
    # Need to iteratively add water molecules to the structure
    ghost_resids = []
    for i in range(n):
        # Add a water molecule to the structure
        struct += struct[":{}".format(water_name)][:water_natoms]
        # Need to translate the newly addedwater molecule to a random point
        new_centre = np.random.rand(3) * box_size
        old_coords = struct.coordinates
        new_coords = old_coords.copy()
        for j in range(-water_natoms, 0):
            new_coords[j,:] = old_coords[j,:] + new_centre - old_coords[-water_natoms,:]
        # Update structure coordinates
        struct.coordinates = new_coords
        ghost_resids.append(len(struct.residues))
    # Save a .prmtop and .inpcrd file
    struct.save("{}.prmtop".format(out))
    struct.save("{}.inpcrd".format(out), format='rst7')
    return ghost_resids


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

