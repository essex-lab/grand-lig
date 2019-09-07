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
from simtk import unit
from simtk.openmm import app
from copy import deepcopy
from scipy.cluster import hierarchy


def get_data_file(filename):
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
        raise Exception("{} does not exist!".format(filepath))


def add_ghosts(topology, positions, ff='tip3p', n=10, pdb='gcmc-extra-wats.pdb'):
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
    water = app.PDBFile(file=get_data_file("{}.pdb".format(ff.lower())))

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

    # Get chain IDs of new waters
    new_chains = []
    for res_id, residue in enumerate(modeller.topology.residues()):
        if res_id in ghosts:
            new_chains.append(chr(ord('a') + residue.chain.index).upper())

    # Write the new topology and positions to a PDB file
    if pdb is not None:
        with open(pdb, 'w') as f:
            water.writeFile(topology=modeller.topology, positions=modeller.positions, file=f)

        # Want to correct the residue IDs of the added waters as this can sometimes cause issues
        with open(pdb, 'r') as f:
            lines = f.readlines()

        max_resid = ghosts[0] + 1  # Start the new resids at the first ghost resid (+1)
        with open(pdb, 'w') as f:
            for line in lines:
                # Automatically write out non-atom lines
                if not any([line.startswith(x) for x in ['ATOM', 'HETATM', 'TER']]):
                    f.write(line)
                else:
                    # Correct the residue ID if this corresponds to an added water
                    if line[21] in new_chains:
                        f.write("{}{:4d}{}".format(line[:22],
                                                   (max_resid % 10000) + 1,
                                                   line[26:]))
                    else:
                        f.write(line)

                    # Need to change the resid if there is a TER line
                    if line.startswith('TER'):
                        max_resid += 1

    return modeller.topology, modeller.positions, ghosts


def remove_ghosts(topology, positions, ghosts=None, pdb='gcmc-removed-ghosts.pdb'):
    """
    Function to remove ghost water molecules from a topology, after a simulation.
    This is so that a structure can then be used to run further analysis without ghost
    waters disturbing the system.

    Parameters
    ----------
    topology : simtk.openmm.app.Topology
        Topology of the initial system
    positions : simtk.unit.Quantity
        Atomic coordinates of the initial system
    ghosts : list
        List of residue IDs for the ghost waters to be deleted
    pdb : str
        Name of the PDB file to write containing the updated system
        This will be useful for visualising the results obtained.

    Returns
    -------
    modeller.topology : simtk.openmm.app.Topology
        Topology of the system after modification
    modeller.positions : simtk.unit.Quantity
        Atomic positions of the system after modification
    """
    # Do nothing if no ghost waters are specified
    if ghosts is None:
        raise Exception("No ghost waters defined! Nothing to do.")

    # Create a Modeller instance
    modeller = app.Modeller(topology=topology, positions=positions)

    # Find the residues which need to be removed, and delete them
    delete_waters = []  # Residue objects for waters to be deleted
    for resid, residue in enumerate(modeller.topology.residues()):
        if resid in ghosts:
            delete_waters.append(residue)
    modeller.delete(toDelete=delete_waters)

    # Save PDB file
    if pdb is not None:
        tip3p_pdb = app.PDBFile(file=get_data_file("tip3p.pdb"))  # Need to start with a PDB so will use the TIP3P one..
        pdbfile = open(pdb, 'w')
        tip3p_pdb.writeFile(topology=modeller.topology, positions=modeller.positions, file=pdbfile)
        pdbfile.close()

    return modeller.topology, modeller.positions


def write_amber_input(pdb, protein_ff="leaprc.protein.ff14SB", ligand_ff="leaprc.gaff",
                      water_ff="leaprc.water.tip3p", other_ffs=[], prepi=None, frcmod=None,
                      outdir="."):
    """
    Take a PDB file (with ghosts having been added) and create AMBER format prmtop
    and inpcrd files, allowing the use of other forcefields and parameter sets

    Parameters
    ----------
    pdb : str
        Name of the PDB file
    protein_ff : str
        Name of the protein force field leaprc file, e.g. 'leaprc.protein.ff14SB'
    ligand_ff : str
        Name of the ligand force field leaprc file, e.g. 'leaprc.gaff'
    water_ff : str
        Name of the water force field leaprc file, e.g. 'leaprc.water.tip3p'
    other_ffs : list
        List of any other force field leaprc files to load
    prepi : str
        Name of the ligand .prepi file
    frcmod : str
        Name of the ligand .frcmod file

    Returns
    -------
    prmtop : str
        Name of the .prmtop file written out
    inpcrd : str
        Name of the .inpcrd file written out
    """
    # Get stem of file name
    file_stem = os.path.join(outdir, os.path.splitext(os.path.basename(pdb))[0])
    pdb_amber = "{}-amber.pdb".format(file_stem)
    prmtop = "{}.prmtop".format(file_stem)
    inpcrd = "{}.inpcrd".format(file_stem)
    tleap = "{}-tleap".format(file_stem)

    # Read in box dimensions from pdb file
    with open(pdb, 'r') as f:
        for line in f.readlines():
            if line.startswith('CRYST'):
                box = [float(line.split()[1]), float(line.split()[2]), float(line.split()[3])]

    # Convert PDB to amber format
    os.system("pdb4amber -i {} -o {}".format(pdb, pdb_amber))

    # Write an input file for tleap using the relevant settings
    with open("{}.in".format(tleap), "w") as f:
        # Load force fields
        for ff in [protein_ff, ligand_ff, water_ff] + other_ffs:
            f.write("source {}\n".format(ff))
        # Load ligand parameters
        if prepi is not None:
            f.write("loadamberprep {}\n".format(prepi))
        if frcmod is not None:
            f.write("loadamberparams {}\n".format(frcmod))
        # Need an atom name map to fix cap atom names being mixed up
        f.write("addPdbAtomMap {{C CH3} {H1 HH31} {H2 HH32} {H3 HH33}}\n")
        # Load system and write to AMBER files
        f.write("mol = loadpdb {}-amber.pdb\n".format(file_stem))
        f.write("set mol box { %f %f %f }\n" % (box[0], box[1], box[2]))  # Have to use % due to {}
        f.write("saveamberparm mol {} {}\n".format(prmtop, inpcrd))
        f.write("quit\n")

    # Pass the file into tleap to create the desired output - check if there is an error
    err = os.system("tleap -s -f {0}.in > {0}.out".format(tleap))

    # The above tleap file doesn't always work, so sometimes the addPdbAtomMap has to be reversed
    if err != 0:
        with open("{}.in".format(tleap), "w") as f:
            # Load force fields
            for ff in [protein_ff, ligand_ff, water_ff] + other_ffs:
                f.write("source {}\n".format(ff))
            # Load ligand parameters
            if prepi is not None:
                f.write("loadamberprep {}\n".format(prepi))
            if frcmod is not None:
                f.write("loadamberparams {}\n".format(frcmod))
            # Need an atom name map to fix cap atom names being mixed up
            f.write("addPdbAtomMap {{CH3 C} {HH31 H1} {HH32 H2} {HH33 H3}}\n")
            # Load system and write to AMBER files
            f.write("mol = loadpdb {}-amber.pdb\n".format(file_stem))
            f.write("set mol box { %f %f %f }\n" % (box[0], box[1], box[2]))  # Have to use % due to {}
            f.write("saveamberparm mol {} {}\n".format(prmtop, inpcrd))
            f.write("quit\n")
        # Re-run tleap
        os.system("tleap -s -f {0}.in > {0}.out".format(tleap))

    # Return names of the .prmtop and .inpcrd files
    return prmtop, inpcrd


def random_rotation_matrix():
    """
    Generate a random axis and angle for rotation of the water coordinates (using the
    method used for this in the ProtoMS source code (www.protoms.org), and then return
    a rotation matrix produced from these

    Returns
    -------
    rot_matrix : numpy.ndarray
        Rotation matrix generated
    """
    # First generate a random axis about which the rotation will occur
    rand1 = rand2 = 2.0

    while (rand1**2 + rand2**2) >= 1.0:
        rand1 = np.random.rand()
        rand2 = np.random.rand()
    rand_h = 2 * np.sqrt(1.0 - (rand1**2 + rand2**2))
    axis = np.array([rand1 * rand_h, rand2 * rand_h, 1 - 2*(rand1**2 + rand2**2)])
    axis /= np.linalg.norm(axis)

    # Get a random angle
    theta = np.pi * (2*np.random.rand() - 1.0)

    # Simplify products & generate matrix
    x, y, z = axis[0], axis[1], axis[2]
    x2, y2, z2 = axis[0]*axis[0], axis[1]*axis[1], axis[2]*axis[2]
    xy, xz, yz = axis[0]*axis[1], axis[0]*axis[2], axis[1]*axis[2]
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    rot_matrix = np.array([[cos_theta + x2*(1-cos_theta),   xy*(1-cos_theta) - z*sin_theta, xz*(1-cos_theta) + y*sin_theta],
                           [xy*(1-cos_theta) + z*sin_theta, cos_theta + y2*(1-cos_theta),   yz*(1-cos_theta) - x*sin_theta],
                           [xz*(1-cos_theta) - y*sin_theta, yz*(1-cos_theta) + x*sin_theta, cos_theta + z2*(1-cos_theta)  ]])

    return rot_matrix


def shift_ghost_waters(ghost_file, topology=None, trajectory=None, t=None, output=None):
    """
    Translate all ghost waters in a trajectory out of the simulation box, to make
    visualisation clearer

    Parameters
    ----------
    ghost_file : str
        Name of the file containing the ghost water residue IDs at each frame
    topology : str
        Name of the topology/connectivity file (e.g. PDB, GRO, etc.)
    trajectory : str
        Name of the trajectory file (e.g. DCD, XTC, etc.)
    t : mdtraj.Trajectory
        Trajectory object, if already loaded
    output : str
        Name of the file to which the new trajectory is written. If None, then a
        Trajectory will be returned

    Returns
    -------
    t : mdtraj.Trajectory
        Will return a trajectory object, if no output file name is given
    """
    # Read in residue IDs for the ghost waters in each frame
    ghost_resids = []
    with open(ghost_file, 'r') as f:
        for line in f.readlines():
            ghost_resids.append([int(resid) for resid in line.split(",")])

    # Read in trajectory data
    if t is None:
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
            t.xyz[frame, index, :] += 5 * t.unitcell_lengths[frame, :]

    # Either return the trajectory or save to file
    if output is None:
        return t
    else:
        t.save(output)
        return None


def wrap_waters(topology=None, trajectory=None, t=None, output=None):
    """
    Wrap water molecules if the coordinates haven't been wrapped by the DCDReporter

    Parameters
    ----------
    topology : str
        Name of the topology/connectivity file (e.g. PDB, GRO, etc.)
    trajectory : str
        Name of the trajectory file (e.g. DCD, XTC, etc.)
    t : mdtraj.Trajectory
        Trajectory object, if already loaded
    output : str
        Name of the file to which the new trajectory is written. If None, then a
        Trajectory will be returned

    Returns
    -------
    t : mdtraj.Trajectory
        Will return a trajectory object, if no output file name is given
    """
    # Load trajectory data, if not already
    if t is None:
        t = mdtraj.load(trajectory, top=topology, discard_overlapping_frames=False)

    n_frames, n_atoms, n_dims = t.xyz.shape

    # Fix all frames
    for f in range(n_frames):
        for residue in t.topology.residues:
            # Skip if this is a protein residue
            if residue.name not in ['WAT', 'HOH']:
                continue

            # Find the maximum and minimum distances between this residue and the reference atom
            for atom in residue.atoms:
                if 'O' in atom.name:
                    pos = t.xyz[f, atom.index, :]

            # Calculate the correction vector based on the separation
            box = t.unitcell_lengths[f, :]

            new_pos = deepcopy(pos)
            for i in range(3):
                while new_pos[i] >= box[i]:
                    new_pos[i] -= box[i]
                while new_pos[i] <= 0:
                    new_pos[i] += box[i]

            correction = new_pos - pos

            # Apply the correction vector to each atom in the residue
            for atom in residue.atoms:
                t.xyz[f, atom.index, :] += correction

    # Either return or save the trajectory
    if output is None:
        return t
    else:
        t.save(output)
        return None


def align_traj(topology=None, trajectory=None, t=None, reference=None, output=None):
    """
    Align a trajectory to the protein

    Parameters
    ----------
    topology : str
        Name of the topology/connectivity file (e.g. PDB, GRO, etc.)
    trajectory : str
        Name of the trajectory file (e.g. DCD, XTC, etc.)
    t : mdtraj.Trajectory
        Trajectory object, if already loaded
    reference : str
        Name of a PDB file to align the protein to. May be better to visualise
    output : str
        Name of the file to which the new trajectory is written. If None, then a
        Trajectory will be returned

    Returns
    -------
    t : mdtraj.Trajectory
        Will return a trajectory object, if no output file name is given
    """
    # Load trajectory data, if not already
    if t is None:
        t = mdtraj.load(trajectory, top=topology, discard_overlapping_frames=False)

    # Align trajectory based on protein IDs
    protein_ids = [atom.index for atom in t.topology.atoms if atom.residue.is_protein]
    if reference is None:
        # If there is no reference then align to the first frame in the trajectory
        t.superpose(t, atom_indices=protein_ids)
    else:
        # Load a reference PDB to align the structure to
        t_ref = mdtraj.load(reference)
        t.superpose(t_ref, atom_indices=protein_ids)

    # Return or save trajectory
    if output is None:
        return t
    else:
        t.save(output)
        return None


def recentre_traj(topology=None, trajectory=None, t=None, resname='ALA', resid=1, output=None):
    """
    Recentre a trajectory based on a specific protein residue. Assumes that the
    protein has not been broken by periodic boundaries.
    Would be best to do this step before aligning a trajectory

    Parameters
    ----------
    topology : str
        Name of the topology/connectivity file (e.g. PDB, GRO, etc.)
    trajectory : str
        Name of the trajectory file (e.g. DCD, XTC, etc.)
    t : mdtraj.Trajectory
        Trajectory object, if already loaded
    resname : str
        Name of the protein residue to centre the trajectory on. Should be a
        binding site residue
    resid : int
        ID of the protein residue to centre the trajectory. Should be a binding
        site residue
    output : str
        Name of the file to which the new trajectory is written. If None, then a
        Trajectory will be returned

    Returns
    -------
    t : mdtraj.Trajectory
        Will return a trajectory object, if no output file name is given
    """
    # Load trajectory
    if t is None:
        t = mdtraj.load(trajectory, top=topology, discard_overlapping_frames=False)
    n_frames, n_atoms, n_dims = t.xyz.shape

    # Get IDs of protein atoms
    protein_ids = [atom.index for atom in t.topology.atoms if atom.residue.is_protein]

    # Find the index of the C-alpha atom of this residue
    ref_idx = None
    for residue in t.topology.residues:
        if residue.name.lower() == resname.lower() and residue.resSeq == resid:
            for atom in residue.atoms:
                if atom.name.lower() == 'ca':
                    ref_idx = atom.index
    if ref_idx is None:
        raise Exception("Could not find residue {}{}!".format(resname.capitalize(), resid))

    # Fix all frames
    for f in range(n_frames):
        for residue in t.topology.residues:
            # Skip if this is a protein residue
            if any([atom.index in protein_ids for atom in residue.atoms]):
                continue

            # Calculate the correction vector based on the separation
            box = t.unitcell_lengths[f, :]
            correction = np.zeros(3)

            cog = np.zeros(3)
            for atom in residue.atoms:
                cog += t.xyz[f, atom.index, :]
            cog /= residue.n_atoms

            vector = cog - t.xyz[f, ref_idx, :]

            for i in range(3):
                if -2 * box[i] < vector[i] < -0.5 * box[i]:
                    correction[i] += box[i]
                elif 0.5 * box[i] < vector[i] < 2 * box[i]:
                    correction[i] -= box[i]

            # Apply the correction vector to each atom in the residue
            for atom in residue.atoms:
                t.xyz[f, atom.index, :] += correction

    # Either return or save the trajectory
    if output is None:
        return t
    else:
        t.save(output)
        return None


def write_sphere_traj(radius, ref_atoms=None, topology=None, trajectory=None, t=None, sphere_centre=None,
                      output='gcmc_sphere.pdb', initial_frame=False):
    """
    Write out a multi-frame PDB file containing the centre of the GCMC sphere

    Parameters
    ----------
    radius : float
        Radius of the GCMC sphere in Angstroms
    ref_atoms : list
        List of reference atoms for the GCMC sphere, as [['name', 'resname', 'resid']]
    topology : str
        Topology of the system, such as a PDB file
    trajectory : str
        Trajectory file, such as DCD
    t : mdtraj.Trajectory
        Trajectory object, if already loaded
    sphere_centre : simtk.unit.Quantity
        Coordinates around which the GCMC sohere is based
    output : str
        Name of the output PDB file
    initial_frame : bool
        Write an extra frame for the topology at the beginning of the trajectory.
        Sometimes necessary when visualising a trajectory loaded onto a PDB
    """
    # Load trajectory
    if t is None:
        t = mdtraj.load(trajectory, top=topology, discard_overlapping_frames=False)
    n_frames, n_atoms, n_dims = t.xyz.shape

    # Get reference atom IDs
    if ref_atoms is not None:
        ref_indices = []
        for ref_atom in ref_atoms:
            found = False
            for residue in t.topology.residues:
                if residue.name == ref_atom['resname'] and str(residue.resSeq) == ref_atom['resid']:
                    for atom in residue.atoms:
                        if atom.name == ref_atom['name']:
                            ref_indices.append(atom.index)
                            found = True
            if not found:
                raise Exception("Atom {} of residue {}{} not found!".format(ref_atom['name'],
                                                                            ref_atom['resname'].capitalize(),
                                                                            ref_atom['resid']))

    # Loop over all frames and write to PDB file
    with open(output, 'w') as f:
        f.write("HEADER GCMC SPHERE\n")
        f.write("REMARK RADIUS = {} ANGSTROMS\n".format(radius))

        # Figure out the initial coordinates if requested
        if initial_frame:
            t_i = mdtraj.load(topology, discard_overlapping_frames=False)
            # Calculate centre
            if sphere_centre is None:
                centre = np.zeros(3)
                for idx in ref_indices:
                    centre += t_i.xyz[0, idx, :]
                centre *= 10 / len(ref_indices)  # Also convert from nm to A
            else:
                centre = sphere_centre.in_units_of(unit.angstroms)._value
            # Write to PDB file
            f.write("MODEL\n")
            f.write("HETATM{:>5d} {:<4s} {:<4s} {:>4d}    {:>8.3f}{:>8.3f}{:>8.3f}\n".format(1, 'CTR', 'SPH', 1,
                                                                                             centre[0], centre[1],
                                                                                             centre[2]))
            f.write("ENDMDL\n")

        # Loop over all frames
        for frame in range(n_frames):
            # Calculate sphere centre
            if sphere_centre is None:
                centre = np.zeros(3)
                for idx in ref_indices:
                    centre += t.xyz[frame, idx, :]
                centre *= 10 / len(ref_indices)  # Also convert from nm to A
            else:
                centre = sphere_centre.in_units_of(unit.angstroms)._value
            # Write to PDB file
            f.write("MODEL {}\n".format(frame+1))
            f.write("HETATM{:>5d} {:<4s} {:<4s} {:>4d}    {:>8.3f}{:>8.3f}{:>8.3f}\n".format(1, 'CTR', 'SPH', 1,
                                                                                             centre[0], centre[1],
                                                                                             centre[2]))
            f.write("ENDMDL\n")

    return None

def cluster_waters(topology, trajectory, sphere_radius, ref_atoms=None, sphere_centre=None, cutoff=2.4,
                   output='gcmc_clusts.pdb'):
    """
    Carry out a clustering analysis on GCMC water molecules with the sphere. Based on the clustering
    code in the ProtoMS software package.

    This function currently assumes that the system has been aligned and centred on the GCMC sphere (approximately).

    Parameters
    ----------
    topology : str
        Topology of the system, such as a PDB file
    trajectory : str
        Trajectory file, such as DCD
    sphere_radius : float
        Radius of the GCMC sphere in Angstroms
    ref_atoms : list
        List of reference atoms for the GCMC sphere, as [['name', 'resname', 'resid']]
    sphere_centre : simtk.unit.Quantity
        Coordinates around which the GCMC sohere is based
    cutoff : float
        Distance cutoff used in the clustering
    output : str
        Name of the output PDB file containing the clusters
    """
    # Load trajectory
    t = mdtraj.load(trajectory, top=topology, discard_overlapping_frames=False)
    n_frames, n_atoms, n_dims = t.xyz.shape

    # Get reference atom IDs
    if ref_atoms is not None:
        ref_indices = []
        for ref_atom in ref_atoms:
            found = False
            for residue in t.topology.residues:
                if residue.name == ref_atom['resname'] and str(residue.resSeq) == ref_atom['resid']:
                    for atom in residue.atoms:
                        if atom.name == ref_atom['name']:
                            ref_indices.append(atom.index)
                            found = True
            if not found:
                raise Exception("Atom {} of residue {}{} not found!".format(ref_atom['name'],
                                                                            ref_atom['resname'].capitalize(),
                                                                            ref_atom['resid']))

    wat_coords = []  # Store a list of water coordinates
    wat_frames = []  # Store a list of the frame that each water is in

    # Get list of water oxygen atom IDs
    wat_ox_ids = []
    for residue in t.topology.residues:
        if residue.name.lower() in ['wat', 'hoh']:
            for atom in residue.atoms:
                if atom.name.lower() == 'o':
                    wat_ox_ids.append(atom.index)

    # Get the coordinates of all GCMC water oxygen atoms
    for f in range(n_frames):

        # Calculate sphere centre for this frame
        if ref_atoms is not None:
            centre = np.zeros(3)
            for idx in ref_indices:
                centre += t.xyz[f, idx, :]
            centre /= len(ref_indices)
        else:
            centre = sphere_centre.in_units_of(unit.nanometer)._value

        # For all waters, check the distance to the sphere centre
        for o in wat_ox_ids:
            # Calculate PBC-corrected vector
            vector = t.xyz[f, o, :] - centre

            # Check length and add to list if within sphere
            if 10*np.linalg.norm(vector) <= sphere_radius:  # *10 to give Angstroms
                wat_coords.append(10 * t.xyz[f, o, :])  # Convert to Angstroms
                wat_frames.append(f)

    # Calculate water-water distances - if the waters are in the same frame are assigned a very large distance
    dist_list = []
    for i in range(len(wat_coords)):
        for j in range(i+1, len(wat_coords)):
            if wat_frames[i] == wat_frames[j]:
                dist = 1e8
            else:
                dist = np.linalg.norm(wat_coords[i] - wat_coords[j])
            dist_list.append(dist)

    # Cluster the waters hierarchically
    tree = hierarchy.linkage(dist_list, method='average')
    wat_clust_ids = hierarchy.fcluster(tree, t=cutoff, criterion='distance')
    n_clusts = max(wat_clust_ids)

    # Sort the clusters by occupancy
    clusts = []
    for i in range(1, n_clusts+1):
        occ = len([wat for wat in wat_clust_ids if wat == i])
        clusts.append([i, occ])
    clusts = sorted(clusts, key=lambda x: -x[1])
    clust_ids_sorted = [x[0] for x in clusts]
    clust_occs_sorted = [x[1] for x in clusts]

    # Calculate the cluster centre and representative position for each cluster
    rep_coords = []
    for i in range(n_clusts):
        clust_id = clust_ids_sorted[i]
        # Calculate the mean position of the cluster
        clust_centre = np.zeros(3)
        for j, wat in enumerate(wat_clust_ids):
            if wat == clust_id:
                clust_centre += wat_coords[j]
        clust_centre /= clust_occs_sorted[i]

        # Find the water observation which is closest to the mean position
        min_dist = 1e8
        rep_wat = None
        for j, wat in enumerate(wat_clust_ids):
            if wat == clust_id:
                dist = np.linalg.norm(wat_coords[j] - clust_centre)
                if dist < min_dist:
                    min_dist = dist
                    rep_wat = j
        rep_coords.append(wat_coords[rep_wat])

    # Write the cluster coordinates to a PDB file
    with open(output, 'w') as f:
        f.write("REMARK Clustered GCMC Water positions written by grand\n")
        for i in range(n_clusts):
            coords = rep_coords[i]
            occ1 = clust_occs_sorted[i]
            occ2 = occ1 / float(n_frames)
            f.write("ATOM  {:>5d} {:<4s} {:<4s} {:>4d}    {:>8.3f}{:>8.3f}{:>8.3f}{:>6.2f}{:>6.2f}\n".format(1, 'O',
                                                                                                             'WAT', i+1,
                                                                                                             coords[0],
                                                                                                             coords[1],
                                                                                                             coords[2],
                                                                                                             occ1, occ2))
            f.write("TER\n")
        f.write("END")

    return None
