# -*- coding: utf-8 -*-

"""
write_pdbs.py
Marley Samways

Description
-----------
Script to write PDBs from a trajectory, either every N frames, or just the last frame
"""

import argparse
import os
import MDAnalysis as mda
from openmm.app import PDBFile
from grandlig.utils import remove_ghosts


# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-c',  '--connect', default='system.pdb',
                    help='Connectivity file for the system')
parser.add_argument('-t',  '--trajectory', default='simulation.dcd',
                    help='Trajectory file for the simulation')
parser.add_argument('-g',  '--ghosts', default=None,
                    help='File describing ghost waters for the simulation')
parser.add_argument('-s', '--skip', type=int, default=1,
                    help='Write out PDBs every this many frame. If -1, will only write the last frame')
parser.add_argument('-o',  '--output', default='frame_',
                    help='File stem for the output PDB file(s)')
args = parser.parse_args()

# Load universe and define the system
u = mda.Universe(args.connect, args.trajectory)
print('x')
print(dir(u.atoms))
u.atoms[u.atoms.types == 'Cl'].masses = 35.5
system = u.select_atoms('all')

# Read in ghost waters from file, if needed
if args.ghosts is not None:
    ghost_wats = []
    with open(args.ghosts, 'r') as f:
        for line in f.readlines():
            ghost_wats.append([int(x) for x in line.split(',')])
    # Also want to count the maximum range between numbers of water molecules, but checking the length of the positions
    min_pos_len = 1e8
    max_pos_len = None

if args.skip == -1:
    # If skip is set equal to -1, then only write out the last frame of the trajectory
    filename = '{}{:07d}.pdb'.format(args.output, len(u.trajectory))
    u.trajectory[-1]
    system.write(filename)
    # Remove the ghost waters, if necessary
    if args.ghosts is not None:
        # Load in PDB file
        pdb = PDBFile(filename)
        # Remove file so that it can be safely overwritten
        os.remove(filename)
        # Remove the ghosts from the file
        pdb.topology, pdb.positions = remove_ghosts(topology=pdb.topology, positions=pdb.positions,
                                                    ghosts=ghost_wats[-1], pdb=filename)
else:
    # Otherwise, write out PDBs every {skip} steps
    for i, ts in enumerate(u.trajectory):
        if i % args.skip != 0:
            continue
        # Write file
        filename = '{}{:07d}.pdb'.format(args.output, i+1)
        system.write(filename)

        # Need to remove the MODEL and ENDMDL lines
        with open(filename, 'r') as f:
            lines = f.readlines()
        with open(filename, 'w') as f:
            for line in lines:
                if 'MODEL' not in line and 'ENDMDL' not in line:
                    f.write(line)

        # Remove ghosts, if needed
        if args.ghosts is not None:
            # Load in PDB file
            pdb = PDBFile(filename)
            # Remove file so that it can be safely overwritten
            os.remove(filename)
            # Remove the ghosts from the file
            pdb.topology, pdb.positions = remove_ghosts(topology=pdb.topology, positions=pdb.positions,
                                                        ghosts=ghost_wats[i], pdb=filename)
            # Check length of the positions
            if len(pdb.positions) < min_pos_len:
                min_pos_len = len(pdb.positions)
            if len(pdb.positions) > max_pos_len:
                max_pos_len = len(pdb.positions)

# Try to count how many water molecules
for i in [3, 4]:
    # Calculate difference in number of atoms
    diff = max_pos_len - min_pos_len
    # Calculate number of waters this corresponds to
    n_wats = diff/float(i)
    # Print this, if it divides evenly
    if n_wats % 1 == 0:
        print('For a {}-site water model, the maximum difference is {} water molecules'.format(i, int(n_wats)))
