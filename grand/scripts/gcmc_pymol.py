# -*- coding: utf-8 -*-

"""
gcmc_pymol.py
Marley Samways

Description
-----------
Script to load a GCMC simulation (carried out in OpenMM) into PyMOL
This script can be used by typing:
    pymol gcmc_pymol.py -- {arguments}
"""

import __main__
import argparse
import pymol
from pymol import cmd

import numpy as np
import mdtraj


__main__.pymol_argv = ['pymol']
pymol.finish_launching()

parser = argparse.ArgumentParser()
parser.add_argument('-top', '--topology', default='system.pdb',
                    help='System topology file.')
parser.add_argument('-trj', '--trajectory', default='simulation.dcd',
                    help='Simulation trajectory file.')
parser.add_argument('-l', '--ligands', default=None, nargs='+',
                    help='Ligand residue names')
parser.add_argument('-s', '--sphere', default=None,
                    help='GCMC sphere PDB file.')
parser.add_argument('-r', '--residues', default=None, nargs='+', type=int,
                    help='Specific residues to show.')
args = parser.parse_args()

# Basic formatting
cmd.set('valence', 0)
cmd.set('stick_h_scale', 1)

# Load trajectory
cmd.load(args.topology, 'system')
cmd.load_traj(args.trajectory, 'system')
cmd.show('cartoon', 'system')

# Format ligands, if any
if args.ligands is not None:
    for resname in args.ligands:
        cmd.show('sticks', 'resn {}'.format(resname))

# Show any residues the user might be interested in
if args.residues is not None:
    for resid in args.residues:
        cmd.show('sticks', 'polymer.protein and resi {}'.format(resid))

# Hide nonpolar hydrogen atoms - syntax can vary for some reason (may need brackets)
cmd.hide('h. and (e. c extend 1)')

# Load GCMC sphere, if given
if args.sphere is not None:
    # Read in the sphere radius
    sphere_coords = []
    with open(args.sphere, 'r') as f:
        for line in f.readlines():
            # Read in radius
            if line.startswith('REMARK RADIUS'):
                radius = float(line.split()[3])
            # Read in sphere coordinates for each frame
            if line.startswith('HETATM'):
                sphere_coords.append(np.array([float(line[30:38]),
                                               float(line[38:46]),
                                               float(line[46:54])]))

    # Load the sphere
    cmd.load(args.sphere, 'sphere')

    # Format the size of the sphere - may need to adjust settings later
    cmd.hide('everything', 'sphere')
    cmd.show('spheres', 'sphere')
    cmd.alter('sphere', 'vdw={}'.format(radius))
    cmd.rebuild()
    cmd.set('sphere_color', 'grey90', 'sphere')
    cmd.set('sphere_transparency', '0.5', 'sphere')

    # Format the trajectory to show the waters within the GCMC sphere as sticks
    cmd.hide('everything', 'resn HOH')  # Hide waters first...
    n_frames = cmd.count_states()
    cmd.mset('1 -{}'.format(n_frames))

    # Need to read in water positions for each frame to check which are within the GCMC region
    t_pdb = mdtraj.load(args.topology)
    t_traj = mdtraj.load(args.trajectory, top=args.topology)

    # Analyse each frame
    for f in range(n_frames):
        sphere_wats = []
        if f == 0:
            t = t_pdb
        else:
            t = t_traj
        for residue in t.topology.residues:
            if residue.name != 'HOH':
                continue
            for atom in residue.atoms:
                if atom.name == 'O':
                    if f == 0:
                        pos = t.xyz[f, atom.index, :]
                    else:
                        pos = t.xyz[f-1, atom.index, :]
            if np.linalg.norm(pos-sphere_coords[f]) < radius:
                sphere_wats.append(residue.resSeq)
        # Set visualisation for this first frame
        command = "hide sticks, resn HOH"
        if len(sphere_wats) > 0:
            command += "; sele gcmcwats, resn HOH and resi {} and not chain A".format(sphere_wats[0])
            if len(sphere_wats) > 1:
                for i in range(1, len(sphere_wats)):
                    command += "; sele gcmcwats, gcmcwats or (resn HOH and resi {} and not chain A)".format(sphere_wats[i])
        command += "; show sticks, gcmcwats"
        cmd.mdo(f+1, command)

    '''
    for f in range(1, n_frames+1):
        # Need to write a command to update the movie to show GCMC waters as sticks
        movie_command = ("hide sticks, resn HOH;"
                         "sele gcmcwats, resn SPH around {} and resn HOH, state={};"
                         "show sticks, gcmcwats").format(radius, f)
        cmd.mdo(f, movie_command)
    '''

