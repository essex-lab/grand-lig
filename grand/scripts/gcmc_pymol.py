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
import os
import argparse
import pymol
from pymol import cmd


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
        #cmd.hide('(h. and (e. c extend 1))', 'resn {}'.format(resname))

# Load GCMC sphere, if given
if args.sphere is not None:
    # Read in the sphere radius
    with open(args.sphere, 'r') as f:
        for line in f.readlines():
            # Read in radius
            if line.startswith('REMARK RADIUS'):
                radius = float(line.split()[3])
                break
    # Load the sphere
    cmd.load(args.sphere, 'sphere')
    # Format the size of the sphere
    cmd.hide('everything', 'sphere')
    cmd.show('spheres', 'sphere')
    cmd.alter('sphere', 'vdw={}'.format(radius))
    cmd.rebuild()
    cmd.set('sphere_color', 'grey90', 'sphere')
    cmd.set('sphere_transparency', '0.5', 'sphere')
