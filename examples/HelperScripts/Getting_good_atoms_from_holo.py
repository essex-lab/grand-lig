"""
Script to find the closest COG between two protein atoms to the COG of a ligand
"""

import MDAnalysis as mda
from MDAnalysis.analysis import rms, align, distances
import argparse
import numpy as np
import math
from openmm.unit import *
import matplotlib.pyplot as plt

# Arguments to be input
parser = argparse.ArgumentParser()
parser.add_argument("-p", "--pdb", help="Input PDB File")
parser.add_argument("-r", "--resn", help="Input ligand resname")
parser.add_argument("-i", "--resi", help="Input ligand resid")


args = parser.parse_args()

u = mda.Universe(args.pdb)


c_alphas = u.select_atoms('protein and name CA')


lig = u.select_atoms(f"resname {args.resn} and resid {args.resi}")
lig_com = lig.center_of_mass()
print(lig_com)

avg_c_alphas = u.select_atoms('protein and name CA')

dist_arr = distances.distance_array(avg_c_alphas.positions, # reference
                                    avg_c_alphas.positions, # configuration
                                    box=u.dimensions)
print(dist_arr.shape)


cas_com = np.zeros((len(avg_c_alphas), len(avg_c_alphas), 3))

for i in range(len(avg_c_alphas)):
    for j in range(len(avg_c_alphas)):
        atom_i = avg_c_alphas[i]
        atom_j = avg_c_alphas[j]
        ij_ag = atom_i + atom_j
        #print(ij_ag)
        com = ij_ag.center_of_mass()
        for k in range(3):
            cas_com[i][j][k] = com[k]


closest_dist = 1e8
closest_indicies = [0, 0]
for i in range(len(avg_c_alphas)):
    for j in range(len(avg_c_alphas)):
        p0 = cas_com[i][j]
        p1 = lig_com
        dist =  np.linalg.norm(p0 - p1)
        if dist < closest_dist:
            closest_dist = dist
            closest_indicies = [i, j]
            print(dist, closest_indicies,  c_alphas[closest_indicies[0]], c_alphas[closest_indicies[1]])

print(closest_indicies, c_alphas[closest_indicies[0]], c_alphas[closest_indicies[1]])
