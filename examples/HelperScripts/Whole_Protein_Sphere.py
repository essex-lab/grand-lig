
import MDAnalysis as mda
from MDAnalysis.analysis import rms, align, distances
import argparse
import numpy as np


# Arguments to be input
parser = argparse.ArgumentParser()
parser.add_argument("-p", "--pdb", help="Input PDB File")
parser.add_argument("-t", "--traj", help="Input trajectory File", default=None)
args = parser.parse_args()


if args.pdb.split(".")[-1] == 'cif':
    from openmm.app import pdbxfile
    pdb = pdbxfile.PDBxFile(args.pdb)
else:
    pdb = args.pdb

if args.traj:
    u = mda.Universe(pdb, args.traj)
    average = align.AverageStructure(u, u, select='protein and name CA',
                                     ref_frame=0).run()
    ref = average.results.universe
    aligner = align.AlignTraj(u, ref,
                              select='protein and name CA',
                              in_memory=True).run()
else:
    u = mda.Universe(pdb)

# Get the COM of the protein
if args.traj:
    c_alphas = ref.select_atoms("protein and name CA")[20:-20]
else:
    c_alphas = u.select_atoms("protein and name CA")[20:-20]

centroid = c_alphas.centroid()

# Get distance to centroid for all CAs
dists = []
dists = [np.linalg.norm(centroid-c_alphas.positions[i]) for i in range(len(c_alphas.positions))]

# Find the closest CA to the centroid
min_dist = np.min(dists)
closest_CA_arg = np.argmin(dists)
closest_CA = c_alphas[closest_CA_arg]

# calc the RMSF to see if the centroid CA is stable
if args.traj:
    R = rms.RMSF(c_alphas).run()
    closest_rmsf = R.results.rmsf[closest_CA_arg]
    if closest_rmsf > 1:
        print(f"WARNING: RMSF fot he closest CA {closest_CA} is greater than 1. ({closest_rmsf})")


closest_positions = closest_CA.position
dists_from_closest = [np.linalg.norm(closest_positions-c_alphas.positions[i]) for i in range(len(c_alphas.positions))]
sphere_rad = np.max(dists_from_closest)

print(f"Sphere should be anchored to {closest_CA} with a radius of {sphere_rad} A")

