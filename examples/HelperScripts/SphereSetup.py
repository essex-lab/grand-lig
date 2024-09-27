import mdtraj
import numpy as np

def write_sphere_traj(radius, ref_atoms=None, pdb=None, t=None, sphere_centre=None,
                      output='gcmc_sphere.pdb', initial_frame=False):
    """
    Write out a multi-frame PDB file containing the centre of the GCMC sphere

    Parameters
    ----------
    radius : float
        Radius of the GCMC sphere in Angstroms
    ref_atoms : list
        List of reference atoms for the GCMC sphere, as [['name', 'resname', 'resid']]
    pdb : str
        Topology of the system, such as a PDB file
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
        t = mdtraj.load(pdb, discard_overlapping_frames=False)
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

        # Calculate sphere centre
        centre = np.zeros(3)
        for idx in ref_indices:
            centre += t.xyz[0, idx, :]
        centre *= 10 / len(ref_indices)  # Also convert from nm to A
        # Write to PDB file
        f.write("MODEL 1\n")
        f.write("HETATM{:>5d} {:<4s} {:<4s} {:>4d}    {:>8.3f}{:>8.3f}{:>8.3f}\n".format(1, 'CTR', 'SPH', 1,
                                                                                         centre[0], centre[1],
                                                                                         centre[2]))
        f.write("ENDMDL\n")

    return None


ref_atoms = [{'name': 'CA', 'resname': 'LYS', 'resid': '55'},
             {'name': 'CA', 'resname': 'LEU', 'resid': '116'}]

write_sphere_traj(radius=4.0, ref_atoms=ref_atoms, pdb='1znk.pdb',
                              output='gcmc_sphere.pdb', initial_frame=True)

