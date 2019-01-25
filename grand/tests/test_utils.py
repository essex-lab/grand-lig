"""
test_utils.py
Marley Samways

This file contains functions written to test the functions in the grand.utils sub-module
"""

import unittest
import numpy as np
import mdtraj
from simtk.openmm import *
from simtk.openmm.app import *
from grand import utils


outdir = os.path.join(os.path.dirname(__file__), 'output', 'utils')


class TestUtils(unittest.TestCase):
    """
    Class to store the tests for grand.utils
    """
    @classmethod
    def setUpClass(cls):
        """
        Get things ready to run these tests
        """
        # Make the output directory if needed
        if not os.path.isdir(os.path.join(os.path.dirname(__file__), 'output')):
            os.mkdir(os.path.join(os.path.dirname(__file__), 'output'))
        # Create a new directory if needed
        if not os.path.isdir(outdir):
            os.mkdir(outdir)
        # If not, then clear any files already in the output directory so that they don't influence tests
        else:
            for file in os.listdir(outdir):
                os.remove(os.path.join(outdir, file))

        return None

    def test_get_data_file(self):
        """
        Test the get_data_file function, designed to retrieve certain package data files
        """
        # Check that a known file is returned
        assert os.path.isfile(utils.get_data_file('tip3p.pdb'))
        # Check that a made up file raises an exception
        self.assertRaises(Exception, lambda: utils.get_data_file('imaginary.file'))

        return None

    def test_add_ghosts(self):
        """
        Test that the add_ghosts() function works fine
        """
        # Need to load some test data
        bpti_orig = PDBFile(utils.get_data_file(os.path.join('tests', 'bpti.pdb')))
        # Count the number of water molecules in this file
        n_wats_orig = 0
        for residue in bpti_orig.topology.residues():
            if residue.name in ["HOH", "WAT"]:
                n_wats_orig += 1
        # Run add_ghosts_function
        n_add = 5  # Add this number of waters
        new_file = os.path.join(outdir, 'bpti-ghosts.pdb')
        # Make sure this file doesn't already exist
        if os.path.isfile(new_file):
            os.remove(new_file)

        # Run function
        topology, positions, ghosts = utils.add_ghosts(topology=bpti_orig.topology, positions=bpti_orig.positions,
                                                       n=n_add, pdb=new_file)
        # Make sure that the number of ghost molecule IDs is correct
        assert len(ghosts) == n_add
        # Count the number of water molecules in the topoology and check it's right
        n_wats_top = 0
        for residue in topology.residues():
            if residue.name in ["HOH", "WAT"]:
                n_wats_top += 1
        assert n_wats_top == n_wats_orig + n_add

        # Make sure that the new file was created
        assert os.path.isfile(new_file)

        return None

    def test_write_amber_input(self):
        """
        Run some tests for the write_amber_input() function
        """
        # Make sure that AMBER files get written out correctly for BPTI (no ligand)
        utils.write_amber_input(pdb=utils.get_data_file(os.path.join('tests', 'bpti-ghosts.pdb')),
                                outdir=outdir)
        assert os.path.isfile(os.path.join(outdir, 'bpti-ghosts.prmtop'))
        assert os.path.isfile(os.path.join(outdir, 'bpti-ghosts.inpcrd'))
        # Make sure that OpenMM can read these back in (but don't do anything with them)
        prmtop = AmberPrmtopFile(os.path.join(outdir, 'bpti-ghosts.prmtop'))
        inpcrd = AmberInpcrdFile(os.path.join(outdir, 'bpti-ghosts.inpcrd'))

        # Now do the same for scytalone (example with a ligand)
        utils.write_amber_input(pdb=utils.get_data_file(os.path.join('tests', 'scytalone.pdb')),
                                prepi=utils.get_data_file(os.path.join('tests', 'mq1.prepi')),
                                frcmod=utils.get_data_file(os.path.join('tests', 'mq1.frcmod')),
                                outdir=outdir)
        assert os.path.isfile(os.path.join(outdir, 'scytalone.prmtop'))
        assert os.path.isfile(os.path.join(outdir, 'scytalone.inpcrd'))
        prmtop = AmberPrmtopFile(os.path.join(outdir, 'scytalone.prmtop'))
        inpcrd = AmberInpcrdFile(os.path.join(outdir, 'scytalone.inpcrd'))

        # Clean up leap.log file if needed
        if os.path.isfile('leap.log'):
            os.remove('leap.log')

        return None

    def test_rotation_matrix(self):
        """
        Test that the random_rotation_matrix() function works as expected
        """
        R = utils.random_rotation_matrix()
        # Matrix must be 3x3
        assert R.shape == (3, 3)
        # Make sure that det(R) is +/- 1
        assert np.isclose(np.linalg.det(R), 1.0) or np.isclose(np.linalg.det(R), -1.0)
        # Check that the inverse is equal to the transpose of R
        assert np.all(np.isclose(np.linalg.inv(R), R.T))
        # Make sure that a different matrix is returned each time
        assert not np.all(np.isclose(R, utils.random_rotation_matrix()))

        return None

    def test_shift_ghost_waters(self):
        """
        Test that the shift_ghost_waters() function works
        """
        # First check that the function can return Trajectory if required
        t = utils.shift_ghost_waters(ghost_file=utils.get_data_file(os.path.join('tests', 'bpti-ghost-wats.txt')),
                                     topology=utils.get_data_file(os.path.join('tests', 'bpti-ghosts.pdb')),
                                     trajectory=utils.get_data_file(os.path.join('tests', 'bpti-raw.dcd')))
        assert isinstance(t, mdtraj.Trajectory)
        # Then make sure that the code can also write a DCD file
        utils.shift_ghost_waters(ghost_file=utils.get_data_file(os.path.join('tests', 'bpti-ghost-wats.txt')),
                                 topology=utils.get_data_file(os.path.join('tests', 'bpti-ghosts.pdb')),
                                 trajectory=utils.get_data_file(os.path.join('tests', 'bpti-raw.dcd')),
                                 output=os.path.join(outdir, 'bpti-shifted.dcd'))
        assert os.path.isfile(os.path.join(outdir, 'bpti-shifted.dcd'))

        # Need to check that the ghost residues are out of the simulation box
        ghost_waters = []
        with open(utils.get_data_file(os.path.join('tests', 'bpti-ghost-wats.txt')), 'r') as f:
            for line in f.readlines():
                ghost_waters.append([int(resid) for resid in line.split(',')])

        # Now want to check that there are no ghost water molecules inside the box
        n_frames, _, _ = t.xyz.shape
        failed_checks = []  # List of all checks done - True if the water is too close to the box
        for f in range(n_frames):
            box = t.unitcell_lengths[f, :]

            for resid, residue in enumerate(t.topology.residues):
                # Only consider ghost waters
                if resid not in ghost_waters[f]:
                    continue

                for atom in residue.atoms:
                    if 'O' in atom.name:
                        pos = t.xyz[f, atom.index, :]

                # Check if this water is too close to the box (within 2 box lengths) in any dimension
                for i in range(3):
                    failed_checks.append(pos[i] < 2 * box[i])
        # Make sure that none of the waters are ever too close
        assert not any(failed_checks)

        return None

    def test_wrap_waters(self):
        """
        Test that the wrap_waters() function works as desired
        """
        # First check that the function can return Trajectory if required
        t = utils.wrap_waters(topology=utils.get_data_file(os.path.join('tests', 'bpti-ghosts.pdb')),
                              trajectory=utils.get_data_file(os.path.join('tests', 'bpti-raw.dcd')))
        assert isinstance(t, mdtraj.Trajectory)
        # Then make sure that the code can also write a DCD file
        utils.wrap_waters(topology=utils.get_data_file(os.path.join('tests', 'bpti-ghosts.pdb')),
                          trajectory=utils.get_data_file(os.path.join('tests', 'bpti-raw.dcd')),
                          output=os.path.join(outdir, 'bpti-wrapped.dcd'))
        assert os.path.isfile(os.path.join(outdir, 'bpti-wrapped.dcd'))

        # Now want to check that there are no water molecules outside the box
        n_frames, _, _ = t.xyz.shape
        failed_checks = []  # List of all checks done - True if the water is outside the box
        for f in range(n_frames):
            box = t.unitcell_lengths[f, :]

            for residue in t.topology.residues:
                if residue.name not in ['WAT', 'HOH']:
                    continue

                for atom in residue.atoms:
                    if 'O' in atom.name:
                        pos = t.xyz[f, atom.index, :]

                # Check if this water is outside the box in any dimension
                for i in range(3):
                    failed_checks.append(pos[i] < 0 or pos[i] > box[i])
        # Make sure that none of the waters are ever outside
        assert not any(failed_checks)

        return None

    def test_align_traj(self):
        """
        Test that the align_traj() function works
        Hard to test this properly, so will just make sure that the function runs without errors
        """
        # First check that the function can return Trajectory if required (with a reference)
        t1 = utils.align_traj(topology=utils.get_data_file(os.path.join('tests', 'bpti-ghosts.pdb')),
                              trajectory=utils.get_data_file(os.path.join('tests', 'bpti-raw.dcd')),
                              reference=utils.get_data_file(os.path.join('tests', 'bpti-ghosts.pdb')))
        assert isinstance(t1, mdtraj.Trajectory)
        # Now check without a reference
        t2 = utils.align_traj(topology=utils.get_data_file(os.path.join('tests', 'bpti-ghosts.pdb')),
                              trajectory=utils.get_data_file(os.path.join('tests', 'bpti-raw.dcd')))
        assert isinstance(t2, mdtraj.Trajectory)
        # Then make sure that the code can also write a DCD file
        utils.align_traj(topology=utils.get_data_file(os.path.join('tests', 'bpti-ghosts.pdb')),
                         trajectory=utils.get_data_file(os.path.join('tests', 'bpti-raw.dcd')),
                         output=os.path.join(outdir, 'bpti-aligned.dcd'))
        assert os.path.isfile(os.path.join(outdir, 'bpti-aligned.dcd'))

        return None

    def test_recentre_traj(self):
        """
        Test that the recentre_traj() function works
        """
        pdb = utils.get_data_file(os.path.join('tests', 'bpti-ghosts.pdb'))
        dcd = utils.get_data_file(os.path.join('tests', 'bpti-raw.dcd'))
        out = os.path.join(outdir, 'bpti-recentred.dcd')
        # Check that there is an Exception if the reference residue doesn't exist (His1 doesn't for BPTI)
        self.assertRaises(Exception, lambda: utils.recentre_traj(topology=pdb, trajectory=dcd, resname='HIS',
                                                                        resid=1))
        # Make sure the function isn't case-sensitive to the residue name
        utils.recentre_traj(topology=pdb, trajectory=dcd, resname='Arg', resid=1, output=out)
        os.remove(out)

        # Make sure that a trajectory object can be returned
        t = utils.recentre_traj(topology=pdb, trajectory=dcd, resname='TYR', resid=10)
        assert isinstance(t, mdtraj.Trajectory)
        # Make sure that a file can be written out
        utils.recentre_traj(topology=pdb, trajectory=dcd, resname='TYR', resid=10, output=out)
        assert os.path.isfile(out)

        # Make sure this residue is at the centre of the trajectory
        n_frames, _, _ = t.xyz.shape
        for residue in t.topology.residues:
            if residue.name == 'TYR' and residue.resSeq == 10:
                for atom in residue.atoms:
                    if atom.name.lower() == 'ca':
                        ref = atom.index

        # Check all residues and make sure that the COG is within 0.5L of the reference atom in each frame
        failed_checks = []  # List of failed checks - True is that the COG is too far from the centre
        for f in range(n_frames):
            box = t.unitcell_lengths[f, :]

            for residue in t.topology.residues:
                # Ignore protein
                if residue.is_protein:
                    continue

                # Calculate centre of geometry
                cog = np.zeros(3)
                for atom in residue.atoms:
                    cog += t.xyz[f, atom.index, :]
                cog /= residue.n_atoms

                # Check distance from the reference
                vector = cog - t.xyz[f, ref, :]
                for i in range(3):
                    failed_checks.append(vector[i] > 0.5*box[i] or vector[i] < -0.5*box[i])
        assert not any(failed_checks)

        return None

    def test_write_sphere_traj(self):
        """
        Test that the write_sphere_traj() function works
        May need to make this function more thorough...
        """
        pdb = utils.get_data_file(os.path.join('tests', 'bpti-ghosts.pdb'))
        dcd = utils.get_data_file(os.path.join('tests', 'bpti-raw.dcd'))
        out = os.path.join(outdir, 'bpti-recentred.dcd')
        radius = 4.0
        # Make sure that the function doesn't work if an incorrect reference is given (should be Tyr10 and Asn43)
        self.assertRaises(Exception, lambda: utils.write_sphere_traj(ref_atoms=[['CA', 'TYR', '9'],
                                                                                ['CA', 'ASN', '43']],
                                                                     radius=radius, topology=pdb, trajectory=dcd))
        # Write a sphere PDB
        utils.write_sphere_traj(ref_atoms=[['CA', 'TYR', '10'], ['CA', 'ASN', '43']], radius=radius, topology=pdb,
                                trajectory=dcd, output=out, initial_frame=True)
        assert os.path.isfile(out)

        # Make sure the sphere radius is
        with open(out, 'r') as f:
            for line in f.readlines():
                if line.startswith('REMARK RADIUS'):
                    r = float(line.split()[3])
                    break
        assert np.isclose(radius, r)

        return None

