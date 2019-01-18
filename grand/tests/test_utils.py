"""
test_utils.py
Marley Samways

This file contains functions written to test the functions in the grand.utils sub-module
"""

import os
import unittest
import numpy as np
from simtk.openmm import *
from simtk.openmm.app import *
from grand import utils


outdir = os.path.join(os.path.dirname(__file__), 'output', 'utils')


class TestUtils(unittest.TestCase):
    """
    Class to store the tests for grand.utils
    """
    def setUp(self):
        """
        Get things ready to run these tests
        """
        # Make the output directory if needed
        if not os.path.isdir(os.path.join(os.path.dirname(__file__), 'output')):
            os.mkdir(os.path.join(os.path.dirname(__file__), 'output'))
        # Create a new directory if needed
        if not os.path.isdir(outdir):
            os.mkdir(outdir)

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
