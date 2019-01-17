"""
test_utils.py
Marley Samways

This file contains functions written to test the functions in the grand.utils sub-module
"""

import os
import unittest
import numpy as np
from grand import utils


class TestUtils(unittest.TestCase):
    """
    Class to store the tests for grand.utils
    """
    def test_get_file(self):
        """
        Test the get_file function, designed to retrieve certain package data files
        """
        # Check that a known file is returned
        assert os.path.isfile(utils.get_file('tip3p.pdb'))
        # Check that a made up file raises an exception
        self.assertRaises(Exception, lambda: utils.get_file('imaginary.file'))

    def test_rotation_matrix(self):
        """
        Test that the random_rotation_matrix functions works as expected
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
