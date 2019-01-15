"""
test_utils.py
Marley Samways

This file contains functions written to test the functions in the grand.utils sub-module
"""

import os
import unittest
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
