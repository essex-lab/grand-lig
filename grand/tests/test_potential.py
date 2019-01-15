"""
test_potential.py
Marley Samways

This file contains functions written to test the functions in the grand.potential sub-module
"""

import unittest
import numpy as np
from grand import potential


class TestPotential(unittest.TestCase):
    """
    Class to store the tests for grand.utils
    """
    def test_get_lambdas(self):
        """
        Test the get_lambda_values() function, designed to retrieve steric and
        electrostatic lambda values from a single lambda value
        """
        # Test several lambda values between 0 and 1 - should interpolate linearly
        assert all(np.isclose(potential.get_lambda_values(1.00), [1.0, 1.0]))
        assert all(np.isclose(potential.get_lambda_values(0.75), [1.0, 0.5]))
        assert all(np.isclose(potential.get_lambda_values(0.50), [1.0, 0.0]))
        assert all(np.isclose(potential.get_lambda_values(0.25), [0.5, 0.0]))
        assert all(np.isclose(potential.get_lambda_values(0.00), [0.0, 0.0]))
        # Test behaviour outside of these limits - should stay within 0 and 1
        assert all(np.isclose(potential.get_lambda_values(2.00), [1.0, 1.0]))
        assert all(np.isclose(potential.get_lambda_values(1.50), [1.0, 1.0]))
        assert all(np.isclose(potential.get_lambda_values(-0.50), [0.0, 0.0]))
        assert all(np.isclose(potential.get_lambda_values(-1.00), [0.0, 0.0]))
