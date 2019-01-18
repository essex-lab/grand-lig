"""
test_potential.py
Marley Samways

This file contains functions written to test the functions in the grand.potential sub-module
"""

import unittest
import numpy as np
from simtk.unit import *
from simtk.openmm.app import *
from simtk.openmm import *
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

        return None

    def test_calc_mu(self):
        """
        Test that the calc_mu function performs sensibly

        Notes
        -----
        Need to add more thorough testing here...
            - Not exactly sure how to do this...
            - May need to break the calc_mu() function up a bit to facilitate
              testing...
        """
        # Try to calculate dG for three platforms
        dG_cuda = None
        dG_opencl = None
        dG_cpu = None
        # Try running on CUDA
        try:
            dG_cuda = potential.calc_mu(model='tip3p', box_len=25 * angstroms, cutoff=12 * angstroms, switch_dist=10 * angstroms,
                                        nb_method=PME, temperature=300 * kelvin, equil_time=20 * femtoseconds,
                                        sample_time=10 * femtoseconds,
                                        n_lambdas=6, n_samples=5, dt=2 * femtoseconds,
                                        platform=Platform.getPlatformByName('CUDA'))
        except:
            pass
        # Then try OpenCL...
        try:
            dG_opencl = potential.calc_mu(model='tip3p', box_len=25 * angstroms, cutoff=12 * angstroms,
                                          switch_dist=10 * angstroms,
                                          nb_method=PME, temperature=300 * kelvin, equil_time=20 * femtoseconds,
                                          sample_time=10 * femtoseconds,
                                          n_lambdas=6, n_samples=5, dt=2 * femtoseconds,
                                          platform=Platform.getPlatformByName('OpenCL'))
        except:
            pass
        # Then try CPU (default)
        try:
            dG_cpu = potential.calc_mu(model='tip3p', box_len=25 * angstroms, cutoff=12 * angstroms,
                                       switch_dist=10 * angstroms,
                                       nb_method=PME, temperature=300 * kelvin, equil_time=20 * femtoseconds,
                                       sample_time=10 * femtoseconds,
                                       n_lambdas=6, n_samples=5, dt=2 * femtoseconds)
        except:
            pass
        # Check that at least the CPU version has worked...
        assert dG_cpu is not None

        # Check that those which have worked each returned a free energy
        for dG in [dG_cuda, dG_opencl, dG_cpu]:
            if dG is not None:
                # Make sure that the returned value has units
                assert isinstance(dG, Quantity)
                # Make sure that the value has units of energy
                assert dG.unit.is_compatible(kilocalorie_per_mole)

        return None
