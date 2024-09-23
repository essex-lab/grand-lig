"""
Description
-----------
This file contains functions written to test the functions in the grandlig.potential sub-module

Marley Samways
"""

import os
import unittest
import numpy as np
from openmm.unit import *
from openmm.app import *
from openmm import *
from grandlig import potential
from grandlig import utils


outdir = os.path.join(os.path.dirname(__file__), "output", "potential")
# if os.path.exists(outdir):
#     os.rmdir(outdir)
# os.makedirs(outdir)

class TestPotential(unittest.TestCase):
    """
    Class to store the tests for grandlig.potential
    """

    @classmethod
    def setUpClass(cls):
        """
        Get things ready to run these tests
        """
        # Make the output directory if needed
        if not os.path.isdir(
            os.path.join(os.path.dirname(__file__), "output")
        ):
            os.mkdir(os.path.join(os.path.dirname(__file__), "output"))
        # Create a new directory if needed
        if not os.path.isdir(outdir):
            os.mkdir(outdir)
        # If not, then clear any files already in the output directory so that they don't influence tests
        else:
            for file in os.listdir(outdir):
                os.remove(os.path.join(outdir, file))

        return None

    def test_calc_mu(self):
        """
        Test that the calc_mu function performs sensibly
        """
        # Need to set up a system first

        # Load a water-ligand box
        pdb = PDBFile(
            utils.get_data_file(os.path.join("tests", "WaterBenzene.pdb"))
        )

        # Set up system
        ff = ForceField(
            "tip3p.xml",
            utils.get_data_file(os.path.join("tests", "Benzene.xml")),
        )
        system = ff.createSystem(
            pdb.topology,
            nonbondedMethod=PME,
            nonbondedCutoff=12.0 * angstroms,
            constraints=HBonds,
            switchDistance=10 * angstroms,
        )

        # Run free energy calculation using grandlig
        log_file = os.path.join(outdir, "free_energy_test.log")
        free_energy = potential.calc_mu_ex(
            system=system,
            topology=pdb.topology,
            positions=pdb.positions,
            resname="L01",
            resid=2135,
            box_vectors=pdb.topology.getPeriodicBoxVectors(),
            temperature=298 * kelvin,
            n_lambdas=11,
            n_samples=5,
            n_equil=1,
            log_file=log_file,
            pressure=1 * bar,
        )

        # Check that a free energy has been returned
        # Make sure that the returned value has units
        assert isinstance(free_energy, Quantity)
        # Make sure that the value has units of energy
        assert free_energy.unit.is_compatible(kilocalorie_per_mole)

        return None

    def test_calc_std_volume(self):
        """
        Test that the calc_std_volume function performs sensibly
        """
        # Need to set up a system first

        # Load a pre-equilibrated water box
        pdb = PDBFile(
            utils.get_data_file(os.path.join("tests", "water_box-eq.pdb"))
        )

        # Set up system
        ff = ForceField("tip3p.xml")
        system = ff.createSystem(
            pdb.topology,
            nonbondedMethod=PME,
            nonbondedCutoff=12.0 * angstroms,
            constraints=HBonds,
            switchDistance=10 * angstroms,
        )

        # Run std volume calculation using grandlig
        std_volume_dict, conc_dict = potential.calc_avg_volume(
            system=system,
            topology=pdb.topology,
            positions=pdb.positions,
            box_vectors=pdb.topology.getPeriodicBoxVectors(),
            temperature=298 * kelvin,
            n_samples=10,
            n_equil=1,
        )

        # Check that a volume has been returned
        # Make sure that the returned value has units
        assert isinstance(std_volume_dict, dict)
        assert isinstance(std_volume_dict["HOH"], Quantity)

        assert isinstance(conc_dict, dict)
        assert isinstance(conc_dict["HOH"], Quantity)

        # Make sure that the value has units of volume
        assert std_volume_dict["HOH"].unit.is_compatible(angstroms**3)
        assert conc_dict["HOH"].unit.is_compatible(molar)
        return None


if __name__ == "__main__":
    unittest.main()
