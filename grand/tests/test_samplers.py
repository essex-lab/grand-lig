"""
test_samplers.py
Marley Samways

This file contains functions written to test the functions in the grand.samplers sub-module

Notes
-----
- Need to make these tests more extensive at some stage...
- Need to add tests for the non-equilibrium sampler (when it's written)
"""

import os
import unittest
import numpy as np
from copy import deepcopy
from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
from grand import samplers
from grand import utils


outdir = os.path.join(os.path.dirname(__file__), 'output', 'samplers')


def setup_GrandCanonicalMonteCarloSampler():
    """
    Set up variables for the GrandCanonicalMonteCarloSampler
    """
    # Make variables global so that they can be used
    global sampler1
    global simulation1

    pdb1 = PDBFile(utils.get_data_file(os.path.join('tests', 'bpti-ghosts.pdb')))
    ff = ForceField('amber10.xml', 'tip3p.xml')
    system1 = ff.createSystem(pdb1.topology, nonbondedMethod=PME, nonbondedCutoff=12 * angstroms,
                              constraints=HBonds)

    sampler1 = samplers.GrandCanonicalMonteCarloSampler(system=system1, topology=pdb1.topology,
                                                        temperature=300 * kelvin,
                                                        ghostFile=os.path.join(outdir, 'bpti-ghost-wats.txt'),
                                                        referenceAtoms=[['CA', 'TYR', '10'],
                                                                        ['CA', 'ASN', '43']],
                                                        sphereRadius=4 * angstrom)

    # Define a simulation
    integrator1 = LangevinIntegrator(300 * kelvin, 1.0 / picosecond, 0.002 * picoseconds)

    try:
        platform1 = Platform.getPlatformByName('OpenCL')
    except:
        platform1 = Platform.getPlatformByName('CPU')

    simulation1 = Simulation(pdb1.topology, system1, integrator1, platform1)
    simulation1.context.setPositions(pdb1.positions)
    simulation1.context.setVelocitiesToTemperature(300 * kelvin)
    simulation1.context.setPeriodicBoxVectors(*pdb1.topology.getPeriodicBoxVectors())
    sampler1.prepareGCMCSphere(simulation1.context, [3054, 3055, 3056, 3057, 3058])

    return None


class TestGrandCanonicalMonteCarloSampler(unittest.TestCase):
    """
    Class to store the tests for the GrandCanonicalMonteCarloSampler class
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

        # Need to create the sampler
        setup_GrandCanonicalMonteCarloSampler()

        return None

    def test_prepareGCMCSphere(self):
        """
        Make sure the GrandCanonicalMonteCarloSampler.prepareGCMCSphere() method works correctly
        """
        # Need ghost waters to simulate, so the function should complain if there are none
        self.assertRaises(Exception, lambda: sampler1.prepareGCMCSphere(simulation1.context, []))

        # Make sure the variables are all updated
        assert isinstance(sampler1.context, Context)
        assert isinstance(sampler1.positions, Quantity)
        assert isinstance(sampler1.sphere_centre, Quantity)

        return None

    def test_deleteWatersInGCMCSphere(self):
        """
        Make sure the GrandCanonicalMonteCarloSampler.deleteWatersInGCMCSphere() method works correctly
        """
        # Now delete the waters in the sphere
        sampler1.deleteWatersInGCMCSphere(simulation1.context)
        new_ghosts = [sampler1.gcmc_resids[id] for id in np.where(sampler1.gcmc_status == 0)[0]]
        # Check that the list of ghosts is correct
        assert new_ghosts == [70, 71, 3054, 3055, 3056, 3057, 3058]
        # Check that the variables match there being no waters in the GCMC region
        assert sampler1.N == 0
        assert all(sampler1.gcmc_status == 0)

        return None

    def test_updateGCMCSphere(self):
        """
        Make sure the GrandCanonicalMonteCarloSampler.updateGCMCSphere() method works correctly
        """
        # Get initial gcmc_resids and status
        gcmc_resids = deepcopy(sampler1.gcmc_resids)
        gcmc_status = deepcopy(sampler1.gcmc_status)
        sphere_centre = deepcopy(sampler1.sphere_centre)
        N = sampler1.N

        # Update the GCMC sphere (shouldn't change as the system won't have moved)
        state = simulation1.context.getState(getPositions=True, getVelocities=True)
        sampler1.updateGCMCSphere(state)

        # Make sure that these values are all still the same
        assert all(np.isclose(gcmc_resids, sampler1.gcmc_resids))
        assert all(np.isclose(gcmc_status, sampler1.gcmc_status))
        assert all(np.isclose(sphere_centre._value, sampler1.sphere_centre._value))
        assert N == sampler1.N

        return None

    def test_move(self):
        """
        Make sure the GrandCanonicalMonteCarloSampler.move() method works correctly
        """
        # Shouldn't be able to run a move with this sampler
        self.assertRaises(NotImplementedError, lambda: sampler1.move(simulation1.context))

        return None

    def test_report(self):
        """
        Make sure the GrandCanonicalMonteCarloSampler.report() method works correctly
        """
        # Get the list of ghost resids
        ghosts = [sampler1.gcmc_resids[id] for id in np.where(sampler1.gcmc_status == 0)[0]]

        # Report
        sampler1.report(simulation1)

        # Check the output to the ghost file
        assert os.path.isfile(os.path.join(outdir, 'bpti-ghost-wats.txt'))
        # Read which ghosts were written
        with open(os.path.join(outdir, 'bpti-ghost-wats.txt'), 'r') as f:
            n_lines = 0
            lines = f.readlines()
            for line in lines:
                if len(line.split()) > 0:
                    n_lines += 1
        assert n_lines == 1
        ghosts_read = [int(resid) for resid in lines[0].split(',')]
        assert all(np.isclose(ghosts, ghosts_read))

        return None


def setup_StandardGCMCSampler():
    """
    Set up variables for the StandardGCMCSampler
    """
    # Need variables to be global so that they can be accessed outside the function
    global sampler2
    global simulation2

    pdb2 = PDBFile(utils.get_data_file(os.path.join('tests', 'bpti-ghosts.pdb')))
    ff = ForceField('amber10.xml', 'tip3p.xml')
    system2 = ff.createSystem(pdb2.topology, nonbondedMethod=PME, nonbondedCutoff=12 * angstroms,
                              constraints=HBonds)

    sampler2 = samplers.StandardGCMCSampler(system=system2, topology=pdb2.topology, temperature=300 * kelvin,
                                            ghostFile=os.path.join(outdir, 'bpti-ghost-wats.txt'),
                                            referenceAtoms=[['CA', 'TYR', '10'],
                                                            ['CA', 'ASN', '43']],
                                            sphereRadius=4 * angstrom)

    # Define a simulation
    integrator2 = LangevinIntegrator(300 * kelvin, 1.0 / picosecond, 0.002 * picoseconds)

    try:
        platform2 = Platform.getPlatformByName('OpenCL')
    except:
        platform2 = Platform.getPlatformByName('CPU')

    simulation2 = Simulation(pdb2.topology, system2, integrator2, platform2)
    simulation2.context.setPositions(pdb2.positions)
    simulation2.context.setVelocitiesToTemperature(300 * kelvin)
    simulation2.context.setPeriodicBoxVectors(*pdb2.topology.getPeriodicBoxVectors())
    sampler2.prepareGCMCSphere(simulation2.context, [3054, 3055, 3056, 3057, 3058])

    return None


class TestStandardGCMCSampler(unittest.TestCase):
    """
    Class to store the tests for the StandardGCMCSampler class
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

        # Create sampler
        setup_StandardGCMCSampler()

        return None

    def test_move(self):
        """
        Make sure the GrandCanonicalMonteCarloSampler.move() method works correctly
        """
        # Attempt a single move
        sampler2.move(simulation2.context)
        # Make sure that one move appears to have been carried out
        assert sampler2.n_moves == 1
        assert len(sampler2.Ns) == 1
        assert sampler2.n_accepted <= sampler2.n_moves

        # Make sure that the values above can be reset
        sampler2.reset()
        assert sampler2.n_moves == 0
        assert len(sampler2.Ns) == 0
        assert sampler2.n_accepted == 0

        # Try to run multiple moves and check that it works fine
        moves = 10
        sampler2.move(simulation2.context, n=moves)
        # Make sure that the right number of moves seem to have been carried out
        assert sampler2.n_moves == moves
        assert len(sampler2.Ns) == moves
        assert sampler2.n_accepted <= sampler2.n_moves

        return None


class TestNonequilibriumGCMCSampler(unittest.TestCase):
    """
    Class to store the tests for the NonequilibriumGCMCSampler class
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
