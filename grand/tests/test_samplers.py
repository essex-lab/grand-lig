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

        return None

    def setUp(self):
        """
        Create necessary variables for each test
        """
        pdb = PDBFile(utils.get_data_file(os.path.join('tests', 'bpti-ghosts.pdb')))
        ff = ForceField('amber10.xml', 'tip3p.xml')
        system = ff.createSystem(pdb.topology, nonbondedMethod=PME, nonbondedCutoff=12*angstroms,
                                 constraints=HBonds)

        self.sampler = samplers.GrandCanonicalMonteCarloSampler(system=system, topology=pdb.topology,
                                                                temperature=300*kelvin,
                                                                ghostFile=os.path.join(outdir, 'bpti-ghost-wats.txt'),
                                                                referenceAtoms=[['CA', 'TYR', '10'],
                                                                                ['CA', 'ASN', '43']],
                                                                sphereRadius=4*angstrom)

        # Define a simulation
        integrator = LangevinIntegrator(300*kelvin, 1.0/picosecond, 0.002*picoseconds)

        try:
            platform = Platform.getPlatformByName('OpenCL')
        except:
            platform = Platform.getPlatformByName('CPU')

        self.simulation = Simulation(pdb.topology, system, integrator, platform)
        self.simulation.context.setPositions(pdb.positions)
        self.simulation.context.setVelocitiesToTemperature(300 * kelvin)
        self.simulation.context.setPeriodicBoxVectors(*pdb.topology.getPeriodicBoxVectors())

        return None

    def test_prepareGCMCSphere(self):
        """
        Make sure the GrandCanonicalMonteCarloSampler.prepareGCMCSphere() method works correctly
        """
        # Need ghost waters to simulate, so the function should complain if there are none
        self.assertRaises(Exception, lambda: self.sampler.prepareGCMCSphere(self.simulation.context, []))

        # Happen to know what the ghost waters are for this example...
        ghosts = [3054, 3055, 3056, 3057, 3058]

        # Make sure the variables are all updated
        self.sampler.prepareGCMCSphere(self.simulation.context, ghosts)
        assert isinstance(self.sampler.context, Context)
        assert isinstance(self.sampler.positions, Quantity)
        assert isinstance(self.sampler.sphere_centre, Quantity)

        return None

    def test_deleteWatersInGCMCSphere(self):
        """
        Make sure the GrandCanonicalMonteCarloSampler.deleteWatersInGCMCSphere() method works correctly
        """
        # Prepare GCMC sphere first
        ghosts = [3054, 3055, 3056, 3057, 3058]
        self.sampler.prepareGCMCSphere(self.simulation.context, ghosts)

        # Now delete the waters in the sphere
        self.sampler.deleteWatersInGCMCSphere(self.simulation.context)
        new_ghosts = [self.sampler.gcmc_resids[id] for id in np.where(self.sampler.gcmc_status == 0)[0]]
        # Check that the list of ghosts is correct
        assert new_ghosts == [70, 71, 3054, 3055, 3056, 3057, 3058]
        # Check that the variables match there being no waters in the GCMC region
        assert self.sampler.N == 0
        assert all(self.sampler.gcmc_status == 0)

        return None

    def test_updateGCMCSphere(self):
        """
        Make sure the GrandCanonicalMonteCarloSampler.updateGCMCSphere() method works correctly
        """
        # Prepare GCMC sphere first
        ghosts = [3054, 3055, 3056, 3057, 3058]
        self.sampler.prepareGCMCSphere(self.simulation.context, ghosts)

        # Get initial gcmc_resids and status
        gcmc_resids = deepcopy(self.sampler.gcmc_resids)
        gcmc_status = deepcopy(self.sampler.gcmc_status)
        sphere_centre = deepcopy(self.sampler.sphere_centre)
        N = self.sampler.N

        # Update the GCMC sphere (shouldn't change as the system won't have moved)
        state = self.simulation.context.getState(getPositions=True, getVelocities=True)
        self.sampler.updateGCMCSphere(state)

        # Make sure that these values are all still the same
        assert all(np.isclose(gcmc_resids, self.sampler.gcmc_resids))
        assert all(np.isclose(gcmc_status, self.sampler.gcmc_status))
        assert all(np.isclose(sphere_centre._value, self.sampler.sphere_centre._value))
        assert N == self.sampler.N

        return None

    def test_move(self):
        """
        Make sure the GrandCanonicalMonteCarloSampler.move() method works correctly
        """
        # Shouldn't be able to run a move with this sampler
        self.assertRaises(NotImplementedError, lambda: self.sampler.move(self.simulation.context, n=1))

        return None

    def test_report(self):
        """
        Make sure the GrandCanonicalMonteCarloSampler.report() method works correctly
        """
        # Prepare GCMC sphere
        self.sampler.prepareGCMCSphere(self.simulation.context, [3054, 3055, 3056, 3057, 3058])

        # Get the list of ghost resids
        ghosts = [self.sampler.gcmc_resids[id] for id in np.where(self.sampler.gcmc_status == 0)[0]]

        # Report
        self.sampler.report()

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

        return None

    def setUp(self):
        """
        Create necessary variables for each test
        """
        pdb = PDBFile(utils.get_data_file(os.path.join('tests', 'bpti-ghosts.pdb')))
        ff = ForceField('amber10.xml', 'tip3p.xml')
        system = ff.createSystem(pdb.topology, nonbondedMethod=PME, nonbondedCutoff=12*angstroms,
                                 constraints=HBonds)

        self.sampler = samplers.StandardGCMCSampler(system=system, topology=pdb.topology,
                                                    temperature=300*kelvin,
                                                    ghostFile=os.path.join(outdir, 'bpti-ghost-wats.txt'),
                                                    referenceAtoms=[['CA', 'TYR', '10'],
                                                                    ['CA', 'ASN', '43']],
                                                    sphereRadius=4*angstrom)

        # Define a simulation
        integrator = LangevinIntegrator(300*kelvin, 1.0/picosecond, 0.002*picoseconds)

        try:
            platform = Platform.getPlatformByName('OpenCL')
        except:
            platform = Platform.getPlatformByName('CPU')

        self.simulation = Simulation(pdb.topology, system, integrator, platform)
        self.simulation.context.setPositions(pdb.positions)
        self.simulation.context.setVelocitiesToTemperature(300 * kelvin)
        self.simulation.context.setPeriodicBoxVectors(*pdb.topology.getPeriodicBoxVectors())

        # Prepare GCMC sphere
        ghosts = [3054, 3055, 3056, 3057, 3058]
        self.sampler.prepareGCMCSphere(self.simulation.context, ghosts)

        return None

    def test_move(self):
        """
        Make sure the GrandCanonicalMonteCarloSampler.move() method works correctly
        """
        # Attempt a single move
        self.sampler.move(self.simulation.context)
        # Make sure that one move appears to have been carried out
        assert self.sampler.n_moves == 1
        assert len(self.sampler.Ns) == 1
        assert self.sampler.n_accepted <= self.sampler.n_moves

        # Make sure that the values above can be reset
        self.sampler.reset()
        assert self.sampler.n_moves == 0
        assert len(self.sampler.Ns) == 0
        assert self.sampler.n_accepted == 0

        # Try to run multiple moves and check that it works fine
        moves = 10
        self.sampler.move(self.simulation.context, n=moves)
        # Make sure that the right number of moves seem to have been carried out
        assert self.sampler.n_moves == moves
        assert len(self.sampler.Ns) == moves
        assert self.sampler.n_accepted <= self.sampler.n_moves

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



