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


def setup_BaseGrandCanonicalMonteCarloSampler():
    """
    Set up variables for the GrandCanonicalMonteCarloSampler
    """
    # Make variables global so that they can be used
    global sampler1
    global simulation1

    pdb = PDBFile(utils.get_data_file(os.path.join('tests', 'bpti-ghosts.pdb')))
    ff = ForceField('amber10.xml', 'tip3p.xml')
    system = ff.createSystem(pdb.topology, nonbondedMethod=PME, nonbondedCutoff=12 * angstroms,
                             constraints=HBonds)

    ref_atoms = [{'name': 'CA', 'resname': 'TYR', 'resid': '10'},
                 {'name': 'CA', 'resname': 'ASN', 'resid': '43'}]

    sampler1 = samplers.BaseGrandCanonicalMonteCarloSampler(system=system, topology=pdb.topology,
                                                            temperature=300*kelvin,
                                                            ghostFile=os.path.join(outdir, 'bpti-ghost-wats.txt'),
                                                            log=os.path.join(outdir, 'basegcmcsampler.log'))

    # Define a simulation
    integrator = LangevinIntegrator(300 * kelvin, 1.0/picosecond, 0.002*picoseconds)

    try:
        platform = Platform.getPlatformByName('CUDA')
    except:
        try:
            platform = Platform.getPlatformByName('OpenCL')
        except:
            platform = Platform.getPlatformByName('CPU')

    simulation1 = Simulation(pdb.topology, system, integrator, platform)
    simulation1.context.setPositions(pdb.positions)
    simulation1.context.setVelocitiesToTemperature(300*kelvin)
    simulation1.context.setPeriodicBoxVectors(*pdb.topology.getPeriodicBoxVectors())
    sampler1.context = simulation1.context
    #sampler1.initialise(simulation1.context, [3054, 3055, 3056, 3057, 3058])

    return None


def setup_GCMCSphereSampler():
    """
    Set up variables for the GrandCanonicalMonteCarloSampler
    """
    # Make variables global so that they can be used
    global sampler2
    global simulation2

    pdb = PDBFile(utils.get_data_file(os.path.join('tests', 'bpti-ghosts.pdb')))
    ff = ForceField('amber10.xml', 'tip3p.xml')
    system = ff.createSystem(pdb.topology, nonbondedMethod=PME, nonbondedCutoff=12 * angstroms,
                              constraints=HBonds)

    ref_atoms = [{'name': 'CA', 'resname': 'TYR', 'resid': '10'},
                 {'name': 'CA', 'resname': 'ASN', 'resid': '43'}]

    sampler2 = samplers.GCMCSphereSampler(system=system, topology=pdb.topology, temperature=300*kelvin,
                                          referenceAtoms=ref_atoms, sphereRadius=4*angstroms,
                                          ghostFile=os.path.join(outdir, 'bpti-ghost-wats.txt'),
                                          log=os.path.join(outdir, 'gcmcspheresampler.log'))

    # Define a simulation
    integrator = LangevinIntegrator(300 * kelvin, 1.0/picosecond, 0.002*picoseconds)

    try:
        platform = Platform.getPlatformByName('CUDA')
    except:
        try:
            platform = Platform.getPlatformByName('OpenCL')
        except:
            platform = Platform.getPlatformByName('CPU')

    simulation2 = Simulation(pdb.topology, system, integrator, platform)
    simulation2.context.setPositions(pdb.positions)
    simulation2.context.setVelocitiesToTemperature(300*kelvin)
    simulation2.context.setPeriodicBoxVectors(*pdb.topology.getPeriodicBoxVectors())
    sampler2.initialise(simulation2.context, [3054, 3055, 3056, 3057, 3058])

    return None


class TestBaseGrandCanonicalMonteCarloSampler(unittest.TestCase):
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
        setup_BaseGrandCanonicalMonteCarloSampler()

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
        Make sure the BaseGrandCanonicalMonteCarloSampler.report() method works correctly
        """
        # Delete some ghost waters so they can be written out
        ghosts = [3054, 3055, 3056, 3057, 3058]
        sampler1.deleteGhostWaters(ghostResids=ghosts)

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


class TestGCMCSphereSampler(unittest.TestCase):
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
        setup_GCMCSphereSampler()

        return None

    def test_initialise(self):
        """
        Make sure the GrandCanonicalMonteCarloSampler.prepareGCMCSphere() method works correctly
        """

        # Make sure the variables are all updated
        assert isinstance(sampler2.context, Context)
        assert isinstance(sampler2.positions, Quantity)
        assert isinstance(sampler2.sphere_centre, Quantity)

        return None

    def test_deleteWatersInGCMCSphere(self):
        """
        Make sure the GrandCanonicalMonteCarloSampler.deleteWatersInGCMCSphere() method works correctly
        """
        # Now delete the waters in the sphere
        sampler2.deleteWatersInGCMCSphere()
        new_ghosts = [sampler2.gcmc_resids[id] for id in np.where(sampler2.gcmc_status == 0)[0]]
        # Check that the list of ghosts is correct
        assert new_ghosts == [70, 71, 3054, 3055, 3056, 3057, 3058]
        # Check that the variables match there being no waters in the GCMC region
        assert sampler2.N == 0
        assert all(sampler2.gcmc_status == 0)

        return None

    def test_updateGCMCSphere(self):
        """
        Make sure the GrandCanonicalMonteCarloSampler.updateGCMCSphere() method works correctly
        """
        # Get initial gcmc_resids and status
        gcmc_resids = deepcopy(sampler2.gcmc_resids)
        gcmc_status = deepcopy(sampler2.gcmc_status)
        sphere_centre = deepcopy(sampler2.sphere_centre)
        N = sampler2.N

        # Update the GCMC sphere (shouldn't change as the system won't have moved)
        state = simulation2.context.getState(getPositions=True, getVelocities=True)
        sampler2.updateGCMCSphere(state)

        # Make sure that these values are all still the same
        assert all(np.isclose(gcmc_resids, sampler2.gcmc_resids))
        assert all(np.isclose(gcmc_status, sampler2.gcmc_status))
        assert all(np.isclose(sphere_centre._value, sampler2.sphere_centre._value))
        assert N == sampler2.N

        return None

    def test_move(self):
        """
        Make sure the GCMCSphereSampler.move() method works correctly
        """
        # Shouldn't be able to run a move with this sampler
        self.assertRaises(NotImplementedError, lambda: sampler2.move(simulation2.context))

        return None


#class TestStandardGCMCSampler(unittest.TestCase):
#    """
#    Class to store the tests for the StandardGCMCSampler class
#    """
#    @classmethod
#    def setUpClass(cls):
#        """
#        Get things ready to run these tests
#        """
#        # Make the output directory if needed
#        if not os.path.isdir(os.path.join(os.path.dirname(__file__), 'output')):
#            os.mkdir(os.path.join(os.path.dirname(__file__), 'output'))
#        # Create a new directory if needed
#        if not os.path.isdir(outdir):
#            os.mkdir(outdir)
#        # If not, then clear any files already in the output directory so that they don't influence tests
#        else:
#            for file in os.listdir(outdir):
#                os.remove(os.path.join(outdir, file))
#
#        # Create sampler
#        setup_StandardGCMCSampler()
#
#        return None
#
#    def test_move(self):
#        """
#        Make sure the GrandCanonicalMonteCarloSampler.move() method works correctly
#        """
#        # Attempt a single move
#        sampler2.move(simulation2.context)
#        # Make sure that one move appears to have been carried out
#        assert sampler2.n_moves == 1
#        assert len(sampler2.Ns) == 1
#        assert sampler2.n_accepted <= sampler2.n_moves
#
#        # Make sure that the values above can be reset
#        sampler2.reset()
#        assert sampler2.n_moves == 0
#        assert len(sampler2.Ns) == 0
#        assert sampler2.n_accepted == 0
#
#        # Try to run multiple moves and check that it works fine
#        moves = 10
#        sampler2.move(simulation2.context, n=moves)
#        # Make sure that the right number of moves seem to have been carried out
#        assert sampler2.n_moves == moves
#        assert len(sampler2.Ns) == moves
#        assert sampler2.n_accepted <= sampler2.n_moves
#
#        return None
#
#
#class TestNonequilibriumGCMCSampler(unittest.TestCase):
#    """
#    Class to store the tests for the NonequilibriumGCMCSampler class
#    """
#    @classmethod
#    def setUpClass(cls):
#        """
#        Get things ready to run these tests
#        """
#        # Make the output directory if needed
#        if not os.path.isdir(os.path.join(os.path.dirname(__file__), 'output')):
#            os.mkdir(os.path.join(os.path.dirname(__file__), 'output'))
#        # Create a new directory if needed
#        if not os.path.isdir(outdir):
#            os.mkdir(outdir)
#        # If not, then clear any files already in the output directory so that they don't influence tests
#        else:
#            for file in os.listdir(outdir):
#                os.remove(os.path.join(outdir, file))
#
#        return None
