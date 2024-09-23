"""
Description
-----------
This file contains functions written to test the functions in the grandlig.samplers submodule

Marley Samways
"""

import os
import unittest
import numpy as np
from copy import deepcopy
from openmm.app import *
from openmm import *
from openmm.unit import *
from grandlig import samplers
from grandlig import utils
from openmmtools.integrators import BAOABIntegrator

outdir = os.path.join(os.path.dirname(__file__), "output", "samplers")
# if os.path.exists(outdir):
#     os.rmdir(outdir)
# os.makedirs(outdir)


def setup_BaseGrandCanonicalMonteCarloSampler(outdir):
    """
    Set up variables for the GrandCanonicalMonteCarloSampler
    """
    # Make variables global so that they can be used
    global base_gcmc_sampler
    global base_gcmc_simulation

    pdb = PDBFile(
        utils.get_data_file(os.path.join("tests", "bpti-ghosts.pdb"))
    )
    ff = ForceField("amber14-all.xml", "amber14/tip3p.xml")
    system = ff.createSystem(
        pdb.topology,
        nonbondedMethod=PME,
        nonbondedCutoff=12 * angstroms,
        constraints=HBonds,
    )

    base_gcmc_sampler = samplers.BaseGrandCanonicalMonteCarloSampler(
        system=system,
        topology=pdb.topology,
        temperature=300 * kelvin,
        ghostFile=os.path.join(outdir, "bpti-ghost-wats.txt"),
        log=os.path.join(outdir, "basegcmcsampler.log"),
    )

    # Define a simulation
    integrator = BAOABIntegrator(
        300 * kelvin, 1.0 / picosecond, 0.002 * picoseconds
    )


    platform = Platform.getPlatformByName("CPU")

    base_gcmc_simulation = Simulation(
        pdb.topology, system, integrator, platform
    )
    base_gcmc_simulation.context.setPositions(pdb.positions)
    base_gcmc_simulation.context.setVelocitiesToTemperature(300 * kelvin)
    base_gcmc_simulation.context.setPeriodicBoxVectors(
        *pdb.topology.getPeriodicBoxVectors()
    )

    # Set up the sampler
    base_gcmc_sampler.context = base_gcmc_simulation.context

    return None


def setup_GCMCSphereSampler(outdir):
    """
    Set up variables for the GCMCSphereSampler
    """
    # Make variables global so that they can be used
    global gcmc_sphere_sampler
    global gcmc_sphere_simulation

    pdb = PDBFile(
        utils.get_data_file(os.path.join("tests", "bpti-ghosts.pdb"))
    )
    ff = ForceField("amber14-all.xml", "amber14/tip3p.xml")
    system = ff.createSystem(
        pdb.topology,
        nonbondedMethod=PME,
        nonbondedCutoff=12 * angstroms,
        constraints=HBonds,
    )

    ref_atoms = [
        {"name": "CA", "resname": "TYR", "resid": "10"},
        {"name": "CA", "resname": "ASN", "resid": "43"},
    ]

    gcmc_sphere_sampler = samplers.GCMCSphereSampler(
        system=system,
        topology=pdb.topology,
        temperature=300 * kelvin,
        referenceAtoms=ref_atoms,
        sphereRadius=4 * angstroms,
        ghostFile=os.path.join(outdir, "bpti-ghost-wats.txt"),
        log=os.path.join(outdir, "gcmcspheresampler.log"),
    )

    # Define a simulation
    integrator = BAOABIntegrator(
        300 * kelvin, 1.0 / picosecond, 0.002 * picoseconds
    )


    platform = Platform.getPlatformByName("CPU")

    gcmc_sphere_simulation = Simulation(
        pdb.topology, system, integrator, platform
    )
    gcmc_sphere_simulation.context.setPositions(pdb.positions)
    gcmc_sphere_simulation.context.setVelocitiesToTemperature(300 * kelvin)
    gcmc_sphere_simulation.context.setPeriodicBoxVectors(
        *pdb.topology.getPeriodicBoxVectors()
    )

    # Set up the sampler
    gcmc_sphere_sampler.initialise(
        gcmc_sphere_simulation.context,
        gcmc_sphere_simulation,
        [3054, 3055, 3056, 3057, 3058],
    )

    return None


def setup_StandardGCMCSphereSampler(outdir):
    """
    Set up variables for the StandardGCMCSphereSampler
    """
    # Make variables global so that they can be used
    global std_gcmc_sphere_sampler
    global std_gcmc_sphere_simulation

    pdb = PDBFile(
        utils.get_data_file(os.path.join("tests", "bpti-ghosts.pdb"))
    )
    ff = ForceField("amber14-all.xml", "amber14/tip3p.xml")
    system = ff.createSystem(
        pdb.topology,
        nonbondedMethod=PME,
        nonbondedCutoff=12 * angstroms,
        constraints=HBonds,
    )

    ref_atoms = [
        {"name": "CA", "resname": "TYR", "resid": "10"},
        {"name": "CA", "resname": "ASN", "resid": "43"},
    ]

    std_gcmc_sphere_sampler = samplers.StandardGCMCSphereSampler(
        system=system,
        topology=pdb.topology,
        temperature=300 * kelvin,
        referenceAtoms=ref_atoms,
        sphereRadius=4 * angstroms,
        ghostFile=os.path.join(outdir, "bpti-ghost-wats.txt"),
        log=os.path.join(outdir, "stdgcmcspheresampler.log"),
    )

    # Define a simulation
    integrator = BAOABIntegrator(
        300 * kelvin, 1.0 / picosecond, 0.002 * picoseconds
    )

    platform = Platform.getPlatformByName("CPU")

    std_gcmc_sphere_simulation = Simulation(
        pdb.topology, system, integrator, platform
    )
    std_gcmc_sphere_simulation.context.setPositions(pdb.positions)
    std_gcmc_sphere_simulation.context.setVelocitiesToTemperature(300 * kelvin)
    std_gcmc_sphere_simulation.context.setPeriodicBoxVectors(
        *pdb.topology.getPeriodicBoxVectors()
    )

    # Set up the sampler
    std_gcmc_sphere_sampler.initialise(
        std_gcmc_sphere_simulation.context,
        std_gcmc_sphere_simulation,
        [3054, 3055, 3056, 3057, 3058],
    )

    return None


def setup_NonequilibriumGCMCSphereSampler(outdir):
    """
    Set up variables for the GrandCanonicalMonteCarloSampler
    """
    # Make variables global so that they can be used
    global neq_gcmc_sphere_sampler
    global neq_gcmc_sphere_simulation

    pdb = PDBFile(
        utils.get_data_file(os.path.join("tests", "bpti-ghosts.pdb"))
    )
    ff = ForceField("amber14-all.xml", "amber14/tip3p.xml")
    system = ff.createSystem(
        pdb.topology,
        nonbondedMethod=PME,
        nonbondedCutoff=12 * angstroms,
        constraints=HBonds,
    )

    ref_atoms = [
        {"name": "CA", "resname": "TYR", "resid": "10"},
        {"name": "CA", "resname": "ASN", "resid": "43"},
    ]

    integrator = BAOABIntegrator(
        300 * kelvin, 1.0 / picosecond, 0.002 * picoseconds
    )

    neq_gcmc_sphere_sampler = samplers.NonequilibriumGCMCSphereSampler(
        system=system,
        topology=pdb.topology,
        temperature=300 * kelvin,
        referenceAtoms=ref_atoms,
        sphereRadius=4 * angstroms,
        integrator=integrator,
        nPropStepsPerPert=10,
        nPertSteps=1,
        ghostFile=os.path.join(outdir, "bpti-ghost-wats.txt"),
        log=os.path.join(outdir, "neqgcmcspheresampler.log"),
    )

    # Define a simulation

    platform = Platform.getPlatformByName("CPU")

    neq_gcmc_sphere_simulation = Simulation(
        pdb.topology, system, neq_gcmc_sphere_sampler.integrator, platform
    )
    neq_gcmc_sphere_simulation.context.setPositions(pdb.positions)
    neq_gcmc_sphere_simulation.context.setVelocitiesToTemperature(300 * kelvin)
    neq_gcmc_sphere_simulation.context.setPeriodicBoxVectors(
        *pdb.topology.getPeriodicBoxVectors()
    )

    # Set up the sampler
    neq_gcmc_sphere_sampler.initialise(
        neq_gcmc_sphere_simulation.context,
        neq_gcmc_sphere_simulation,
        [3054, 3055, 3056, 3057, 3058],
    )

    return None


def setup_GCMCSystemSampler(outdir):
    """
    Set up variables for the GCMCSystemSampler
    """
    # Make variables global so that they can be used
    global gcmc_system_sampler
    global gcmc_system_simulation

    pdb = PDBFile(
        utils.get_data_file(os.path.join("tests", "water-ghosts.pdb"))
    )
    ff = ForceField("tip3p.xml")
    system = ff.createSystem(
        pdb.topology,
        nonbondedMethod=PME,
        nonbondedCutoff=12 * angstroms,
        constraints=HBonds,
    )

    gcmc_system_sampler = samplers.GCMCSystemSampler(
        system=system,
        topology=pdb.topology,
        temperature=300 * kelvin,
        boxVectors=np.array(pdb.topology.getPeriodicBoxVectors()),
        ghostFile=os.path.join(outdir, "water-ghost-wats.txt"),
        log=os.path.join(outdir, "gcmcsystemsampler.log"),
    )

    # Define a simulation
    integrator = BAOABIntegrator(
        300 * kelvin, 1.0 / picosecond, 0.002 * picoseconds
    )
    platform = Platform.getPlatformByName("CPU")


    gcmc_system_simulation = Simulation(
        pdb.topology, system, integrator, platform
    )
    gcmc_system_simulation.context.setPositions(pdb.positions)
    gcmc_system_simulation.context.setVelocitiesToTemperature(300 * kelvin)
    gcmc_system_simulation.context.setPeriodicBoxVectors(
        *pdb.topology.getPeriodicBoxVectors()
    )

    # Set up the sampler
    gcmc_system_sampler.initialise(
        gcmc_system_simulation.context,
        gcmc_system_simulation,
        [2094, 2095, 2096, 2097, 2098],
    )

    return None


def setup_StandardGCMCSystemSampler(outdir):
    """
    Set up variables for the StandardGCMCSystemSampler
    """
    # Make variables global so that they can be used
    global std_gcmc_system_sampler
    global std_gcmc_system_simulation
    outdir = os.path.join(
        os.path.dirname(__file__), "output", "samplers/std_system"
    )
    pdb = PDBFile(
        utils.get_data_file(os.path.join("tests", "water-ghosts.pdb"))
    )
    ff = ForceField("tip3p.xml")
    system = ff.createSystem(
        pdb.topology,
        nonbondedMethod=PME,
        nonbondedCutoff=12 * angstroms,
        constraints=HBonds,
    )

    std_gcmc_system_sampler = samplers.StandardGCMCSystemSampler(
        system=system,
        topology=pdb.topology,
        temperature=300 * kelvin,
        boxVectors=np.array(pdb.topology.getPeriodicBoxVectors()),
        ghostFile=os.path.join(outdir, "water-ghost-wats.txt"),
        log=os.path.join(outdir, "stdgcmcsystemsampler.log"),
    )

    # Define a simulation
    integrator = BAOABIntegrator(
        300 * kelvin, 1.0 / picosecond, 0.002 * picoseconds
    )

    platform = Platform.getPlatformByName("CPU")

    std_gcmc_system_simulation = Simulation(
        pdb.topology, system, integrator, platform
    )
    std_gcmc_system_simulation.context.setPositions(pdb.positions)
    std_gcmc_system_simulation.context.setVelocitiesToTemperature(300 * kelvin)
    std_gcmc_system_simulation.context.setPeriodicBoxVectors(
        *pdb.topology.getPeriodicBoxVectors()
    )

    # Set up the sampler
    std_gcmc_system_sampler.initialise(
        std_gcmc_system_simulation.context,
        std_gcmc_system_simulation,
        [2094, 2095, 2096, 2097, 2098],
    )

    return None


def setup_NonequilibriumGCMCSystemSampler(outdir):
    """
    Set up variables for the StandardGCMCSystemSampler
    """
    # Make variables global so that they can be used
    global neq_gcmc_system_sampler
    global neq_gcmc_system_simulation
    outdir = os.path.join(
        os.path.dirname(__file__), "output", "samplers/noneq_system"
    )

    pdb = PDBFile(
        utils.get_data_file(os.path.join("tests", "water-ghosts.pdb"))
    )
    ff = ForceField("tip3p.xml")
    system = ff.createSystem(
        pdb.topology,
        nonbondedMethod=PME,
        nonbondedCutoff=12 * angstroms,
        constraints=HBonds,
    )

    integrator = BAOABIntegrator(
        300 * kelvin, 1.0 / picosecond, 0.002 * picoseconds
    )

    neq_gcmc_system_sampler = samplers.NonequilibriumGCMCSystemSampler(
        system=system,
        topology=pdb.topology,
        temperature=300 * kelvin,
        integrator=integrator,
        boxVectors=np.array(pdb.topology.getPeriodicBoxVectors()),
        ghostFile=os.path.join(outdir, "water-ghost-wats.txt"),
        log=os.path.join(outdir, "neqgcmcsystemsampler.log"),
    )

    # Define a simulation

    platform = Platform.getPlatformByName("CPU")

    neq_gcmc_system_simulation = Simulation(
        pdb.topology, system, neq_gcmc_system_sampler.integrator, platform
    )
    neq_gcmc_system_simulation.context.setPositions(pdb.positions)
    neq_gcmc_system_simulation.context.setVelocitiesToTemperature(300 * kelvin)
    neq_gcmc_system_simulation.context.setPeriodicBoxVectors(
        *pdb.topology.getPeriodicBoxVectors()
    )

    # Set up the sampler
    neq_gcmc_system_sampler.initialise(
        neq_gcmc_system_simulation.context,
        neq_gcmc_system_simulation,
        [2094, 2095, 2096, 2097, 2098],
    )

    return None


class TestBaseGrandCanonicalMonteCarloSampler(unittest.TestCase):
    """
    Class to store the tests for the GrandCanonicalMonteCarloSampler class
    """

    outdir = os.path.join(os.path.dirname(__file__), "output", "samplers/base")

    @classmethod
    def setUpClass(cls):
        """
        Get things ready to run these tests
        """
        outdir = os.path.join(
            os.path.dirname(__file__), "output", "samplers/base"
        )

        # Make the output directory if needed
        if not os.path.isdir(
            os.path.join(os.path.dirname(__file__), "output")
        ):
            os.makedirs(os.path.join(os.path.dirname(__file__), "output"))
        # Create a new directory if needed
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        # If not, then clear any files already in the output directory so that they don't influence tests
        else:
            for file in os.listdir(outdir):
                os.remove(os.path.join(outdir, file))

        # Need to create the sampler
        setup_BaseGrandCanonicalMonteCarloSampler(outdir)

        return None

    def test_move(self):
        """
        Make sure the GrandCanonicalMonteCarloSampler.move() method works correctly
        """
        # Shouldn't be able to run a move with this sampler
        self.assertRaises(
            NotImplementedError,
            lambda: base_gcmc_sampler.move(base_gcmc_simulation.context),
        )

        return None

    def test_report(self):
        """
        Make sure the BaseGrandCanonicalMonteCarloSampler.report() method works correctly

        """
        outdir = os.path.join(
            os.path.dirname(__file__), "output", "samplers/base"
        )
        # Delete some ghost waters so they can be written out
        ghosts = [3054, 3055, 3056, 3057, 3058]
        base_gcmc_sampler.deleteGhostMolecules(ghostResids=ghosts)

        # Report
        base_gcmc_sampler.report(base_gcmc_simulation)

        # Check the output to the ghost file
        assert os.path.isfile(os.path.join(outdir, "bpti-ghost-wats.txt"))
        # Read which ghosts were written
        with open(os.path.join(outdir, "bpti-ghost-wats.txt"), "r") as f:
            n_lines = 0
            lines = f.readlines()
            for line in lines:
                if len(line.split()) > 0:
                    n_lines += 1
        assert n_lines == 1
        ghosts_read = [int(resid) for resid in lines[0].split(",")]
        assert all(np.isclose(ghosts, ghosts_read))

        return None

    def test_reset(self):
        """
        Make sure the BaseGrandCanonicalMonteCarloSampler.reset() method works correctly
        """
        keys = []
        types = []
        for key in base_gcmc_sampler.tracked_variables.keys():
            keys.append(key)
            types.append(type(base_gcmc_sampler.tracked_variables[key]))

        for i in range(len(keys)):
            if types[i] == list:
                base_gcmc_sampler.tracked_variables[keys[i]] = [1, 2, 3, 4]
            elif types[i] == int:
                base_gcmc_sampler.tracked_variables[keys[i]] = 99
            elif types[i] != list or types[i] != int:
                raise Exception(
                    "Found a tracked variable that is not a list or an integer."
                )

        # Reset base_gcmc_sampler
        base_gcmc_sampler.reset()
        for i in range(len(keys)):
            if types[i] == list:
                assert base_gcmc_sampler.tracked_variables[keys[i]] == []
            elif types[i] == int:
                assert base_gcmc_sampler.tracked_variables[keys[i]] == 0
            elif types[i] != list or types[i] != int:
                raise Exception(
                    "Found a tracked variable that is not a list or an integer."
                )

        # Check that some specific values have been reset
        assert base_gcmc_sampler.tracked_variables["n_accepted"] == 0
        assert base_gcmc_sampler.tracked_variables["n_moves"] == 0
        assert len(base_gcmc_sampler.tracked_variables["Ns"]) == 0

        return None


class TestGCMCSphereSampler(unittest.TestCase):
    """
    Class to store the tests for the GCMCSphereSampler class
    """

    @classmethod
    def setUpClass(cls):
        """
        Get things ready to run these tests
        """
        outdir = os.path.join(
            os.path.dirname(__file__), "output", "samplers/sphere"
        )

        # Make the output directory if needed
        if not os.path.isdir(
            os.path.join(os.path.dirname(__file__), "output")
        ):
            os.makedirs(os.path.join(os.path.dirname(__file__), "output"))
        # Create a new directory if needed
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        # If not, then clear any files already in the output directory so that they don't influence tests
        else:
            for file in os.listdir(outdir):
                os.remove(os.path.join(outdir, file))

        # Need to create the sampler
        setup_GCMCSphereSampler(outdir)

        return None

    def test_initialise(self):
        """
        Make sure the GCMCSphereSampler.initialise() method works correctly
        """

        # Make sure the variables are all updated
        assert isinstance(gcmc_sphere_sampler.context, Context)
        assert isinstance(gcmc_sphere_sampler.positions, Quantity)
        assert isinstance(gcmc_sphere_sampler.sphere_centre, Quantity)

        return None

    def test_deleteMoleculesInGCMCSphere(self):
        """
        Make sure the GCMCSphereSampler.deleteWatersInGCMCSphere() method works correctly
        """
        # Now delete the waters in the sphere
        gcmc_sphere_sampler.deleteMoleculesInGCMCSphere()
        new_ghosts = gcmc_sphere_sampler.getMolStatusResids(0)
        # Check that the list of ghosts is correct
        assert new_ghosts == [70, 71, 3054, 3055, 3056, 3057, 3058]
        # Check that the variables match there being no waters in the GCMC region
        assert gcmc_sphere_sampler.N == 0
        assert all(
            [x in [0, 2] for x in gcmc_sphere_sampler.mol_status.values()]
        )

        # turn then back on so we have something to delete later on
        gcmc_sphere_sampler.setMolStatus(70, 1)
        gcmc_sphere_sampler.setMolStatus(71, 1)

        state = gcmc_sphere_simulation.context.getState(
            getPositions=True, getVelocities=True
        )
        gcmc_sphere_sampler.updateGCMCSphere(state)

        return None

    def test_updateGCMCSphere(self):
        """
        Make sure the GCMCSphereSampler.updateGCMCSphere() method works correctly
        """
        # Get initial gcmc_resids and status
        gcmc_resids = deepcopy(gcmc_sphere_sampler.getMolStatusResids(1))
        sphere_centre = deepcopy(gcmc_sphere_sampler.sphere_centre)
        N = gcmc_sphere_sampler.N

        # Update the GCMC sphere (shouldn't change as the system won't have moved)
        state = gcmc_sphere_simulation.context.getState(
            getPositions=True, getVelocities=True
        )
        gcmc_sphere_sampler.updateGCMCSphere(state)

        # Make sure that these values are all still the same
        assert all(
            np.isclose(gcmc_resids, gcmc_sphere_sampler.getMolStatusResids(1))
        )
        assert all(
            np.isclose(
                sphere_centre._value, gcmc_sphere_sampler.sphere_centre._value
            )
        )
        assert N == gcmc_sphere_sampler.N

        return None

    def test_move(self):
        """
        Make sure the GCMCSphereSampler.move() method works correctly
        """
        # Shouldn't be able to run a move with this sampler
        self.assertRaises(
            NotImplementedError,
            lambda: gcmc_sphere_sampler.move(gcmc_sphere_simulation.context),
        )

        return None

    def test_insertRandomMolecule(self):
        """
        Make sure the GCMCSphereSampler.insertRandomWater() method works correctly
        """
        # Insert a random water
        new_positions, gcmc_id = gcmc_sphere_sampler.insertRandomMolecule()

        # Check that the indices returned are integers - may not be type int
        assert gcmc_id == int(gcmc_id)

        atom_ids = gcmc_sphere_sampler.mol_atom_ids[gcmc_id]
        # Check that the new positions are different to the old positions
        assert all(
            [
                any(
                    [
                        new_positions[i][j]
                        != gcmc_sphere_sampler.positions[i][j]
                        for j in range(3)
                    ]
                )
                for i in atom_ids
            ]
        )
        assert all(
            [
                all(
                    [
                        new_positions[i][j]
                        == gcmc_sphere_sampler.positions[i][j]
                        for j in range(3)
                    ]
                )
                for i in range(len(new_positions))
                if i not in atom_ids
            ]
        )

        return None

    def test_deleteRandomMolecule(self):
        """
        Make sure the GCMCSphereSampler.deleteRandomWater() method works correctly
        """
        # Delete a random water
        gcmc_id = gcmc_sphere_sampler.deleteRandomMolecule()

        # Check that the indices returned are integers
        assert gcmc_id == int(gcmc_id)

        return None


class TestStandardGCMCSphereSampler(unittest.TestCase):
    """
    Class to store the tests for the StandardGCMCSphereSampler class
    """

    @classmethod
    def setUpClass(cls):
        """
        Get things ready to run these tests
        """
        outdir = os.path.join(
            os.path.dirname(__file__), "output", "samplers/std_sphere"
        )

        # Make the output directory if needed
        if not os.path.isdir(
            os.path.join(os.path.dirname(__file__), "output")
        ):
            os.makedirs(os.path.join(os.path.dirname(__file__), "output"))
        # Create a new directory if needed
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        # If not, then clear any files already in the output directory so that they don't influence tests
        else:
            for file in os.listdir(outdir):
                os.remove(os.path.join(outdir, file))

        # Create sampler
        setup_StandardGCMCSphereSampler(outdir)

        return None

    def test_move(self):
        """
        Make sure the StandardGCMCSphereSampler.move() method works correctly
        """
        # Run a handful of GCMC moves
        n_moves = 10
        std_gcmc_sphere_sampler.move(
            std_gcmc_sphere_simulation.context, n_moves
        )

        # Check that all of the appropriate variables seem to have been updated
        # Hard to test individual moves as they are rarely accepted - just need to check the overall behaviour
        assert std_gcmc_sphere_sampler.tracked_variables["n_moves"] == n_moves
        assert (
            0
            <= std_gcmc_sphere_sampler.tracked_variables["n_accepted"]
            <= n_moves
        )
        assert len(std_gcmc_sphere_sampler.tracked_variables["Ns"]) == n_moves
        assert (
            len(
                std_gcmc_sphere_sampler.tracked_variables[
                    "acceptance_probabilities"
                ]
            )
            == n_moves
        )
        assert isinstance(std_gcmc_sphere_sampler.energy, Quantity)
        assert std_gcmc_sphere_sampler.energy.unit.is_compatible(
            kilocalories_per_mole
        )

        return None


class TestNonequilibriumGCMCSphereSampler(unittest.TestCase):
    """
    Class to store the tests for the NonequilibriumGCMCSphereSampler class
    """

    @classmethod
    def setUpClass(cls):
        """
        Get things ready to run these tests
        """
        outdir = os.path.join(
            os.path.dirname(__file__), "output", "samplers/noneq_sphere"
        )

        # Make the output directory if needed
        if not os.path.isdir(
            os.path.join(os.path.dirname(__file__), "output")
        ):
            os.makedirs(os.path.join(os.path.dirname(__file__), "output"))
        # Create a new directory if needed
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        # If not, then clear any files already in the output directory so that they don't influence tests
        else:
            for file in os.listdir(outdir):
                os.remove(os.path.join(outdir, file))

        # Create sampler
        setup_NonequilibriumGCMCSphereSampler(outdir)

        return None

    def test_move(self):
        """
        Make sure the NonequilibriumGCMCSphereSampler.move() method works correctly
        """
        neq_gcmc_sphere_sampler.reset()

        # Just run one move, as they are a bit more expensive
        neq_gcmc_sphere_sampler.move(neq_gcmc_sphere_simulation.context, 1)

        # Check some of the variables have been updated as appropriate
        assert neq_gcmc_sphere_sampler.tracked_variables["n_moves"] == 1
        assert (
            0 <= neq_gcmc_sphere_sampler.tracked_variables["n_accepted"] <= 1
        )
        assert len(neq_gcmc_sphere_sampler.tracked_variables["Ns"]) == 1
        assert (
            len(
                neq_gcmc_sphere_sampler.tracked_variables[
                    "acceptance_probabilities"
                ]
            )
            == 1
        )

        # Check the NCMC-specific variables
        assert isinstance(neq_gcmc_sphere_sampler.velocities, Quantity)
        assert neq_gcmc_sphere_sampler.velocities.unit.is_compatible(
            nanometers / picosecond
        )
        
        if neq_gcmc_sphere_sampler.tracked_variables["n_explosions"] != 1:
            assert (
                len(neq_gcmc_sphere_sampler.tracked_variables["insert_works"]) == 1
                or len(neq_gcmc_sphere_sampler.tracked_variables["delete_works"])
                == 1
            )
        assert (
            0
            <= neq_gcmc_sphere_sampler.tracked_variables["n_left_sphere"]
            <= 1
        )
        assert (
            0 <= neq_gcmc_sphere_sampler.tracked_variables["n_explosions"] <= 1
        )

        return None

    def test_insertionMove(self):
        """
        Make sure the NonequilibriumGCMCSphereSampler.insertionMove() method works correctly
        """
        # Prep for a move
        # Read in positions
        neq_gcmc_sphere_sampler.context = neq_gcmc_sphere_simulation.context
        state = neq_gcmc_sphere_sampler.context.getState(
            getPositions=True, enforcePeriodicBox=True, getVelocities=True
        )
        neq_gcmc_sphere_sampler.positions = deepcopy(
            state.getPositions(asNumpy=True)
        )
        neq_gcmc_sphere_sampler.velocities = deepcopy(
            state.getVelocities(asNumpy=True)
        )

        # Update GCMC region based on current state
        neq_gcmc_sphere_sampler.updateGCMCSphere(state)

        # # Set to NCMC integrator
        # neq_gcmc_sphere_sampler.compound_integrator.setCurrentIntegrator(1)

        # Just run one move to make sure it doesn't crash
        neq_gcmc_sphere_sampler.insertionMove()

        # # Reset the compound integrator
        # neq_gcmc_sphere_sampler.compound_integrator.setCurrentIntegrator(0)

        return None

    def test_deletionMove(self):
        """
        Make sure the NonequilibriumGCMCSphereSampler.deletionMove() method works correctly
        """
        # Prep for a move
        # Read in positions
        neq_gcmc_sphere_sampler.context = neq_gcmc_sphere_simulation.context
        state = neq_gcmc_sphere_sampler.context.getState(
            getPositions=True, enforcePeriodicBox=True, getVelocities=True
        )
        neq_gcmc_sphere_sampler.positions = deepcopy(
            state.getPositions(asNumpy=True)
        )
        neq_gcmc_sphere_sampler.velocities = deepcopy(
            state.getVelocities(asNumpy=True)
        )

        # Update GCMC region based on current state
        neq_gcmc_sphere_sampler.updateGCMCSphere(state)

        # # Set to NCMC integrator
        # neq_gcmc_sphere_sampler.integrator.setCurrentIntegrator(1)

        # Just run one move to make sure it doesn't crash
        neq_gcmc_sphere_sampler.deletionMove()

        # # Reset the compound integrator
        # neq_gcmc_sphere_sampler.integrator.setCurrentIntegrator(0)

        return None


class TestGCMCSystemSampler(unittest.TestCase):
    """
    Class to store the tests for the GCMCSystemSampler class
    """

    @classmethod
    def setUpClass(cls):
        """
        Get things ready to run these tests
        """
        outdir = os.path.join(
            os.path.dirname(__file__), "output", "samplers/system"
        )
        # Make the output directory if needed
        if not os.path.isdir(
            os.path.join(os.path.dirname(__file__), "output")
        ):
            os.makedirs(os.path.join(os.path.dirname(__file__), "output"))
        # Create a new directory if needed
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        # If not, then clear any files already in the output directory so that they don't influence tests
        else:
            for file in os.listdir(outdir):
                os.remove(os.path.join(outdir, file))

        # Need to create the sampler
        setup_GCMCSystemSampler(outdir)

        return None

    def test_initialise(self):
        """
        Make sure the GCMCSystemSampler.initialise() method works correctly
        """
        # Make sure the variables are all updated
        assert isinstance(gcmc_system_sampler.context, Context)
        assert isinstance(gcmc_system_sampler.positions, Quantity)
        assert isinstance(gcmc_system_sampler.simulation_box, Quantity)

        return None

    def test_move(self):
        """
        Make sure the GCMCSystemSampler.move() method works correctly
        """
        # Shouldn't be able to run a move with this sampler
        self.assertRaises(
            NotImplementedError,
            lambda: gcmc_system_sampler.move(gcmc_system_simulation.context),
        )

        return None

    def test_insertRandomMolecule(self):
        """
        Make sure the GCMCSystemSampler.insertRandomWater() method works correctly
        """
        # Insert a random water
        new_positions, gcmc_id = gcmc_system_sampler.insertRandomMolecule()

        # Check that the indices returned are integers - may not be type int
        assert gcmc_id == int(gcmc_id)

        atom_ids = gcmc_system_sampler.mol_atom_ids[gcmc_id]
        # Check that the new positions are different to the old positions
        assert all(
            [
                any(
                    [
                        new_positions[i][j]
                        != gcmc_system_sampler.positions[i][j]
                        for j in range(3)
                    ]
                )
                for i in atom_ids
            ]
        )
        assert all(
            [
                all(
                    [
                        new_positions[i][j]
                        == gcmc_system_sampler.positions[i][j]
                        for j in range(3)
                    ]
                )
                for i in range(len(new_positions))
                if i not in atom_ids
            ]
        )

        return None

    def test_deleteRandomMolecule(self):
        """
        Make sure the GCMCSystemSampler.deleteRandomWater() method works correctly
        """
        # Delte a random water
        gcmc_id = gcmc_system_sampler.deleteRandomMolecule()

        # Check that the indices returned are integers
        assert gcmc_id == int(gcmc_id)

        return None


class TestStandardGCMCSystemSampler(unittest.TestCase):
    """
    Class to store the tests for the StandardGCMCSystemSampler class
    """

    @classmethod
    def setUpClass(cls):
        """
        Get things ready to run these tests
        """
        outdir = os.path.join(
            os.path.dirname(__file__), "output", "samplers/std_system"
        )
        # Make the output directory if needed
        if not os.path.isdir(
            os.path.join(os.path.dirname(__file__), "output")
        ):
            os.makedirs(os.path.join(os.path.dirname(__file__), "output"))
        # Create a new directory if needed
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        # If not, then clear any files already in the output directory so that they don't influence tests
        else:
            for file in os.listdir(outdir):
                os.remove(os.path.join(outdir, file))

        # Create sampler
        setup_StandardGCMCSystemSampler(outdir)

        return None

    def test_move(self):
        """
        Make sure the StandardGCMCSystemSampler.move() method works correctly
        """
        # Run a handful of GCMC moves
        n_moves = 10
        std_gcmc_system_sampler.move(
            std_gcmc_system_simulation.context, n_moves
        )

        # Check that all of the appropriate variables seem to have been updated
        # Hard to test individual moves as they are rarely accepted - just need to check the overall behaviour
        assert std_gcmc_system_sampler.tracked_variables["n_moves"] == n_moves
        assert (
            0
            <= std_gcmc_system_sampler.tracked_variables["n_accepted"]
            <= n_moves
        )
        assert len(std_gcmc_system_sampler.tracked_variables["Ns"]) == n_moves
        assert (
            len(
                std_gcmc_system_sampler.tracked_variables[
                    "acceptance_probabilities"
                ]
            )
            == n_moves
        )
        assert isinstance(std_gcmc_system_sampler.energy, Quantity)
        assert std_gcmc_system_sampler.energy.unit.is_compatible(
            kilocalories_per_mole
        )

        return None


class TestNonequilibriumGCMCSystemSampler(unittest.TestCase):
    """
    Class to store the tests for the NonequilibriumGCMCSystemSampler class
    """

    @classmethod
    def setUpClass(cls):
        """
        Get things ready to run these tests
        """
        outdir = os.path.join(
            os.path.dirname(__file__), "output", "samplers/noneq_system"
        )
        # Make the output directory if needed
        if not os.path.isdir(
            os.path.join(os.path.dirname(__file__), "output")
        ):
            os.makedirs(os.path.join(os.path.dirname(__file__), "output"))
        # Create a new directory if needed
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        # If not, then clear any files already in the output directory so that they don't influence tests
        else:
            for file in os.listdir(outdir):
                os.remove(os.path.join(outdir, file))

        # Create sampler
        setup_NonequilibriumGCMCSystemSampler(outdir)

        return None

    def test_move(self):
        """
        Make sure the NonequilibriumGCMCSystemSampler.move() method works correctly
        """
        neq_gcmc_system_sampler.reset()

        # Just run one move, as they are a bit more expensive
        neq_gcmc_system_sampler.move(neq_gcmc_system_simulation.context, 1)

        # Check some of the variables have been updated as appropriate
        assert neq_gcmc_system_sampler.tracked_variables["n_moves"] == 1
        assert (
            0 <= neq_gcmc_system_sampler.tracked_variables["n_accepted"] <= 1
        )
        assert len(neq_gcmc_system_sampler.tracked_variables["Ns"]) == 1
        assert (
            len(
                neq_gcmc_system_sampler.tracked_variables[
                    "acceptance_probabilities"
                ]
            )
            == 1
        )

        # Check the NCMC-specific variables
        assert isinstance(neq_gcmc_system_sampler.velocities, Quantity)
        assert neq_gcmc_system_sampler.velocities.unit.is_compatible(
            nanometers / picosecond
        )
        if neq_gcmc_system_sampler.tracked_variables["n_explosions"] != 1:
            assert (
                len(neq_gcmc_system_sampler.tracked_variables["insert_works"]) == 1
                or len(neq_gcmc_system_sampler.tracked_variables["delete_works"])
                == 1
            )
        assert (
            0 <= neq_gcmc_system_sampler.tracked_variables["n_explosions"] <= 1
        )

        return None

    def test_insertionMove(self):
        """
        Make sure the NonequilibriumGCMCSystemSampler.insertionMove() method works correctly
        """
        # Prep for a move
        # Read in positions
        neq_gcmc_system_sampler.context = neq_gcmc_system_simulation.context
        state = neq_gcmc_system_sampler.context.getState(
            getPositions=True, enforcePeriodicBox=True, getVelocities=True
        )
        neq_gcmc_system_sampler.positions = deepcopy(
            state.getPositions(asNumpy=True)
        )
        neq_gcmc_system_sampler.velocities = deepcopy(
            state.getVelocities(asNumpy=True)
        )

        # # Set to NCMC integrator
        # neq_gcmc_system_sampler.compound_integrator.setCurrentIntegrator(1)

        # Just run one move to make sure it doesn't crash
        neq_gcmc_system_sampler.insertionMove()

        # # Reset the compound integrator
        # neq_gcmc_sphere_sampler.compound_integrator.setCurrentIntegrator(0)

        return None

    def test_deletionMove(self):
        """
        Make sure the NonequilibriumGCMCSystemSampler.deletionMove() method works correctly
        """
        # Prep for a move
        # Read in positions
        neq_gcmc_system_sampler.context = neq_gcmc_system_simulation.context
        state = neq_gcmc_system_sampler.context.getState(
            getPositions=True, enforcePeriodicBox=True, getVelocities=True
        )
        neq_gcmc_system_sampler.positions = deepcopy(
            state.getPositions(asNumpy=True)
        )
        neq_gcmc_system_sampler.velocities = deepcopy(
            state.getVelocities(asNumpy=True)
        )

        # # Set to NCMC integrator
        # neq_gcmc_system_sampler.compound_integrator.setCurrentIntegrator(1)

        # Just run one move to make sure it doesn't crash
        neq_gcmc_system_sampler.deletionMove()

        # # Reset the compound integrator
        # neq_gcmc_sphere_sampler.compound_integrator.setCurrentIntegrator(0)

        return None


if __name__ == "__main__":
    unittest.main()
