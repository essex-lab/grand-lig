# -*- coding: utf-8 -*-

"""
Description
-----------
This module is written to execute GCMC moves with molecules in OpenMM, via a series of
Sampler objects.

Marley Samways
Ollie Melling
Will Poole
"""
import copy

import numpy as np
import mdtraj
import os
import logging
import parmed
import math
from copy import deepcopy
from simtk import unit
from simtk import openmm
from openmmtools.integrators import NonequilibriumLangevinIntegrator
from rdkit import Chem
from grand import utils
from grand import potential


class BaseGrandCanonicalMonteCarloSampler(object):
    """
    Base class for carrying out GCMC moves in OpenMM.
    All other Sampler objects are derived from this
    """
    def __init__(self, system, topology, temperature, resname="HOH", ghostFile="gcmc-ghost-wats.txt", log='gcmc.log',
                 createCustomForces=True, dcd=None, rst=None, overwrite=False, dihedrals=[], distribution={}, conf=None):
        """
        Initialise the object to be used for sampling insertion/deletion moves

        Parameters
        ----------
        system : simtk.openmm.System
            System object to be used for the simulation
        topology : simtk.openmm.app.Topology
            Topology object for the system to be simulated
        temperature : simtk.unit.Quantity
            Temperature of the simulation, must be in appropriate units
        resname : str
            Resname of the molecule of interest. Default = "HOH"
        ghostFile : str
            Name of a file to write out the residue IDs of ghost molecules. This is
            useful if you want to visualise the sampling, as you can then remove these molecules
            from view, as they are non-interacting. Default is 'gcmc-ghost-wats.txt'
        log : str
            Log file to write out
        createCustomForces : bool
            If True (default), will create CustomForce objects to handle interaction switching. If False, these forces
            must be created elsewhere
        dcd : str
            Name of the DCD file to write the system out to
        rst : str
            Name of the restart file to write out (.pdb or .rst7)
        overwrite : bool
            Overwrite any data already present
        """
        # Create logging object
        if os.path.isfile(log):
            if overwrite:
                os.remove(log)
            else:
                raise Exception("File {} already exists, not overwriting...".format(log))

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        file_handler = logging.FileHandler(log)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s: %(message)s'))
        self.logger.addHandler(file_handler)

        # Set important variables here
        self.system = system
        self.topology = topology
        self.positions = None  # Store no positions upon initialisation
        self.velocities = None
        self.context = None
        self.kT = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA * temperature
        self.simulation_box = np.zeros(3) * unit.nanometer  # Set to zero for now

        self.logger.info("kT = {}".format(self.kT.in_units_of(unit.kilocalorie_per_mole)))

        # Find NonbondedForce - needs to be updated to switch molecules on/off
        for f in range(system.getNumForces()):
            force = system.getForce(f)
            if force.__class__.__name__ == "NonbondedForce":
                self.nonbonded_force = force
            # Flag an error if not simulating at constant volume
            elif "Barostat" in force.__class__.__name__:
                self.raiseError("GCMC must be used at constant volume - {} cannot be used!".format(force.__class__.__name__))

        # Set GCMC-specific variables
        self.N = 0  # Initialise N as zero
        self.Ns = []  # Store all observed values of N
        self.n_moves = 0
        self.n_accepted = 0
        self.acceptance_probabilities = []  # Store acceptance probabilities

        # Get residue IDs & assign statuses to each
        self.mol_resids = self.getMoleculeResids(resname)  # All molecules
        print(self.mol_resids)
        # Assign each water a status: 0: ghost water, 1: GCMC water, 2: Water not under GCMC tracking (out of sphere)
        self.mol_status = {x: 1 for x in self.mol_resids}  # Initially assign all to 1
        print('Testing the new updates babyyyyyyyy!!!!!!!!!!!!!')
        #self.mol_status = np.ones_like(self.mol_resids)  # 1 indicates on, 0 indicates off
        self.gcmc_resids = []  # GCMC molecules

        # Need to customised forces to handle softcore steric interactions and exceptions
        self.mol_params = []  # Stores nonbonded parameters for each atom
        self.custom_nb_force = None
        self.vdw_except_force = None
        self.ele_except_force = None
        self.mol_vdw_excepts = {}  # Store the vdW exception IDs for each molecule
        self.mol_ele_excepts = {}  # Store the electrostatic exception IDs for each molecule

        # Check get atom IDs for each molecule
        self.mol_atom_ids = {}  # Store atom IDs for each molecule
        self.mol_heavy_ids = {}  # Store heavy atom IDs for each molecule
        self.getMoleculeAtoms()

        # Create the custom forces, if requested (default)
        if createCustomForces:
            # Get molecule parameters
            self.mol_params = self.getMoleculeParameters(resname)
            # Create the custom forces
            _, self.custom_nb_force, self.ele_except_force, self.vdw_except_force = utils.create_custom_forces(system,
                                                                                                               topology,
                                                                                                               [resname])
            # Also need to assign exception IDs to each molecule ID
            self.getMoleculeExceptions()
        else:
            self.logger.info("Custom Force objects not created in Sampler __init__() function. These must be set "
                             "using the self.setCustomForces() function!")

        # Need to open the file to store ghost molecule IDs
        self.ghost_file = ghostFile
        # Check whether to overwrite if the file already exists
        if os.path.isfile(self.ghost_file) and not overwrite:
            self.raiseError("File {} already exists, not overwriting...".format(self.ghost_file))
        else:
            with open(self.ghost_file, 'w') as f:
                pass

        # Store reporters for DCD and restart output
        if dcd is not None:
            # Check whether to overwrite
            if os.path.isfile(dcd):
                if overwrite:
                    # Need to remove before overwriting, so there isn't any mix up
                    os.remove(dcd)
                    self.dcd = mdtraj.reporters.DCDReporter(dcd, 0)
                else:
                    self.raiseError("File {} already exists, not overwriting...".format(dcd))
            else:
                self.dcd = mdtraj.reporters.DCDReporter(dcd, 0)
        else:
            self.dcd = None

        if rst is not None:
            # Check whether to overwrite
            if os.path.isfile(rst) and not overwrite:
                self.raiseError("File {} already exists, not overwriting...".format(rst))
            else:
                # Check whether to use PDB or RST7 for the restart file
                rst_ext = os.path.splitext(rst)[1]
                if rst_ext == '.rst7':
                    self.restart = parmed.openmm.reporters.RestartReporter(rst, 0)
                elif rst_ext == '.pdb':
                    self.restart = utils.PDBRestartReporter(rst, self.topology)
                else:
                    self.raiseError("File extension {} not recognised for restart file".format(rst))
        else:
            self.restart = None

        print(f'Dihedrals in the BaseSampler : {dihedrals}')
        self.dihedrals = dihedrals
        # Define dihedral sampling stuff - Maybe expand this across all samplers in future but only need it here for now
        if len(dihedrals) > 0:  # If some dihedrals have actually been defined
            print(self.dihedrals)
            self.dihedral_distribution = distribution
            self.rdkit_conf = conf

        self.logger.info("BaseGrandCanonicalMonteCarloSampler object initialised")

    def setCustomForces(self, param_list, custom_nb_force, elec_bond_force, steric_bond_force):
        """
        Set the custom force objects to forces created elsewhere - if createCustomForces was set to False, this function
        must be run before the Simulation object is created

        Parameters
        ----------
        param_list : list
            List of parameters for each atom (in correct order) - each entry must be a dictionary containing 'charge',
            'sigma', and 'epsilon'
        custom_sterics : simtk.openmm.CustomNonbondedForce
            Handles the softcore LJ interactions
        electrostatc_exceptions : simtk.openmm.CustomBondForce
            Handles the electrostatic exceptions (if relevant, None otherwise)
        steric_exceptions : simtk.openmm.CustomBondForce
            Handles the steric exceptions (if relevant, None otherwise)
        """
        # Set the molecule parameters
        self.mol_params = param_list

        # Check that the Forces haven't already been created - don't want to worry about overwriting for the time being
        if any([force is not None for force in [self.custom_nb_force, self.ele_except_force, self.vdw_except_force]]):
            raise Exception("Error! Custom Force objects have already been assigned!")

        # Set the forces to the appropriate objects
        self.custom_nb_force = custom_nb_force
        self.ele_except_force = elec_bond_force
        self.vdw_except_force = steric_bond_force

        # Read in which Exceptions correspond to which molecules
        self.getMoleculeExceptions()

        # Report to logger that this has been sorted
        self.logger.info("Custom Force objects assigned.")

        return None

    def reset(self):
        """
        Reset counted values (such as number of total or accepted moves) to zero
        """
        self.logger.info('Resetting any tracked variables...')
        self.n_accepted = 0
        self.n_moves = 0
        self.Ns = []
        self.acceptance_probabilities = []
        
        return None

    def getMoleculeParameters(self, resname):
        """
        Get the non-bonded parameters for each of the atoms in the water model used

        Parameters
        ----------
        resname : str
            Name of the molecule residues
    
        Returns
        -------
        wat_params : list
            List of dictionaries containing the charge, sigma and epsilon for each water atom
        """
        mol_params = []  # Store parameters in a list
        for residue in self.topology.residues():
            if residue.name == resname:
                for atom in residue.atoms():
                    # Store the parameters of each atom
                    atom_params = self.nonbonded_force.getParticleParameters(atom.index)
                    mol_params.append({'charge' : atom_params[0],
                                       'sigma' : atom_params[1],
                                       'epsilon' : atom_params[2]})
                break  # Don't need to continue past the first instance
        return mol_params

    def getMoleculeResids(self, resname):
        """
        Get the residue IDs of all molecule molecules in the system

        Parameters
        ----------
        resname : str
            Name of the molecule residues

        Returns
        -------
        resid_list : list
            List of residue ID numbers
        """
        resid_list = []
        for resid, residue in enumerate(self.topology.residues()):
            if residue.name == resname:
                resid_list.append(resid)
        return resid_list

    def getMoleculeExceptions(self):
        """
        Find out the IDs of atoms and exceptions belonging to each molecule
        """
        for resid, residue in enumerate(self.topology.residues()):
            # Only interested in GCMC molecules
            if resid not in self.mol_resids:
                continue

            # Loop over atoms to get the IDs for this molecule
            atom_ids = []
            for atom in residue.atoms():
                atom_ids.append(atom.index)

            print(atom_ids)
            # Loop over vdW exceptions to find those which correspond to this molecule
            vdw_exceptions = []
            if self.vdw_except_force is not None:
                for b in range(self.vdw_except_force.getNumBonds()):
                    # Get the parameters for this 'bond'
                    i, j, [sigma, epsilon, lambda_value] = self.vdw_except_force.getBondParameters(b)

                    # Make sure that we don't have inter-molecular exceptions
                    if (i in atom_ids and j not in atom_ids) or (i not in atom_ids and j in atom_ids):
                        raise Exception("Currently not supporting inter-molecular exceptions")

                    # Check if this corresponds to the molecule
                    if i in atom_ids and j in atom_ids:
                        vdw_exceptions.append(b)

            # Loop over vdW exceptions to find those which correspond to this molecule
            ele_exceptions = []
            if self.ele_except_force is not None:
                for b in range(self.ele_except_force.getNumBonds()):
                    # Get the parameters for this 'bond'
                    i, j, [chargeprod, sigma, lambda_value] = self.ele_except_force.getBondParameters(b)

                    # Make sure that we don't have inter-molecular exceptions
                    if (i in atom_ids and j not in atom_ids) or (i not in atom_ids and j in atom_ids):
                        raise Exception("Currently not supporting inter-molecular exceptions")

                    # Check if this corresponds to the molecule
                    if i in atom_ids and j in atom_ids:
                        ele_exceptions.append(b)

            # Save these IDs
            self.mol_atom_ids[resid] = atom_ids
            self.mol_vdw_excepts[resid] = vdw_exceptions
            self.mol_ele_excepts[resid] = ele_exceptions

        return None

    def setMolStatus(self, resid, new_value):
        """
        Set the status of a particular water to a particular value

        Parameters
        ----------
        resid : int
            Residue to update the status for
        new_value : int
            New value of the water status. 0: ghost, 1: GCMC water, 2: Non-tracked water
        """
        self.mol_status[resid] = new_value
        return None

    def getMolStatusResids(self, value):
        """
        Get a list of resids which have a particular status value

        Parameters
        ----------
        value : int
            Value of the water status. 0: ghost, 1: GCMC water, 2: Non-tracked water

        Returns
        -------
        resids : numpy.array
            List of residues which match that status
        """
        resids = [x[0] for x in self.mol_status.items() if x[1] == value]
        return resids

    def getMolStatusValue(self, resid):
        """
        Get the status value of a particular resid

        Parameters
        ----------
        resid : int
            Residue to update the status for

        Returns
        -------
        value : int
            Value of the water status. 0: ghost, 1: GCMC water, 2: Non-tracked water
        """
        value = self.mol_status[resid]
        return value

    def getMoleculeAtoms(self):
        """
        Get all atom IDs for each molecule, noting heavy atoms in a second list
        """
        # Get the elements for all atoms in the system
        elements = [atom.element.name for atom in self.topology.atoms()]

        # For each residue, get the IDs of all atoms, and also separately those of heavy atoms
        for resid, residue in enumerate(self.topology.residues()):
            all_atoms = []
            heavy_atoms = []

            # Make sure we care about this residue
            if resid not in self.mol_resids:
                continue

            for atom in residue.atoms():
                # Add to the atom list
                all_atoms.append(atom.index)
                # Add to the heavy list, if appropriate
                if elements[atom.index].lower() != 'hydrogen':
                    heavy_atoms.append(atom.index)

            # Update the dictionaries for this residue
            self.mol_atom_ids[resid] = all_atoms
            self.mol_heavy_ids[resid] = heavy_atoms

        return None

    def deleteGhostMolecules(self, ghostResids=None, ghostFile=None):
        """
        Switch off nonbonded interactions involving the ghost molecules initially added
        This function should be executed before beginning the simulation, to prevent any
        explosions.

        Parameters
        ----------
        context : simtk.openmm.Context
            Current context of the simulation
        ghostResids : list
            List of residue IDs corresponding to the ghost molecules added
        ghostFile : str
            File containing residue IDs of ghost molecules. Will switch off those on the
            last line. This will be useful in restarting simulations

        Returns
        -------
        context : simtk.openmm.Context
            Updated context, with ghost molecules switched off
        """
        # Get a list of all ghost residue IDs supplied from list and file
        ghost_resids = []
        # Read in list
        if ghostResids is not None:
            for resid in ghostResids:
                ghost_resids.append(resid)

        # Read residues from file if needed
        if ghostFile is not None:
            with open(ghostFile, 'r') as f:
                lines = f.readlines()
                for resid in lines[-1].split(","):
                    ghost_resids.append(int(resid))

        # Add ghost residues to list of GCMC residues
        for resid in ghost_resids:
            self.gcmc_resids.append(resid)

        # Switch off the interactions involving ghost molecules
        for resid, residue in enumerate(self.topology.residues()):
            if resid in ghost_resids:
                #  Switch off nonbonded interactions involving this molecule
                self.adjustSpecificMolecule(resid, 0.0)
                # Mark that this molecule has been switched off
                self.setMolStatus(resid, 0)

        # Calculate N
        self.N = len(self.getMolStatusResids(1))  # Gets all the resids that are status 1

        return None

    def adjustSpecificMolecule(self, resid, new_lambda, ele=None, vdw=None):
        """
        Adjust the coupling of a specific molecule, by adjusting the lambda value

        Parameters
        ----------
        resid : int
            Resid of the molecule to be adjusted
        new_lambda : float
            Value to set lambda to for this particle
        """
        if ele == None or vdw == None:
            # Get lambda values
            lambda_vdw, lambda_ele = utils.get_lambda_values(new_lambda)
        else:
            lambda_vdw = vdw
            lambda_ele = ele

        #print(lambda_ele)
        # Update per-atom nonbonded parameters first
        atoms = self.mol_atom_ids[resid]
        #print(atoms)
        for i, atom_idx in enumerate(atoms):
            # Obtain original parameters
            atom_params = self.mol_params[i]
            #print(atom_params)
            # Update charge in NonbondedForce
            #self.nonbonded_force.setParticleParameterOffset(atom_idx, 'lambda_ele', atom_idx, atom_params["charge"] * (lambda_ele -1), 0.0, 0.0)
            #print(i, lambda_ele, atom_idx, self.nonbonded_force.getParticleParameterOffset(atom_idx))
            #print(self.nonbonded_force.getParticleParameters(atom_idx))
            self.nonbonded_force.setParticleParameters(atom_idx,
                                                       charge=(lambda_ele * atom_params["charge"]),
                                                       sigma=atom_params["sigma"],
                                                       epsilon=abs(0.0))
            # Update lambda in CustomNonbondedForce
            self.custom_nb_force.setParticleParameters(atom_idx,
                                                       [atom_params["sigma"], atom_params["epsilon"], lambda_vdw])

        # Update context with new parameters
        self.nonbonded_force.updateParametersInContext(self.context)
        self.custom_nb_force.updateParametersInContext(self.context)

        # Update the exceptions, where relevant
        if self.vdw_except_force is not None:
            # Update vdW exceptions
            vdw_exceptions = self.mol_vdw_excepts[resid]
            for exception_id in vdw_exceptions:
                # Get atom IDs and parameters
                i, j, [sigma, epsilon, old_lambda] = self.vdw_except_force.getBondParameters(exception_id)
                # Set the new value of lambda
                self.vdw_except_force.setBondParameters(exception_id, i, j, [sigma, epsilon, lambda_vdw])

            # Update electrostatic exceptions
            ele_exceptions = self.mol_ele_excepts[resid]
            for exception_id in ele_exceptions:
                # Get atom IDs and parameters
                i, j, [chargeprod, sigma, old_lambda] = self.ele_except_force.getBondParameters(exception_id)
                # Set the new value of lambda
                self.ele_except_force.setBondParameters(exception_id, i, j, [chargeprod, sigma, lambda_ele])

            # Update context with new parameters
            self.vdw_except_force.updateParametersInContext(self.context)
            self.ele_except_force.updateParametersInContext(self.context)
        
        return None

    def report(self, simulation):
        """
        Function to report any useful data

        Parameters
        ----------
        simulation : simtk.openmm.app.Simulation
            Simulation object being used
        """
        # Calculate rounded acceptance rate and mean N
        if self.n_moves > 0:
            acc_rate = np.round(self.n_accepted * 100.0 / self.n_moves, 4)
        else:
            acc_rate = np.nan
        mean_N = np.round(np.mean(self.Ns), 4)
        # Print out a line describing the acceptance rate and sampling of N
        msg = "{} move(s) completed ({} accepted ({:.4f} %)). Current N = {}. Average N = {:.3f}".format(self.n_moves,
                                                                                                         self.n_accepted,
                                                                                                         acc_rate,
                                                                                                         self.N,
                                                                                                         mean_N)
        print(msg)
        self.logger.info(msg)

        # Write to the file describing which molecules are ghosts through the trajectory
        self.writeGhostMoleculeResids()

        # Append to the DCD and update the restart file
        state = simulation.context.getState(getPositions=True, getVelocities=True)
        if self.dcd is not None:
            self.dcd.report(simulation, state)
        if self.restart is not None:
            self.restart.report(simulation, state)

        return None

    def raiseError(self, error_msg):
        """
        Make it nice and easy to report an error in a consisent way - also easier to manage error handling in future

        Parameters
        ----------
        error_msg : str
            Message describing the error
        """
        # Write to the log file
        self.logger.error(error_msg)
        # Raise an Exception
        raise Exception(error_msg)
        return None

    def writeGhostMoleculeResids(self):
        """
        Write out a comma-separated list of the residue IDs of molecules which are
        non-interacting, so that they can be removed from visualisations. It is important 
        to execute this function when writing to trajectory files, so that each line
        in the ghost molecule file corresponds to a frame in the trajectory
        """
        # Need to write this function
        with open(self.ghost_file, 'a') as f:
            ghost_resids = self.getMolStatusResids(0)
            if len(ghost_resids) > 0:
                f.write("{}".format(ghost_resids[0]))
                if len(ghost_resids) > 1:
                    for resid in ghost_resids[1:]:
                        f.write(",{}".format(resid))
            f.write("\n")

        return None

    def move(self, context, n=1):
        """
        Returns an error if someone attempts to execute a move with the parent object
        Parameters are designed to match the signature of the inheriting classes

        Parameters
        ----------
        context : simtk.openmm.Context
            Current context of the simulation
        n : int
            Number of moves to execute
        """
        error_msg = ("GrandCanonicalMonteCarloSampler is not designed to sample!")
        self.logger.error(error_msg)
        raise NotImplementedError(error_msg)

    def calculateCOG(self, resid):
        """
        Calculate centre of geometry (COG) of a resid based on heavy atoms

        Parameters
        ----------
        resid : int
            Residue ID of interest

        Returns
        -------
        heavy_cog : simtk.unit.Quantity
            Centre of geometry, based on heavy atoms
        """
        # Calculate centre of geometry (COG), based on heavy atoms
        heavy_atoms = self.mol_heavy_ids[resid]
        heavy_cog = np.zeros(3) * unit.angstroms
        for index in heavy_atoms:
            heavy_cog += self.positions[index]
        heavy_cog /= len(heavy_atoms)
        return heavy_cog

    def randomMolecularRotation(self, resid, new_centre=None):
        """
        Rotate a molecule randomly about it's centre of geometry (based only on heavy atoms)

        Parameters
        ----------
        resid : int
            Residue of interest
        new_centre : simtk.unit.Quantity
            Translate the molecule to this position after the rotation, if given

        Returns
        -------
        new_positions : simtk.unit.Quantity
            New positions for the whole context, including this rotation
        """
        # Calculate centre of geometry (COG), based on heavy atoms
        heavy_cog = self.calculateCOG(resid)

        # If no new position is given, set it to the COG
        if new_centre is None:
            new_centre = heavy_cog

        #  Generate a random rotation matrix
        R = utils.random_rotation_matrix()

        # Scramble the molecular orientation
        new_positions = deepcopy(self.positions)
        for index in self.mol_atom_ids[resid]:
            #  Translate coordinates to an origin defined by the COG, and normalise
            atom_position = self.positions[index] - heavy_cog

            # Rotate about the COG
            vec_length = np.linalg.norm(atom_position.in_units_of(unit.angstroms)) * unit.angstroms
            # If the length of the vector is zero, then we don't need to rotate, as it is sat on the COG
            if vec_length != 0.0 * unit.angstroms:
                atom_position = atom_position / vec_length
                # Rotate coordinates & restore length
                atom_position = vec_length * np.dot(R, atom_position)

            # Translate to new position
            new_positions[index] = atom_position + new_centre

        return new_positions

    def randomiseMoleculeConformer(self, resid):
        dihedral_list = list(self.dihedral_distribution.keys())
        #  Normalise probabilities distribution
        probs = np.asarray(list(self.dihedral_distribution.values())) / 100
        probs = probs / sum(probs)
        rand_conf_id = np.random.choice(np.arange(len(self.dihedral_distribution.keys())),
                                        p=probs)  # Get the position in the conformation keys list of th
        rand_conf = dihedral_list[rand_conf_id]  # Got the actual dihedrals to insert
        print(self.n_moves, rand_conf_id, rand_conf)
        new_positions = copy.deepcopy(self.positions)
        # Now need to set the positions of the RDKit molecule to be the same as the inserted molecule
        for i, index in enumerate(self.mol_atom_ids[resid]):
            self.rdkit_conf.SetAtomPosition(i, self.positions[index]._value * 10)
        for i, dihedral in enumerate(self.dihedrals):   # Set the dihedrals
            Chem.rdMolTransforms.SetDihedralDeg(self.rdkit_conf, *dihedral, float(rand_conf[i]))

        # Get the new positions back out of RDKit
        for i, index in enumerate(self.mol_atom_ids[resid]):
            new_rd_positions = self.rdkit_conf.GetAtomPosition(i)
            new_rd_xyz = np.asarray([float(new_rd_positions.x), float(new_rd_positions.y), float(new_rd_positions.z)]) / 10
            new_positions[index] = new_rd_xyz * unit.nanometers

        return new_positions

    def randomiseAtomVelocities(self, atom_ids):
        """
        Assign random velocities (from the Maxwell-Boltzmann) to a subset of atoms

        Parameters
        ----------
        atom_ids : list
            List of atom indices to assign random velocities to. All other velocities will be unchanged
        """
        # Get temperature from kT
        temperature = self.kT / (unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA)

        # Randomise velocities (using the OpenMM functionality)
        self.context.setVelocitiesToTemperature(temperature)
        random_velocities = self.context.getState(getVelocities=True).getVelocities(asNumpy=True)

        # For each atom of interest, replace the original velocity with the random one
        for idx in atom_ids:
            self.velocities[idx] = random_velocities[idx]

        # Put the correct velocities back into the Context
        self.context.setVelocities(self.velocities)

        return None


########################################################################################################################
########################################################################################################################
########################################################################################################################

class GCMCSphereSampler(BaseGrandCanonicalMonteCarloSampler):
    """
    Base class for carrying out GCMC moves in OpenMM, using a GCMC sphere to sample the system
    """
    def __init__(self, system, topology, temperature, adams=None,
                 excessChemicalPotential=-6.09*unit.kilocalories_per_mole,
                 standardVolume=30.345*unit.angstroms**3, adamsShift=0.0, resname='HOH',
                 ghostFile="gcmc-ghost-wats.txt", referenceAtoms=None, sphereRadius=None, sphereCentre=None,
                 log='gcmc.log', createCustomForces=True, dcd=None, rst=None, overwrite=False,
                 dihedrals=[], distribution={}, conf=None):
        """
        Initialise the object to be used for sampling water insertion/deletion moves

        Parameters
        ----------
        system : simtk.openmm.System
            System object to be used for the simulation
        topology : simtk.openmm.app.Topology
            Topology object for the system to be simulated
        temperature : simtk.unit.Quantity
            Temperature of the simulation, must be in appropriate units
        adams : float
            Adams B value for the simulation (dimensionless). Default is None,
            if None, the B value is calculated from the box volume and chemical
            potential
        excessChemicalPotential : simtk.unit.Quantity
            Excess chemical potential of the system that the simulation should be in equilibrium with, default is
            -6.09 kcal/mol. This should be the hydration free energy of water, and may need to be changed for specific
            simulation parameters.
        standardVolume : simtk.unit.Quantity
            Standard volume of water - corresponds to the volume per water molecule in bulk. The default value is 30.345 A^3
        adamsShift : float
            Shift the B value from Bequil, if B isn't explicitly set. Default is 0.0
        resname : str
            Resname of the molecule of interest. Default = "HOH"
        ghostFile : str
            Name of a file to write out the residue IDs of ghost water molecules. This is
            useful if you want to visualise the sampling, as you can then remove these waters
            from view, as they are non-interacting. Default is 'gcmc-ghost-wats.txt'
        referenceAtoms : list
            List containing dictionaries describing the atoms to use as the centre of the GCMC region
            Must contain 'name' and 'resname' as keys, and optionally 'resid' (recommended) and 'chain'
            e.g. [{'name': 'C1', 'resname': 'LIG', 'resid': '123'}]
        sphereRadius : simtk.unit.Quantity
            Radius of the spherical GCMC region
        sphereCentre : simtk.unit.Quantity
            Coordinates around which the GCMC sphere is based
        log : str
            Log file to write out
        createCustomForces : bool
            If True (default), will create CustomForce objects to handle interaction switching. If False, these forces
            must be created elsewhere
        dcd : str
            Name of the DCD file to write the system out to
        rst : str
            Name of the restart file to write out (.pdb or .rst7)
        overwrite : bool
            Overwrite any data already present
        """
        # Initialise base
        BaseGrandCanonicalMonteCarloSampler.__init__(self, system, topology, temperature, resname=resname, ghostFile=ghostFile,
                                                     log=log, createCustomForces=createCustomForces, dcd=dcd, rst=rst,
                                                     overwrite=overwrite, dihedrals=dihedrals, distribution=distribution, conf=conf)

        # Initialise variables specific to the GCMC sphere
        self.sphere_radius = sphereRadius
        self.sphere_centre = None
        volume = (4 * np.pi * sphereRadius ** 3) / 3

        if referenceAtoms is not None:
            # Define sphere based on reference atoms
            self.ref_atoms = self.getReferenceAtomIndices(referenceAtoms)
            self.logger.info("GCMC sphere is based on reference atom IDs: {}".format(self.ref_atoms))
        elif sphereCentre is not None:
            # Define sphere based on coordinates
            assert len(sphereCentre) == 3, "Sphere coordinates must be 3D"
            self.sphere_centre = sphereCentre
            self.ref_atoms = None
            self.logger.info("GCMC sphere is fixed in space and centred on {}".format(self.sphere_centre))
        else:
            self.raiseError("A set of atoms or coordinates must be used to define the centre of the sphere!")

        self.logger.info("GCMC sphere radius is {}".format(self.sphere_radius))

        # Set or calculate the Adams value for the simulation
        if adams is not None:
            self.B = adams
        else:
            # Calculate Bequil from the chemical potential and volume
            self.B = excessChemicalPotential / self.kT + math.log(volume / standardVolume)
            # Shift B from Bequil if necessary
            self.B += adamsShift

        self.logger.info("Simulating at an Adams (B) value of {}".format(self.B))

        self.logger.info("GCMCSphereSampler object initialised")

    def getReferenceAtomIndices(self, ref_atoms):
        """
        Get the index of the atom used to define the centre of the GCMC box

        Parameters
        ----------
        ref_atoms : list
            List of dictionaries containing the atom name, residue name and (optionally) residue ID and chain,
            as marked by keys 'name', 'resname', 'resid' and 'chain'

        Returns
        -------
        atom_indices : list
            Indices of the atoms chosen
        """
        atom_indices = []
        # Convert to list of lists, if not already
        if not all(type(x) == dict for x in ref_atoms):
            self.raiseError("Reference atoms must be a list of dictionaries! {}".format(ref_atoms))

        # Find atom index for each of the atoms used
        for atom_dict in ref_atoms:
            found = False  # Checks if the atom has been found
            # Read in atom data
            name = atom_dict['name']
            resname = atom_dict['resname']
            # Residue ID and chain may not be present
            try:
                resid = atom_dict['resid']
            except:
                resid = None
            try:
                chain = atom_dict['chain']
            except:
                chain = None

            # Loop over all atoms to find one which matches these criteria
            for c, chain_obj in enumerate(self.topology.chains()):
                # Check chain, if specified
                if chain is not None:
                    if c != chain:
                        continue
                for residue in chain_obj.residues():
                    # Check residue name
                    if residue.name != resname:
                        continue
                    # Check residue ID, if specified
                    if resid is not None:
                        if residue.id != resid:
                            continue
                    # Loop over all atoms in this residue to find the one with the right name
                    for atom in residue.atoms():
                        if atom.name == name:
                            atom_indices.append(atom.index)
                            found = True
            if not found:
                self.raiseError("Atom {} of residue {}{} not found!".format(atom_dict['name'],
                                                                            atom_dict['resname'].capitalize(),
                                                                            atom_dict['resid']))

        if len(atom_indices) == 0:
            self.raiseError("No GCMC reference atoms found")

        return atom_indices

    def getSphereCentre(self):
        """
        Update the coordinates of the sphere centre
        Need to make sure it isn't affected by the reference atoms being split across PBCs
        """
        if self.ref_atoms is None:
            self.raiseError("No reference atoms defined, cannot get sphere coordinates...")

        # Calculate the mean coordinate
        self.sphere_centre = np.zeros(3) * unit.nanometers
        for i, atom in enumerate(self.ref_atoms):
            # Need to add on a correction in case the atoms get separated
            correction = np.zeros(3) * unit.nanometers
            if i != 0:
                # Vector from the first reference atom
                vec = self.positions[self.ref_atoms[0]] - self.positions[atom]
                # Correct for PBCs
                for j in range(3):
                    if vec[j] > 0.5 * self.simulation_box[j]:
                        correction[j] = self.simulation_box[j]
                    elif vec[j] < -0.5 * self.simulation_box[j]:
                        correction[j] = -self.simulation_box[j]

            # Add vector and correction onto the running sum
            self.sphere_centre += self.positions[atom] + correction

        # Calculate the average coordinate
        self.sphere_centre /= len(self.ref_atoms)

        return None

    def initialise(self, context, simulation, ghostResids=[]):
        """
        Prepare the GCMC sphere for simulation by loading the coordinates from a
        Context object.

        Parameters
        ----------
        context : simtk.openmm.Context
            Current context of the simulation
        ghostResids : list
            List of residue IDs corresponding to the ghost waters added
        """

        # Load context into sampler
        self.context = context
        # Load simulation into the sampler
        self.simulation = simulation

        # Load in positions and box vectors from context
        state = self.context.getState(getPositions=True, getVelocities=True, enforcePeriodicBox=True)
        self.positions = deepcopy(state.getPositions(asNumpy=True))
        self.velocities = deepcopy(state.getVelocities(asNumpy=True))
        box_vectors = state.getPeriodicBoxVectors(asNumpy=True)

        # Check the symmetry of the box - currently only tolerate cuboidal boxes
        # All off-diagonal box vector components must be zero
        for i in range(3):
            for j in range(3):
                if i == j:
                    continue
                if not np.isclose(box_vectors[i, j]._value, 0.0):
                    self.raiseError("grand only accepts cuboidal simulation cells at this time.")

        # Get sphere-specific variables
        self.updateGCMCSphere(state)
        # Delete ghost waters
        if len(ghostResids) > 0:
            self.deleteGhostMolecules(ghostResids)

        return None

    def deleteMoleculesInGCMCSphere(self):
        """
        Function to delete all of the molecules currently present in the GCMC region
        This may be useful the plan is to generate a distribution for this
        region from scratch. If so, it would be recommended to interleave the GCMC
        sampling with coordinate propagation, as this will converge faster.

        Parameters
        ----------
        context : simtk.openmm.Context
            Current context of the system. Only needs to be supplied if the context
            has changed since the last update

        Returns
        -------
        context : simtk.openmm.Context
            Updated context after deleting the relevant waters
        """
        #  Read in positions of the context and update GCMC box
        state = self.context.getState(getPositions=True, enforcePeriodicBox=True)
        self.positions = deepcopy(state.getPositions(asNumpy=True))

        # Loop over all residues to find those of interest
        for resid, residue in enumerate(self.topology.residues()):
            if resid not in self.mol_resids:  # Make sure its a molecule of interest
                continue

            # Make sure its a GCMC molecules (ghost/in sphere)
            if self.getMolStatusValue(resid) != 1:
                continue

            self.adjustSpecificMolecule(resid, 0.0)
            # Update relevant parameters
            self.setMolStatus(resid, 0)
            self.N -= 1

        return None

    def updateGCMCSphere(self, state):
        """
        Update the relevant GCMC-sphere related parameters. This also involves monitoring
        which molecules are in/out of the region
        """
        # Get the sphere centre, if using reference atoms, otherwise this will be fine
        if self.ref_atoms is not None:
            self.getSphereCentre()

        box_vectors = state.getPeriodicBoxVectors(asNumpy=True)
        self.simulation_box = np.array([box_vectors[0, 0]._value,
                                        box_vectors[1, 1]._value,
                                        box_vectors[2, 2]._value]) * unit.nanometer

        # Check which molecules are in the GCMC region
        for resid, residue in enumerate(self.topology.residues()):
            # Make sure its a molecule of interesrt
            if resid not in self.mol_resids:
                continue

            # Ghost molecules automatically count as GCMC waters
            if self.getMolStatusValue(resid) == 0:
                continue

            # Check if the molecule is within the sphere
            vector = self.calculateCOG(resid) - self.sphere_centre
            #  Correct PBCs of this vector - need to make this part cleaner
            for i in range(3):
                if vector[i] >= 0.5 * self.simulation_box[i]:
                    vector[i] -= self.simulation_box[i]
                elif vector[i] <= -0.5 * self.simulation_box[i]:
                    vector[i] += self.simulation_box[i]
            # Set molecule status as appropriate
            if np.linalg.norm(vector.in_units_of(unit.angstroms)) * unit.angstrom <= self.sphere_radius:
                self.setMolStatus(resid, 1)  # GCMC to be tracked i.e its on and in sphere
            else:
                self.setMolStatus(resid, 2)  # Not being tracked


        # Update lists
        self.N = len(self.getMolStatusResids(1))

        return None

    def insertRandomMolecule(self):
        """
        Translate a random ghost to a random point in the GCMC sphere to allow subsequent insertion

        Returns
        -------
        new_positions : simtk.unit.Quantity
            Positions following the 'insertion' of the ghost water
        insert_mol : int
            Resid of the molecule to insert
        """
        # Select a ghost molecule to insert
        ghost_mols = self.getMolStatusResids(0)  # Find all mols that are turned off (ghosts)
        # Check that there are any ghosts present
        if len(ghost_mols) == 0:
            self.raiseError("No ghost molecules left, so insertion moves cannot occur - add more ghost molecules")

        insert_mol = np.random.choice(ghost_mols)  # Position in list of GCMC molecules

        # Select a point to insert the water (based on O position)
        rand_nums = np.random.randn(3)
        insert_point = self.sphere_centre + (
                self.sphere_radius * np.power(np.random.rand(), 1.0 / 3) * rand_nums) / np.linalg.norm(rand_nums)

        new_positions = self.randomMolecularRotation(insert_mol, insert_point)

        if len(self.dihedrals) > 0:
            self.positions = new_positions
            new_positions = self.randomiseMoleculeConformer(insert_mol)

        return new_positions, insert_mol

    def deleteRandomMolecule(self):
        """
        Choose a random water to be deleted

        Returns
        -------
        delete_mol : int
            Resid of the molecule to delete
        gcmc_id : int
            GCMC ID for this molecule
        mol_id : int
            Overall ID for this molecule
        """
        # Cannot carry out deletion if there are no GCMC waters on
        gcmc_mols = self.getMolStatusResids(1)  # Get all the 'on' resids
        if len(gcmc_mols) == 0:
            return None

        # Select a water residue to delete
        delete_mol = np.random.choice(gcmc_mols)  # Position in list of GCMC molecules

        atom_indices = []  # Dont think i Need
        for resid, residue in enumerate(self.topology.residues()):
            if resid == delete_mol:
                for atom in residue.atoms():
                    atom_indices.append(atom.index)

        return delete_mol


########################################################################################################################

class StandardGCMCSphereSampler(GCMCSphereSampler):
    """
    Class to carry out instantaneous GCMC moves in OpenMM
    """
    def __init__(self, system, topology, temperature, adams=None, excessChemicalPotential=-6.09*unit.kilocalories_per_mole,
                 standardVolume=30.345*unit.angstroms**3, adamsShift=0.0, resname='HOH', ghostFile="gcmc-ghost-wats.txt",
                 referenceAtoms=None, sphereRadius=None, sphereCentre=None, log='gcmc.log', createCustomForces=True,
                 dcd=None, rst=None, overwrite=False):
        """
        Initialise the object to be used for sampling instantaneous water insertion/deletion moves

        Parameters
        ----------
        system : simtk.openmm.System
            System object to be used for the simulation
        topology : simtk.openmm.app.Topology
            Topology object for the system to be simulated
        temperature : simtk.unit.Quantity
            Temperature of the simulation, must be in appropriate units
        adams : float
            Adams B value for the simulation (dimensionless). Default is None,
            if None, the B value is calculated from the box volume and chemical
            potential
        excessChemicalPotential : simtk.unit.Quantity
            Excess chemical potential of the system that the simulation should be in equilibrium with, default is
            -6.09 kcal/mol. This should be the hydration free energy of water, and may need to be changed for specific
            simulation parameters.
        standardVolume : simtk.unit.Quantity
            Standard volume of water - corresponds to the volume per water molecule in bulk. The default value is 30.345 A^3
        adamsShift : float
            Shift the B value from Bequil, if B isn't explicitly set. Default is 0.0
        ghostFile : str
            Name of a file to write out the residue IDs of ghost water molecules. This is
            useful if you want to visualise the sampling, as you can then remove these waters
            from view, as they are non-interacting. Default is 'gcmc-ghost-wats.txt'
        referenceAtoms : list
            List containing dictionaries describing the atoms to use as the centre of the GCMC region
            Must contain 'name' and 'resname' as keys, and optionally 'resid' (recommended) and 'chain'
            e.g. [{'name': 'C1', 'resname': 'LIG', 'resid': '123'}]
        sphereRadius : simtk.unit.Quantity
            Radius of the spherical GCMC region
        sphereCentre : simtk.unit.Quantity
            Coordinates around which the GCMC sphere is based
        log : str
            Name of the log file to write out
        createCustomForces : bool
            If True (default), will create CustomForce objects to handle interaction switching. If False, these forces
            must be created elsewhere
        dcd : str
            Name of the DCD file to write the system out to
        rst : str
            Name of the restart file to write out (.pdb or .rst7)
        overwrite : bool
            Indicates whether to overwrite already existing data
        """
        # Initialise base class - don't need any more initialisation for the instantaneous sampler
        GCMCSphereSampler.__init__(self, system, topology, temperature, adams=adams,
                                   excessChemicalPotential=excessChemicalPotential, standardVolume=standardVolume,
                                   adamsShift=adamsShift, resname=resname, ghostFile=ghostFile, referenceAtoms=referenceAtoms,
                                   sphereRadius=sphereRadius, sphereCentre=sphereCentre, log=log,
                                   createCustomForces=createCustomForces, dcd=dcd, rst=rst, overwrite=overwrite)

        self.energy = None  # Need to save energy
        self.logger.info("StandardGCMCSphereSampler object initialised")

    def move(self, context, n=1):
        """
        Execute a number of GCMC moves on the current system

        Parameters
        ----------
        context : simtk.openmm.Context
            Current context of the simulation
        n : int
            Number of moves to execute
        """
        # Read in positions
        self.context = context
        state = self.context.getState(getPositions=True, getVelocities=True, enforcePeriodicBox=True, getEnergy=True)
        self.positions = deepcopy(state.getPositions(asNumpy=True))
        self.velocities = deepcopy(state.getVelocities(asNumpy=True))
        self.energy = state.getPotentialEnergy()

        # Update GCMC region based on current state
        self.updateGCMCSphere(state)

        # Check change in N
        if len(self.Ns) > 0:
            dN = self.N - self.Ns[-1]
            if abs(dN) > 0:
                self.logger.info('Change in N of {:+} between GCMC batches'.format(dN))

        # Execute moves
        for i in range(n):
            # Insert or delete a water, based on random choice
            if np.random.randint(2) == 1:
                # Attempt to insert a water
                self.insertionMove()
            else:
                # Attempt to delete a water
                self.deletionMove()
            self.n_moves += 1
            self.Ns.append(self.N)

        return None

    def insertionMove(self):
        """
        Carry out a random water insertion move on the current system
        """
        # Choose a random site in the sphere to insert a molecule
        new_positions, insert_mol = self.insertRandomMolecule()

        # Recouple this water
        self.adjustSpecificMolecule(insert_mol, 1.0)

        self.context.setPositions(new_positions)
        # Calculate new system energy and acceptance probability
        final_energy = self.context.getState(getEnergy=True).getPotentialEnergy()
        acc_prob = math.exp(self.B) * math.exp(-(final_energy - self.energy) / self.kT) / (self.N + 1)
        self.acceptance_probabilities.append(acc_prob)

        if acc_prob < np.random.rand() or np.isnan(acc_prob):
            # Need to revert the changes made if the move is to be rejected
            # Switch off nonbonded interactions involving this molecule
            self.adjustSpecificMolecule(insert_mol, 0.0)
            self.context.setPositions(self.positions)
        else:
            # Update some variables if move is accepted
            self.positions = deepcopy(new_positions)
            self.setMolStatus(insert_mol, 1)
            self.N += 1
            self.n_accepted += 1
            # Update energy
            self.energy = final_energy
            # Assign random velocities to the inserted atoms (not dependent on acceptance, just more efficient this way)
            self.randomiseAtomVelocities(self.mol_atom_ids[insert_mol])

        return None

    def deletionMove(self):
        """
        Carry out a random water deletion move on the current system
        """
        # Choose a random molecule in the sphere to be deleted
        delete_mol = self.deleteRandomMolecule()
        # Deletion may not be possible
        if delete_mol is None:
            return None

        # Switch water off
        self.adjustSpecificMolecule(delete_mol, 0.0)
        # Calculate energy of new state and acceptance probability
        final_energy = self.context.getState(getEnergy=True).getPotentialEnergy()
        acc_prob = self.N * math.exp(-self.B) * math.exp(-(final_energy - self.energy) / self.kT)
        self.acceptance_probabilities.append(acc_prob)

        if acc_prob < np.random.rand() or np.isnan(acc_prob):
            # Switch the molecule back on if the move is rejected
            self.adjustSpecificMolecule(delete_mol, 1.0)
        else:
            # Update some variables if move is accepted
            self.setMolStatus(delete_mol, 0)
            self.N -= 1
            self.n_accepted += 1
            # Update energy
            self.energy = final_energy

        return None


########################################################################################################################

class NonequilibriumGCMCSphereSampler(GCMCSphereSampler):
    """
    Class to carry out GCMC moves in OpenMM, using nonequilibrium candidate Monte Carlo (NCMC)
    to boost acceptance rates
    """
    def __init__(self, system, topology, temperature, integrator, adams=None,
                 excessChemicalPotential=-6.09*unit.kilocalories_per_mole, standardVolume=30.345*unit.angstroms**3,
                 adamsShift=0.0, nPertSteps=1, nPropStepsPerPert=1, timeStep=2 * unit.femtoseconds, lambdas=None,
                 resname='HOH', ghostFile="gcmc-ghost-wats.txt", referenceAtoms=None, sphereRadius=None, sphereCentre=None,
                 log='gcmc.log', createCustomForces=True, dcd=None, rst=None, overwrite=False,
                 dihedrals=[], distribution={}, conf=None, maxN=999, recordTraj=False):
        """
        Initialise the object to be used for sampling NCMC-enhanced water insertion/deletion moves

        Parameters
        ----------
        system : simtk.openmm.System
            System object to be used for the simulation
        topology : simtk.openmm.app.Topology
            Topology object for the system to be simulated
        temperature : simtk.unit.Quantity
            Temperature of the simulation, must be in appropriate units
        integrator : simtk.openmm.CustomIntegrator
            Integrator to use to propagate the dynamics of the system. Currently want to make sure that this
            is the customised Langevin integrator found in openmmtools which uses BAOAB (VRORV) splitting.
        adams : float
            Adams B value for the simulation (dimensionless). Default is None,
            if None, the B value is calculated from the box volume and chemical
            potential
        excessChemicalPotential : simtk.unit.Quantity
            Excess chemical potential of the system that the simulation should be in equilibrium with, default is
            -6.09 kcal/mol. This should be the hydration free energy of water, and may need to be changed for specific
            simulation parameters.
        standardVolume : simtk.unit.Quantity
            Standard volume of water - corresponds to the volume per water molecule in bulk. The default value is 30.345 A^3
        adamsShift : float
            Shift the B value from Bequil, if B isn't explicitly set. Default is 0.0
        nPertSteps : int
            Number of pertubation steps over which to shift lambda between 0 and 1 (or vice versa).
        nPropStepsPerPert : int
            Number of propagation steps to carry out for
        timeStep : simtk.unit.Quantity
            Time step to use for non-equilibrium integration during the propagation steps
        lambdas : list
            Series of lambda values corresponding to the pathway over which the molecules are perturbed
        resname : str
            Resname of the molecule of interest. Default = "HOH"
        ghostFile : str
            Name of a file to write out the residue IDs of ghost water molecules. This is
            useful if you want to visualise the sampling, as you can then remove these waters
            from view, as they are non-interacting. Default is 'gcmc-ghost-wats.txt'
        referenceAtoms : list
            List containing dictionaries describing the atoms to use as the centre of the GCMC region
            Must contain 'name' and 'resname' as keys, and optionally 'resid' (recommended) and 'chain'
            e.g. [{'name': 'C1', 'resname': 'LIG', 'resid': '123'}]
        sphereRadius : simtk.unit.Quantity
            Radius of the spherical GCMC region
        sphereCentre : simtk.unit.Quantity
            Coordinates around which the GCMC sphere is based
        log : str
            Name of the log file to write out
        createCustomForces : bool
            If True (default), will create CustomForce objects to handle interaction switching. If False, these forces
            must be created elsewhere
        dcd : str
            Name of the DCD file to write the system out to
        rst : str
            Name of the restart file to write out (.pdb or .rst7)
        overwrite : bool
            Indicates whether to overwrite already existing data
        dihedrals : list
            List of dihedrals in the insertion molecule
        distribution : dict
            Dictionary of the conformation distribution in the moelcule and their populations
        conf : rdkit thing
            RDKit conformer for use when inserting flexible molecules
        maxN : int
            User can supply the maximum number of N molecules to have in the sphere - this will massively speed up calculations
            to prevent un-needed insertions.
        recordTraj : bool
            User can specifiy if they want the insertion/deletion trajectories to be recorded. Note requires large amount
            of disk space.
        """
        # Initialise base class
        GCMCSphereSampler.__init__(self, system, topology, temperature, adams=adams,
                                   excessChemicalPotential=excessChemicalPotential, standardVolume=standardVolume,
                                   adamsShift=adamsShift, resname=resname, ghostFile=ghostFile, referenceAtoms=referenceAtoms,
                                   sphereRadius=sphereRadius, sphereCentre=sphereCentre, log=log,
                                   createCustomForces=createCustomForces, dcd=dcd, rst=rst, overwrite=overwrite,
                                   dihedrals=dihedrals, distribution=distribution, conf=conf)

        # Load in extra NCMC variables
        if lambdas is not None:
            # Read in set of lambda values, if specified
            assert np.isclose(lambdas[0], 0.0) and np.isclose(lambdas[-1], 1.0), "Lambda series must start at 0 and end at 1"
            self.lambdas = lambdas
            self.n_pert_steps = len(self.lambdas) - 1
        else:
            # Otherwise, assume they are evenly distributed
            self.n_pert_steps = nPertSteps
            self.lambdas = np.linspace(0.0, 1.0, self.n_pert_steps + 1)

        self.maxN = maxN
        self.n_pert_steps = nPertSteps
        self.n_prop_steps_per_pert = nPropStepsPerPert
        self.time_step = timeStep.in_units_of(unit.picosecond)
        self.protocol_time = (self.n_pert_steps + 1) * self.n_prop_steps_per_pert * self.time_step
        self.logger.info("Each NCMC move will be executed over a total of {}".format(self.protocol_time))

        self.insert_works = []  # Store work values of moves
        self.delete_works = []
        self.n_explosions = 0
        self.n_left_sphere = 0  # Number of moves rejected because the water left the sphere
        self.record = recordTraj

        self.integrator = integrator

        self.logger.info("NonequilibriumGCMCSphereSampler object initialised")
        print('Developer MODE!')


    def move(self, context, n=1):
        """
        Carry out a nonequilibrium GCMC move

        Parameters
        ----------
        context : simtk.openmm.Context
            Current context of the simulation
        n : int
            Number of moves to execute
        """
        # Read in positions
        self.context = context
        state = self.context.getState(getPositions=True, enforcePeriodicBox=True, getVelocities=True)
        self.positions = deepcopy(state.getPositions(asNumpy=True))
        self.velocities = deepcopy(state.getVelocities(asNumpy=True))

        # Update GCMC region based on current state
        self.updateGCMCSphere(state)


        #  Execute moves
        for i in range(n):
            if self.record:  # If we want to record traj
                self.moveDCD, self.dcd_name = utils.setupmoveTraj(self.n_moves) # Run the function to setup a move trajectory which is hidden in utils
            # Insert or delete a water, based on random choice
            if np.random.randint(2) == 1:
                # Attempt to insert a water
                self.insertionMove()
            else:
                # Attempt to delete a water
                self.deletionMove()
            self.n_moves += 1
            self.Ns.append(self.N)

        return None

    def insertionMove(self):
        """
        Carry out a nonequilibrium insertion move for a random water molecule
        """
        # Store initial positions
        old_positions = deepcopy(self.positions)

        if self.N >= self.maxN:  # If we know we're at the max for the sphere, dont bother trying to insert!
            self.logger.info('Insertion move not attempted because binding site is full.')
            return None

        # Choose a random site in the sphere to insert a molecule
        new_positions, insert_mol = self.insertRandomMolecule()
        #print(insert_mol)

        #with open(f'insertion_{self.n_moves}.pdb', 'w') as f:
         #   openmm.app.PDBFile.writeFile(self.topology, new_positions, f)

        # Need to update the context positions
        self.context.setPositions(new_positions)
        # Assign random velocities to the inserted atoms
        self.randomiseAtomVelocities(self.mol_atom_ids[insert_mol])
        if self.record:
            current_state = self.simulation.context.getState(enforcePeriodicBox=True, getPositions=True)
            self.moveDCD.report(self.simulation, current_state)
        # Start running perturbation and propagation kernels
        protocol_work = 0.0 * unit.kilocalories_per_mole
        explosion = False
        #self.ncmc_integrator.step(self.n_prop_steps_per_pert)
        self.integrator.step(self.n_prop_steps_per_pert)
        for i in range(self.n_pert_steps):
            state = self.context.getState(getEnergy=True)
            energy_initial = state.getPotentialEnergy()
            # Adjust interactions of this molecule
            self.adjustSpecificMolecule(insert_mol, self.lambdas[i+1])
            state = self.context.getState(getEnergy=True)
            energy_final = state.getPotentialEnergy()
            protocol_work += energy_final - energy_initial
            # Propagate the system
            try:
                #self.ncmc_integrator.step(self.n_prop_steps_per_pert)
                for j in range(self.n_prop_steps_per_pert):
                    self.integrator.step(1)
                    if self.record:
                        if j % 10 == 0:
                            current_state = self.simulation.context.getState(enforcePeriodicBox=True, getPositions=True)
                            self.moveDCD.report(self.simulation, current_state)
            except:
                print("Caught explosion!")
                explosion = True
                self.n_explosions += 1
                break

        # Store the protocol work
        self.logger.info("Insertion work = {}".format(protocol_work))
        self.insert_works.append(protocol_work)

        # Update variables and GCMC sphere
        self.setMolStatus(insert_mol, 1)
        state = self.context.getState(getPositions=True, enforcePeriodicBox=True)
        self.positions = state.getPositions(asNumpy=True)
        self.updateGCMCSphere(state)

        # Check which waters are still in the GCMC sphere
        gcmc_mols_new = self.getMolStatusResids(1)

        # Calculate acceptance probability
        if insert_mol not in gcmc_mols_new:
            # If the inserted water leaves the sphere, the move cannot be reversed and therefore cannot be accepted
            acc_prob = -1
            self.n_left_sphere += 1
            self.logger.info("Move rejected due to molecule leaving the GCMC sphere")
        elif explosion:
            acc_prob = -1
            self.logger.info("Move rejected due to an instability during integration")
        else:
            # Calculate acceptance probability based on protocol work
            acc_prob = math.exp(self.B) * math.exp(-protocol_work/self.kT) / self.N  # Here N is the new value

        self.acceptance_probabilities.append(acc_prob)

        # Update or reset the system, depending on whether the move is accepted or rejected
        if acc_prob < np.random.rand() or np.isnan(acc_prob):
            if self.record:
                os.remove(self.dcd_name)
            # Need to revert the changes made if the move is to be rejected
            self.adjustSpecificMolecule(insert_mol, 0.0)
            self.context.setPositions(old_positions)
            self.context.setVelocities(-self.velocities)  # Reverse velocities on rejection
            self.positions = deepcopy(old_positions)
            self.velocities = -self.velocities
            state = self.context.getState(getPositions=True, enforcePeriodicBox=True)
            self.setMolStatus(insert_mol, 0)  # rejected so its off
            self.updateGCMCSphere(state)
        else:
            # Update some variables if move is accepted
            if self.record:
                os.rename(self.dcd_name, '{}_accepted_insertion.dcd'.format(self.dcd_name))
            self.N = len(gcmc_mols_new)
            self.n_accepted += 1
            state = self.context.getState(getPositions=True, enforcePeriodicBox=True, getVelocities=True)
            self.positions = deepcopy(state.getPositions(asNumpy=True))
            self.velocities = deepcopy(state.getVelocities(asNumpy=True))
            self.updateGCMCSphere(state)

        return None

    def deletionMove(self):
        """
        Carry out a nonequilibrium deletion move for a random water molecule
        """
        # Store initial positions
        old_positions = deepcopy(self.positions)

        # Choose a random water in the sphere to be deleted
        delete_mol = self.deleteRandomMolecule()

        # Deletion may not be possible
        if delete_mol is None:
            return None

        # Start running perturbation and propagation kernels
        protocol_work = 0.0 * unit.kilocalories_per_mole
        explosion = False
        #self.ncmc_integrator.step(self.n_prop_steps_per_pert)
        self.integrator.step(self.n_prop_steps_per_pert)
        for i in range(self.n_pert_steps):
            state = self.context.getState(getEnergy=True)
            energy_initial = state.getPotentialEnergy()
            # Adjust interactions of this water
            self.adjustSpecificMolecule(delete_mol, self.lambdas[-(2+i)])
            state = self.context.getState(getEnergy=True)
            energy_final = state.getPotentialEnergy()
            protocol_work += energy_final - energy_initial
            if self.record:
                current_state = self.simulation.context.getState(enforcePeriodicBox=True, getPositions=True)
                self.moveDCD.report(self.simulation, current_state)
            # Propagate the system
            try:
                #self.ncmc_integrator.step(self.n_prop_steps_per_pert)
                for j in range(self.n_prop_steps_per_pert):
                    self.integrator.step(1)
                    if self.record:
                        if j % 10 == 0:
                            current_state = self.simulation.context.getState(enforcePeriodicBox=True, getPositions=True)
                            self.moveDCD.report(self.simulation, current_state)
            except:
                print("Caught explosion!")
                explosion = True
                self.n_explosions += 1
                break

        # Get the protocol work
        self.logger.info("Deletion work = {}".format(protocol_work))
        self.delete_works.append(protocol_work)

        # Update variables and GCMC sphere
        # Leaving the molecule as 'on' here to check that the deleted water doesn't leave
        state = self.context.getState(getPositions=True, enforcePeriodicBox=True)
        self.positions = state.getPositions(asNumpy=True)
        old_N = self.N
        self.updateGCMCSphere(state)

        # Check which waters are still in the GCMC sphere
        gcmc_mols_new = self.getMolStatusResids(1)

        # Calculate acceptance probability
        if delete_mol not in gcmc_mols_new:
            # If the deleted water leaves the sphere, the move cannot be reversed and therefore cannot be accepted
            acc_prob = 0
            self.n_left_sphere += 1
            self.logger.info("Move rejected due to molecule leaving the GCMC sphere")
        elif explosion:
            acc_prob = 0
            self.logger.info("Move rejected due to an instability during integration")
        else:
            # Calculate acceptance probability based on protocol work
            acc_prob = old_N * math.exp(-self.B) * math.exp(-protocol_work/self.kT)  # N is the old value

        self.acceptance_probabilities.append(acc_prob)

        # Update or reset the system, depending on whether the move is accepted or rejected
        if acc_prob < np.random.rand() or np.isnan(acc_prob):
            if self.record:
                os.remove(self.dcd_name)
            # Need to revert the changes made if the move is to be rejected
            self.adjustSpecificMolecule(delete_mol, 1.0)
            self.context.setPositions(old_positions)
            self.context.setVelocities(-self.velocities)  # Reverse velocities on rejection
            self.positions = deepcopy(old_positions)
            self.velocities = -self.velocities
            state = self.context.getState(getPositions=True, enforcePeriodicBox=True)
            self.updateGCMCSphere(state)
        else:
            # Update some variables if move is accepted
            if self.record:
                os.rename(self.dcd_name, '{}_accepted_deletion.dcd'.format(self.dcd_name))
            self.setMolStatus(delete_mol, 0)
            self.N = len(gcmc_mols_new) - 1  # Accounting for the deleted water
            self.n_accepted += 1
            state = self.context.getState(getPositions=True, enforcePeriodicBox=True, getVelocities=True)
            self.positions = deepcopy(state.getPositions(asNumpy=True))
            self.velocities = deepcopy(state.getVelocities(asNumpy=True))
            self.updateGCMCSphere(state)

        return None

    def reset(self):
        """
        Reset counted values (such as number of total or accepted moves) to zero
        """
        self.logger.info('Resetting any tracked variables...')
        self.n_accepted = 0
        self.n_moves = 0
        self.Ns = []
        self.acceptance_probabilities = []

        # NCMC-specific variables
        self.insert_works = []
        self.delete_works = []
        self.n_explosions = 0
        self.n_left_sphere = 0

        return None


########################################################################################################################
########################################################################################################################
########################################################################################################################

class GCMCSystemSampler(BaseGrandCanonicalMonteCarloSampler):
    """
    Base class for carrying out GCMC moves in OpenMM, sampling the whole system with GCMC
    """
    def __init__(self, system, topology, temperature, resname="HOH", adams=None,
                 excessChemicalPotential=-6.09*unit.kilocalories_per_mole,
                 standardVolume=30.345*unit.angstroms**3, adamsShift=0.0, boxVectors=None,
                 ghostFile="gcmc-ghost-wats.txt", log='gcmc.log', dcd=None, createCustomForces=True, rst=None,
                 overwrite=False, dihedrals=[], distribution={}, conf=None):
        """
        Initialise the object to be used for sampling water insertion/deletion moves

        Parameters
        ----------
        system : simtk.openmm.System
            System object to be used for the simulation
        topology : simtk.openmm.app.Topology
            Topology object for the system to be simulated
        temperature : simtk.unit.Quantity
            Temperature of the simulation, must be in appropriate units
        resname : str
            Resname of the molecule of interest. Default = "HOH"
        adams : float
            Adams B value for the simulation (dimensionless). Default is None,
            if None, the B value is calculated from the box volume and chemical
            potential
        excessChemicalPotential : simtk.unit.Quantity
            Excess chemical potential of the system that the simulation should be in equilibrium with, default is
            -6.09 kcal/mol (water). This should be the hydration free energy of the molecule, and may need to be changed
        standardVolume : simtk.unit.Quantity
            Standard volume of the molecule - corresponds to the volume per molecule in bulk solution. The default value is
            30.345 A^3 (water)
        adamsShift : float
            Shift the B value from Bequil, if B isn't explicitly set. Default is 0.0
        boxVectors : simtk.unit.Quantity
            Box vectors for the simulation cell
        ghostFile : str
            Name of a file to write out the residue IDs of ghost water molecules. This is
            useful if you want to visualise the sampling, as you can then remove these waters
            from view, as they are non-interacting. Default is 'gcmc-ghost-wats.txt'
        log : str
            Log file to write out
        createCustomForces : bool
            If True (default), will create CustomForce objects to handle interaction switching. If False, these forces
            must be created elsewhere
        dcd : str
            Name of the DCD file to write the system out to
        rst : str
            Name of the restart file to write out (.pdb or .rst7)
        overwrite : bool
            Overwrite any data already present
        """
        BaseGrandCanonicalMonteCarloSampler.__init__(self, system, topology, temperature, resname=resname,
                                                     ghostFile=ghostFile, log=log,
                                                     createCustomForces=createCustomForces, dcd=dcd, rst=rst,
                                                     overwrite=overwrite, dihedrals=dihedrals, distribution=distribution, conf=conf)

        # Read in simulation box lengths
        self.simulation_box = np.array([boxVectors[0, 0]._value,
                                        boxVectors[1, 1]._value,
                                        boxVectors[2, 2]._value]) * unit.nanometer
        volume = self.simulation_box[0] * self.simulation_box[1] * self.simulation_box[2]

        # Set or calculate the Adams value for the simulation
        if adams is not None:
            self.B = adams
        else:
            # Calculate Bequil from the chemical potential and volume
            self.B = excessChemicalPotential / self.kT + math.log(volume / standardVolume)
            # Shift B from Bequil if necessary
            self.B += adamsShift

        self.logger.info("Simulating at an Adams (B) value of {}".format(self.B))

        self.logger.info("GCMCSystemSampler object initialised")

    def initialise(self, context, simulation, ghostResids):
        """
        Prepare the GCMC SYSTEM for simulation by loading the coordinates from a
        Context object.

        Parameters
        ----------
        context : simtk.openmm.Context
            Current context of the simulation
        ghostResids : list
            List of residue IDs corresponding to the ghost waters added
        """
        print('Checking that developer mode is on....')
        if len(ghostResids) == 0 or ghostResids is None:
            self.raiseError("No ghost molecules given! Cannot insert molecules without any ghosts!")
        # Load context into sampler
        self.context = context
        self.simulation = simulation

        # Load in positions and box vectors from context
        state = self.context.getState(getPositions=True, getVelocities=True, enforcePeriodicBox=True)
        self.positions = deepcopy(state.getPositions(asNumpy=True))
        self.velocities = deepcopy(state.getVelocities(asNumpy=True))
        box_vectors = state.getPeriodicBoxVectors(asNumpy=True)

        # Check the symmetry of the box - currently only tolerate cuboidal boxes
        # All off-diagonal box vector components must be zero
        for i in range(3):
            for j in range(3):
                if i == j:
                    continue
                if not np.isclose(box_vectors[i, j]._value, 0.0):
                    self.raiseError("grand only accepts cuboidal simulation cells at this time.")

        self.simulation_box = np.array([box_vectors[0, 0]._value,
                                        box_vectors[1, 1]._value,
                                        box_vectors[2, 2]._value]) * unit.nanometer

        # Delete ghost waters
        self.deleteGhostMolecules(ghostResids)

        # Count N
        self.N = len(self.getMolStatusResids(1))

        return None

    def insertRandomMolecule(self):
        """
        Translate a random ghost to a random point in the simulation box to allow subsequent insertion

        Returns
        -------
        new_positions : simtk.unit.Quantity
            Positions following the 'insertion' of the ghost water
        insert_mol : int
            Resid of the molecule to insert
        """
        # Select a ghost water to insert
        ghost_mols = self.getMolStatusResids(0)
        # Check that there are any ghosts present
        if len(ghost_mols) == 0:
            self.raiseError("No ghost molecules left, so insertion moves cannot occur - add more ghosts")

        insert_mol = np.random.choice(ghost_mols)  # Position in list of GCMC waters

        # Select a point to insert the molecule (based on centre of heavy atoms)
        insert_point = np.random.rand(3) * self.simulation_box

        # Randomly rotate the molecule, and shift to the insertion point
        new_positions = self.randomMolecularRotation(insert_mol, insert_point)

        return new_positions, insert_mol

    def deleteRandomMolecule(self):
        """
        Choose a random water to be deleted

        Returns
        -------
        delete_mol : int
            Resid of the molecule to delete
        """
        # Cannot carry out deletion if there are no GCMC molecules on
        gcmc_mols = self.getMolStatusResids(1)  # Get on mols
        if len(gcmc_mols) == 0:
            return None

        # Select a water residue to delete
        delete_mol = np.random.choice(gcmc_mols)  # Position in list of GCMC molecules
        return delete_mol


########################################################################################################################

class StandardGCMCSystemSampler(GCMCSystemSampler):
    """
    Class to carry out instantaneous GCMC moves in OpenMM
    """
    def __init__(self, system, topology, temperature, resname="HOH", adams=None,
                 excessChemicalPotential=-6.09*unit.kilocalories_per_mole,
                 standardVolume=30.345*unit.angstroms**3, adamsShift=0.0, boxVectors=None,
                 ghostFile="gcmc-ghost-wats.txt", log='gcmc.log', createCustomForces=True, dcd=None, rst=None,
                 overwrite=False):
        """
        Initialise the object to be used for sampling instantaneous water insertion/deletion moves

        Parameters
        ----------
        system : simtk.openmm.System
            System object to be used for the simulation
        topology : simtk.openmm.app.Topology
            Topology object for the system to be simulated
        temperature : simtk.unit.Quantity
            Temperature of the simulation, must be in appropriate units
        resname : str
            Resname of the molecule of interest. Default = "HOH"
        adams : float
            Adams B value for the simulation (dimensionless). Default is None,
            if None, the B value is calculated from the box volume and chemical
            potential
        excessChemicalPotential : simtk.unit.Quantity
            Excess chemical potential of the system that the simulation should be in equilibrium with, default is
            -6.09 kcal/mol (water). This should be the hydration free energy of the molecule, and may need to be
            changed
        standardVolume : simtk.unit.Quantity
            Standard volume of the molecule - corresponds to the volume per molecule in bulk solution. The default value is
            30.345 A^3 (water)
        adamsShift : float
            Shift the B value from Bequil, if B isn't explicitly set. Default is 0.0
        boxVectors : simtk.unit.Quantity
            Box vectors for the simulation cell
        ghostFile : str
            Name of a file to write out the residue IDs of ghost water molecules. This is
            useful if you want to visualise the sampling, as you can then remove these waters
            from view, as they are non-interacting. Default is 'gcmc-ghost-wats.txt'
        log : str
            Name of the log file to write out
        createCustomForces : bool
            If True (default), will create CustomForce objects to handle interaction switching. If False, these forces
            must be created elsewhere
        dcd : str
            Name of the DCD file to write the system out to
        rst : str
            Name of the restart file to write out (.pdb or .rst7)
        overwrite : bool
            Indicates whether to overwrite already existing data
        """
        # Initialise base class - don't need any more initialisation for the instantaneous sampler
        GCMCSystemSampler.__init__(self, system, topology, temperature, resname=resname, adams=adams,
                                   excessChemicalPotential=excessChemicalPotential, standardVolume=standardVolume,
                                   adamsShift=adamsShift, boxVectors=boxVectors, ghostFile=ghostFile, log=log,
                                   createCustomForces=createCustomForces, dcd=dcd, rst=rst, overwrite=overwrite)

        self.energy = None  # Need to save energy
        self.logger.info("StandardGCMCSystemSampler object initialised")

    def move(self, context, n=1):
        """
        Execute a number of GCMC moves on the current system

        Parameters
        ----------
        context : simtk.openmm.Context
            Current context of the simulation
        n : int
            Number of moves to execute
        """
        # Read in positions
        self.context = context
        state = self.context.getState(getPositions=True, enforcePeriodicBox=True, getEnergy=True)
        self.positions = deepcopy(state.getPositions(asNumpy=True))
        self.velocities = deepcopy(state.getVelocities(asNumpy=True))
        self.energy = state.getPotentialEnergy()

        # Execute moves
        for i in range(n):
            # Insert or delete a water, based on random choice
            if np.random.randint(2) == 1:
                # Attempt to insert a water
                self.insertionMove()
            else:
                # Attempt to delete a water
                self.deletionMove()
            self.n_moves += 1
            self.Ns.append(self.N)

        return None

    def insertionMove(self):
        """
        Carry out a random water insertion move on the current system
        """
        # Insert a ghost water to a random site
        new_positions, insert_mol = self.insertRandomMolecule()

        # Recouple this water
        self.adjustSpecificMolecule(insert_mol, 1.0)

        self.context.setPositions(new_positions)
        # Calculate new system energy and acceptance probability
        final_energy = self.context.getState(getEnergy=True).getPotentialEnergy()
        acc_prob = math.exp(self.B) * math.exp(-(final_energy - self.energy) / self.kT) / (self.N + 1)
        self.acceptance_probabilities.append(acc_prob)

        if acc_prob < np.random.rand() or np.isnan(acc_prob):
            # Need to revert the changes made if the move is to be rejected
            # Switch off nonbonded interactions involving this water
            self.adjustSpecificMolecule(insert_mol, 0.0)
            self.context.setPositions(self.positions)  # Not sure this is necessary...
        else:
            # Update some variables if move is accepted
            self.positions = deepcopy(new_positions)
            self.setMolStatus(insert_mol, 1)  # Set the mol status to on
            self.N += 1
            self.n_accepted += 1
            # Update energy
            self.energy = final_energy
            # Assign random velocities to the inserted atoms (not dependent on acceptance, just more efficient this way)
            self.randomiseAtomVelocities(self.mol_atom_ids[insert_mol])

        return None

    def deletionMove(self):
        """
        Carry out a random water deletion move on the current system
        """
        # Choose a random water to be deleted
        delete_mol = self.deleteRandomMolecule()
        # Deletion may not be possible
        if delete_mol is None:
            return None

        # Switch water off
        self.adjustSpecificMolecule(delete_mol, 0.0)
        # Calculate energy of new state and acceptance probability
        final_energy = self.context.getState(getEnergy=True).getPotentialEnergy()
        acc_prob = self.N * math.exp(-self.B) * math.exp(-(final_energy - self.energy) / self.kT)
        self.acceptance_probabilities.append(acc_prob)

        if acc_prob < np.random.rand() or np.isnan(acc_prob):
            # Switch the water back on if the move is rejected
            self.adjustSpecificMolecule(delete_mol, 1.0)
        else:
            # Update some variables if move is accepted
            self.setMolStatus(delete_mol, 0)  # Set it to off cause its been deleted
            self.N -= 1
            self.n_accepted += 1
            # Update energy
            self.energy = final_energy

        return None


########################################################################################################################

class NonequilibriumGCMCSystemSampler(GCMCSystemSampler):
    """
    Class to carry out GCMC moves in OpenMM, using nonequilibrium candidate Monte Carlo (NCMC)
    to boost acceptance rates
    """
    def __init__(self, system, topology, temperature, integrator, resname="HOH", adams=None,
                 excessChemicalPotential=-6.09*unit.kilocalories_per_mole, standardVolume=30.345*unit.angstroms**3,
                 adamsShift=0.0, nPertSteps=1, nPropStepsPerPert=1, timeStep=2 * unit.femtoseconds, boxVectors=None,
                 ghostFile="gcmc-ghost-wats.txt", log='gcmc.log', createCustomForces=True, dcd=None, rst=None,
                 overwrite=False, lambdas=None, dihedrals=[], distribution={}, conf=None):
        """
        Initialise the object to be used for sampling NCMC-enhanced water insertion/deletion moves

        Parameters
        ----------
        system : simtk.openmm.System
            System object to be used for the simulation
        topology : simtk.openmm.app.Topology
            Topology object for the system to be simulated
        temperature : simtk.unit.Quantity
            Temperature of the simulation, must be in appropriate units
        integrator : simtk.openmm.CustomIntegrator
            Integrator to use to propagate the dynamics of the system. Currently want to make sure that this
            is the customised Langevin integrator found in openmmtools which uses BAOAB (VRORV) splitting.
        resname : str
            Resname of the molecule of interest. Default = "HOH"
        adams : float
            Adams B value for the simulation (dimensionless). Default is None,
            if None, the B value is calculated from the box volume and chemical
            potential
        excessChemicalPotential : simtk.unit.Quantity
            Excess chemical potential of the system that the simulation should be in equilibrium with, default is
            -6.09 kcal/mol. This should be the hydration free energy of water, and may need to be changed for specific
            simulation parameters.
        standardVolume : simtk.unit.Quantity
            Standard volume of water - corresponds to the volume per water molecule in bulk. The default value is 30.345 A^3
        adamsShift : float
            Shift the B value from Bequil, if B isn't explicitly set. Default is 0.0
        nPertSteps : int
            Number of pertubation steps over which to shift lambda between 0 and 1 (or vice versa).
        nPropStepsPerPert : int
            Number of propagation steps to carry out for
        timeStep : simtk.unit.Quantity
            Time step to use for non-equilibrium integration during the propagation steps
        lambdas : list
            Series of lambda values corresponding to the pathway over which the molecules are perturbed
        boxVectors : simtk.unit.Quantity
            Box vectors for the simulation cell
        ghostFile : str
            Name of a file to write out the residue IDs of ghost water molecules. This is
            useful if you want to visualise the sampling, as you can then remove these waters
            from view, as they are non-interacting. Default is 'gcmc-ghost-wats.txt'
        log : str
            Name of the log file to write out
        createCustomForces : bool
            If True (default), will create CustomForce objects to handle interaction switching. If False, these forces
            must be created elsewhere
        dcd : str
            Name of the DCD file to write the system out to
        rst : str
            Name of the restart file to write out (.pdb or .rst7)
        overwrite : bool
            Indicates whether to overwrite already existing data
        """
        # Initialise base class
        GCMCSystemSampler.__init__(self, system, topology, temperature, resname=resname, adams=adams,
                                   excessChemicalPotential=excessChemicalPotential, standardVolume=standardVolume,
                                   adamsShift=adamsShift, boxVectors=boxVectors, ghostFile=ghostFile, log=log,
                                   createCustomForces=createCustomForces, dcd=dcd, rst=rst, overwrite=overwrite,
                                   dihedrals=dihedrals, distribution=distribution, conf=conf)

        # Load in extra NCMC variables
        if lambdas is not None:
            # Read in set of lambda values, if specified
            assert np.isclose(lambdas[0], 0.0) and np.isclose(lambdas[-1], 1.0), "Lambda series must start at 0 and end at 1"
            self.lambdas = lambdas
            self.n_pert_steps = len(self.lambdas) - 1
        else:
            # Otherwise, assume they are evenly distributed
            self.n_pert_steps = nPertSteps
            self.lambdas = np.linspace(0.0, 1.0, self.n_pert_steps + 1)

        self.n_prop_steps_per_pert = nPropStepsPerPert
        self.time_step = timeStep.in_units_of(unit.picosecond)
        self.protocol_time = (self.n_pert_steps + 1) * self.n_prop_steps_per_pert * self.time_step
        self.logger.info("Each NCMC move will be executed over a total of {}".format(self.protocol_time))

        self.insert_works = []  # Store work values of moves
        self.delete_works = []
        self.n_explosions = 0


        self.integrator = integrator

        self.logger.info("NonequilibriumGCMCSystemSampler object initialised")

    def move(self, context, n=1):
        """
        Carry out a nonequilibrium GCMC move

        Parameters
        ----------
        context : simtk.openmm.Context
            Current context of the simulation
        n : int
            Number of moves to execute
        """
        # Read in positions
        self.context = context
        state = self.context.getState(getPositions=True, enforcePeriodicBox=True, getVelocities=True)
        self.positions = deepcopy(state.getPositions(asNumpy=True))
        self.velocities = deepcopy(state.getVelocities(asNumpy=True))


        #  Execute moves
        for i in range(n):
            # Insert or delete a water, based on random choice
            if np.random.randint(2) == 1:
                # Attempt to insert a water
                self.insertionMove()
            else:
                # Attempt to delete a water
                self.deletionMove()
            self.n_moves += 1
            self.Ns.append(self.N)

        return None

    def insertionMove(self):
        """
        Carry out a nonequilibrium insertion move for a random water molecule
        """
        # Insert a ghost water to a random site
        new_positions, insert_mol = self.insertRandomMolecule()

        # Need to update the context positions
        self.context.setPositions(new_positions)
        # Assign random velocities to the inserted atoms
        self.randomiseAtomVelocities(self.mol_atom_ids[insert_mol])

        # Start running perturbation and propagation kernels
        protocol_work = 0.0 * unit.kilocalories_per_mole
        explosion = False

        self.integrator.step(self.n_prop_steps_per_pert)
        for i in range(self.n_pert_steps):
            state = self.context.getState(getEnergy=True)
            energy_initial = state.getPotentialEnergy()
            # Adjust interactions of this water
            self.adjustSpecificMolecule(insert_mol, self.lambdas[i+1])
            state = self.context.getState(getEnergy=True)
            energy_final = state.getPotentialEnergy()
            protocol_work += energy_final - energy_initial
            # Propagate the system
            try:
                self.integrator.step(self.n_prop_steps_per_pert)
            except:
                print("Caught explosion!")
                explosion = True
                self.n_explosions += 1
                break

        # Get the protocol work
        #self.logger.info("Insertion work = {}".format(protocol_work))
        self.insert_works.append(protocol_work)

        if explosion:
            acc_prob = -1
            self.logger.info("Move rejected due to an instability during integration")
        else:
            # Calculate acceptance probability based on protocol work
            acc_prob = math.exp(self.B) * math.exp(-protocol_work/self.kT) / (self.N + 1)  # Here N is the old value

        self.acceptance_probabilities.append(acc_prob)

        # Update or reset the system, depending on whether the move is accepted or rejected
        if acc_prob < np.random.rand() or np.isnan(acc_prob):
            # Need to revert the changes made if the move is to be rejected
            self.adjustSpecificMolecule(insert_mol, 0.0)
            self.context.setPositions(self.positions)
            self.context.setVelocities(-self.velocities)  # Reverse velocities on rejection
            self.positions = deepcopy(self.positions)
            self.velocities = -self.velocities
        else:
            # Update some variables if move is accepted
            self.N += 1
            self.n_accepted += 1
            state = self.context.getState(getPositions=True, enforcePeriodicBox=True, getVelocities=True)
            self.positions = deepcopy(state.getPositions(asNumpy=True))
            self.velocities = deepcopy(state.getVelocities(asNumpy=True))
            self.setMolStatus(insert_mol, 1)  # Set status to on

        return None

    def deletionMove(self):
        """
        Carry out a nonequilibrium deletion move for a random water molecule
        """
        # Choose a random water to be deleted
        delete_mol = self.deleteRandomMolecule()
        # Deletion may not be possible
        if delete_mol is None:
            return None

        # Start running perturbation and propagation kernels
        protocol_work = 0.0 * unit.kilocalories_per_mole
        explosion = False
        self.integrator.step(self.n_prop_steps_per_pert)
        for i in range(self.n_pert_steps):
            state = self.context.getState(getEnergy=True)
            energy_initial = state.getPotentialEnergy()
            # Adjust interactions of this water
            self.adjustSpecificMolecule(delete_mol, self.lambdas[-(2+i)])
            state = self.context.getState(getEnergy=True)
            energy_final = state.getPotentialEnergy()
            protocol_work += energy_final - energy_initial
            # Propagate the system
            try:
                self.integrator.step(self.n_prop_steps_per_pert)
            except:
                print("Caught explosion!")
                explosion = True
                self.n_explosions += 1
                break

        # Get the protocol work
        #self.logger.info("Deletion work = {}".format(protocol_work))
        self.delete_works.append(protocol_work)

        if explosion:
            acc_prob = 0
            self.logger.info("Move rejected due to an instability during integration")
        else:
            # Calculate acceptance probability based on protocol work
            acc_prob = self.N * math.exp(-self.B) * math.exp(-protocol_work/self.kT)  # N is the old value

        self.acceptance_probabilities.append(acc_prob)

        # Update or reset the system, depending on whether the move is accepted or rejected
        if acc_prob < np.random.rand() or np.isnan(acc_prob):
            # Need to revert the changes made if the move is to be rejected
            self.adjustSpecificMolecule(delete_mol, 1.0)
            self.context.setPositions(self.positions)
            self.context.setVelocities(-self.velocities)  # Reverse velocities on rejection
            self.positions = deepcopy(self.positions)
            self.velocities = -self.velocities
        else:
            # Update some variables if move is accepted
            self.setMolStatus(delete_mol, 0)
            self.N -= 1
            self.n_accepted += 1
            state = self.context.getState(getPositions=True, enforcePeriodicBox=True, getVelocities=True)
            self.positions = deepcopy(state.getPositions(asNumpy=True))
            self.velocities = deepcopy(state.getVelocities(asNumpy=True))

        return None

    def reset(self):
        """
        Reset counted values (such as number of total or accepted moves) to zero
        """
        self.logger.info('Resetting any tracked variables...')
        self.n_accepted = 0
        self.n_moves = 0
        self.Ns = []
        self.acceptance_probabilities = []

        # NCMC-specific variables
        self.insert_works = []
        self.delete_works = []
        self.n_explosions = 0

        return None

########################################################################################################################
########################################################################################################################
########################################################################################################################

class GCMCCylinderSampler(BaseGrandCanonicalMonteCarloSampler):
    """
    Base class for carrying out GCMC moves in OpenMM, using a GCMC Cyclinder to sample the system - ideal for membrane bound
    proteins
    """
    def __init__(self, system, topology, temperature, adams=None,
                 excessChemicalPotential=-6.09*unit.kilocalories_per_mole,
                 standardVolume=30.345*unit.angstroms**3, adamsShift=0.0, resname='HOH',
                 ghostFile="gcmc-ghost-wats.txt", faceCenter=None, faceRadius=None, MaxZCoord=None, cylinderHeight=None,
                 log='gcmc.log', createCustomForces=True, dcd=None, rst=None, overwrite=False,
                 dihedrals=[], distribution={}, conf=None):
        """
        Initialise the object to be used for sampling water insertion/deletion moves

        Parameters
        ----------
        system : simtk.openmm.System
            System object to be used for the simulation
        topology : simtk.openmm.app.Topology
            Topology object for the system to be simulated
        temperature : simtk.unit.Quantity
            Temperature of the simulation, must be in appropriate units
        adams : float
            Adams B value for the simulation (dimensionless). Default is None,
            if None, the B value is calculated from the box volume and chemical
            potential
        excessChemicalPotential : simtk.unit.Quantity
            Excess chemical potential of the system that the simulation should be in equilibrium with, default is
            -6.09 kcal/mol. This should be the hydration free energy of water, and may need to be changed for specific
            simulation parameters.
        standardVolume : simtk.unit.Quantity
            Standard volume of water - corresponds to the volume per water molecule in bulk. The default value is 30.345 A^3
        adamsShift : float
            Shift the B value from Bequil, if B isn't explicitly set. Default is 0.0
        resname : str
            Resname of the molecule of interest. Default = "HOH"
        ghostFile : str
            Name of a file to write out the residue IDs of ghost water molecules. This is
            useful if you want to visualise the sampling, as you can then remove these waters
            from view, as they are non-interacting. Default is 'gcmc-ghost-wats.txt'
        referenceAtoms : list
            List containing dictionaries describing the atoms to use as the centre of the GCMC region
            Must contain 'name' and 'resname' as keys, and optionally 'resid' (recommended) and 'chain'
            e.g. [{'name': 'C1', 'resname': 'LIG', 'resid': '123'}]
        sphereRadius : simtk.unit.Quantity
            Radius of the spherical GCMC region
        sphereCentre : simtk.unit.Quantity
            Coordinates around which the GCMC sphere is based
        log : str
            Log file to write out
        createCustomForces : bool
            If True (default), will create CustomForce objects to handle interaction switching. If False, these forces
            must be created elsewhere
        dcd : str
            Name of the DCD file to write the system out to
        rst : str
            Name of the restart file to write out (.pdb or .rst7)
        overwrite : bool
            Overwrite any data already present
        """
        # Initialise base
        BaseGrandCanonicalMonteCarloSampler.__init__(self, system, topology, temperature, resname=resname, ghostFile=ghostFile,
                                                     log=log, createCustomForces=createCustomForces, dcd=dcd, rst=rst,
                                                     overwrite=overwrite, dihedrals=dihedrals, distribution=distribution, conf=conf)

        # Initialise variables specific to the GCMC sphere
        self.face_center = faceCenter
        self.cylinder_radius = faceRadius
        self.cylinder_height = cylinderHeight
        self.maxZ = MaxZCoord
        self.minZ = self.maxZ - cylinderHeight
        self.cylinder_center = self.getCylinderCentre()
        volume = np.pi * (self.cylinder_radius ** 2) * self.cylinder_height  # pir^2h

        self.logger.info("GCMC cylinder radius is {} with height {} starting at {}".format(self.cylinder_radius,
                                                                                           self.cylinder_height,
                                                                                           self.maxZ))

        # Set or calculate the Adams value for the simulation
        if adams is not None:
            self.B = adams
        else:
            # Calculate Bequil from the chemical potential and volume
            self.B = excessChemicalPotential / self.kT + math.log(volume / standardVolume)
            # Shift B from Bequil if necessary
            self.B += adamsShift

        self.logger.info("Simulating at an Adams (B) value of {}".format(self.B))

        self.logger.info("GCMCCylinderSampler object initialised")

    def getReferenceAtomIndices(self, ref_atoms):
        """
        Get the index of the atom used to define the centre of the GCMC box

        Parameters
        ----------
        ref_atoms : list
            List of dictionaries containing the atom name, residue name and (optionally) residue ID and chain,
            as marked by keys 'name', 'resname', 'resid' and 'chain'

        Returns
        -------
        atom_indices : list
            Indices of the atoms chosen
        """
        atom_indices = []
        # Convert to list of lists, if not already
        if not all(type(x) == dict for x in ref_atoms):
            self.raiseError("Reference atoms must be a list of dictionaries! {}".format(ref_atoms))

        # Find atom index for each of the atoms used
        for atom_dict in ref_atoms:
            found = False  # Checks if the atom has been found
            # Read in atom data
            name = atom_dict['name']
            resname = atom_dict['resname']
            # Residue ID and chain may not be present
            try:
                resid = atom_dict['resid']
            except:
                resid = None
            try:
                chain = atom_dict['chain']
            except:
                chain = None

            # Loop over all atoms to find one which matches these criteria
            for c, chain_obj in enumerate(self.topology.chains()):
                # Check chain, if specified
                if chain is not None:
                    if c != chain:
                        continue
                for residue in chain_obj.residues():
                    # Check residue name
                    if residue.name != resname:
                        continue
                    # Check residue ID, if specified
                    if resid is not None:
                        if residue.id != resid:
                            continue
                    # Loop over all atoms in this residue to find the one with the right name
                    for atom in residue.atoms():
                        if atom.name == name:
                            atom_indices.append(atom.index)
                            found = True
            if not found:
                self.raiseError("Atom {} of residue {}{} not found!".format(atom_dict['name'],
                                                                            atom_dict['resname'].capitalize(),
                                                                            atom_dict['resid']))

        if len(atom_indices) == 0:
            self.raiseError("No GCMC reference atoms found")

        return atom_indices

    def getCylinderCentre(self):
        """
        Update the coordinates of the sphere centre
        Need to make sure it isn't affected by the reference atoms being split across PBCs
        """

        # Calculate the coordinate
        cylinder_centre = np.asarray([self.face_center[0], self.face_center[1], ((self.maxZ - self.minZ)/2)._value]) * unit.angstroms
        print(f'Cylinder Center = {cylinder_centre}')
        return cylinder_centre

    def initialise(self, context, simulation, ghostResids=[]):
        """
        Prepare the GCMC sphere for simulation by loading the coordinates from a
        Context object.

        Parameters
        ----------
        context : simtk.openmm.Context
            Current context of the simulation
        ghostResids : list
            List of residue IDs corresponding to the ghost waters added
        """

        # Load context into sampler
        self.context = context
        # Load simulation into the sampler
        self.simulation = simulation

        # Load in positions and box vectors from context
        state = self.context.getState(getPositions=True, getVelocities=True, enforcePeriodicBox=True)
        self.positions = deepcopy(state.getPositions(asNumpy=True))
        self.velocities = deepcopy(state.getVelocities(asNumpy=True))
        box_vectors = state.getPeriodicBoxVectors(asNumpy=True)

        # Check the symmetry of the box - currently only tolerate cuboidal boxes
        # All off-diagonal box vector components must be zero
        for i in range(3):
            for j in range(3):
                if i == j:
                    continue
                if not np.isclose(box_vectors[i, j]._value, 0.0):
                    self.raiseError("grand only accepts cuboidal simulation cells at this time.")

        # Get sphere-specific variables
        self.updateGCMCCylinder(state)
        # Delete ghost waters
        if len(ghostResids) > 0:
            print('Deleting Ghosts')
            self.deleteGhostMolecules(ghostResids)

        self.updateGCMCCylinder(state)
        return None

    def deleteMoleculesInGCMCCylinder(self):
        """
        Function to delete all of the molecules currently present in the GCMC region
        This may be useful the plan is to generate a distribution for this
        region from scratch. If so, it would be recommended to interleave the GCMC
        sampling with coordinate propagation, as this will converge faster.

        Parameters
        ----------
        context : simtk.openmm.Context
            Current context of the system. Only needs to be supplied if the context
            has changed since the last update

        Returns
        -------
        context : simtk.openmm.Context
            Updated context after deleting the relevant waters
        """
        #  Read in positions of the context and update GCMC box
        state = self.context.getState(getPositions=True, enforcePeriodicBox=True)
        self.positions = deepcopy(state.getPositions(asNumpy=True))

        # Loop over all residues to find those of interest
        for resid, residue in enumerate(self.topology.residues()):
            if resid not in self.mol_resids:  # Make sure its a molecule of interest
                continue

            # Make sure its a GCMC molecules (ghost/in sphere)
            if self.getMolStatusValue(resid) != 1:
                continue

            self.adjustSpecificMolecule(resid, 0.0)
            # Update relevant parameters
            self.setMolStatus(resid, 0)
            self.N -= 1

        return None

    def updateGCMCCylinder(self, state):
        """
        Update the relevant GCMC-sphere related parameters. This also involves monitoring
        which molecules are in/out of the region
        """
        box_vectors = state.getPeriodicBoxVectors(asNumpy=True)
        self.simulation_box = np.array([box_vectors[0, 0]._value,
                                        box_vectors[1, 1]._value,
                                        box_vectors[2, 2]._value]) * unit.nanometer

        # Check which molecules are in the GCMC region
        for resid, residue in enumerate(self.topology.residues()):
            # Make sure its a molecule of interesrt
            if resid not in self.mol_resids:
                continue

            # Ghost molecules automatically count as GCMC waters
            if self.getMolStatusValue(resid) == 0:
                #print(f'Found a Ghost: {resid}')
                continue

            # Check if the molecule is within the sphere
            COG = self.calculateCOG(resid)
            #print(COG, self.cylinder_center)
            xy_vector = COG[:2] - self.cylinder_center[:2] # x, y vector
            z_vector = COG[2] - self.cylinder_center[2]
            #  Correct PBCs of this vector - need to make this part cleaner
            for i in range(2):
                if xy_vector[i] >= 0.5 * self.simulation_box[i]:
                    xy_vector[i] -= self.simulation_box[i]
                elif xy_vector[i] <= -0.5 * self.simulation_box[i]:
                    xy_vector[i] += self.simulation_box[i]

            # Correct Z vector?
            if z_vector >= 0.5 * self.simulation_box[2]:
                z_vector -= self.simulation_box[2]
            elif z_vector <= -0.5 * self.simulation_box[2]:
                z_vector += self.simulation_box[2]

            #print(z_vector)
            # Set molecule status as appropriate
            if np.linalg.norm(xy_vector.in_units_of(unit.angstroms)) * unit.angstrom <= self.cylinder_radius:  # If in the cirlce
                if z_vector <= self.cylinder_height/2:  # If within height
                    print(f'Setting {resid} as on and in cylinder')
                    self.setMolStatus(resid, 1)  # GCMC to be tracked i.e its on and in cylinder
            else:
                self.setMolStatus(resid, 2)  # Not being tracked

        # Update lists
        self.N = len(self.getMolStatusResids(1))

        return None

    def insertRandomMolecule(self):
        """
        Translate a random ghost to a random point in the GCMC sphere to allow subsequent insertion

        Returns
        -------
        new_positions : simtk.unit.Quantity
            Positions following the 'insertion' of the ghost water
        insert_mol : int
            Resid of the molecule to insert
        """
        # Select a ghost molecule to insert
        ghost_mols = self.getMolStatusResids(0)  # Find all mols that are turned off (ghosts)
        # Check that there are any ghosts present
        if len(ghost_mols) == 0:
            self.raiseError("No ghost molecules left, so insertion moves cannot occur - add more ghost molecules")

        insert_mol = np.random.choice(ghost_mols)  # Position in list of GCMC molecules

        # Select a point to insert the water (based on O position)
        xy_rand_nums = np.random.randn(2)
        xy_insert_point = self.cylinder_center[:2] + (
                self.cylinder_radius * np.power(np.random.rand(), 1.0 / 2) * xy_rand_nums) / np.linalg.norm(xy_rand_nums)

        print(self.maxZ)
        z_insert_point = np.random.uniform(self.maxZ._value, self.minZ._value)
        #print(f'z_insert_point : {z_insert_point}')
        insert_point = np.asarray([xy_insert_point[0]._value, xy_insert_point[1]._value, z_insert_point]) * unit.angstroms
        print(insert_point)
        new_positions = self.randomMolecularRotation(insert_mol, insert_point)

        if len(self.dihedrals) > 0:
            self.positions = new_positions
            new_positions = self.randomiseMoleculeConformer(insert_mol)

        return new_positions, insert_mol

    def deleteRandomMolecule(self):
        """
        Choose a random water to be deleted

        Returns
        -------
        delete_mol : int
            Resid of the molecule to delete
        gcmc_id : int
            GCMC ID for this molecule
        mol_id : int
            Overall ID for this molecule
        """
        # Cannot carry out deletion if there are no GCMC waters on
        gcmc_mols = self.getMolStatusResids(1)  # Get all the 'on' resids
        if len(gcmc_mols) == 0:
            return None

        # Select a water residue to delete
        delete_mol = np.random.choice(gcmc_mols)  # Position in list of GCMC molecules

        atom_indices = []  # Dont think i Need
        for resid, residue in enumerate(self.topology.residues()):
            if resid == delete_mol:
                for atom in residue.atoms():
                    atom_indices.append(atom.index)

        return delete_mol


########################################################################################################################

class StandardGCMCCylinderSampler(GCMCCylinderSampler):
    """
    Class to carry out instantaneous GCMC moves in OpenMM
    """
    def __init__(self, system, topology, temperature, adams=None, excessChemicalPotential=-6.09*unit.kilocalories_per_mole,
                 standardVolume=30.345*unit.angstroms**3, adamsShift=0.0, resname='HOH', ghostFile="gcmc-ghost-wats.txt",
                 faceCenter=None, faceRadius=None, MaxZCoord=None, cylinderHeight=None,
                 log='gcmc.log', createCustomForces=True, dcd=None, rst=None, overwrite=False):
        """
        Initialise the object to be used for sampling instantaneous water insertion/deletion moves

        Parameters
        ----------
        system : simtk.openmm.System
            System object to be used for the simulation
        topology : simtk.openmm.app.Topology
            Topology object for the system to be simulated
        temperature : simtk.unit.Quantity
            Temperature of the simulation, must be in appropriate units
        adams : float
            Adams B value for the simulation (dimensionless). Default is None,
            if None, the B value is calculated from the box volume and chemical
            potential
        excessChemicalPotential : simtk.unit.Quantity
            Excess chemical potential of the system that the simulation should be in equilibrium with, default is
            -6.09 kcal/mol. This should be the hydration free energy of water, and may need to be changed for specific
            simulation parameters.
        standardVolume : simtk.unit.Quantity
            Standard volume of water - corresponds to the volume per water molecule in bulk. The default value is 30.345 A^3
        adamsShift : float
            Shift the B value from Bequil, if B isn't explicitly set. Default is 0.0
        ghostFile : str
            Name of a file to write out the residue IDs of ghost water molecules. This is
            useful if you want to visualise the sampling, as you can then remove these waters
            from view, as they are non-interacting. Default is 'gcmc-ghost-wats.txt'
        referenceAtoms : list
            List containing dictionaries describing the atoms to use as the centre of the GCMC region
            Must contain 'name' and 'resname' as keys, and optionally 'resid' (recommended) and 'chain'
            e.g. [{'name': 'C1', 'resname': 'LIG', 'resid': '123'}]
        sphereRadius : simtk.unit.Quantity
            Radius of the spherical GCMC region
        sphereCentre : simtk.unit.Quantity
            Coordinates around which the GCMC sphere is based
        log : str
            Name of the log file to write out
        createCustomForces : bool
            If True (default), will create CustomForce objects to handle interaction switching. If False, these forces
            must be created elsewhere
        dcd : str
            Name of the DCD file to write the system out to
        rst : str
            Name of the restart file to write out (.pdb or .rst7)
        overwrite : bool
            Indicates whether to overwrite already existing data
        """
        # Initialise base class - don't need any more initialisation for the instantaneous sampler
        GCMCCylinderSampler.__init__(self, system, topology, temperature, adams=adams,
                                     excessChemicalPotential=excessChemicalPotential, standardVolume=standardVolume,
                                     adamsShift=adamsShift, resname=resname, ghostFile=ghostFile,
                                     faceCenter=faceCenter, faceRadius=faceRadius, MaxZCoord=MaxZCoord, cylinderHeight=cylinderHeight,
                                     log=log, createCustomForces=createCustomForces, dcd=dcd, rst=rst, overwrite=overwrite)

        self.energy = None  # Need to save energy
        self.logger.info("StandardGCMCCylinderSampler object initialised")

    def move(self, context, n=1):
        """
        Execute a number of GCMC moves on the current system

        Parameters
        ----------
        context : simtk.openmm.Context
            Current context of the simulation
        n : int
            Number of moves to execute
        """
        # Read in positions
        self.context = context
        state = self.context.getState(getPositions=True, getVelocities=True, enforcePeriodicBox=True, getEnergy=True)
        self.positions = deepcopy(state.getPositions(asNumpy=True))
        self.velocities = deepcopy(state.getVelocities(asNumpy=True))
        self.energy = state.getPotentialEnergy()

        # Update GCMC region based on current state
        self.updateGCMCCylinder(state)

        # Check change in N
        if len(self.Ns) > 0:
            dN = self.N - self.Ns[-1]
            if abs(dN) > 0:
                self.logger.info('Change in N of {:+} between GCMC batches'.format(dN))

        # Execute moves
        for i in range(n):
            # Insert or delete a water, based on random choice
            if np.random.randint(2) == 1:
                # Attempt to insert a water
                self.insertionMove()
            else:
                # Attempt to delete a water
                self.deletionMove()
            self.n_moves += 1
            self.Ns.append(self.N)

        return None

    def insertionMove(self):
        """
        Carry out a random water insertion move on the current system
        """
        # Choose a random site in the sphere to insert a molecule
        new_positions, insert_mol = self.insertRandomMolecule()

        # Recouple this water
        self.adjustSpecificMolecule(insert_mol, 1.0)

        self.context.setPositions(new_positions)
        # Calculate new system energy and acceptance probability
        final_energy = self.context.getState(getEnergy=True).getPotentialEnergy()
        acc_prob = math.exp(self.B) * math.exp(-(final_energy - self.energy) / self.kT) / (self.N + 1)
        self.acceptance_probabilities.append(acc_prob)

        if acc_prob < np.random.rand() or np.isnan(acc_prob):
            # Need to revert the changes made if the move is to be rejected
            # Switch off nonbonded interactions involving this molecule
            self.adjustSpecificMolecule(insert_mol, 0.0)
            self.context.setPositions(self.positions)
        else:
            # Update some variables if move is accepted
            self.positions = deepcopy(new_positions)
            self.setMolStatus(insert_mol, 1)
            self.N += 1
            self.n_accepted += 1
            # Update energy
            self.energy = final_energy
            # Assign random velocities to the inserted atoms (not dependent on acceptance, just more efficient this way)
            self.randomiseAtomVelocities(self.mol_atom_ids[insert_mol])

        return None

    def deletionMove(self):
        """
        Carry out a random water deletion move on the current system
        """
        # Choose a random molecule in the sphere to be deleted
        delete_mol = self.deleteRandomMolecule()
        # Deletion may not be possible
        if delete_mol is None:
            return None

        # Switch water off
        self.adjustSpecificMolecule(delete_mol, 0.0)
        # Calculate energy of new state and acceptance probability
        final_energy = self.context.getState(getEnergy=True).getPotentialEnergy()
        acc_prob = self.N * math.exp(-self.B) * math.exp(-(final_energy - self.energy) / self.kT)
        self.acceptance_probabilities.append(acc_prob)

        if acc_prob < np.random.rand() or np.isnan(acc_prob):
            # Switch the molecule back on if the move is rejected
            self.adjustSpecificMolecule(delete_mol, 1.0)
        else:
            # Update some variables if move is accepted
            self.setMolStatus(delete_mol, 0)
            self.N -= 1
            self.n_accepted += 1
            # Update energy
            self.energy = final_energy

        return None


########################################################################################################################

class NonequilibriumGCMCCylinderSampler(GCMCCylinderSampler):
    """
    Class to carry out GCMC moves in OpenMM, using nonequilibrium candidate Monte Carlo (NCMC)
    to boost acceptance rates
    """
    def __init__(self, system, topology, temperature, integrator, adams=None,
                 excessChemicalPotential=-6.09*unit.kilocalories_per_mole, standardVolume=30.345*unit.angstroms**3,
                 adamsShift=0.0, nPertSteps=1, nPropStepsPerPert=1, timeStep=2 * unit.femtoseconds, lambdas=None,
                 resname='HOH', ghostFile="gcmc-ghost-wats.txt", faceCenter=None, faceRadius=None, MaxZCoord=None, cylinderHeight=None,
                 log='gcmc.log', createCustomForces=True, dcd=None, rst=None, overwrite=False,
                 dihedrals=[], distribution={}, conf=None, maxN=999, recordTraj=False):
        """
        Initialise the object to be used for sampling NCMC-enhanced water insertion/deletion moves

        Parameters
        ----------
        system : simtk.openmm.System
            System object to be used for the simulation
        topology : simtk.openmm.app.Topology
            Topology object for the system to be simulated
        temperature : simtk.unit.Quantity
            Temperature of the simulation, must be in appropriate units
        integrator : simtk.openmm.CustomIntegrator
            Integrator to use to propagate the dynamics of the system. Currently want to make sure that this
            is the customised Langevin integrator found in openmmtools which uses BAOAB (VRORV) splitting.
        adams : float
            Adams B value for the simulation (dimensionless). Default is None,
            if None, the B value is calculated from the box volume and chemical
            potential
        excessChemicalPotential : simtk.unit.Quantity
            Excess chemical potential of the system that the simulation should be in equilibrium with, default is
            -6.09 kcal/mol. This should be the hydration free energy of water, and may need to be changed for specific
            simulation parameters.
        standardVolume : simtk.unit.Quantity
            Standard volume of water - corresponds to the volume per water molecule in bulk. The default value is 30.345 A^3
        adamsShift : float
            Shift the B value from Bequil, if B isn't explicitly set. Default is 0.0
        nPertSteps : int
            Number of pertubation steps over which to shift lambda between 0 and 1 (or vice versa).
        nPropStepsPerPert : int
            Number of propagation steps to carry out for
        timeStep : simtk.unit.Quantity
            Time step to use for non-equilibrium integration during the propagation steps
        lambdas : list
            Series of lambda values corresponding to the pathway over which the molecules are perturbed
        resname : str
            Resname of the molecule of interest. Default = "HOH"
        ghostFile : str
            Name of a file to write out the residue IDs of ghost water molecules. This is
            useful if you want to visualise the sampling, as you can then remove these waters
            from view, as they are non-interacting. Default is 'gcmc-ghost-wats.txt'
        referenceAtoms : list
            List containing dictionaries describing the atoms to use as the centre of the GCMC region
            Must contain 'name' and 'resname' as keys, and optionally 'resid' (recommended) and 'chain'
            e.g. [{'name': 'C1', 'resname': 'LIG', 'resid': '123'}]
        sphereRadius : simtk.unit.Quantity
            Radius of the spherical GCMC region
        sphereCentre : simtk.unit.Quantity
            Coordinates around which the GCMC sphere is based
        log : str
            Name of the log file to write out
        createCustomForces : bool
            If True (default), will create CustomForce objects to handle interaction switching. If False, these forces
            must be created elsewhere
        dcd : str
            Name of the DCD file to write the system out to
        rst : str
            Name of the restart file to write out (.pdb or .rst7)
        overwrite : bool
            Indicates whether to overwrite already existing data
        dihedrals : list
            List of dihedrals in the insertion molecule
        distribution : dict
            Dictionary of the conformation distribution in the moelcule and their populations
        conf : rdkit thing
            RDKit conformer for use when inserting flexible molecules
        maxN : int
            User can supply the maximum number of N molecules to have in the sphere - this will massively speed up calculations
            to prevent un-needed insertions.
        recordTraj : bool
            User can specifiy if they want the insertion/deletion trajectories to be recorded. Note requires large amount
            of disk space.
        """
        # Initialise base class
        GCMCCylinderSampler.__init__(self, system, topology, temperature, adams=adams,
                                   excessChemicalPotential=excessChemicalPotential, standardVolume=standardVolume,
                                   adamsShift=adamsShift, resname=resname, ghostFile=ghostFile,
                                   faceCenter=faceCenter, faceRadius=faceRadius, MaxZCoord=MaxZCoord, cylinderHeight=cylinderHeight,
                                   log=log, createCustomForces=createCustomForces, dcd=dcd, rst=rst, overwrite=overwrite,
                                   dihedrals=dihedrals, distribution=distribution, conf=conf)

        # Load in extra NCMC variables
        if lambdas is not None:
            # Read in set of lambda values, if specified
            assert np.isclose(lambdas[0], 0.0) and np.isclose(lambdas[-1], 1.0), "Lambda series must start at 0 and end at 1"
            self.lambdas = lambdas
            self.n_pert_steps = len(self.lambdas) - 1
        else:
            # Otherwise, assume they are evenly distributed
            self.n_pert_steps = nPertSteps
            self.lambdas = np.linspace(0.0, 1.0, self.n_pert_steps + 1)

        self.maxN = maxN
        self.n_pert_steps = nPertSteps
        self.n_prop_steps_per_pert = nPropStepsPerPert
        self.time_step = timeStep.in_units_of(unit.picosecond)
        self.protocol_time = (self.n_pert_steps + 1) * self.n_prop_steps_per_pert * self.time_step
        self.logger.info("Each NCMC move will be executed over a total of {}".format(self.protocol_time))

        self.insert_works = []  # Store work values of moves
        self.delete_works = []
        self.n_explosions = 0
        self.n_left_sphere = 0  # Number of moves rejected because the water left the sphere
        self.record = recordTraj

        self.integrator = integrator

        self.logger.info("NonequilibriumGCMCCylinderSampler object initialised")
        print('Developer MODE!')


    def move(self, context, n=1):
        """
        Carry out a nonequilibrium GCMC move

        Parameters
        ----------
        context : simtk.openmm.Context
            Current context of the simulation
        n : int
            Number of moves to execute
        """
        # Read in positions
        self.context = context
        state = self.context.getState(getPositions=True, enforcePeriodicBox=True, getVelocities=True)
        self.positions = deepcopy(state.getPositions(asNumpy=True))
        self.velocities = deepcopy(state.getVelocities(asNumpy=True))

        # Update GCMC region based on current state
        self.updateGCMCCylinder(state)


        #  Execute moves
        for i in range(n):
            if self.record:  # If we want to record traj
                self.moveDCD, self.dcd_name = utils.setupmoveTraj(self.n_moves) # Run the function to setup a move trajectory which is hidden in utils
            # Insert or delete a water, based on random choice
            if np.random.randint(2) == 1:
                # Attempt to insert a water
                self.insertionMove()
            else:
                # Attempt to delete a water
                self.deletionMove()
            self.n_moves += 1
            self.Ns.append(self.N)

        return None

    def insertionMove(self):
        """
        Carry out a nonequilibrium insertion move for a random water molecule
        """
        # Store initial positions
        old_positions = deepcopy(self.positions)

        if self.N >= self.maxN:  # If we know we're at the max for the sphere, dont bother trying to insert!
            self.logger.info('Insertion move not attempted because binding site is full.')
            return None

        # Choose a random site in the sphere to insert a molecule
        new_positions, insert_mol = self.insertRandomMolecule()
        #print(insert_mol)

        #with open(f'insertion_{self.n_moves}.pdb', 'w') as f:
         #   openmm.app.PDBFile.writeFile(self.topology, new_positions, f)

        # Need to update the context positions
        self.context.setPositions(new_positions)
        # Assign random velocities to the inserted atoms
        self.randomiseAtomVelocities(self.mol_atom_ids[insert_mol])
        if self.record:
            current_state = self.simulation.context.getState(enforcePeriodicBox=True, getPositions=True)
            self.moveDCD.report(self.simulation, current_state)
        # Start running perturbation and propagation kernels
        protocol_work = 0.0 * unit.kilocalories_per_mole
        explosion = False
        #self.ncmc_integrator.step(self.n_prop_steps_per_pert)
        self.integrator.step(self.n_prop_steps_per_pert)
        for i in range(self.n_pert_steps):
            state = self.context.getState(getEnergy=True)
            energy_initial = state.getPotentialEnergy()
            # Adjust interactions of this molecule
            self.adjustSpecificMolecule(insert_mol, self.lambdas[i+1])
            state = self.context.getState(getEnergy=True)
            energy_final = state.getPotentialEnergy()
            protocol_work += energy_final - energy_initial
            # Propagate the system
            try:
                #self.ncmc_integrator.step(self.n_prop_steps_per_pert)
                for j in range(self.n_prop_steps_per_pert):
                    self.integrator.step(1)
                    if self.record:
                        if j % 10 == 0:
                            current_state = self.simulation.context.getState(enforcePeriodicBox=True, getPositions=True)
                            self.moveDCD.report(self.simulation, current_state)
            except:
                print("Caught explosion!")
                explosion = True
                self.n_explosions += 1
                break

        # Store the protocol work
        self.logger.info("Insertion work = {}".format(protocol_work))
        self.insert_works.append(protocol_work)

        # Update variables and GCMC sphere
        self.setMolStatus(insert_mol, 1)
        state = self.context.getState(getPositions=True, enforcePeriodicBox=True)
        self.positions = state.getPositions(asNumpy=True)
        self.updateGCMCCylinder(state)

        # Check which waters are still in the GCMC sphere
        gcmc_mols_new = self.getMolStatusResids(1)

        # Calculate acceptance probability
        if insert_mol not in gcmc_mols_new:
            # If the inserted water leaves the sphere, the move cannot be reversed and therefore cannot be accepted
            acc_prob = -1
            self.n_left_sphere += 1
            self.logger.info("Move rejected due to molecule leaving the GCMC sphere")
        elif explosion:
            acc_prob = -1
            self.logger.info("Move rejected due to an instability during integration")
        else:
            # Calculate acceptance probability based on protocol work
            acc_prob = math.exp(self.B) * math.exp(-protocol_work/self.kT) / self.N  # Here N is the new value

        self.acceptance_probabilities.append(acc_prob)

        # Update or reset the system, depending on whether the move is accepted or rejected
        if acc_prob < np.random.rand() or np.isnan(acc_prob):
            if self.record:
                os.remove(self.dcd_name)
            # Need to revert the changes made if the move is to be rejected
            self.adjustSpecificMolecule(insert_mol, 0.0)
            self.context.setPositions(old_positions)
            self.context.setVelocities(-self.velocities)  # Reverse velocities on rejection
            self.positions = deepcopy(old_positions)
            self.velocities = -self.velocities
            state = self.context.getState(getPositions=True, enforcePeriodicBox=True)
            self.setMolStatus(insert_mol, 0)  # rejected so its off
            self.updateGCMCCylinder(state)
        else:
            # Update some variables if move is accepted
            if self.record:
                os.rename(self.dcd_name, '{}_accepted_insertion.dcd'.format(self.dcd_name))
            self.N = len(gcmc_mols_new)
            self.n_accepted += 1
            state = self.context.getState(getPositions=True, enforcePeriodicBox=True, getVelocities=True)
            self.positions = deepcopy(state.getPositions(asNumpy=True))
            self.velocities = deepcopy(state.getVelocities(asNumpy=True))
            self.updateGCMCCylinder(state)

        return None

    def deletionMove(self):
        """
        Carry out a nonequilibrium deletion move for a random water molecule
        """
        # Store initial positions
        old_positions = deepcopy(self.positions)

        # Choose a random water in the sphere to be deleted
        delete_mol = self.deleteRandomMolecule()

        # Deletion may not be possible
        if delete_mol is None:
            return None

        # Start running perturbation and propagation kernels
        protocol_work = 0.0 * unit.kilocalories_per_mole
        explosion = False
        #self.ncmc_integrator.step(self.n_prop_steps_per_pert)
        self.integrator.step(self.n_prop_steps_per_pert)
        for i in range(self.n_pert_steps):
            state = self.context.getState(getEnergy=True)
            energy_initial = state.getPotentialEnergy()
            # Adjust interactions of this water
            self.adjustSpecificMolecule(delete_mol, self.lambdas[-(2+i)])
            state = self.context.getState(getEnergy=True)
            energy_final = state.getPotentialEnergy()
            protocol_work += energy_final - energy_initial
            if self.record:
                current_state = self.simulation.context.getState(enforcePeriodicBox=True, getPositions=True)
                self.moveDCD.report(self.simulation, current_state)
            # Propagate the system
            try:
                #self.ncmc_integrator.step(self.n_prop_steps_per_pert)
                for j in range(self.n_prop_steps_per_pert):
                    self.integrator.step(1)
                    if self.record:
                        if j % 10 == 0:
                            current_state = self.simulation.context.getState(enforcePeriodicBox=True, getPositions=True)
                            self.moveDCD.report(self.simulation, current_state)
            except:
                print("Caught explosion!")
                explosion = True
                self.n_explosions += 1
                break

        # Get the protocol work
        self.logger.info("Deletion work = {}".format(protocol_work))
        self.delete_works.append(protocol_work)

        # Update variables and GCMC sphere
        # Leaving the molecule as 'on' here to check that the deleted water doesn't leave
        state = self.context.getState(getPositions=True, enforcePeriodicBox=True)
        self.positions = state.getPositions(asNumpy=True)
        old_N = self.N
        self.updateGCMCCylinder(state)

        # Check which waters are still in the GCMC sphere
        gcmc_mols_new = self.getMolStatusResids(1)

        # Calculate acceptance probability
        if delete_mol not in gcmc_mols_new:
            # If the deleted water leaves the sphere, the move cannot be reversed and therefore cannot be accepted
            acc_prob = 0
            self.n_left_sphere += 1
            self.logger.info("Move rejected due to molecule leaving the GCMC sphere")
        elif explosion:
            acc_prob = 0
            self.logger.info("Move rejected due to an instability during integration")
        else:
            # Calculate acceptance probability based on protocol work
            acc_prob = old_N * math.exp(-self.B) * math.exp(-protocol_work/self.kT)  # N is the old value

        self.acceptance_probabilities.append(acc_prob)

        # Update or reset the system, depending on whether the move is accepted or rejected
        if acc_prob < np.random.rand() or np.isnan(acc_prob):
            if self.record:
                os.remove(self.dcd_name)
            # Need to revert the changes made if the move is to be rejected
            self.adjustSpecificMolecule(delete_mol, 1.0)
            self.context.setPositions(old_positions)
            self.context.setVelocities(-self.velocities)  # Reverse velocities on rejection
            self.positions = deepcopy(old_positions)
            self.velocities = -self.velocities
            state = self.context.getState(getPositions=True, enforcePeriodicBox=True)
            self.updateGCMCCylinder(state)
        else:
            # Update some variables if move is accepted
            if self.record:
                os.rename(self.dcd_name, '{}_accepted_deletion.dcd'.format(self.dcd_name))
            self.setMolStatus(delete_mol, 0)
            self.N = len(gcmc_mols_new) - 1  # Accounting for the deleted water
            self.n_accepted += 1
            state = self.context.getState(getPositions=True, enforcePeriodicBox=True, getVelocities=True)
            self.positions = deepcopy(state.getPositions(asNumpy=True))
            self.velocities = deepcopy(state.getVelocities(asNumpy=True))
            self.updateGCMCCylinder(state)

        return None

    def reset(self):
        """
        Reset counted values (such as number of total or accepted moves) to zero
        """
        self.logger.info('Resetting any tracked variables...')
        self.n_accepted = 0
        self.n_moves = 0
        self.Ns = []
        self.acceptance_probabilities = []

        # NCMC-specific variables
        self.insert_works = []
        self.delete_works = []
        self.n_explosions = 0
        self.n_left_sphere = 0

        return None

