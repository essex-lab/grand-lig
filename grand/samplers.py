# -*- coding: utf-8 -*-

"""
samplers.py
Marley Samways

Description
-----------
This code is written to execute GCMC moves with water molecules in OpenMM, in a 
way that can easily be included with other OpenMM simulations or implemented
methods, with minimal extra effort.

References
----------
1 - G. A. Ross, M. S. Bodnarchuk and J. W. Essex, J. Am. Chem. Soc., 2015,
    137, 14930-14943
2 - G. A. Ross, H. E. Bruce Macdonald, C. Cave-Ayland, A. I. Cabedo Martinez
    and J. W. Essex, J. Chem. Theory Comput., 2017, 13, 6373-6381
"""

import numpy as np
import mdtraj
import os
import logging
from copy import deepcopy
from simtk import unit
from simtk import openmm
from parmed.openmm.reporters import RestartReporter
from openmmtools.integrators import NonequilibriumLangevinIntegrator

from grand.utils import random_rotation_matrix
from grand.potential import get_lambda_values


class GrandCanonicalMonteCarloSampler(object):
    """
    Base class for carrying out GCMC moves in OpenMM
    """
    def __init__(self, system, topology, temperature, adams=None, chemicalPotential=-6.3*unit.kilocalories_per_mole,
                 adamsShift=0.0, waterName="HOH", ghostFile="gcmc-ghost-wats.txt", referenceAtoms=None,
                 sphereRadius=None, sphereCentre=None, log='gcmc.log', dcd=None, rst7=None, overwrite=False):
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
        chemicalPotential : simtk.unit.Quantity
            Chemical potential of the simulation, default is -6.3 kcal/mol. This should
            be the hydration free energy of water, and may need to be changed for specific
            simulation parameters.
        adamsShift : float
            Shift the B value from Bequil, if B isn't explicitly set. Default is 0.0
        waterName : str
            Name of the water residues. Default is 'HOH'
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
            Coordinates around which the GCMC sohere is based
        log : str
            Log file to write out
        dcd : str
            Name of the DCD file to write the system out to
        rst7 : str
            Name of the AMBER restart file to write out
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
        self.context = None
        self.kT = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA * temperature
        self.simulation_box = np.zeros(3) * unit.nanometer  # Set to zero for now

        self.logger.info("kT = {}".format(self.kT.in_units_of(unit.kilocalorie_per_mole)))

        # Find NonbondedForce - needs to be updated to switch waters on/off
        for f in range(system.getNumForces()):
            force = system.getForce(f)
            if force.__class__.__name__ == "NonbondedForce":
                self.nonbonded_force = force
            # Flag an error if not simulating at constant volume
            elif "Barostat" in force.__class__.__name__:
                self.logger.error("GCMC must be used at constant volume - {} cannot be used!".format(force.__class__.__name__))
                raise Exception("GCMC must be used at constant volume - {} cannot be used!".format(force.__class__.__name__))
        
        # Calculate GCMC-specific variables
        self.N = 0  # Initialise N as zero
        # Read in sphere-specific parameters
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
            self.logger.error("A set of atoms or coordinates must be used to define the centre of the sphere!")
            raise Exception("A set of atoms or coordinates must be used to define the centre of the sphere!")

        self.logger.info("GCMC sphere radius is {}".format(self.sphere_radius))

        # Set or calculate the Adams value for the simulation
        if adams is not None:
            self.B = adams
        else:
            # Calculate Bequil from the chemical potential and volume
            self.B = chemicalPotential/self.kT + np.log(volume / (30.0 * unit.angstrom ** 3))
            # Shift B from Bequil if necessary
            self.B += adamsShift

        self.logger.info("Simulating at an Adams (B) value of {}".format(self.B))

        # Other variables
        self.n_moves = 0
        self.n_accepted = 0
        self.Ns = []  # Store all observed values of N
        
        # Get parameters for the water model
        self.water_params = self.getWaterParameters(waterName)

        # Get water residue IDs & assign statuses to each
        self.water_resids = self.getWaterResids(waterName)  # All waters
        self.water_status = np.ones_like(self.water_resids)  # 1 indicates on, 0 indicates off
        self.gcmc_resids = []  # GCMC waters
        self.gcmc_status = []  # 1 indicates on, 0 indicates off

        # Need to create a customised force to handle softcore steric interactions of water molecules
        # This should prevent any 0/0 energy evaluations
        self.custom_nb_force = None
        self.customiseForces()

        # Need to open the file to store ghost water IDs
        self.ghost_file = ghostFile
        # Check whether to overwrite if the file already exists
        if os.path.isfile(self.ghost_file) and not overwrite:
            self.logger.error("File {} already exists, not overwriting...".format(self.ghost_file))
            raise Exception("File {} already exists, not overwriting...".format(self.ghost_file))
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
                    self.logger.error("File {} already exists, not overwriting...".format(dcd))
                    raise Exception("File {} already exists, not overwriting...".format(dcd))
            else:
                self.dcd = mdtraj.reporters.DCDReporter(dcd, 0)
        else:
            self.dcd = None

        if rst7 is not None:
            # Check whether to overwrite
            if os.path.isfile(dcd) and not overwrite:
                self.logger.error("File {} already exists, not overwriting...".format(rst7))
                raise Exception("File {} already exists, not overwriting...".format(rst7))
            else:
                self.rst = RestartReporter(rst7, 0)
        else:
            self.rst = None

    def customiseForces(self):
        """
        Create a CustomNonbondedForce to handle water-water interactions and modify the original NonbondedForce
        to ignore water interactions
        """
        #  Need to make sure that the electrostatics are handled using PME (for now)
        if self.nonbonded_force.getNonbondedMethod() != openmm.NonbondedForce.PME:
            raise Exception("Currently only supporting PME for long range electrostatics")

        # Define the energy expression for the softcore sterics
        energy_expression = ("U;"
                             "U = (lambda^soft_a) * 4 * epsilon * x * (x-1.0);"  # Softcore energy
                             "x = (sigma/reff)^6;"  # Define x as sigma/r(effective)
                             "reff = sigma*((soft_alpha*(1.0-lambda)^soft_b + (r/sigma)^soft_c))^(1/soft_c);"  # Effective r
                             # Define combining rules
                             "sigma = 0.5*(sigma1+sigma2); epsilon = sqrt(epsilon1*epsilon2); lambda = lambda1*lambda2")

        # Create a customised sterics force
        custom_sterics = openmm.CustomNonbondedForce(energy_expression)
        # Add necessary particle parameters
        custom_sterics.addPerParticleParameter("sigma")
        custom_sterics.addPerParticleParameter("epsilon")
        custom_sterics.addPerParticleParameter("lambda")
        # Assume that the system is periodic (for now)
        custom_sterics.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)
        # Transfer properties from the original force
        custom_sterics.setUseSwitchingFunction(self.nonbonded_force.getUseSwitchingFunction())
        custom_sterics.setCutoffDistance(self.nonbonded_force.getCutoffDistance())
        custom_sterics.setSwitchingDistance(self.nonbonded_force.getSwitchingDistance())
        custom_sterics.setUseLongRangeCorrection(self.nonbonded_force.getUseDispersionCorrection())
        # Set softcore parameters
        custom_sterics.addGlobalParameter('soft_alpha', 0.5)
        custom_sterics.addGlobalParameter('soft_a', 1)
        custom_sterics.addGlobalParameter('soft_b', 1)
        custom_sterics.addGlobalParameter('soft_c', 6)

        # Get a list of all water and non-water atom IDs
        water_atom_ids = []
        nonwater_atom_ids = []
        for resid, residue in enumerate(self.topology.residues()):
            if resid in self.water_resids:
                for atom in residue.atoms():
                    water_atom_ids.append(atom.index)
            else:
                for atom in residue.atoms():
                    nonwater_atom_ids.append(atom.index)

        # Copy all water-water and water-nonwater steric interactions into the custom force
        for atom_idx in range(self.nonbonded_force.getNumParticles()):
            # Get atom parameters
            [charge, sigma, epsilon] = self.nonbonded_force.getParticleParameters(atom_idx)
            # Make sure that sigma is not equal to zero
            if np.isclose(sigma._value, 0.0):
                sigma = 1.0 * unit.angstrom
            # Add particle to the custom force (with lambda=1 for now)
            custom_sterics.addParticle([sigma, epsilon, 1.0])
            # Disable steric interactions of waters in the original force by setting epsilon=0
            # We keep the charges for PME purposes
            if atom_idx in water_atom_ids:
                self.nonbonded_force.setParticleParameters(atom_idx, charge, sigma, abs(0))

        # Copy over all exceptions into the new force as exclusions
        for exception_idx in range(self.nonbonded_force.getNumExceptions()):
            [i, j, chargeprod, sigma, epsilon] = self.nonbonded_force.getExceptionParameters(exception_idx)
            custom_sterics.addExclusion(i, j)

        # Define interaction groups for the custom force and add to the system
        custom_sterics.addInteractionGroup(water_atom_ids, water_atom_ids)
        custom_sterics.addInteractionGroup(water_atom_ids, nonwater_atom_ids)
        self.system.addForce(custom_sterics)
        self.custom_nb_force = custom_sterics

        return None

    def reset(self):
        """
        Reset counted values (such as number of total or accepted moves) to zero
        """
        self.logger.info('Resetting any tracked variables...')
        self.n_accepted = 0
        self.n_moves = 0
        self.Ns = []
        
        return None

    def getReferenceAtomIndices(self, ref_atoms):
        """
        Get the index of the atom used to define the centre of the GCMC box
        
        Notes
        -----
        Should make this more efficient at some stage.
        
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
            raise Exception("Reference atoms must be a list of dictionaries! {}".format(ref_atoms))

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
            for residue in self.topology.residues():
                # Check residue name
                if residue.name != resname:
                    continue
                # Check residue ID, if specified
                if resid is not None:
                    if residue.id != resid:
                        continue
                # Check chain, if specified
                if chain is not None:
                    if residue.chain != resid:
                        continue
                # Loop over all atoms in this residue to find the one with the right name
                for atom in residue.atoms():
                    if atom.name == name:
                        atom_indices.append(atom.index)
                        found = True
            if not found:
                self.logger.error("Atom {} of residue {}{} not found!".format(atom_dict['name'],
                                                                              atom_dict['resname'].capitalize(),
                                                                              atom_dict['resid']))
                raise Exception("Atom {} of residue {}{} not found!".format(atom_dict['name'],
                                                                            atom_dict['resname'].capitalize(),
                                                                            atom_dict['resid']))

        if len(atom_indices) == 0:
            self.logger.error("No GCMC reference atoms found")
            raise Exception("No GCMC reference atoms found")

        return atom_indices

    def getSphereCentre(self):
        """
        Update the coordinates of the sphere centre
        Need to make sure it isn't affected by the reference atoms being split across PBCs
        """
        if self.ref_atoms is None:
            self.logger.error("No reference atoms defined, cannot get sphere coordinates...")
            raise Exception("No reference atoms defined, cannot get sphere coordinates...")

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

    def prepareGCMCSphere(self, context, ghostResids):
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
        if len(ghostResids) == 0 or ghostResids is None:
            self.logger.error("No ghost waters given! Cannot insert waters without any ghosts!")
            raise Exception("No ghost waters given! Cannot insert waters without any ghosts!")
        # Load context into sampler
        self.context = context

        # Load in positions and box vectors from context
        state = self.context.getState(getPositions=True, enforcePeriodicBox=True)
        self.positions = deepcopy(state.getPositions(asNumpy=True))
        box_vectors = state.getPeriodicBoxVectors(asNumpy=True)
        self.simulation_box = np.array([box_vectors[0,0]._value,
                                        box_vectors[1,1]._value,
                                        box_vectors[2,2]._value]) * unit.nanometer

        # Calculate the centre of the GCMC sphere, if using reference atoms
        if self.ref_atoms is not None:
            self.getSphereCentre()

        # Loop over waters and check which are in/out of the GCMC sphere at the beginning - may be able to replace this with updateGCMCSphere?
        for resid, residue in enumerate(self.topology.residues()):
            if resid not in self.water_resids:
                continue
            for atom in residue.atoms():
                ox_index = atom.index
                break

            vector = self.positions[ox_index]-self.sphere_centre
            # Correct PBCs of this vector - need to make this part cleaner
            for i in range(3):
                if vector[i] >= 0.5 * self.simulation_box[i]:
                    vector[i] -= self.simulation_box[i]
                elif vector[i] <= -0.5 * self.simulation_box[i]:
                    vector[i] += self.simulation_box[i]
            if np.linalg.norm(vector)*unit.nanometer <= self.sphere_radius:
                self.gcmc_resids.append(resid)  # Add to list of GCMC waters

        # Delete ghost waters
        self.deleteGhostWaters(ghostResids)
        return None

    def getWaterParameters(self, water_resname="HOH"):
        """
        Get the non-bonded parameters for each of the atoms in the water model used

        Parameters
        ----------
        water_resname : str
            Name of the water residues
    
        Returns
        -------
        wat_params : list
            List of dictionaries containing the charge, sigma and epsilon for each water atom
        """
        wat_params = []  # Store parameters in a list
        for residue in self.topology.residues():
            if residue.name == water_resname:
                for atom in residue.atoms():
                    # Store the parameters of each atom
                    atom_params = self.nonbonded_force.getParticleParameters(atom.index)
                    wat_params.append({'charge' : atom_params[0],
                                       'sigma' : atom_params[1],
                                       'epsilon' : atom_params[2]})
                break  # Don't need to continue past the first instance
        return wat_params

    def getWaterResids(self, water_resname="HOH"):
        """
        Get the residue IDs of all water molecules in the system

        Parameters
        ----------
        water_resname : str
            Name of the water residues

        Returns
        -------
        resid_list : list
            List of residue ID numbers
        """
        resid_list = []
        for resid, residue in enumerate(self.topology.residues()):
            if residue.name == water_resname:
                resid_list.append(resid)
        return resid_list

    def deleteGhostWaters(self, ghostResids=None, ghostFile=None):
        """
        Switch off nonbonded interactions involving the ghost molecules initially added
        This function should be executed before beginning the simulation, to prevent any
        explosions.

        Parameters
        ----------
        context : simtk.openmm.Context
            Current context of the simulation
        ghostResids : list
            List of residue IDs corresponding to the ghost waters added
        ghostFile : str
            File containing residue IDs of ghost waters. Will switch off those on the
            last line. This will be useful in restarting simulations

        Returns
        -------
        context : simtk.openmm.Context
            Updated context, with ghost waters switched off
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
        self.gcmc_status = np.ones_like(self.gcmc_resids, dtype=np.int_)  # Store status of each GCMC water

        # Switch off the interactions involving ghost waters
        for resid, residue in enumerate(self.topology.residues()):
            if resid in ghost_resids:
                #  Switch off nonbonded interactions involving this water
                atom_ids = []
                for i, atom in enumerate(residue.atoms()):
                    atom_ids.append(atom.index)
                self.adjustSpecificWater(atom_ids, 0.0)
                # Mark that this water has been switched off
                gcmc_id = np.where(np.array(self.gcmc_resids) == resid)[0]
                wat_id = np.where(np.array(self.water_resids) == resid)[0]
                self.gcmc_status[gcmc_id] = 0
                self.water_status[wat_id] = 0

        # Calculate N
        self.N = np.sum(self.gcmc_status)

        return None

    def deleteWatersInGCMCSphere(self, context=None):
        """
        Function to delete all of the waters currently present in the GCMC region
        This may be useful the plan is to generate a water distribution for this
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
        # Read in positions of the context and update GCMC box
        state = self.context.getState(getPositions=True, enforcePeriodicBox=True)
        self.positions = deepcopy(state.getPositions(asNumpy=True))
        # Loop over all residues to find those of interest
        for resid, residue in enumerate(self.topology.residues()):
            if resid not in self.gcmc_resids:
                continue  # Only concerned with GCMC waters
            gcmc_id = np.where(np.array(self.gcmc_resids) == resid)[0][0]   # Position in list of GCMC waters
            wat_id = np.where(np.array(self.water_resids) == resid)[0][0]  # Position in list of all waters
            if self.gcmc_status[gcmc_id] == 1:
                atom_ids = []
                for atom in residue.atoms():
                    # Switch off interactions involving the atoms of this residue
                    atom_ids.append(atom.index)
                self.adjustSpecificWater(atom_ids, 0.0)
                # Update relevant parameters
                self.gcmc_status[gcmc_id] = 0
                self.water_status[wat_id] = 0
                self.N -= 1

        return None

    def updateGCMCSphere(self, state):
        """
        Update the relevant GCMC-sphere related parameters. This also involves monitoring
        which water molecules are in/out of the region
        """
        # Get the sphere centre, if using reference atoms, otherwise this will be fine
        if self.ref_atoms is not None:
            #self.sphere_centre = np.zeros(3) * unit.nanometers
            #for atom in self.ref_atoms:
            #    self.sphere_centre += self.positions[atom]
            #self.sphere_centre /= len(self.ref_atoms)
            self.getSphereCentre()

        # Update gcmc_resids and gcmc_status
        gcmc_resids = []
        gcmc_status = []

        box_vectors = state.getPeriodicBoxVectors(asNumpy=True)
        self.simulation_box = np.array([box_vectors[0, 0]._value,
                                        box_vectors[1, 1]._value,
                                        box_vectors[2, 2]._value]) * unit.nanometer

        # Check which waters are in the GCMC region
        for resid, residue in enumerate(self.topology.residues()):
            if resid not in self.water_resids:
                continue
            for atom in residue.atoms():
                ox_index = atom.index
                break
            wat_id = np.where(np.array(self.water_resids) == resid)[0][0]

            # Ghost waters automatically count as GCMC waters
            if self.water_status[wat_id] == 0:
                gcmc_resids.append(resid)
                gcmc_status.append(0)
                continue

            # Check if the water is within the sphere
            vector = self.positions[ox_index] - self.sphere_centre
            #  Correct PBCs of this vector - need to make this part cleaner
            for i in range(3):
                if vector[i] >= 0.5 * self.simulation_box[i]:
                    vector[i] -= self.simulation_box[i]
                elif vector[i] <= -0.5 * self.simulation_box[i]:
                    vector[i] += self.simulation_box[i]
            # Update lists if this water is in the sphere
            if np.linalg.norm(vector)*unit.nanometer <= self.sphere_radius:
                gcmc_resids.append(resid)  # Add to list of GCMC waters
                gcmc_status.append(self.water_status[wat_id])

        # Update lists
        self.gcmc_resids = deepcopy(gcmc_resids)
        self.gcmc_status = np.array(gcmc_status)
        self.N = np.sum(self.gcmc_status)

        return None

    def adjustSpecificWater(self, atoms, new_lambda):
        """
        Adjust the coupling of a specific water molecule, by adjusting the lambda value

        Parameters
        ----------
        atoms : list
            List of the atom indices of the water to be adjusted
        new_lambda : float
            Value to set lambda to for this particle
        """
        # Get lambda values
        lambda_vdw, lambda_ele = get_lambda_values(new_lambda)

        # Loop over parameters
        for i, atom_idx in enumerate(atoms):
            # Obtain original parameters
            atom_params = self.water_params[i]
            # Update charge in NonbondedForce
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
            acc_rate = np.round(self.n_accepted * 100.0 / self.n_moves, 3)
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

        # Write to the file describing which waters are ghosts through the trajectory
        self.writeGhostWaterResids()

        # Append to the DCD and update the restart file
        state = simulation.context.getState(getPositions=True, getVelocities=True)
        if self.dcd is not None:
            self.dcd.report(simulation, state)
        if self.rst is not None:
            self.rst.report(simulation, state)

        return None

    def writeGhostWaterResids(self):
        """
        Write out a comma-separated list of the residue IDs of waters which are
        non-interacting, so that they can be removed from visualisations. It is important 
        to execute this function when writing to trajectory files, so that each line
        in the ghost water file corresponds to a frame in the trajectory
        """
        # Need to write this function
        with open(self.ghost_file, 'a') as f:
            gcmc_ids = np.where(self.gcmc_status == 0)[0]
            ghost_resids = [self.gcmc_resids[id] for id in gcmc_ids]
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
        error_msg = ("GrandCanonicalMonteCarloSampler is not designed to sample! Use StandardGCMCSampler or "
                     "NonequilibriumGCMCSampler")
        self.logger.error(error_msg)
        raise NotImplementedError(error_msg)


class StandardGCMCSampler(GrandCanonicalMonteCarloSampler):
    """
    Class to carry out instantaneous GCMC moves in OpenMM
    """
    def __init__(self, system, topology, temperature, adams=None, chemicalPotential=-6.3*unit.kilocalories_per_mole,
                 adamsShift=0.0, waterName="HOH", ghostFile="gcmc-ghost-wats.txt", referenceAtoms=None,
                 sphereRadius=None, sphereCentre=None, log='gcmc.log', dcd=None, rst7=None, overwrite=False):
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
        chemicalPotential : simtk.unit.Quantity
            Chemical potential of the simulation, default is -6.3 kcal/mol. This should
            be the hydration free energy of water, and may need to be changed for specific
            simulation parameters.
        adamsShift : float
            Shift the B value from Bequil, if B isn't explicitly set. Default is 0.0
        waterName : str
            Name of the water residues. Default is 'HOH'
        waterModel : str
            Name of the water model being used. This is used to identify the equilibrium chemical
            potential to use. Default: TIP3P. Accepted: SPCE, TIP3P, TIP4Pew
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
            Coordinates around which the GCMC sohere is based
        log : str
            Name of the log file to write out
        dcd : str
            Name of the DCD file to write the system out to
        rst7 : str
            Name of the AMBER restart file to write out
        overwrite : bool
            Indicates whether to overwrite already existing data
        """
        # Initialise base class - don't need any more initialisation for the instantaneous sampler
        GrandCanonicalMonteCarloSampler.__init__(self, system, topology, temperature, adams=adams,
                                                 chemicalPotential=chemicalPotential, adamsShift=adamsShift,
                                                 waterName=waterName, ghostFile=ghostFile,
                                                 referenceAtoms=referenceAtoms, sphereRadius=sphereRadius,
                                                 sphereCentre=sphereCentre, log=log, dcd=dcd, rst7=rst7,
                                                 overwrite=overwrite)

        self.energy = None  # Need to save energy
        self.acceptance_probabilities = []  # Store acceptance probabilities
        self.logger.info("StandardGCMCSampler object initialised")

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
        self.logger.info("Starting a batch of {} GCMC moves".format(n))
        old_accepted = self.n_accepted

        # Read in positions
        self.context = context
        state = self.context.getState(getPositions=True, enforcePeriodicBox=True, getEnergy=True)
        self.positions = deepcopy(state.getPositions(asNumpy=True))
        self.energy = state.getPotentialEnergy()

        # Update GCMC region based on current state
        self.updateGCMCSphere(state)

        #  Execute moves
        for i in range(n):
            # Insert or delete a water, based on random choice
            if np.random.randint(2) == 1:
                # Attempt to insert a water
                self.insertRandomWater()
            else:
                # Attempt to delete a water
                self.deleteRandomWater()
            self.n_moves += 1
            self.Ns.append(self.N)

        self.logger.info("{}/{} moves accepted from this batch".format(self.n_accepted-old_accepted, n))

        return None

    def insertRandomWater(self):
        """
        Carry out a random water insertion move on the current system

        Notes
        -----
        Need to double-check (and fix) any issues relating to periodic boundaries
        and the inserted coordinates
        """
        # Select a ghost water to insert
        ghost_wats = np.where(self.gcmc_status == 0)[0]
        # Check that there are any ghosts present
        if len(ghost_wats) == 0:
            self.logger.error("No ghost water molecules left, so insertion moves cannot occur - add more ghost waters")
            raise Exception("No ghost water molecules left, so insertion moves cannot occur - add more ghost waters")

        gcmc_id = np.random.choice(ghost_wats)  # Position in list of GCMC waters
        insert_water = self.gcmc_resids[gcmc_id]
        wat_id = np.where(np.array(self.water_resids) == insert_water)[0][0]  # Position in list of all waters
        atom_indices = []
        for resid, residue in enumerate(self.topology.residues()):
            if resid == insert_water:
                for atom in residue.atoms():
                    atom_indices.append(atom.index)

        # Select a point to insert the water (based on O position)
        rand_nums = np.random.randn(3)
        insert_point = self.sphere_centre + (
                    self.sphere_radius * np.power(np.random.rand(), 1.0 / 3) * rand_nums) / np.linalg.norm(rand_nums)
        #  Generate a random rotation matrix
        R = random_rotation_matrix()
        new_positions = deepcopy(self.positions)
        for i, index in enumerate(atom_indices):
            #  Translate coordinates to an origin defined by the oxygen atom, and normalise
            atom_position = self.positions[index] - self.positions[atom_indices[0]]
            # Rotate about the oxygen position
            if i != 0:
                vec_length = np.linalg.norm(atom_position)
                atom_position = atom_position / vec_length
                # Rotate coordinates & restore length
                atom_position = vec_length * np.dot(R, atom_position) * unit.nanometer
            # Translate to new position
            new_positions[index] = atom_position + insert_point

        # Recouple this water
        self.adjustSpecificWater(atom_indices, 1.0)

        self.context.setPositions(new_positions)
        # Calculate new system energy and acceptance probability
        final_energy = self.context.getState(getEnergy=True).getPotentialEnergy()
        acc_prob = np.exp(self.B) * np.exp(-(final_energy - self.energy) / self.kT) / (self.N + 1)
        self.acceptance_probabilities.append(acc_prob)

        if acc_prob < np.random.rand() or np.isnan(acc_prob):
            # Need to revert the changes made if the move is to be rejected
            # Switch off nonbonded interactions involving this water
            self.adjustSpecificWater(atom_indices, 0.0)
            self.context.setPositions(self.positions)
        else:
            # Update some variables if move is accepted
            self.positions = deepcopy(new_positions)
            self.gcmc_status[gcmc_id] = 1
            self.water_status[wat_id] = 1
            self.N += 1
            self.n_accepted += 1
            # Update energy
            self.energy = final_energy

        return None

    def deleteRandomWater(self):
        """
        Carry out a random water deletion move on the current system
        """
        # Cannot carry out deletion if there are no GCMC waters on
        if np.sum(self.gcmc_status) == 0:
            return None

        # Select a water residue to delete
        gcmc_id = np.random.choice(np.where(self.gcmc_status == 1)[0])  # Position in list of GCMC waters
        delete_water = self.gcmc_resids[gcmc_id]
        wat_id = np.where(np.array(self.water_resids) == delete_water)[0][0]  # Position in list of all waters
        atom_indices = []
        for resid, residue in enumerate(self.topology.residues()):
            if resid == delete_water:
                for atom in residue.atoms():
                    atom_indices.append(atom.index)

        # Switch water off
        self.adjustSpecificWater(atom_indices, 0.0)
        # Calculate energy of new state and acceptance probability
        final_energy = self.context.getState(getEnergy=True).getPotentialEnergy()
        acc_prob = self.N * np.exp(-self.B) * np.exp(-(final_energy - self.energy) / self.kT)
        self.acceptance_probabilities.append(acc_prob)

        if acc_prob < np.random.rand() or np.isnan(acc_prob):
            # Switch the water back on if the move is rejected
            self.adjustSpecificWater(atom_indices, 1.0)
        else:
            # Update some variables if move is accepted
            self.gcmc_status[gcmc_id] = 0
            self.water_status[wat_id] = 0
            self.N -= 1
            self.n_accepted += 1
            # Update energy
            self.energy = final_energy

        return None


class NonequilibriumGCMCSampler(GrandCanonicalMonteCarloSampler):
    """
    Class to carry out GCMC moves in OpenMM, using nonequilibrium candidate Monte Carlo (NCMC)
    to boost acceptance rates
    """
    def __init__(self, system, topology, temperature, integrator, adams=None,
                 chemicalPotential=-6.3*unit.kilocalories_per_mole, adamsShift=0.0, nPertSteps=1, nPropSteps=1,
                 waterName="HOH", ghostFile="gcmc-ghost-wats.txt", referenceAtoms=None, sphereRadius=None,
                 sphereCentre=None, log='gcmc.log', dcd=None, rst7=None, overwrite=False):
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
        chemicalPotential : simtk.unit.Quantity
            Chemical potential of the simulation, default is -6.3 kcal/mol. This should
            be the hydration free energy of water, and may need to be changed for specific
            simulation parameters.
        adamsShift : float
            Shift the B value from Bequil, if B isn't explicitly set. Default is 0.0
        nPertSteps : int
            Number of pertubation steps over which to shift lambda between 0 and 1 (or vice versa).
        nPropSteps : int
            Number of propagation steps to carry out for
        waterName : str
            Name of the water residues. Default is 'HOH'
        waterModel : str
            Name of the water model being used. This is used to identify the equilibrium chemical
            potential to use. Default: TIP3P. Accepted: SPCE, TIP3P, TIP4Pew
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
            Coordinates around which the GCMC sohere is based
        log : str
            Name of the log file to write out
        dcd : str
            Name of the DCD file to write the system out to
        rst7 : str
            Name of the AMBER restart file to write out
        overwrite : bool
            Indicates whether to overwrite already existing data
        """
        # Initialise base class
        GrandCanonicalMonteCarloSampler.__init__(self, system, topology, temperature, adams=adams,
                                                 chemicalPotential=chemicalPotential, adamsShift=adamsShift,
                                                 waterName=waterName, ghostFile=ghostFile,
                                                 referenceAtoms=referenceAtoms, sphereRadius=sphereRadius,
                                                 sphereCentre=sphereCentre, log=log, dcd=dcd, rst7=rst7,
                                                 overwrite=overwrite)

        self.velocities = None  # Need to store velocities for this type of sampling

        # Load in extra NCMC variables
        self.n_pert_steps = nPertSteps
        self.n_prop_steps = nPropSteps
        self.works = []  # Store work values of moves
        self.acceptance_probabilities = []  # Store acceptance probabilities
        self.n_explosions = 0
        self.n_left_sphere = 0  # Number of moves rejected because the water left the sphere

        # Define a compound integrator
        self.compound_integrator = openmm.CompoundIntegrator()
        # Add the MD integrator
        self.compound_integrator.addIntegrator(integrator)
        # Create and add the nonequilibrium integrator
        self.ncmc_integrator = NonequilibriumLangevinIntegrator(temperature=temperature,
                                                                collision_rate=1.0/unit.picosecond,
                                                                timestep=2*unit.femtoseconds, splitting="V R O R V")
        self.compound_integrator.addIntegrator(self.ncmc_integrator)
        # Set the compound integrator to the MD integrator
        self.compound_integrator.setCurrentIntegrator(0)

        self.logger.info("NonequilibriumGCMCSampler object initialised")

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
        self.logger.info("Starting a batch of {} nonequilibrium GCMC moves".format(n))
        old_accepted = self.n_accepted

        # Read in positions
        self.context = context
        state = self.context.getState(getPositions=True, enforcePeriodicBox=True, getVelocities=True)
        self.positions = deepcopy(state.getPositions(asNumpy=True))
        self.velocities = deepcopy(state.getVelocities(asNumpy=True))

        # Update GCMC region based on current state
        self.updateGCMCSphere(state)

        # Set to NCMC integrator
        self.compound_integrator.setCurrentIntegrator(1)

        #  Execute moves
        for i in range(n):
            # Insert or delete a water, based on random choice
            if np.random.randint(2) == 1:
                # Attempt to insert a water
                self.insertRandomWater()
            else:
                # Attempt to delete a water
                self.deleteRandomWater()
            self.n_moves += 1
            self.Ns.append(self.N)

        # Set to MD integrator
        self.compound_integrator.setCurrentIntegrator(0)

        self.logger.info("{}/{} moves accepted from this batch".format(self.n_accepted-old_accepted, n))

        return None

    def insertRandomWater(self):
        """
        Carry out a nonequilibrium insertion move for a random water molecule
        """
        # Select a ghost water to insert
        ghost_wats = np.where(self.gcmc_status == 0)[0]
        # Check that there are any ghosts present
        if len(ghost_wats) == 0:
            self.logger.error("No ghost water molecules left, so insertion moves cannot occur - add more ghost waters")
            raise Exception("No ghost water molecules left, so insertion moves cannot occur - add more ghost waters")

        gcmc_id = np.random.choice(ghost_wats)  # Position in list of GCMC waters
        insert_water = self.gcmc_resids[gcmc_id]
        wat_id = np.where(np.array(self.water_resids) == insert_water)[0][0]  # Position in list of all waters
        atom_indices = []
        for resid, residue in enumerate(self.topology.residues()):
            if resid == insert_water:
                for atom in residue.atoms():
                    atom_indices.append(atom.index)

        # Select a point to insert the water (based on O position)
        rand_nums = np.random.randn(3)
        insert_point = self.sphere_centre + (
                self.sphere_radius * np.power(np.random.rand(), 1.0 / 3) * rand_nums) / np.linalg.norm(rand_nums)
        #  Generate a random rotation matrix
        R = random_rotation_matrix()
        old_positions = deepcopy(self.positions)
        new_positions = deepcopy(self.positions)
        for i, index in enumerate(atom_indices):
            #  Translate coordinates to an origin defined by the oxygen atom, and normalise
            atom_position = self.positions[index] - self.positions[atom_indices[0]]
            # Rotate about the oxygen position
            if i != 0:
                vec_length = np.linalg.norm(atom_position)
                atom_position = atom_position / vec_length
                # Rotate coordinates & restore length
                atom_position = vec_length * np.dot(R, atom_position) * unit.nanometer
            # Translate to new position
            new_positions[index] = atom_position + insert_point

        # Need to update the context positions
        self.context.setPositions(new_positions)

        # Set lambda values for each perturbation
        lambdas = np.linspace(0.0, 1.0, self.n_pert_steps + 1)

        # Start running perturbation and propagation kernels
        protocol_work = 0.0 * unit.kilocalories_per_mole
        explosion = False
        self.ncmc_integrator.step(self.n_prop_steps)
        for i in range(self.n_pert_steps):
            state = self.context.getState(getEnergy=True)
            energy_initial = state.getPotentialEnergy()
            # Adjust interactions of this water
            self.adjustSpecificWater(atom_indices, lambdas[i+1])
            state = self.context.getState(getEnergy=True)
            energy_final = state.getPotentialEnergy()
            protocol_work += energy_final - energy_initial
            # Propagate the system
            try:
                self.ncmc_integrator.step(self.n_prop_steps)
            except:
                print("Caught explosion!")
                explosion = True
                self.n_explosions += 1
                break

        # Get the protocol work (in units of kT)
        #protocol_work = self.ncmc_integrator.get_protocol_work(dimensionless=True)
        self.logger.info("Insertion work = {}".format(protocol_work))
        self.works.append(protocol_work)

        # Update variables and GCMC sphere
        #self.gcmc_status[gcmc_id] = 1
        self.water_status[wat_id] = 1
        state = self.context.getState(getPositions=True, enforcePeriodicBox=True)
        self.positions = state.getPositions(asNumpy=True)
        self.updateGCMCSphere(state)

        # Check which waters are still in the GCMC sphere
        gcmc_wats_new = [wat for i, wat in enumerate(self.gcmc_resids) if self.gcmc_status[i] == 1]

        # Calculate acceptance probability
        if insert_water not in gcmc_wats_new:
            # If the inserted water leaves the sphere, the move cannot be reversed and therefore cannot be accepted
            acc_prob = -1
            self.n_left_sphere += 1
            self.logger.info("Move rejected due to water leaving the GCMC sphere")
        elif explosion:
            acc_prob = -1
            self.logger.info("Move rejected due to an instability during integration")
        else:
            # Calculate acceptance probability based on protocol work
            acc_prob = np.exp(self.B) * np.exp(-protocol_work/self.kT) / self.N  # Here N is the new value

        self.acceptance_probabilities.append(acc_prob)

        # Update or reset the system, depending on whether the move is accepted or rejected
        if acc_prob < np.random.rand() or np.isnan(acc_prob):
            # Need to revert the changes made if the move is to be rejected
            self.adjustSpecificWater(atom_indices, 0.0)
            self.context.setPositions(old_positions)
            self.context.setVelocities(-self.velocities)  # Reverse velocities on rejection
            self.positions = deepcopy(old_positions)
            self.velocities *= -1
            state = self.context.getState(getPositions=True, enforcePeriodicBox=True)
            #self.gcmc_status[gcmc_id] = 0
            self.water_status[wat_id] = 0
            self.updateGCMCSphere(state)
        else:
            # Update some variables if move is accepted
            self.N = len(gcmc_wats_new)
            self.n_accepted += 1
            state = self.context.getState(getPositions=True, enforcePeriodicBox=True, getVelocities=True)
            self.positions = deepcopy(state.getPositions(asNumpy=True))
            self.velocities = deepcopy(state.getVelocities(asNumpy=True))
            self.updateGCMCSphere(state)

        return None

    def deleteRandomWater(self):
        """
        Carry out a nonequilibrium deletion move for a random water molecule
        """
        # Cannot carry out deletion if there are no GCMC waters on
        if np.sum(self.gcmc_status) == 0:
            return None

        old_positions = deepcopy(self.positions)

        # Select a water residue to delete
        gcmc_id = np.random.choice(np.where(self.gcmc_status == 1)[0])  # Position in list of GCMC waters
        delete_water = self.gcmc_resids[gcmc_id]
        wat_id = np.where(np.array(self.water_resids) == delete_water)[0][0]  # Position in list of all waters
        atom_indices = []
        for resid, residue in enumerate(self.topology.residues()):
            if resid == delete_water:
                for atom in residue.atoms():
                    atom_indices.append(atom.index)

        # Set lambda values for each perturbation
        lambdas = np.linspace(1.0, 0.0, self.n_pert_steps + 1)

        # Start running perturbation and propagation kernels
        protocol_work = 0.0 * unit.kilocalories_per_mole
        explosion = False
        self.ncmc_integrator.step(self.n_prop_steps)
        for i in range(self.n_pert_steps):
            state = self.context.getState(getEnergy=True)
            energy_initial = state.getPotentialEnergy()
            # Adjust interactions of this water
            self.adjustSpecificWater(atom_indices, lambdas[i + 1])
            state = self.context.getState(getEnergy=True)
            energy_final = state.getPotentialEnergy()
            protocol_work += energy_final - energy_initial
            # Propagate the system
            try:
                self.ncmc_integrator.step(self.n_prop_steps)
            except:
                print("Caught explosion!")
                explosion = True
                self.n_explosions += 1
                break

        # Get the protocol work (in units of kT)
        #protocol_work = self.ncmc_integrator.get_protocol_work(dimensionless=True)
        self.logger.info("Deletion work = {}".format(protocol_work))
        self.works.append(protocol_work)

        # Update variables and GCMC sphere
        #self.gcmc_status[gcmc_id] = 1  # Leaving the water as 'on' here to check
        self.water_status[wat_id] = 1  # that the deleted water doesn't leave
        state = self.context.getState(getPositions=True, enforcePeriodicBox=True)
        self.positions = state.getPositions(asNumpy=True)
        old_N = self.N
        self.updateGCMCSphere(state)

        # Check which waters are still in the GCMC sphere
        gcmc_wats_new = [wat for i, wat in enumerate(self.gcmc_resids) if self.gcmc_status[i] == 1]

        # Calculate acceptance probability
        if delete_water not in gcmc_wats_new:
            # If the deleted water leaves the sphere, the move cannot be reversed and therefore cannot be accepted
            acc_prob = 0
            self.n_left_sphere += 1
            self.logger.info("Move rejected due to water leaving the GCMC sphere")
        elif explosion:
            acc_prob = 0
            self.logger.info("Move rejected due to an instability during integration")
        else:
            # Calculate acceptance probability based on protocol work
            acc_prob = old_N * np.exp(-self.B) * np.exp(-protocol_work/self.kT)  # N is the old value

        print("\tP = {}".format(acc_prob))
        self.acceptance_probabilities.append(acc_prob)

        # Update or reset the system, depending on whether the move is accepted or rejected
        if acc_prob < np.random.rand() or np.isnan(acc_prob):
            # Need to revert the changes made if the move is to be rejected
            self.adjustSpecificWater(atom_indices, 1.0)
            self.context.setPositions(old_positions)
            self.context.setVelocities(-self.velocities)  # Reverse velocities on rejection
            self.positions = deepcopy(old_positions)
            self.velocities *= -1
            state = self.context.getState(getPositions=True, enforcePeriodicBox=True)
            self.updateGCMCSphere(state)
        else:
            # Update some variables if move is accepted
            #self.gcmc_status[gcmc_id] = 0
            self.water_status[wat_id] = 0
            self.N = len(gcmc_wats_new) - 1  # Accounting for the deleted water
            self.n_accepted += 1
            state = self.context.getState(getPositions=True, enforcePeriodicBox=True, getVelocities=True)
            self.positions = deepcopy(state.getPositions(asNumpy=True))
            self.velocities = deepcopy(state.getVelocities(asNumpy=True))
            self.updateGCMCSphere(state)

        return None

