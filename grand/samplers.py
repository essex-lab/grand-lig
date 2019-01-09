# -*- coding: utf-8 -*-

"""
sampler.py
Marley Samways

Description
-----------
This code is written to execute GCMC moves with water molecules in OpenMM, in a 
way that can easily be included with other OpenMM simulations or implemented
methods, with minimal extra effort.

TODO
----
    * Double check all thermodynamic/GCMC parameters
    * Fix explosion problem
    * Decide if restraints are necessary

References
----------
1 - G. A. Ross, M. S. Bodnarchuk and J. W. Essex, J. Am. Chem. Soc., 2015,
    137, 14930-14943
2 - G. A. Ross, H. E. Bruce Macdonald, C. Cave-Ayland, A. I. Cabedo Martinez
    and J. W. Essex, J. Chem. Theory Comput., 2017, 13, 6373-6381
"""

import numpy as np
from copy import deepcopy
from simtk import unit
from simtk import openmm
from openmmtools.integrators import NonequilibriumLangevinIntegrator


class GrandCanonicalMonteCarloSampler(object):
    """
    Base class for carrying out GCMC moves in OpenMM
    """
    def __init__(self, system, topology, temperature, adams=None, chemicalPotential=None, waterName="HOH",
                 waterModel="tip3p", ghostFile="gcmc-ghost-wats.txt", referenceAtoms=None, sphereRadius=None,
                 useRestraint=False):
        """
        Initialise the object to be used for sampling water insertion/deletion moves

        Notes
        -----
        At some point the mu' values below will have to be replaced with calculated
        hydration free energies for each of the water models.

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
            Chemical potential of the simulation, default is None. This is to be used
            if you don't want to use the equilibrium value or if using a water model
            other than SPCE, TIP3P, TIP4Pew (need to add others).
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
            List containing details of the atom to use as the centre of the GCMC region
            Must contain atom name, residue name and (optionally) residue ID,
            e.g. ['C1', 'LIG', 123] or just ['C1', 'LIG']
        sphereRadius : simtk.unit.Quantity
            Radius of the spherical GCMC region
        useRestraint : bool
            Indicates whether or not to use half-harmonic restraints to keep the GCMC region
            impermeable
        """
        # Set important variables here
        self.system = system
        self.topology = topology
        self.positions = None  # Store no positions upon initialisation
        self.energy = None
        self.context = None
        self.kT = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA * temperature
        self.simulation_box = np.zeros(3) * unit.nanometer  # Set to zero for now

        # Find NonbondedForce - needs to be updated to switch waters on/off
        for f in range(system.getNumForces()):
            force = system.getForce(f)
            if force.__class__.__name__ == "NonbondedForce":
                self.nonbonded_force = force
            # Flag an error if not simulating at constant volume
            elif "Barostat" in force.__class__.__name__:
                raise Exception("GCMC must be used at constant volume!")
        
        # Calculate GCMC-specific variables
        self.N = 0  # Initialise N as zero
        # Read in sphere-specific parameters
        self.sphere_radius = sphereRadius
        self.sphere_centre = None
        volume = (4 * np.pi * sphereRadius ** 3) / 3
        if referenceAtoms is not None:
            self.ref_atoms = self.getReferenceAtomIndices(referenceAtoms)
            # Define custom forces to keep GCMC waters in and non-GCMC waters out, if required:
            if useRestraint:
                self.use_restraint = True
                force_constant = 10000.0  # kJ mol^-1 nm^-2
                self.exclude_bonds = []
                self.exclude_force = None
                self.include_bonds = []
                self.include_force = None
                self.createRestraintForces(force_constant)
            else:
                self.use_restraint = False
        else:
            raise Exception("A set of atoms must be used to define the centre of the sphere!")
        
        # Calculate Adams value, if needed
        if adams is None:
            if chemicalPotential is None:
                assert waterModel.lower() in ['spce', 'tip3p', 'tip4pew'],\
                    "Unsupported water model. Must define mu' manually"
                mu_dict = {"spce": -6.2 * unit.kilocalorie_per_mole,
                           "tip3p": -6.2 * unit.kilocalorie_per_mole,
                           "tip4pew": -6.2 * unit.kilocalorie_per_mole}
                mu = mu_dict[waterModel.lower()]
            else:
                mu = chemicalPotential
            self.B = mu/self.kT + np.log(volume / (30.0 * unit.angstrom ** 3))
        else:
            self.B = adams

        # Other variables
        self.n_moves = 0
        self.n_accepted = 0
        
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
        with open(self.ghost_file, 'w') as f:
            pass

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
        # Transfer properties from the original force
        custom_sterics.setUseSwitchingFunction(self.nonbonded_force.getUseSwitchingFunction())
        custom_sterics.setCutoffDistance(self.nonbonded_force.getCutoffDistance())
        custom_sterics.setSwitchingDistance(self.nonbonded_force.getSwitchingDistance())
        #custom_sterics.setUseLongRangeCorrection(self.nonbonded_force.getUseDispersionCorrection())
        custom_sterics.setUseLongRangeCorrection(False)
        # Assume that the system is periodic (for now)
        custom_sterics.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)
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
                sigma = 1.0 * unit.nanometer
            # Add particle to the custom force (with lambda=1 for now)
            custom_sterics.addParticle([sigma, epsilon, 1.0])
            # Disable steric interactions of waters in the original force by setting epsilon=0
            # We keep the charges for PME purposes
            if atom_idx in water_atom_ids:
                self.nonbonded_force.setParticleParameters(atom_idx, charge, sigma, abs(0.0))

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
        self.n_accepted = 0
        self.n_moves = 0

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
            List containing the atom name, residue name and (optionally) residue ID, in that order.
            May be a list of these values for each reference atom.
        
        Returns
        -------
        atom_indices : list
            Indices of the atoms chosen
        """
        atom_indices = []
        # Convert to list of lists, if not already
        if type(ref_atoms[0]) != list:
            ref_atoms = [ref_atoms]

        # Find atom index for each of the atoms used
        for _atom in ref_atoms:
            found = False
            for residue in self.topology.residues():
                if residue.name != _atom[1]:
                    continue
                if len(_atom) > 2 and residue.id != _atom[2]:
                    continue
                for atom in residue.atoms():
                    if atom.name == _atom[0]:
                        atom_indices.append(atom.index)
                        found = True
            if not found:
                raise Exception("Atom {} of residue {}{} not found!".format(_atom[0], _atom[1].capitalize(), _atom[2]))

        if len(atom_indices) == 0:
            raise Exception("No GCMC reference atoms found")

        return atom_indices

    def createRestraintForces(self, force_constant):
        """
        Create a set of restraints to keep the GCMC sphere impermeable
        This uses two CustomCentroidBondForce objects, one to keep the bulk
        water out, and one to keep the GCMC water in.

        Parameters
        ----------
        force_constant : float
            Force constant for the half-harmonic restraint, in units of kJ/mol/nm^2
        """
        excluder = openmm.CustomCentroidBondForce(2, "k*min(0, distance(g1,g2)-r0)^2")
        excluder.addPerBondParameter("k")
        excluder.addPerBondParameter("r0")
        # Add the reference atoms as a group
        ref_group = excluder.addGroup(self.ref_atoms, list(np.ones_like(self.ref_atoms)))
        for residue in self.topology.residues():
            if residue.name != 'HOH':
                continue
            for atom in residue.atoms():
                # Add a group containing the oxygen water atom
                wat_group = excluder.addGroup([atom.index])
                bond_id = excluder.addBond([ref_group, wat_group], [force_constant, self.sphere_radius / unit.nanometer])
                self.exclude_bonds.append(bond_id)
                break  # Only want the oxygen atom for each water
        self.system.addForce(excluder)
        self.exclude_force = excluder

        # Similarly define a custom force to keep GCMC waters in
        includer = openmm.CustomCentroidBondForce(2, "k*max(0, distance(g1,g2)-r0)^2")
        includer.addPerBondParameter("k")
        includer.addPerBondParameter("r0")
        # Add the reference atoms as a group
        ref_group = includer.addGroup(self.ref_atoms, list(np.ones_like(self.ref_atoms)))
        for residue in self.topology.residues():
            if residue.name != 'HOH':
                continue
            for atom in residue.atoms():
                # Add a group containing the oxygen water atom
                wat_group = includer.addGroup([atom.index])
                bond_id = includer.addBond([ref_group, wat_group], [force_constant, self.sphere_radius / unit.nanometer])
                self.include_bonds.append(bond_id)
                break  # Only want the oxygen atom for each water
        self.system.addForce(includer)
        self.include_force = includer

        return None

    def prepareGCMCSphere(self, context, ghostResids):
        """
        Prepare the GCMC sphere for simulation by loading the coordinates from a
        Context object. Also activates any restraint forces and deletes all ghost
        water molecules.
        """
        # Load context into sampler
        self.context = context

        # Load in positions and box vectors from context
        state = self.context.getState(getPositions=True, enforcePeriodicBox=True)
        self.positions = deepcopy(state.getPositions(asNumpy=True))
        box_vectors = state.getPeriodicBoxVectors(asNumpy=True)
        self.simulation_box = np.array([box_vectors[0,0]._value,
                                        box_vectors[1,1]._value,
                                        box_vectors[2,2]._value]) * unit.nanometer

        # Calculate the centre of the GCMC sphere
        self.sphere_centre = np.zeros(3) * unit.nanometers
        for atom in self.ref_atoms:
            self.sphere_centre += self.positions[atom]
        self.sphere_centre /= len(self.ref_atoms)

        # Loop over waters and check which are in/out of the GCMC sphere at the beginning
        for resid, residue in enumerate(self.topology.residues()):
            if resid not in self.water_resids:
                continue
            for atom in residue.atoms():
                ox_index = atom.index
                break
            wat_id = np.where(np.array(self.water_resids) == resid)[0][0]

            # Remove restraints on ghosts
            if resid in ghostResids:
                if self.use_restraint:
                    self.include_force.setBondParameters(self.include_bonds[wat_id], [0, wat_id+1], [0.0, self.sphere_radius/unit.nanometer])
                    self.exclude_force.setBondParameters(self.exclude_bonds[wat_id], [0, wat_id+1], [0.0, self.sphere_radius/unit.nanometer])
                continue

            # Remove inclusive or exclusive restraint, depending on water position
            vector = self.positions[ox_index]-self.sphere_centre
            # Correct PBCs of this vector - need to make this part cleaner
            for i in range(3):
                if vector[i] >= 0.5 * self.simulation_box[i]:
                    vector[i] -= self.simulation_box[i]
                elif vector[i] <= -0.5 * self.simulation_box[i]:
                    vector[i] += self.simulation_box[i]
            if np.linalg.norm(vector)*unit.nanometer <= self.sphere_radius:
                if self.use_restraint:
                    self.exclude_force.setBondParameters(self.exclude_bonds[wat_id], [0, wat_id+1], [0.0, self.sphere_radius/unit.nanometer])
                self.gcmc_resids.append(resid)  # Add to list of GCMC waters
            elif self.use_restraint:
                self.include_force.setBondParameters(self.include_bonds[wat_id], [0, wat_id+1], [0.0, self.sphere_radius/unit.nanometer])

        # Update parameters for restraints, if necessary
        if self.use_restraint:
            self.include_force.updateParametersInContext(self.context)
            self.exclude_force.updateParametersInContext(self.context)

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
                wat_id = np.where(np.array(self.gcmc_resids) == resid)[0]
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
            gcmc_id = np.where(np.array(self.gcmc_resids) == resid)[0]   # Position in list of GCMC waters
            wat_id = np.where(np.array(self.water_resids) == resid)[0][0]  # Position in list of all waters
            if self.gcmc_status[gcmc_id] == 1:
                atom_ids = []
                for atom in residue.atoms():
                    # Switch off interactions involving the atoms of this residue
                    atom_ids.append(atom.index)
                self.adjustSpecificWater(atom_ids, 0.0)
                # Remove restraint on the water
                if self.use_restraint:
                    self.include_force.setBondParameters(self.include_bonds[wat_id], [0, wat_id+1], [0.0, self.sphere_radius/unit.nanometer])
                    self.include_force.updateParametersInContext(self.context)
                # Update relevant parameters
                self.gcmc_status[gcmc_id] = 0
                self.water_status[wat_id] = 0
                self.N -= 1

        return None

    def updateGCMCSphere(self, state):
        """
        Update the relevant GCMC-sphere related parameters. This also involves monitoring
        which water molecules are in/out of the region when not using restraints
        """
        self.sphere_centre = np.zeros(3) * unit.nanometers
        for atom in self.ref_atoms:
            self.sphere_centre += self.positions[atom]
        self.sphere_centre /= len(self.ref_atoms)

        # If not using restraints, we need to check which waters are in the region or not
        if not self.use_restraint:
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
            self.gcmc_resids = gcmc_resids
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
        # Loop over parameters
        for i, atom_idx in enumerate(atoms):
            # Obtain original parameters
            atom_params = self.water_params[i]
            # Update charge in NonbondedForce
            self.nonbonded_force.setParticleParameters(atom_idx,
                                                       charge=(new_lambda * atom_params["charge"]),
                                                       sigma=atom_params["sigma"],
                                                       epsilon=abs(0.0))
            # Update lambda in CustomNonbondedForce
            self.custom_nb_force.setParticleParameters(atom_idx,
                                                       [atom_params["sigma"], atom_params["epsilon"], new_lambda])

        # Update context with new parameters
        self.nonbonded_force.updateParametersInContext(self.context)
        self.custom_nb_force.updateParametersInContext(self.context)
        
        return None

    def getRandomRotationMatrix(self):
        """
        Generate a random axis and angle for rotation of the water coordinates (using the
        method used for this in the ProtoMS source code (www.protoms.org), and then return
        a rotation matrix produced from these

        Returns
        -------
        rot_matrix : numpy.ndarray
            Rotation matrix generated
        """
        # First generate a random axis about which the rotation will occur
        rand1 = rand2 = 2.0

        while (rand1**2 + rand2**2) >= 1.0:
            rand1 = np.random.rand()
            rand2 = np.random.rand()
        rand_h = 2 * np.sqrt(1.0 - (rand1**2 + rand2**2))
        axis = np.array([rand1 * rand_h, rand2 * rand_h, 1 - 2*(rand1**2 + rand2**2)])
        axis /= np.linalg.norm(axis)

        # Get a random angle
        theta = np.pi * (2*np.random.rand() - 1.0)

        # Simplify products & generate matrix
        x, y, z = axis[0], axis[1], axis[2]
        x2, y2, z2 = axis[0]*axis[0], axis[1]*axis[1], axis[2]*axis[2]
        xy, xz, yz = axis[0]*axis[1], axis[0]*axis[2], axis[1]*axis[2]
        cos_theta, sin_theta = np.cos(theta), np.sin(theta)
        rot_matrix = np.array([[cos_theta + x2*(1-cos_theta),   xy*(1-cos_theta) - z*sin_theta, xz*(1-cos_theta) + y*sin_theta],
                               [xy*(1-cos_theta) + z*sin_theta, cos_theta + y2*(1-cos_theta),   yz*(1-cos_theta) - x*sin_theta],
                               [xz*(1-cos_theta) - y*sin_theta, yz*(1-cos_theta) + x*sin_theta, cos_theta + z2*(1-cos_theta)  ]])

        return rot_matrix

    def report(self):
        """
        Function to report any useful data
        """
        self.writeGhostWaterResids()
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
            f.write("{}".format(ghost_resids[0]))
            for resid in ghost_resids[1:]:
                f.write(",{}".format(resid))
            f.write("\n")

        return None


class StandardGCMCSampler(GrandCanonicalMonteCarloSampler):
    """
    Class to carry out instantaneous GCMC moves in OpenMM
    """
    def __init__(self, system, topology, temperature, adams=None, chemicalPotential=None, waterName="HOH",
                 waterModel="tip3p", ghostFile="gcmc-ghost-wats.txt", referenceAtoms=None, sphereRadius=None,
                 useRestraint=False):
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
            Chemical potential of the simulation, default is None. This is to be used
            if you don't want to use the equilibrium value or if using a water model
            other than SPCE, TIP3P, TIP4Pew (need to add others).
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
            List containing details of the atom to use as the centre of the GCMC region
            Must contain atom name, residue name and (optionally) residue ID,
            e.g. ['C1', 'LIG', 123] or just ['C1', 'LIG']
        sphereRadius : simtk.unit.Quantity
            Radius of the spherical GCMC region
        useRestraint : bool
            Indicates whether or not to use half-harmonic restraints to keep the GCMC region
            impermeable
        """
        # Initialise base class - don't need any more for the instantaneous sampler
        GrandCanonicalMonteCarloSampler.__init__(self, system, topology, temperature, adams=adams,
                                                 chemicalPotential=chemicalPotential, waterName=waterName,
                                                 waterModel=waterModel, ghostFile=ghostFile,
                                                 referenceAtoms=referenceAtoms, sphereRadius=sphereRadius,
                                                 useRestraint=useRestraint)

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
        gcmc_id = np.random.choice(np.where(self.gcmc_status == 0)[0])  # Position in list of GCMC waters
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
        R = self.getRandomRotationMatrix()
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
            # Add in restraint to keep the water in the box
            if self.use_restraint:
                self.include_force.setBondParameters(self.include_bonds[wat_id], [0, wat_id + 1],
                                                     [10000.0, self.sphere_radius / unit.nanometer])
                self.include_force.updateParametersInContext(self.context)
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

        if acc_prob < np.random.rand() or np.isnan(acc_prob):
            # Switch the water back on if the move is rejected
            self.adjustSpecificWater(atom_indices, 1.0)
        else:
            # Update some variables if move is accepted
            self.gcmc_status[gcmc_id] = 0
            self.water_status[wat_id] = 0
            self.N -= 1
            self.n_accepted += 1
            # Remove the restraint, now that the water is non-interacting
            if self.use_restraint:
                self.include_force.setBondParameters(self.include_bonds[wat_id], [0, wat_id + 1],
                                                     [0.0, self.sphere_radius / unit.nanometer])
                self.include_force.updateParametersInContext(self.context)
            # Update energy
            self.energy = final_energy

        return None


class NonequilibriumGCMCSampler(GrandCanonicalMonteCarloSampler):
    """
    Class to carry out GCMC moves in OpenMM, using nonequilibrium candidate Monte Carlo (NCMC)
    to boost acceptance rates
    """
    def __init__(self, system, topology, temperature, integrator, adams=None, chemicalPotential=None, nPertSteps=1,
                 nPropSteps=1, waterName="HOH", waterModel="tip3p", ghostFile="gcmc-ghost-wats.txt",
                 referenceAtoms=None, sphereRadius=None, useRestraint=False):
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
            Chemical potential of the simulation, default is None. This is to be used
            if you don't want to use the equilibrium value or if using a water model
            other than SPCE, TIP3P, TIP4Pew (need to add others).
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
            List containing details of the atom to use as the centre of the GCMC region
            Must contain atom name, residue name and (optionally) residue ID,
            e.g. ['C1', 'LIG', 123] or just ['C1', 'LIG']
        sphereRadius : simtk.unit.Quantity
            Radius of the spherical GCMC region
        useRestraint : bool
            Indicates whether or not to use half-harmonic restraints to keep the GCMC region
            impermeable
        """
        # Initialise base class
        GrandCanonicalMonteCarloSampler.__init__(self, system, topology, temperature, adams=adams,
                                                 chemicalPotential=chemicalPotential, waterName=waterName,
                                                 waterModel=waterModel, ghostFile=ghostFile,
                                                 referenceAtoms=referenceAtoms, sphereRadius=sphereRadius,
                                                 useRestraint=useRestraint)

        # Load in extra NCMC variables
        self.integrator = integrator
        self.n_pert_steps = nPertSteps
        self.n_prop_steps = nPropSteps

        # Need to make sure that the integrator is a Nonequilibrium Langevin integrator
        assert isinstance(self.integrator, NonequilibriumLangevinIntegrator),\
            "Must use a NonequilibriumLangevinIntegrator for nonquilibrium GCMC moves!"

        def move(self, context):
            """
            Carry out a nonequilibrium GCMC move

            Need to write this function...
            """
            return None

        def insertRandomWater(self):
            """
            Carry out a nonequilibrium insertion move for a random water molecule

            Need to write this function...
            """
            return None

        def deleteRandomWater(self):
            """
            Carry out a nonequilibrium deletion move for a random water molecule

            Need to write this function...
            """
            return None
