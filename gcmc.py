# -*- coding: utf-8 -*-

"""
gcmc.py
Marley Samways

Description
-----------
This code is written to execute GCMC moves with water molecules in OpenMM, in a 
way that can easily be included with other OpenMM simulations or implemented
methods, with minimal extra effort.

(Need to add more description on how to use and how this works...)

Notes
-----
To Do:
    - Double check all thermodynamic/GCMC parameters
    - Write out a list of ghost waters to file

References
----------
1 - G. A. Ross, M. S. Bodnarchuk and J. W. Essex, J. Am. Chem. Soc., 2015,
    137, 14930-14943
2 - G. A. Ross, H. E. Bruce Macdonald, C. Cave-Ayland, A. I. Cabedo Martinez
    and J. W. Essex, J. Chem. Theory Comput., 2017, 13, 6373-6381
"""

import numpy as np
from simtk import unit
from copy import deepcopy


class GrandCanonicalMonteCarloSampler(object):
    """
    Class to carry out the GCMC moves in OpenMM
    """
    def __init__(self, system, topology, temperature, boxSize, boxAtoms=None, boxCentre=None,
                 adams=None, chemicalPotential=None, waterName="HOH", waterModel="tip3p",
                 ghostFile="gcmc-ghost-wats.txt"):
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
        boxsize : simtk.unit.Quantity
            Size of the GCMC region in all three dimensions. Must be a 3D
            vector with appropriate units
        boxatoms : list
            List containing details of the atom to use as the centre of the GCMC region
            Must contain atom name, residue name and (optionally) residue ID (numbered from 0),
            e.g. ['C1', 'LIG', 123] or just ['C1', 'LIG']
        boxcentre : simtk.unit.Quantity
            Define coordinates for the centre of the GCMC region. Default is None. If not
            None, this will override the reference atoms.
        adams : float
            Adams B value for the simulation (dimensionless). Default is None,
            if None, the B value is calculated from the box volume and chemical
            potential
        mu : simtk.unit.Quantity
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
        """
        # Set important variables here
        self.system = system
        self.topology = topology
        self.positions = 0  # Store no positions upon initialisation
        self.kT = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA * temperature
        self.simulation_box = np.zeros(3) * unit.nanometer  # Set to zero for now

        # Calculate GCMC-specific variables
        self.N = 0  # Initialise N as zero
        self.box_size = boxSize   # Size of the GCMC region
        # Check whether to use reference atoms for the GCMC box or not
        if boxCentre is not None:
            self.box_atoms = None
            self.box_centre = boxCentre
            self.box_origin = self.box_centre - 0.5 * self.box_size  # Store the box origin
        elif boxAtoms is not None:
            self.box_atoms = self.getReferenceAtomIndices(boxAtoms)  # Atoms around which the GCMC region is centred
            self.box_centre = np.zeros(3) * unit.nanometers  # Store the coordinates of the box centre
            self.box_origin = np.zeros(3) * unit.nanometers  # Initialise the origin of the box
        else:
            raise Exception("Either a set of atoms or coordinates must be used to define the GCMC box!")
        if adams is None:
            if chemicalPotential is None:
                assert waterModel.lower() in ['spce', 'tip3p', 'tip4pew'], "Unsupported water model. Must define mu' manually"
                mu_dict = {"spce" : -6.2 * unit.kilocalorie_per_mole,
                           "tip3p" : -6.2 * unit.kilocalorie_per_mole,
                           "tip4pew" : -6.2 * unit.kilocalorie_per_mole}
                mu = mu_dict[waterModel.lower()]
            else:
                mu = chemicalPotential
            self.adams = mu/self.kT + np.log(boxSize[0]*boxSize[1]*boxSize[2] / 30.0 * unit.angstrom ** 3)
        else:
            self.adams = adams

        # Other variables
        self.n_moves = 0
        self.n_accepted = 0
        
        # Find nonbonded force - needs to be updated when inserting/deleting
        for f in range(system.getNumForces()):
            force = system.getForce(f)
            if force.__class__.__name__ == "NonbondedForce":
                self.nonbonded_force = force
            # Flag an error if not simulating at constant volume
            elif force.__class__.__name__ == "MonteCarloBarostat":
                raise Exception("GCMC must be used at constant volume!")
        
        # Get parameters for the water model
        self.water_params = self.getWaterParameters(waterName)

        # Get water residue IDs & assign statuses to each
        self.water_resids = self.getWaterResids(waterName)
        self.water_status = np.ones(len(self.water_resids))  # 1 indicates on, 0 indicates off

        # Need a list to store the IDs of waters in the GCMC region
        self.waters_in_box = []  # Empty for now

        # Need to open the file to store ghost water IDs
        self.ghost_file = ghostFile

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
        for box_atom in ref_atoms:
            for resid, residue in enumerate(self.topology.residues()):
                if residue.name != box_atom[1]:
                    continue
                if len(box_atom) > 2 and resid != box_atom[2]:
                    continue
                for atom in residue.atoms():
                    if atom.name == box_atom[0]:
                        atom_indices.append(atom.index)
        if len(atom_indices) == 0:
            raise Exception("No GCMC reference atoms found")
        return atom_indices

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

    def deleteGhostWaters(self, context, ghostResids=None, ghostFile=None):
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
        # Switch of the interactions involving ghost waters
        for resid, residue in enumerate(self.topology.residues()):
            if resid in ghost_resids:
                for i, atom in enumerate(residue.atoms()):
                    # Switch off nonbonded interactions involving this water
                    self.nonbonded_force.setParticleParameters(atom.index,
                                                               charge=0*unit.elementary_charge,
                                                               sigma=1*unit.angstrom,
                                                               epsilon=0*unit.kilojoule_per_mole)
                # Mark that this water has been switched off
                for i in range(len(self.water_resids)):
                    if resid == self.water_resids[i]:
                        self.water_status[i] = 0
                        break
        # Update the context with the new parameters and return it
        self.nonbonded_force.updateParametersInContext(context)
        return context

    def deleteWatersInGCMCBox(self, context):
        """
        Function to delete all of the waters currently present in the GCMC region
        This may be useful the plan is to generate a water distribution for this
        region from scratch. If so, it would be recommended to interleave the GCMC
        sampling with coordinate propagation, as this will converge faster.
        
        Parameters
        ----------
        context : simtk.openmm.Context
            Current context of the system
        
        Returns
        -------
        context : simtk.openmm.Context
            Updated context after deleting the relevant waters
        """
        # Read in positions of the context and update GCMC box
        state = context.getState(getPositions=True, enforcePeriodicBox=True)
        self.positions = deepcopy(state.getPositions(asNumpy=True))
        self.updateGCMCBox(context)
        # Loop over all residues to find those of interest
        for resid, residue in enumerate(self.topology.residues()):
            #if resid == 0: continue  # Just using this for testing - delete this later..........................
            if resid in self.waters_in_box:
                for atom in residue.atoms():
                    # Switch off interactions involving the atoms of this residue
                    self.nonbonded_force.setParticleParameters(atom.index,
                                                               charge=0*unit.elementary_charge,
                                                               sigma=1*unit.angstrom,
                                                               epsilon=0*unit.kilojoule_per_mole)
                # Update relevant parametera
                wat_id = np.where(np.array(self.water_resids) == resid)[0]
                self.water_status[wat_id] = 0
                self.N -= 1
                self.waters_in_box = [i for i in self.waters_in_box if i != resid]
        # Update the context with the modified forces
        self.nonbonded_force.updateParametersInContext(context)
        return context

    def updateGCMCBox(self, context):
        """
        Update the centre and origin of the GCMC box. Also count the number of
        water molecules currently present in the box

        Notes
        -----
        Need to double-check (and fix) any issues relating to periodic boundaries
        and binning the waters

        Parameters
        ----------
        context : simtk.openmm.Context
            Current context of the simulation
        """
        # Update box centre and origin - if using reference atoms
        if self.box_atoms is not None:
            self.box_centre = self.positions[self.box_atoms[0]]
            for i in range(1, len(self.box_atoms)):
                self.box_centre += self.positions[self.box_atoms[i]]
            self.box_centre /= len(self.box_atoms)
            self.box_origin = self.box_centre - 0.5 * self.box_size
        # Check which waters are on inside the GCMC box
        # Need to add a PBC correction, but this will do for now
        self.waters_in_box = []
        for resid, residue in enumerate(self.topology.residues()):
            if resid not in self.water_resids:
                # Ignore anything that isn't water
                continue
            # Need the position of this water in the lists stored
            wat_id = np.where(np.array(self.water_resids) == resid)[0]
            if self.water_status[wat_id] != 1:
                # Ignore waters which are switched off
                continue
            for atom in residue.atoms():
                # Need the index of the oxygen atom
                ox_index = atom.index
                break
            # Consider the distance of the water from the GCMC box centre
            vector = self.positions[ox_index] - self.box_centre
            # Correct PBCs of this vector - need to make this part cleaner
            for i in range(3):
                if vector[i] >= 0.5 * self.simulation_box[i]:
                    vector[i] -= self.simulation_box[i]
                elif vector[i] <= -0.5 * self.simulation_box[i]:
                    vector[i] += self.simulation_box[i]
            # Check if the water is sufficiently close to the box centre in each dimension
            if np.all([abs(vector[i]) <= 0.5*self.box_size for i in range(3)]):
                self.waters_in_box.append(resid)
        self.N = len(self.waters_in_box)
        return None

    def move(self, context, n=1):
        """
        Execute one GCMC move on the current system
        
        Parameters
        ----------
        context : simtk.openmm.Context
            Current context of the simulation
        
        Returns
        -------
        context : simtk.openmm.Context
            Updated context after carrying out the move
        n : int
            Number of moves to execute
        """
        # Read in positions and box vectors
        state = context.getState(getPositions=True, enforcePeriodicBox=True)
        self.positions = deepcopy(state.getPositions(asNumpy=True))
        box_vectors = state.getPeriodicBoxVectors(asNumpy=True)
        self.simulation_box = np.array([box_vectors[0,0]._value,
                                        box_vectors[1,1]._value,
                                        box_vectors[2,2]._value]) * unit.nanometer
        # Update GCMC region based on current state
        self.updateGCMCBox(context)
        for i in range(n):
            # Get initial positions and energy
            state = context.getState(getPositions=True, enforcePeriodicBox=True, getEnergy=True)
            self.positions = deepcopy(state.getPositions(asNumpy=True))
            initial_energy = state.getPotentialEnergy()
            # Insert or delete a water, based on random choice
            if np.random.randint(2) == 1:
                # Attempt to insert a water
                context = self.insertRandomWater(context, initial_energy)
            else:
                # Attempt to delete a water
                context = self.deleteRandomWater(context, initial_energy)
            self.n_moves += 1
        return context

    def insertRandomWater(self, context, initial_energy):
        """
        Carry out a random water insertion move on the current system

        Notes
        -----
        Need to double-check (and fix) any issues relating to periodic boundaries
        and the inserted coordinates
        
        Parameters
        ----------
        context : simtk.openmm.Context
            Current context of the system
        initial_energy : simtk.unit.Quantity
            Potential energy of the system before the move
        
        Returns
        -------
        context : simtk.openmm.Context
            Updated context after the move
        """
        # Select a ghost water to insert
        box_fractions = np.random.rand(3)
        insert_point = self.box_origin + box_fractions * self.box_size
        wat_id = np.random.choice(np.where(self.water_status == 0)[0])
        insert_water = self.water_resids[wat_id]
        if insert_water == 0: return context
        atom_indices = []
        for resid, residue in enumerate(self.topology.residues()):
            if resid == insert_water:
                for atom in residue.atoms():
                    atom_indices.append(atom.index)
        # Select a point to insert the water (based on O position)
        box_fractions = np.random.rand(3)
        insert_point = self.box_origin + box_fractions * self.box_size
        # Generate a random rotation matrix
        R = self.getRandomRotationMatrix()
        new_positions = deepcopy(self.positions)
        for i, index in enumerate(atom_indices):
            # Translate coordinates to an origin defined by the oxygen atom, and normalise
            atom_position = self.positions[index] - self.positions[atom_indices[0]]
            # Rotate about the oxygen position
            if i != 0:
                vec_length = np.linalg.norm(atom_position)
                atom_position = atom_position / vec_length
                # Rotate coordinates & restore length
                atom_position = vec_length * np.dot(R, atom_position) * unit.nanometer
            # Translate to new position 
            new_positions[index] = atom_position + insert_point
            # Switch atom's interactions on
            atom_params = self.water_params[i]
            self.nonbonded_force.setParticleParameters(index,
                                                       charge=atom_params["charge"],
                                                       sigma=atom_params["sigma"],
                                                       epsilon=atom_params["epsilon"])
        # Update context to include the insertion
        self.nonbonded_force.updateParametersInContext(context)
        context.setPositions(new_positions)
        # Calculate new system energy and acceptance probability
        final_energy = context.getState(getEnergy=True).getPotentialEnergy()
        acc_prob = np.exp(self.adams) * np.exp(-(final_energy - initial_energy)/self.kT) / (self.N + 1)
        if acc_prob < np.random.rand() or np.isnan(acc_prob):
            # Need to revert the changes made if the move is to be rejected
            # Switch off nonbonded interactions involving this water
            for i, index in enumerate(atom_indices):
                self.nonbonded_force.setParticleParameters(index,
                                                           charge=0*unit.elementary_charge,
                                                           sigma=1*unit.angstrom,
                                                           epsilon=0*unit.kilojoule_per_mole)
            self.nonbonded_force.updateParametersInContext(context)
            context.setPositions(self.positions)
        else:
            # Update some variables if move is accepted
            self.water_status[wat_id] = 1
            self.N += 1
            self.n_accepted += 1
            self.waters_in_box.append(insert_water)
        return context

    def deleteRandomWater(self, context, initial_energy):
        """
        Carry out a random water deletion move on the current system
        
        Parameters
        ----------
        context : simtk.openmm.Context
            Current context of the system
        initial_energy : simtk.unit.Quantity
            Potential energy of the system before the move
        
        Returns
        -------
        context : simtk.openmm.Context
            Updated context after the move
        """
        # Select a water residue to delete
        if len(self.waters_in_box) == 0:
            # No waters to delete
            return context
        delete_water = np.random.choice(self.waters_in_box)
        #if delete_water == 0: return context  # need to remove this later - just used in testing........
        wat_id = np.where(np.array(self.water_resids) == delete_water)[0]
        atom_indices = []
        for resid, residue in enumerate(self.topology.residues()):
            if resid == delete_water:
                for atom in residue.atoms():
                    atom_indices.append(atom.index)
        # Switch water off
        for index in atom_indices:
            self.nonbonded_force.setParticleParameters(index,
                                                       charge=0*unit.elementary_charge,
                                                       sigma=1*unit.angstrom,
                                                       epsilon=0*unit.kilojoule_per_mole)
        self.nonbonded_force.updateParametersInContext(context)
        # Calculate energy of new state and acceptance probability
        final_energy = context.getState(getEnergy=True).getPotentialEnergy()
        acc_prob = self.N * np.exp(-self.adams) * np.exp(-(final_energy - initial_energy)/self.kT)
        if acc_prob < np.random.rand() or np.isnan(acc_prob):
            # Switch the water back on if the move is rejected
            for i, index in enumerate(atom_indices):
                atom_params = self.water_params[i]
                self.nonbonded_force.setParticleParameters(index,
                                                           charge=atom_params["charge"],
                                                           sigma=atom_params["sigma"],
                                                           epsilon=atom_params["epsilon"])
            self.nonbonded_force.updateParametersInContext(context)
        else:
            # Update some variables if move is accepted
            self.water_status[wat_id] = 0
            self.N -= 1
            self.n_accepted += 1
            self.waters_in_box = [resid for resid in self.waters_in_box if resid != delete_water]
        return context

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
        # First generate a random axis about which the rotation will occur
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

    def writeGhostWaterResids(self):
        """
        Write out a comma-separated list of the residue IDs of waters which are
        non-interacting, so that they can be removed from visualisations. It is important 
        to execute this function when writing to trajectory files, so that each line
        in the ghost water file corresponds to a frame in the trajectory
        """
        # Need to write this function
        with open(self.ghost_file, 'a') as f:
            ghost_resids = np.where(self.water_status == 0)[0]
            f.write("{}".format(ghost_resids[0]))
            for resid in ghost_resids[1:]:
                f.write(",{}".format(resid))
            f.write("\n")
        return None

