# -*- coding: utf-8 -*-

"""
Description
-----------
Functions to provide support for grandlig canonical sampling in OpenMM.
These functions are not used during the simulation, but will be relevant in setting up
simulations and processing results

Will Poole
Marley Samways
Ollie Melling
"""
import os
import numpy as np
import mdtraj
import parmed
from openmm import unit
import openmm
from openmm import app
from copy import deepcopy
from scipy.cluster import hierarchy
from tqdm import tqdm


class PDBRestartReporter(object):
    """
    *Very* basic class to write PDB files as a basic form of restarter
    """

    def __init__(self, filename, topology):
        """
        Load in the name of the file and the topology of the system

        Parameters
        ----------
        filename : str
            Name of the PDB file to write out
        topology : openmm.app.Topology
            Topology object for the system of interest
        """
        self.filename = filename
        self.topology = topology

    def report(self, simulation, state):
        """
        Write out a PDB of the current state

        Parameters
        ----------
        simulation : openmm.app.Simulation
            Simulation object being used
        state : openmm.State
            Current State of the simulation
        """
        # Read the positions from the state
        positions = state.getPositions()
        # Write the PDB out
        with open(self.filename, "w") as f:
            app.PDBFile.writeFile(topology=self.topology, positions=positions, file=f)

        return None


def get_data_file(filename):
    """
    Get the absolute path of one of the data files included in the package

    Parameters
    ----------
    filename : str
        Name of the file

    Returns
    -------
    filepath : str
        Name of the file including the path
    """
    filepath = os.path.join(os.path.dirname(__file__), "data", filename)
    if os.path.isfile(filepath):
        return filepath
    else:
        raise Exception("{} does not exist!".format(filepath))


def add_ghosts(
    topology, positions, molfile="tip3p.pdb", n=10, pdb="gcmc-extra-wats.pdb"
):
    """
    Function to add molecules to a topology, as extras for GCMC
    This is to avoid changing the number of particles throughout a simulation
    Instead, we can just switch between 'ghost' and 'real' molecules...

    Parameters
    ----------
    topology : openmm.app.Topology
        Topology of the initial system
    positions : openmm.unit.Quantity
        Atomic coordinates of the initial system
    molfile : str
        Name of the file defining the molecule to insert. Must be a
        PDB file containing a single molecule
    n : int
        Number of ghosts to add to the system
    pdb : str
        Name of the PDB file to write containing the updated system
        This will be useful for visualising the results obtained.

    Returns
    -------
    modeller.topology : openmm.app.Topology
        Topology of the system after modification
    modeller.positions : openmm.unit.Quantity
        Atomic positions of the system after modification
    ghosts : list
        List of the residue numbers (counting from 0) of the ghost
        molecules added to the system.
    """
    # Create a Modeller instance of the system
    modeller = app.Modeller(topology=topology, positions=positions)

    # Get box vectors based on the min positions of whats in the box, not the PDB.
    # nanometers
    xmin = min([v[0] for v in modeller.positions])
    ymin = min([v[1] for v in modeller.positions])
    zmin = min([v[2] for v in modeller.positions])
    translation = np.array([xmin._value, ymin._value, zmin._value]) * unit.nanometers

    # print(translation)
    # Read chain IDs
    chain_ids = []
    for chain in modeller.topology.chains():
        chain_ids.append(chain.id)
    # print(chain_ids)

    # Read in simulation box size
    box_vectors = (
        np.asarray(topology.getPeriodicBoxVectors()._value) * unit.nanometers
    )  # nanometers
    # box_size = (
    #     np.array(
    #         [
    #             box_vectors[0][0]._value,
    #             box_vectors[1][1]._value,
    #             box_vectors[2][2]._value,
    #         ]
    #     )
    #     * unit.nanometer
    # )

    # print(box_vectors)

    # Make sure that this molecule file exists
    if not os.path.isfile(molfile):
        # If not, check if it exists in the data directory  (This is where a water and some std ligands will be)
        if os.path.isfile(get_data_file(molfile)):
            molfile = get_data_file(molfile)
        else:
            # Raise an error otherwise
            raise Exception("File {} does not exist".format(molfile))

    # Load the PDB for the molecule
    molecule = app.PDBFile(molfile)

    # Calculate the centre of geometry of the molecule
    cog = np.zeros(3) * unit.nanometers
    for i in range(len(molecule.positions)):
        cog += molecule.positions[i]
    cog /= len(molecule.positions)

    # Add multiple copies of the same molecule, then write out a pdb (for visualisation)
    ghosts = []
    for i in range(n):
        # Read in template molecule positions
        positions = molecule.positions

        # Need to translate the molecule to a random point in the simulation box
        new_centre = (
            np.matmul(np.random.rand(3), box_vectors._value) + translation._value
        ) * unit.nanometers

        new_positions = deepcopy(molecule.positions)
        for i in range(len(positions)):
            # print(positions[i])
            # print(new_centre)
            # print(cog)
            new_positions[i] = positions[i] + new_centre - cog

        # Add the molecule to the model and include the resid in a list
        modeller.add(addTopology=molecule.topology, addPositions=new_positions)
        ghosts.append(modeller.topology._numResidues - 1)

    # Take the ghost chain as the one after the last chain (alphabetically)
    new_chain = chr(((ord(chain_ids[-1]) - 64) % 26) + 65)
    # Renumber all ghost molecules and assign them to the new chain
    for resid, residue in enumerate(modeller.topology.residues()):
        if resid in ghosts:
            residue.id = str(((resid - 1) % 9999) + 1)
            residue.chain.id = new_chain

    # Write the new topology and positions to a PDB file
    if pdb is not None:
        with open(pdb, "w") as f:
            app.PDBFile.writeFile(
                topology=modeller.topology,
                positions=modeller.positions,
                file=f,
                keepIds=True,
            )

        with open(pdb.split(".pdb")[0] + ".cif", "w") as f:
            app.PDBxFile.writeFile(
                topology=modeller.topology,
                positions=modeller.positions,
                file=f,
                keepIds=True,
            )

        # # Want to correct the residue IDs of the added molecules as this can sometimes cause issues
        # with open(pdb, 'r') as f:
        #     lines = f.readlines()
        #
        # max_resid = ghosts[0] + 1  # Start the new resids at the first ghost resid (+1)
        # with open(pdb, 'w') as f:
        #     for line in lines:
        #         # Automatically write out non-atom lines
        #         if not any([line.startswith(x) for x in ['ATOM', 'HETATM', 'TER']]):
        #             f.write(line)
        #         else:
        #             # Correct the residue ID if this corresponds to an added water
        #             if line[21] in new_chains:
        #                 f.write("{}{:4d}{}".format(line[:22],
        #                                            (max_resid % 9999) + 1,
        #                                            line[26:]))
        #             else:
        #                 f.write(line)
        #
        #             # Need to change the resid if there is a TER line
        #             if line.startswith('TER'):
        #                 max_resid += 1

    return modeller.topology, modeller.positions, ghosts


def remove_ghosts(topology, positions, ghosts=None, pdb="gcmc-removed-ghosts.pdb"):
    """
    Function to remove ghost molecules from a topology, after a simulation.
    This is so that a structure can then be used to run further analysis without ghosts
    disturbing the system.

    Parameters
    ----------
    topology : openmm.app.Topology
        Topology of the initial system
    positions : openmm.unit.Quantity
        Atomic coordinates of the initial system
    ghosts : list
        List of residue IDs for the ghost molecules to be deleted
    pdb : str
        Name of the PDB file to write containing the updated system
        This will be useful for visualising the results obtained.

    Returns
    -------
    modeller.topology : openmm.app.Topology
        Topology of the system after modification
    modeller.positions : openmm.unit.Quantity
        Atomic positions of the system after modification
    """
    # Do nothing if no ghost molecules are specified
    if ghosts is None:
        raise Exception("No ghost molecules defined! Nothing to do.")

    # Create a Modeller instance
    modeller = app.Modeller(topology=topology, positions=positions)

    # Find the residues which need to be removed, and delete them
    delete_mols = []  # Residue objects for molecules to be deleted
    for resid, residue in enumerate(modeller.topology.residues()):
        if resid in ghosts:
            delete_mols.append(residue)
    modeller.delete(toDelete=delete_mols)

    # Save PDB file
    if pdb is not None:
        with open(pdb, "w") as f:
            app.PDBFile.writeFile(
                topology=modeller.topology,
                positions=modeller.positions,
                file=f,
            )

    return modeller.topology, modeller.positions


def read_ghosts_from_file(ghost_file):
    """
    Read in the ghost molecules residue IDs from each from in the ghost file

    Parameters
    ----------
    ghost_file : str
        File containing the IDs of the ghost residues in each frame

    Returns
    -------
    ghost_resids : list
        List of lists, containing residue IDs for each frame
    """
    # Read in residue IDs for the ghost molecules in each frame
    ghost_resids = []
    with open(ghost_file, "r") as f:
        for line in f.readlines():
            ghost_resids.append([int(resid) for resid in line.split(",")])

    return ghost_resids


def convert_conc_to_volume(conc):
    """
    Calculate the concentration of a molecule from a given average volume per ligand
    (can be calculated exactly by rearrangement)

    Parameters
    ----------
    conc : openmm.unit.Quantity
        Concentration of interest

    Returns
    -------
    v_per_mol : openmm.unit.Quantity
        Volume per molecule
    """
    # Make sure that the concentration has units of M - this should raise an error otherwise
    conc = conc.in_units_of(unit.molar)

    # Convert to volume per molecule
    v_per_mol = (1 / (conc * unit.AVOGADRO_CONSTANT_NA)).in_units_of(unit.angstroms**3)

    return v_per_mol


def convert_vol_to_conc(v_per_mol):
    """
    Calculate the volume per molecule from a given concentration
    (can be calculated exactly by rearrangement)

    Parameters
    ----------
    v_per_mol : openmm.unit.Quantity
        Volume per molecule

    Returns
    -------
    conc : openmm.unit.Quantity
        Concentration of interest
    """
    v_per_mol = v_per_mol.in_units_of(unit.angstroms**3)

    # Make sure that the concentration has units of M - this should raise an error otherwise
    conc = (1 / (v_per_mol * unit.AVOGADRO_CONSTANT_NA)).in_units_of(unit.molar)

    return conc


def read_prepi(filename):
    """
    Function to read in some atomic data and bonding information from an AMBER prepi file

    Parameters
    ----------
    filename : str
        Name of the prepi file

    Returns
    -------
    atom_data : list
        A list containing a list for each atom, of the form [name, type, charge], where each are strings
    bonds : list
        A list containing one list per bond, of the form [name1, name2]
    """
    with open(filename, "r") as f:
        lines = f.readlines()

    atom_dict = {}  #  Indicates the ID number of each atom name
    atom_data = []  #  Will store the name, type and charge of each atom
    bonds = []  #  List of bonds between atoms
    for i, line_i in enumerate(lines):
        line_data = line_i.split()
        # First read in the data from the atom lines
        if len(line_data) > 10:
            atom_id = line_data[0]
            atom_name = line_data[1]
            atom_type = line_data[2]
            bond_id = line_data[4]
            atom_charge = line_data[10]

            # Ignore dummies
            if atom_type == "DU":
                continue

            atom_dict[atom_id] = atom_name
            atom_data.append([atom_name, atom_type, atom_charge])
            # Double checking the atom isn't bonded to a dummy before writing
            if int(bond_id) > 3:
                bond_name = atom_dict[bond_id]
                bonds.append([atom_name, bond_name])
        # Now read in the data from the loop-completion lines
        elif line_i.startswith("LOOP"):
            for line_j in lines[i + 1 :]:
                if len(line_j.split()) == 2:
                    bonds.append(line_j.split())
                else:
                    break

    return atom_data, bonds


def write_conect(pdb, resname, prepi, output):
    """
    Take in a PDB and write out a new one, including CONECT lines for a specified residue, given a .prepi file
    Should make it easy to run this on more residues at a time - though for now it can just be run separately per residue
    but this isn't ideal...

    Parameters
    ----------
    pdb : str
        Name of the input PDB file
    resname : str
        Name of the residue of interest
    prepi : str
        Name of the prepi file
    output : str
        Name of the output PDB file
    """
    # Read in bonds from prepi file
    _, bond_list = read_prepi(prepi)

    resids_done = []  # List of completed residues

    conect_lines = []  # List of CONECT lines to add

    with open(pdb, "r") as f:
        pdb_lines = f.readlines()

    for i, line_i in enumerate(pdb_lines):
        if not any([line_i.startswith(x) for x in ["ATOM", "HETATM"]]):
            continue

        if line_i[17:21].strip() == resname:
            resid_i = int(line_i[22:26])
            if resid_i in resids_done:
                continue
            residue_atoms = {}  # List of atom names & IDs for this residue
            for line_j in pdb_lines[i:]:
                # Make sure this is an atom line
                if not any([line_j.startswith(x) for x in ["ATOM", "HETATM"]]):
                    break
                # Make sure the following lines correspond to this resname and resid
                resid_j = int(line_j[22:26])
                if resid_j != resid_i or line_j[17:21].strip() != resname:
                    break
                # Read the atom data in for this residue
                atom_name = line_j[12:16].strip()
                atom_id = int(line_j[6:11])
                residue_atoms[atom_name] = atom_id
            # Add CONECT lines
            for bond in bond_list:
                conect_lines.append(
                    "CONECT{:>5d}{:>5d}\n".format(
                        residue_atoms[bond[0]], residue_atoms[bond[1]]
                    )
                )
            resids_done.append(resid_i)

    # Write out the new PDB file, including CONECT lines
    with open(output, "w") as f:
        for line in pdb_lines:
            if not line.startswith("END"):
                f.write(line)
            else:
                for line_c in conect_lines:
                    f.write(line_c)
                f.write(line)

    return None


def create_ligand_xml(prmtop, prepi, resname="LIG", output="lig.xml"):
    """
    Takes two AMBER parameter files (.prmtop and .prepi) for a small molecule and uses them to create an XML file
    which can be used to load the force field parameters for the ligand into OpenMM
    This function could do with some tidying at some point...

    Parameters
    ----------
    prmtop : str
        Name of the .prmtop file
    prepi : str
        Name of the .prepi file
    resname : str
        Residue name of the small molecule
    output : str
        Name of the output XML file
    """
    prmtop = parmed.load_file(prmtop)
    openmm_params = parmed.openmm.OpenMMParameterSet.from_structure(prmtop)
    tmp_xml = os.path.splitext(output)[0] + "-tmp.xml"
    openmm_params.write(tmp_xml)

    # Need to add some more changes here though, as the XML is incomplete
    atom_data, bond_list = read_prepi(prepi)

    # Read the temporary XML data back in
    with open(tmp_xml, "r") as f:
        tmp_xml_lines = f.readlines()

    with open(output, "w") as f:
        # First few lines get written out automatically
        for line in tmp_xml_lines[:4]:
            f.write(line)

        # First, we worry about the <AtomTypes> section
        f.write("  <AtomTypes>\n")
        for line in tmp_xml_lines:
            # Loop over the lines for each atom class
            if "<Type " in line:
                # Read in the data for this XML line
                type_data = {}
                for x in line.split():
                    if "=" in x:
                        key = x.split("=")[0]
                        item = x.split("=")[1].strip("/>").strip('"')
                        type_data[key] = item

                # For each atom with this type, we write out a new line - can probably avoid doing this...
                for atom in atom_data:
                    if atom[1] != type_data["class"]:
                        continue
                    new_line = '    <Type name="{}-{}" class="{}" element="{}" mass="{}"/>\n'.format(
                        resname,
                        atom[0],
                        type_data["class"],
                        type_data["element"],
                        type_data["mass"],
                    )
                    f.write(new_line)
            elif "</AtomTypes>" in line:
                f.write("  </AtomTypes>\n")
                break

        # Now need to generate the actual residue template
        f.write(" <Residues>\n")
        f.write('  <Residue name="{}">\n'.format(resname))
        # First, write the atoms
        for atom in atom_data:
            f.write(
                '   <Atom name="{0}" type="{1}-{0}" charge="{2}"/>\n'.format(
                    atom[0], resname, atom[2]
                )
            )
        # Then the bonds
        for bond in bond_list:
            f.write(
                '   <Bond atomName1="{}" atomName2="{}"/>\n'.format(bond[0], bond[1])
            )
        f.write("  </Residue>\n")
        f.write(" </Residues>\n")

        # Now we can write out the rest, from the <HarmonicBondForce> section onwards
        for i, line_i in enumerate(tmp_xml_lines):
            if "<HarmonicBondForce>" in line_i:
                for line_j in tmp_xml_lines[i:]:
                    # Some lines need the typeX swapped for classX
                    f.write(line_j.replace("type", "class"))
                break

    # Remove temporary file
    os.remove(tmp_xml)

    return None


def create_custom_forces(system, topology, resnames):
    """
    Modify a system's forces to handle alchemical decoupling of certain residues

    Parameters
    ----------
    system : openmm.System
        System of interest
    topology : openmm.app.Topology
        Topology for the system of interest
    resnames : list
        List of residue names that will be switched

    Returns
    -------
    param_dict : dict
        Dictionary containing parameters for each atom of each molecule. The keys are the residue names and the items
        are lists. Each list contains (in order) dictionaries storing the charge, sigma and epsilon parameters for each
        atom of that residue
    custom_sterics : openmm.CustomNonbondedForce
        Handles the softcore LJ interactions
    """
    # Find NonbondedForce - needs to be updated to switch molecules on/off
    for f in range(system.getNumForces()):
        force = system.getForce(f)
        if force.__class__.__name__ == "NonbondedForce":
            nonbonded_force = force

    # Get the parameters corresponding to each molecule type
    param_dict = {}
    for resname in resnames:
        # Create an entry for this residue
        param_dict[resname] = []
        for residue in topology.residues():
            if residue.name == resname:
                for atom in residue.atoms():
                    # Read the parameters of this atom and add to the list of this residue
                    atom_params = nonbonded_force.getParticleParameters(atom.index)
                    param_dict[resname].append(
                        {
                            "charge": atom_params[0],
                            "sigma": atom_params[1],
                            "epsilon": atom_params[2],
                        }
                    )
                # Break this loop, as we only need to read one instance
                break

    #  Need to make sure that the electrostatics are handled using PME (for now)
    if nonbonded_force.getNonbondedMethod() != openmm.NonbondedForce.PME:
        raise Exception("Currently only supporting PME for long range electrostatics")

    # Define the energy expression for the softcore sterics
    # lj_energy = ("U;"
    #              "U = (lambda^soft_a) * 4 * epsilon * x * (x-1.0);"  # Softcore energy
    #              "x = (sigma/reff)^6;"  # Define x as sigma/r(effective)
    #              # Calculate effective distance
    #              "reff = sigma*((soft_alpha*(1.0-lambda)^soft_b + (r/sigma)^soft_c))^(1/soft_c)")

    lj_energy = (
        "U;"
        "U = (lambda) * 4 * epsilon * x * (x-1.0);"  # Softcore energy
        "x = (sigma/reff)^6;"  # Define x as sigma/r(effective)
        # Calculate effective distance
        "reff = sigma*((0.5*(1.0-lambda) + (r/sigma)^6))^(1/6)"
    )
    # Define combining rules
    lj_combining = "; sigma = 0.5*(sigma1+sigma2); epsilon = sqrt(epsilon1*epsilon2); lambda = lambda1*lambda2"

    # Create a customised sterics force
    custom_sterics = openmm.CustomNonbondedForce(lj_energy + lj_combining)
    # Add necessary particle parameters
    custom_sterics.addPerParticleParameter("sigma")
    custom_sterics.addPerParticleParameter("epsilon")
    custom_sterics.addPerParticleParameter("lambda")
    # Assume that the system is periodic (for now)
    custom_sterics.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)
    # Transfer properties from the original force
    custom_sterics.setUseSwitchingFunction(nonbonded_force.getUseSwitchingFunction())
    custom_sterics.setCutoffDistance(nonbonded_force.getCutoffDistance())
    custom_sterics.setSwitchingDistance(nonbonded_force.getSwitchingDistance())
    nonbonded_force.setUseDispersionCorrection(False)
    custom_sterics.setUseLongRangeCorrection(
        nonbonded_force.getUseDispersionCorrection()
    )
    # Set softcore parameters  Dont need them if hard coded into energy function. Speeds things up a it.
    # custom_sterics.addGlobalParameter('soft_alpha', 0.5)
    # custom_sterics.addGlobalParameter('soft_a', 1)
    # custom_sterics.addGlobalParameter('soft_b', 1)
    # custom_sterics.addGlobalParameter('soft_c', 6)

    # Get a list of all molecule atom IDs
    mol_atom_ids = []
    for residue in topology.residues():
        if residue.name in resnames:
            for atom in residue.atoms():
                mol_atom_ids.append(atom.index)

    # Copy all steric interactions into the custom force, and remove them from the original force
    for atom_idx in range(nonbonded_force.getNumParticles()):
        # Get atom parameters
        [charge, sigma, epsilon] = nonbonded_force.getParticleParameters(atom_idx)

        # Make sure that sigma is not equal to zero (sometimes an issue with the reference force, this fixes it.)
        if np.isclose(sigma._value, 0.0):
            sigma = 1.0 * unit.angstrom

        # Add particle to the custom force (with lambda=1 for now)
        custom_sterics.addParticle([sigma, epsilon, 1.0])
        # Dont get rid of the interactions in the original force yet, because we need that information for the exceptions below

    # Get a dictionary of atom pairs subject to exceptions
    exception_dict = {}
    num_excepts = nonbonded_force.getNumExceptions()
    for exception_idx in range(num_excepts):
        [i, j, chargeprod, sigma, epsilon] = nonbonded_force.getExceptionParameters(
            exception_idx
        )
        exception_dict[exception_idx] = [i, j]

    # Make sure all intramolecular interactions for the molecules of interest are set to exceptions, so they aren't switched off
    for residue in topology.residues():
        if residue.name in resnames:
            # Get a list of atom IDs for this residue
            atom_ids = [atom.index for atom in residue.atoms()]
            # Loop over all possible interactions between atoms in this molecule
            for x, atom_x in enumerate(atom_ids):
                for atom_y in atom_ids[x + 1 :]:
                    # Check if there is an exception already for this interaction
                    except_id = None
                    """
                    for exception_idx in range(nonbonded_force.getNumExceptions()):
                        [i, j, chargeprod, sigma, epsilon] = nonbonded_force.getExceptionParameters(exception_idx)
                        if atom_x in [i, j] and atom_y in [i, j]:
                            except_id = exception_idx
                            break
                    """
                    for exception_idx in range(num_excepts):
                        if (
                            atom_x in exception_dict[exception_idx]
                            and atom_y in exception_dict[exception_idx]
                        ):
                            except_id = exception_idx
                            break
                    # Add an exception if there is not one already
                    if except_id is None:
                        [charge_x, sigma_x, epsilon_x] = (
                            nonbonded_force.getParticleParameters(atom_x)
                        )
                        [charge_y, sigma_y, epsilon_y] = (
                            nonbonded_force.getParticleParameters(atom_y)
                        )
                        # Combine parameters (Lorentz-Berthelot)
                        chargeprod = charge_x * charge_y
                        sigma = 0.5 * (sigma_x + sigma_y)
                        epsilon = (epsilon_x * epsilon_y).sqrt()
                        # Create the exception
                        nonbonded_force.addException(
                            atom_x, atom_y, chargeprod, sigma, epsilon
                        )

    # Copy over all exceptions into the new force as exclusions and add to the exception forces, where necessary
    for exception_idx in range(nonbonded_force.getNumExceptions()):
        [i, j, chargeprod, sigma, epsilon] = nonbonded_force.getExceptionParameters(
            exception_idx
        )

        # Make sure that sigma is not equal to zero
        if np.isclose(sigma._value, 0.0):
            sigma = 1.0 * unit.angstrom

        # Copy this over as an exclusion so it isn't counted by the CustomNonbonded Force
        custom_sterics.addExclusion(i, j)

    # Turn off everything in the nonbonded force to avoid double counting with the custom nonbonded force
    for atom_idx in range(nonbonded_force.getNumParticles()):
        # Get atom parameters
        [charge, sigma, epsilon] = nonbonded_force.getParticleParameters(atom_idx)

        # Make sure that sigma is not equal to zero
        if np.isclose(sigma._value, 0.0):
            sigma = 1.0 * unit.angstrom

        # Disable steric interactions in the original force by setting epsilon=0 (keep the charges for PME purposes)
        nonbonded_force.setParticleParameters(atom_idx, charge, sigma, abs(0))

    # Add the custom force to the system
    system.addForce(custom_sterics)

    # return param_dict, custom_sterics, electrostatic_exceptions, steric_exceptions
    return param_dict, custom_sterics


def LinearAlchemicalFunction(start, end, lambda_in):
    """
    Evaluates the alchemical function.

    Parameters
    ----------
    start : float
        The lambda value at and below which the function will have a value of 0.
    end : float
        The lambda value at and above which the function will have a value of 1.
    lambda_in : float
        The value at which to evaluate the LinearAlchemicalFunction.

    Returns
    -------
    y : float
        The evaluated LinearAlchemicalFunction.
    """
    if lambda_in <= start:
        return 0.0
    elif lambda_in >= end:
        return 1.0
    else:
        return (lambda_in - start) / (end - start)


def get_lambda_values(lambda_in, vdw_end=0.75):
    """
    Calculate the lambda_sterics and lambda_electrostatics values for a given lambda.
    For a deletion move: Electrostatics are decoupled from lambda=1 to 0.75, and sterics are decoupled from lambda=0.75 to 0.
    For an Insertion move: Sterics are coupled from lambda=0 to 0.75. Electrostatics are coupled from 0.75 to 1.

    Parameters
    ----------
    lambda_in : float
        Input lambda value

    Returns
    -------
    lambda_vdw : float
        Lambda value for steric interactions
    lambda_ele : float
        Lambda value for electrostatic interactions
    """
    if lambda_in > 1.0:
        # Set both values to 1.0 if lambda > 1
        lambda_vdw = 1.0
        lambda_ele = 1.0
    elif lambda_in < 0.0:
        # Set both values to 0.0 if lambda < 0
        lambda_vdw = 0.0
        lambda_ele = 0.0
    else:
        # Evaluate the LinearAlchemicalFunction
        lambda_vdw = LinearAlchemicalFunction(0, vdw_end, lambda_in)
        lambda_ele = LinearAlchemicalFunction(vdw_end, 1, lambda_in)
    return lambda_vdw, lambda_ele


def random_rotation_matrix():
    """
    Generate a random axis and angle for rotation of the molecules coordinates (using the
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
    axis = np.array([rand1 * rand_h, rand2 * rand_h, 1 - 2 * (rand1**2 + rand2**2)])
    axis /= np.linalg.norm(axis)

    # Get a random angle
    theta = np.pi * (2 * np.random.rand() - 1.0)

    # Simplify products & generate matrix
    x, y, z = axis[0], axis[1], axis[2]
    x2, y2, z2 = axis[0] * axis[0], axis[1] * axis[1], axis[2] * axis[2]
    xy, xz, yz = axis[0] * axis[1], axis[0] * axis[2], axis[1] * axis[2]
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    rot_matrix = np.array(
        [
            [
                cos_theta + x2 * (1 - cos_theta),
                xy * (1 - cos_theta) - z * sin_theta,
                xz * (1 - cos_theta) + y * sin_theta,
            ],
            [
                xy * (1 - cos_theta) + z * sin_theta,
                cos_theta + y2 * (1 - cos_theta),
                yz * (1 - cos_theta) - x * sin_theta,
            ],
            [
                xz * (1 - cos_theta) - y * sin_theta,
                yz * (1 - cos_theta) + x * sin_theta,
                cos_theta + z2 * (1 - cos_theta),
            ],
        ]
    )

    return rot_matrix


def shift_ghost_molecules(
    ghost_file, topology=None, trajectory=None, t=None, output=None
):
    """
    Translate all ghost molecules in a trajectory out of the simulation box, to make
    visualisation clearer

    Parameters
    ----------
    ghost_file : str
        Name of the file containing the ghost water residue IDs at each frame
    topology : str
        Name of the topology/connectivity file (e.g. PDB, GRO, etc.)
    trajectory : str
        Name of the trajectory file (e.g. DCD, XTC, etc.)
    t : mdtraj.Trajectory
        Trajectory object, if already loaded
    output : str
        Name of the file to which the new trajectory is written. If None, then a
        Trajectory will be returned

    Returns
    -------
    t : mdtraj.Trajectory
        Will return a trajectory object, if no output file name is given
    """
    # Read in residue IDs for the ghost waters in each frame
    ghost_resids = read_ghosts_from_file(ghost_file)

    # Read in trajectory data
    if t is None:
        t = mdtraj.load(trajectory, top=topology, discard_overlapping_frames=False)

    # Identify which atoms need to be moved out of sight
    ghost_atom_ids = []
    for frame in range(len(ghost_resids)):
        atom_ids = []
        for i, residue in enumerate(t.topology.residues):
            if i in ghost_resids[frame]:
                for atom in residue.atoms:
                    atom_ids.append(atom.index)
        ghost_atom_ids.append(atom_ids)

    # Shift coordinates of ghost atoms by several unit cells and write out trajectory
    for frame, atom_ids in enumerate(ghost_atom_ids):
        for index in atom_ids:
            t.xyz[frame, index, :] += 5 * t.unitcell_lengths[frame, :]

    # Either return the trajectory or save to file
    if output is None:
        return t
    else:
        t.save(output)
        return None


def wrap_waters(topology=None, trajectory=None, t=None, output=None):
    """
    Wrap water molecules if the coordinates haven't been wrapped by the DCDReporter

    Parameters
    ----------
    topology : str
        Name of the topology/connectivity file (e.g. PDB, GRO, etc.)
    trajectory : str
        Name of the trajectory file (e.g. DCD, XTC, etc.)
    t : mdtraj.Trajectory
        Trajectory object, if already loaded
    output : str
        Name of the file to which the new trajectory is written. If None, then a
        Trajectory will be returned

    Returns
    -------
    t : mdtraj.Trajectory
        Will return a trajectory object, if no output file name is given
    """
    # Load trajectory data, if not already
    if t is None:
        t = mdtraj.load(trajectory, top=topology, discard_overlapping_frames=False)

    n_frames, n_atoms, n_dims = t.xyz.shape

    # Fix all frames
    for f in range(n_frames):
        for residue in t.topology.residues:
            # Skip if this is a protein residue
            if residue.name not in ["WAT", "HOH", "L02"]:
                continue

            # Find the maximum and minimum distances between this residue and the reference atom
            for atom in residue.atoms:
                if "O" in atom.name:
                    pos = t.xyz[f, atom.index, :]

            # Calculate the correction vector based on the separation
            box = t.unitcell_lengths[f, :]

            new_pos = deepcopy(pos)
            for i in range(3):
                while new_pos[i] >= box[i]:
                    new_pos[i] -= box[i]
                while new_pos[i] <= 0:
                    new_pos[i] += box[i]

            correction = new_pos - pos

            # Apply the correction vector to each atom in the residue
            for atom in residue.atoms:
                t.xyz[f, atom.index, :] += correction

    # Either return or save the trajectory
    if output is None:
        return t
    else:
        t.save(output)
        return None


def align_traj(topology=None, trajectory=None, t=None, reference=None, output=None):
    """
    Align a trajectory to the protein

    Parameters
    ----------
    topology : str
        Name of the topology/connectivity file (e.g. PDB, GRO, etc.)
    trajectory : str
        Name of the trajectory file (e.g. DCD, XTC, etc.)
    t : mdtraj.Trajectory
        Trajectory object, if already loaded
    reference : str
        Name of a PDB file to align the protein to. May be better to visualise
    output : str
        Name of the file to which the new trajectory is written. If None, then a
        Trajectory will be returned

    Returns
    -------
    t : mdtraj.Trajectory
        Will return a trajectory object, if no output file name is given
    """
    # Load trajectory data, if not already
    if t is None:
        t = mdtraj.load(trajectory, top=topology, discard_overlapping_frames=False)

    # Align trajectory based on protein C-alpha atoms
    protein_ids = [
        atom.index
        for atom in t.topology.atoms
        if atom.residue.is_protein and atom.name == "CA"
    ]
    if reference is None:
        # If there is no reference then align to the first frame in the trajectory
        t.superpose(t, atom_indices=protein_ids)
    else:
        # Load a reference PDB to align the structure to
        t_ref = mdtraj.load(reference)
        t.superpose(t_ref, atom_indices=protein_ids)

    # Return or save trajectory
    if output is None:
        return t
    else:
        t.save(output)
        return None


def recentre_traj(
    topology=None,
    trajectory=None,
    t=None,
    name="CA",
    resname="ALA",
    resid=1,
    output=None,
):
    """
    Recentre a trajectory based on a specific protein residue. Assumes that the
    protein has not been broken by periodic boundaries.
    Would be best to do this step before aligning a trajectory

    Parameters
    ----------
    topology : str
        Name of the topology/connectivity file (e.g. PDB, GRO, etc.)
    trajectory : str
        Name of the trajectory file (e.g. DCD, XTC, etc.)
    t : mdtraj.Trajectory
        Trajectory object, if already loaded
    resname : str
        Name of the atom to centre the trajectorname.lower(
    resname : str
        Name of the protein residue to centre the trajectory on. Should be a
        binding site residue
    resid : int
        ID of the protein residue to centre the trajectory. Should be a binding
        site residue
    output : str
        Name of the file to which the new trajectory is written. If None, then a
        Trajectory will be returned

    Returns
    -------
    t : mdtraj.Trajectory
        Will return a trajectory object, if no output file name is given
    """
    # Load trajectory
    if t is None:
        t = mdtraj.load(trajectory, top=topology, discard_overlapping_frames=False)
    n_frames, n_atoms, n_dims = t.xyz.shape

    # Get IDs of protein atoms
    protein_ids = [atom.index for atom in t.topology.atoms if atom.residue.is_protein]

    # Find the index of the C-alpha atom of this residue
    ref_idx = None
    for residue in t.topology.residues:
        if residue.name.lower() == resname.lower() and residue.resSeq == resid:
            for atom in residue.atoms:
                if atom.name.lower() == name.lower():
                    ref_idx = atom.index
    if ref_idx is None:
        raise Exception(
            "Could not find atom {} of residue {}{}!".format(
                name, resname.capitalize(), resid
            )
        )

    # Fix all frames
    for f in range(n_frames):
        # Box dimensions for this frame
        box = t.unitcell_lengths[f, :]

        # Recentre all protein chains
        for chain in t.topology.chains:
            # Skip if this is a non-protein chain
            if not all([atom.index in protein_ids for atom in chain.atoms]):
                continue

            # Find the closest distance between this chain and the reference
            min_dists = 1e8 * np.ones(3)
            for atom in chain.atoms:
                # Distance between this atom and reference
                v = t.xyz[f, atom.index, :] - t.xyz[f, ref_idx, :]
                for i in range(3):
                    if abs(v[i]) < min_dists[i]:
                        min_dists[i] = v[i]

            # Calculate the correction vector based on the separation
            correction = np.zeros(3)

            for i in range(3):
                if -2 * box[i] < min_dists[i] < -0.5 * box[i]:
                    correction[i] += box[i]
                elif 0.5 * box[i] < min_dists[i] < 2 * box[i]:
                    correction[i] -= box[i]

            # Apply the correction vector to each atom in the residue
            for atom in chain.atoms:
                t.xyz[f, atom.index, :] += correction

        # Recentre all non-protein residues
        for residue in t.topology.residues:
            # Skip if this is a protein residue
            if any([atom.index in protein_ids for atom in residue.atoms]):
                continue

            # Find the closest distance between this residue and the reference
            min_dists = 1e8 * np.ones(3)
            for atom in residue.atoms:
                # Distance between this atom and reference
                v = t.xyz[f, atom.index, :] - t.xyz[f, ref_idx, :]
                for i in range(3):
                    if abs(v[i]) < min_dists[i]:
                        min_dists[i] = v[i]

            # Calculate the correction vector based on the separation
            correction = np.zeros(3)

            for i in range(3):
                if -2 * box[i] < min_dists[i] < -0.5 * box[i]:
                    correction[i] += box[i]
                elif 0.5 * box[i] < min_dists[i] < 2 * box[i]:
                    correction[i] -= box[i]

            # Apply the correction vector to each atom in the residue
            for atom in residue.atoms:
                t.xyz[f, atom.index, :] += correction

    # Either return or save the trajectory
    if output is None:
        return t
    else:
        t.save(output)
        return None


def recentre_traj_new(topology=None, trajectory=None, t=None, output=None):
    """
    Recentre a trajectory based on a specific protein residue. Assumes that the
    protein has not been broken by periodic boundaries.
    Would be best to do this step before aligning a trajectory

    Parameters
    ----------
    topology : str
        Name of the topology/connectivity file (e.g. PDB, GRO, etc.)
    trajectory : str
        Name of the trajectory file (e.g. DCD, XTC, etc.)
    t : mdtraj.Trajectory
        Trajectory object, if already loaded
    output : str
        Name of the file to which the new trajectory is written. If None, then a
        Trajectory will be returned

    Returns
    -------
    t : mdtraj.Trajectory
        Will return a trajectory object, if no output file name is given
    """
    # Load trajectory
    if t is None:
        t = mdtraj.load(trajectory, top=topology, discard_overlapping_frames=False)
    n_frames, n_atoms, n_dims = t.xyz.shape

    topology = t.topology
    # Get IDs of protein atoms
    protein_atom_ids = topology.select("protein")
    atomset = [topology.atom(i) for i in protein_atom_ids]
    atomset = [set(atomset)]
    t.image_molecules(inplace=True, anchor_molecules=atomset)

    # Either return or save the trajectory
    if output is None:
        return t
    else:
        t.save(output)
        return None


def write_sphere_traj(
    radius,
    ref_atoms=None,
    topology=None,
    trajectory=None,
    t=None,
    sphere_centre=None,
    output="gcmc_sphere.pdb",
    initial_frame=False,
):
    """
    Write out a multi-frame PDB file containing the centre of the GCMC sphere

    Parameters
    ----------
    radius : float
        Radius of the GCMC sphere in Angstroms
    ref_atoms : list
        List of reference atoms for the GCMC sphere (list of dictionaries)
    topology : str
        Topology of the system, such as a PDB file
    trajectory : str
        Trajectory file, such as DCD
    t : mdtraj.Trajectory
        Trajectory object, if already loaded
    sphere_centre : openmm.unit.Quantity
        Coordinates around which the GCMC sphere is based
    output : str
        Name of the output PDB file
    initial_frame : bool
        Write an extra frame for the topology at the beginning of the trajectory.
        Sometimes necessary when visualising a trajectory loaded onto a PDB
    """
    # Load trajectory
    if t is None:
        t = mdtraj.load(trajectory, top=topology, discard_overlapping_frames=False)
    n_frames, n_atoms, n_dims = t.xyz.shape

    # Get reference atom IDs
    if ref_atoms is not None:
        ref_indices = []
        for ref_atom in ref_atoms:
            found = False
            for residue in t.topology.residues:
                if (
                    residue.name == ref_atom["resname"]
                    and str(residue.resSeq) == ref_atom["resid"]
                ):
                    for atom in residue.atoms:
                        if atom.name == ref_atom["name"]:
                            ref_indices.append(atom.index)
                            found = True
            if not found:
                raise Exception(
                    "Atom {} of residue {}{} not found!".format(
                        ref_atom["name"],
                        ref_atom["resname"].capitalize(),
                        ref_atom["resid"],
                    )
                )

    # Loop over all frames and write to PDB file
    with open(output, "w") as f:
        f.write("HEADER GCMC SPHERE\n")
        f.write("REMARK RADIUS = {} ANGSTROMS\n".format(radius))

        # Figure out the initial coordinates if requested
        if initial_frame:
            t_i = mdtraj.load(topology, discard_overlapping_frames=False)
            # Calculate centre
            if sphere_centre is None:
                centre = np.zeros(3)
                for idx in ref_indices:
                    centre += t_i.xyz[0, idx, :]
                centre *= 10 / len(ref_indices)  # Also convert from nm to A
            else:
                centre = sphere_centre.in_units_of(unit.angstroms)._value
            # Write to PDB file
            f.write("MODEL\n")
            f.write(
                "HETATM{:>5d} {:<4s} {:<4s} {:>4d}    {:>8.3f}{:>8.3f}{:>8.3f}\n".format(
                    1, "CTR", "SPH", 1, centre[0], centre[1], centre[2]
                )
            )
            f.write("ENDMDL\n")

        # Loop over all frames
        for frame in range(n_frames):
            # Calculate sphere centre
            if sphere_centre is None:
                centre = np.zeros(3)
                for idx in ref_indices:
                    centre += t.xyz[frame, idx, :]
                centre *= 10 / len(ref_indices)  # Also convert from nm to A
            else:
                centre = sphere_centre.in_units_of(unit.angstroms)._value
            # Write to PDB file
            f.write("MODEL {}\n".format(frame + 1))
            f.write(
                "HETATM{:>5d} {:<4s} {:<4s} {:>4d}    {:>8.3f}{:>8.3f}{:>8.3f}\n".format(
                    1, "CTR", "SPH", 1, centre[0], centre[1], centre[2]
                )
            )
            f.write("ENDMDL\n")

    return None


def cluster_waters(
    topology,
    trajectory,
    sphere_radius,
    ref_atoms=None,
    sphere_centre=None,
    cutoff=2.4,
    output="gcmc_clusts.pdb",
):
    """
    Carry out a clustering analysis on GCMC water molecules with the sphere. Based on the clustering
    code in the ProtoMS software package.

    This function currently assumes that the system has been aligned and centred on the GCMC sphere (approximately).

    Parameters
    ----------
    topology : str
        Topology of the system, such as a PDB file
    trajectory : str
        Trajectory file, such as DCD
    sphere_radius : float
        Radius of the GCMC sphere in Angstroms
    ref_atoms : list
        List of reference atoms for the GCMC sphere (list of dictionaries)
    sphere_centre : openmm.unit.Quantity
        Coordinates around which the GCMC sphere is based
    cutoff : float
        Distance cutoff used in the clustering
    output : str
        Name of the output PDB file containing the clusters
    """
    # Load trajectory
    t = mdtraj.load(trajectory, top=topology, discard_overlapping_frames=False)
    n_frames, n_atoms, n_dims = t.xyz.shape

    # Get reference atom IDs
    if ref_atoms is not None:
        ref_indices = []
        for ref_atom in ref_atoms:
            found = False
            for residue in t.topology.residues:
                if (
                    residue.name == ref_atom["resname"]
                    and str(residue.resSeq) == ref_atom["resid"]
                ):
                    for atom in residue.atoms:
                        if atom.name == ref_atom["name"]:
                            ref_indices.append(atom.index)
                            found = True
            if not found:
                raise Exception(
                    "Atom {} of residue {}{} not found!".format(
                        ref_atom["name"],
                        ref_atom["resname"].capitalize(),
                        ref_atom["resid"],
                    )
                )

    # wat_coords = []  # Store a list of water coordinates
    # wat_frames = []  # Store a list of the frame that each water is in

    # Get list of water oxygen atom IDs
    wat_ox_ids = []
    for residue in t.topology.residues:
        if residue.name.lower() in ["wat", "hoh", "sol"]:
            for atom in residue.atoms:
                if atom.name.lower() == "o":
                    wat_ox_ids.append(atom.index)

    try:
        wat_coords = np.load("wat_coords.npy")
        wat_frames = np.load("wat_frames.npy")
    except:
        wat_coords = []  # Store a list of water coordinates
        wat_frames = []  # Store a list of the frame that each water is in

        # Get the coordinates of all GCMC water oxygen atoms
        for f in tqdm(range(n_frames)):

            # Calculate sphere centre for this frame
            if ref_atoms is not None:
                centre = np.zeros(3)
                for idx in ref_indices:
                    centre += t.xyz[f, idx, :]
                centre /= len(ref_indices)
            else:
                centre = sphere_centre.in_units_of(unit.nanometer)._value

            # For all waters, check the distance to the sphere centre
            for o in wat_ox_ids:
                # Calculate PBC-corrected vector
                vector = t.xyz[f, o, :] - centre

                # Check length and add to list if within sphere
                if (
                    10 * np.linalg.norm(vector) <= sphere_radius
                ):  # *10 to give Angstroms
                    wat_coords.append(10 * t.xyz[f, o, :])  # Convert to Angstroms
                    wat_frames.append(f)

        np.save("wat_coords.npy", wat_coords)
        np.save("wat_frames.npy", wat_frames)

    # Calculate water-water distances - if the waters are in the same frame are assigned a very large distance
    try:
        dist_list = np.load("dist_list.npy")
    except:
        dist_list = []
        for i in tqdm(range(len(wat_coords))):
            for j in range(i + 1, len(wat_coords)):
                if wat_frames[i] == wat_frames[j]:
                    dist = 1e8
                else:
                    dist = np.linalg.norm(wat_coords[i] - wat_coords[j])
                dist_list.append(dist)

        np.save("dist_list.npy", dist_list)
    # Cluster the waters hierarchically
    tree = hierarchy.linkage(dist_list, method="average")
    wat_clust_ids = hierarchy.fcluster(tree, t=cutoff, criterion="distance")
    n_clusts = max(wat_clust_ids)

    # Sort the clusters by occupancy
    clusts = []
    for i in tqdm(range(1, n_clusts + 1)):
        occ = len([wat for wat in wat_clust_ids if wat == i])
        clusts.append([i, occ])
    clusts = sorted(clusts, key=lambda x: -x[1])
    clust_ids_sorted = [x[0] for x in clusts]
    clust_occs_sorted = [x[1] for x in clusts]
    print(clust_occs_sorted)
    # Calculate the cluster centre and representative position for each cluster
    rep_coords = []
    for i in tqdm(range(n_clusts)):
        clust_id = clust_ids_sorted[i]
        # Calculate the mean position of the cluster
        clust_centre = np.zeros(3)
        for j, wat in enumerate(wat_clust_ids):
            if wat == clust_id:
                clust_centre += wat_coords[j]
        clust_centre /= clust_occs_sorted[i]

        # Find the water observation which is closest to the mean position
        min_dist = 1e8
        rep_wat = None
        for j, wat in enumerate(wat_clust_ids):
            if wat == clust_id:
                dist = np.linalg.norm(wat_coords[j] - clust_centre)
                if dist < min_dist:
                    min_dist = dist
                    rep_wat = j
        rep_coords.append(wat_coords[rep_wat])

    # Write the cluster coordinates to a PDB file
    with open(output, "w") as f:
        f.write("REMARK Clustered GCMC water positions written by grandlig\n")
        for i in range(n_clusts):
            coords = rep_coords[i]
            occ1 = clust_occs_sorted[i]
            occ2 = occ1 / float(n_frames)
            f.write(
                "ATOM  {:>5d} {:<4s} {:<4s} {:>4d}    {:>8.3f}{:>8.3f}{:>8.3f}{:>6.2f}{:>6.2f}\n".format(
                    1,
                    "O",
                    "WAT",
                    i + 1,
                    coords[0],
                    coords[1],
                    coords[2],
                    occ2,
                    occ2,
                )
            )
            f.write("TER\n")
        f.write("END")

    return None


def cluster_molecules(
    topology,
    trajectory,
    sphere_radius,
    resname="L02",
    ref_atoms=None,
    cutoff=2.4,
    output="gcmc_clusts.pdb",
):
    """
    Carry out a clustering analysis on GCMC molecules with the sphere. Based on the clustering
    code in the ProtoMS software package.

    This function only does center of geometry clustering and cannot distinguish binding modes.

    It is provided as a basic helper function and may not be appropriate for certain types of analysis.

    This function currently assumes that the system has been aligned and centred on the GCMC sphere (approximately).

    Parameters
    ----------
    topology : str
        Topology of the system, such as a PDB file
    trajectory : str
        Trajectory file, such as DCD
    sphere_radius : float
        Radius of the GCMC sphere in Angstroms
    resname : str
        Resname of the molecule you want to cluster
    ref_atoms : list
        List of reference atoms for the GCMC sphere (list of dictionaries)
    cutoff : float
        Distance cutoff used in the clustering
    output : str
        Name of the output PDB file containing the clusters
    """
    # Load trajectory
    t = mdtraj.load(trajectory, top=topology, discard_overlapping_frames=False)
    n_frames, n_atoms, n_dims = t.xyz.shape

    # Get reference atom IDs
    if ref_atoms is not None:
        ref_indices = []
        for ref_atom in ref_atoms:
            found = False
            for residue in t.topology.residues:
                if (
                    residue.name == ref_atom["resname"]
                    and str(residue.resSeq) == ref_atom["resid"]
                ):
                    for atom in residue.atoms:
                        if atom.name == ref_atom["name"]:
                            ref_indices.append(atom.index)
                            found = True
            if not found:
                raise Exception(
                    "Atom {} of residue {}{} not found!".format(
                        ref_atom["name"],
                        ref_atom["resname"].capitalize(),
                        ref_atom["resid"],
                    )
                )

    # Get the COG's of all the GCMC molcules since were only going to do centroid clustering for now
    mols = {}
    for residue in t.topology.residues:
        if residue.name == resname:
            mols[residue.index] = []
            for atom in residue.atoms:
                if atom.element.name != "hydrogen":
                    mols[residue.index].append(atom.index)
                else:
                    continue
    n_frags = len(mols.keys())

    frag_coords = np.zeros((n_frames, n_frags, 3))
    for frame in range(n_frames):
        for i, resid in enumerate(mols.keys()):
            coords = np.zeros(3)
            for atom_id in mols[resid]:
                coords += t.xyz[frame, atom_id, :]
            frag_coords[frame, i, :] = coords / len(mols[resid])

    mol_coords = []  # Store a list of water coordinates
    mol_frames = []  # Store a list of the frame that each water is in

    # Get the coordinates of all GCMC water oxygen atoms
    for f in range(n_frames):
        # Calculate sphere centre for this frame
        if ref_atoms is not None:
            centre = np.zeros(3)
            for idx in ref_indices:
                centre += t.xyz[f, idx, :]
            centre /= len(ref_indices)

        for i, resid in enumerate(mols.keys()):
            vector = frag_coords[f, i, :] - centre
            if 10 * np.linalg.norm(vector) <= sphere_radius:
                mol_coords.append(10 * frag_coords[f, i, :])
                mol_frames.append(f)

    # Calculate water-water distances - if the waters are in the same frame are assigned a very large distance
    dist_list = []
    for i in range(len(mol_coords)):
        for j in range(i + 1, len(mol_coords)):
            if mol_frames[i] == mol_frames[j]:
                dist = 1e8
            else:
                dist = np.linalg.norm(mol_coords[i] - mol_coords[j])
            dist_list.append(dist)

    # Cluster the waters hierarchically
    tree = hierarchy.linkage(dist_list, method="average")
    mol_clust_ids = hierarchy.fcluster(tree, t=cutoff, criterion="distance")
    n_clusts = max(mol_clust_ids)

    # Sort the clusters by occupancy
    clusts = []
    for i in range(1, n_clusts + 1):
        occ = len([wat for wat in mol_clust_ids if wat == i])
        clusts.append([i, occ])
    clusts = sorted(clusts, key=lambda x: -x[1])
    clust_ids_sorted = [x[0] for x in clusts]
    clust_occs_sorted = [x[1] for x in clusts]

    # Calculate the cluster centre and representative position for each cluster
    rep_coords = []
    for i in range(n_clusts):
        clust_id = clust_ids_sorted[i]
        # Calculate the mean position of the cluster
        clust_centre = np.zeros(3)
        for j, wat in enumerate(mol_clust_ids):
            if wat == clust_id:
                clust_centre += mol_coords[j]
        clust_centre /= clust_occs_sorted[i]

        # Find the water observation which is closest to the mean position
        min_dist = 1e8
        rep_wat = None
        for j, wat in enumerate(mol_clust_ids):
            if wat == clust_id:
                dist = np.linalg.norm(mol_coords[j] - clust_centre)
                if dist < min_dist:
                    min_dist = dist
                    rep_wat = j
        rep_coords.append(mol_coords[rep_wat])

    # Write the cluster coordinates to a PDB file
    with open(output, "w") as f:
        f.write("REMARK Clustered GCMC water positions written by grandlig\n")
        for i in range(n_clusts):
            coords = rep_coords[i]
            occ1 = clust_occs_sorted[i]
            occ2 = occ1 / float(n_frames)
            f.write(
                "ATOM  {:>5d} {:<4s} {:<4s} {:>4d}    {:>8.3f}{:>8.3f}{:>8.3f}{:>6.2f}{:>6.2f}\n".format(
                    1,
                    "C",
                    "CLU",
                    i + 1,
                    coords[0],
                    coords[1],
                    coords[2],
                    occ2,
                    occ2,
                )
            )
            f.write("TER\n")
        f.write("END")

    return None


def setupmoveTraj(n_moves):
    """
    Set up a trajectory for an individual GCNCMC move to be written to. This function is called by sampler objects when
    the user wishes to vizulize an insertion or deletion move.
    todo: improve!

    Parameters:
    -----------
    n_moves : int
        Move number

    Returns:
    --------
    moveDCD: mdtraj.reporters.DCDReporter
        MDTraj DCD reporter to report to
    name: str
        Name of the trajecotry file
    """
    name = f"move-{n_moves+1}.dcd"
    moveDCD = mdtraj.reporters.DCDReporter(f"move-{n_moves+1}.dcd", 0)
    return moveDCD, name


def calculateBFromConc(mu, r, V_L, T):
    """
    Function calculate the Adams (B) value for a given excess chemical potential and avg. volume per ligand. This assumes as sphere as the GCMC region.

    Parameters
    ----------
    mu: openmm.unit.Quantity
        Excess chemical potential of the molecules
    r: openmm.unit.Quantity
        Sphere radius
    V_L: openmm.unit.Quantity
        Average volume per ligand
    T: openmm.unit.Quantity
        Temperature

    Returns
    -------
    B: openmm.unit.Quantity
        The calculated Adams value

    """
    beta = 1 / (unit.BOLTZMANN_CONSTANT_kB * T * unit.AVOGADRO_CONSTANT_NA)
    B = (beta * mu) + np.log(calcSphereVol(r) / V_L)
    return B


def calcSphereVol(radius):
    """
    Calculate the volume of a sphere given a radius according to 4/3*pi*r**3

    Parameters
    ----------
    radius: float or openmm.unit.Quantity
        Radius of the sphere

    Returns
    -------
    V: float or openmm.unit.Quantity
        Volume of the sphere

    """
    V = 4 / 3 * np.pi * radius**3
    return V
