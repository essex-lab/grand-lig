import argparse
import grandlig as grand
from openmm import *
from openmm.app import *
from openmm.unit import *
import openmmtools
import numpy as np
from openmmtools.integrators import BAOABIntegrator
import math

parser = argparse.ArgumentParser()
parser.add_argument("--pdb", help="Input PDB file of solvated protein/complex")
parser.add_argument("--lig", help="PDB file of ligand to add as ghosts")
parser.add_argument("--xml", help="XML file containing parameters for the ligands")
parser.add_argument("--mu", help="Ligand mu_ex in units of kcal/mol", type=float)
parser.add_argument("--conc", help="Concentration in units of M", type=int)
parser.add_argument(
    "--st",
    help="Switching time in ps. The default settings use n_prop = 50 and will calculate n_pert automattically",
    type=int,
)
parser.add_argument(
    "--sph_resns",
    help="Resnames of residues to define sphere betweeen",
    type=str,
    nargs="+",
    default=None,
)
parser.add_argument(
    "--sph_resis",
    help="Resids of residues to define sphere betweeen",
    type=str,
    nargs="+",
    default=None,
)
parser.add_argument("--rad", help="GCMC Sphere radius in Angstroms", type=float)
parser.add_argument("--hmr", help="Use HMR?", action="store_true")
parser.add_argument(
    "--nmoves", help="Number of production GCNCMC moves", type=int, default=2000
)
parser.add_argument(
    "--mdpermove",
    help="Number of MD steps to perform between each GCNCMC move",
    type=int,
    default=5000,
)
args = parser.parse_args()

print(f"Using HMR: {args.hmr}")

# Basic System Configuration
nonbondedMethod = PME
nonbondedCutoff = 1.2 * nanometers
switchDistance = 10.0 * angstroms
ewaldErrorTolerance = 0.0005  # Default
constraints = HBonds
rigidWater = True  # Default
hydrogenMass = 1.0 * amu

# Integration Options
dt = 0.002 * picoseconds
temperature = 298 * kelvin
friction = 1.0 / picosecond

# HMR?
if args.hmr:
    dt = 0.004 * picoseconds
    hydrogenMass = 2.0 * amu

# GCNCMC params
mu_ex = args.mu * kilocalories_per_mole
conc = args.conc * molar
v_per_lig = grand.utils.convert_conc_to_volume(conc)

switching_time = args.st * picoseconds
n_prop = 50
n_pert = int((switching_time / (n_prop * dt)) - 1)
print(
    f"Running at a switching time of {switching_time}. n_pert = {n_pert}, n_prop = {n_prop}"
)

n_gcncmc_moves = args.nmoves
md_per_move = args.mdpermove

# GCMC Sphere
s_resns = args.sph_resns
s_resis = args.sph_resis
if len(s_resns) != len(s_resis):
    raise Exception("Length of sphere resnames must be the same as resids")

ref_atoms = []
for i in range(len(s_resns)):
    ref_atoms.append(
        {"name": "CA", "resname": f"{s_resns[i]}", "resid": f"{s_resis[i]}"}
    )

sphere_rad = args.rad * angstroms

# Input Files
protein_file = args.pdb
lig_file = args.lig
xml_file = args.xml
pdb = PDBFile(protein_file)

# Get platform
platform = Platform.getPlatformByName("CUDA")
platformProperties = {"Precision": "mixed"}
platform.setPropertyDefaultValue("Precision", "mixed")
print(f"Running on {platform.getName()} platform.")

# Prepare Simulation and add ghost molecules for GCMC.
pdb.topology, pdb.positions, ghosts = grand.utils.add_ghosts(
    pdb.topology, pdb.positions, molfile=lig_file, n=50, pdb="Protein_Ghosts.pdb"
)

# Use the ghost resids to get the ligand resname
for residue in pdb.topology.residues():
    if residue.index == ghosts[0]:
        resname = residue.name


# Load force field and create system
ff = ForceField("amber14-all.xml", "amber14/tip3p.xml", xml_file)
system = ff.createSystem(
    pdb.topology,
    nonbondedMethod=nonbondedMethod,
    nonbondedCutoff=nonbondedCutoff,
    switchDistance=switchDistance,
    constraints=constraints,
    rigidWater=rigidWater,
    hydrogenMass=hydrogenMass,
)

# Langevin Integrator
print(f"Setting Integrator Step size to {dt}")
integrator = BAOABIntegrator(
    temperature, friction, dt
)  # 0.002 ps because we want to equil at this


# Set up NCMC Sphere sampler
print("Setting up sphere sampler object...")
ncmc_mover = grand.samplers.NonequilibriumGCMCSphereSampler(
    system=system,
    topology=pdb.topology,
    temperature=temperature,
    integrator=integrator,
    excessChemicalPotential=mu_ex,
    adams=None,
    standardVolume=v_per_lig,
    referenceAtoms=ref_atoms,
    sphereRadius=sphere_rad,
    nPertSteps=n_pert,
    nPropStepsPerPert=n_prop,
    timeStep=dt,
    resname=resname,
    ghostFile="ncmc-ghost-ligs.txt",
    log="gcncmc.log",
    rst="checkpoint.rst7",
    dcd="gcncmc_raw.dcd",
    overwrite=True,
    recordTraj=False,
    maxN=999,
)

# Set up simulation object
simulation = Simulation(
    pdb.topology,
    system,
    ncmc_mover.integrator,
    platform,
    platformProperties=platformProperties,
)
simulation.context.setPositions(pdb.positions)
simulation.context.setVelocitiesToTemperature(298 * kelvin)
simulation.context.setPeriodicBoxVectors(*pdb.topology.getPeriodicBoxVectors())


simulation.reporters.append(
    StateDataReporter(
        "simulation_log.txt",
        1000,
        step=True,
        time=True,
        potentialEnergy=True,
        temperature=True,
        volume=True,
    )
)

# Switch off ghosts
ncmc_mover.initialise(simulation.context, simulation, ghosts)
ncmc_mover.deleteMoleculesInGCMCSphere()  # If there are any 'on' molecules in the sphere e.g. in the original PDB file, these will be removed.

print("Minimising...")
simulation.minimizeEnergy(
    tolerance=0.0001 * kilojoule / mole, maxIterations=10000
)  # Quick Minimisation


print("Equilibration...")
# Run some MD for equil
equil_time = 2 * nanosecond
equil_md_steps = math.ceil((equil_time) / dt)

print(f"Simulating for {equil_md_steps} MD equil steps at {dt} ({equil_time})")
simulation.step(equil_md_steps)

print("Running GCNCMC Equilibration")
for i in range(150):
    simulation.step(1000)
    ncmc_mover.move(simulation.context, 1)


print(
    "{}/{} equilibration NCMC moves accepted. N = {}".format(
        ncmc_mover.n_accepted, ncmc_mover.n_moves, ncmc_mover.N
    )
)

print(f"Simulating for {equil_md_steps / 2} MD equil steps")
simulation.step(int(equil_md_steps / 2))

ncmc_mover.reset()  # Reset stats
print("\nProduction....")

print("Running NCMC...")
for i in range(n_gcncmc_moves):
    simulation.step(md_per_move)
    ncmc_mover.move(simulation.context, 1)
    ncmc_mover.report(simulation, data=True)


# Setup the output files
# Move ghost waters out of the simulation cell
trj = grand.utils.shift_ghost_waters(
    ghost_file="ncmc-ghost-ligs.txt",
    topology="Protein_Ghosts.pdb",
    trajectory="gcncmc_raw.dcd",
)

# Recentre the trajectory on a particular residue
# trj = grand.utils.recentre_traj(t=trj, resname='ILE', resid=221)

# Align the trajectory to the protein
grand.utils.align_traj(t=trj, output="gcncmc.dcd", reference="Protein_Ghosts.pdb")

# Write out a trajectory of the GCMC sphere
grand.utils.write_sphere_traj(
    radius=sphere_rad,
    ref_atoms=ref_atoms,
    topology="Protein_Ghosts.pdb",
    trajectory="gcncmc.dcd",
    output="gcmc_sphere.pdb",
    initial_frame=True,
)
