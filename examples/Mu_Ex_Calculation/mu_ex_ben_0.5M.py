import grandlig as grand
from openmm.app import *
from openmm import *
from openmm.unit import *
from sys import stdout
import numpy as np
import openmmtools
import pandas as pd

pdb = PDBFile("0.5M_Benzene.pdb")

# Set MD and GCNCMC variables
nonbondedMethod = PME
nonbondedCutoff = 1.2 * nanometers
switchDistance = 10.0 * angstroms
ewaldErrorTolerance = 0.0005  # Default
constraints = HBonds
rigidWater = True  # Default

ff = ForceField("amber14/tip3p.xml", "Benzene.xml")

system = ff.createSystem(
    pdb.topology,
    nonbondedMethod=nonbondedMethod,
    nonbondedCutoff=nonbondedCutoff,
    switchDistance=switchDistance,
    constraints=constraints,
    rigidWater=rigidWater,
)

list_of_resis = (
    []
)  # Get a list of resids so we can choose at= random one to decouple
resname = "L01"

for residue in pdb.topology.residues():
    if residue.name == resname:
        list_of_resis.append(residue.index)

resid = np.random.choice(list_of_resis)

mu_ex = grand.potential.calc_mu_ex(
    system=system,
    topology=pdb.topology,
    positions=pdb.positions,
    resname="L01",
    resid=resid,
    box_vectors=pdb.topology.getPeriodicBoxVectors(),
    temperature=298*kelvin,
    pressure=1 * bar,
    n_lambdas=20,
    n_samples=5000,
    n_equil=400,
    log_file="dg.txt",
    turnOff=True
)


