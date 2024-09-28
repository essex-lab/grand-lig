import pandas as pd
import argparse
from pymbar.timeseries import statistical_inefficiency, subsample_correlated_data, detect_equilibration
from pymbar import MBAR
import numpy as np
import os
from openmm.unit import *

def calc_mu_ex():
    WORK_DIR = os.getcwd()
    ligand = id
    #print(WORK_DIR)
    #print(ligand)
    # Load first np array to get info from
    
    u_ = np.load(f"lambda_0/U_matrix_0.npy")
    n_lambdas, _, n_samples = u_.shape
    lambdas = np.linspace(0, 1, n_lambdas)


    kT = AVOGADRO_CONSTANT_NA * BOLTZMANN_CONSTANT_kB * 298 * kelvin  # Define kT
    kT_kcal = kT.in_units_of(kilocalories_per_mole)

    dg_repeats = []
    u_kln_list = []
    
    U_kln = np.zeros((n_lambdas, n_lambdas, n_samples))
    for i in range(0, n_lambdas):  # Load in all the data to the big U_kln array
        npy_array = f'lambda_{i}/U_matrix_{i}.npy'
        if not os.path.isfile(npy_array):
            raise Exception(f"Cannot find file: {npy_array}")
        u_kln_ = np.load(npy_array)
        U_kln += u_kln_

    np.save("Final_U_kln.npy", U_kln)
    u_kln_list.append(U_kln)
    # Perform pymbar analysis
    N_k = np.zeros(n_lambdas, np.int32)
    for i in range(n_lambdas):
        A_t = U_kln[i, i, :]
        n_equil, g, neff_max = detect_equilibration(A_t, nskip=1)  # Detect equilibrated states for this lambda
        indicies = subsample_correlated_data(A_t, g=g)  # identify uncorrelated data
        if len(indicies) < 500:
            print(i, len(indicies))
        N_k[i] = len(indicies)

        U_kln[i, :, 0:N_k[i]] = U_kln[i, :, indicies].T

    mbar = MBAR(U_kln, N_k, maximum_iterations=100000)
    [deltaG_ij, ddeltaG_ij, theta_ij] = mbar.compute_free_energy_differences(return_theta=True).values()

    for i in range(n_lambdas):
        dG_i = (deltaG_ij[0, i] * kT_kcal).in_units_of(kilocalorie_per_mole)
        print('Free energy ({:.3f} -> {:.3f}) = {}'.format(lambdas[0], lambdas[i], dG_i))

    dg = deltaG_ij[0, -1] * kT_kcal
    dg_repeats.append(dg._value)
    os.chdir(WORK_DIR)

    mean_dg = np.mean(dg_repeats)
    std_error = np.std(dg_repeats) / np.sqrt(len(dg_repeats))
    print(f"{ligand}: {mean_dg:.3f} +/- {std_error:.3f}")
    return mean_dg, std_error


# get working dir
wd = os.getcwd()
print(wd)



mu, err = calc_mu_ex()

print(f"Final free energy estimate: {mu} +/- {err}")

