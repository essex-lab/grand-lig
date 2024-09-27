# Modules to import
import argparse
import os

import scipy.stats
from simtk.unit import *
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from scipy.optimize import curve_fit
from openmmtools.constants import STANDARD_STATE_VOLUME
from tqdm import tqdm


small_font = 16
medium_font = 18
large_font = 20

plt.rc("figure", titlesize=large_font)
plt.rc("font", size=small_font)
plt.rc("axes", titlesize=small_font)
plt.rc("axes", labelsize=medium_font)
plt.rc("xtick", labelsize=small_font)
plt.rc("ytick", labelsize=small_font)
plt.rc("legend", fontsize=small_font)


def calc_VGCMC(radius):
    V = 4 / 3 * np.pi * radius**3
    return V


def calc_LigVol(conc):
    V_l = 1 / (AVOGADRO_CONSTANT_NA * conc)
    return V_l.in_units_of(angstroms**3)


def calc_c_from_B(B, HFE, sphere_rad):
    c = np.exp(B - (beta * HFE)) / (AVOGADRO_CONSTANT_NA * calc_VGCMC(sphere_rad))
    return c.in_units_of(molar)


def calcB(HFE, sphere_rad, V_L):
    test = calc_VGCMC(sphere_rad) / (V_L)
    B = (beta * HFE) + np.log(test)
    return B


def sigmoid(x, x0, k):
    """
    1 / 1 + exp(-k(B-betadF))
    Parameters
    ----------
    x
    x0
    k

    Returns
    -------

    """
    y = 1 / (1 + np.exp(-k * (x - x0)))
    return y


def inverse_sigmoid(Y, x0, k):
    return (-np.log((1 - Y) / Y) / k) + x0


def bootstrap(n, occupancies, Bs):
    booted_params = []
    for i in tqdm(range(n)):
        B_sample, occ_sample = [], []
        for j, B in enumerate(Bs):
            B_sample.append(B)
            rand_data_set = np.random.randint(0, len(occupancies))
            occ_sample.append(occupancies[rand_data_set][j])
        try:  # Add in initial fits and get x number of fits per bootstrap and take the best?
            initial_guess = [np.median(Bs), 1]
            popt, pcov = curve_fit(
                sigmoid, Bs, occ_sample, p0=initial_guess, maxfev=10000
            )
            booted_params.append(popt)
        except:
            continue
    return booted_params


T = 298 * kelvin
kT = BOLTZMANN_CONSTANT_kB * T * AVOGADRO_CONSTANT_NA
beta = 1 / kT

Bs = [
    -21.31,
    -19.01,
    -16.71,
    -14.4,
    -13.7,
    -12.99,
    -12.28,
    -11.57,
    -10.86,
    -10.15,
    -9.44,
    -8.74,
    -8.03,
    -7.32,
    -6.61,
    -5.9,
    -5.19,
    -2.89,
    -0.59,
    1.71,
]

# INPUT OCCUPANCIES HERE / ADAPT SCRIPT - List of lists required. [n_repeats, N_bs]
occs = [
    [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.019,
        0.003,
        0.008,
        0.009,
        0.014,
        0.02,
        0.15,
        0.0,
        0.316,
        0.341,
        0.939,
        0.726,
        1.0,
        1.0,
        1.0,
    ],
    [
        0.0,
        0.0,
        0.0,
        0.0,
        0.001,
        0.0,
        0.0,
        0.004,
        0.001,
        0.003,
        0.022,
        0.042,
        0.319,
        0.268,
        0.364,
        0.609,
        0.965,
        1.0,
        1.0,
        1.0,
    ],
    [
        0.0,
        0.0,
        0.0,
        0.0,
        0.001,
        0.005,
        0.024,
        0.0,
        0.1,
        0.014,
        0.038,
        0.405,
        0.41,
        0.837,
        0.56,
        0.609,
        0.979,
        0.998,
        1.0,
        1.0,
    ],
    [
        0.0,
        0.0,
        0.0,
        0.0,
        0.001,
        0.005,
        0.024,
        0.053,
        0.1,
        0.014,
        0.038,
        0.405,
        0.41,
        0.837,
        0.56,
        0.609,
        0.979,
        0.998,
        1.0,
        1.0,
    ],
]

all_Bs = np.concatenate((Bs, np.tile(Bs, len(occs) - 1)))
all_occs = []
for repeat in occs:
    for i in repeat:
        all_occs.append(i)

tau = scipy.stats.kendalltau(all_Bs, all_occs, nan_policy="omit")[0]

params = bootstrap(1000, occs, Bs)
params = np.array(params)
N_fit_mean = np.mean(params, axis=0)

x = np.linspace(min(Bs), max(Bs), 100)

N_fit = []
betadfs = []
for set in params:
    N_fit.append(sigmoid(x, *set))
    beta_df = inverse_sigmoid(0.5, *set)
    betadfs.append(beta_df)

betadF_trans = N_fit_mean[0]  # Also B50
dF_trans = (betadF_trans / beta).in_units_of(kilocalories_per_mole)

std_B50s = np.std(betadfs)
dF_trans_err = ((std_B50s) / beta).in_units_of(kilocalories_per_mole)



print(
    N_fit_mean, betadF_trans, (betadF_trans / beta).in_units_of(kilocalories_per_mole)
)


fig = plt.figure(figsize=(20, 10))

ax1 = fig.add_subplot(111)
plt.xlabel("Adams Value (B)")
plt.ylabel("Site Occupancy")

colors = {
    "Crimson": "d7263d",
    "Brandy": "914542",
    "Portland Orange": "f46036",
    "Space Cadet": "2e294e",
    "Persian Green": "1b998b",
    "Malachite": "04e762",
    "June Bud": "c5d86d",
}
colors = list(colors.values())

y = sigmoid(x, *N_fit_mean)
y_plus = sigmoid(x, *N_fit_mean) + np.std(N_fit, axis=0)
y_minus = sigmoid(x, *N_fit_mean) - np.std(N_fit, axis=0)

ax1.plot(x, sigmoid(x, *N_fit_mean), "-", color="#" + colors[-2])
ax1.fill_between(x, y_minus, y_plus, alpha=0.5, lw=0, color="#000080")

ax1.plot(
    all_Bs,
    all_occs,
    marker="x",
    linestyle="None",
    c="#" + colors[0],
    label="Raw data " + r"($\tau={:.3f}$)".format(tau),
)
x50 = [betadF_trans, betadF_trans]
y50 = [0, 0.5]
ax1.plot(x50, y50, linestyle="--", c="k")
text_for_graph = (
    r"$B_{{50}} = \beta \Delta F_{{trans}} = {:.2f} \pm {:.2f}$".format(
        betadF_trans, std_B50s
    )
    + "\n"
    + r"$\Delta F_{{trans}} = {:.2f} \pm {:.2f}\ kcal\ mol^{{-1}}$".format(
        dF_trans._value, dF_trans_err._value
    )
)
ax1.text(betadF_trans - 15, np.median(y50), text_for_graph)

ax1.axhline(0.5)
plt.legend(loc="upper left")

plt.show()
