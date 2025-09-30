#!/usr/bin/env python3
import os
import pdb
import sys
import numpy as np
from desc.equilibrium import Equilibrium
import jax.numpy as jnp
from desc.grid import Grid, LinearGrid
from matplotlib import pyplot as plt

from matplotlib.ticker import LogLocator, ScalarFormatter

mu0 = 4 * np.pi * 1e-7

comparison = True

keyword = "HBT-EP"


len0 = int(200)
radial_grid = LinearGrid(L=len0)
rho = np.linspace(0, 1, len0 + 1)

data_keys = ["p", "iota"]

eq = Equilibrium.load("beak_equilibrium_A8_beta1p0_2100.0_reducedFalse.h5")
data = eq.compute(data_keys, grid=radial_grid)

pressure = data["p"]/ 10**6
plt.plot(rho, pressure, "-k", linewidth=3)
plt.xticks(fontsize=22)
plt.yticks(np.linspace(0.0, 0.0035, 6), fontsize=22)
plt.xlabel(r"$\rho$", fontsize=26)
plt.ylabel(r"$p (\mathrm{MPa})$", fontsize=26)
plt.tight_layout()
# plt.savefig(f"input_profiles/{keyword}_pressure_profile.png", dpi=400)
#plt.savefig(f"{keyword}_pressure_profile.pdf", dpi=400)
plt.savefig(f"{keyword}_pressure_profile.png", dpi=300)
plt.close()

plt.figure()
plt.plot(rho, np.abs(data["iota"]), "-k", linewidth=3)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.xlabel(r"$\rho$", fontsize=26)
plt.ylabel(r"$\iota$", fontsize=26)
plt.tight_layout()
# plt.savefig(f"input_profiles/{keyword}_iota_profile.png", dpi=400)
#plt.savefig(f"{keyword}_iota_profile.pdf", dpi=400)
plt.savefig(f"{keyword}_iota_profile.png", dpi=300)
plt.close()
