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

data_keys = ["p", "iota", "iota vacuum", "iota current"]

eq0 = Equilibrium.load("eq_final3_3kA.h5")
eq = Equilibrium.load("eq_final3_3kA_re-solved.h5")

data0 = eq0.compute(data_keys, grid=radial_grid)
data = eq.compute(data_keys, grid=radial_grid)

plt.figure()
#plt.plot(rho, np.abs(data0["iota"]), "-r", linewidth=3)
plt.plot(rho, np.abs(data["iota"]), "-k", linewidth=3)
plt.plot(rho, np.abs(data["iota vacuum"]), "-r", linewidth=3)
plt.plot(rho, np.abs(data["iota current"]), "-b", linewidth=3)


plt.yscale("log")

#plt.plot(rho, 1/np.abs(data0["iota"]), "-r", linewidth=3)
#plt.plot(rho, 1/np.abs(data["iota"]), "-k", linewidth=3)

#plt.plot(rho, 1/np.abs(data0["iota"])-1/np.abs(data["iota"]), "-r", linewidth=3)
#plt.plot(rho, 1/np.abs(data0["iota"])-1/np.abs(data["iota"]), "-k", linewidth=3)

plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.xlabel(r"$\rho$", fontsize=26)
plt.ylabel(r"$\iota$", fontsize=26)
plt.legend(["total", "shaping", "current"], fontsize=24)
plt.tight_layout()
# plt.savefig(f"input_profiles/{keyword}_iota_profile.png", dpi=400)
#plt.savefig(f"{keyword}_iota_profile.pdf", dpi=400)
plt.savefig(f"{keyword}_iota_profile_comparison.png", dpi=300)
plt.show()
plt.close()
