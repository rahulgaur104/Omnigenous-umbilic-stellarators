#!/usr/bin/env python3

import pdb
import numpy as np

from desc.equilibrium import Equilibrium
from desc.grid import LinearGrid

from desc.plotting import *
import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.ticker as ticker
from matplotlib import pyplot as plt


# --------------------------------------------------------------------
# Force Matplotlib to use its default sans-serif font (DejaVu Sans).
# Also configure math text to be sans-serif, and disable LaTeX engine.
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["DejaVu Sans"]
mpl.rcParams["mathtext.fontset"] = "dejavusans"
mpl.rcParams["text.usetex"] = False

eq_new0 = Equilibrium.load("eq_high-res_initial.h5")
eq_new1 = Equilibrium.load("eq_high-res_optimized.h5")

L0=100
rho = np.linspace(0, 1, L0+1)
grid0 = LinearGrid(L=L0)
grid1 = LinearGrid(L=L0)

iota0 = eq_new0.compute("iota", grid=grid0)["iota"]
iota1 = eq_new1.compute("iota", grid=grid1)["iota"]

plt.figure(figsize=(6, 5))

plt.plot(rho, np.abs(iota0), 'r', linewidth=2, label='initial')
plt.plot(rho, np.abs(iota1), 'g', linewidth=2, label='optimized')

plt.axhline(y=0.3333, xmin=0, xmax=0.98, color='k', linestyle='--', linewidth=2, label=r'$\iota = 1/3$')

plt.legend(fontsize=22)
plt.xticks(np.linspace(0., 1.00, 6), fontsize=28)
plt.yticks(np.linspace(0., 0.35, 6), fontsize=27)

plt.xlim([0., 1.0])
plt.ylim([0., 0.35])

# Control how many decimal places are shown on x/y ticks (e.g., 2 decimals)
ax = plt.gca()
ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))

#plt.axis('equal')
#plt.yscale("log")
plt.xlabel(r'$\rho$',fontsize=32)
plt.ylabel(r'$\iota$',fontsize=32)

#plt.savefig("iota_comparison.pdf", dpi=400)
plt.savefig("iota_comparison.svg", dpi=400)
#plt.show()



