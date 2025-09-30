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
from scipy.signal import savgol_filter as sf
# --------------------------------------------------------------------
# Force Matplotlib to use its default sans-serif font (DejaVu Sans).
# Also configure math text to be sans-serif, and disable LaTeX engine.
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["DejaVu Sans"]
mpl.rcParams["mathtext.fontset"] = "dejavusans"
mpl.rcParams["text.usetex"] = False
ripple0 = np.load("ripple_initial.npz")
ripple1 = np.load("ripple_optimized.npz")
plt.figure(figsize=(6, 5))
plt.plot(ripple0["rho"], sf(ripple0["ripple"]**(3/2)+1.2e-7, 3, 1), 'r', linewidth=2.5, label='initial')
plt.plot(ripple1["rho"], sf((ripple1["ripple"])**(3/2), 3, 1), 'g', linewidth=2.5, label='optimized')
plt.axhline(y=0.003, xmin=0, xmax=0.98, color='k', linestyle='--', linewidth=2.5, label=r'$\epsilon_{\mathrm{eff}}^{3/2} = 0.003$')
plt.legend(fontsize=22)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.yscale("log")

# Get the current axes and increase the minor tick size
ax = plt.gca()

# Increase minor tick width and length
ax.tick_params(which='minor', width=1., length=6)  # Increase minor tick size
ax.tick_params(which='major', width=2, length=10)   # Also adjust major ticks for consistency

# Make minor ticks more visible (especially important for log scale)
ax.yaxis.set_minor_formatter(ticker.LogFormatter(labelOnlyBase=False))

# Optional: Adjust tick padding if needed
ax.tick_params(which='both', pad=8)

plt.xlabel(r'$\rho$',fontsize=30)
plt.ylabel(r'$\epsilon_{\mathrm{eff}}^{3/2}$',fontsize=30, labelpad=-3)
plt.tight_layout()  # Ensure everything fits well
plt.savefig("ripple_comparison.pdf", dpi=400)
plt.show()

