#!/usr/bin/env python3


import numpy as np
from desc.equilibrium import Equilibrium
from desc.grid import LinearGrid
from matplotlib import pyplot as plt
import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.ticker as ticker
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter as sf

#eq = Equilibrium.load("eq_initial_high-res_m2_NFP5.h5")
#eq = Equilibrium.load("eq_optimized_high-res_m2_NFP5.h5")

#eq = Equilibrium.load("eq_omni_m2_NFP5_14.h5")

eq0 = Equilibrium.load("eq_initial_high-res_m2_NFP5.h5")
eq1 = Equilibrium.load("eq_final_high-res.h5")

L = 100
rho = np.linspace(0, 1, L+1)
grid0 = LinearGrid(L=L)


data0 = eq0.compute(["iota current", "iota vacuum"], grid=grid0)
iota_current0 = data0["iota current"]
iota_vacuum0 = data0["iota vacuum"]

data1 = eq1.compute(["iota current", "iota vacuum"], grid=grid0)
iota_current1 = data1["iota current"]
iota_vacuum1 = data1["iota vacuum"]



plt.figure(figsize=(6, 5))
plt.plot(rho, iota_vacuum0/(iota_current0+iota_vacuum0),'r', linewidth=2.5, label='initial')
plt.plot(rho, iota_vacuum1/(iota_current1+iota_vacuum1),'g', linewidth=2.5, label='optimized')
plt.legend(fontsize=22)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)


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
plt.ylabel(r'$\iota_{\mathrm{shaping}}/\iota$',fontsize=30, labelpad=0)
plt.tight_layout()  # Ensure everything fits well
plt.savefig("iota_fraction_UT225.pdf", dpi=400)
plt.show()








