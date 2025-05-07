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

eq0 = Equilibrium.load("beak_equilibrium_A8_beta1p0_2100.0_reducedFalse.h5")
eq1 = Equilibrium.load("beak_equilibrium_A8_beta1p0_2100.0_increased4100_fixcur.h5")
eq2 = Equilibrium.load("beak_equilibrium_A8_beta1p0_2100.0_increased4100_fixcur.h5")
eq3 = Equilibrium.load("beak_equilibrium_A8_beta1p0_2100.0_increased6100_fixcur.h5")
eq4 = Equilibrium.load("beak_equilibrium_A8_beta1p0_2100.0_increased6100_fixcur.h5")


L = 10
rho = np.linspace(0, 1, L+1)
grid0 = LinearGrid(L=L)


data0 = eq0.compute(["iota current", "iota vacuum"], grid=grid0)
iota_current0 = data0["iota current"][-1]
iota_vacuum0 = data0["iota vacuum"][-1]
frac0 = iota_vacuum0/(iota_current0+iota_vacuum0)

data1 = eq1.compute(["iota current", "iota vacuum"], grid=grid0)
iota_current1 = data1["iota current"][-1]
iota_vacuum1 = data1["iota vacuum"][-1]
frac1 = iota_vacuum1/(iota_current1+iota_vacuum1)

data2 = eq2.compute(["iota current", "iota vacuum"], grid=grid0)
iota_current2 = data2["iota current"][-1]
iota_vacuum2 = data2["iota vacuum"][-1]
frac2 = iota_vacuum2/(iota_current2+iota_vacuum2)

data3 = eq3.compute(["iota current", "iota vacuum"], grid=grid0)
iota_current3 = data3["iota current"][-1]
iota_vacuum3 = data3["iota vacuum"][-1]
frac3 = iota_vacuum3/(iota_current3+iota_vacuum3)

data4 = eq4.compute(["iota current", "iota vacuum"], grid=grid0)
iota_current4 = data4["iota current"][-1]
iota_vacuum4 = data4["iota vacuum"][-1]
frac4 = iota_vacuum4/(iota_current4+iota_vacuum4)


#x_array = np.array([2100, 3100, 4100, 5100, 6100])
x_array = np.array([2100, 4100, 6100])
y_array = np.array([frac0, frac2, frac4])

plt.figure(figsize=(6, 5))
plt.plot(x_array, y_array, '-ok', ms=8, linewidth=3)
#plt.legend(fontsize=22)
plt.xticks(x_array, fontsize=24)
# Replace the current plt.xticks line with these lines:
#plt.xticks(x_array, ['2.1', '3.1', '4.1',  '6.1'], fontsize=24)

plt.yticks(np.linspace(0., 0.08, 5), fontsize=24)


# Get the current axes and increase the minor tick size
ax = plt.gca()

# Increase minor tick width and length
ax.tick_params(which='minor', width=1., length=6)  # Increase minor tick size
ax.tick_params(which='major', width=2, length=10)   # Also adjust major ticks for consistency

# Make minor ticks more visible (especially important for log scale)
ax.yaxis.set_minor_formatter(ticker.LogFormatter(labelOnlyBase=False))

# Optional: Adjust tick padding if needed
ax.tick_params(which='both', pad=8)

plt.xlabel(r'$I_{\mathrm{umbilic}}(A)$',fontsize=30)
#ax = plt.gca()
#ax.xaxis.set_label_text(r'$I_{\mathrm{umbilic}}(A) \times 10^3$', fontsize=30)
plt.ylabel(r'$\iota_{\mathrm{shaping}}/\iota$',fontsize=30, labelpad=0)
plt.tight_layout()  # Ensure everything fits well
plt.savefig("iota_fraction_w_umbilic_current.pdf", dpi=400)
plt.show()








