#!/usr/bin/env python3

import pdb
import numpy as np

#from desc.geometry import FourierUmbilicCurve
from desc.equilibrium import Equilibrium
from desc.grid import LinearGrid, Grid

from desc.plotting import *
from matplotlib import pyplot as plt
import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.ticker as ticker


# --------------------------------------------------------------------
# Force Matplotlib to use its default sans-serif font (DejaVu Sans).
# Also configure math text to be sans-serif, and disable LaTeX engine.
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["DejaVu Sans"]
mpl.rcParams["mathtext.fontset"] = "dejavusans"
mpl.rcParams["text.usetex"] = False


eq_new0 = Equilibrium.load("eq_initial_high-res_m2_NFP5.h5")
eq_new1 = Equilibrium.load("eq_omni_m2_NFP5_14.h5")



phi = np.array([0, np.pi/2*1/eq_new0.NFP, np.pi*1/eq_new0.NFP, 3*np.pi/2*1/eq_new0.NFP])

fig, ax, data0 = plot_boundary(eq_new0, phi=phi, return_data = True)
plt.close()
fig, ax, data1 = plot_boundary(eq_new1, phi=phi, return_data = True)
plt.close()

linecolor = ['-r', '-g', '-b', '-k']
#labels = [r'$\phi = 0$', r'$\phi = \pi/2$', r'$\phi = \pi$', r'$\phi = 3 \pi/2$']
labels = [r'$\phi = 0$', r'$\phi = \pi/4$', r'$\phi = \pi/2$', r'$\phi = 3 \pi/4$']
#labels = [r'$\zeta = 0$', r'$\zeta = \pi/2$', r'$\zeta = \pi$', r'$\zeta = 3 \pi/2$']



theta0 = np.linspace(0, 2*np.pi, 100)
zeta0 = np.linspace(0, 2*np.pi, 100)

num_flux_surfs = 7

for i in range(4):
    for j in range(num_flux_surfs):
        grid2 = LinearGrid(rho = np.array([(1/num_flux_surfs)*(j+1)]), theta=theta0, zeta=phi[i])
        data2 = eq_new1.compute(["R","Z"],  grid=grid2)
        if j == num_flux_surfs-1:
            plt.plot(data2["R"], data2["Z"], linecolor[i], linewidth=2, label=labels[i])
        else:
            plt.plot(data2["R"], data2["Z"], linecolor[i], linewidth=0.5)


x_min = np.min(data1["R"][:, 1, :].flatten())
x_max = np.max(data1["R"][:, 1, :].flatten())

y_min = np.min(data1["Z"][:, 1, :].flatten())
y_max = np.max(data1["Z"][:, 1, :].flatten())

plt.xlim([x_min*0.98, x_max*1.02])
plt.ylim([y_min*1.02, y_max*1.02])

plt.xticks(np.linspace(x_min, x_max, 4), fontsize=22)
plt.yticks(np.linspace(y_min, y_max, 5), fontsize=22)

# Control how many decimal places are shown on x/y ticks (e.g., 2 decimals)
ax = plt.gca()
ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))

# Add a legend
#legend = plt.legend(fontsize=18, loc='lower left', bbox_to_anchor=(-0.05, -0.1))
legend = plt.legend(ncol=2, fontsize=14, loc='upper center', bbox_to_anchor=(0.7, 1.1), handlelength=1.5)
#legend = plt.legend(fontsize=18, loc='upper left')
legend.get_frame().set_alpha(0.3)  # Change 0.5 to any value between 0 and 1
# Now, customize the line thickness in the legend
for handle in legend.legend_handles:
    handle.set_linewidth(4.0)  # Set the line thickness (default is usually 1-2)


plt.xlabel("R", fontsize=26)
plt.ylabel("Z", fontsize=26)

## Get the current axis
#ax = plt.gca()
#
## Adjust the axis to maintain equal aspect ratio but fill the figure box
#ax.set_aspect("equal", adjustable="box")
#
#plt.tight_layout()
#plt.savefig("boundary_plots/boundary_optimized.pdf", dpi=300)
#plt.close()
#

##plt.figure(figsize=(5, 5))
#linecolor = ['--r', '--g', '--b', '--k']
#
#for i in range(4):
#    plt.plot(data0["R"][:, 1, i], data0["Z"][:, 1, i], linecolor[i], linewidth=2, label=labels[i])
#
##x_min = np.min(data0["R"][:, 1, :].flatten())
##x_max = np.max(data0["R"][:, 1, :].flatten())
##
##y_min = np.min(data0["Z"][:, 1, :].flatten())
##y_max = np.max(data0["Z"][:, 1, :].flatten())

plt.xlim([x_min*0.98, x_max*1.02])
plt.ylim([y_min*1.02, y_max*1.02])

#plt.xticks(np.linspace(x_min, x_max, 4), fontsize=22)
#plt.yticks(np.linspace(y_min, y_max, 5), fontsize=22)

plt.xticks(np.linspace(x_min, x_max, 4), fontsize=18)
plt.yticks(np.linspace(y_min, y_max, 5), fontsize=18)

# Control how many decimal places are shown on x/y ticks (e.g., 2 decimals)
ax = plt.gca()
ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))

## Add a legend
#legend = plt.legend(fontsize=16, loc='best')
#legend.get_frame().set_alpha(0.5)  # Change 0.5 to any value between 0 and 1

#plt.xlabel("R", fontsize=26)
#plt.ylabel("Z", fontsize=26)

plt.xlabel("R", fontsize=19)
plt.ylabel("Z", fontsize=19)

# Get the current axis
ax = plt.gca()

# Adjust the axis to maintain equal aspect ratio but fill the figure box
ax.set_aspect("equal", adjustable="box")

plt.tight_layout()
#plt.savefig("boundary_plots/boundary_initial.pdf", dpi=300)
#plt.savefig("boundary_plots/boundary_optimized_225.png", dpi=300)
plt.savefig("boundary_plots/boundary_optimized_225.pdf", dpi=300)
plt.show()
plt.close()





