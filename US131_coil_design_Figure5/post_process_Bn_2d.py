#!/usr/bin/env python3

import os
import pdb
import numpy as np
from desc.grid import LinearGrid

from desc.equilibrium import Equilibrium
from desc.coils import CoilSet

from desc.plotting import *

from matplotlib import pyplot as plt
from matplotlib import ticker
import matplotlib as mpl

import plotly.graph_objects as go

from scipy.signal import find_peaks

eq = Equilibrium.load("eq_high-res_optimized.h5")
field = CoilSet.load("optimized_coilset9.h5")

#fig = plot_3d(eq,"|B|")


coil_grid = LinearGrid(N=60)
plot_grid = LinearGrid(M=92, N=92, NFP=1, endpoint=True)

fig, ax, data1 = plot_2d(
    eq.surface,
    "B*n",
    field=field,
    field_grid=coil_grid, 
    grid=plot_grid,
    return_data=True
)
plt.close()


Ntheta = 600
Nzeta = 600
zeta = np.linspace(0, 2*np.pi, Nzeta)
theta = np.linspace(0, 2*np.pi, Ntheta)

grid0 = LinearGrid(rho=np.array([1.]), theta=theta, zeta=zeta)
grid_array = np.zeros((Ntheta, Nzeta))
list0 = [] 

curvature = np.array(eq.compute("curvature_k2_rho", grid=grid0)["curvature_k2_rho"])
curvature = np.reshape(curvature, (Nzeta, Ntheta))

offset = int(Ntheta/3)
for i in range(Nzeta):
    curvature1 = np.concatenate((curvature[i], curvature[i][:offset]))
    curvature1[curvature1 >= -25] = 0
    min_curvature_idxs = np.unique(np.mod(find_peaks(curvature1)[0], Ntheta))
    list0.append(theta[min_curvature_idxs])

plt.figure(figsize=(7, 6))
#plt.plot(zeta, np.sort(np.array(list0), axis=0))
plt.plot(zeta, 2*np.pi-1*np.sort(np.array(list0), axis=0)[:, 0], '--k', linewidth=3)
plt.plot(zeta, 2*np.pi-1*np.sort(np.array(list0), axis=0)[:, 1], '--k', linewidth=3)
plt.plot(zeta, 2*np.pi-1*np.sort(np.array(list0), axis=0)[:, 2], '--k', linewidth=3, label=r"field line")
#plt.show()


contour = plt.contourf(data1["zeta"], data1["theta"], data1["B*n"], cmap="coolwarm", levels=96, vmin=-0.012, vmax=0.012)
ax = plt.gca()
ax.set_aspect("equal", adjustable="box")


## Adding a colorbar
#cbar = fig.colorbar(contour, ax=ax, orientation="vertical")
#tick_locator = ticker.MaxNLocator(nbins=7)
#cbar.locator = tick_locator
#cbar.update_ticks()
#cbar.ax.tick_params(labelsize=18)  # Change colorbar tick size

# Set explicit tick locations and labels for x and y axes (multiples of π)
# For x-axis (zeta)
x_ticks = [0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi]
x_labels = ['0', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$']
ax.set_xticks(x_ticks)
ax.set_xticklabels(x_labels, fontsize=38)

# For y-axis (theta)
y_ticks = [0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi]
y_labels = ['0', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$']
ax.set_yticks(y_ticks)
ax.set_yticklabels(y_labels, fontsize=38)



plt.xlabel(r"$\zeta$", fontsize=46)
plt.ylabel(r"$\theta$", fontsize=46, labelpad=-3)

plt.tight_layout()

plt.savefig("US131_Bn_umbilic_curve.svg")


#plt.show()
#pdb.set_trace()




