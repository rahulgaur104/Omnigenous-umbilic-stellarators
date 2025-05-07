#!/usr/bin/env python3

import numpy as np
from desc.grid import LinearGrid
from desc.equilibrium import Equilibrium
from desc.plotting import *
from matplotlib import pyplot as plt
from matplotlib import ticker
import matplotlib as mpl
from scipy.signal import find_peaks
import pdb


# --------------------------------------------------------------------
# Force Matplotlib to use its default sans-serif font (DejaVu Sans).
# Also configure math text to be sans-serif, and disable LaTeX engine.
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["DejaVu Sans"]
mpl.rcParams["mathtext.fontset"] = "dejavusans"
mpl.rcParams["text.usetex"] = False

eq = Equilibrium.load("eq_final_high-res.h5")

Ntheta = 800
Nzeta = 600
zeta = np.linspace(0, 2*np.pi, Nzeta)
theta = np.linspace(0, 2*np.pi, Ntheta)

grid0 = LinearGrid(rho=np.array([1.]), theta=theta, zeta=zeta, sym=False)
grid_array = np.zeros((Ntheta, Nzeta))
list0 = [] 

curvature = np.array(eq.compute("curvature_k2_rho", grid=grid0)["curvature_k2_rho"])
curvature = np.reshape(curvature, (Nzeta, Ntheta))

offset = int(Ntheta/5)
for i in range(Nzeta):
    curvature1 = np.concatenate((curvature[i], curvature[i][:offset]))
    curvature1[curvature1 >= -20] = 0
    min_curvature_idxs = np.unique(np.mod(find_peaks(curvature1)[0], Ntheta))
    #print(min_curvature_idxs)
    list0.append(theta[min_curvature_idxs].tolist())

plt.figure(figsize=(5, 4))
#plt.plot(zeta, np.sort(np.array(list0), axis=0))
plt.plot(zeta, np.array(list0), '-k', linewidth=3)
#plt.plot(zeta+2*np.pi, np.sort(np.array(list0), axis=0)[:, 1], '-k', linewidth=3)
#plt.plot(zeta+4*np.pi, np.sort(np.array(list0), axis=0)[:, 2], '-k', linewidth=3)
#plt.plot(zeta+6*np.pi, np.sort(np.array(list0), axis=0)[:, 3], '-k', linewidth=3)
#plt.plot(zeta+8*np.pi, np.sort(np.array(list0), axis=0)[:, 4], '-k', linewidth=3, label=r"field line")
#pdb.set_trace()

#plt.plot(zeta, np.sort(np.array(list0), axis=0)[:, 0], '-k', linewidth=3)
#plt.plot(zeta+2*np.pi, np.sort(np.array(list0), axis=0)[:, 1], '-k', linewidth=3)
#plt.plot(zeta+4*np.pi, np.sort(np.array(list0), axis=0)[:, 2], '-k', linewidth=3)
#plt.plot(zeta+6*np.pi, np.sort(np.array(list0), axis=0)[:, 3], '-k', linewidth=3)
#plt.plot(zeta+8*np.pi, np.sort(np.array(list0), axis=0)[:, 4], '-k', linewidth=3, label=r"field line")
##plt.show()

iota = eq.compute("iota", grid=LinearGrid(rho=np.array([1.])))["iota"]

zeta_full = np.linspace(0, 2*np.pi*3, Nzeta)
grid1 = eq._get_rtz_grid(np.array([1.]), np.array([0.]), zeta_full, coordinates="raz",
                        period=(np.inf, 2*np.pi, np.inf), iota=iota)


ax = plt.gca()
#plt.plot(grid1.nodes[:, 2], -grid1.nodes[:, 1], '--', "#ff39ffff", linewidth=3, ms=2, label="umbilic edge")
plt.plot(np.mod(grid1.nodes[:, 2], 2*np.pi), grid1.nodes[:, 1], '--', color="purple", linewidth=3, ms=2, label="umbilic edge")
##plt.plot(np.mod(grid1.nodes[:, 2], 2*np.pi), grid1.nodes[:, 1], 'or', ms=2)

plt.xlabel(r"$\zeta$", fontsize=28, labelpad=-2)
plt.ylabel(r"$\theta$", fontsize=28, labelpad=-2)

## Set explicit tick locations and labels for x and y axes (multiples of π)
## For x-axis (zeta)
#x_ticks = [0, 2*np.pi, 4*np.pi, 6*np.pi]
#x_labels = ['0', r'$2\pi$', r'$4\pi$', r'$6\pi$']
#ax.set_xticks(x_ticks)
#ax.set_xticklabels(x_labels, fontsize=26)
#
#y_ticks = [0, 2*np.pi/3, 4*np.pi/3, 2*np.pi]
#y_labels = ['0', r'$2\pi/3$', r'$4\pi/3$', r'$2\pi$']
#ax.set_yticks(y_ticks)
#ax.set_yticklabels(y_labels, fontsize=26)
#
### Control how many decimal places are shown on x/y ticks (e.g., 2 decimals)
##ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
##ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
##
plt.xlim([0, 10*np.pi])
plt.ylim([0, 2*np.pi])

plt.grid()

from matplotlib.lines import Line2D
#legend_elements = [
#    Line2D([0], [0], color='k', lw=4, label='Field line'),
#    Line2D([0], [0], color='b', lw=4, label='Umbilic edge')
#]
#legend = plt.legend(handles=legend_elements, fontsize=20, loc='lower right')
legend = plt.legend(fontsize=20, loc='best')
legend.get_frame().set_alpha(0.5)  # Change 0.5 to any value between 0 and 1
plt.tight_layout()
plt.savefig("fieldline_umbilicedge.svg", dpi=400)
plt.show()
