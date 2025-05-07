#!/usr/bin/env python3

import pdb
import numpy as np

from desc.coils import CoilSet, FourierPlanarCoil, FourierRZCoil
from desc.geometry import FourierUmbilicCurve, FourierRZCurve

from desc.equilibrium import Equilibrium

from desc.grid import LinearGrid, Grid
from desc.plotting import *
from desc.backend import *

from matplotlib import pyplot as plt

m = 1
NFP_umbilic_factor = 1
n = NFP_umbilic_factor
nphi = 100

eq_new = Equilibrium.load("beak_equilibrium_A8_beta1p0_2100.0_reducedFalse.h5")
curve_opt = curve_opt = FourierUmbilicCurve.load("beak_equilibrium_umbilic_curve_A8_beta1p0.h5")
#phi_arr1 = np.linspace(0, 2*np.pi*1, nphi)
phi_arr1 = np.linspace(0, 2 * np.pi * NFP_umbilic_factor, 4*nphi)
grid_new = eq_new.get_rtz_grid(np.array([1.]), np.array([0.]), phi_arr1, coordinates="raz", period=(np.inf, 2*np.pi, np.inf), iota = eq_new.compute("iota", grid=LinearGrid(rho=np.linspace(0, 1,2)))["iota"][-1], )

fig = plot_3d(eq_new,"curvature_k2_rho")
phi1 = phi_arr1.flatten()
data_curve_opt0 = curve_opt.compute(["UC"], grid = LinearGrid(zeta = phi1, NFP=m, NFP_umbilic_factor=NFP_umbilic_factor), override_grid=False)
theta1 = (data_curve_opt0["UC"] - m*phi1)/NFP_umbilic_factor
custom_grid = Grid(jnp.array([jnp.ones_like(phi1), theta1, phi1]).T)
curve_data = eq_new.compute(["R", "Z"], grid=custom_grid)
R1 = curve_data["R"]
Z1 = curve_data["Z"]
data_curve_opt1 = np.zeros((len(phi1), 3))

arr1 = np.array([R1, phi1, Z1]).T
data_curve_opt1[:, :] = arr1

curve_opt1 = FourierRZCurve.from_values(coords=jnp.array(data_curve_opt1), N=25, NFP=m)
grid_nodes = np.array(grid_new.nodes)

custom_grid2 = Grid(grid_nodes)
curve_data2 = eq_new.compute(["R", "Z"], grid=custom_grid2)
R2 = curve_data2["R"]
Z2 = curve_data2["Z"]
data_curve_opt2 = np.zeros((len(phi1), 3))

arr2 = np.array([R2, phi1, Z2]).T
data_curve_opt2[:, :] = arr2


#curve_opt1 = FourierRZCurve.from_values(coords=jnp.array(data_curve_opt1), N=25, NFP=1)
#curve_opt2 = FourierRZCurve.from_values(coords=jnp.array(data_curve_opt2), N=25, NFP=1)
####plot_coils(curve_opt1, grid=custom_grid)
###plot_coils(curve_opt1, fig=fig,grid=custom_grid)
###plot_coils(curve_opt2, fig=fig,grid=custom_grid, **{"color":"green"})
#
##fig = plot_coils(curve_opt1, fig=fig, grid=custom_grid,  **{"color":"red"})
#fig = plot_coils(curve_opt1,  grid=custom_grid,  **{"color":"red"})
#
#plot_coils(curve_opt2, fig=fig, grid=custom_grid2, **{"color":"blue"})


plt.figure(dpi=300)
plt.plot(phi1, grid_nodes[:, 1], 'or', ms=1.5, label="fieldline")
plt.plot(phi1, np.mod(theta1, 2*np.pi),'ob', ms=0.5, label="umbilic_curve")
#plt.plot(phi1, np.mod(m/n*phi1[::-1], 2*np.pi),'og', ms=0.5, label="straight line")
plt.ylabel(r"$\theta_{\mathrm{DESC}}$", fontsize=20)
plt.xlabel(r"$\zeta$", fontsize=20)
plt.legend()
plt.title("deviation of the field line from the umbilic curve")
plt.show()


