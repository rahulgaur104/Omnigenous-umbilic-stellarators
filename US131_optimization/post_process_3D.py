#!/usr/bin/env python3


import numpy as np

from desc.plotting import *
from matplotlib import pyplot as plt

#from desc.geometry import FourierUmbilicCurve

from desc.equilibrium import Equilibrium
from desc.grid import LinearGrid, Grid

from desc.backend import *
#from desc.curve import Curve
#eq_new = Equilibrium.load("eq_ripple_NFP3_10.h5")
#eq_new = Equilibrium.load("eq_final_high-res.h5")
eq_new = Equilibrium.load("eq_high-res_optimized.h5")
curve_opt = FourierUmbilicCurve.load("curve_ripple_NFP3_10.h5")

fig, ax = plot_section(eq_new, name="|F|", norm_F=True, log=True)
plt.show()


NFP_umbilic_factor = int(3)
restart_idx = int(0)
m = 1

nphi =int(200)
phi1 = np.linspace(0, 2 * np.pi * NFP_umbilic_factor, nphi)
curve_grid = LinearGrid(zeta = phi1, NFP_umbilic_factor=NFP_umbilic_factor)

# Plotting optimized eq + curve combo
#fig = plot_3d(eq_new,"|B|")
fig = plot_3d(eq_new,"curvature_k2_rho")
phi_arr1 = np.linspace(0, 2 * np.pi * NFP_umbilic_factor, nphi)
phi1 = phi_arr1.flatten()

data_curve_opt = curve_opt.compute(["UC"], grid = LinearGrid(zeta = phi1, NFP_umbilic_factor=NFP_umbilic_factor), override_grid=False)
theta1 = (data_curve_opt["UC"] - m * phi1)/NFP_umbilic_factor
custom_grid = Grid(jnp.array([jnp.ones_like(phi1), theta1, phi1]).T)
curve_data = eq_new.compute(["R", "Z"], grid=custom_grid)
R1 = curve_data["R"]
Z1 = curve_data["Z"]
data_curve_opt1 = np.zeros((len(phi1), 3))

arr1 = np.array([R1, phi1, Z1]).T
data_curve_opt1[:, :] = arr1

fig.add_scatter3d(
x=R1*np.cos(phi1),
y=R1*np.sin(phi1),
z=Z1,
marker=dict(
size=0,
opacity=0,
),
line=dict(
color="black",
width=0.1,
dash="solid",
),
showlegend=False,)

fig.write_html(f"test.html")
plt.show()
plt.close()
