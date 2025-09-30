#!/usr/bin/env python3


import numpy as np

from desc.plotting import *
from matplotlib import pyplot as plt

from desc.geometry import FourierUmbilicCurve

from desc.equilibrium import Equilibrium
from desc.grid import LinearGrid, Grid

from desc.backend import *

#from desc.curve import Curve
#eq_new = Equilibrium.load("eq_ripple_NFP3_10.h5")
#eq_new = Equilibrium.load("eq_final_high-res.h5")
eq_new = Equilibrium.load("eq_high-res_optimized.h5")
#eq_new = Equilibrium.load("eq_limiota_m1_n3_L14_M14_N14_QA_init.h5")
curve_opt = FourierUmbilicCurve.load("curve_ripple_NFP3_10.h5")

#fig, ax = plot_section(eq_new, name="|F|", norm_F=True, log=True)
#plt.show()


NFP_umbilic_factor = int(3)
restart_idx = int(0)
m = 1

nphi =int(200)
phi1 = np.linspace(0, 2 * np.pi * NFP_umbilic_factor, nphi)
curve_grid = LinearGrid(zeta = phi1, NFP_umbilic_factor=NFP_umbilic_factor)


grid0 = LinearGrid(M = 50, N=50)

# Plotting optimized eq + curve combo
#fig = plot_3d(eq_new, "|B|", grid=grid0)
fig = plot_3d(eq_new, "curvature_k2_rho", grid=grid0)

fig.update_traces(
    colorbar=dict(
        tickfont=dict(size=58),  # Adjust the size value as needed
        title_font=dict(size=58),  # Adjust the size value as needed
    )
)

#fig = plot_3d(eq_new,"curvature_k2_rho")
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
size=0.8,
opacity=1,
),
line=dict(
color="black",
width=5,
dash="solid",
),
showlegend=False,)



config = {
    "toImageButtonOptions": {
        "filename": f"modB_3d_optimized",
        "format": "svg",
        "scale": 2,
    }
}


#fig.write_html(f"test_op.html")
fig.write_html(
    #"test_op.html", config=config, include_plotlyjs=True, full_html=True
    "test_in.html", config=config, include_plotlyjs=True, full_html=True
)
plt.show()
plt.close()
