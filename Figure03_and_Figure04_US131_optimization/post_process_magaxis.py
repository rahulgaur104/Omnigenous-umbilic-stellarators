#!/usr/bin/env python3


import numpy as np

from desc.plotting import *
from matplotlib import pyplot as plt

from desc.equilibrium import Equilibrium
from desc.grid import LinearGrid, Grid

from desc.backend import *

eq_new = Equilibrium.load("eq_high-res_optimized.h5")


nphi = 300
phi = np.linspace(0, 2*np.pi, nphi)
grid0 = LinearGrid(rho=np.array([0.01]), M=1, zeta=phi)


# Plotting optimized eq + curve combo
#fig = plot_3d(eq_new,"|B|")
fig = plot_3d(eq_new,"curvature_k2_rho", grid=grid0)


grid1 = LinearGrid(rho=np.array([0.01]), M=0, zeta=phi)
curve_data = eq_new.compute(["R", "Z"], grid=grid1)

R = curve_data["R"]
Z = curve_data["Z"]
data_curve_opt1 = np.zeros((len(phi), 3))

arr1 = np.array([R, phi, Z]).T

fig.add_scatter3d(
x=R*np.cos(phi),
y=R*np.sin(phi),
z=Z,
marker=dict(
size=0,
opacity=0,
),
line=dict(
color="black",
width=10,
dash="solid",
),
showlegend=False,)

# Update layout to remove grid and colorbar
fig.update_layout(
    scene=dict(
        xaxis=dict(showbackground=False, showgrid=False, visible=False),
        yaxis=dict(showbackground=False, showgrid=False, visible=False),
        zaxis=dict(showbackground=False, showgrid=False, visible=False),
    ),
    coloraxis_showscale=False  # This removes the colorbar
)



fig.show()

#fig.write_html(f"test.html")
#plt.show()
#plt.close()


