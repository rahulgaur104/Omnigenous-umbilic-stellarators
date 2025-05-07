#!/usr/bin/env python3

import pdb
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker

from desc.equilibrium import Equilibrium
from desc.grid import LinearGrid, Grid

from desc.geometry import FourierUmbilicCurve, FourierRZCurve

from scipy.interpolate import griddata

from desc.plotting import *
from desc.backend import *

#from plotly.io import write_html

eq0 = Equilibrium.load("eq_initial_high-res_m2_NFP5.h5")
eq1 = Equilibrium.load("eq_omni_m2_NFP5_14.h5")

NFP = eq1.NFP

legend_list = ["initial", "optimized"]
eq_list = [eq0, eq1]
scale_list = [2, 2]

keyword = "OP"

eq = eq1
legend = "optimized"
scale = 2

#for eq, legend, scale in zip(eq_list, legend_list, scale_list):
plt.figure()
theta_grid = np.linspace(0, 2 * np.pi, 300)
zeta_grid = np.linspace(0, 2 * np.pi, 300)
grid = LinearGrid(rho=1.0, theta=theta_grid, zeta=zeta_grid)
fig = plot_3d(eq, name="curvature_k2_rho", grid=grid)

# fig.update_xaxes(showgrid=True, gridwidth=2, gridcolor='lightgray', linewidth=2, linecolor='black')
# fig.update_yaxes(showgrid=True, gridwidth=2, gridcolor='lightgray', linewidth=2, linecolor='black')
# fig.update_layout(coloraxis_colorbar=dict(len=0.8, thickness=20))
fig.update_traces(
    colorscale='Plasma',
    colorbar=dict(
        tickfont=dict(size=58),  # Adjust the size value as needed
        title_font=dict(size=58),  # Adjust the size value as needed
        nticks=6
    )
)

fig.update_layout(font=dict(size=18, color="black", family="Arial, sans-serif"))

fig.data[0].update(
    lighting=dict(
        ambient=0.85,        # Ambient light - overall brightness
        diffuse=1.0,        # Diffuse light - light scattered evenly
        roughness=0.5,      # Surface roughness - lower is more reflective
        fresnel=0.5,        # Fresnel effect - edge highlighting
        specular=0.6,       # Specular light - concentrated reflection
        facenormalsepsilon=1e-6,  # For face normals calculation
        vertexnormalsepsilon=1e-6  # For vertex normals calculation
    )
)



m = 2
NFP_umbilic_factor = 5
n = NFP_umbilic_factor
nphi = 300

curve0 = FourierUmbilicCurve.load("curve_omni_m2_NFP5_14.h5")
phi_arr1 = np.linspace(0, 2 * np.pi * NFP_umbilic_factor, nphi)
phi1 = phi_arr1.flatten()
data_curve0 = curve0.compute(["UC"], grid = LinearGrid(zeta = phi1, NFP=1, NFP_umbilic_factor=n))
theta1 = np.mod((1*data_curve0["UC"] - m * NFP * phi1)/n, 2*np.pi)

custom_grid = Grid(jnp.array([jnp.ones_like(phi1), theta1, phi1]).T)
curve_data = eq1.compute(["R", "Z"], grid=custom_grid)
R1 = curve_data["R"]
Z1 = curve_data["Z"]
data_curve1 = np.zeros((len(phi1), 3))

arr1 = np.array([R1, phi1, Z1]).T
data_curve1[:, :] = arr1 

curve2 = FourierRZCurve.from_values(coords=jnp.array(data_curve1), N=15, NFP=1)

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
    width=0.5,
    dash="solid",
    ),
    showlegend=False,
)




config = {
    "toImageButtonOptions": {
        "filename": f"modB_3d_{keyword}_{legend}",
        "format": "svg",
        "scale": scale,
    }
}



save_path_html = os.getcwd() + f"/curvature_3d_{keyword}_{legend}.html"
fig.write_html(
    save_path_html, config=config, include_plotlyjs=True, full_html=True
)
fig.show()
# save_path_png = os.getcwd() + f"/3D_modB/modB_3d_{keyword}_{legend}.png"
# fig.write_image(save_path_png, scale=scale)
#plt.close()
