#!/usr/bin/env python3
import pdb
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import matplotlib.colors as colors
import matplotlib.ticker as ticker

from desc.equilibrium import Equilibrium
from desc.grid import LinearGrid
from desc.plotting import *

#mpl.rcParams['font.family'] = 'sans-serif'
#mpl.rcParams['font.sans-serif'] = ['DejaVu Sans']

# --------------------------------------------------------------------
# Force Matplotlib to use its default sans-serif font (DejaVu Sans).
# Also configure math text to be sans-serif, and disable LaTeX engine.
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["DejaVu Sans"]
mpl.rcParams["mathtext.fontset"] = "dejavusans"
mpl.rcParams["text.usetex"] = False



minor_radius = 0.1
r1 = 0.133
r2 = 0.01
r3 = 0.0
r4 = 0.01
r5 = 0.1
r6 = 0
r7 = -0.00


# Define the parametrization of the umbilic torus
ntheta = int(300) 
nphi = int(300)

phi = 0.0
t = np.linspace(0, 2 * np.pi, ntheta)

NFP = int(1) # For the surface parametrization, we don't use umbilic
n = int(3)
m = int(1)

# First, we parametrize the surface.
R =  1 + 2*minor_radius*np.cos(np.pi/(2*n)) * np.cos((t + np.pi*(2*np.floor(n*t/(2*np.pi)) + 1)/n)/2 + (m*phi + r1*np.sin(phi) + r4*np.sin(2*phi))/n ) - minor_radius*np.cos(np.pi*(2*np.floor(n*t/(2*np.pi)) + 1)/n + (m*phi + r1*np.sin(phi) + r4*np.sin(2*phi))/n) + r2*np.cos(phi) + r7*np.cos(2*phi)
Z = 2*minor_radius*np.cos(np.pi/(2*n)) * np.sin((t + np.pi*(2*np.floor(n*t/(2*np.pi)) + 1)/n)/2 + (m*phi + r5*np.sin(phi))/n) - minor_radius*np.sin(np.pi*(2*np.floor(n*t/(2*np.pi)) + 1)/n + (m*phi + r5*np.sin(phi))/n) + r3*np.sin(2*phi) + r6*np.sin(phi)

Rb_UToL = R 
Zb_UToL = Z


eq_new0 = Equilibrium.load("eq_limiota_m1_n3_L14_M14_N14_QA_init.h5")

# Use plot_section once and close its default figure to get "data"
fig_tmp, ax_tmp, data = plot_section(eq_new0, name="|F|", norm_F=True, log=True, return_data=True)
plt.close(fig_tmp)  # We don't want that figure, just the data

# Compute the boundary in R, Z
grid0 = LinearGrid(rho=np.array([1.0]), theta=np.linspace(0, 2*np.pi, 200), zeta=np.array([0.]))
data_keys_bdry = ["R", "Z"]
data_bdry = eq_new0.compute(data_keys_bdry, grid=grid0)
Rb = data_bdry["R"]
Zb = data_bdry["Z"]


# Suppose data["|F|"] is shape (n_theta, n_zeta), i.e. 2D. We'll call it fvals here:
fvals = np.abs(data["|F|"][:, :, 0])  # if the last dimension is for zeta, pick index 0


# Create figure
plt.figure(figsize=(11, 11))

# Choose min/max for the color scale. Adjust to suit your data range.
vmin = 1e-5  # e.g. smallest |F|
vmax = 1e-1   # e.g. largest |F|

# Choose discrete contour levels in powers of ten
# e.g. 1e-4, 1e-3, 1e-2, 1e-1, 1e0
levels = np.logspace(-5, -1, 5)

# Contourf the data in log scale
cf = plt.contourf(
    data["R"][:, :, 0],  # X-coords
    data["Z"][:, :, 0],  # Y-coords
    fvals,               # values of |F|
    levels=levels,
    norm=colors.LogNorm(vmin=vmin, vmax=vmax),
    cmap="hot"
)

# Plot boundary in solid black
plt.plot(Rb, Zb, "k-", linewidth=2, label=r"$\mathrm{DESC}$" +" boundary")

# Plot Umbilic Torus shape in thick dotted black
plt.plot(Rb_UToL, Zb_UToL, "b--", linewidth=4, label="UToL boundary")

plt.xlabel("R", fontsize=46)
plt.ylabel("Z", fontsize=46)

plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off

#plt.tick_params(
#    axis='y',          # changes apply to the x-axis
#    which='both',      # both major and minor ticks are affected
#    bottom=False,      # ticks along the bottom edge are off
#    top=False,         # ticks along the top edge are off
#    labelbottom=False) # labels along the bottom edge are off

plt.axis('off')


plt.axis("equal")
legend = plt.legend(fontsize=30, loc='upper right')
legend.get_frame().set_alpha(0.2)

plt.xlim([1.10, 1.11])
plt.ylim([-0.01, 0.01])

plt.tight_layout()

#plt.savefig("normF_UToL131.png", dpi=400)
#plt.savefig("normF_UToL131.svg", dpi=400)
plt.savefig("normF_UToL131_zoom.svg", dpi=400)
#plt.savefig("normF_UToL131.svg", dpi=400)
plt.show()
plt.close()

#plt.show()



