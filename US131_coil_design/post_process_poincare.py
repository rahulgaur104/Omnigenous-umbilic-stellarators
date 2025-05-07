#!/usr/bin/env python3
import pdb
import numpy as np
from desc.equilibrium import Equilibrium
from desc.grid import LinearGrid

import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.ticker as ticker
from matplotlib import pyplot as plt


# --------------------------------------------------------------------
# Force Matplotlib to use its default sans-serif font (DejaVu Sans).
# Also configure math text to be sans-serif, and disable LaTeX engine.
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["DejaVu Sans"]
mpl.rcParams["mathtext.fontset"] = "dejavusans"
mpl.rcParams["text.usetex"] = False

eq = Equilibrium.load("eq_high-res_optimized.h5")

data0 = np.load("data_co.npy", allow_pickle=True)
data1 = np.load("data_counter.npy", allow_pickle=True)



rho_array = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
N_rho = len(rho_array)

N_theta = 100
N_zeta = 1

phi_poincare = np.linspace(0, 2*np.pi, 6, endpoint=False)

section_idx = 1
phi_section = phi_poincare[section_idx]
print("phi = ",phi_section)

grid = LinearGrid(rho=rho_array, theta=np.linspace(0, 2*np.pi, N_theta), zeta=np.linspace(phi_section, 2*np.pi, N_zeta))

data_out = eq.compute(["R", "Z"], grid=grid)


R = np.reshape(data_out["R"], (N_rho, N_theta))
Z = np.reshape(data_out["Z"], (N_rho, N_theta))

fig = plt.figure(figsize=(10, 10))

for i in range(N_rho):
    plt.plot(R[i], Z[i], '-r', linewidth=4)


plt.plot(data0.item()["R"][::, section_idx, 1:16], data0.item()["Z"][::, section_idx, 1:16], '.k', ms=1.0)
plt.plot(data1.item()["R"][::, section_idx, 1:16], data1.item()["Z"][::, section_idx, 1:16], '.k', ms=1.0)
#plt.plot(data0.item()["R"][::, section_idx, 1:16]+0.0012, data0.item()["Z"][::, section_idx, 1:16], '.k', ms=1.0)
#plt.plot(data1.item()["R"][::, section_idx, 1:16]+0.0012, data1.item()["Z"][::, section_idx, 1:16], '.k', ms=1.0)

Rb = R[-1]
Zb = Z[-1]

from matplotlib.path import Path


# Create a Path object from the boundary coordinates
boundary = Path(np.column_stack((Rb, Zb)))

# Process each index separately (avoiding arrays for marker size)
inside_indices = []
outside_indices = []

for i in range(49):
    R_point = data1.item()['R'][0, section_idx, 16+i]
    Z_point = data1.item()['Z'][0, section_idx, 16+i]
    
    # Test if the point is inside the boundary
    is_inside = boundary.contains_point((R_point, Z_point))
    
    if is_inside:
        inside_indices.append(16+i)
    else:
        outside_indices.append(16+i)

# Plot with different sizes based on indices
if inside_indices:
    plt.plot(data0.item()["R"][::, section_idx, inside_indices]+0.0012, 
             data0.item()["Z"][::, section_idx, inside_indices], 
             '.k', ms=0.5)
    plt.plot(data1.item()["R"][::, section_idx, inside_indices]+0.0012, 
             data1.item()["Z"][::, section_idx, inside_indices], 
             '.k', ms=0.5)

if outside_indices:
    plt.plot(data0.item()["R"][::, section_idx, outside_indices]+0.0012, 
             data0.item()["Z"][::, section_idx, outside_indices], 
             '.k', ms=3.0)
    plt.plot(data1.item()["R"][::, section_idx, outside_indices]+0.0012, 
             data1.item()["Z"][::, section_idx, outside_indices], 
             '.k', ms=3.0)


#plt.ylim([-0.14, 0.14])
#plt.xlim([1.20, 1.45])

ax = plt.gca()
ax.set_aspect("equal", adjustable="box")

## Add a legend
##legend = plt.legend(fontsize=18, loc='lower left', bbox_to_anchor=(-0.05, -0.1))
#legend = plt.legend(fontsize=14, loc='upper center', bbox_to_anchor=(0.8, 1.1), handlelength=0.8)
#legend.get_frame().set_alpha(0.3)  # Change 0.5 to any value between 0 and 1
##legend = plt.legend(fontsize=18, loc='upper left')

plt.xticks(np.linspace(1.20, 1.45, 5), fontsize=38)
plt.yticks(np.linspace(-0.14, 0.14, 5), fontsize=38)

# Control how many decimal places are shown on x/y ticks (e.g., 2 decimals)
ax = plt.gca()
ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))


plt.xlabel("R", fontsize=48)
plt.ylabel("Z", fontsize=48, labelpad=-3)

plt.tight_layout()

plt.savefig(f"US131_poincare_trace_section_{section_idx}.svg")
plt.show()

