import numpy as np
import matplotlib.pyplot as plt
import h5py
import pdb

from desc.equilibrium import Equilibrium
from desc.grid import LinearGrid, Grid

from desc.plotting import *
from matplotlib import pyplot as plt
import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.ticker as ticker
from desc.coils import CoilSet


# --------------------------------------------------------------------
# Force Matplotlib to use its default sans-serif font (DejaVu Sans).
# Also configure math text to be sans-serif, and disable LaTeX engine.
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["DejaVu Sans"]
mpl.rcParams["mathtext.fontset"] = "dejavusans"
mpl.rcParams["text.usetex"] = False


### plotting the free=boundary solution
eq0 = Equilibrium.load("beak_equilibrium_A8_beta1p0_2100.0_increased6100_fixcur.h5")

grid0 = LinearGrid(L=100)
print(eq0.compute("iota current", grid=grid0)["iota current"])
phi = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
fig, ax, data0 = plot_boundary(eq=eq0, phi=phi, return_data = True)
plt.close()



#for i in range(4):
i = 0
#plt.plot(data0["R"][:, 1, i], data0["Z"][:, 1, i], '-r', linewidth=2)
num_flux_surfs = 10
theta0 = np.linspace(0, 2*np.pi, 200)
for j in range(num_flux_surfs):
    grid2 = LinearGrid(rho = np.array([(1/num_flux_surfs)*(j+1)]), theta=theta0, zeta=phi[i])
    data2 = eq0.compute(["R","Z"],  grid=grid2)
    if j == num_flux_surfs-1:
        plt.plot(data2["R"], data2["Z"], '-r', linewidth=2)
    else:
        plt.plot(data2["R"], data2["Z"], '-r', linewidth=1)


# plotting the umbilic coil location
field = CoilSet.load("../medium_beak_optimized_coilset_2p100kA_N12_mindist0p020_PFcurrent_59kA_shorterTF.h5")[3]
field.current = 6100
grid0 = LinearGrid(zeta=phi[i])
data_curve = field.compute(["R", "Z"], grid=grid0)
R_coil = data_curve["R"]
Z_coil = data_curve["Z"]
plt.plot(R_coil, Z_coil, 'ok', ms=8)

plt.xlim([0.75, 1.25])
#plt.xlim([0.75, 1.1])
#plt.ylim([-0.20, 0.20])
plt.ylim([-0.20, 0.25])

plt.xticks(np.linspace(0.75, 1.25, 5), fontsize=20)
#plt.xticks(np.linspace(0.75, 1.25, 5), fontsize=18)
plt.yticks(np.linspace(-0.2, 0.2, 5), fontsize=20)

# Control how many decimal places are shown on x/y ticks (e.g., 2 decimals)
ax = plt.gca()
ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))

plt.xlabel("R", fontsize=24)
plt.ylabel("Z", fontsize=24)

plt.savefig("poincare-cross-section_6100.svg", dpi=200)
#plt.savefig("poincare-cross-section_0.eps", dpi=600)
plt.show()
