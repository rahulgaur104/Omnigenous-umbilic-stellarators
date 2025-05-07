"""
Plot cross sections and magnetic axis position
"""
import pdb
import numpy as np

#from desc.geometry import FourierUmbilicCurve
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


from desc.plotting import *

#eq0 = Equilibrium.load("beak_equilibrium_A8_beta1p0_2100.0_reducedFalse.h5")
#eq0 = Equilibrium.load("beak_equilibrium_A8_beta1p0_6_umbilic-8100_fixcur.h5")
#eq0 = Equilibrium.load("beak_equilibrium_A8_beta1p0_6_umbilic-16100_fixcur.h5")
#eq1 = Equilibrium.load("beak_equilibrium_A8_beta1p0_6_umbilic-16100_fixcur.h5")

#eq0 = Equilibrium.load("beak_equilibrium_A8_beta1p0_2100.0_increased4100_fixcur.h5")
#eq1 = Equilibrium.load("beak_equilibrium_A8_beta1p0_2100.0_increased4100_fixcur.h5")

#eq0 = Equilibrium.load("beak_equilibrium_A8_beta1p0_2100.0_increased6100_fixcur.h5")
#eq1 = Equilibrium.load("beak_equilibrium_A8_beta1p0_2100.0_increased6100_fixcur.h5")

eq0 = Equilibrium.load("beak_equilibrium_A8_beta1p0_4_umbilic-8100_fixcur.h5")
eq1 = Equilibrium.load("beak_equilibrium_A8_beta1p0_4_umbilic-8100_fixcur.h5")

#fig, ax = plot_section(eq0, name="|F|", norm_F=True, log=False)

#field = CoilSet.load("medium_beak_optimized_coilset_2p100kA_N12_mindist0p021_PFcurrent_59kA_shorterTF.h5")[3]
#field = CoilSet.load("medium_beak_optimized_coilset_2p100kA_N12_mindist0p020_PFcurrent_59kA_shorterTF.h5")[3]
field = CoilSet.load("beak_beta1p0_coilset_reversed_current4.h5")[3]


fig, ax = plot_boozer_surface(eq0)
plt.show()

grid0 = LinearGrid(L=100)
print(eq0.compute("iota current", grid=grid0)["iota current"])
print(eq0.compute("iota vacuum", grid=grid0)["iota vacuum"])
#fig, ax  = plot_comparison(eqs=[eq1])

phi = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
fig, ax, data0  = plot_boundary(eq=eq0, phi=phi, return_data=True)
plt.close()
fig, ax, data1 = plot_boundary(eq=eq1, phi=phi, return_data = True)
plt.close()


grid0 = LinearGrid(zeta=phi)
data_curve = field.compute(["R", "Z"], grid=grid0)
R_coil = data_curve["R"]
Z_coil = data_curve["Z"]

linecolor = ['-r', '-g', '-b', '-k']
#labels = [r'$\phi = 0$', r'$\phi = \pi/2$', r'$\phi = \pi$', r'$\phi = 3 \pi/2$']
labels = [r'$\zeta = 0$', r'$\zeta = \pi/2$', r'$\zeta = \pi$', r'$\zeta = 3 \pi/2$']

for i in range(4):
    plt.plot(data0["R"][:, 1, i], data0["Z"][:, 1, i], linecolor[i], linewidth=2, label=labels[i])


x_min = np.min(data0["R"][:, 1, :].flatten())
x_max = np.max(data0["R"][:, 1, :].flatten())

y_min = np.min(data0["Z"][:, 1, :].flatten())
y_max = np.max(data0["Z"][:, 1, :].flatten())

plt.xlim([x_min*0.92, x_max*1.15])
plt.ylim([y_min*1.8, y_max*1.8])

plt.xticks(np.linspace(x_min*0.92, x_max*1.15, 4), fontsize=18)
plt.yticks(np.linspace(y_min*1.8, y_max*1.8, 5), fontsize=18)

# Control how many decimal places are shown on x/y ticks (e.g., 2 decimals)
ax = plt.gca()
ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))

# Add a legend
#legend = plt.legend(fontsize=18, loc='lower left', bbox_to_anchor=(-0.05, -0.1))
legend = plt.legend(fontsize=14, loc='upper center', bbox_to_anchor=(0.8, 1.1), handlelength=0.8)
legend.get_frame().set_alpha(0.3)  # Change 0.5 to any value between 0 and 1
#legend = plt.legend(fontsize=18, loc='upper left')

plt.xlabel("R", fontsize=22)
plt.ylabel("Z", fontsize=22)

##plt.figure(figsize=(5, 5))
#linecolor = ['--r', '--g', '--b', '--k']
#
#for i in range(4):
#    plt.plot(data1["R"][:, 1, i], data1["Z"][:, 1, i], linecolor[i], linewidth=0.5, label=labels[i])
#
##x_min = np.min(data0["R"][:, 1, :].flatten())
##x_max = np.max(data0["R"][:, 1, :].flatten())
##
##y_min = np.min(data0["Z"][:, 1, :].flatten())
##y_max = np.max(data0["Z"][:, 1, :].flatten())
#
##plt.xlim([x_min*0.98, x_max*1.02])
##plt.ylim([y_min*1.02, y_max*1.02])
#
##plt.xticks(np.linspace(x_min, x_max, 4), fontsize=22)
##plt.yticks(np.linspace(y_min, y_max, 5), fontsize=22)
#
##plt.xticks(np.linspace(x_min, x_max, 4), fontsize=18)
##plt.yticks(np.linspace(y_min, y_max, 5), fontsize=18)
#
## Control how many decimal places are shown on x/y ticks (e.g., 2 decimals)
#ax = plt.gca()
#ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
#ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
#
## Add a legend
#legend = plt.legend(fontsize=16, loc='best')
#legend.get_frame().set_alpha(0.5)  # Change 0.5 to any value between 0 and 1
#
#plt.xlabel("R", fontsize=26)
#plt.ylabel("Z", fontsize=26)

plt.plot(R_coil, Z_coil, 'xk', ms=8)

# Get the current axis
ax = plt.gca()

# Adjust the axis to maintain equal aspect ratio but fill the figure box
ax.set_aspect("equal", adjustable="box")

#plt.savefig("Xsection_optimized_reverse.svg", dpi=400)
#plt.savefig("Xsection_optimized.pdf", dpi=400)
plt.show()
