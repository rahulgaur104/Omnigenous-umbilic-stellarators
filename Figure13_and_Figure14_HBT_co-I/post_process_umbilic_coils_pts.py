"""
Save the umbilic coil location (X, Y, Z)
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

eq0 = Equilibrium.load("beak_equilibrium_A8_beta1p0_4_umbilic-8100_fixcur.h5")
eq1 = Equilibrium.load("beak_equilibrium_A8_beta1p0_4_umbilic-8100_fixcur.h5")

field = CoilSet.load("beak_beta1p0_coilset_reversed_current4.h5")[3]



phi = np.linspace(0, 2*np.pi, 1024)

grid0 = LinearGrid(zeta=phi)
data_curve = field.compute(["R", "Z"], grid=grid0)
R_coil = data_curve["R"]
Z_coil = data_curve["Z"]

X_coil = R_coil * np.cos(phi)
Y_coil = R_coil * np.sin(phi)
Z_coil = Z_coil

coil_position = np.array([X_coil, Y_coil, Z_coil])
np.save("coil_position.npy", coil_position)

