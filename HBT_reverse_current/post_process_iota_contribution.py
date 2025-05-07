#!/usr/bin/env python3


import numpy as np
from desc.equilibrium import Equilibrium
from desc.grid import LinearGrid
from matplotlib import pyplot as plt
import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.ticker as ticker
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter as sf
from desc.plotting import *

eq0 = Equilibrium.load("beak_equilibrium_A8_beta1p0_4_umbilic-8100.h5")

fig, ax = plot_section(eq0, name="|F|", norm_F=True, log=True)
plt.show()

L = 10
rho = np.linspace(0, 1, L+1)
grid0 = LinearGrid(L=L)


data0 = eq0.compute(["iota current", "iota vacuum"], grid=grid0)
iota_current0 = data0["iota current"][-1]
iota_vacuum0 = data0["iota vacuum"][-1]
frac0 = iota_vacuum0/(iota_current0+iota_vacuum0)

print(frac0)

