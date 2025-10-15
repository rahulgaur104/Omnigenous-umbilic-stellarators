#!/usr/bin/env python3


import numpy as np
from desc.plotting import *
from desc.equilibrium import Equilibrium
from matplotlib import pyplot as plt


#eq = Equilibrium.load("../beak_equilibrium_A8_beta1p0_2100.0_increased4100.h5")
eq = Equilibrium.load("../beak_equilibrium_A8_beta1p0_2100.0_reducedFalse.h5")
#eq = Equilibrium.load("/home/rgaur/HBT_umbilic_stellarator_reversed2/beak_equilibrium_A8_beta1p0_6_umbilic-2100_fixcur.h5")

fig, ax = plot_section(eq, name="|F|", norm_F=True, log=True)

plt.show()
