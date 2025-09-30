#!/usr/bin/env python3

import numpy as np
from desc.equilibrium import Equilibrium
from desc.grid import LinearGrid

eq = Equilibrium.load("beak_equilibrium_A8_beta1p0_2100.0_reducedFalse.h5")
rho = np.linspace(0, 1, 100)
grid0 = LinearGrid(rho=rho, M=eq.M_grid, N=eq.N_grid)
data = eq.compute("current", grid=grid0)

print(grid0.compress(data["current"]))

