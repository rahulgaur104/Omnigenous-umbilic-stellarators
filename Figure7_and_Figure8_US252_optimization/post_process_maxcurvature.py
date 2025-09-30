#!/usr/bin/env python3

import pdb
import numpy as np
from desc.grid import LinearGrid
from desc.equilibrium import Equilibrium


eq = Equilibrium.load("eq_final_high-res.h5")

ntheta = 2000
nzeta = 1000

grid0 = LinearGrid(rho=1.0, theta = np.linspace(0, 2*np.pi, ntheta), zeta = np.linspace(0, 2*np.pi, nzeta))

data = eq.compute("curvature_k2_rho", grid=grid0)["curvature_k2_rho"]

data_reshaped1 = np.reshape(data, (ntheta, nzeta))
data_reshaped2 = np.reshape(data, (nzeta, ntheta))

print(np.min(data_reshaped1, axis=0))
print(np.min(data_reshaped2, axis=1))

print(np.mean(data_reshaped1))

pdb.set_trace()
