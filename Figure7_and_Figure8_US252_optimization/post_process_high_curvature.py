#!/usr/bin/env python3


import numpy as np
from matplotlib import pyplot as plt

from desc.equilibrium import Equilibrium
from desc.grid import LinearGrid

from matplotlib import pyplot as plt


eq = Equilibrium.load("eq_omni_m2_NFP5_14.h5")

theta_arr = np.linspace(0, 2*np.pi, 100)
phi_arr = np.linspace(0, 2*np.pi, 100)
grid0 = LinearGrid(rho = np.array([1.0], theta = theta_arr, zeta=phi_arr)

curvature = eq.compute("curvature_k2_rho", grid = grid0)["curvature_k2_rho"]



