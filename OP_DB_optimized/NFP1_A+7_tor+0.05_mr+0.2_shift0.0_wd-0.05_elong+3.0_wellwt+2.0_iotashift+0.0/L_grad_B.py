#!/usr/bin/env python3

import glob
import numpy as np
import pdb
from desc.equilibrium import Equilibrium
from desc.grid import LinearGrid

from desc.examples import get
from desc.compat import rescale

#eq = get("precise_QA")
#eq = get("precise_QH")
#eq = get("W7-X")

#eq = Equilibrium.load("/home/rgaur/Enhanced-stability-DESC-data/equilibria/OP_nfp3/eq_OP_ball3_033_initial.h5")
file0 = glob.glob("eq_*vacuum_final2.h5")[0]

eq = Equilibrium.load(file0)

eq = rescale(eq, L=("a", 1.704), B=("<B>", 5.865))

grid0 = LinearGrid(rho = np.array([1.0]), M=2*eq.M_grid, N=2*eq.N_grid, NFP=eq.NFP)

data = eq.compute("L_grad(B)", grid=grid0)["L_grad(B)"]

#print(eq.compute("R0"))

print(np.min(data))

