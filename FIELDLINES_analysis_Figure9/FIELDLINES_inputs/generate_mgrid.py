#!/usr/bin/env python3

import pdb
import os
import numpy as np

from desc.backend import jax
from desc.equilibrium import Equilibrium
from desc.magnetic_fields import _MagneticField
from desc.coils import CoilSet
from desc.grid import LinearGrid

import subprocess as spr
from desc.plotting import *

#keyword_arr = ["OP"]

field0 = CoilSet.load("medium_beak_optimized_coilset_2p100kA_N12_mindist0p020_PFcurrent_59kA_shorterTF.h5")

print(field0[3].current)
field0[3].current = 6100
#plot_grid = LinearGrid(
#    M=40, N=80, NFP=1, endpoint=True
#)  # a smaller than usual plot grid to reduce memory of the notebook file
#coil_grid = LinearGrid(N=50)
#fig = plot_3d(eq1, "B*n", field=field0, field_grid=coil_grid, grid=plot_grid)
#fig = plot_coils(field0, fig=fig)
#fig.show()


field0.save_mgrid(f"mgrid_medbeak_DESC_6100.nc", Rmin=0.65, Rmax = 1.3, Zmin=-0.3, Zmax=0.3, nR=128, nZ=128, nphi=196)

