#!/usr/bin/env python3

import numpy as np
from desc.equilibrium import Equilibrium
from desc.vmec import VMECIO 

eq = Equilibrium.load("eq_high-res_optimized.h5")

VMECIO.save(eq, "wout_umbilic_131.nc", surfs=256)
