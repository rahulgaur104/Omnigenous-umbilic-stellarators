#!/usr/bin/env python3

import numpy as np
from desc.equilibrium import Equilibrium
from desc.vmec import VMECIO 

#eq = Equilibrium.load("beak_equilibrium_A8_beta1p0_2100.0_reducedFalse.h5")
#eq = Equilibrium.load("beak_equilibrium_A8_beta1p0_2100.0_increased3100.h5")

#value_list = [4100, 5100, 6100]
#for value in value_list:
##eq = Equilibrium.load(f"beak_equilibrium_A8_beta1p0_2100.0_increased{value}.h5")
eq = Equilibrium.load(f"beak_equilibrium_A8_beta1p0_2100.0_reducedFalse.h5")
VMECIO.save(eq, f"wout_fieldlines_2100.nc", surfs=1024)
