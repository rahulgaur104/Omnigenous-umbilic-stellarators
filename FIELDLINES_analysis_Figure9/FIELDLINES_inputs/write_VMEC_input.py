#!/usr/bin/env python3

from desc.vmec import VMECIO
from desc.equilibrium import Equilibrium

eq = Equilibrium.load("beak_equilibrium_A8_beta1p0_2100.0_reducedFalse.h5")

VMECIO.write_vmec_input(eq, fname="input.medbeak")
