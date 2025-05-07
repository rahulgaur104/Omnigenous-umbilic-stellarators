#!/usr/bin/env python3

import os
from desc.grid import LinearGrid

from desc.equilibrium import Equilibrium
from desc.coils import CoilSet

from desc.plotting import *

from matplotlib import pyplot as plt

import plotly.graph_objects as go


eq = Equilibrium.load("beak_equilibrium_A8_beta1p0_2100.0_reducedFalse.h5")
field = CoilSet.load("medium_beak_optimized_coilset_2p100kA_N12_mindist0p020_PFcurrent_59kA_shorterTF.h5")


CoilSet.save_in_makegrid_format(field, "umbilic_coilset")





