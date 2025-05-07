#!/usr/bin/env python3

import pdb
import numpy as np
from netCDF4 import Dataset as ds
from matplotlib import pyplot as plt

rtg = ds("mgrid_medbeak_DESC_6100.nc")

#X = rtg.variables["px_grid"][0, :, :]
#Y = rtg.variables["py_grid"][0, :, :]
#R = np.sqrt(X**2 + Y**2)
#Z = rtg.variables["pz_grid"][0, :, :]
#
#plt.plot(R, Z, '.k')
#plt.show()

#nr = rtg.dimensions["r"].size
#nz = rtg.dimensions["z"].size
#nphi = rtg.dimensions["phi"].size

nr = 128
nz = 128
nphi = 196

rmin = rtg.variables["rmin"]
rmax = rtg.variables["rmax"]
zmin = rtg.variables["zmin"]
zmax = rtg.variables["zmax"]

R0 = np.linspace(rmin, rmax, nr)
Z0 = np.linspace(zmin, zmax, nz)
phi0 = np.linspace(0, 2*np.pi, nphi)

#R, Z = np.meshgrid((R0, Z0))
#pdb.set_trace()
Bz = rtg.variables["bz_001"][0, :, :].data
Br = rtg.variables["br_001"][0, :, :].data
#Bp[Bp > 0] = 0

#plt.contour(R0, Z0, Bp,'-k', levels=[0.])
#plt.contourf(R0, Z0, Bp, levels=128, cmap='coolwarm')
plt.quiver(Z0, R0, Br, Bz, scale=0.5)
#plt.xlim([0.8, 1.6])
#plt.ylim([-0.4, 0.4])
plt.colorbar()
plt.show()

