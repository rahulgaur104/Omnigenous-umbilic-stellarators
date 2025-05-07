#!/usr/bin/env python3

import pdb
import numpy as np
from netCDF4 import Dataset as ds
from scipy.constants import mu_0
from scipy.interpolate import RectBivariateSpline

# Enclosed toroidal current by the boundary
I = -3.5 * 10000

# Radius icrular torus enclosing the plasma
a_1 = 0.14

# Major radius
R_0 = 0.94

# Aspect ratio of the circular torus
A = R_0/a_1

theta = np.linspace(0, 2*np.pi, 100)

# B_p from Ampere's Law
B_p = mu_0 * I/(2 * np.pi * a_1) * (1 - 1/A * np.cos(theta))


# Coil field data load
rtg = ds("mgrid_medbeak_DESC4_-8100.nc")

Bphi = rtg['bp_001'][:]
Br = rtg['br_001'][:]
Bz = rtg['bz_001'][:]

R = np.linspace(0.7, 1.2, 128)
Z = np.linspace(-0.25, 0.25, 128)

nphi = np.shape(Bphi)[0]

iota_net = 0

for i in range(nphi):

    spl_Br = RectBivariateSpline(R, Z, Br[0])
    spl_Bz = RectBivariateSpline(R, Z, Bz[0])
    spl_Bphi = RectBivariateSpline(R, Z, Bphi[0])
    
    R_interp = R_0 + a_1 * np.cos(theta)
    Z_interp = a_1 * np.sin(theta)
    
    Br_interp = spl_Br.ev(R_interp, Z_interp)
    Bz_interp = spl_Bz.ev(R_interp, Z_interp)
    
    Bphi_interp = spl_Bphi.ev(R_interp, Z_interp)

    B_p_net = B_p + np.sqrt(Br_interp**2 + Bz_interp**2)
    #B_p_net = B_p 

    iota_net += np.mean(A * B_p_net/Bphi_interp)


print(iota_net/nphi)
