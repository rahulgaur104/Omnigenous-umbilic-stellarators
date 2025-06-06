import numpy as np
import matplotlib.pyplot as plt
import h5py
import pdb

from desc.equilibrium import Equilibrium
from desc.grid import LinearGrid, Grid

from desc.plotting import *
from matplotlib import pyplot as plt
import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.ticker as ticker
from desc.coils import CoilSet


# --------------------------------------------------------------------
# Force Matplotlib to use its default sans-serif font (DejaVu Sans).
# Also configure math text to be sans-serif, and disable LaTeX engine.
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["DejaVu Sans"]
mpl.rcParams["mathtext.fontset"] = "dejavusans"
mpl.rcParams["text.usetex"] = False

def read_fieldlines(filename):
    data = {}
    
    # 1) Open the HDF5 file and read everything into a dictionary
    with h5py.File(filename, 'r') as f:
        for key in f.keys():
            data[key] = f[key][...]  # load dataset as numpy array
    
    # 2) Extract nlines, nsteps from single-element datasets (the file likely has them).
    #    Suppose the file's "nlines" and "nsteps" arrays each contain exactly one value:
    nlines_file = int(data['nlines'][0])  # e.g. 24
    nsteps_file = int(data['nsteps'][0])  # e.g. 50931
    
    # 3) Now fix R_lines, PHI_lines, Z_lines, B_lines if they exist
    #    so they match shape = (nlines, nsteps).
    #    Right now they are shape=(50931, 24) => (nsteps, nlines).
    #    We only do this if they have shape (nsteps_file, nlines_file).
    #    If they already have shape (nlines_file, nsteps_file), we do nothing.
    
    for key_ in ['R_lines', 'PHI_lines', 'Z_lines', 'B_lines']:
        if key_ in data:
            arr = data[key_]
            if arr.shape == (nsteps_file, nlines_file):
                data[key_] = arr.T  # now shape => (nlines, nsteps)
            elif arr.shape == (nlines_file, nsteps_file):
                # It's already the correct orientation. do nothing
                pass
            else:
                # The file might be inconsistent, or we guessed incorrectly.
                print(f"WARNING: {key_} has shape={arr.shape}, "
                      f"but we expected {(nsteps_file, nlines_file)} or "
                      f"{(nlines_file, nsteps_file)}. Not transposing.")
    
    # 4) Overwrite data['nlines'] and data['nsteps'] with correct int values
    data['nlines'] = nlines_file
    data['nsteps'] = nsteps_file
    
    # 5) Create X_lines, Y_lines using the (nlines, nsteps)-shaped arrays
    if 'R_lines' in data and 'PHI_lines' in data:
        data['X_lines'] = data['R_lines'] * np.cos(data['PHI_lines'])
        data['Y_lines'] = data['R_lines'] * np.sin(data['PHI_lines'])
    
    # 6) Compute phiend, dphi if PHI_lines is present
    #    With shape (nlines, nsteps), the last column is index -1
    if 'PHI_lines' in data:
        data['phiend'] = data['PHI_lines'][:, -1]
        data['dphi']   = data['phiend'] / float(data['nsteps'] - 1)
    return data


def plot_fieldlines(data, plottype='basic', cutplane_idx=1, 
                    skip=1, line_color='k'):
    """
    Plot 2D Poincaré data from read_fieldlines (in R,Z space).
    
    Parameters
    ----------
    data : dict
        Dictionary with keys like 'R_lines', 'Z_lines', 'nlines', 'nsteps', 
        possibly 'npoinc', etc.
        R_lines, Z_lines assumed shape = (nlines, nsteps).
    plottype : str
        Either 'basic' or 'cutplane'.
        - 'basic': Gathers data from the first cutplane or from every 
                   npoinc step if given.
        - 'cutplane': Gathers data from a user-defined plane index.
    cutplane_idx : int
        For 'cutplane', which cross-section index to plot (1-based).
        E.g. 1 => columns 0 in Python, 2 => columns 1, etc.
    skip : int
        Plot every 'skip'-th field line to reduce clutter in the figure 
        (i.e., the row step).
    line_color : str
        Color to use for the points/lines if not coloring by something else.
    """

    # 1) Extract arrays
    R_lines = data['R_lines']  # shape = (nlines, nsteps)
    Z_lines = data['Z_lines']  # shape = (nlines, nsteps)
    nlines  = data['nlines']   # e.g., 24
    nsteps  = data['nsteps']   # e.g., 50931

    # 2) Grab npoinc if it exists, else default to 1
    #    If your file or code sets npoinc=96, for instance, 
    #    we assume it is an int.
    npoinc = data.get('npoinc', 1)
    if isinstance(npoinc, (np.ndarray, list)):
        npoinc = int(npoinc[0])  # if stored in an array

    # 3) Figure out which columns we want to plot based on plottype
    #    In the original MATLAB logic:
    #       line_dex = nphi : npoinc : nsteps
    #    but we have to adapt for 0-based Python indexing.
    if plottype == 'basic':
        # "Basic" means the first cutplane or every npoinc step.
        # For a first cutplane in Python => column 0
        # then column = 0 + npoinc, etc.
        line_dex = range(0, nsteps, npoinc)
        #title_str = "Basic Poincaré Plot"
    elif plottype == 'cutplane':
        # 'cutplane_idx' is 1-based, so the first column is cutplane_idx-1
        # Then we go in steps of npoinc.
        col_start = cutplane_idx - 1  # make it 0-based
        line_dex = range(col_start, nsteps, npoinc)
        #title_str = f"Cutplane (index={cutplane_idx})"
    else:
        raise ValueError(f"Unknown plottype '{plottype}'. Use 'basic' or 'cutplane'.")

    # 4) Slice out the data
    #    We skip lines by picking [::skip] in the first dimension
    #    and pick only line_dex in the second dimension.
    #R_subset = R_lines[0:nlines:skip, line_dex]  # shape => (# lines plotted, # points per line)
    #Z_subset = Z_lines[0:nlines:skip, line_dex]

    #indices = np.array([0, 1, 2, 3, 4, 5, 7, 8, 10, 12, 14, 15, 16, 17, 18, 19, ]) 
    R_subset = R_lines[0:nlines:skip, line_dex]  # shape => (# lines plotted, # points per line)
    Z_subset = Z_lines[0:nlines:skip, line_dex]
    # 5) Plot
    plt.figure()
    # We'll just do a point plot, same as MATLAB plot(...,'.')
    plt.plot(R_subset, Z_subset, '.', color=line_color, lw=0.2, markersize=0.5)
    #Z_filtered = Z_subset[R_subset>0]
    #R_filtered = R_subset[R_subset>0]
    #plt.plot(R_filtered, Z_filtered, '.', color=line_color, lw=0.2, markersize=1.0)
    #plt.plot(R_filtered, Z_filtered, '-.', color=line_color, lw=0.05, markersize=1.0)

    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel("R [m]")
    plt.ylabel("Z [m]")
    #plt.show()



data = read_fieldlines("fieldlines_fieldlines.h5")
#plot_fieldlines(data, plottype='cutplane', cutplane_idx=1, skip=1)
#plot_fieldlines(data, plottype='cutplane', cutplane_idx=5, skip=1)
plot_fieldlines(data, plottype='cutplane', cutplane_idx=9, skip=1)
#plot_fieldlines(data, plottype='cutplane', cutplane_idx=1, skip=1)




### plotting the free=boundary solution
eq0 = Equilibrium.load("beak_equilibrium_A8_beta1p0_2100.0_reducedFalse.h5")

grid0 = LinearGrid(L=100)
print(eq0.compute("iota current", grid=grid0)["iota current"])
phi = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
fig, ax, data0 = plot_boundary(eq=eq0, phi=phi, return_data = True)
plt.close()



#for i in range(4):
i = 2
#plt.plot(data0["R"][:, 1, i], data0["Z"][:, 1, i], '-r', linewidth=2)
num_flux_surfs = 10
theta0 = np.linspace(0, 2*np.pi, 200)
for j in range(num_flux_surfs):
    grid2 = LinearGrid(rho = np.array([(1/num_flux_surfs)*(j+1)]), theta=theta0, zeta=phi[i])
    data2 = eq0.compute(["R","Z"],  grid=grid2)
    if j == num_flux_surfs-1:
        plt.plot(data2["R"], data2["Z"], '-r', linewidth=2)
    else:
        plt.plot(data2["R"], data2["Z"], '-r', linewidth=1)


# plotting the umbilic coil location
field = CoilSet.load("medium_beak_optimized_coilset_2p100kA_N12_mindist0p020_PFcurrent_59kA_shorterTF.h5")[3]
grid0 = LinearGrid(zeta=phi[i])
data_curve = field.compute(["R", "Z"], grid=grid0)
R_coil = data_curve["R"]
Z_coil = data_curve["Z"]
plt.plot(R_coil, Z_coil, 'ok', ms=6)

#plt.xlim([0.75, 1.25])
plt.xlim([0.75, 1.1])
#plt.ylim([-0.20, 0.20])
plt.ylim([-0.20, 0.25])

plt.xticks(np.linspace(0.75, 1.15, 5), fontsize=18)
#plt.xticks(np.linspace(0.75, 1.25, 5), fontsize=18)
plt.yticks(np.linspace(-0.2, 0.2, 5), fontsize=18)

# Control how many decimal places are shown on x/y ticks (e.g., 2 decimals)
ax = plt.gca()
ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))

plt.xlabel("R", fontsize=22)
plt.ylabel("Z", fontsize=22)

plt.savefig("poincare-cross-section_2.svg", dpi=200)
#plt.savefig("poincare-cross-section_1.svg", dpi=200)
#plt.savefig("poincare-cross-section_0.eps", dpi=600)
plt.show()
