#!/usr/bin/env python3
import pdb
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
import matplotlib as mpl


## --------------------------------------------------------------------
## Force Matplotlib to use its default sans-serif font (DejaVu Sans).
## Also configure math text to be sans-serif, and disable LaTeX engine.
#mpl.rcParams["text.usetex"] = True  # Enable LaTeX rendering
#mpl.rcParams["font.family"] = "serif"  # Use serif font family (typical for LaTeX)
#mpl.rcParams["font.serif"] = ["Computer Modern Roman"]  # Standard LaTeX font
#mpl.rcParams["mathtext.fontset"] = "cm"  # Use Computer Modern math font

# Modify these settings in your script
mpl.rcParams["text.usetex"] = False  # Don't use actual LaTeX
mpl.rcParams["mathtext.fontset"] = "cm"  # Still use Computer Modern math font
mpl.rcParams["font.family"] = "serif"  # Use serif font

# Load data
#data = np.load("Jpar_UT131_rho0p5.npz")
#data = np.load("Jpar_UT225_rho0p75.npz")
#data = np.load("Jpar_OP_DB_NFP2.npz")
data = np.load("Jpar_OP_DB_NFP1.npz")

#pdb.set_trace()

#data = np.load("Jpar_UT225_rho0p75.npz")
data_full = data["data_full"]  # shape: (nPitch, nAlpha)
alpha     = data["alpha_arr"]      # shape: (nAlpha,)
minB      = data["minB"][0]      # shape: (nPitch,)
maxB      = data["maxB"][0]      # shape: (nPitch,)
inv_pitch     = np.linspace(minB, maxB, data["num_pitch"]) 
threshold = 1e-3 

# Create a figure
plt.figure(figsize=(6, 5))
data_transposed = data_full  # shape: (nAlpha, nPitch)


# Copy the plasma colormap so we can edit it
cmap = plt.get_cmap("plasma").copy()

# Make any values below 'vmin' be displayed in white
cmap.set_under("white")

# Create a normalization that sets vmin = threshold
norm = plt.Normalize(vmin=threshold, vmax=data_transposed.max())
extent = [inv_pitch.min(), inv_pitch.max(), alpha.min(), alpha.max()]

# Plot the image with no interpolation so each cell is a solid color
plt.imshow(
    data_transposed,
    origin="lower",            # so alpha increases upward
    extent=extent,
    aspect="auto",            # allows the aspect ratio to stretch or shrink
    cmap=cmap,
    norm=norm,
    interpolation="nearest"    # no smoothing/interpolation
)

# Add a colorbar to show the data_full scale
cbar = plt.colorbar()

# Increase colorbar tick label font size
cbar.ax.tick_params(labelsize=22)  # Increase colorbar tick font size


# Control the number of ticks and format
ax = plt.gca()  # Get current axis

# Format colorbar ticks to show only 3 significant digits
import matplotlib.ticker as ticker
cbar.ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
cbar.ax.yaxis.get_major_formatter().set_powerlimits((0, 0))  # Use scientific notation
cbar.ax.yaxis.set_major_locator(ticker.MaxNLocator(6))  # Set maximum number of ticks

y_ticks = [0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi]
y_labels = ['0', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$']
ax.set_yticks(y_ticks)
ax.set_yticklabels(y_labels, fontsize=26)

# Label the axes
plt.xlabel(r"$1/\lambda$", fontsize=24)
plt.ylabel(r"$\alpha$", fontsize=28, labelpad=-5)


### Increase axis tick label font size
##plt.xticks(np.linspace(1/maxB, 1/minB-0.01, 6),fontsize=18)  # Increase x-axis tick font size
plt.xticks(np.linspace(minB+0.01, maxB-0.04, 5),fontsize=20)  # Increase x-axis tick font size
##plt.yticks(fontsize=18)  # Increase y-axis tick font size


# Format x-ticks to show only 2 decimal places
ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
## Limit the number of x-ticks
#ax.xaxis.set_major_locator(MaxNLocator(5))

#plt.xlim([1/maxB, 1/minB-0.1])

## Format y-ticks to show only 1 decimal place
#ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
## Limit the number of y-ticks
#ax.yaxis.set_major_locator(MaxNLocator(6))

ax.set_xlim(minB, maxB-0.04)  # Use ax.set_xlim instead of plt.xlim

plt.tight_layout()

#plt.savefig("Jpar_UT225.svg", dpi=330)
#plt.savefig("Jpar_UT131.svg", dpi=330)
#plt.savefig("Jpar_OP_DB_NFP2.svg", dpi=330)
plt.savefig("Jpar_OP_DB_NFP1.svg", dpi=330)
plt.show()

