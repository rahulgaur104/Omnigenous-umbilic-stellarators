#!/usr/bin/env python3

import pdb
import numpy as np

#from desc.geometry import FourierUmbilicCurve
from desc.equilibrium import Equilibrium
from desc.grid import LinearGrid, Grid

from desc.plotting import *
from matplotlib import pyplot as plt
import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.ticker as ticker

eq_new0 = Equilibrium.load("eq_high-res_initial.h5")
eq_new1 = Equilibrium.load("eq_high-res_optimized.h5")

print(eq_new1.compute("R0/a")["R0/a"])
exit()

#fig, ax, data = plot_comparison(eqs=[eq_new0, eq_new1], return_data = True)
fig, ax, data0 = plot_comparison(eqs=[eq_new0], return_data = True)
plt.close()
fig, ax, data1 = plot_comparison(eqs=[eq_new1], return_data = True)
plt.close()

for section_idx in range(4):
    #section_idx = 4
    
    plt.figure(figsize=())
    plt.plot(data0["rho_R_coords"][0][ :, :, section_idx], data0["rho_Z_coords"][0][ :, :, section_idx], 'r', linewidth=2);
    plt.plot(data0["vartheta_R_coords"][0][:, :, section_idx].T, data0["vartheta_Z_coords"][0][:, :, section_idx].T, 'r', linewidth=2); 
    
    plt.plot(data1["rho_R_coords"][0][ :, :, section_idx], data1["rho_Z_coords"][0][ :, :, section_idx], 'g', linewidth=2);
    plt.plot(data1["vartheta_R_coords"][0][:, :, section_idx].T, data1["vartheta_Z_coords"][0][:, :, section_idx].T, 'g', linewidth=2); 
    #plt.axis('equal')
    
    #x_min = np.min(np.concatenate((data0["rho_R_coords"][0][ :, :, section_idx].flatten(), data1["rho_R_coords"][0][ :, :, section_idx].flatten())))
    #x_max = np.max(np.concatenate((data0["rho_R_coords"][0][ :, :, section_idx].flatten(), data1["rho_R_coords"][0][ :, :, section_idx].flatten())))
    #
    #y_min = np.min(np.concatenate((data0["rho_Z_coords"][0][ :, :, section_idx].flatten(), data1["rho_Z_coords"][0][ :, :, section_idx].flatten())))
    #y_max = np.max(np.concatenate((data0["rho_Z_coords"][0][ :, :, section_idx].flatten(), data1["rho_Z_coords"][0][ :, :, section_idx].flatten())))
    
    x_min = 0.6
    x_max = 1.4
    y_min = -0.5
    y_max =  0.5

    plt.xlabel("R", fontsize=22)
    plt.ylabel("Z", fontsize=22)
    
    #plt.title(r"$\phi = \pi*{section_idx}/3$")
    
    plt.xlim([x_min, x_max])
    plt.ylim([y_min, y_max])
    
    plt.xticks(np.linspace(x_min, x_max, 4), fontsize=20)
    plt.yticks(np.linspace(y_min, y_max, 5), fontsize=20)
    
    # Control how many decimal places are shown on x/y ticks (e.g., 2 decimals)
    ax = plt.gca()
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    
    #plt.axis("equal")
    # Get the current axis
    ax = plt.gca()
    
    # Adjust the axis to maintain equal aspect ratio but fill the figure box
    ax.set_aspect("equal", adjustable="box")
    
    # After your plotting commands but before plt.show()
    from matplotlib.lines import Line2D
    
    # Create custom legend handles
    legend_elements = [
        Line2D([0], [0], color='r', lw=2, label='Initial'),
        Line2D([0], [0], color='g', lw=2, label='Final')
    ]
    
    # Add a legend
    legend = plt.legend(handles=legend_elements, fontsize=16, loc='best')
    legend.get_frame().set_alpha(0.5)  # Change 0.5 to any value between 0 and 1


    plt.tight_layout()
    plt.savefig(f"Xsection_{section_idx}.svg", dpi=300)
    plt.show()
    
