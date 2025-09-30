#!/usr/bin/env python3
"""
This script plots the |B| contours on the plasma boundary in Boozer coordinates
"""
import pdb
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
import matplotlib as mpl

from desc.equilibrium import Equilibrium
from desc.grid import LinearGrid

from scipy.interpolate import griddata

from desc.plotting import *


eq_new0 = Equilibrium.load("beak_equilibrium_A8_beta1p0_2100.0_reducedFalse.h5")
eq_new1 = Equilibrium.load("beak_equilibrium_A8_beta1p0_2100.0_reducedFalse.h5")

N = int(200)
grid = LinearGrid(L=N)
rho = np.linspace(0, 1, N + 1)

data_keys = ["iota", "D_Mercier"]

data0 = eq_new0.compute(data_keys, grid=grid)
data1 = eq_new1.compute(data_keys, grid=grid)

iota = data0["iota"]

rho0 = 1.0
fig, ax, Boozer_data0 = plot_boozer_surface(eq_new0, rho=rho0, return_data=True, fieldlines=1)
#plt.show()
plt.close()

fig, ax, Boozer_data1 = plot_boozer_surface(eq_new1, rho=rho0, return_data=True)
plt.close()

# Boozer_data_list = [Boozer_data0, Boozer_data1, Boozer_data2]
Boozer_data_list = [Boozer_data0, Boozer_data1]

for i, Boozer_data in enumerate(Boozer_data_list):

    print(i)
    theta_B0 = Boozer_data["theta_B"]
    zeta_B0 = Boozer_data["zeta_B"]
    B0 = Boozer_data["|B|"]

    Theta = theta_B0
    Zeta = zeta_B0

    fig, ax = plt.subplots(figsize=(6, 5))
    contour = ax.contour(
        Zeta,
        Theta,
        B0,
        levels=np.linspace(np.min(B0), np.max(B0), 30)[:],
        cmap="jet",
    )

    # Adding a colorbar
    cbar = fig.colorbar(contour, ax=ax, orientation="vertical")
    tick_locator = ticker.MaxNLocator(nbins=7)
    cbar.locator = tick_locator
    cbar.update_ticks()
    cbar.ax.tick_params(labelsize=18)  # Change colorbar tick size

    # Set explicit tick locations and labels for x and y axes (multiples of π)
    # For x-axis (zeta)
    x_ticks = [0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi]
    x_labels = ['0', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$']
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, fontsize=28)

    # For y-axis (theta)
    y_ticks = [0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi]
    y_labels = ['0', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$']
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=28)

    # Labeling axes
    ax.set_xlabel(r"$\zeta_{\mathrm{Boozer}}$", fontsize=34, labelpad=-3)
    ax.set_ylabel(r"$\theta_{\mathrm{Boozer}}$", fontsize=34, labelpad=-1)

    # Adjust figure to make room for labels
    plt.subplots_adjust(left=0.15, right=0.85, bottom=0.15, top=0.9)

    # Increase resolution to publication quality
    if i == 0:
        #plt.savefig(
        #    f"Boozer_contours/Boozer_contour_plot_rho{rho0}_initial.png",
        #    dpi=300,
        #)
        #plt.savefig(
        #    f"Boozer_contours/Boozer_contour_plot_rho{rho0}_initial.pdf",
        #    dpi=400,
        #)
        plt.savefig(
            f"Boozer_contours/Boozer_contour_plot_rho{rho0}_initial.svg",
        )
        #plt.show()
        plt.close()
    else:
        #plt.savefig(
        #    f"Boozer_contours/Boozer_contour_plot_rho{rho0}_optimized.pdf",
        #    dpi=400,
        #)
        plt.savefig(
            f"Boozer_contours/Boozer_contour_plot_rho{rho0}_optimized.svg",
        )
        #plt.show()
        plt.close()
