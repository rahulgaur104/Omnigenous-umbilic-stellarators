#!/usr/bin/env python3
import numpy as np
import pyvista as pv
from desc.plotting import plot_coils

def export_coil_to_paraview(x, y, z, current, filename="coil.vtp"):
    # Stack coordinates into shape (N, 3)
    points = np.column_stack((x, y, z))
    n_points = len(points)

    # Define connectivity: a single line through all points
    # The format is: [n_points, p0, p1, p2, ..., pN]
    # For a polyline with N segments, there are N+1 points
    lines = np.hstack(([n_points], np.arange(n_points)))

    # Create PolyData object
    poly = pv.PolyData()
    poly.points = points
    poly.lines = lines

    # Add current as a scalar field
    current_array = np.full(n_points, current)
    poly["current"] = current_array

    # Save to VTP
    poly.save(filename)

currents = np.abs(np.concatenate(coils.current))  # this may be different depending on the coil object
for i, (X, Y, Z) in enumerate(zip(data["X"], data["Y"], data["Z"])):
    print(f"coil {i}")
    current  = currents[i]
    export_coil_to_paraview(X, Y, Z, current, filename=f"coil{i}.vtp")
