#!/usr/bin/env python3

import pdb
import numpy as np
from svgpathtools import svg2paths
from matplotlib import pyplot as plt

#bulge_size = "small"
#bulge_size = "medium"
bulge_size = "large"

# Load the SVG file
svg_file = f"deformed_HBT_{bulge_size}_bulge.svg"
paths, attributes = svg2paths(svg_file)

# Function to sample points densely from a path
def sample_path_densely(path, num_points=1000):
    """Sample points densely along a path."""
    points = []
    for t in np.linspace(0, 1, num_points):  # Sample at `num_points` along the path
        point = path.point(t)  # Get the (x, y) coordinate at parameter t
        points.append((point.real, point.imag))  # Store as (x, y)
    return points

N0 = 100

# Extract points from all paths in the SVG
all_points = []
for path in paths:
    all_points.extend(sample_path_densely(path, num_points=N0))  # Adjust `num_points` as needed


data0 = np.array(all_points)

# Load the SVG file
svg_file = "HBT_circle.svg"
paths, attributes = svg2paths(svg_file)

# Function to sample points densely from a path
def sample_path_densely(path, num_points=1000):
    """Sample points densely along a path."""
    points = []
    for t in np.linspace(0, 1, num_points):  # Sample at `num_points` along the path
        point = path.point(t)  # Get the (x, y) coordinate at parameter t
        points.append((point.real, point.imag))  # Store as (x, y)
    return points

# Extract points from all paths in the SVG
all_points = []
for path in paths:
    all_points.extend(sample_path_densely(path, num_points=2*N0))  # Adjust `num_points` as needed


data1 = np.array(all_points[:])

x0 = np.mean(data1[:, 0])
y0 = np.mean(data1[:, 1])

avg_radius = (np.max(data1[:, 0]) - np.min(data1[:, 0]))/2
minor_radius = 0.125

data0[:, 0] = minor_radius*(data0[:, 0] - x0)/avg_radius
data0[:, 1] = minor_radius*(data0[:, 1] - y0)/avg_radius

data1[:, 0] = minor_radius*(data1[:, 0] - x0)/avg_radius 
data1[:, 1] = minor_radius*(data1[:, 1] - y0)/avg_radius


#plt.plot(data0[:200, 0], data0[:200, 1], '-or', ms=1)
plt.plot(data0[:, 0], data0[:, 1], 'or', ms=1)
plt.plot(data1[:, 0], data1[:, 1], 'og', ms=1)
plt.plot(np.mean(data1[:, 0]), np.mean(data1[:, 1]), 'xk')

plt.axis("equal")
plt.show()

np.savez(f"beaked_{bulge_size}_{minor_radius}.npz", R=data0[:, 0], Z=data0[:, 1])

