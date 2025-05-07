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

fig = plot_3d(eq,"|B|")
#fig = plot_3d(eq,"curvature_k2_rho")


fig.update_traces(
    colorbar=dict(
        tickfont=dict(size=52),  # Adjust the size value as needed
        title_font=dict(size=52),  # Adjust the size value as needed
        len=0.8,  
    )
)


fig.update_layout(font=dict(size=18, color="black", family="Arial, sans-serif"))


fig.data[0].update(
    lighting=dict(
        ambient=0.85,        # Ambient light - overall brightness
        diffuse=1.0,        # Diffuse light - light scattered evenly
        roughness=0.5,      # Surface roughness - lower is more reflective
        fresnel=0.5,        # Fresnel effect - edge highlighting
        specular=0.6,       # Specular light - concentrated reflection
        facenormalsepsilon=1e-6,  # For face normals calculation
        vertexnormalsepsilon=1e-6  # For vertex normals calculation
    )
)


fig = plot_coils(field, fig=fig)


## Assuming you already have a fig object with a 3D surface
## Get the first trace (assuming it's a surface)
#fig.data[0].update(
#    lighting=dict(
#        ambient=0.3,        # Ambient light - overall brightness
#        diffuse=0.8,        # Diffuse light - light scattered evenly
#        roughness=0.1,      # Surface roughness - lower is more reflective
#        fresnel=0.2,        # Fresnel effect - edge highlighting
#        specular=1.0,       # Specular light - concentrated reflection
#        # Light position:
#        facenormalsepsilon=1e-6,
#        position=dict(x=100, y=100, z=1000)
#    )
#)



import numpy as np
import plotly.graph_objects as go

def create_tube_mesh(x, y, z, radius=2, n_segments=16):
    """Create a 3D tube along a curve defined by points (x,y,z)"""
    points = np.array([x, y, z]).T
    n_points = len(points)
    
    # Initialize lists for vertices and triangles
    vertices = []
    triangles = []
    
    # For each point along the curve (except the last one)
    for i in range(n_points - 1):
        # Calculate direction vector
        direction = points[i+1] - points[i]
        direction = direction / np.linalg.norm(direction)
        
        # Find perpendicular vectors
        # First try using z-axis as reference
        reference = np.array([0, 0, 1])
        if np.abs(np.dot(direction, reference)) > 0.99:
            # If direction is nearly parallel to z-axis, use x-axis
            reference = np.array([1, 0, 0])
            
        # Cross product to get first perpendicular vector
        perp1 = np.cross(direction, reference)
        perp1 = perp1 / np.linalg.norm(perp1)
        
        # Second perpendicular vector
        perp2 = np.cross(direction, perp1)
        
        # Create circle points at current position
        circle_vertices_i = []
        for j in range(n_segments):
            theta = 2 * np.pi * j / n_segments
            # Point on circle
            circle_point = points[i] + radius * (np.cos(theta) * perp1 + np.sin(theta) * perp2)
            vertices.append(circle_point)
            circle_vertices_i.append(len(vertices) - 1)
        
        # If not the first point, connect to previous circle
        if i > 0:
            prev_circle = circle_vertices_i[0] - n_segments
            for j in range(n_segments):
                # Get indices for quad vertices
                j_next = (j + 1) % n_segments
                
                # Vertex indices
                v00 = prev_circle + j
                v01 = prev_circle + j_next
                v10 = circle_vertices_i[j]
                v11 = circle_vertices_i[j_next]
                
                # Add two triangles for the quad
                triangles.append([v00, v10, v11])
                triangles.append([v00, v11, v01])
    
    # Convert lists to arrays
    vertices = np.array(vertices)
    triangles = np.array(triangles)
    
    return {
        'vertices': vertices,
        'triangles': triangles
    }

# Now use this function with your coil traces
# Assuming 'fig' is your existing figure with the coil lines
coil_traces = [trace for trace in fig.data if (
    trace.type == 'scatter3d' and 
    trace.mode == 'lines' and 
    (trace.line.color == 'black' or not hasattr(trace.line, 'color'))
)]

## Create a new figure with all non-coil traces
new_fig = go.Figure()
#for trace in fig.data:
#    if trace not in coil_traces:
#        new_fig.add_trace(trace)

# First, add the equilibrium surface (usually the first trace)
new_fig.add_trace(fig.data[0])  # This adds the colored toroidal surface

# Add any other non-coil traces you need (like axes, etc.)
# This is often unnecessary but included for completeness
for i, trace in enumerate(fig.data):
    if i != 0 and trace not in coil_traces:
        new_fig.add_trace(trace)


# Modified approach to identify coil traces
coil_traces = []
for i, trace in enumerate(fig.data):
    # Check if this trace is part of the coil representation
    # Add more conditions based on your specific figure structure
    if (trace.type == 'scatter3d' or
        (hasattr(trace, 'name') and 'coil' in trace.name.lower()) or
        (hasattr(trace, 'line') and hasattr(trace.line, 'color') and trace.line.color == 'black')):
        coil_traces.append(trace)
        #print(f"Found coil trace: {i}")


# For each coil, create a 3D tube
for i, coil in enumerate(coil_traces):
 
    # Extract the coordinates
    x, y, z = coil.x, coil.y, coil.z
    
    # Create the tube mesh
    tube_mesh = create_tube_mesh(x, y, z, radius=0.01, n_segments=16)

    if i == 22:

        # Add the tube as a mesh3d trace
        new_fig.add_trace(go.Mesh3d(
            x=tube_mesh['vertices'][:, 0],
            y=tube_mesh['vertices'][:, 1],
            z=tube_mesh['vertices'][:, 2],
            i=tube_mesh['triangles'][:, 0],
            j=tube_mesh['triangles'][:, 1],
            k=tube_mesh['triangles'][:, 2],
            color='black',
            lighting=dict(
                ambient=0.8,
                diffuse=0.7,
                roughness=0.3,
                fresnel=0.4,
                specular=2.0,
                facenormalsepsilon=1e-6,
                vertexnormalsepsilon=1e-6
            ),
            name=f"Coil {i}"
        ))
    else:
        # Add the tube as a mesh3d trace
        new_fig.add_trace(go.Mesh3d(
            x=tube_mesh['vertices'][:, 0],
            y=tube_mesh['vertices'][:, 1],
            z=tube_mesh['vertices'][:, 2],
            i=tube_mesh['triangles'][:, 0],
            j=tube_mesh['triangles'][:, 1],
            k=tube_mesh['triangles'][:, 2],
            color='gray',
            lighting=dict(
                ambient=0.8,
                diffuse=0.7,
                roughness=0.3,
                fresnel=0.4,
                specular=2.0,
                facenormalsepsilon=1e-6,
                vertexnormalsepsilon=1e-6
            ),
            name=f"Coil {i}"
        ))
        continue

# Copy the layout from the original figure
new_fig.update_layout(
    scene=fig.layout.scene,
    title=fig.layout.title,
    width=fig.layout.width,
    height=fig.layout.height
)



config = {
    "toImageButtonOptions": {
        "filename": f"modB_3d_plot",
        #"filename": f"curvature_plot",
        "format": "svg",
        "scale": 2,
    }
}

save_path_html = os.getcwd() + f"/3D_plot1.html"
new_fig.write_html(
    save_path_html, config=config, include_plotlyjs=True, full_html=True
)


## Show the new figure
#new_fig.show()

