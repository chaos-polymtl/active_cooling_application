# utility.py 
import numpy as np

def flatten_index(i, j, k, nx, ny):
        """
        Convert a 3D index (i, j, k) to a 1D index for a grid of size (nx, ny, nz).
        
        Parameters:
            i (int): Index in the x-direction.
            j (int): Index in the y-direction.
            k (int): Index in the z-direction.
            nx (int): Number of points along the x-axis.
            ny (int): Number of points along the y-axis.
            
        Returns:
            int: The flattened 1D index.
        """
        return i + j * nx + k * nx * ny

def compute_neighbor(index, boundary_id, nx, ny):
    """
    Compute the neighbor index of a given index along a specified axis.
    """

    if boundary_id == 0:
        return index + 1
    elif boundary_id == 1:
        return index - 1
    elif boundary_id == 2:
        return index + nx
    elif boundary_id == 3:
        return index - nx
    elif boundary_id == 4:
        return index + nx * ny
    elif boundary_id == 5:
        return index - nx * ny
    
def compute_boundary_dict(nx, ny, nz):
    """
    Create a dictionary with the boundary points for each face.
    """
    dict_boundary_points = {}

    boundary_points_x0 = np.empty((ny * nz), dtype=int)
    boundary_points_xn = np.empty((ny * nz), dtype=int)

    boundary_points_y0 = np.empty((nx * nz), dtype=int)
    boundary_points_yn = np.empty((nx * nz), dtype=int)
    
    boundary_points_z0 = np.empty((nx * ny), dtype=int)
    boundary_points_zn = np.empty((nx * ny), dtype=int)

    i = 0
    for j in range(ny):
        for k in range(nz):
            boundary_points_x0[i] = flatten_index(0, j, k, nx, ny)
            boundary_points_xn[i] = flatten_index(nx-1, j, k, nx, ny)
            i += 1

    i = 0
    for j in range(nx):
        for k in range(nz):
            boundary_points_y0[i] = flatten_index(j, 0, k, nx, ny)
            boundary_points_yn[i] = flatten_index(j, ny-1, k, nx, ny)
            i += 1

    i = 0
    for j in range(nx):
        for k in range(ny):
            boundary_points_z0[i] = flatten_index(j, k, 0, nx, ny)
            boundary_points_zn[i] = flatten_index(j, k, nz-1, nx, ny)
            i += 1

    dict_boundary_points[0] = boundary_points_x0
    dict_boundary_points[1] = boundary_points_xn
    dict_boundary_points[2] = boundary_points_y0
    dict_boundary_points[3] = boundary_points_yn
    dict_boundary_points[4] = boundary_points_z0
    dict_boundary_points[5] = boundary_points_zn
    
    return dict_boundary_points