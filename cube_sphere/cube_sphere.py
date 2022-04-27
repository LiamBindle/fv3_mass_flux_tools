import numpy as np
import sklearn.preprocessing
import scipy.spatial.transform


def rotate(v, theta, k):
    theta = theta[:, np.newaxis]
    term1 = v*np.cos(theta)
    term2 = np.cross(k, v) * np.sin(theta)
    term3 = k * np.dot(v, k) * (1-np.cos(theta))
    return  term1 + term2 + term3


def normalize(v):
    return v/np.linalg.norm(v)


def cart2sph(v):
    v = np.atleast_2d(v)
    y_sph = np.arcsin(v[..., 2])
    x_sph = np.arctan2(v[..., 1], v[..., 0])
    return np.moveaxis([x_sph, y_sph], 0, -1) 


def sph2cart(v):
    v = np.atleast_2d(v)
    x_car = np.cos(v[..., 1]) * np.cos(v[..., 0])
    y_car = np.cos(v[..., 1]) * np.sin(v[..., 0])
    z_car = np.sin(v[..., 1])
    return np.moveaxis([x_car, y_car, z_car], 0, -1) 


# def make_a_grid(size: int):

#     origin_u = np.array([
#         [0.0,  1.0,  0.0],
#         [0.0,  0.0,  1.0],
#         [1.0,  0.0,  0.0],
#         [1.0,  0.0,  0.0],
#         [1.0,  0.0,  0.0],
#         [-1.0, 0.0,  0.0],
#     ])

#     origin_v = np.array([
#         [0.0,  0.0,  1.0],
#         [0.0, -1.0,  0.0],
#         [0.0,  0.0,  1.0],
#         [0.0,  0.0, -1.0],
#         [0.0, -1.0,  0.0],
#         [0.0, -1.0,  0.0],
#     ])

#     origin_u[0] = rotate(origin_u[0], np.deg2rad(np.array([-10])), np.array([0, 0, 1]))

#     rot_angles = np.deg2rad(np.linspace(-45, 45, size))

#     grid = np.zeros((6, size, size, 2))
    
#     for nf in range(6):
#         origin = normalize(np.cross(origin_u[nf], origin_v[nf]))
#         rot_i = rotate(origin_u[nf], rot_angles, origin_v[nf])
#         rot_j = rotate(origin_v[nf], rot_angles, origin_u[nf])

#         for i in range(size):
#             for j in range(size):
#                 v = normalize(np.cross(rot_i[i], rot_j[j]))
#                 v = cart2sph(v)
#                 v = np.rad2deg(v)
#                 grid[nf, i, j] = v

#     return grid
