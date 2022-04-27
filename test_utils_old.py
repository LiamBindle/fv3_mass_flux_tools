import numpy as np
import xarray as xr
import cube_sphere

def test_rotations():
    theta = np.array([np.pi/2])
    v = np.array([1, 0, 0])
    k = np.array([0, 1, 0])
    result = cube_sphere.rotate(v, theta, k)
    answer = np.array([[0, 0, -1]])
    assert np.allclose(result, answer)

def test_multi_rotations():
    theta = np.array([np.pi/2, np.pi, 0])
    v = np.array([1, 0, 0])
    k = np.array([0, 1, 0])
    result = cube_sphere.rotate(v, theta, k)
    answer = np.array([[0, 0, -1], [-1, 0, 0], v])
    assert np.allclose(result, answer)

def test_cart2sph():
    v = [[0, 0, -1], [0, 1, 0]]
    result = cube_sphere.cart2sph(v)
    answer = [[0, -np.pi/2], [np.pi/2, 0]]
    assert np.allclose(result, answer)


def test_sph2cart():
    v = [[0, -np.pi/2], [np.pi/2, 0]]
    result = cube_sphere.sph2cart(v)
    answer = [[0, 0, -1], [0, 1, 0]]
    assert np.allclose(result, answer)


#def test_make_c24():
#    calculated_c24_edges = cube_sphere.make_a_grid(25)
#    calculated_tile1_lon = calculated_c24_edges[0, :, :, 0]
#    calculated_tile1_lon[calculated_tile1_lon < 0] += 360
#    calculated_tile1_lat = calculated_c24_edges[0, :, :, 1]
#    c24_tile1 = xr.open_dataset("data/c24.tile1.nc")
#    c24_tile1_lon = c24_tile1.lons[0::2, 0::2]
#    c24_tile1_lat = c24_tile1.lats[0::2, 0::2]
#    assert np.allclose(calculated_tile1_lon, c24_tile1_lon)
