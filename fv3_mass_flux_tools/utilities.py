import numpy as np
import xarray as xr
from fv3_mass_flux_tools.constants import R_EARTH

def convex_central_angle(angle: np.ndarray, deg=True, axis=0):
    full_circle = 360.0 if deg else np.pi*2.0
    half_circle = full_circle/2.0
    diff_angle = np.abs(np.diff(angle % full_circle, axis=axis))
    is_reflex_angle = diff_angle > half_circle
    diff_angle[is_reflex_angle] = full_circle - diff_angle[is_reflex_angle]  # convex angle assumption
    return diff_angle


def central_angle(lon0, lat0, lon1, lat1):
    lon0 = np.deg2rad(lon0)
    lon1 = np.deg2rad(lon1)
    lat0 = np.deg2rad(lat0)
    lat1 = np.deg2rad(lat1)
    angle_rad = np.arccos(np.sin(lat0) * np.sin(lat1) + np.cos(lat0) * np.cos(lat1) * np.cos(np.abs(lon0-lon1))) 
    return np.rad2deg(angle_rad)


def great_circle_distance(lon0, lat0, lon1, lat1):
    angle_rad = np.deg2rad(central_angle(lon0, lat0, lon1, lat1))
    return angle_rad * R_EARTH


def get_dx_dy_grid_spacing(grid: xr.Dataset):
    y0_lon = grid.lons.isel(yc=slice(0,   -2, 2), xc=slice(0, None, 2))
    y0_lat = grid.lats.isel(yc=slice(0,   -2, 2), xc=slice(0, None, 2))
    y1_lon = grid.lons.isel(yc=slice(2, None, 2), xc=slice(0, None, 2))
    y1_lat = grid.lats.isel(yc=slice(2, None, 2), xc=slice(0, None, 2))
    dy = great_circle_distance(y0_lon, y0_lat, y1_lon, y1_lat)

    x0_lon = grid.lons.isel(yc=slice(0, None, 2), xc=slice(0,   -2, 2))
    x0_lat = grid.lats.isel(yc=slice(0, None, 2), xc=slice(0,   -2, 2))
    x1_lon = grid.lons.isel(yc=slice(0, None, 2), xc=slice(2, None, 2))
    x1_lat = grid.lats.isel(yc=slice(0, None, 2), xc=slice(2, None, 2))
    dx = great_circle_distance(x0_lon, x0_lat, x1_lon, x1_lat)

    dx = np.asarray(dx)
    dy = np.asarray(dy)

    return dx, dy
