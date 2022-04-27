from re import I
import numpy as np
import xarray as xr

R_EARTH=6371000.7900

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


def unstagger_cubic_interp(x):
    assert len(x.shape) == 1
    assert len(x) >= 4
    a1 = 0.5625
    a2 = -0.0625
    x_interp = np.zeros((len(x)-1,))
    x_interp[0] = 0.5 * (x[0] + x[1])
    x_interp[-1] = 0.5 * (x[-2] + x[-1])
    x0 = x[:-3]
    x1 = x[1:-2]
    x2 = x[2:-1]
    x3 = x[3:]
    x_interp[1:-1] = a2 * (x0 + x3) + a1 * (x1 + x2)
    return x_interp


def unstagger_linear_interp(x):
    assert len(x.shape) == 1
    assert len(x) >= 4
    x_interp = 0.5 * (x[0:-1] + x[1:])
    return x_interp


def get_dx_dy(grid: xr.Dataset):
    # This only works for the first and second face. dx and dy need to be rotated for higher faces.
    phi = np.deg2rad(grid.lats)
    lam = np.deg2rad(grid.lons)

    # todo: review these slices
    dphi = convex_central_angle(phi[0::2,1::2], axis=0, deg=False)
    cosphi = np.cos(phi[1::2,0:-1:2])  # last slice should be 0:-1:2 or 2::2, depending on if C-grid data includes the first or last coordinate in the supergrid
    dlam = convex_central_angle(lam[1::2,0::2], axis=1, deg=False)

    dy = R_EARTH*dphi
    dx = R_EARTH*cosphi*dlam

    dx = np.asarray(dx)
    dy = np.asarray(dy)
    
    return dx, dy


def get_dx_dy_better(grid: xr.Dataset):
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


def mass_fluxes_to_winds(tavg_1hr_ctm: xr.Dataset, grid: xr.Dataset):
    NUM_FV3_TIME_STEPS=8

    mfxc = tavg_1hr_ctm.MFXC
    mfyc = tavg_1hr_ctm.MFYC
    delp = tavg_1hr_ctm.DELP

    #dx, dy = get_dx_dy(grid)

    mfxc = mfxc.pad(Xdim=(0, 1), constant_values=np.nan)
    mfyc = mfyc.pad(Ydim=(0, 1), constant_values=np.nan)
    delpx = delp.pad(Xdim=(0, 1), constant_values=np.nan)
    delpy = delp.pad(Ydim=(0, 1), constant_values=np.nan)
    dx, dy = get_dx_dy_better(grid)
    
    # mfxc and mfyc appear to be in units of degrees accidentally; therefore, convert to radians
    mfxc = np.deg2rad(mfxc)
    mfyc = np.deg2rad(mfyc)

    uc = mfxc/(delpx*dy)
    vc = mfyc/(delpy*dx)
    uc /= NUM_FV3_TIME_STEPS
    vc /= NUM_FV3_TIME_STEPS
    return uc, vc


def cgrid_to_agrid(uc, vc, cubic=True):
    interp = unstagger_cubic_interp if cubic else unstagger_linear_interp
    ua = np.apply_along_axis(interp, -1, uc)
    va = np.apply_along_axis(interp, -2, vc)
    return ua, va
