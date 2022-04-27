import numpy as np
import xarray as xr


def convex_central_angle(angle: np.ndarray, deg=True, axis=0):
    full_circle = 360.0 if deg else np.pi*2.0
    half_circle = full_circle/2.0
    diff_angle = np.abs(np.diff(angle % full_circle, axis=axis))
    is_reflex_angle = diff_angle > half_circle
    diff_angle[is_reflex_angle] = full_circle - diff_angle[is_reflex_angle]  # convex angle assumption
    return diff_angle


def get_dx_dy(grid: xr.Dataset, verbose=False):
    R_EARTH=6371000.7900
    phi = np.deg2rad(grid.lats.isel(nf=0).values)
    lam = np.deg2rad(grid.lons.isel(nf=0).values)

    # todo: review these slices
    dphi = convex_central_angle(phi[0::2,1::2], axis=0, deg=False)
    cosphi = np.cos(phi[1::2,0:-1:2])  # last slice should be 0:-1:2 or 2::2, depending on if C-grid data includes the first or last coordinate in the supergrid
    dlam = convex_central_angle(lam[1::2,0::2], axis=1, deg=False)

    dy = R_EARTH*dphi
    dx = R_EARTH*cosphi*dlam
    
    return dx, dy


def mass_fluxes_to_winds(tavg_1hr_ctm: xr.Dataset, grid: xr.Dataset, verbose=False):
    NUM_FV3_TIME_STEPS=8

    mfxc = tavg_1hr_ctm.MFXC.isel(nf=0).values
    mfyc = tavg_1hr_ctm.MFYC.isel(nf=0).values
    delp = tavg_1hr_ctm.DELP.isel(nf=0).values

    dx, dy = get_dx_dy(grid, verbose)
    
    # mfxc and mfyc appear to be in units of degrees accidentally; therefore, convert to radians
    mfxc = np.deg2rad(mfxc)
    mfyc = np.deg2rad(mfyc)

    uc = mfxc/(delp*dy)
    vc = mfyc/(delp*dx)
    uc /= NUM_FV3_TIME_STEPS
    vc /= NUM_FV3_TIME_STEPS
    return uc, vc
