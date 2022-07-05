import numpy as np
import xarray as xr

from fv3_mass_flux_tools.utilities import get_dx_dy_grid_spacing


def get_full_mfxc_delpx(ds_mf: xr.Dataset, nf: int):
    mfxc = ds_mf.MFXC
    mfyc = ds_mf.MFYC
    delp = ds_mf.DELP
    mf_face = mfxc.isel(nf=nf).values
    delp_face = delp.isel(nf=nf).values
    if nf == 0:
        extra_slice = {'nf': 1, 'Xdim': 0, 'Ydim': slice(None)}
        mf_extra_var = mfxc
    elif nf == 1:
        extra_slice = {'nf': 3, 'Xdim': slice(None, None, -1), 'Ydim': 0}
        mf_extra_var = mfyc
    elif nf == 2:
        extra_slice = {'nf': 3, 'Xdim': 0, 'Ydim': slice(None)}
        mf_extra_var = mfxc
    elif nf == 3:
        extra_slice = {'nf': 5, 'Xdim': slice(None, None, -1), 'Ydim': 0} 
        mf_extra_var = mfyc
    elif nf == 4:
        extra_slice = {'nf': 5, 'Xdim': 0, 'Ydim': slice(None)}
        mf_extra_var = mfxc
    elif nf == 5:
        extra_slice = {'nf': 1, 'Xdim': slice(None, None, -1), 'Ydim': 0}
        mf_extra_var = mfyc

    mf_extra = mf_extra_var.isel(**extra_slice).values
    delp_extra = delp.isel(**extra_slice).values
    mf_face = np.concatenate((mf_face, mf_extra[..., np.newaxis]), axis=-1)
    delp_face = np.concatenate((delp_face, delp_extra[..., np.newaxis]), axis=-1)
    return mf_face, delp_face


def get_full_mfyc_delpy(ds_mf: xr.Dataset, nf: int):
    mfxc = ds_mf.MFXC
    mfyc = ds_mf.MFYC
    delp = ds_mf.DELP
    mf_face = mfyc.isel(nf=nf).values
    delp_face = delp.isel(nf=nf).values
    if nf == 0:
        extra_slice = {'nf': 2, 'Xdim': 0, 'Ydim': slice(None, None, -1)}
        mf_extra_var = mfxc
    elif nf == 1:
        extra_slice = {'nf': 2, 'Xdim': slice(None), 'Ydim': 0}
        mf_extra_var = mfyc
    elif nf == 2:
        extra_slice = {'nf': 4, 'Xdim': 0, 'Ydim': slice(None, None, -1)}
        mf_extra_var = mfxc
    elif nf == 3:
        extra_slice = {'nf': 4, 'Xdim': slice(None), 'Ydim': 0}
        mf_extra_var = mfyc
    elif nf == 4:
        extra_slice = {'nf': 0, 'Xdim': 0, 'Ydim': slice(None, None, -1)}
        mf_extra_var = mfxc
    elif nf == 5:
        extra_slice = {'nf': 0, 'Xdim': slice(None), 'Ydim': 0}
        mf_extra_var = mfyc

    mf_extra = mf_extra_var.isel(**extra_slice).values
    delp_extra = delp.isel(**extra_slice).values
    mf_face = np.concatenate((mf_face, mf_extra[..., np.newaxis, :]), axis=-2)
    delp_face = np.concatenate((delp_face, delp_extra[..., np.newaxis, :]), axis=-2)
    return mf_face, delp_face


def convert_mass_fluxes_to_winds(ds_mf: xr.Dataset, ds_grid: xr.Dataset, nf: int, NUM_FV3_TIME_STEPS: int = 8):
    mfxc, delpx = get_full_mfxc_delpx(ds_mf, nf)
    mfyc, delpy = get_full_mfyc_delpy(ds_mf, nf)

    dx, dy = get_dx_dy_grid_spacing(ds_grid.isel(nf=nf))
    
    # mfxc and mfyc appear to be in units of degrees accidentally; therefore, convert to radians
    mfxc = np.deg2rad(mfxc)
    mfyc = np.deg2rad(mfyc)

    uc = mfxc/(delpx*dy)
    vc = mfyc/(delpy*dx)
    uc /= NUM_FV3_TIME_STEPS
    vc /= NUM_FV3_TIME_STEPS
    return uc, vc

def convert_winds_to_mass_fluxes(ds_mf: xr.Dataset, ds_grid: xr.Dataset, nf: int, uc, vc, NUM_FV3_TIME_STEPS: int = 8):
    # assume delpx and delpy are exactly correct
    _, delpx = get_full_mfxc_delpx(ds_mf, nf)
    _, delpy = get_full_mfyc_delpy(ds_mf, nf)

    uc *= NUM_FV3_TIME_STEPS
    vc *= NUM_FV3_TIME_STEPS

    dx, dy = get_dx_dy_grid_spacing(ds_grid.isel(nf=nf))
    mfxc = uc * (delpx*dy)
    mfyc = vc * (delpy*dx)

    mfxc = np.rad2deg(mfxc)
    mfyc = np.rad2deg(mfyc)

    return mfxc, mfyc
