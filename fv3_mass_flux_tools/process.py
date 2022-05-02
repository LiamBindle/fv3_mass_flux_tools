import xarray as xr
import numpy as np
from tqdm import tqdm

import fv3_mass_flux_tools.basis_vectors
import fv3_mass_flux_tools.convert
import fv3_mass_flux_tools.interpolate


def convert_mass_fluxes_to_wind(tavg_1hr_ctm: xr.Dataset, grid: xr.Dataset, change_of_basis='ronchi', cubic_unstagger=True):
    ds_out = xr.Dataset(coords=tavg_1hr_ctm.coords)
    dims = tavg_1hr_ctm.MFXC.dims
    nan_array = np.ones_like(tavg_1hr_ctm.MFXC) * np.nan
    ds_out['UA'] = xr.DataArray(nan_array.copy(), dims=dims)
    ds_out['VA'] = xr.DataArray(nan_array.copy(), dims=dims)
    for nf in tqdm(range(6), unit='cube_sphere_face', desc='Processing'):
        uc, vc = fv3_mass_flux_tools.convert.convert_mass_fluxes_to_winds(tavg_1hr_ctm, grid, nf)
        ua, va = fv3_mass_flux_tools.interpolate.cgrid_to_agrid(uc, vc)

        if change_of_basis == 'simple_rotate':
            M = fv3_mass_flux_tools.basis_vectors.uv_to_eastnorth_rotation_matrix(nf)
            M = np.broadcast_to(M, (*ua.shape, 2, 2))
        elif change_of_basis == 'ronchi':
            x = grid.isel(nf=nf, yc=slice(1, None, 2), xc=slice(1, None, 2)).lons.squeeze().values
            x[x > 180] -= 360
            y = grid.isel(nf=nf, yc=slice(1, None, 2), xc=slice(1, None, 2)).lats.squeeze().values
            M = fv3_mass_flux_tools.basis_vectors.spherical_to_gmao(np.deg2rad(x+10), np.deg2rad(90-y), gmao_face=nf)
            M = np.linalg.inv(M)
        elif change_of_basis == 'none':
            M = np.eye(2)
        else:
            raise ValueError(f"Unsupported change of basis '{change_of_basis}'")

        M = np.broadcast_to(M, (*ua.shape, 2, 2))
        uv = np.concatenate((ua[..., np.newaxis], va[..., np.newaxis]), axis=-1)
        uv = np.einsum('tlxyij,tlxyj->tlxyi', M, uv)
        
        assert np.all(np.isfinite(uv))
        ds_out['UA'][:, :, nf, :, :] = uv[..., 0]
        ds_out['VA'][:, :, nf, :, :] = uv[..., 1]
    
    return ds_out


if __name__ == '__main__':
    tavg_1hr_ctm = xr.open_dataset("data/GEOS.fp.asm.tavg_1hr_ctm_c0720_v72.20210401_0030.V01.nc4")
    grid = xr.open_mfdataset([f"data/c720.tile{n}.nc" for n in range(1,7)], concat_dim='nf', combine='nested')
    convert_mass_fluxes_to_wind(tavg_1hr_ctm, grid, change_of_basis='ronchi')
