import xarray as xr
import numpy as np
import mf2w.transform
import mf2w.ronchi
from tqdm import tqdm


def convert_mass_fluxes_to_wind(tavg_1hr_ctm: xr.Dataset, grid: xr.Dataset, change_of_basis='simple_rotate'):
    ds_out = xr.Dataset(coords=tavg_1hr_ctm.coords)
    dims = tavg_1hr_ctm.MFXC.dims
    nan_array = np.ones_like(tavg_1hr_ctm.MFXC) * np.nan
    ds_out['UA'] = xr.DataArray(nan_array.copy(), dims=dims)
    ds_out['VA'] = xr.DataArray(nan_array.copy(), dims=dims)
    for nf in tqdm(range(6), unit='cube_sphere_face', desc='Processing'):
        uc, vc = mf2w.transform.mass_fluxes_to_winds(tavg_1hr_ctm, grid, nf)
        ua, va = mf2w.transform.cgrid_to_agrid(uc, vc)

        if change_of_basis == 'simple_rotate':
            M = mf2w.ronchi.uv_face_rotations(nf+1)
            M = np.broadcast_to(M, (*ua.shape, 2, 2))
        elif change_of_basis == 'ronchi':
            x = grid.isel(nf=nf, yc=slice(1, None, 2), xc=slice(1, None, 2)).lons.squeeze().values
            x[x > 180] -= 360
            y = grid.isel(nf=nf, yc=slice(1, None, 2), xc=slice(1, None, 2)).lats.squeeze().values
            M = mf2w.spherical_to_gmao(np.deg2rad(x+10), np.deg2rad(90-y), gmao_face=nf+1)
            M = np.linalg.inv(M)
        elif change_of_basis == 'none':
            M = np.eye(2)

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
