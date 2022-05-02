import sys
import os.path
import datetime
import xarray as xr
import numpy as np
from tqdm import tqdm

import fv3_mass_flux_tools.basis_vectors
import fv3_mass_flux_tools.convert
import fv3_mass_flux_tools.interpolate


VERSION_NUMBER="0.0.0"


def create_derived_wind_dataset(tavg_1hr_ctm: xr.Dataset, grid: xr.Dataset, change_of_basis='ronchi', cubic_unstagger=True):
    now = datetime.datetime.now()
    command = " ".join(sys.argv)
    file_attrs = dict(
        title='GEOS FP (Forward Processing) Derived Wind Fields',
        institution='Washington University in St. Louis',
        source=f'calculated from GEOS.fp.asm.tavg_1hr_ctm_c0720_v72 data collection using liambindle/fv3_mass_flux_tools:{VERSION_NUMBER}',
        history=f'[{now}] {command}',
        references='https://github.com/LiamBindle/fv3_mass_flux_tools\nhttp://gmao.gsfc.nasa.gov\nhttps://gmao.gsfc.nasa.gov/operations/GEOS5_V1_File_Specification.pdf',
        comment='See https://github.com/LiamBindle/fv3_mass_flux_tools for details.'
    )

    common_var_attrs = dict(
        fmissing_value=np.float32(1.0e15),
        vmin=np.float32(-1.0e15),
        vmax=np.float32(1.0e15),
        valid_range=np.array([np.float32(-1.0e15), np.float32(1.0e15)], dtype=np.float32),
        units='m s-1',
    )

    ua_attrs = dict(
        long_name='eastward_wind',
        standard_name='eastward_wind',
        **common_var_attrs
    )

    va_attrs = dict(
        long_name='northward_wind',
        standard_name='northward_wind',
        **common_var_attrs
    )
    var_encoding = dict(
        dtype='float32',
        _FillValue=1.0e15
    )

    ds_out = xr.Dataset(coords=tavg_1hr_ctm.coords, attrs=file_attrs)
    dims = tavg_1hr_ctm.MFXC.dims
    nan_array = np.ones_like(tavg_1hr_ctm.MFXC) * np.nan
    ds_out['UA'] = xr.DataArray(nan_array.copy(), dims=dims, attrs=ua_attrs)
    ds_out['UA'].encoding = var_encoding
    ds_out['VA'] = xr.DataArray(nan_array.copy(), dims=dims, attrs=va_attrs)
    ds_out['VA'].encoding = var_encoding
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
    if len(sys.argv) != 4:
        print(f"error: there are 3 required arguments (tavg_1hr_ctm_file gridspec_tile_dir output_dir) but {len(sys.argv)-1} were provided")
        exit(1)
    tavg_1hr_ctm_file = sys.argv[1]
    print(f"Input file: {tavg_1hr_ctm_file}")
    gridspec_tile_dir = sys.argv[2]
    gridspec_tile_files = [f"{gridspec_tile_dir}/c720.tile{n}.nc" for n in range(1,7)]
    print(f"Gridspec tile files: {gridspec_tile_dir}/c720.tile[123456].nc")
    output_dir = sys.argv[3]
    print(f"Output dir: {output_dir}")
    for required_file in [tavg_1hr_ctm_file, output_dir, *gridspec_tile_files]:
        if not os.path.exists(required_file):
            print(f"error: '{required_file}' does not exist")
            exit(1)

    print("Opening files...")
    tavg_1hr_ctm = xr.open_dataset(tavg_1hr_ctm_file)
    grid = xr.open_mfdataset(gridspec_tile_files, concat_dim='nf', combine='nested')
    print("Processing wind fields...")
    ds = create_derived_wind_dataset(tavg_1hr_ctm, grid)
    print("Writing NetCDF file...")
    output_filename=os.path.basename(tavg_1hr_ctm_file).replace("tavg_1hr_ctm_c0720_v72", "tavg_1hr_derivedwind_c0720_v72")
    ds.to_netcdf(f'{output_dir}/{output_filename}')
    print("Done.")
