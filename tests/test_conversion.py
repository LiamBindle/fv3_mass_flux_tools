import numpy as np
import xarray as xr
from fv3_mass_flux_tools.convert import *
from fv3_mass_flux_tools.interpolate import cgrid_to_agrid


def test_mass_flux_conversion_first_face():
    import xarray as xr
    tavg_1hr_ctm = xr.open_dataset("data/GEOS.fp.asm.tavg_1hr_ctm_c0720_v72.20210401_0030.V01.nc4")
    grid = xr.open_mfdataset([f"data/c720.tile1.nc"], concat_dim='nf', combine='nested')
    uc, vc = convert_mass_fluxes_to_winds(tavg_1hr_ctm, grid, 0)
    ua, va = cgrid_to_agrid(uc, vc)
    magnitude = np.sqrt(ua*ua + va*va)
    avg_magnitude_surface = np.mean(magnitude[0,-1,:,:])
    assert 5.0 < avg_magnitude_surface < 7.0,f"calculated avg. wind speed: {avg_magnitude_surface} m/s; expected: ~7.5 m/s"  # actual magnitude should be ~7.5 m/s


def test_loading_mass_fluxes():
    import xarray as xr
    tavg_1hr_ctm = xr.open_dataset("data/GEOS.fp.asm.tavg_1hr_ctm_c0720_v72.20210401_0030.V01.nc4")
    mfxc, delpx = get_full_mfxc_delpx(tavg_1hr_ctm, 0)
    mfyc, delpy = get_full_mfyc_delpy(tavg_1hr_ctm, 0)

    assert mfxc.shape == (1, 72, 720, 721)
    assert delpx.shape == (1, 72, 720, 721)
    assert mfyc.shape == (1, 72, 721, 720)
    assert delpy.shape == (1, 72, 721, 720)
