# Data for fv3_mass_flux_tools

The tests and demo notebook use files described here.

## Grid Data and Regridding Weights
The GEOS-FP mass flux collections (GEOS.fp.asm.tavg_1hr_ctm_c0720_v72 and GEOS.fp.asm.inst_1hr_ctm_c0720_v72) use a C720 gnomonic cube sphere grid.
The gridspec file for this grid was generated with the [gridspec](https://github.com/LiamBindle/gridspec) cli:

```console
$ gridspec-create gcs 720  # generates c720_gridspec.nc and c720.tile[123456].nc
```

The GEOS-FP 3-hour wind collection (GEOS.fp.asm.tavg3_3d_asm_Nv) uses a 721x1152 
dataline-centered and pole-centered latitude-longitude grid. The gridspec file
for this grid was generated with

```console
$ gridspec-create latlon 721 1152 -dc -pc  # generates regular_lat_lon_721x1152.nc
```

Regridding weights for converting between the two grids were generated with `ESMF_RegridWeightGen`:

```console
$ ESMF_RegridWeightGen -s c720_gridspec.nc -d regular_lat_lon_721x1152.nc -w weights_cs2ll.nc -m conserve
$ ESMF_RegridWeightGen -d c720_gridspec.nc -s regular_lat_lon_721x1152.nc -w weights_ll2cs.nc -m conserve
```

# Mass Flux and Wind Data

Sample 1-hour mass fluxes:
* http://geoschemdata.wustl.edu/ExtData/GEOS_C720/GEOS_FP_Native/Y2022/M04/D01/GEOS.fp.asm.tavg_1hr_ctm_c0720_v72.20210401_0030.V01.nc4
* http://geoschemdata.wustl.edu/ExtData/GEOS_C720/GEOS_FP_Native/Y2022/M04/D01/GEOS.fp.asm.tavg_1hr_ctm_c0720_v72.20210401_0130.V01.nc4
* http://geoschemdata.wustl.edu/ExtData/GEOS_C720/GEOS_FP_Native/Y2022/M04/D01/GEOS.fp.asm.tavg_1hr_ctm_c0720_v72.20210401_0230.V01.nc4

These were combined into a 3-hour time average with
```python
from dask.diagnostics import ProgressBar
import xarray as xr
tavg_3hr_ctm = xr.open_mfdataset(["data/GEOS.fp.asm.tavg_1hr_ctm_c0720_v72.20210401_0030.V01.nc4",
                                  "data/GEOS.fp.asm.tavg_1hr_ctm_c0720_v72.20210401_0130.V01.nc4",
                                  "data/GEOS.fp.asm.tavg_1hr_ctm_c0720_v72.20210401_0230.V01.nc4"], data_vars=['MFXC', 'MFYC', 'DELP'], compat='override', coords=['time'])
tavg_3hr_ctm = tavg_3hr_ctm.drop(['cubed_sphere', 'contacts', 'orientation', 'anchor', 'TAITIME', 'PS'])
tavg_3hr_ctm = tavg_3hr_ctm.mean(dim=['time']).expand_dims(dim='time', axis=0)
delayed = tavg_3hr_ctm.to_netcdf("data/GEOS.fp.asm.tavg_3hr_ctm_c0720_v72.20210401_0130.V01.nc", compute=False)
with ProgressBar():
    delayed.compute()
```

Sample 3-hour winds:
* http://geoschemdata.wustl.edu/ExtData/GEOS_0.25x0.3125/GEOS_FP_Native/Y2022/M04/D01/GEOS.fp.asm.tavg3_3d_asm_Nv.20210401_0130.V01.nc4
