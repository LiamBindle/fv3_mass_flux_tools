from audioop import avg
import numpy as np
from dataclasses import dataclass
import mf2w

@dataclass
class meridion_difference:
    angle1: float
    angle2: float
    answer: list

    def run_test(self):
        angles = np.array([self.angle1, self.angle2, self.angle1], dtype=float)
        angles_rad = np.deg2rad(angles)
        answer = np.array([self.answer]*2, dtype=float)
        answer_rad = np.deg2rad(answer)
        result = mf2w.convex_central_angle(angles)
        result_rad = mf2w.convex_central_angle(angles_rad, deg=False)
        assert np.allclose(answer, result),f"result: {result.tolist()}; expected: {answer.tolist()}"
        assert np.allclose(answer_rad, result_rad),f"result: {result_rad.tolist()}; expected: {answer_rad.tolist()}"


def test_meridional_difference_simple():
    meridion_difference(angle1=0, angle2=5, answer=5).run_test()
    meridion_difference(angle1=5, angle2=11, answer=6).run_test()
    meridion_difference(angle1=-121, angle2=-130, answer=9).run_test()
    meridion_difference(angle1=250, angle2=253, answer=3).run_test()
    meridion_difference(angle1=359, angle2=358, answer=1).run_test()


def test_meridional_difference_across_pm():
    meridion_difference(angle1=359, angle2=5, answer=6).run_test()
    meridion_difference(angle1=5, angle2=300, answer=65).run_test()


def test_great_circle_distance():
    lat0 = np.array([78.0])
    lon0 = np.array([-110.1])
    lat1 = np.array([72.0])
    lon1 = np.array([70.0])
    answer = np.array([3349970])
    result = mf2w.great_circle_distance(lon0, lat0, lon1, lat1)
    assert np.allclose(answer, result, rtol=0.01)

def test_unstagger_cubic():
    x = np.ones((4,))
    x_unstaggered = mf2w.unstagger_cubic_interp(x)
    assert np.allclose(x_unstaggered, [1, 1, 1])

    x = np.array([1, 0, 0, 0])
    x_unstaggered = mf2w.unstagger_cubic_interp(x)
    assert np.allclose(x_unstaggered, [0.5, -0.0625, 0])

    x = np.array([0, 1, 0, 0])
    x_unstaggered = mf2w.unstagger_cubic_interp(x)
    assert np.allclose(x_unstaggered, [0.5, 0.5625, 0])

    x = np.array([0, 0, 1, 0])
    x_unstaggered = mf2w.unstagger_cubic_interp(x)
    assert np.allclose(x_unstaggered, [0, 0.5625, 0.5])

    x = np.array([0, 0, 0, 1])
    x_unstaggered = mf2w.unstagger_cubic_interp(x)
    assert np.allclose(x_unstaggered, [0, -0.0625, 0.5])

def test_unstagger_linear():
    x = np.ones((4,))
    x_unstaggered = mf2w.unstagger_linear_interp(x)
    assert np.allclose(x_unstaggered, [1, 1, 1])

    x = np.array([1, 0, 0, 0])
    x_unstaggered = mf2w.unstagger_linear_interp(x)
    assert np.allclose(x_unstaggered, [0.5, 0, 0])

    x = np.array([0, 1, 0, 0])
    x_unstaggered = mf2w.unstagger_linear_interp(x)
    assert np.allclose(x_unstaggered, [0.5, 0.5, 0])

    x = np.array([0, 0, 1, 0])
    x_unstaggered = mf2w.unstagger_linear_interp(x)
    assert np.allclose(x_unstaggered, [0, 0.5, 0.5])

    x = np.array([0, 0, 0, 1])
    x_unstaggered = mf2w.unstagger_linear_interp(x)
    assert np.allclose(x_unstaggered, [0, 0, 0.5])


def test_dx_dx_calculation_first_face():
    import xarray as xr
    grid = xr.open_mfdataset([f"data/c720.tile{n}.nc" for n in range(1,7)], concat_dim='nf', combine='nested').isel(nf=0)
    dx, dy = mf2w.get_dx_dy(grid)
    avg_dx = np.mean(dx) / 1e3
    avg_dy = np.mean(dy) / 1e3
    assert 12 < avg_dx < 13, f"false: 12 km < avg_dx={avg_dx} km < 13 km"
    assert 12 < avg_dy < 13, f"false: 12 km < avg_dy={avg_dy} km < 13 km"


def test_mass_flux_conversion_first_face():
    import xarray as xr
    tavg_1hr_ctm = xr.open_dataset("data/GEOS.fp.asm.tavg_1hr_ctm_c0720_v72.20210401_0030.V01.nc4").isel(nf=0)
    grid = xr.open_mfdataset([f"data/c720.tile{n}.nc" for n in range(1,7)], concat_dim='nf', combine='nested').isel(nf=0)
    uc, vc = mf2w.mass_fluxes_to_winds(tavg_1hr_ctm, grid)
    #magnitude = np.sqrt(uc*uc + vc*vc)
    #magnitude = np.sqrt(uc.isel(Ydim=slice(0, -1))**2 + vc.isel(Xdim=slice(0, -1))**2)
    ua, va = mf2w.cgrid_to_agrid(uc, vc)
    magnitude = np.sqrt(ua*ua + va*va)
    avg_magnitude_surface = np.nanmean(magnitude[0,-1,:,:])
    assert 5.0 < avg_magnitude_surface < 7.0,f"calculated avg. wind speed: {avg_magnitude_surface} m/s; expected: ~7.5 m/s"  # actual magnitude should be ~7.5 m/s
