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


def test_dx_dx_calculation():
    import xarray as xr
    grid = xr.open_mfdataset([f"data/c720.tile{n}.nc" for n in range(1,7)], concat_dim='nf', combine='nested')
    dx, dy = mf2w.get_dx_dy(grid, verbose=True)
    avg_dx = np.mean(dx) / 1e3
    avg_dy = np.mean(dy) / 1e3
    assert 12 < avg_dx < 13, f"false: 12 km < avg_dx={avg_dx} km < 13 km"
    assert 12 < avg_dy < 13, f"false: 12 km < avg_dy={avg_dy} km < 13 km"


def test_mass_flux_conversion_first_face():
    import xarray as xr
    tavg_1hr_ctm = xr.open_dataset("data/GEOS.fp.asm.tavg_1hr_ctm_c0720_v72.20210401_0030.V01.nc4")
    grid = xr.open_mfdataset([f"data/c720.tile{n}.nc" for n in range(1,7)], concat_dim='nf', combine='nested')
    uc, vc = mf2w.mass_fluxes_to_winds(tavg_1hr_ctm, grid, True)
    magnitude = np.sqrt(uc*uc + vc*vc)
    avg_magnitude_surface = np.mean(magnitude[0,-1,:,:])
    assert 5.0 < avg_magnitude_surface < 7.0,f"calculated avg. wind speed: {avg_magnitude_surface} m/s; expected: ~7.5 m/s"  # actual magnitude should be ~7.5 m/s
