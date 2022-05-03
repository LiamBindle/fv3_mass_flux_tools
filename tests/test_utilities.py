from dataclasses import dataclass
import numpy as np
from fv3_mass_flux_tools.utilities import *


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
        result = convex_central_angle(angles)
        result_rad = convex_central_angle(angles_rad, deg=False)
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
    result = great_circle_distance(lon0, lat0, lon1, lat1)
    assert np.allclose(answer, result, rtol=0.01)


def test_dx_dx_calculation_first_face():
    import xarray as xr
    grid = xr.open_mfdataset([f"data/c720.tile1.nc"], concat_dim='nf', combine='nested').isel(nf=0)
    dx, dy = get_dx_dy_grid_spacing(grid)
    avg_dx = np.mean(dx) / 1e3
    avg_dy = np.mean(dy) / 1e3
    assert 12 < avg_dx < 13, f"false: 12 km < avg_dx={avg_dx} km < 13 km"
    assert 12 < avg_dy < 13, f"false: 12 km < avg_dy={avg_dy} km < 13 km"
