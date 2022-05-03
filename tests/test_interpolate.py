import numpy as np
from fv3_mass_flux_tools.interpolate import *

def test_unstagger_cubic():
    x = np.ones((4,))
    x_unstaggered = unstagger_cubic_interp(x)
    assert np.allclose(x_unstaggered, [1, 1, 1])

    x = np.array([1, 0, 0, 0])
    x_unstaggered = unstagger_cubic_interp(x)
    assert np.allclose(x_unstaggered, [0.5, -0.0625, 0])

    x = np.array([0, 1, 0, 0])
    x_unstaggered = unstagger_cubic_interp(x)
    assert np.allclose(x_unstaggered, [0.5, 0.5625, 0])

    x = np.array([0, 0, 1, 0])
    x_unstaggered = unstagger_cubic_interp(x)
    assert np.allclose(x_unstaggered, [0, 0.5625, 0.5])

    x = np.array([0, 0, 0, 1])
    x_unstaggered = unstagger_cubic_interp(x)
    assert np.allclose(x_unstaggered, [0, -0.0625, 0.5])

def test_unstagger_linear():
    x = np.ones((4,))
    x_unstaggered = unstagger_linear_interp(x)
    assert np.allclose(x_unstaggered, [1, 1, 1])

    x = np.array([1, 0, 0, 0])
    x_unstaggered = unstagger_linear_interp(x)
    assert np.allclose(x_unstaggered, [0.5, 0, 0])

    x = np.array([0, 1, 0, 0])
    x_unstaggered = unstagger_linear_interp(x)
    assert np.allclose(x_unstaggered, [0.5, 0.5, 0])

    x = np.array([0, 0, 1, 0])
    x_unstaggered = unstagger_linear_interp(x)
    assert np.allclose(x_unstaggered, [0, 0.5, 0.5])

    x = np.array([0, 0, 0, 1])
    x_unstaggered = unstagger_linear_interp(x)
    assert np.allclose(x_unstaggered, [0, 0, 0.5])
