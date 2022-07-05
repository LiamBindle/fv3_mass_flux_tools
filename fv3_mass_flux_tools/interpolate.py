import numpy as np

def unstagger_cubic_interp(x):
    assert len(x.shape) == 1
    assert len(x) >= 4
    a1 = 0.5625
    a2 = -0.0625
    x_interp = np.zeros((len(x)-1,))
    x_interp[0] = 0.5 * (x[0] + x[1])
    x_interp[-1] = 0.5 * (x[-2] + x[-1])
    x0 = x[:-3]
    x1 = x[1:-2]
    x2 = x[2:-1]
    x3 = x[3:]
    x_interp[1:-1] = a2 * (x0 + x3) + a1 * (x1 + x2)
    return x_interp


def unstagger_linear_interp(x):
    assert len(x.shape) == 1
    assert len(x) >= 4
    x_interp = 0.5 * (x[0:-1] + x[1:])
    return x_interp


def cgrid_to_agrid(uc, vc, cubic=True):
    interp = unstagger_cubic_interp if cubic else unstagger_linear_interp
    ua = np.apply_along_axis(interp, -1, uc)
    va = np.apply_along_axis(interp, -2, vc)
    return ua, va

def agrid_to_cgrid_interior_only(ua, va, cubic=True):
    interp = unstagger_cubic_interp if cubic else unstagger_linear_interp
    uc = np.apply_along_axis(interp, -1, ua)
    vc = np.apply_along_axis(interp, -2, va)
    return uc, vc
