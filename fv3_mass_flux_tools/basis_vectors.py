import numpy as np

# region "Cube-sphere basis vectors from Ronchi et al. (1995)"
# doi: https://doi.org/10.1006/jcph.1996.0047


def auxiliary_delta(X, Y):
    # Eqn. (3)
    return 1 + X**2 + Y**2


def auxiliary_C(X):
    # Eqn. (4)
    return np.sqrt(1 + X**2)


def auxiliary_D(Y):
    # Eqn. (5)
    return np.sqrt(1 + Y**2)


def spherical_to_local_equatorial(X, Y):
    # Eqn. (7)
    delta = auxiliary_delta(X, Y)
    C = auxiliary_C(X)
    D = auxiliary_D(Y)
    M = np.array([[np.zeros_like(X), C*D/np.sqrt(delta)], [-np.ones_like(X), X*Y/np.sqrt(delta)]])
    M = np.moveaxis(M, 0, -1)   # roll back once
    M = np.moveaxis(M, 0, -1)   # roll back again
    return M


def spherical_to_local_north_pole(X, Y):
    # Eqn. (12)
    delta = auxiliary_delta(X, Y)
    C = auxiliary_C(X)
    D = auxiliary_D(Y)
    M = np.array([[D*X, -D*Y/np.sqrt(delta)], [C*Y, C*X/np.sqrt(delta)]]) / np.sqrt(delta - 1)
    M = np.moveaxis(M, 0, -1)   # roll back once
    M = np.moveaxis(M, 0, -1)   # roll back again
    return M


def spherical_to_local_south_pole(X, Y):
    # Eqn. (14)
    delta = auxiliary_delta(X, Y)
    C = auxiliary_C(X)
    D = auxiliary_D(Y)
    M = np.array([[-D*X, D*Y/np.sqrt(delta)], [-C*Y, -C*X/np.sqrt(delta)]]) / np.sqrt(delta - 1)
    M = np.moveaxis(M, 0, -1)   # roll back once
    M = np.moveaxis(M, 0, -1)   # roll back again
    return M


def XY_I(phi, theta):
    # Eqn. (6)
    X = np.tan(phi)
    Y = 1/(np.tan(theta) * np.cos(phi))
    return X, Y


def XY_II(phi, theta):
    # Eqn. (8)
    X = -1/np.tan(phi)
    Y = 1/(np.tan(theta) * np.sin(phi))
    return X, Y


def XY_III(phi, theta):
    # Eqn. (9)
    X = np.tan(phi)
    Y = -1/(np.tan(theta) * np.cos(phi))
    return X, Y


def XY_IV(phi, theta):
    # Eqn. (10)
    X = -1/np.tan(phi)
    Y = -1/(np.tan(theta) * np.sin(phi))
    return X, Y


def XY_V(phi, theta):
    # Eqn. (11)
    X = np.tan(theta) * np.sin(phi)
    Y = -np.tan(theta) * np.cos(phi)
    return X, Y


def XY_VI(phi, theta):
    # Eqn. (13)
    X = -np.tan(theta) * np.sin(phi)
    Y = -np.tan(theta) * np.cos(phi)
    return X, Y


def spherical_to_ronchi(phi, theta, ronchi_face: int):
    # Ronchi basis to lon,lat
    if ronchi_face < 5:
        if ronchi_face == 1:
            X, Y = XY_I(phi, theta)
        elif ronchi_face == 2:
            X, Y = XY_II(phi, theta)
        elif ronchi_face == 3:
            X, Y = XY_III(phi, theta)
        elif ronchi_face == 4:
            X, Y = XY_IV(phi, theta)
        R = spherical_to_local_equatorial(X, Y)
    elif ronchi_face == 5:
        X, Y = XY_V(phi, theta)
        R = spherical_to_local_north_pole(X, Y)
    elif ronchi_face == 6:
        X, Y = XY_VI(phi, theta)
        R = spherical_to_local_south_pole(X, Y)
    R = np.flip(R, axis=-1)                 # RH: colat,lon to lon,colat
    Rm = R @ np.array([[1, 0], [0, -1]])    # RH: lon,colat to lon,lat
    return Rm


# endregion


FACE_INDEX_TRANSLATION_GMAO_TO_RONCHI = {
    0: 1,
    1: 2,
    2: 5,
    3: 3,
    4: 4,
    5: 6,
}


FACE_ROTATION_GMAO_TO_RONCHI = {
    0: np.array([[1, 0],[ 0, 1]]),
    1: np.array([[1, 0],[ 0, 1]]),
    2: np.array([[0,-1],[ 1, 0]]),
    3: np.array([[0, 1],[-1, 0]]),
    4: np.array([[0, 1],[-1, 0]]),
    5: np.array([[1, 0],[ 0, 1]]),
}


def spherical_to_gmao(phi, theta, gmao_face: int):
    ronchi_face = FACE_INDEX_TRANSLATION_GMAO_TO_RONCHI[gmao_face]
    gmao_to_ronchi = FACE_ROTATION_GMAO_TO_RONCHI[gmao_face]
    Rm = spherical_to_ronchi(phi, theta, ronchi_face=ronchi_face)
    G  = np.linalg.inv(gmao_to_ronchi) @ Rm
    mag = np.sqrt(np.einsum('ijkl,ijkl->ijk', G, G))
    return G/np.expand_dims(mag, -1)


def uv_to_eastnorth_rotation_matrix(gmao_face: int):
    return FACE_ROTATION_GMAO_TO_RONCHI[gmao_face]
