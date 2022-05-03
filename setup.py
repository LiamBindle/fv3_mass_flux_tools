from setuptools import setup

setup(
    name='fv3_mass_flux_tools',
    version='0.0.0',
    author="Liam Bindle",
    author_email="liam.bindle@gmail.com",
    description="Utilities for working with FV3 mass fluxes.",
    url="https://github.com/LiamBindle/fv3_mass_flux_tools",
    project_urls={
        "Bug Tracker": "https://github.com/LiamBindle/fv3_mass_flux_tools/issues",
    },
    packages=['fv3_mass_flux_tools'],
    install_requires=[
        'numpy',
        'netcdf4',
        'xarray',
        'dask',
        'tqdm',
    ],
)
