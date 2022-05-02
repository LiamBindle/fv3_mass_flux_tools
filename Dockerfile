FROM python:3.9.12

COPY ./fv3_mass_flux_tools /install_fv3_mass_flux_tools/fv3_mass_flux_tools
COPY ./setup.py /install_fv3_mass_flux_tools

RUN pip install /install_fv3_mass_flux_tools

ENTRYPOINT [ "python", "-m", "fv3_mass_flux_tools.process" ]
