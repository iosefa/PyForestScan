FROM jupyter/base-notebook:latest
LABEL maintainer="Iosefa Percival"
LABEL repo="https://github.com/iosefa/PyForestScan"

USER root
RUN apt-get update -y && apt-get install -y \
    gcc g++ make \
    libgdal-dev libgl1 sqlite3 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}"

USER 1000
RUN mamba install -c conda-forge sqlite gdal pdal -y && \
    pip install --no-cache-dir pyforestscan jupyter-server-proxy && \
    mamba update -c conda-forge -y && \
    jupyter server extension enable --sys-prefix jupyter_server_proxy

RUN mkdir ./examples
COPY /docs/examples ./examples

RUN mkdir ./example_data
COPY /docs/example_data ./example_data

ENV PROJ_LIB='/opt/conda/share/proj'
ENV JUPYTER_ENABLE_LAB=yes

USER root
RUN chown -R ${NB_UID} ${HOME}
USER ${NB_USER}