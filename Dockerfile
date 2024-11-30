FROM jupyter/base-notebook:latest
LABEL maintainer="Iosefa Percival"
LABEL repo="https://github.com/iosefa/PyForestScan"

USER 1000

RUN mamba install -c conda-forge pdal gdal -y && \
    pip install -U pyforestscan jupyter-server-proxy && \
    mamba update -c conda-forge -y && \
    jupyter server extension enable --sys-prefix jupyter_server_proxy && \
    fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}"

RUN mkdir ./examples
COPY /docs/examples ./examples

ENV PROJ_LIB='/opt/conda/share/proj'
ENV JUPYTER_ENABLE_LAB=yes

USER root
RUN chown -R ${NB_UID} ${HOME}
USER ${NB_USER}