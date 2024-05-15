# hash:sha256:a7fcc65dee8f866645954da10e381c3bc9cf5915bd451b87bc1876e4c3057bbe
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/mambaforge3:23.1.0-4-python3.10.12-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG AWS_ACCESS_KEY_ID
ARG AWS_DEFAULT_REGION
ARG AWS_SECRET_ACCESS_KEY

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential=12.9ubuntu3 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -U --no-cache-dir \
    arviz \
    matplotlib \
    scipy \
    seaborn \
    sklearn

RUN pip3 install -U --no-cache-dir \
    numpy==1.26.4 \
    pandas==2.2.1 \
    pystan==3.9.0 \
    statsmodels
