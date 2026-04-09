# hash:sha256:07fa441915b694378062160a8bfb899de444b0746f3c98c2d011ec361e5c5888
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/mambaforge3:23.1.0-4-python3.10.12-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG AWS_ACCESS_KEY_ID
ARG AWS_DEFAULT_REGION
ARG AWS_SECRET_ACCESS_KEY

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-ask-pass /

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential=12.9ubuntu3 \
        git \
        curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -U --no-cache-dir \
    arviz==0.21.0 \
    matplotlib==3.9.1 \
    scipy==1.14.1 \
    seaborn==0.13.2 \
    scikit-learn==1.5.2 \
    codeocean==0.14.0 \
    PyPDF2==3.0.1 \
    aind-dynamic-foraging-data-utils==0.1.4
    

RUN pip3 install -U --no-cache-dir \
    numpy==1.26.4 \
    pandas==2.2.1 \
    pystan==3.9.0 \
    statsmodels==0.14.5 \
    natsort==8.4.0

COPY postInstall /
RUN /postInstall
