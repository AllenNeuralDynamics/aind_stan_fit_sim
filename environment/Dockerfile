# hash:sha256:f74e2a48695dcf108431d8125d628b00c202168ea2e2e99e10a015d39ad536c7
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/mambaforge3:23.1.0-4-python3.10.12-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-ask-pass /

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential=12.9ubuntu3 \
    && rm -rf /var/lib/apt/lists/*

RUN --mount=type=secret,id=git-access-token \
    GIT_ACCESS_TOKEN="$(cat /run/secrets/git-access-token)" \
    pip3 install -U --no-cache-dir \
    aind-dynamic-foraging-data-utils==0.1.4 \
    arviz==0.21.0 \
    matplotlib==3.9.1 \
    natsort==8.4.0 \
    numpy==1.26.4 \
    open-ephys-python-tools==0.1.10 \
    pandas==2.2.1 \
    pypdf2==3.0.1 \
    pystan==3.9.0 \
    scikit-learn==1.5.2 \
    scipy==1.14.1 \
    seaborn==0.13.2 \
    spikeinterface[full]==0.103.0 \
    statsmodels==0.14.5 \
    -e git+https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-basic-analysis@b3e2e4f29f062764f9a40bfdc630dd0b6f039989#egg=aind-dynamic-foraging-basic-analysis

COPY postInstall /
RUN --mount=type=secret,id=secrets . /run/secrets/secrets \
    && /postInstall
