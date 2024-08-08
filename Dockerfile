# Base image with apex and transformer engine, but without NeMo or Megatron-LM.
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:24.02-py3
FROM ${BASE_IMAGE} AS bionemo2-base

# Install NeMo dependencies.
WORKDIR /build

ARG MAX_JOBS=4
ENV MAX_JOBS=${MAX_JOBS}

# See NeMo readme for the latest tested versions of these libraries
ARG APEX_COMMIT=810ffae374a2b9cb4b5c5e28eaeca7d7998fca0c
RUN git clone https://github.com/NVIDIA/apex.git && \
  cd apex && \
  git checkout ${APEX_COMMIT} && \
  pip install . -v --no-build-isolation --disable-pip-version-check --no-cache-dir \
  --config-settings "--build-option=--cpp_ext --cuda_ext --fast_layer_norm --distributed_adam --deprecated_fused_adam --group_norm"

# Transformer Engine pre-1.7.0. 1.7 standardizes the meaning of bits in the attention mask to match
ARG TE_COMMIT=7d576ed25266a17a7b651f2c12e8498f67e0baea
RUN git clone https://github.com/NVIDIA/TransformerEngine.git && \
  cd TransformerEngine && \
  git fetch origin ${TE_COMMIT} && \
  git checkout FETCH_HEAD && \
  git submodule init && git submodule update && \
  NVTE_FRAMEWORK=pytorch NVTE_WITH_USERBUFFERS=1 MPI_HOME=/usr/local/mpi pip install .

# Install core apt packages.
RUN apt-get update \
  && apt-get install -y \
  libsndfile1 \
  ffmpeg \
  git \
  curl \
  pre-commit \
  sudo \
  && rm -rf /var/lib/apt/lists/*

# Check the nemo dependency for causal conv1d and make sure this checkout
# tag matches. If not, update the tag in the following line.
RUN CAUSAL_CONV1D_FORCE_BUILD=TRUE pip --disable-pip-version-check --no-cache-dir install \
  git+https://github.com/Dao-AILab/causal-conv1d.git@v1.2.0.post2

# Mamba dependancy installation
RUN pip --disable-pip-version-check --no-cache-dir install \
  git+https://github.com/state-spaces/mamba.git@v2.0.3



# Create a non-root user to use inside a devcontainer.
ARG USERNAME=bionemo
ARG USER_UID=1000
ARG USER_GID=$USER_UID
RUN groupadd --gid $USER_GID $USERNAME \
  && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
  && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
  && chmod 0440 /etc/sudoers.d/$USERNAME

ENV PATH="/home/bionemo/.local/bin:${PATH}"

FROM bionemo2-base AS dev

COPY requirements-dev.lock ./
RUN sed '/-e/d' requirements-dev.lock > requirements-install.lock \
  && PYTHONDONTWRITEBYTECODE=1 pip install --no-cache-dir --disable-pip-version-check -r requirements-install.lock
RUN rm requirements*.lock

# Create a release image with bionemo2 installed.
FROM bionemo2-base AS release

COPY requirements.lock ./
RUN sed '/-e/d' requirements.lock > requirements-install.lock \
  && PYTHONDONTWRITEBYTECODE=1 pip install --no-cache-dir --disable-pip-version-check -r requirements-install.lock
RUN rm requirements*.lock

# Install 3rd-party deps
COPY ./3rdparty /build
WORKDIR /build/Megatron-LM
RUN pip install --disable-pip-version-check --no-cache-dir .

WORKDIR /build/NeMo
RUN pip install --disable-pip-version-check --no-cache-dir .[all]
WORKDIR /workspace
RUN rm -rf /build

# Install bionemo2 submodules
WORKDIR /workspace/bionemo2/
COPY ./sub-packages /workspace/bionemo2/sub-packages
# Dynamically install the code for each bionemo namespace package.
RUN for sub in $(ls sub-packages/); do pushd sub-packages/${sub} && pip install --no-build-isolation --no-cache-dir --disable-pip-version-check --no-deps . && popd; done

WORKDIR /workspace/bionemo2/
COPY ./scripts ./scripts
COPY ./README.md ./
