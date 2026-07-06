# syntax=docker/dockerfile:1

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Development container for radar.
#
# Extends the Holoscan SDK image with the build tools needed to configure,
# compile, and test the module from a live source mount. No source is copied
# into the image; the project tree is bind-mounted by the holoscan-cli at
# /workspace/radar when the container is launched.
#
# Build & run via the holoscan-cli wrapper (recommended):
#   ./holohub run-container
#
# Manual build (without the CLI):
#   docker build -t holoscan-radar .

# Holoscan SDK 5.x pre-release development container (SDK preinstalled).
# 5.x dev containers are published per-arch on the internal GitLab registry and
# tagged by the 9-char main-5x git SHA (there is no friendly 5.0.0 release yet).
# Note the x86_64 repo has no `-dgpu` suffix (x86_64 is dgpu-only); the aarch64
# repo is `dev-cu13-aarch64-dgpu`. Not every commit is published, so the tag is
# the newest main-5x commit that has a built image. Pulling it requires:
#   docker login gitlab-master.nvidia.com:5005
# The holohub wrapper forwards this same reference via HOLOSCAN_CLI_BASE_IMAGE;
# override BASE_IMAGE (or that env) to bump the pin or target another arch.
ARG BASE_IMAGE_TAG=570e5ec1c
ARG BASE_IMAGE=gitlab-master.nvidia.com:5005/holoscan/holoscan-sdk/dev-cu13-x86_64:${BASE_IMAGE_TAG}
FROM ${BASE_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive

# Install holoscan-cli developer tools
ARG HOLOSCAN_CLI_INSTALL_ARGS="--pre --extra-index-url https://pypi.nvidia.com holoscan-cli>4.2.0"
RUN if ! python3 -m holoscan_cli --help 2>/dev/null | grep -qw build; then \
    python3 -m pip install ${HOLOSCAN_CLI_INSTALL_ARGS} \
    ; fi

# TODO: add module-specific runtime / build dependencies below.
RUN apt-get update \
    && apt-get install --no-install-recommends -y \
    libgtest-dev \
    clang-format \
    pybind11-dev \
    && rm -rf /var/lib/apt/lists/*

# Python tooling: pytest-timeout for tests, build + scikit-build-core for
# `./holohub package --pkg-generator WHEEL` (drives `python -m build`).
RUN pip install --no-cache-dir pytest-timeout build scikit-build-core
