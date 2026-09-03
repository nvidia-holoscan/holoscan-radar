# syntax=docker/dockerfile:1

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Development container for radar.
#
# Extends the Holoscan SDK image with the build tools needed to configure,
# compile, and test the module from a live source mount. No source is copied
# into the image; the project tree is bind-mounted by the holoscan-cli at
# /workspace/holoscan-radar when the container is launched.
#
# Build & run via the holoscan-cli wrapper (recommended):
#   ./holohub run-container
#
# Manual build (without the CLI):
#   docker build -t holoscan-radar .

# NGC publishes v4.3.0 as the cuda13 tag (no -dgpu variant at this version).
ARG BASE_IMAGE=nvcr.io/nvidia/clara-holoscan/holoscan:v4.3.0-cuda13
FROM ${BASE_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive

# Non-root development user (team convention: work as `dev`, not root).
# The base Holoscan image already ships a UID/GID 1000 account (`ubuntu`), so we
# free those ids and reassign them to `dev` — matching the typical host user and
# keeping the bind-mounted source tree writable. Hosts whose user isn't 1000 are
# handled at container-create time by `updateRemoteUserUID` (see .devcontainer/).
# The default USER stays root so `./holohub run-container` and packaging are
# unaffected.
ARG DEV_USER=dev
ARG DEV_UID=1000
ARG DEV_GID=1000

# Install holoscan-cli developer tools
ARG HOLOSCAN_CLI_INSTALL_ARGS="--pre --extra-index-url https://pypi.nvidia.com holoscan-cli>4.2.0"
# Disable pathname expansion while preserving the documented whitespace-separated
# pip argument override.
RUN if ! python3 -m holoscan_cli --help 2>/dev/null | grep -qw build; then \
        set -f; \
        python3 -m pip install ${HOLOSCAN_CLI_INSTALL_ARGS} \
    ; fi

# TODO: add module-specific runtime / build dependencies below.
#
# The last three serve `./holohub test` (ci/container.ctest):
#   ninja-build — the driver's CMake generator
#   xvfb        — holoscan-cli wraps ctest in `xvfb-run -a` unless --no-xvfb.
#                 This satisfies that wrapper; it does NOT make Holoviz work
#                 headlessly (that needs Vulkan), so smoke runs keep
#                 visualization disabled.
#   ccache      — compiler cache; the driver wires it in via
#                 CMAKE_<LANG>_COMPILER_LAUNCHER and the holohub wrapper points
#                 CCACHE_DIR at the gitignored .cache/, so it survives the
#                 build-test/ wipe.
RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        libgtest-dev \
        clang-format \
        pybind11-dev \
        sudo \
        ninja-build \
        xvfb \
        ccache \
    && rm -rf /var/lib/apt/lists/*

# Python tooling: pytest-timeout for tests, build + scikit-build-core for
# `./holohub package --pkg-generator WHEEL` (drives `python -m build`).
RUN pip install --no-cache-dir pytest-timeout build scikit-build-core

# Create the `dev` user with passwordless sudo at UID/GID ${DEV_UID}/${DEV_GID},
# first freeing those ids if the base image assigned them to another account.
RUN set -eux; \
    if getent passwd "${DEV_UID}" >/dev/null 2>&1; then \
        userdel -r "$(getent passwd "${DEV_UID}" | cut -d: -f1)" 2>/dev/null || true; \
    fi; \
    if getent group "${DEV_GID}" >/dev/null 2>&1; then \
        groupdel "$(getent group "${DEV_GID}" | cut -d: -f1)" 2>/dev/null || true; \
    fi; \
    groupadd --gid "${DEV_GID}" "${DEV_USER}"; \
    useradd --uid "${DEV_UID}" --gid "${DEV_GID}" --create-home --shell /bin/bash "${DEV_USER}"; \
    echo "${DEV_USER} ALL=(root) NOPASSWD:ALL" > "/etc/sudoers.d/99-${DEV_USER}"; \
    chmod 0440 "/etc/sudoers.d/99-${DEV_USER}"
