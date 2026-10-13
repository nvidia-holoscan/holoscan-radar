# syntax=docker/dockerfile:1

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Development container for radar.
#
# Extends the Holoscan SDK image with the build tools needed to configure,
# compile, and test the module from a live source mount. No source is copied
# into the image apart from pyproject.toml for CLI dependency installation;
# the project tree is bind-mounted when the container is launched.
#
# Build & run via the Holoscan CLI:
#   holoscan run-container --base-img <HSDK-5-image>
#
# Manual build (without the CLI):
#   docker build --build-arg BASE_IMAGE=<HSDK-5-image> -t holoscan-radar .

# Supply the temporary HSDK 5 image with --base-img or --build-arg BASE_IMAGE.
# The default remains the published SDK image until a public HSDK 5 image exists.
ARG BASE_IMAGE=nvcr.io/nvidia/clara-holoscan/holoscan:v4.3.0-cuda13
FROM ${BASE_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive

# Non-root development user (team convention: work as `dev`, not root).
# The base Holoscan image already ships a UID/GID 1000 account (`ubuntu`), so we
# free those ids and reassign them to `dev` — matching the typical host user and
# keeping the bind-mounted source tree writable. Hosts whose user isn't 1000 are
# handled at container-create time by `updateRemoteUserUID` (see .devcontainer/).
# The default USER stays root so `holoscan run-container` and packaging are
# unaffected.
ARG DEV_USER=dev
ARG DEV_UID=1000
ARG DEV_GID=1000

# Install the CLI and Python test dependencies declared by this module.
# pip 25.1+ reads PEP 735 groups without building the radar wheel.
COPY pyproject.toml /opt/holoscan-radar/pyproject.toml
RUN python3 -m pip install --no-cache-dir --upgrade 'pip>=25.1' \
    && python3 -m pip install --no-cache-dir \
        --extra-index-url https://pypi.nvidia.com \
        --group /opt/holoscan-radar/pyproject.toml:cli \
        --group /opt/holoscan-radar/pyproject.toml:test

# TODO: add module-specific runtime / build dependencies below.
#
# The last three serve `holoscan test` (ci/container.ctest):
#   ninja-build — the driver's CMake generator
#   xvfb        — holoscan-cli wraps ctest in `xvfb-run -a` unless --no-xvfb.
#                 This satisfies that wrapper; it does NOT make Holoviz work
#                 headlessly (that needs Vulkan), so smoke runs keep
#                 visualization disabled.
#   ccache      — compiler cache; the driver wires it in via
#                 CMAKE_<LANG>_COMPILER_LAUNCHER and the CTest driver points
#                 CCACHE_DIR at the gitignored .cache/, so it survives the
#                 build-test/ wipe.
#   libyaml-cpp-dev — demo apps parse YAML configs with yaml-cpp. HSDK 4.x
#                 exported it from the SDK install; HSDK 5 keeps yaml-cpp
#                 private to config_io, so the apps need the system package.
#   libtinyxml2-dev — CPHDReaderOp parses CPHD XML metadata; the HSDK 5
#                 development image no longer supplies this dependency.
#   openssh-client — git-over-SSH and host tooling (dotfiles clone, remotes);
#                 the runtime base image does not ship ssh.
RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        libgtest-dev \
        libyaml-cpp-dev \
        libtinyxml2-dev \
        openssh-client \
        clang-format \
        pybind11-dev \
        sudo \
        ninja-build \
        xvfb \
        ccache \
    && rm -rf /var/lib/apt/lists/*

# Wheel packaging tooling: `holoscan package --pkg-generator WHEEL`
# drives `python -m build`; its backend is declared in pyproject.toml.
RUN pip install --no-cache-dir build scikit-build-core

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
