# syntax=docker/dockerfile:1

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
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
# UID/GID are build args so the dev container can match them to the host user,
# keeping the bind-mounted source tree writable (see .devcontainer/). Defaults to
# 1000 for a plain `docker build`; the dev container overrides via Compose.
ARG DEV_USER=dev
ARG DEV_UID=1000
ARG DEV_GID=1000

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
        sudo \
    && rm -rf /var/lib/apt/lists/*

# Python tooling: pytest-timeout for tests, build + scikit-build-core for
# `./holohub package --pkg-generator WHEEL` (drives `python -m build`).
RUN pip install --no-cache-dir pytest-timeout build scikit-build-core

# Create the `dev` user with passwordless sudo. Idempotent against base images
# that already ship a user/group at either the requested name or id.
# The default USER stays root so `./holohub run-container` and packaging are
# unaffected; the dev container opts in to `dev` via remoteUser.
RUN set -eux; \
    if getent passwd "${DEV_USER}" >/dev/null 2>&1; then \
        current_uid="$(getent passwd "${DEV_USER}" | cut -d: -f3)"; \
        current_gid="$(getent passwd "${DEV_USER}" | cut -d: -f4)"; \
        [ "${current_uid}" = "${DEV_UID}" ] && [ "${current_gid}" = "${DEV_GID}" ] || userdel -r "${DEV_USER}"; \
    fi; \
    if getent passwd "${DEV_UID}" >/dev/null 2>&1; then \
        user="$(getent passwd "${DEV_UID}" | cut -d: -f1)"; \
        [ "${user}" = "${DEV_USER}" ] || userdel -r "${user}"; \
    fi; \
    if getent group "${DEV_USER}" >/dev/null 2>&1; then \
        gid="$(getent group "${DEV_USER}" | cut -d: -f3)"; \
        [ "${gid}" = "${DEV_GID}" ] || groupdel "${DEV_USER}" || true; \
    fi; \
    if getent group "${DEV_GID}" >/dev/null 2>&1; then \
        group="$(getent group "${DEV_GID}" | cut -d: -f1)"; \
        [ "${group}" = "${DEV_USER}" ] || groupdel "${group}" || true; \
    fi; \
    getent group "${DEV_GID}" >/dev/null 2>&1 || groupadd --gid "${DEV_GID}" "${DEV_USER}"; \
    getent passwd "${DEV_UID}" >/dev/null 2>&1 || \
        useradd --uid "${DEV_UID}" --gid "${DEV_GID}" -m -s /bin/bash "${DEV_USER}"; \
    install -d -m 0755 -o "${DEV_UID}" -g "${DEV_GID}" "/home/${DEV_USER}"; \
    echo "${DEV_USER} ALL=(root) NOPASSWD:ALL" > "/etc/sudoers.d/99-${DEV_USER}"; \
    chmod 0440 "/etc/sudoers.d/99-${DEV_USER}"
