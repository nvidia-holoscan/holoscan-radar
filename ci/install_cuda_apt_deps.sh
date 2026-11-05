#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Install the prerequisites for building and packaging Holoscan Radar on a
# CUDA devel image (for example nvcr.io/nvidia/cuda:13.0.2-devel-ubuntu24.04)
# with the Holoscan SDK from the CUDA APT repository. See
# DEVELOPER.md#cuda-devel-image-with-the-apt-sdk. Safe to re-run.
#
# Override any of these through the environment:
#   HOLOSCAN_MIN_VERSION  minimum `holoscan` Debian package version
#   MATX_VERSION          MatX tag built from source (no APT package exists)
#   MATX_PREFIX           MatX install prefix
#   TOOLS_VENV            Python venv that receives CMake and the Holoscan CLI
#   CMAKE_SPEC            pip requirement for CMake (Ubuntu 24.04 ships 3.28)

set -euo pipefail

HOLOSCAN_MIN_VERSION="${HOLOSCAN_MIN_VERSION:-5.0.0}"
MATX_VERSION="${MATX_VERSION:-v1.0.0}"
MATX_PREFIX="${MATX_PREFIX:-/usr/local}"
TOOLS_VENV="${TOOLS_VENV:-/opt/holoscan-radar-tools}"
CMAKE_SPEC="${CMAKE_SPEC:-cmake>=4,<5}"

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export DEBIAN_FRONTEND=noninteractive

sudo=""
if [[ "$(id -u)" -ne 0 ]]; then
    sudo="sudo"
fi

${sudo} apt-get update

# CUDA images already configure the CUDA APT repository; other Ubuntu hosts
# need NVIDIA's cuda-keyring package first.
if [[ -z "$(apt-cache policy holoscan | awk '/Candidate:/ && $2 != "(none)" {print $2}')" ]]; then
    . /etc/os-release
    case "$(uname -m)" in
        x86_64) cuda_arch=x86_64 ;;
        aarch64) cuda_arch=sbsa ;;
        *) echo "error: unsupported architecture $(uname -m)" >&2; exit 1 ;;
    esac
    ${sudo} apt-get install --no-install-recommends -y ca-certificates curl
    keyring_deb="$(mktemp --suffix=.deb)"
    curl -fsSL -o "${keyring_deb}" \
        "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu${VERSION_ID//./}/${cuda_arch}/cuda-keyring_1.1-1_all.deb"
    ${sudo} dpkg -i "${keyring_deb}"
    rm -f "${keyring_deb}"
    ${sudo} apt-get update
fi

${sudo} apt-get install --no-install-recommends -y \
    holoscan \
    libyaml-cpp-dev \
    libtinyxml2-dev \
    libgtest-dev \
    ninja-build \
    git \
    python3-venv \
    python3-pip

holoscan_version="$(dpkg-query -W -f='${Version}' holoscan)"
if ! dpkg --compare-versions "${holoscan_version}" ge "${HOLOSCAN_MIN_VERSION}"; then
    echo "error: holoscan ${holoscan_version} is older than ${HOLOSCAN_MIN_VERSION}" >&2
    exit 1
fi

# CMake 4 and the Holoscan CLI pinned by pyproject.toml's `cli` group.
if [[ ! -x "${TOOLS_VENV}/bin/python" ]]; then
    ${sudo} python3 -m venv "${TOOLS_VENV}"
fi
${sudo} "${TOOLS_VENV}/bin/python" -m pip install --quiet --upgrade 'pip>=25.1'
${sudo} "${TOOLS_VENV}/bin/python" -m pip install --quiet \
    --extra-index-url https://pypi.nvidia.com \
    "${CMAKE_SPEC}" \
    --group "${repo_root}/pyproject.toml:cli"

# MatX is header-only; build only its CMake package.
matx_config="${MATX_PREFIX}/lib/cmake/matx/matx-config-version.cmake"
if [[ -f "${matx_config}" ]] &&
    grep -q "PACKAGE_VERSION \"${MATX_VERSION#v}\"" "${matx_config}"; then
    echo "MatX ${MATX_VERSION} already installed under ${MATX_PREFIX}"
else
    matx_src="$(mktemp -d)"
    git clone --quiet --depth 1 --branch "${MATX_VERSION}" \
        https://github.com/NVIDIA/MatX.git "${matx_src}/src"
    "${TOOLS_VENV}/bin/cmake" -S "${matx_src}/src" -B "${matx_src}/build" -G Ninja \
        -DCMAKE_INSTALL_PREFIX="${MATX_PREFIX}" \
        -DCMAKE_CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES:-75}" \
        -DMATX_BUILD_TESTS=OFF \
        -DMATX_BUILD_EXAMPLES=OFF \
        -DMATX_BUILD_DOCS=OFF \
        -DMATX_BUILD_BENCHMARKS=OFF
    ${sudo} "${TOOLS_VENV}/bin/cmake" --install "${matx_src}/build"
    rm -rf "${matx_src}"
fi

echo "Installed holoscan ${holoscan_version} and MatX ${MATX_VERSION}."
echo "Add ${TOOLS_VENV}/bin to PATH for cmake and the Holoscan CLI."
