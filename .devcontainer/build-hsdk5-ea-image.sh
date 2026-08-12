#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# [EA HSDK5 ONLY] Build the local HSDK 5 base image used by the devcontainer.
# Delete this entire file before the HSDK 5 public release.

set -euo pipefail

readonly HSDK_REPOSITORY="ssh://git@gitlab-master.nvidia.com:12051/holoscan/holoscan-sdk.git"
readonly HSDK_TAG="v5.0.0.1"
readonly OUTPUT_IMAGE="holoscan-radar-hsdk5-ea:${HSDK_TAG}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
HSDK_ROOT="${REPO_ROOT}/.cache/holoscan-sdk-${HSDK_TAG}"

for command in git docker; do
    if ! command -v "${command}" >/dev/null 2>&1; then
        echo "error: '${command}' is required" >&2
        exit 1
    fi
done

if [[ ! -d "${HSDK_ROOT}/.git" ]]; then
    if [[ -e "${HSDK_ROOT}" ]]; then
        echo "error: '${HSDK_ROOT}' exists but is not a Git checkout" >&2
        exit 1
    fi

    git clone --branch "${HSDK_TAG}" --depth 1 \
        "${HSDK_REPOSITORY}" "${HSDK_ROOT}"
fi

CHECKED_OUT_TAG="$(
    git -C "${HSDK_ROOT}" describe --tags --exact-match 2>/dev/null || true
)"
if [[ "${CHECKED_OUT_TAG}" != "${HSDK_TAG}" ]]; then
    echo "error: cached HSDK checkout is not at ${HSDK_TAG}" >&2
    echo "       remove '${HSDK_ROOT}' and run this script again" >&2
    exit 1
fi

if [[ -n "$(git -C "${HSDK_ROOT}" status --porcelain)" ]]; then
    echo "error: cached HSDK checkout has local changes" >&2
    echo "       remove '${HSDK_ROOT}' and run this script again" >&2
    exit 1
fi

echo "[EA HSDK5 ONLY] Building ${HSDK_TAG}"
(
    cd "${HSDK_ROOT}"
    ./public/run build
)

BASE_IMAGE="$(
    cd "${HSDK_ROOT}"
    printf '%s:%s' \
        "$(./public/run get_build_img_name)" \
        "$(./public/run get_git_sha)"
)"
INSTALL_DIR="$(
    cd "${HSDK_ROOT}"
    ./public/run get_install_dir
)"

if [[ ! -d "${HSDK_ROOT}/public/${INSTALL_DIR}" ]]; then
    echo "error: HSDK install tree 'public/${INSTALL_DIR}' was not created" >&2
    exit 1
fi

docker build \
    --file "${SCRIPT_DIR}/sdk-installed.Dockerfile" \
    --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
    --build-arg "HSDK_INSTALL_DIR=${INSTALL_DIR}" \
    --tag "${OUTPUT_IMAGE}" \
    "${HSDK_ROOT}/public"

echo
echo "[EA HSDK5 ONLY] Built ${OUTPUT_IMAGE}"
echo "Rebuild and reopen the holoscan-radar devcontainer."
