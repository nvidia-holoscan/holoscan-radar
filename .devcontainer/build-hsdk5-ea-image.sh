#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# [EA HSDK5 ONLY] Build the local HSDK 5 base image used by the devcontainer.
# Delete this entire file before the HSDK 5 public release.

set -euo pipefail

readonly HSDK_REPOSITORY="ssh://git@gitlab-master.nvidia.com:12051/holoscan/holoscan-sdk.git"
readonly DEFAULT_PIN="deustice/5x/optional-port-contracts"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

usage() {
    cat <<'USAGE'
Usage: build-hsdk5-ea-image.sh [--pin <tag|branch|sha>] [--print-pin]

  --pin <ref>   HSDK 5 tag, branch, or commit sha to build.
                Default: $HSDK_PIN, else deustice/5x/optional-port-contracts.
  --print-pin   Resolve and verify the pin, print the commit and the image it
                would produce, then exit without building.

A tag or sha pin is immutable, so a cached checkout is verified with no network
access. A branch pin is resolved with `git ls-remote` and must sit exactly on
the branch tip; its short sha goes into the image tag so two tips cannot claim
one image. Record the printed image name in devcontainer.json's BASE_IMAGE.
USAGE
}

PIN="${HSDK_PIN:-${DEFAULT_PIN}}"
PRINT_ONLY=false
while (($#)); do
    case "$1" in
        --pin)
            if [[ $# -lt 2 ]]; then
                echo "error: --pin needs a value" >&2
                exit 2
            fi
            PIN="$2"
            shift 2
            ;;
        --pin=*)
            PIN="${1#*=}"
            shift
            ;;
        --print-pin)
            PRINT_ONLY=true
            shift
            ;;
        -h | --help)
            usage
            exit 0
            ;;
        *)
            echo "error: unknown argument '$1'" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ -z "${PIN}" ]]; then
    echo "error: the pin is empty" >&2
    exit 2
fi

required_commands=(git)
if [[ "${PRINT_ONLY}" == false ]]; then
    required_commands+=(docker)
fi
for command in "${required_commands[@]}"; do
    if ! command -v "${command}" >/dev/null 2>&1; then
        echo "error: '${command}' is required" >&2
        exit 1
    fi
done

# A sha pin is a bare hex prefix; anything else is a tag or a branch name, and
# which one it is only matters for verification (below).
pin_is_sha() {
    [[ "${PIN}" =~ ^[0-9a-f]{7,40}$ ]]
}

# Docker tags and directory names take no slashes.
sanitize() {
    local value="${1//[^A-Za-z0-9_.-]/-}"
    printf '%s' "${value}"
}

remote_tip_sha() {
    local listed
    listed="$(
        git ls-remote "${HSDK_REPOSITORY}" \
            "refs/heads/${PIN}" "refs/tags/${PIN}" 2>/dev/null || true
    )"
    printf '%s' "${listed%%$'\t'*}"
}

HSDK_ROOT="${REPO_ROOT}/.cache/holoscan-sdk-$(sanitize "${PIN}")"

if [[ ! -d "${HSDK_ROOT}/.git" ]]; then
    if [[ -e "${HSDK_ROOT}" ]]; then
        echo "error: '${HSDK_ROOT}' exists but is not a Git checkout" >&2
        exit 1
    fi
    if [[ "${PRINT_ONLY}" == true ]]; then
        echo "error: no checkout at '${HSDK_ROOT}' to verify" >&2
        echo "       run without --print-pin to clone it" >&2
        exit 1
    fi

    if pin_is_sha; then
        # --branch takes no sha, and a shallow default-branch clone need not
        # contain the commit, so fetch the commit itself and fall back to a
        # full fetch when the server refuses an arbitrary sha.
        git init --quiet "${HSDK_ROOT}"
        git -C "${HSDK_ROOT}" remote add origin "${HSDK_REPOSITORY}"
        git -C "${HSDK_ROOT}" fetch --depth 1 origin "${PIN}" ||
            git -C "${HSDK_ROOT}" fetch origin
        git -C "${HSDK_ROOT}" checkout --quiet --detach "${PIN}"
    else
        git clone --branch "${PIN}" --depth 1 \
            "${HSDK_REPOSITORY}" "${HSDK_ROOT}"
    fi
fi

CHECKED_OUT_SHA="$(git -C "${HSDK_ROOT}" rev-parse HEAD)"
SHORT_SHA="$(git -C "${HSDK_ROOT}" rev-parse --short HEAD)"

# Verify the checkout is the pin, and name the image after what is immutable
# about it: the tag when the pin is a tag, the commit otherwise.
if pin_is_sha; then
    if [[ "${CHECKED_OUT_SHA}" != "${PIN}"* ]]; then
        echo "error: cached HSDK checkout is ${SHORT_SHA}, not ${PIN}" >&2
        echo "       remove '${HSDK_ROOT}' and run this script again" >&2
        exit 1
    fi
    PIN_KIND="sha"
    IMAGE_TAG="${SHORT_SHA}"
elif [[ -n "$(git -C "${HSDK_ROOT}" tag --list "${PIN}")" ]]; then
    CHECKED_OUT_TAG="$(
        git -C "${HSDK_ROOT}" describe --tags --exact-match 2>/dev/null || true
    )"
    if [[ "${CHECKED_OUT_TAG}" != "${PIN}" ]]; then
        echo "error: cached HSDK checkout is not at ${PIN}" >&2
        echo "       remove '${HSDK_ROOT}' and run this script again" >&2
        exit 1
    fi
    PIN_KIND="tag"
    IMAGE_TAG="$(sanitize "${PIN}")"
else
    TIP_SHA="$(remote_tip_sha)"
    if [[ -z "${TIP_SHA}" ]]; then
        echo "error: could not resolve branch '${PIN}'" >&2
        echo "       no such branch in ${HSDK_REPOSITORY}, or no SSH" \
            "access to it" >&2
        exit 1
    fi
    if [[ "${TIP_SHA}" != "${CHECKED_OUT_SHA}" ]]; then
        echo "error: branch '${PIN}' has moved to ${TIP_SHA:0:9};" \
            "the checkout is ${SHORT_SHA}" >&2
        echo "       remove '${HSDK_ROOT}' to take the new tip, or pin the" \
            "sha you want" >&2
        exit 1
    fi
    PIN_KIND="branch"
    IMAGE_TAG="$(sanitize "${PIN}")-${SHORT_SHA}"
fi

OUTPUT_IMAGE="holoscan-radar-hsdk5-ea:${IMAGE_TAG}"

if [[ -n "$(git -C "${HSDK_ROOT}" status --porcelain)" ]]; then
    echo "error: cached HSDK checkout has local changes" >&2
    echo "       remove '${HSDK_ROOT}' and run this script again" >&2
    exit 1
fi

echo "[EA HSDK5 ONLY] pin ${PIN} (${PIN_KIND}) → ${CHECKED_OUT_SHA}"
echo "[EA HSDK5 ONLY] image ${OUTPUT_IMAGE}"

if [[ "${PRINT_ONLY}" == true ]]; then
    exit 0
fi

echo "[EA HSDK5 ONLY] Building ${PIN}"
(
    cd "${HSDK_ROOT}"
    ./public/run build
)

BUILD_DIR="$(
    cd "${HSDK_ROOT}"
    ./public/run get_build_dir
)"
INSTALL_DIR="$(
    cd "${HSDK_ROOT}"
    ./public/run get_install_dir
)"

# [EA HSDK5 ONLY] Visualization is built by the current SDK checkout but is
# not yet included in public/run's installed component set. Install the
# generated component without modifying the SDK source checkout. public/run
# launch mounts public/ at /workspace/holoscan-sdk in the build container.
if grep -Rqs --include=cmake_install.cmake \
    'CMAKE_INSTALL_COMPONENT.*"holoscan-visualization"' \
    "${HSDK_ROOT}/public/${BUILD_DIR}"; then
    echo "[EA HSDK5 ONLY] Installing HSDK visualization component"
    (
        cd "${HSDK_ROOT}"
        ./public/run launch "${BUILD_DIR}" --run-cmd \
            "cmake --install . \
                --prefix /workspace/holoscan-sdk/${INSTALL_DIR} \
                --component holoscan-visualization"
    )
else
    echo "[EA HSDK5 ONLY] Visualization component was not generated; skipping"
fi

BASE_IMAGE="$(
    cd "${HSDK_ROOT}"
    printf '%s:%s' \
        "$(./public/run get_build_img_name)" \
        "$(./public/run get_git_sha)"
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
if ! grep -q "\"BASE_IMAGE\": \"${OUTPUT_IMAGE}\"" \
    "${SCRIPT_DIR}/devcontainer.json"; then
    echo "note: devcontainer.json does not name this image yet — set its" \
        "BASE_IMAGE build arg to ${OUTPUT_IMAGE}"
fi
echo "Rebuild and reopen the holoscan-radar devcontainer."
