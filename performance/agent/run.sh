#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# One-shot: profile an operator and render its perf report — runs the four
# analysis stages (runner -> parse -> diagnose -> report) in sequence.
#
# Run from the repo root, inside ./holohub run-container (needs nsys and the
# built perf_<operator>; build it first with -DHOLOSCAN_RADAR_BUILD_PERFORMANCE=ON
# -DHOLOSCAN_RADAR_PERF_NVTX=ON). runner.py asserts the toolchain in preflight.
#
# Usage:
#   performance/agent/run.sh <operator> [sizes]
# e.g.
#   performance/agent/run.sh fmcw_range_processor xs,md,xl
#   BUILD_DIR=build-perf performance/agent/run.sh cfar_detector xs,lg,xlw
#
# Env overrides: BUILD_DIR (default build-perf), OUT_DIR (default
# performance/agent/runs). Extra args after <sizes> pass through to runner.py
# (e.g. --no-nsys, --profiled-frames N).
set -euo pipefail

OP="${1:?usage: performance/agent/run.sh <operator> [sizes] [runner-args...]}"
SIZES="${2:-xs,md,xl}"
shift $(( $# > 2 ? 2 : $# ))  # remaining args pass through to runner.py

BUILD_DIR="${BUILD_DIR:-build-perf}"
OUT_DIR="${OUT_DIR:-performance/agent/runs}"
HERE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
RUN="${OUT_DIR}/${OP}"

python3 "${HERE}/runner.py" --operator "${OP}" --build-dir "${BUILD_DIR}" \
    --out-dir "${OUT_DIR}" --sizes "${SIZES}" "$@"
python3 "${HERE}/parse.py"    --run-dir "${RUN}"
python3 "${HERE}/diagnose.py" --run-dir "${RUN}"
python3 "${HERE}/report.py"   --run-dir "${RUN}"

echo "report: ${RUN}/report.md"
