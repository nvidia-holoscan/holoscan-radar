#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Materialize the "before" (pre-optimization) tree and build its perf binaries.
#
# Why a worktree + overlay rather than a plain old checkout: the optimization
# branch also changed the perf harness itself (size ladders, the async-mempool
# retain, live/sustained modes). Measuring an old checkout against the current
# one would compare operators *and* harness at once. So we check out the last
# pre-optimization commit for everything, then overlay the *current* HEAD's
# performance/ and skills/ on top. The harness sources are then byte-identical
# on both sides and only the operator implementations differ.
#
# Layout (build-before/ is already gitignored via the build-* rule):
#   build-before/src    git worktree at the baseline commit + current harness
#   build-before/build  its CMake binary dir
#
# Usage:
#   performance/agent/setup_before_build.sh [BASELINE_REF] [HARNESS_REF]
#
# BASELINE_REF defaults to the `deustice/perf-skills` branch tip, which is
# exactly the split point: it carries all the perf-harness and skills work but
# none of the operator-level optimizations (the first of those, "Operators:
# Pool per-tick output buffers in FMCWRangeProcessorOp", is the first commit
# after it). Naming the branch rather than a SHA keeps this correct across
# rebases of either branch. HARNESS_REF defaults to HEAD.

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

BASELINE_REF="${1:-deustice/perf-skills}"
HARNESS_REF="${2:-HEAD}"

SRC="build-before/src"
BUILD="build-before/build"

# Fall back to the remote-tracking ref for a fresh clone that has not checked
# the branch out locally. Once perf-skills merges and is deleted, neither
# resolves — pass the merge commit (or any pre-optimization SHA) explicitly.
if ! BASELINE_SHA="$(git rev-parse --verify -q "$BASELINE_REF^{commit}")"; then
    if ! BASELINE_SHA="$(git rev-parse --verify -q \
        "origin/$BASELINE_REF^{commit}")"; then
        echo "error: cannot resolve baseline ref '$BASELINE_REF' (tried it" \
            "and origin/$BASELINE_REF)." >&2
        echo "       Pass an explicit pre-optimization commit as \$1." >&2
        exit 1
    fi
fi
HARNESS_SHA="$(git rev-parse --verify "$HARNESS_REF^{commit}")"

echo "baseline (operators): $(git log -1 --format='%h %s' "$BASELINE_SHA")"
echo "harness  (overlay):   $(git log -1 --format='%h %s' "$HARNESS_SHA")"

# The overlay only makes sense if the baseline is behind the harness ref; if
# they have diverged, "before" is not a prefix of "after" and any delta mixes
# in unrelated work.
if ! git merge-base --is-ancestor "$BASELINE_SHA" "$HARNESS_SHA"; then
    echo "error: baseline is not an ancestor of $HARNESS_REF — the two have" \
        "diverged, so the comparison would not isolate the optimizations." >&2
    exit 1
fi

if [ -d "$SRC" ]; then
    echo "==> reusing existing worktree at $SRC"
    git -C "$SRC" checkout --detach "$BASELINE_SHA"
else
    echo "==> creating worktree at $SRC"
    git worktree add --detach "$SRC" "$BASELINE_SHA"
fi

# Overlay the current harness. Both dirs are wholly measurement tooling: no
# operator or shared-library source lives under them, so this cannot smuggle an
# optimization into the baseline.
echo "==> overlaying $HARNESS_SHA:performance/ and skills/"
git -C "$SRC" checkout "$HARNESS_SHA" -- performance/ skills/

# Sanity check: the harness must be identical to the "after" side, else the
# comparison is not apples-to-apples.
if ! git -C "$SRC" diff --quiet "$HARNESS_SHA" -- performance/; then
    echo "error: performance/ differs from $HARNESS_REF after overlay" >&2
    exit 1
fi

# Flags mirror the default ./build tree so the two are directly comparable.
echo "==> configuring $BUILD"
cmake -S "$SRC" -B "$BUILD" \
    -DHOLOSCAN_RADAR_BUILD_TESTING=ON \
    -DHOLOSCAN_RADAR_BUILD_PERFORMANCE=ON \
    -DHOLOSCAN_RADAR_PERF_NVTX="${PERF_NVTX:-OFF}" >/dev/null

mapfile -t TARGETS < <(
    find "$SRC/performance/cpp" -maxdepth 1 -name 'perf_*.cpp' -printf '%f\n' |
        sed 's/\.cpp$//' | sort
)
# cmake prints its own [ nn%] per-target progress; add the piece it cannot
# know — how many targets and how long the whole thing took. A cold CUDA build
# of the full set runs on the order of half an hour.
echo "==> building ${#TARGETS[@]} perf binaries with -j$(nproc)"
BUILD_T0=$SECONDS
cmake --build "$BUILD" -j"$(nproc)" --target "${TARGETS[@]}"
BUILD_ELAPSED=$((SECONDS - BUILD_T0))
printf '==> built %d targets in %dm%02ds\n' \
    "${#TARGETS[@]}" $((BUILD_ELAPSED / 60)) $((BUILD_ELAPSED % 60))

echo
echo "done. compare with:"
echo "  python performance/agent/compare_builds.py \\"
echo "      --before-build $BUILD --after-build build --sizes sm,md,xl"
