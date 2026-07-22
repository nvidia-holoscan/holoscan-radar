// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Source-count (model-order) estimation for the subspace angle estimators —
// MUSIC / Root-MUSIC / ESPRIT — via the Exponential Fitting Test (EFT).
// Settled decision: docs/planning/STATUS.md S2.
//
// EFT (Quinlan, Barbot, Larzabal & Haardt, "Model Order Selection for Short
// Data: An Exponential Fitting Test (EFT)", EURASIP JASP 2007,
// doi:10.1155/2007/71953) exploits the fact that, for white Gaussian noise and
// a short sample (few snapshots N relative to channel count M), the *ordered*
// noise-only eigenvalue profile is well approximated by an exponential law.
// Starting from the smallest observed eigenvalue (assumed noise), the test
// recursively predicts the next-largest theoretical noise eigenvalue from
// that law and compares it to what was actually observed; a significant
// mismatch marks the first signal eigenvalue, so the noise-subspace
// dimension — and hence the source count — is read off directly.
//
// Fidelity note (do not silently "fix" this without a team decision): the
// original paper calibrates its per-step acceptance thresholds η_P by
// Monte Carlo (~10^5 trials per (P, N, target Pfa) triple — the same
// complexity class as this codebase's GO/SO/OS CFAR alpha_lut calibration,
// see cfar_alpha_mc.cu). This implementation instead uses the paper's own
// closed-form noise-eigenvalue-profile model (eq. 25-31) for the *prediction*
// step, but derives the accept/reject threshold from a first-order Gaussian
// approximation of the profile's fluctuation (eq. 22-24) rather than Monte
// Carlo calibration. This trades exact Pfa control for a cheap, allocation-
// free, closed-form test that is fast enough to run per item, per tick, with
// no calibration cache — the resulting `pfa` parameter is a *target*, not a
// guarantee. Tightening this to the exact MC-calibrated test later (if the
// team wants precise Pfa control) is a documented follow-up, not a contract
// change: the operator-facing `num_sources: auto` / `source_count_pfa`
// contract (OPERATOR_API.md) does not pin the calibration method.
//
// CUDA-free (lives in radar_common); the identical per-item decision, batched
// over many items with the eigenvalues already resident on the GPU (as
// produced by matx::eig), is also available as a device kernel in
// source_count_device.{hpp,cu} for the operator hot path — this header is the
// host reference / spec the device kernel is validated against.

#pragma once

#include <cstdint>
#include <vector>

namespace holoscan::radar {

// Estimate the number of sources (model order) from the ascending eigenvalue
// spectrum of a sample covariance matrix, via the closed-form-profile /
// Gaussian-threshold EFT variant described above.
//
// `eigenvalues_ascending` — the M = eigenvalues.size() eigenvalues of the
//   sample covariance, smallest first (matches matx::eig's convention and
//   MVDREstimatorOp's `eigenvalues` output).
// `num_snapshots` — N, the number of (valid) snapshots the covariance was
//   estimated from. Must be >= 1.
// `pfa` — target per-trial false-alarm probability (the EFT test's Pfa
//   budget is split uniformly over the up to M-1 recursive steps, mirroring
//   the paper's own per-step decomposition, eq. 37). Must be in (0, 1).
// `max_sources` — clamp on the returned value. Must be >= 1.
//
// Returns d in [1, min(max_sources, M-1)]: with M eigenvalues at most M-1 can
// be signal (at least one must remain to define the noise floor), and the
// contract clamps the floor at 1 even in the trivial noise-only case (a
// radar operator reports "at least one" DOA, never "zero sources", from the
// dominant-eigenvector direction).
//
// Does not implement the "rank-deficient covariance (K < num_channels) falls
// back to 1 and warns" contract clause — callers know K and must apply that
// fallback (and the warning) themselves before calling this on a full-rank
// covariance's eigenvalues.
int64_t eft_source_count(const std::vector<double>& eigenvalues_ascending,
                         int64_t num_snapshots, double pfa,
                         int64_t max_sources);

}  // namespace holoscan::radar
