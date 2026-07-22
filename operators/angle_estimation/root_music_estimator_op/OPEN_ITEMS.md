# RootMUSICEstimatorOp — open items

## 1. Batched polynomial root solver (performance)

**Context:** v1 forms the Root-MUSIC conjugate-reciprocal polynomial on device,
builds a Frobenius companion matrix, and calls `cusolverDnXgeev` once per item.
NVIDIA documents `Xgeev` as a hybrid CPU–GPU algorithm with required host
workspace; it is not batched across items.

**Measured impact (RTX PRO 6000 Blackwell, float, 2026-07-22):**

| channels | items | companion | median / frame | approximate / item |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 64 | 6 × 6 | 25 ms | 0.40 ms |
| 8 | 128 | 14 × 14 | 111 ms | 0.87 ms |
| 16 | 256 | 30 × 30 | 437 ms | 1.71 ms |
| 32 | 512 | 62 × 62 | 2456 ms | 4.80 ms |
| 32 | 1024 | 62 × 62 | 4851 ms | 4.74 ms |

An `nsys` trace of the 4-channel case recorded about 39,900 CUDA kernel
launches and 13,000 stream synchronizations across the profiled frames.
Covariance, batched Hermitian eig, projector, polynomial construction, and root
selection together accounted for less than 1% of GPU kernel time; the serial
`Xgeev` calls dominate. There is no `cusolverDnXgeevBatched`; the available
batched eigensolvers are symmetric/Hermitian and cannot preserve the complex
companion eigenvalues.

**Candidate resolution:** replace only the root stage with a device-resident
one-block-per-polynomial solver. A balanced implicit Hessenberg QR solver is
the production-oriented option. Aberth–Ehrlich is simpler, but the initial
prototype did not converge reliably on Root-MUSIC's clustered,
near-unit-circle reciprocal roots. Accuracy must be checked against `Xgeev`
over close sources, high SNR, degree 6–62, and float/double before switching.
Multi-stream `Xgeev` is not favored: the routine is hybrid and synchronizes
internally, so useful overlap would require multiple host threads, handles,
streams, and workspaces while retaining the launch-heavy design.

**Operational mitigations:** reduce `num_items` through cluster-then-angle /
`DetectionReductionOp`, use a smaller uniform subarray when aperture loss is
acceptable, or select MUSIC when gridless output is not required.

**Status:** postponed by team decision on 2026-07-22. The correct `Xgeev` path
remains the v1 implementation; revisit when Root-MUSIC is selected for a
latency-sensitive application. This is a performance follow-up, not a missing
contract feature, so it does not block checklist completeness.

## 2. `source_count_pfa` contract listing

**Context:** `OPERATOR_API.md` references EFT/auto inside `num_sources` text
for Root-MUSIC/ESPRIT but does not list `source_count_pfa` as its own
parameter row (MUSIC lists it explicitly).

**Impact:** Implementation exposes `source_count_pfa` **[live]** like MUSIC;
no behavioral gap.

**Status:** open — doc harmonization only.

## 3. 2-D Root-MUSIC variant

**Context:** Contract: native 2-D az/el needs a 2-D-capable variant; v1 accepts
filled **1-D ULA** only (`array_geometry` with `geometry_type::kUla` or detected
uniform line). URA / 2-D aperture → `std::invalid_argument`; use
`SubarraySelectionOp` upstream.

**Status:** open — deferred variant, not a v1 blocker.

## 4. MatX eig cache retirement

**Context:** Reuses MUSIC's `retire_music_matx_caches()` after batched
`matx::eig` (same MatX plan-lifetime caveat as `MUSICEstimatorOp`).

**Status:** landed — same pattern as MUSIC; Python `install_matx_cache_atexit()`.
