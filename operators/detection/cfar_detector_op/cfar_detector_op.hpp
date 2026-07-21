// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <string_view>
#include <type_traits>
#include <vector>

#include <holoscan/holoscan.hpp>

namespace holoscan::radar {

namespace detail {
struct DetectionIdxCompactState;
}  // namespace detail

// CFAR noise-estimation rule. Static per instance (selects the kernel); to
// run different variants, instantiate separate detectors.
enum class CfarVariant {
  kCA,  // cell-averaging
  kGO,  // greatest-of
  kSO,  // smallest-of
  kOS,  // order-statistic
};

constexpr std::string_view to_string(CfarVariant variant) {
  switch (variant) {
    case CfarVariant::kCA: return "CA";
    case CfarVariant::kGO: return "GO";
    case CfarVariant::kSO: return "SO";
    case CfarVariant::kOS: return "OS";
  }
  return "unknown";
}

// Cache key for the host-built per-training-count alpha LUT. Shared between
// operator instances via detail::get_cached_alpha_lut.
struct CfarAlphaLutKey {
  CfarVariant variant{CfarVariant::kCA};
  double pfa{-1.0};
  int64_t fold{0};
  int64_t os_rank{0};
  int guard_d{0};
  int guard_r{0};
  int train_d{0};
  int train_r{0};
  int num_doppler{0};
  int num_range{0};

  bool operator==(const CfarAlphaLutKey& other) const {
    return variant == other.variant && pfa == other.pfa &&
           fold == other.fold && os_rank == other.os_rank &&
           guard_d == other.guard_d && guard_r == other.guard_r &&
           train_d == other.train_d && train_r == other.train_r &&
           num_doppler == other.num_doppler && num_range == other.num_range;
  }
  bool operator!=(const CfarAlphaLutKey& other) const {
    return !(*this == other);
  }
};

// 2-D CFAR detector on the (Doppler, range) plane. This header is the
// contract reference for the operator's ports, shapes, dtypes, and
// parameters; module-wide conventions live in operators/common/README.md.
//
// In:  rd_maps        — complex64/complex128 (square-law power formed
//                       internally) or real float32/float64 (already-formed
//                       power map), (..., num_doppler, num_range_bins). The
//                       CFAR plane is the last two axes; leading axes are
//                       batched, or non-coherently integrated per nci_axes.
//                       Path auto-selected by dtype (no parameter).
// Out: detection_mask  — bool, (..., num_doppler, num_range_bins), with the
//                       nci_axes reduced. True where detected
//                       (detection_score > 1).
// Out: detection_score — float32/float64, same shape as detection_mask.
//                       P / (threshold_scale * alpha * noise_estimate); >1
//                       ⇔ detected, doubles as the per-detection strength.
// Out: noise_estimate  — float32/float64, same shape as detection_mask.
//                       Per-cell training-cell noise (N cells).
// Out: detection_idx   — int64, (num_detections, 2 [+ n_batch]),
//                       (doppler_bin, range_bin) per row. Always emitted,
//                       sized to the detection count and reallocated per
//                       tick. Capped by max_detections when set.
//
// Precision is fixed at compile time by the template instantiation
// (T = float → complex64/float32, T = double → complex128/float64).
//
// Implementation status: all four variants (CA, GO, SO, OS) are implemented for
// arbitrary leading axes with optional non-coherent integration (nci_axes),
// Gamma(K) thresholding, and batch-widened detection_idx.
template <typename T>
class CFARDetectorOp : public holoscan::Operator {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                "CFARDetectorOp supports float and double precision");

 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(CFARDetectorOp)

  CFARDetectorOp() = default;
  ~CFARDetectorOp() override;

  // Registers the argument setters for the custom parameter types
  // (radar_common/sdk/arg_setters.hpp, plus the local CfarVariant enum)
  // before the base class processes args.
  void initialize() override;
  // Pre-calibrates the alpha LUT at startup (before the first tick) so a slow
  // Monte Carlo build does not stall the data plane on tick one. Assumes the RD
  // map is at least as large as the CFAR window; a smaller map triggers a
  // map-specific rebuild in compute() with a warning.
  void start() override;

  void setup(OperatorSpec& spec) override;
  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override;

 private:
  // Parameters per the locked contract. cfar_variant is static; the rest are
  // overridable (each has a same-named non-blocking override input port
  // declared in setup()).
  Parameter<CfarVariant> cfar_variant_;
  Parameter<int64_t> os_rank_;                  // OS only
  Parameter<std::vector<int64_t>> guard_cells_;    // (doppler, range)
  Parameter<std::vector<int64_t>> training_cells_;  // (doppler, range)
  Parameter<double> pfa_;
  Parameter<std::vector<double>> alpha_lut_;
  Parameter<double> threshold_scale_;
  Parameter<std::vector<int64_t>> nci_axes_;
  // Optional NCI fold hint (K = product of nci_axes sizes). Static. Only used
  // to pre-warm the alpha LUT at start() when nci_axes is non-empty (the fold
  // is otherwise unknown until the first tick). <= 0 → unset (defer to first
  // tick); ignored when nci_axes is empty (K = 1).
  Parameter<int64_t> nci_fold_hint_;
  Parameter<int64_t> max_detections_;  // <= 0 → uncapped
  // When false (default), a [live] LUT-key change serves the previous alpha LUT
  // on that tick while the new table calibrates in the background (one-shot
  // warn). When true, compute() blocks until the new LUT is ready.
  Parameter<bool> blocking_alpha_lut_rebuild_;

  // Effective override values as of the last tick (override-aware), used to
  // detect override-triggered window-kernel rebuilds for the
  // warn-once-and-cache rule.
  std::vector<int64_t> effective_guard_cells_{};
  std::vector<int64_t> effective_training_cells_{};
  double effective_pfa_{1e-6};
  std::vector<double> effective_alpha_lut_{};
  double effective_threshold_scale_{1.0};
  int64_t effective_os_rank_{0};
  std::vector<int64_t> effective_nci_axes_{};
  int64_t effective_max_detections_{0};

  // Set once the effective_* values have been seeded from the parameters
  // (first tick); overrides are applied on top each tick thereafter.
  bool seeded_{false};

  // Cached device copy of the NCI batch_in lookup table. Rebuilt only when the
  // integration layout changes (plan.batch_in_table differs), so a stable
  // nci_axes + input shape incurs no per-tick device alloc/free or upload.
  std::shared_ptr<holoscan::Tensor> nci_table_dev_{};
  std::vector<int64_t> nci_table_host_cache_{};

  // Host-built alpha(N) LUT. Rebuilt when pfa, fold, window footprint, map
  // size, or variant changes. Host table is shared across operator instances
  // via detail::get_cached_alpha_lut.
  using AlphaLutKey = CfarAlphaLutKey;
  AlphaLutKey alpha_lut_key_{};
  AlphaLutKey pending_lut_key_{};
  bool pending_lut_active_{false};
  bool stale_lut_warned_{false};
  bool pfa_lut_slow_warned_{false};
  // Warns once per key transition when a LUT build is forced during compute()
  // (a runtime rebuild / stall) rather than served from the start-time warm
  // cache. Reset whenever the effective LUT key changes.
  bool runtime_build_warned_{false};
  // Warns once when an RD map smaller than the CFAR window arrives, so the
  // start-time (map-fits-window) pre-warm does not apply and a map-specific LUT
  // is built with reduced-N edge thresholds.
  bool window_exceeds_map_warned_{false};
  bool using_external_alpha_lut_{false};
  bool pfa_ignored_while_alpha_lut_warned_{false};
  bool alpha_lut_cleared_pfa_active_warned_{false};
  // Cache that external alpha_lut was validated for the current table +
  // geometry/fold so we skip the O(max_index) checks on unchanged ticks.
  bool external_alpha_lut_validated_{false};
  CfarVariant validated_external_variant_{CfarVariant::kCA};
  int64_t validated_external_fold_{0};
  int64_t validated_external_num_doppler_{0};
  int64_t validated_external_num_range_{0};
  int validated_external_guard_d_{0};
  int validated_external_guard_r_{0};
  int validated_external_train_d_{0};
  int validated_external_train_r_{0};
  std::vector<double> alpha_lut_host_cache_{};
  std::shared_ptr<holoscan::Tensor> alpha_lut_dev_{};

  // Device-side detection_idx compaction scratch (CUB temps + scan offsets).
  // shared_ptr (not unique_ptr) so the incomplete type is OK in TUs that only
  // see the forward declaration — the deleter is type-erased at creation in
  // the .cu.
  std::shared_ptr<detail::DetectionIdxCompactState> detection_idx_compact_;

  // Ping-ponged dense workspaces. score/noise/mask are emitted, so two sets
  // avoid overwriting a buffer a downstream consumer may still hold. power /
  // integrated_power are scratch-only (same slot is fine within a tick).
  // Reallocated only when (batch_in, out_batch, D, R) changes.
  struct DenseWorkspace {
    std::shared_ptr<holoscan::Tensor> score;
    std::shared_ptr<holoscan::Tensor> noise;
    std::shared_ptr<holoscan::Tensor> mask;
    std::shared_ptr<holoscan::Tensor> power;
    std::shared_ptr<holoscan::Tensor> integrated_power;
    int64_t batch_in{0};
    int64_t out_batch{0};
    int64_t num_doppler{0};
    int64_t num_range{0};
  };
  DenseWorkspace dense_workspace_[2]{};
  int dense_workspace_idx_{0};

  // Keeps the previous tick's input alive until this tick's kernels are
  // enqueued (stream-ordered lifetime guard).
  std::shared_ptr<holoscan::Tensor> retained_input_;
};

// Explicit instantiations live in cfar_detector_op.cu.
extern template class CFARDetectorOp<float>;
extern template class CFARDetectorOp<double>;

}  // namespace holoscan::radar
