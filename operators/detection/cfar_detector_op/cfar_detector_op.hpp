// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string_view>
#include <type_traits>
#include <vector>

#include <holoscan/core/domain/dlpack.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/parameter.hpp>
#include <holoscan/core/parameter_schema.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/core/tensor_output_loan.hpp>

#include "radar_common/descriptors/max_tensor_extents.hpp"
#include "radar_common/sdk/live_override_payloads.hpp"

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
    case CfarVariant::kCA:
      return "CA";
    case CfarVariant::kGO:
      return "GO";
    case CfarVariant::kSO:
      return "SO";
    case CfarVariant::kOS:
      return "OS";
  }
  return "unknown";
}

// Neighborhood for the optional peak-grouping (non-maximum-suppression) stage
// (peak_only). Static per instance. cross4 = the 2 range + 2 Doppler orthogonal
// neighbors; square8 = the full 8-cell neighborhood.
enum class CfarPeakNeighborhood {
  kCross4,   // 4-connected (2 range + 2 Doppler)
  kSquare8,  // 8-connected (full 3x3 minus the center)
};

constexpr std::string_view to_string(CfarPeakNeighborhood neighborhood) {
  switch (neighborhood) {
    case CfarPeakNeighborhood::kCross4:
      return "cross4";
    case CfarPeakNeighborhood::kSquare8:
      return "square8";
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
// Out: detection_mask  — uint8 (0/1), (..., num_doppler, num_range_bins), with
//                       the nci_axes reduced. 1 where detected
//                       (detection_score > 1). uint8 rather than bool so the
//                       mask is NumPy-visible. When peak_only is set, the
//                       mask instead marks only the retained peak cells.
// Out: detection_score — float32/float64, same shape as detection_mask.
// Out: noise_estimate  — float32/float64, same shape as detection_mask.
// Out: cfar_power      — float32/float64, same shape as detection_mask. The
//                       exact surface the threshold is applied to. Always
//                       emitted (was optional kNone on HSDK 4.x).
// Out: detection_idx   — int64, (num_detections, 2 [+ n_batch]),
//                       (doppler_bin, range_bin) per row.
//
// Live override ports use Exact + SafeDefault.
// Pair-valued windows use CfarCellWindowOverride POD payloads. Variable-length
// nci_axes / alpha_lut use rank-1 pageable-host config Tensors.
//
// Precision is fixed at compile time (T = float | double).

struct CFARParams {
  CfarVariant cfar_variant{CfarVariant::kCA};
  int64_t os_rank{0};  // OS only
  std::vector<int64_t> guard_cells{};
  std::vector<int64_t> training_cells{};
  double pfa{1e-6};
  std::vector<double> alpha_lut{};
  double threshold_scale{1.0};
  std::vector<int64_t> nci_axes{};
  // Optional NCI fold hint (K). Static. YAML name nci_fold.
  int64_t nci_fold{0};
  int64_t max_detections{0};  // <= 0 → uncapped
  bool blocking_alpha_lut_rebuild{false};
  bool peak_only{false};
  CfarPeakNeighborhood peak_neighborhood{CfarPeakNeighborhood::kSquare8};
  bool sticky_pfa{true};
  bool sticky_threshold_scale{true};
  bool sticky_os_rank{true};
  bool sticky_max_detections{true};
  bool sticky_guard_cells{true};
  bool sticky_training_cells{true};
  bool sticky_nci_axes{true};
  bool sticky_alpha_lut{true};
};

HOLOSCAN_PARAM_SCHEMA(
    CFARParams,
    holoscan::param_schema("com.nvidia.holoscan.radar.CFARParams", 1),
    holoscan::field<&CFARParams::cfar_variant>("cfar_variant")
        .description(
            "Noise-estimation rule {CA, GO, SO, OS}. Static: selects the "
            "kernel; instantiate separate detectors to vary it"),
    holoscan::field<&CFARParams::os_rank>("os_rank")
        .description(
            "Rank into the sorted training cells; used only for OS"),
    holoscan::field<&CFARParams::guard_cells>("guard_cells")
        .description(
            "Guard extent each side of the CUT (doppler, range)"),
    holoscan::field<&CFARParams::training_cells>("training_cells")
        .description(
            "Training/reference extent each side (doppler, range)"),
    holoscan::field<&CFARParams::pfa>("pfa").description(
        "Probability of false alarm; sets alpha when alpha_lut is unset"),
    holoscan::field<&CFARParams::alpha_lut>("alpha_lut")
        .description(
            "Optional threshold multiplier table; when non-empty, used "
            "directly and pfa is ignored for alpha calibration"),
    holoscan::field<&CFARParams::threshold_scale>("threshold_scale")
        .description(
            "Empirical multiplier on top of the Pfa-derived threshold; "
            "1.0 = pure CFAR"),
    holoscan::field<&CFARParams::nci_axes>("nci_axes")
        .description(
            "Leading-axis indices to non-coherently integrate before "
            "thresholding"),
    holoscan::field<&CFARParams::nci_fold>("nci_fold")
        .description(
            "Optional integration fold K for start-time alpha LUT "
            "pre-warm when nci_axes is set; <= 0 → unset; static"),
    holoscan::field<&CFARParams::max_detections>("max_detections")
        .description("Cap on emitted detections (<= 0 → uncapped)"),
    holoscan::field<&CFARParams::blocking_alpha_lut_rebuild>(
        "blocking_alpha_lut_rebuild")
        .description(
            "When false, serve the previous LUT while a new one calibrates "
            "after a config change; when true, block until ready; static"),
    holoscan::field<&CFARParams::peak_only>("peak_only")
        .description(
            "When true, collapse each contiguous above-threshold blob to its "
            "local-maximum cell; static"),
    holoscan::field<&CFARParams::peak_neighborhood>("peak_neighborhood")
        .description(
            "Neighborhood for peak_only NMS {cross4, square8}; static"),
    holoscan::field<&CFARParams::sticky_pfa>("sticky_pfa")
        .description("Keep the last Exact pfa match on a miss."),
    holoscan::field<&CFARParams::sticky_threshold_scale>(
        "sticky_threshold_scale")
        .description("Keep the last Exact threshold_scale match on a miss."),
    holoscan::field<&CFARParams::sticky_os_rank>("sticky_os_rank")
        .description("Keep the last Exact os_rank match on a miss."),
    holoscan::field<&CFARParams::sticky_max_detections>(
        "sticky_max_detections")
        .description("Keep the last Exact max_detections match on a miss."),
    holoscan::field<&CFARParams::sticky_guard_cells>("sticky_guard_cells")
        .description("Keep the last Exact guard_cells match on a miss."),
    holoscan::field<&CFARParams::sticky_training_cells>(
        "sticky_training_cells")
        .description("Keep the last Exact training_cells match on a miss."),
    holoscan::field<&CFARParams::sticky_nci_axes>("sticky_nci_axes")
        .description("Keep the last Exact nci_axes match on a miss."),
    holoscan::field<&CFARParams::sticky_alpha_lut>("sticky_alpha_lut")
        .description("Keep the last Exact alpha_lut match on a miss."));

static_assert(holoscan::param_schema_availability<CFARParams>() ==
                  holoscan::ParamSchemaAvailability::kDeclared,
              "HOLOSCAN_PARAM_SCHEMA must be in CFARParams's namespace");

// 2-D CFAR detector on the (Doppler, range) plane. Fat map outputs
// (detection_mask/score/noise/cfar_power) are sized from
// MaxTensorExtents at graph.op() (ctor capture); detection_idx uses
// max_detections. Dense path may include named leading az/el axes.
//
// Precision is fixed at compile time (T = float | double).

template <typename T>
class CFARDetectorOp : public holoscan::Operator<CFARParams> {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                "CFARDetectorOp supports float and double precision");

 public:
  // Required: sizes the output Tensor pools at setup().
  explicit CFARDetectorOp(const MaxTensorExtents& extents);
  ~CFARDetectorOp() override;

  // Pre-calibrates the alpha LUT at startup (before the first tick) so a slow
  // Monte Carlo build does not stall the data plane on tick one.
  void start() override;
  // Drains any in-flight background alpha-LUT build before Holoscan tears down
  // the CUDA context.
  void stop() override;

  void setup(OperatorSpec& spec) override;
  [[nodiscard]] Contract contract() const override;
  [[nodiscard]] expected<void, Error> compute(
      ExecutionContext& context) override;

  Input<Tensor> rd_maps{"rd_maps"};
  Output<Tensor> detection_mask{"detection_mask"};
  Output<Tensor> detection_score{"detection_score"};
  Output<Tensor> noise_estimate{"noise_estimate"};
  Output<Tensor> detection_idx{"detection_idx"};
  // Always emitted on HSDK 5 (was optional kNone on 4.x). Tests and apps must
  // bind / drain this port.
  Output<Tensor> cfar_power{"cfar_power"};
  // [live] Exact overrides (OPERATOR_API.md).
  Input<double> pfa{"pfa"};
  Input<double> threshold_scale{"threshold_scale"};
  Input<int64_t> os_rank{"os_rank"};
  Input<int64_t> max_detections{"max_detections"};
  Input<CfarCellWindowOverride> guard_cells{"guard_cells"};
  Input<CfarCellWindowOverride> training_cells{"training_cells"};
  Input<Tensor> nci_axes{"nci_axes"};
  Input<Tensor> alpha_lut{"alpha_lut"};

 private:
  MaxTensorExtents extents_;
  std::uint32_t max_map_rank_{2U};
  std::size_t max_map_byte_span_{0U};
  std::size_t max_idx_byte_span_{0U};
  std::size_t max_leading_{0U};
  std::int64_t max_alpha_lut_extent_{0};

  using AlphaLutKey = CfarAlphaLutKey;

  // Sticky [live] POD scalars: the last Exact match holds while sticky_*.
  std::optional<double> sticky_pfa_;
  std::optional<double> sticky_threshold_scale_;
  std::optional<int64_t> sticky_os_rank_;
  std::optional<int64_t> sticky_max_detections_;
  std::optional<std::vector<int64_t>> sticky_guard_cells_;
  std::optional<std::vector<int64_t>> sticky_training_cells_;
  std::optional<std::vector<int64_t>> sticky_nci_axes_;
  std::optional<std::vector<double>> sticky_alpha_lut_;
  // Tracks pfa sticky transitions for the "pfa ignored while alpha_lut"
  // warn-once.
  double last_seen_pfa_{-1.0};
  bool pfa_seen_{false};

  // Cached device copy of the NCI batch_in lookup table.
  std::shared_ptr<holoscan::Tensor> nci_table_dev_{};
  std::vector<int64_t> nci_table_host_cache_{};

  AlphaLutKey alpha_lut_key_{};
  AlphaLutKey pending_lut_key_{};
  bool pending_lut_active_{false};
  bool stale_lut_warned_{false};
  bool pfa_lut_slow_warned_{false};
  bool runtime_build_warned_{false};
  bool window_exceeds_map_warned_{false};
  bool using_external_alpha_lut_{false};
  bool pfa_ignored_while_alpha_lut_warned_{false};
  bool alpha_lut_cleared_pfa_active_warned_{false};
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

  std::shared_ptr<detail::DetectionIdxCompactState> detection_idx_compact_;

  // Ping-ponged dense scratch workspaces. Outputs are loaned from the pool
  // each tick (HSDK 5); these buffers hold stage results before D2D emit.
  struct DenseWorkspace {
    std::shared_ptr<holoscan::Tensor> score;
    std::shared_ptr<holoscan::Tensor> noise;
    std::shared_ptr<holoscan::Tensor> mask;
    std::shared_ptr<holoscan::Tensor> peak_mask;
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
  Tensor retained_input_;
};

extern template class CFARDetectorOp<float>;
extern template class CFARDetectorOp<double>;

}  // namespace holoscan::radar
