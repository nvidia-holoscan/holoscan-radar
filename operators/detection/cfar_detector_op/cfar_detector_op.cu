// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "detection/cfar_detector_op/cfar_detector_op.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/expected_macros.hpp>
#include <holoscan/core/payload_options.hpp>
#include <holoscan/logger/logger.hpp>

#include "detection/cfar_detector_op/cfar_alpha.hpp"
#include "detection/cfar_detector_op/cfar_compact.cuh"
#include "detection/cfar_detector_op/cfar_stage.hpp"
#include "detection/cfar_detector_op/nci_plan.hpp"
#include "radar_common/sdk/override_param.hpp"
#include "radar_common/sdk/tensor_io.hpp"
#include "radar_common/tensor/dlpack_buffer.hpp"
#include "radar_common/tensor/extents_layout.hpp"
#include "radar_common/tensor/max_byte_span.hpp"
#include "radar_common/tensor/tensor_utils.hpp"

namespace holoscan::radar {

namespace {

constexpr const char* kOpName = "CFARDetectorOp";
// detection_mask is emitted as uint8 (0/1), not kDLBool: byte-identical to the
// 0/1 bytes kernels write; NumPy-visible.
constexpr DLDataType kMaskDtype{kDLUInt, 8, 1};
constexpr DLDataType kInt64Dtype{kDLInt, 64, 1};

expected<void, Error> fail_invalid(const std::string& message) {
  return make_unexpected(Error{ErrorCode::kInvalidArgument, message});
}

expected<void, Error> fail_runtime(const std::string& message) {
  return make_unexpected(Error{ErrorCode::kFailure, message});
}

expected<void*, Error> writable_ptr(holoscan::Tensor& tensor,
                                    const char* what) {
  auto mut = tensor.mutable_data();
  if (!mut) {
    return make_unexpected(Error{
        ErrorCode::kFailure,
        std::string(kOpName) + ": " + what + " is not writable: " +
            mut.error().message});
  }
  return *mut;
}

bool uses_slow_pfa_lut_calibration(CfarVariant variant, int64_t fold) {
  return variant == CfarVariant::kGO || variant == CfarVariant::kSO ||
         (variant == CfarVariant::kOS && fold > 1);
}

int64_t max_alpha_lut_index(CfarVariant variant, int num_doppler,
                            int num_range, int guard_d, int guard_r,
                            int train_d, int train_r) {
  if (variant == CfarVariant::kGO || variant == CfarVariant::kSO) {
    return detail::max_go_lut_index(num_doppler, num_range, guard_d, guard_r,
                                    train_d, train_r);
  }
  return detail::max_training_cell_count(num_doppler, num_range, guard_d,
                                         guard_r, train_d, train_r);
}

expected<void, Error> validate_external_alpha_lut(
    CfarVariant variant, int64_t fold, const std::vector<double>& lut,
    int64_t max_lut_index) {
  if (max_lut_index <= 0) {
    return {};
  }
  if (lut.size() <= static_cast<size_t>(max_lut_index)) {
    return fail_invalid(
        "CFARDetectorOp: alpha_lut length must exceed the maximum LUT index "
        "for the configured variant, footprint, and map size");
  }
  for (int64_t idx = 1; idx <= max_lut_index; ++idx) {
    const double alpha = lut[static_cast<size_t>(idx)];
    if (!std::isfinite(alpha) || alpha < 0.0) {
      return fail_invalid(
          "CFARDetectorOp: alpha_lut entries for used indices must be finite "
          "and non-negative");
    }
  }
  const bool sparse_ok = variant == CfarVariant::kGO ||
                         variant == CfarVariant::kSO ||
                         (variant == CfarVariant::kOS && fold > 1);
  if (sparse_ok) {
    return {};
  }
  for (int64_t idx = 1; idx <= max_lut_index; ++idx) {
    if (lut[static_cast<size_t>(idx)] <= 0.0) {
      return fail_invalid(
          "CFARDetectorOp: alpha_lut entries for used indices must be finite "
          "and positive");
    }
  }
  return {};
}

expected<void, Error> require_pair(const std::vector<int64_t>& v,
                                   const char* name) {
  if (v.size() != 2) {
    return fail_invalid(std::string("CFARDetectorOp: '") + name +
                        "' must have exactly 2 entries (doppler, range), got " +
                        std::to_string(v.size()));
  }
  if (v[0] < 0 || v[1] < 0) {
    return fail_invalid(std::string("CFARDetectorOp: '") + name +
                        "' entries must be >= 0");
  }
  return {};
}

expected<TensorOutputLoan, Error> prepare_copy_device(
    Output<Tensor>& port, ExecutionContext& context,
    std::span<const int64_t> shape, DLDataType dtype, const void* src,
    cudaStream_t stream) {
  return prepare_device_tensor(
      port, context, shape, dtype,
      [&](void* dst, std::size_t bytes) -> expected<void, Error> {
        if (bytes == 0) {
          return {};
        }
        if (cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice,
                            stream) != cudaSuccess) {
          return fail_runtime(std::string(kOpName) +
                              ": output D2D copy failed on '" +
                              std::string(port.name()) + "'");
        }
        return {};
      });
}

}  // namespace

template <typename T>
CFARDetectorOp<T>::CFARDetectorOp(const MaxTensorExtents& extents) {
  const char* who = "CFARDetectorOp";
  const std::int64_t max_doppler =
      require_extent(extents.max_doppler_bins, "max_doppler_bins",
                     who);
  const std::int64_t max_range =
      require_extent(extents.max_range_bins, "max_range_bins", who);
  extents_ = extents;

  // Named leading axes: none, N_rx, or (M_tx, N_rx). Same scope as
  // ClutterFilter; the dense beamscan path (az/el leading) uses a
  // separate CFARDetectorOp instance with tighter az/el-only extents
  // (those instances would not set max_tx / max_rx).
  const std::size_t leading = named_tx_rx_leading_rank(extents);
  max_leading_ = leading;
  max_map_rank_ = static_cast<std::uint32_t>(2U + leading);

  std::int64_t max_tx = 1;
  std::int64_t max_rx = 1;
  require_tx_rx_pool_extents(extents, who, &max_tx, &max_rx);
  const std::int64_t leading_product = max_tx * max_rx;
  max_map_byte_span_ = byte_span_real_cube<T>(
      {leading_product, max_doppler, max_range}, who);

  const std::int64_t max_dets =
      extents.max_detections.value_or(
          leading_product * max_doppler * max_range);
  const std::int64_t idx_cols =
      2 + static_cast<std::int64_t>(leading);
  max_idx_byte_span_ = byte_span_elems(
      product_extents({max_dets, idx_cols}, who),
      sizeof(std::int64_t), who);
}

template <typename T>
CFARDetectorOp<T>::~CFARDetectorOp() = default;

template <typename T>
void CFARDetectorOp<T>::start() {
  // Pre-calibrate the alpha LUT at startup so the first tick doesn't stall on a
  // Monte Carlo build. Only the pfa-derived path needs calibration (an external
  // alpha_lut is used as-is). GO/SO cannot be pre-warmed without the RD-map
  // shape.
  const CFARParams& pinned = params();
  if (!pinned.alpha_lut.empty()) {
    return;
  }
  const CfarVariant variant = pinned.cfar_variant;
  if (variant != CfarVariant::kCA && variant != CfarVariant::kOS) {
    HOLOSCAN_LOG_INFO(
        "CFARDetectorOp: {} calibrates its alpha LUT on the first tick "
        "(go_lut_index tables are map-dependent and cannot be pre-warmed "
        "without the RD-map shape); supply alpha_lut to avoid a runtime build.",
        to_string(variant));
    return;
  }
  const std::vector<int64_t>& guard = pinned.guard_cells;
  const std::vector<int64_t>& train = pinned.training_cells;
  if (guard.size() != 2 || train.size() != 2) {
    return;
  }

  int64_t fold = 1;
  if (!pinned.nci_axes.empty()) {
    const int64_t hint = pinned.nci_fold;
    if (hint <= 0) {
      HOLOSCAN_LOG_INFO(
          "CFARDetectorOp: nci_axes is set but nci_fold hint is unset; "
          "deferring alpha-LUT calibration to the first tick (set nci_fold to "
          "the integration fold K to pre-warm at start).");
      return;
    }
    fold = hint;
  }

  const int gd = static_cast<int>(guard[0]);
  const int gr = static_cast<int>(guard[1]);
  const int td = static_cast<int>(train[0]);
  const int tr = static_cast<int>(train[1]);
  const int win_d = 2 * (gd + td) + 1;
  const int win_r = 2 * (gr + tr) + 1;

  const AlphaLutKey warm_key{variant, pinned.pfa, fold, pinned.os_rank,
                             gd,      gr,         td,   tr,
                             win_d,   win_r};
  try {
    (void)detail::get_cached_alpha_lut(warm_key);
    HOLOSCAN_LOG_INFO(
        "CFARDetectorOp: pre-calibrated alpha LUT at start (variant={}, "
        "pfa={}, K={}).",
        to_string(variant), pinned.pfa, fold);
  } catch (const std::exception& e) {
    HOLOSCAN_LOG_WARN(
        "CFARDetectorOp: start-time alpha-LUT pre-calibration skipped ({}); "
        "the first tick will calibrate.",
        e.what());
  }
}

template <typename T>
void CFARDetectorOp<T>::stop() {
  detail::join_inflight_alpha_lut_builds();
}

template <typename T>
void CFARDetectorOp<T>::setup(OperatorSpec& spec) {
  constexpr TensorPortLayout kDevice{
      .memory_kind = MemoryKind::kCudaDevice,
      .dtype = std::nullopt,
      .rank = std::nullopt,
  };
  spec.input(rd_maps).queue_depth(2U).expects_tensor(kDevice);
  const TensorAllocationBounds map_bounds{
      .max_rank = max_map_rank_,
      .max_byte_span = max_map_byte_span_};
  // detection_mask is uint8 — map_bounds (real T) is a safe ceiling.
  spec.output(detection_mask)
      .produces_tensor(kDevice)
      .tensor_allocation(map_bounds);
  spec.output(detection_score)
      .produces_tensor(kDevice)
      .tensor_allocation(map_bounds);
  spec.output(noise_estimate)
      .produces_tensor(kDevice)
      .tensor_allocation(map_bounds);
  spec.output(cfar_power)
      .produces_tensor(kDevice)
      .tensor_allocation(map_bounds);
  spec.output(detection_idx)
      .produces_tensor(kDevice)
      .tensor_allocation(TensorAllocationBounds{
          .max_rank = 2U,
          .max_byte_span = max_idx_byte_span_});
  // [live] POD override ports. Vector lives are Params-only (HSDK5_PORT §9).
  spec.input(pfa).queue_depth(2U);
  spec.input(threshold_scale).queue_depth(2U);
  spec.input(os_rank).queue_depth(2U);
  spec.input(max_detections).queue_depth(2U);
}

template <typename T>
Contract CFARDetectorOp<T>::contract() const {
  Contract result;
  result.trigger(OnEach{rd_maps});
  maybe_add_live_override_to_contract(result, pfa, params().expect_pfa);
  maybe_add_live_override_to_contract(
      result, threshold_scale, params().expect_threshold_scale);
  maybe_add_live_override_to_contract(
      result, os_rank, params().expect_os_rank);
  maybe_add_live_override_to_contract(
      result, max_detections, params().expect_max_detections);
  return result;
}

template <typename T>
expected<void, Error> CFARDetectorOp<T>::compute(ExecutionContext& context) {
  HOLOSCAN_TRY(auto input, receive_tensor(rd_maps, kOpName));

  const CFARParams& pinned = context.params<CFARParams>();
  HOLOSCAN_TRY(auto effective_pfa,
               resolve_expected_override(pinned.expect_pfa, pfa, pinned.pfa,
                                         sticky_pfa_, pinned.sticky_pfa));
  HOLOSCAN_TRY(auto effective_threshold_scale,
               resolve_expected_override(pinned.expect_threshold_scale,
                                         threshold_scale,
                                         pinned.threshold_scale,
                                         sticky_threshold_scale_,
                                         pinned.sticky_threshold_scale));
  HOLOSCAN_TRY(auto effective_os_rank,
               resolve_expected_override(pinned.expect_os_rank, os_rank,
                                         pinned.os_rank, sticky_os_rank_,
                                         pinned.sticky_os_rank));
  HOLOSCAN_TRY(auto effective_max_detections,
               resolve_expected_override(pinned.expect_max_detections,
                                         max_detections, pinned.max_detections,
                                         sticky_max_detections_,
                                         pinned.sticky_max_detections));
  // Params-only vectors (HSDK5_PORT §9).
  const std::vector<int64_t>& effective_guard_cells = pinned.guard_cells;
  const std::vector<int64_t>& effective_training_cells =
      pinned.training_cells;
  const std::vector<double>& effective_alpha_lut = pinned.alpha_lut;
  const std::vector<int64_t>& effective_nci_axes = pinned.nci_axes;

  bool pfa_override_changed = false;
  if (pfa_seen_ && effective_pfa != last_seen_pfa_) {
    pfa_override_changed = true;
  }
  last_seen_pfa_ = effective_pfa;
  pfa_seen_ = true;

  HOLOSCAN_TRY(require_pair(effective_guard_cells, "guard_cells"));
  HOLOSCAN_TRY(require_pair(effective_training_cells, "training_cells"));

  InputPrecision precision;
  try {
    precision =
        classify_input_dtype<T>(input, "rd_maps", /*allow_real=*/true);
    require_cuda_accessible(input, "rd_maps", kOpName);
    require_contiguous(input, "rd_maps", kOpName);
  } catch (const std::invalid_argument& e) {
    return fail_invalid(e.what());
  }
  const bool input_is_complex = (precision == InputPrecision::kComplex);

  detail::NciPlan plan;
  try {
    plan = detail::NciPlan::from_shape(input.shape(), effective_nci_axes,
                                       "rd_maps");
  } catch (const std::invalid_argument& e) {
    return fail_invalid(e.what());
  }
  const std::size_t total_rank = input.shape().size();
  if (total_rank < 2) {
    return fail_invalid(
        "CFARDetectorOp: rd_maps must have at least 2 axes "
        "(num_doppler, num_range)");
  }
  const std::size_t leading_rank = total_rank - 2;
  if (leading_rank > max_leading_) {
    return fail_invalid(
        unexpected_leading_message(kOpName, leading_rank,
                                   max_leading_));
  }
  const int64_t num_doppler = input.shape()[total_rank - 2];
  const int64_t num_range = input.shape()[total_rank - 1];
  if (extents_.max_doppler_bins.has_value() &&
      num_doppler > *extents_.max_doppler_bins) {
    return fail_invalid(
        "CFARDetectorOp: input doppler_bins exceed "
        "MaxTensorExtents");
  }
  if (extents_.max_range_bins.has_value() &&
      num_range > *extents_.max_range_bins) {
    return fail_invalid(
        "CFARDetectorOp: input range_bins exceed "
        "MaxTensorExtents");
  }
  const std::vector<int64_t> output_shape =
      plan.output_shape(num_doppler, num_range);

  cudaStream_t stream = context.cuda_stream().get();
  if (stream == nullptr) {
    return make_unexpected(
        Error{ErrorCode::kNotSupported,
              "CFARDetectorOp: no CUDA compute stream"});
  }

  dense_workspace_idx_ ^= 1;
  DenseWorkspace& ws = dense_workspace_[dense_workspace_idx_];
  const bool ws_hit = ws.score && ws.batch_in == plan.batch_in &&
                      ws.out_batch == plan.out_batch &&
                      ws.num_doppler == num_doppler &&
                      ws.num_range == num_range;
  if (!ws_hit) {
    try {
      ws.score = make_device_tensor(output_shape, real_dtype<T>());
      ws.noise = make_device_tensor(output_shape, real_dtype<T>());
      ws.mask = make_device_tensor(output_shape, kMaskDtype);
      ws.peak_mask = make_device_tensor(output_shape, kMaskDtype);
      ws.power = make_device_tensor(
          plan.fold > 1
              ? std::vector<int64_t>{plan.batch_in, num_doppler, num_range}
              : output_shape,
          real_dtype<T>());
      ws.integrated_power =
          make_device_tensor(output_shape, real_dtype<T>());
    } catch (const std::exception& e) {
      return fail_runtime(e.what());
    }
    ws.batch_in = plan.batch_in;
    ws.out_batch = plan.out_batch;
    ws.num_doppler = num_doppler;
    ws.num_range = num_range;
  }
  auto& score = ws.score;
  auto& noise = ws.noise;
  auto& mask = ws.mask;
  auto& peak_mask = ws.peak_mask;
  auto& power = ws.power;
  auto& integrated_power = ws.integrated_power;

  HOLOSCAN_TRY(auto score_ptr, writable_ptr(*score, "score"));
  HOLOSCAN_TRY(auto noise_ptr, writable_ptr(*noise, "noise"));
  HOLOSCAN_TRY(auto mask_ptr, writable_ptr(*mask, "mask"));
  HOLOSCAN_TRY(auto peak_mask_ptr, writable_ptr(*peak_mask, "peak_mask"));
  HOLOSCAN_TRY(auto power_ptr, writable_ptr(*power, "power"));
  HOLOSCAN_TRY(auto integrated_power_ptr,
               writable_ptr(*integrated_power, "integrated_power"));

  const bool peak_only = pinned.peak_only;
  const bool peak_full8 =
      pinned.peak_neighborhood == CfarPeakNeighborhood::kSquare8;

  const int guard_d = static_cast<int>(effective_guard_cells[0]);
  const int guard_r = static_cast<int>(effective_guard_cells[1]);
  const int train_d = static_cast<int>(effective_training_cells[0]);
  const int train_r = static_cast<int>(effective_training_cells[1]);

  const CfarVariant variant = pinned.cfar_variant;
  if (variant == CfarVariant::kOS) {
    if (effective_os_rank < 0) {
      return fail_invalid(
          "CFARDetectorOp: os_rank must be non-negative for OS variant");
    }
    const int64_t max_train = detail::max_training_cell_count(
        static_cast<int>(num_doppler), static_cast<int>(num_range), guard_d,
        guard_r, train_d, train_r);
    if (max_train > detail::kMaxOsTrainingCells) {
      return fail_invalid(
          "CFARDetectorOp: OS window footprint exceeds the supported "
          "training-cell limit");
    }
    if (effective_os_rank >= max_train) {
      return fail_invalid(
          "CFARDetectorOp: os_rank must be less than the maximum training-"
          "cell count for the configured window footprint");
    }
  }

  const bool map_independent_variant =
      variant == CfarVariant::kCA || variant == CfarVariant::kOS;
  const int win_d = 2 * (guard_d + train_d) + 1;
  const int win_r = 2 * (guard_r + train_r) + 1;
  const bool use_canonical_dims =
      map_independent_variant && num_doppler >= win_d && num_range >= win_r;
  if (map_independent_variant && !use_canonical_dims &&
      !window_exceeds_map_warned_) {
    HOLOSCAN_LOG_WARN(
        "CFARDetectorOp: CFAR window ({}x{}) exceeds the RD-map extent "
        "({}x{}); building a map-specific alpha LUT with reduced-N edge "
        "thresholds (the start-time pre-warmed table does not apply).",
        win_d, win_r, num_doppler, num_range);
    window_exceeds_map_warned_ = true;
  }
  const int key_num_doppler =
      use_canonical_dims ? win_d : static_cast<int>(num_doppler);
  const int key_num_range =
      use_canonical_dims ? win_r : static_cast<int>(num_range);

  AlphaLutKey lut_key{variant,
                      effective_pfa,
                      plan.fold,
                      effective_os_rank,
                      guard_d,
                      guard_r,
                      train_d,
                      train_r,
                      key_num_doppler,
                      key_num_range};

  auto upload_alpha_lut =
      [&](const std::vector<double>& host_lut) -> expected<void, Error> {
    std::vector<T> lut_t(host_lut.size());
    for (size_t i = 0; i < host_lut.size(); ++i) {
      lut_t[i] = static_cast<T>(host_lut[i]);
    }
    try {
      alpha_lut_dev_ = make_device_tensor(
          {static_cast<int64_t>(lut_t.size())}, real_dtype<T>());
    } catch (const std::exception& e) {
      return fail_runtime(e.what());
    }
    HOLOSCAN_TRY(auto lut_mut, writable_ptr(*alpha_lut_dev_, "alpha_lut"));
    if (cudaMemcpyAsync(lut_mut, lut_t.data(), lut_t.size() * sizeof(T),
                        cudaMemcpyHostToDevice, stream) != cudaSuccess) {
      return fail_runtime("CFARDetectorOp: alpha LUT H2D copy failed");
    }
    return {};
  };

  const bool had_external_alpha_lut = using_external_alpha_lut_;
  const bool has_external_alpha_lut = !effective_alpha_lut.empty();

  if (had_external_alpha_lut && !has_external_alpha_lut) {
    if (!alpha_lut_cleared_pfa_active_warned_) {
      HOLOSCAN_LOG_WARN(
          "CFARDetectorOp: alpha_lut cleared; pfa-derived alpha calibration is "
          "now active and may use the slow Monte Carlo path for GO, SO, or OS "
          "with K > 1");
      alpha_lut_cleared_pfa_active_warned_ = true;
    }
    using_external_alpha_lut_ = false;
    alpha_lut_host_cache_.clear();
    alpha_lut_dev_.reset();
    alpha_lut_key_ = {};
    pending_lut_key_ = {};
    pending_lut_active_ = false;
    stale_lut_warned_ = false;
    external_alpha_lut_validated_ = false;
  }

  if (pfa_override_changed && has_external_alpha_lut &&
      !pfa_ignored_while_alpha_lut_warned_) {
    HOLOSCAN_LOG_WARN(
        "CFARDetectorOp: pfa override has no effect while alpha_lut is "
        "supplied");
    pfa_ignored_while_alpha_lut_warned_ = true;
  }

  if (has_external_alpha_lut) {
    const bool lut_changed = effective_alpha_lut != alpha_lut_host_cache_;
    const bool geometry_changed =
        !external_alpha_lut_validated_ ||
        validated_external_variant_ != variant ||
        validated_external_fold_ != plan.fold ||
        validated_external_num_doppler_ != num_doppler ||
        validated_external_num_range_ != num_range ||
        validated_external_guard_d_ != guard_d ||
        validated_external_guard_r_ != guard_r ||
        validated_external_train_d_ != train_d ||
        validated_external_train_r_ != train_r;
    if (lut_changed || geometry_changed) {
      const int64_t max_lut_index = max_alpha_lut_index(
          variant, static_cast<int>(num_doppler), static_cast<int>(num_range),
          guard_d, guard_r, train_d, train_r);
      HOLOSCAN_TRY(validate_external_alpha_lut(
          variant, plan.fold, effective_alpha_lut, max_lut_index));
      external_alpha_lut_validated_ = true;
      validated_external_variant_ = variant;
      validated_external_fold_ = plan.fold;
      validated_external_num_doppler_ = num_doppler;
      validated_external_num_range_ = num_range;
      validated_external_guard_d_ = guard_d;
      validated_external_guard_r_ = guard_r;
      validated_external_train_d_ = train_d;
      validated_external_train_r_ = train_r;
      if (lut_changed) {
        alpha_lut_host_cache_ = effective_alpha_lut;
        HOLOSCAN_TRY(upload_alpha_lut(alpha_lut_host_cache_));
      }
    }
    using_external_alpha_lut_ = true;
  } else {
    using_external_alpha_lut_ = false;
    const bool blocking_lut = pinned.blocking_alpha_lut_rebuild;
    if (lut_key != alpha_lut_key_) {
      if (!(pending_lut_active_ && pending_lut_key_ == lut_key)) {
        runtime_build_warned_ = false;
        stale_lut_warned_ = false;
      }
      const auto adopt_ready_lut =
          [&](std::vector<double> lut) -> expected<void, Error> {
        alpha_lut_host_cache_ = std::move(lut);
        alpha_lut_key_ = lut_key;
        pending_lut_key_ = {};
        pending_lut_active_ = false;
        stale_lut_warned_ = false;
        return upload_alpha_lut(alpha_lut_host_cache_);
      };
      const auto warn_slow_pfa_lut = [&]() {
        if (pfa_lut_slow_warned_ ||
            !uses_slow_pfa_lut_calibration(lut_key.variant, lut_key.fold)) {
          return;
        }
        HOLOSCAN_LOG_WARN(
            "CFARDetectorOp: deriving the alpha LUT from pfa for {} is the "
            "slow calibration path (Monte Carlo LUT build); for low-latency "
            "live updates supply a precomputed alpha_lut",
            to_string(lut_key.variant));
        pfa_lut_slow_warned_ = true;
      };
      const auto warn_runtime_rebuild = [&]() {
        if (runtime_build_warned_) {
          return;
        }
        HOLOSCAN_LOG_WARN(
            "CFARDetectorOp: calibrating the alpha LUT at runtime (the "
            "configuration was not pre-warmed at start — e.g. a [live] "
            "override, an NCI fold not covered by nci_fold, or a sub-window "
            "map); this stalls the tick or serves a stale LUT while building.");
        runtime_build_warned_ = true;
      };
      detail::AlphaLutSnapshot snap = detail::poll_cached_alpha_lut(lut_key);
      if (snap.ready) {
        HOLOSCAN_TRY(adopt_ready_lut(std::move(snap.lut)));
      } else if (blocking_lut || alpha_lut_host_cache_.empty()) {
        warn_runtime_rebuild();
        warn_slow_pfa_lut();
        try {
          HOLOSCAN_TRY(
              adopt_ready_lut(detail::get_cached_alpha_lut(lut_key)));
        } catch (const std::exception& e) {
          return fail_runtime(e.what());
        }
      } else {
        warn_runtime_rebuild();
        warn_slow_pfa_lut();
        detail::ensure_alpha_lut_build(lut_key);
        pending_lut_key_ = lut_key;
        pending_lut_active_ = true;
        if (!stale_lut_warned_) {
          HOLOSCAN_LOG_WARN(
              "CFARDetectorOp: alpha LUT for the new configuration is still "
              "calibrating; using the previous LUT for this tick");
          stale_lut_warned_ = true;
        }
      }
    } else if (pending_lut_active_ && pending_lut_key_ == lut_key) {
      detail::AlphaLutSnapshot snap = detail::poll_cached_alpha_lut(lut_key);
      if (snap.ready) {
        alpha_lut_host_cache_ = std::move(snap.lut);
        alpha_lut_key_ = lut_key;
        pending_lut_key_ = {};
        pending_lut_active_ = false;
        stale_lut_warned_ = false;
        HOLOSCAN_TRY(upload_alpha_lut(alpha_lut_host_cache_));
      }
    }
  }

  if (!alpha_lut_dev_) {
    return fail_runtime("CFARDetectorOp: alpha LUT device buffer is unset");
  }

  const int64_t* nci_table_dev_ptr = nullptr;
  if (plan.fold > 1) {
    if (plan.batch_in_table != nci_table_host_cache_) {
      try {
        nci_table_dev_ = make_device_tensor(
            {static_cast<int64_t>(plan.batch_in_table.size())}, kInt64Dtype);
      } catch (const std::exception& e) {
        return fail_runtime(e.what());
      }
      HOLOSCAN_TRY(auto nci_mut, writable_ptr(*nci_table_dev_, "nci_table"));
      if (cudaMemcpyAsync(nci_mut, plan.batch_in_table.data(),
                          plan.batch_in_table.size() * sizeof(int64_t),
                          cudaMemcpyHostToDevice, stream) != cudaSuccess) {
        return fail_runtime("CFARDetectorOp: NCI table H2D copy failed");
      }
      nci_table_host_cache_ = plan.batch_in_table;
    }
    nci_table_dev_ptr = static_cast<const int64_t*>(nci_table_dev_->data());
  }

  try {
    detail::run_cfar_stage<T>(
        variant, input.data(), input_is_complex, plan.batch_in, plan.out_batch,
        num_doppler, num_range, guard_d, guard_r, train_d, train_r,
        effective_os_rank, effective_threshold_scale, plan.fold, nci_table_dev_ptr,
        alpha_lut_dev_->data(),
        static_cast<int64_t>(alpha_lut_host_cache_.size()), power_ptr,
        integrated_power_ptr, score_ptr, noise_ptr, mask_ptr, peak_only,
        peak_full8, peak_only ? peak_mask_ptr : nullptr, stream);
  } catch (const std::exception& e) {
    return fail_runtime(e.what());
  }

  const void* det_mask_src =
      peak_only ? peak_mask->data() : mask->data();
  const bool* det_mask = static_cast<const bool*>(det_mask_src);

  const int64_t plane = num_doppler * num_range;
  const int64_t out_elems = plan.out_batch * plane;
  const int64_t idx_cols = 2 + plan.n_preserved;

  if (!detection_idx_compact_) {
    detection_idx_compact_ =
        std::make_shared<detail::DetectionIdxCompactState>();
  }
  int64_t total_detections = 0;
  try {
    total_detections = detail::count_mask_detections(
        det_mask, out_elems, detection_idx_compact_.get(), stream);
  } catch (const std::exception& e) {
    return fail_runtime(e.what());
  }

  const bool capped = effective_max_detections > 0;
  const int64_t num_detections =
      capped ? std::min(total_detections, effective_max_detections)
             : total_detections;
  if (capped && total_detections > effective_max_detections) {
    HOLOSCAN_LOG_WARN(
        "CFARDetectorOp: detections exceeded max_detections ({}); surplus "
        "dropped",
        effective_max_detections);
  }

  const std::vector<int64_t> idx_shape = {num_detections, idx_cols};
  // Prepare every output loan (allocate / fill / commit) before any emit so a
  // late fill failure cannot leave detection_idx already published.
  auto idx_loan = detection_idx.allocate_tensor(
      TensorLoanRequest{.shape = idx_shape, .dtype = kInt64Dtype});
  if (!idx_loan) {
    return make_unexpected(std::move(idx_loan).error());
  }
  auto idx_writer = idx_loan->write(context.cuda_stream());
  if (!idx_writer) {
    return make_unexpected(std::move(idx_writer).error());
  }
  int64_t* idx_ptr = static_cast<int64_t*>(idx_writer->data());
  try {
    detail::compact_detection_idx(det_mask, out_elems, num_doppler, num_range,
                                  plan.preserved_dims, num_detections, idx_ptr,
                                  detection_idx_compact_.get(), stream);
  } catch (const std::exception& e) {
    return fail_runtime(e.what());
  }
  auto idx_committed = std::move(*idx_writer).commit();
  if (!idx_committed) {
    return make_unexpected(std::move(idx_committed).error());
  }

  const void* power_src =
      plan.fold > 1 ? integrated_power->data() : power->data();
  HOLOSCAN_TRY(auto power_loan,
               prepare_copy_device(cfar_power, context, output_shape,
                                  real_dtype<T>(), power_src, stream));
  HOLOSCAN_TRY(auto mask_loan,
               prepare_copy_device(detection_mask, context, output_shape,
                                  kMaskDtype, det_mask_src, stream));
  HOLOSCAN_TRY(auto score_loan,
               prepare_copy_device(detection_score, context, output_shape,
                                  real_dtype<T>(), score->data(), stream));
  HOLOSCAN_TRY(auto noise_loan,
               prepare_copy_device(noise_estimate, context, output_shape,
                                  real_dtype<T>(), noise->data(), stream));

  HOLOSCAN_TRY(
      emit_prepared_tensor(detection_idx, context, std::move(*idx_loan)));
  HOLOSCAN_TRY(
      emit_prepared_tensor(cfar_power, context, std::move(power_loan)));
  HOLOSCAN_TRY(
      emit_prepared_tensor(detection_mask, context, std::move(mask_loan)));
  HOLOSCAN_TRY(
      emit_prepared_tensor(detection_score, context, std::move(score_loan)));
  HOLOSCAN_TRY(
      emit_prepared_tensor(noise_estimate, context, std::move(noise_loan)));

  retained_input_ = std::move(input);
  return {};
}

template class CFARDetectorOp<float>;
template class CFARDetectorOp<double>;

}  // namespace holoscan::radar
