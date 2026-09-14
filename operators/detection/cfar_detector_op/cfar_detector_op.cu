// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "detection/cfar_detector_op/cfar_detector_op.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
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
#include "radar_common/sdk/config_tensor.hpp"
#include "radar_common/sdk/override_param.hpp"
#include "radar_common/sdk/tensor_io.hpp"
#include "radar_common/tensor/dlpack_buffer.hpp"
#include "radar_common/tensor/extents_layout.hpp"
#include "radar_common/tensor/max_byte_span.hpp"
#include "radar_common/tensor/tensor_utils.hpp"

namespace holoscan::radar {

namespace detail {

struct CfarDetectorState {
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

  double last_seen_pfa{-1.0};
  bool pfa_seen{false};
  std::shared_ptr<holoscan::Tensor> nci_table_dev{};
  std::vector<int64_t> nci_table_host_cache{};
  CfarAlphaLutKey alpha_lut_key{};
  CfarAlphaLutKey pending_lut_key{};
  bool pending_lut_active{false};
  bool stale_lut_warned{false};
  bool pfa_lut_slow_warned{false};
  bool runtime_build_warned{false};
  bool window_exceeds_map_warned{false};
  bool using_external_alpha_lut{false};
  bool pfa_ignored_while_alpha_lut_warned{false};
  bool alpha_lut_cleared_pfa_active_warned{false};
  bool external_alpha_lut_validated{false};
  CfarVariant validated_external_variant{CfarVariant::kCA};
  int64_t validated_external_fold{0};
  int64_t validated_external_num_doppler{0};
  int64_t validated_external_num_range{0};
  int validated_external_guard_d{0};
  int validated_external_guard_r{0};
  int validated_external_train_d{0};
  int validated_external_train_r{0};
  std::vector<double> alpha_lut_host_cache{};
  std::shared_ptr<holoscan::Tensor> alpha_lut_dev{};
  DetectionIdxCompactState detection_idx_compact{};
  DenseWorkspace dense_workspace[2]{};
  int dense_workspace_idx{0};
};

}  // namespace detail

namespace {

constexpr const char* kOpName = "CFARDetectorOp";
// detection_mask is emitted as uint8 (0/1), not kDLBool: byte-identical to the
// 0/1 bytes kernels write; NumPy-visible.
constexpr DLDataType kMaskDtype{kDLUInt, 8, 1};
constexpr DLDataType kInt64Dtype{kDLInt, 64, 1};
constexpr DLDataType kFloat64Dtype{kDLFloat, 64, 1};
constexpr const char* kMissingOsRankMessage =
    "CFARDetectorOp: os_rank is required for OS variant; "  // NOLINT(whitespace/indent_namespace)
    "set a 0-based rank into the ascending-sorted "  // NOLINT(whitespace/indent_namespace)
    "training cells";  // NOLINT(whitespace/indent_namespace)

struct CfarStagePlan {
  detail::CfarStageConfig stage;
  detail::NciPlan nci;
  bool input_is_complex;
  int64_t effective_fold;
  int64_t num_doppler;
  int64_t num_range;
  int guard_d;
  int guard_r;
  int train_d;
  int train_r;
  int key_num_doppler;
  int key_num_range;
  std::vector<int64_t> output_shape;
};

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

struct PreparedDenseWorkspace {
  detail::CfarDetectorState::DenseWorkspace* tensors;
  void* score;
  void* noise;
  void* mask;
  void* peak_mask;
  void* power;
  void* integrated_power;
};

template <typename T>
expected<PreparedDenseWorkspace, Error> prepare_dense_workspace(
    detail::CfarDetectorState& state, const CfarStagePlan& plan) {
  state.dense_workspace_idx ^= 1;
  auto& ws = state.dense_workspace[state.dense_workspace_idx];
  const bool cache_hit = ws.score && ws.batch_in == plan.nci.batch_in &&
                         ws.out_batch == plan.nci.out_batch &&
                         ws.num_doppler == plan.num_doppler &&
                         ws.num_range == plan.num_range;
  if (!cache_hit) {
    try {
      ws.score = make_device_tensor(plan.output_shape, real_dtype<T>());
      ws.noise = make_device_tensor(plan.output_shape, real_dtype<T>());
      ws.mask = make_device_tensor(plan.output_shape, kMaskDtype);
      ws.peak_mask = make_device_tensor(plan.output_shape, kMaskDtype);
      ws.power = make_device_tensor(
          plan.nci.fold > 1
              ? std::vector<int64_t>{plan.nci.batch_in, plan.num_doppler,
                                     plan.num_range}
              : plan.output_shape,
          real_dtype<T>());
      ws.integrated_power =
          make_device_tensor(plan.output_shape, real_dtype<T>());
    } catch (const std::exception& e) {
      return make_unexpected(Error{ErrorCode::kFailure, e.what()});
    }
    ws.batch_in = plan.nci.batch_in;
    ws.out_batch = plan.nci.out_batch;
    ws.num_doppler = plan.num_doppler;
    ws.num_range = plan.num_range;
  }

  HOLOSCAN_TRY(auto score, writable_ptr(*ws.score, "score"));
  HOLOSCAN_TRY(auto noise, writable_ptr(*ws.noise, "noise"));
  HOLOSCAN_TRY(auto mask, writable_ptr(*ws.mask, "mask"));
  HOLOSCAN_TRY(auto peak_mask, writable_ptr(*ws.peak_mask, "peak_mask"));
  HOLOSCAN_TRY(auto power, writable_ptr(*ws.power, "power"));
  HOLOSCAN_TRY(auto integrated_power,
               writable_ptr(*ws.integrated_power, "integrated_power"));
  return PreparedDenseWorkspace{
      .tensors = &ws,
      .score = score,
      .noise = noise,
      .mask = mask,
      .peak_mask = peak_mask,
      .power = power,
      .integrated_power = integrated_power};
}

expected<TensorOutputLoan, Error> prepare_detection_idx(
    Output<Tensor>& output, ExecutionContext& context,
    const CfarStagePlan& plan, const bool* detection_mask,
    detail::DetectionIdxCompactState& compact_state, cudaStream_t stream) {
  const int64_t plane = plan.num_doppler * plan.num_range;
  const int64_t out_elems = plan.nci.out_batch * plane;
  const int64_t idx_cols = 2 + plan.nci.n_preserved;
  int64_t total_detections = 0;
  try {
    total_detections = detail::count_mask_detections(
        detection_mask, out_elems, &compact_state, stream);
  } catch (const std::exception& e) {
    return make_unexpected(Error{ErrorCode::kFailure, e.what()});
  }

  const bool capped = plan.stage.max_detections > 0;
  const int64_t num_detections =
      capped ? std::min(total_detections, plan.stage.max_detections)
             : total_detections;
  if (capped && total_detections > plan.stage.max_detections) {
    HOLOSCAN_LOG_WARN(
        "CFARDetectorOp: detections exceeded max_detections ({}); surplus "
        "dropped",
        plan.stage.max_detections);
  }

  const std::vector<int64_t> idx_shape = {num_detections, idx_cols};
  auto loan = output.allocate_tensor(
      TensorLoanRequest{.shape = idx_shape, .dtype = kInt64Dtype});
  if (!loan) {
    return make_unexpected(std::move(loan).error());
  }
  auto writer = loan->write(context.cuda_stream());
  if (!writer) {
    return make_unexpected(std::move(writer).error());
  }
  int64_t* idx_ptr = static_cast<int64_t*>(writer->data());
  try {
    detail::compact_detection_idx(
        detection_mask, out_elems, plan.num_doppler, plan.num_range,
        plan.nci.preserved_dims, num_detections, idx_ptr, &compact_state,
        stream);
  } catch (const std::exception& e) {
    return make_unexpected(Error{ErrorCode::kFailure, e.what()});
  }
  auto committed = std::move(*writer).commit();
  if (!committed) {
    return make_unexpected(std::move(committed).error());
  }
  return std::move(*loan);
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
    int64_t max_lut_index, int num_doppler, int num_range, int guard_d,
    int guard_r, int train_d, int train_r) {
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
  std::vector<int64_t> required_indices;
  if (variant == CfarVariant::kGO || variant == CfarVariant::kSO) {
    required_indices = detail::reachable_go_lut_indices(
        num_doppler, num_range, guard_d, guard_r, train_d, train_r);
  } else if (variant == CfarVariant::kOS && fold > 1) {
    required_indices = detail::reachable_training_cell_counts(
        num_doppler, num_range, guard_d, guard_r, train_d, train_r);
  } else {
    required_indices.reserve(static_cast<size_t>(max_lut_index));
    for (int64_t idx = 1; idx <= max_lut_index; ++idx) {
      required_indices.push_back(idx);
    }
  }
  for (const int64_t idx : required_indices) {
    if (lut[static_cast<size_t>(idx)] <= 0.0) {
      return fail_invalid(
          "CFARDetectorOp: alpha_lut entries for reachable indices must be "
          "finite and positive");
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

expected<void, Error> require_window_bounds(
    const std::vector<int64_t>& value, const char* name,
    int64_t max_doppler, int64_t max_range) {
  HOLOSCAN_TRY(require_pair(value, name));
  if (value[0] > max_doppler || value[1] > max_range) {
    return fail_invalid(std::string("CFARDetectorOp: '") + name +
                        "' exceeds MaxTensorExtents map bounds");
  }
  return {};
}

// resolve_live_override_via for a bounded host-Tensor vector payload: read and
// validate a matched sample, otherwise fall back to the Params seed.
template <typename T>
expected<std::vector<T>, Error> resolve_config_tensor_override(
    Input<Tensor>& port, const std::vector<T>& seed,
    DLDataType dtype, int64_t max_extent, const char* field) {
  return resolve_live_override_via(
      port, seed, [&](const Tensor& value) {
        return read_host_config_vector<T>(value, dtype, max_extent, field);
      });
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

expected<detail::CfarStageConfig, Error> resolve_params(
    Input<int64_t>& os_rank, Input<CfarCellWindowOverride>& guard_cells,
    Input<CfarCellWindowOverride>& training_cells, Input<double>& pfa,
    Input<Tensor>& alpha_lut, Input<double>& threshold_scale,
    Input<Tensor>& nci_axes, Input<int64_t>& max_detections,
    const CFARParams& pinned, std::size_t max_leading,
    int64_t max_alpha_lut_extent, const MaxTensorExtents& extents) {
  detail::CfarStageConfig stage;
  stage.cfar_variant = pinned.cfar_variant;
  HOLOSCAN_TRY(stage.os_rank,
               resolve_live_override(os_rank, pinned.os_rank));
  HOLOSCAN_TRY(
      stage.guard_cells,
      resolve_live_override_via(
          guard_cells, pinned.guard_cells,
          [](const CfarCellWindowOverride& value) { return from_wire(value); }));
  HOLOSCAN_TRY(
      stage.training_cells,
      resolve_live_override_via(
          training_cells, pinned.training_cells,
          [](const CfarCellWindowOverride& value) { return from_wire(value); }));
  HOLOSCAN_TRY(stage.pfa, resolve_live_override(pfa, pinned.pfa));

  int64_t alpha_lut_extent_bound = max_alpha_lut_extent;
  if (stage.cfar_variant == CfarVariant::kOS) {
    alpha_lut_extent_bound =
        static_cast<int64_t>(detail::kMaxOsTrainingCells) + 1;
  } else if (stage.cfar_variant == CfarVariant::kGO ||
             stage.cfar_variant == CfarVariant::kSO) {
    alpha_lut_extent_bound =
        std::max(*extents.max_doppler_bins, *extents.max_range_bins) + 1;
  }
  HOLOSCAN_TRY(stage.alpha_lut,
               resolve_config_tensor_override(
                   alpha_lut, pinned.alpha_lut, kFloat64Dtype,
                   alpha_lut_extent_bound, "alpha_lut"));
  HOLOSCAN_TRY(stage.threshold_scale,
               resolve_live_override(threshold_scale,
                                     pinned.threshold_scale));
  HOLOSCAN_TRY(stage.nci_axes,
               resolve_config_tensor_override(
                   nci_axes, pinned.nci_axes, kInt64Dtype,
                   static_cast<int64_t>(max_leading), "nci_axes"));
  stage.preintegrated_looks = pinned.preintegrated_looks;
  HOLOSCAN_TRY(stage.max_detections,
               resolve_live_override(max_detections,
                                     pinned.max_detections));
  stage.blocking_alpha_lut_rebuild = pinned.blocking_alpha_lut_rebuild;
  stage.peak_only = pinned.peak_only;
  stage.peak_neighborhood = pinned.peak_neighborhood;
  return stage;
}

template <typename T>
expected<CfarStagePlan, Error> build_cfar_stage_plan(
    detail::CfarStageConfig stage, const Tensor& input,
    const MaxTensorExtents& extents, std::size_t max_leading) {
  try {
    const InputPrecision precision =
        classify_input_dtype<T>(input, "rd_maps", /*allow_real=*/true);
    require_cuda_accessible(input, "rd_maps", kOpName);
    require_contiguous(input, "rd_maps", kOpName);

    HOLOSCAN_TRY(require_window_bounds(
        stage.guard_cells, "guard_cells", *extents.max_doppler_bins,
        *extents.max_range_bins));
    HOLOSCAN_TRY(require_window_bounds(
        stage.training_cells, "training_cells", *extents.max_doppler_bins,
        *extents.max_range_bins));

    const std::size_t total_rank = input.shape().size();
    if (total_rank < 2) {
      return make_unexpected(Error{
          ErrorCode::kInvalidArgument,
          "CFARDetectorOp: rd_maps must have at least 2 axes "
          "(num_doppler, num_range)"});
    }
    const std::size_t leading_rank = total_rank - 2;
    if (leading_rank > max_leading) {
      return make_unexpected(
          Error{ErrorCode::kInvalidArgument,
                unexpected_leading_message(kOpName, leading_rank,
                                           max_leading)});
    }
    const int64_t num_doppler = input.shape()[total_rank - 2];
    const int64_t num_range = input.shape()[total_rank - 1];
    if (num_doppler > *extents.max_doppler_bins) {
      return make_unexpected(Error{
          ErrorCode::kInvalidArgument,
          "CFARDetectorOp: input doppler_bins exceed MaxTensorExtents"});
    }
    if (num_range > *extents.max_range_bins) {
      return make_unexpected(Error{
          ErrorCode::kInvalidArgument,
          "CFARDetectorOp: input range_bins exceed MaxTensorExtents"});
    }
    const int64_t max_rx = extent_rx_or_channels(extents);
    const int64_t max_channels = extents.max_channels.value_or(max_rx);
    if (leading_rank == 1 && input.shape()[0] > max_channels) {
      return make_unexpected(Error{
          ErrorCode::kInvalidArgument,
          std::string(kOpName) +
              ": input leading axis exceeds max_rx/max_channels from "
              "MaxTensorExtents"});
    }
    if (leading_rank == 2 &&
        (input.shape()[0] > extent_or_one(extents.max_tx) ||
         input.shape()[1] > max_rx)) {
      return make_unexpected(Error{
          ErrorCode::kInvalidArgument,
          std::string(kOpName) +
              ": input leading axes exceed max_tx and max_rx/max_channels "
              "from MaxTensorExtents"});
    }
    if (extents.max_detections.has_value() &&
        stage.max_detections > *extents.max_detections) {
      return make_unexpected(Error{
          ErrorCode::kInvalidArgument,
          "CFARDetectorOp: max_detections exceeds "
          "MaxTensorExtents.max_detections"});
    }

    detail::NciPlan nci =
        detail::NciPlan::from_shape(input.shape(), stage.nci_axes, "rd_maps");
    if (stage.preintegrated_looks <= 0) {
      return make_unexpected(Error{
          ErrorCode::kInvalidArgument,
          "CFARDetectorOp: preintegrated_looks must be positive"});
    }
    if (precision == InputPrecision::kComplex &&
        stage.preintegrated_looks != 1) {
      return make_unexpected(Error{
          ErrorCode::kInvalidArgument,
          "CFARDetectorOp: preintegrated_looks applies only to real power "
          "input"});
    }
    if (nci.fold >
        std::numeric_limits<int64_t>::max() / stage.preintegrated_looks) {
      return make_unexpected(Error{
          ErrorCode::kInvalidArgument,
          "CFARDetectorOp: effective integration fold overflows int64"});
    }
    const int64_t effective_fold =
        nci.fold * stage.preintegrated_looks;
    const int guard_d = static_cast<int>(stage.guard_cells[0]);
    const int guard_r = static_cast<int>(stage.guard_cells[1]);
    const int train_d = static_cast<int>(stage.training_cells[0]);
    const int train_r = static_cast<int>(stage.training_cells[1]);
    if (stage.cfar_variant == CfarVariant::kOS) {
      if (stage.os_rank == -1) {
        HOLOSCAN_LOG_ERROR("{}", kMissingOsRankMessage);
        return make_unexpected(Error{
            ErrorCode::kInvalidArgument, kMissingOsRankMessage});
      }
      if (stage.os_rank < -1) {
        return make_unexpected(Error{
            ErrorCode::kInvalidArgument,
            "CFARDetectorOp: os_rank must be non-negative for OS variant"});
      }
      const int64_t max_train = detail::max_training_cell_count(
          static_cast<int>(num_doppler), static_cast<int>(num_range), guard_d,
          guard_r, train_d, train_r);
      if (max_train > detail::kMaxOsTrainingCells) {
        return make_unexpected(Error{
            ErrorCode::kInvalidArgument,
            "CFARDetectorOp: OS window footprint exceeds the supported "
            "training-cell limit"});
      }
      if (stage.os_rank >= max_train) {
        return make_unexpected(Error{
            ErrorCode::kInvalidArgument,
            "CFARDetectorOp: os_rank must be less than the maximum training-"
            "cell count for the configured window footprint"});
      }
    }

    const bool map_independent =
        stage.cfar_variant == CfarVariant::kCA ||
        stage.cfar_variant == CfarVariant::kOS;
    const int win_d = 2 * (guard_d + train_d) + 1;
    const int win_r = 2 * (guard_r + train_r) + 1;
    const bool canonical =
        map_independent && num_doppler >= win_d && num_range >= win_r;
    std::vector<int64_t> output_shape =
        nci.output_shape(num_doppler, num_range);
    return CfarStagePlan{
        .stage = std::move(stage),
        .nci = std::move(nci),
        .input_is_complex = precision == InputPrecision::kComplex,
        .effective_fold = effective_fold,
        .num_doppler = num_doppler,
        .num_range = num_range,
        .guard_d = guard_d,
        .guard_r = guard_r,
        .train_d = train_d,
        .train_r = train_r,
        .key_num_doppler =
            canonical ? win_d : static_cast<int>(num_doppler),
        .key_num_range = canonical ? win_r : static_cast<int>(num_range),
        .output_shape = std::move(output_shape)};
  } catch (const std::invalid_argument& e) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument, std::string(e.what())});
  }
}

template <typename T>
expected<void, Error> upload_alpha_lut(
    detail::CfarDetectorState& state, const std::vector<double>& host_lut,
    cudaStream_t stream) {
  std::vector<T> lut_t(host_lut.size());
  for (size_t i = 0; i < host_lut.size(); ++i) {
    lut_t[i] = static_cast<T>(host_lut[i]);
  }
  try {
    state.alpha_lut_dev = make_device_tensor(
        {static_cast<int64_t>(lut_t.size())}, real_dtype<T>());
  } catch (const std::exception& e) {
    return fail_runtime(e.what());
  }
  HOLOSCAN_TRY(auto lut_mut,
               writable_ptr(*state.alpha_lut_dev, "alpha_lut"));
  if (cudaMemcpyAsync(lut_mut, lut_t.data(), lut_t.size() * sizeof(T),
                      cudaMemcpyHostToDevice, stream) != cudaSuccess) {
    return fail_runtime("CFARDetectorOp: alpha LUT H2D copy failed");
  }
  return {};
}

template <typename T>
expected<void, Error> update_alpha_lut(
    detail::CfarDetectorState& state, const CfarStagePlan& plan,
    cudaStream_t stream) {
  const auto& stage = plan.stage;
  const bool pfa_override_changed =
      state.pfa_seen && stage.pfa != state.last_seen_pfa;
  state.last_seen_pfa = stage.pfa;
  state.pfa_seen = true;

  const bool map_independent =
      stage.cfar_variant == CfarVariant::kCA ||
      stage.cfar_variant == CfarVariant::kOS;
  const int win_d = 2 * (plan.guard_d + plan.train_d) + 1;
  const int win_r = 2 * (plan.guard_r + plan.train_r) + 1;
  if (map_independent &&
      (plan.key_num_doppler != win_d || plan.key_num_range != win_r) &&
      !state.window_exceeds_map_warned) {
    HOLOSCAN_LOG_WARN(
        "CFARDetectorOp: CFAR window ({}x{}) exceeds the RD-map extent "
        "({}x{}); building a map-specific alpha LUT with reduced-N edge "
        "thresholds (the start-time pre-warmed table does not apply).",
        win_d, win_r, plan.num_doppler, plan.num_range);
    state.window_exceeds_map_warned = true;
  }

  const CfarAlphaLutKey lut_key{
      stage.cfar_variant, stage.pfa,         plan.effective_fold,
      stage.os_rank,      plan.guard_d,      plan.guard_r,
      plan.train_d,       plan.train_r,      plan.key_num_doppler,
      plan.key_num_range};
  const bool had_external_alpha_lut = state.using_external_alpha_lut;
  const bool has_external_alpha_lut = !stage.alpha_lut.empty();

  if (had_external_alpha_lut && !has_external_alpha_lut) {
    if (!state.alpha_lut_cleared_pfa_active_warned) {
      HOLOSCAN_LOG_WARN(
          "CFARDetectorOp: alpha_lut cleared; pfa-derived alpha calibration is "
          "now active and may use the slow Monte Carlo path for GO, SO, or OS "
          "with K > 1");
      state.alpha_lut_cleared_pfa_active_warned = true;
    }
    state.using_external_alpha_lut = false;
    state.alpha_lut_host_cache.clear();
    state.alpha_lut_dev.reset();
    state.alpha_lut_key = {};
    state.pending_lut_key = {};
    state.pending_lut_active = false;
    state.stale_lut_warned = false;
    state.external_alpha_lut_validated = false;
  }

  if (pfa_override_changed && has_external_alpha_lut &&
      !state.pfa_ignored_while_alpha_lut_warned) {
    HOLOSCAN_LOG_WARN(
        "CFARDetectorOp: pfa override has no effect while alpha_lut is "
        "supplied");
    state.pfa_ignored_while_alpha_lut_warned = true;
  }

  if (has_external_alpha_lut) {
    const bool lut_changed = stage.alpha_lut != state.alpha_lut_host_cache;
    const bool geometry_changed =
        !state.external_alpha_lut_validated ||
        state.validated_external_variant != stage.cfar_variant ||
        state.validated_external_fold != plan.effective_fold ||
        state.validated_external_num_doppler != plan.num_doppler ||
        state.validated_external_num_range != plan.num_range ||
        state.validated_external_guard_d != plan.guard_d ||
        state.validated_external_guard_r != plan.guard_r ||
        state.validated_external_train_d != plan.train_d ||
        state.validated_external_train_r != plan.train_r;
    if (lut_changed || geometry_changed) {
      const int64_t max_lut_index = max_alpha_lut_index(
          stage.cfar_variant, static_cast<int>(plan.num_doppler),
          static_cast<int>(plan.num_range), plan.guard_d, plan.guard_r,
          plan.train_d, plan.train_r);
      HOLOSCAN_TRY(validate_external_alpha_lut(
          stage.cfar_variant, plan.effective_fold, stage.alpha_lut,
          max_lut_index,
          static_cast<int>(plan.num_doppler),
          static_cast<int>(plan.num_range), plan.guard_d, plan.guard_r,
          plan.train_d, plan.train_r));
      state.external_alpha_lut_validated = true;
      state.validated_external_variant = stage.cfar_variant;
      state.validated_external_fold = plan.effective_fold;
      state.validated_external_num_doppler = plan.num_doppler;
      state.validated_external_num_range = plan.num_range;
      state.validated_external_guard_d = plan.guard_d;
      state.validated_external_guard_r = plan.guard_r;
      state.validated_external_train_d = plan.train_d;
      state.validated_external_train_r = plan.train_r;
      if (lut_changed) {
        state.alpha_lut_host_cache = stage.alpha_lut;
        HOLOSCAN_TRY(upload_alpha_lut<T>(
            state, state.alpha_lut_host_cache, stream));
      }
    }
    state.using_external_alpha_lut = true;
  } else {
    state.using_external_alpha_lut = false;
    if (lut_key != state.alpha_lut_key) {
      if (!(state.pending_lut_active &&
            state.pending_lut_key == lut_key)) {
        state.runtime_build_warned = false;
        state.stale_lut_warned = false;
      }
      const auto adopt_ready_lut =
          [&](std::vector<double> lut) -> expected<void, Error> {
        state.alpha_lut_host_cache = std::move(lut);
        state.alpha_lut_key = lut_key;
        state.pending_lut_key = {};
        state.pending_lut_active = false;
        state.stale_lut_warned = false;
        return upload_alpha_lut<T>(
            state, state.alpha_lut_host_cache, stream);
      };
      const auto warn_slow_pfa_lut = [&]() {
        if (state.pfa_lut_slow_warned ||
            !uses_slow_pfa_lut_calibration(lut_key.variant, lut_key.fold)) {
          return;
        }
        HOLOSCAN_LOG_WARN(
            "CFARDetectorOp: deriving the alpha LUT from pfa for {} is the "
            "slow calibration path (Monte Carlo LUT build); for low-latency "
            "live updates supply a precomputed alpha_lut",
            to_string(lut_key.variant));
        state.pfa_lut_slow_warned = true;
      };
      const auto warn_runtime_rebuild = [&]() {
        if (state.runtime_build_warned) {
          return;
        }
        HOLOSCAN_LOG_WARN(
            "CFARDetectorOp: calibrating the alpha LUT at runtime (the "
            "configuration was not pre-warmed at start — e.g. a [live] "
            "override, an NCI fold not covered by nci_fold, or a sub-window "
            "map); this stalls the tick or serves a stale LUT while building.");
        state.runtime_build_warned = true;
      };
      detail::AlphaLutSnapshot snap =
          detail::poll_cached_alpha_lut(lut_key);
      if (snap.ready) {
        HOLOSCAN_TRY(adopt_ready_lut(std::move(snap.lut)));
      } else if (stage.blocking_alpha_lut_rebuild ||
                 state.alpha_lut_host_cache.empty()) {
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
        state.pending_lut_key = lut_key;
        state.pending_lut_active = true;
        if (!state.stale_lut_warned) {
          HOLOSCAN_LOG_WARN(
              "CFARDetectorOp: alpha LUT for the new configuration is still "
              "calibrating; using the previous LUT for this tick");
          state.stale_lut_warned = true;
        }
      }
    } else if (state.pending_lut_active &&
               state.pending_lut_key == lut_key) {
      detail::AlphaLutSnapshot snap =
          detail::poll_cached_alpha_lut(lut_key);
      if (snap.ready) {
        state.alpha_lut_host_cache = std::move(snap.lut);
        state.alpha_lut_key = lut_key;
        state.pending_lut_key = {};
        state.pending_lut_active = false;
        state.stale_lut_warned = false;
        HOLOSCAN_TRY(upload_alpha_lut<T>(
            state, state.alpha_lut_host_cache, stream));
      }
    }
  }

  if (!state.alpha_lut_dev) {
    return fail_runtime("CFARDetectorOp: alpha LUT device buffer is unset");
  }
  return {};
}

expected<const int64_t*, Error> update_nci_table(
    detail::CfarDetectorState& state, const detail::NciPlan& plan,
    cudaStream_t stream) {
  if (plan.fold <= 1) {
    return nullptr;
  }
  if (plan.batch_in_table != state.nci_table_host_cache) {
    try {
      state.nci_table_dev = make_device_tensor(
          {static_cast<int64_t>(plan.batch_in_table.size())}, kInt64Dtype);
    } catch (const std::exception& e) {
      return make_unexpected(Error{ErrorCode::kFailure, e.what()});
    }
    HOLOSCAN_TRY(auto nci_mut,
                 writable_ptr(*state.nci_table_dev, "nci_table"));
    if (cudaMemcpyAsync(nci_mut, plan.batch_in_table.data(),
                        plan.batch_in_table.size() * sizeof(int64_t),
                        cudaMemcpyHostToDevice, stream) != cudaSuccess) {
      return make_unexpected(Error{
          ErrorCode::kFailure, "CFARDetectorOp: NCI table H2D copy failed"});
    }
    state.nci_table_host_cache = plan.batch_in_table;
  }
  return static_cast<const int64_t*>(state.nci_table_dev->data());
}

}  // namespace

template <typename T>
CFARDetectorOp<T>::CFARDetectorOp(const MaxTensorExtents& extents) {
  state_ = std::make_shared<detail::CfarDetectorState>();
  const char* who = "CFARDetectorOp";
  const std::int64_t max_doppler =
      require_extent(extents.max_doppler_bins, "max_doppler_bins",
                     who);
  const std::int64_t max_range =
      require_extent(extents.max_range_bins, "max_range_bins", who);
  extents_ = extents;

  // Named leading axes: none, N_rx, or (M_tx, N_rx). Dense az/el instances
  // use max_tx/max_rx as the documented max_az/max_el pool-sizing stand-in.
  const std::size_t leading = named_tx_rx_leading_rank(extents);
  max_leading_ = leading;
  max_map_rank_ = static_cast<std::uint32_t>(2U + leading);
  if (max_doppler > std::numeric_limits<int>::max() ||
      max_range > std::numeric_limits<int>::max()) {
    throw std::invalid_argument(
        "CFARDetectorOp: map extents must fit int kernel dimensions");
  }

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

  const int64_t max_map_cells =
      product_extents({max_doppler, max_range}, who);
  if (max_map_cells == std::numeric_limits<int64_t>::max()) {
    throw std::invalid_argument(
        "CFARDetectorOp: alpha_lut extent bound overflows int64");
  }
  max_alpha_lut_extent_ = max_map_cells + 1;
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
    HOLOSCAN_LOG_WARN(
        "CFARDetectorOp: {} calibrates its alpha LUT on the first tick "
        "(go_lut_index tables are map-dependent and cannot be pre-warmed "
        "without the RD-map shape); supply alpha_lut to avoid a runtime build.",
        to_string(variant));
    return;
  }
  if (variant == CfarVariant::kOS && pinned.os_rank < 0) {
    // The tick planner reports the targeted required-parameter error. Avoid
    // attempting a meaningless start-time calibration with the unset sentinel.
    return;
  }
  const std::vector<int64_t>& guard = pinned.guard_cells;
  const std::vector<int64_t>& train = pinned.training_cells;
  if (guard.size() != 2 || train.size() != 2) {
    return;
  }

  if (pinned.preintegrated_looks <= 0) {
    HOLOSCAN_LOG_WARN(
        "CFARDetectorOp: preintegrated_looks must be positive; deferring "
        "validation to the first tick.");
    return;
  }
  int64_t fold = pinned.preintegrated_looks;
  if (!pinned.nci_axes.empty()) {
    const int64_t hint = pinned.nci_fold;
    if (hint <= 0) {
      HOLOSCAN_LOG_WARN(
          "CFARDetectorOp: nci_axes is set but nci_fold hint is unset; "
          "deferring alpha-LUT calibration to the first tick (set nci_fold to "
          "the integration fold K to pre-warm at start).");
      return;
    }
    if (hint >
        std::numeric_limits<int64_t>::max() / pinned.preintegrated_looks) {
      HOLOSCAN_LOG_WARN(
          "CFARDetectorOp: effective integration fold overflows int64; "
          "deferring validation to the first tick.");
      return;
    }
    fold *= hint;
  }

  const int gd = static_cast<int>(guard[0]);
  const int gr = static_cast<int>(guard[1]);
  const int td = static_cast<int>(train[0]);
  const int tr = static_cast<int>(train[1]);
  const int win_d = 2 * (gd + td) + 1;
  const int win_r = 2 * (gr + tr) + 1;

  const CfarAlphaLutKey warm_key{variant, pinned.pfa, fold, pinned.os_rank,
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
  constexpr TensorRepresentation kDevice{
      .memory_kind = MemoryKind::kCudaDevice,
      .dtype = std::nullopt,
      .rank = std::nullopt,
  };
  const TensorRepresentation real_device{
      .memory_kind = MemoryKind::kCudaDevice,
      .dtype = real_dtype<T>(),
      .rank = std::nullopt,
  };
  constexpr TensorRepresentation kMaskDevice{
      .memory_kind = MemoryKind::kCudaDevice,
      .dtype = kMaskDtype,
      .rank = std::nullopt,
  };
  constexpr TensorRepresentation kIndexDevice{
      .memory_kind = MemoryKind::kCudaDevice,
      .dtype = kInt64Dtype,
      .rank = 2U,
  };
  spec.input(rd_maps).queue_depth(2U).expects_tensor(tensor_input(kDevice));
  const TensorBounds map_bounds{
      .max_rank = max_map_rank_,
      .max_byte_span = max_map_byte_span_};
  // detection_mask is uint8 — map_bounds (real T) is a safe ceiling.
  spec.output(detection_mask)
      .produces_tensor(pooled_tensor_output(kMaskDevice, map_bounds));
  spec.output(detection_score)
      .produces_tensor(pooled_tensor_output(real_device, map_bounds));
  spec.output(noise_estimate)
      .produces_tensor(pooled_tensor_output(real_device, map_bounds));
  spec.output(cfar_power)
      .produces_tensor(pooled_tensor_output(real_device, map_bounds));
  spec.output(detection_idx)
      .produces_tensor(
          pooled_tensor_output(kIndexDevice, 2U, max_idx_byte_span_));
  constexpr TensorRepresentation kHostInt64Vector{
      .memory_kind = MemoryKind::kHost,
      .dtype = kInt64Dtype,
      .rank = 1U,
  };
  constexpr TensorRepresentation kHostFloat64Vector{
      .memory_kind = MemoryKind::kHost,
      .dtype = kFloat64Dtype,
      .rank = 1U,
  };
  // [live] Exact overrides, Params order (schema and POD alike).
  spec.input(os_rank).optional().queue_depth(2U);
  spec.input(guard_cells).optional().queue_depth(2U);
  spec.input(training_cells).optional().queue_depth(2U);
  spec.input(pfa).optional().queue_depth(2U);
  spec.input(alpha_lut)
      .optional()
      .queue_depth(2U)
      .expects_tensor(tensor_input(kHostFloat64Vector));
  spec.input(threshold_scale).optional().queue_depth(2U);
  spec.input(nci_axes)
      .optional()
      .queue_depth(2U)
      .expects_tensor(tensor_input(kHostInt64Vector));
  spec.input(max_detections).optional().queue_depth(2U);
}

template <typename T>
Contract CFARDetectorOp<T>::contract() const {
  Contract result;
  result.trigger(OnEach{rd_maps});
  add_live_override_to_contract(result, os_rank);
  add_live_override_to_contract(result, guard_cells);
  add_live_override_to_contract(result, training_cells);
  add_live_override_to_contract(result, pfa);
  add_live_override_to_contract(result, alpha_lut);
  add_live_override_to_contract(result, threshold_scale);
  add_live_override_to_contract(result, nci_axes);
  add_live_override_to_contract(result, max_detections);
  return result;
}

template <typename T>
expected<void, Error> CFARDetectorOp<T>::compute(ExecutionContext& context) {
  // Receive.
  HOLOSCAN_TRY(auto input, receive_tensor(rd_maps, kOpName));

  // Resolve this tick's Exact overrides over pinned Params.
  const CFARParams& pinned = context.params<CFARParams>();
  HOLOSCAN_TRY(auto stage,
               resolve_params(
                   os_rank, guard_cells, training_cells, pfa, alpha_lut,
                   threshold_scale, nci_axes, max_detections, pinned,
                   max_leading_, max_alpha_lut_extent_, extents_));

  // Plan input admission, NCI, and output geometry.
  HOLOSCAN_TRY(auto plan,
               build_cfar_stage_plan<T>(
                   std::move(stage), input, extents_, max_leading_));

  // Select the runtime stream.
  cudaStream_t stream = context.cuda_stream().get();
  if (stream == nullptr) {
    return make_unexpected(
        Error{ErrorCode::kNotSupported,
              "CFARDetectorOp: no CUDA compute stream"});
  }

  // Prepare workspace and LUT/table caches, then run the stage.
  detail::CfarDetectorState& state = *state_;
  HOLOSCAN_TRY(auto workspace, prepare_dense_workspace<T>(state, plan));

  HOLOSCAN_TRY(update_alpha_lut<T>(state, plan, stream));
  HOLOSCAN_TRY(auto nci_table_dev_ptr,
               update_nci_table(state, plan.nci, stream));

  // Run the CFAR stage.
  try {
    detail::run_cfar_stage<T>(
        plan.stage, input.data(), plan.input_is_complex, plan.nci.batch_in,
        plan.nci.out_batch, plan.num_doppler, plan.num_range, plan.nci.fold,
        nci_table_dev_ptr, state.alpha_lut_dev->data(),
        static_cast<int64_t>(state.alpha_lut_host_cache.size()),
        workspace.power, workspace.integrated_power, workspace.score,
        workspace.noise, workspace.mask,
        plan.stage.peak_only ? workspace.peak_mask : nullptr, stream);
  } catch (const std::exception& e) {
    return fail_runtime(e.what());
  }

  // Get the detection mask from the workspace.
  const void* det_mask_src =
      plan.stage.peak_only ? workspace.tensors->peak_mask->data()
                           : workspace.tensors->mask->data();
  const bool* det_mask = static_cast<const bool*>(det_mask_src);

  // Every output is optional/latched. Prepare every still-active loan before
  // any emit so a late fill failure cannot publish a partial output set.
  // detection_idx additionally gates its count sync and compaction.
  std::optional<TensorOutputLoan> idx_loan;
  if (detection_idx_out_.active()) {
    HOLOSCAN_TRY(auto loan,
                 prepare_detection_idx(detection_idx, context, plan, det_mask,
                                       state.detection_idx_compact, stream));
    idx_loan.emplace(std::move(loan));
  }

  // Prepare output loans. We do this prior to emitting to ensure that the
  // loans are ready when the emit is called.
  std::optional<TensorOutputLoan> power_loan;
  std::optional<TensorOutputLoan> mask_loan;
  std::optional<TensorOutputLoan> score_loan;
  std::optional<TensorOutputLoan> noise_loan;
  const void* power_src =
      plan.nci.fold > 1 ? workspace.tensors->integrated_power->data()
                        : workspace.tensors->power->data();
  if (cfar_power_out_.active()) {
    HOLOSCAN_TRY(auto loan,
                 prepare_copy_device(cfar_power, context, plan.output_shape,
                                     real_dtype<T>(), power_src, stream));
    power_loan.emplace(std::move(loan));
  }
  if (detection_mask_out_.active()) {
    HOLOSCAN_TRY(auto loan,
                 prepare_copy_device(detection_mask, context, plan.output_shape,
                                     kMaskDtype, det_mask_src, stream));
    mask_loan.emplace(std::move(loan));
  }
  if (detection_score_out_.active()) {
    HOLOSCAN_TRY(auto loan,
                 prepare_copy_device(detection_score, context,
                                     plan.output_shape, real_dtype<T>(),
                                     workspace.tensors->score->data(), stream));
    score_loan.emplace(std::move(loan));
  }
  if (noise_estimate_out_.active()) {
    HOLOSCAN_TRY(auto loan,
                 prepare_copy_device(noise_estimate, context,
                                     plan.output_shape, real_dtype<T>(),
                                     workspace.tensors->noise->data(), stream));
    noise_loan.emplace(std::move(loan));
  }

  // Emit all active output loans.
  if (idx_loan) {
    detection_idx_out_.emit(detection_idx, context, std::move(*idx_loan),
                            "CFARDetectorOp.detection_idx");
  }
  if (power_loan) {
    cfar_power_out_.emit(cfar_power, context, std::move(*power_loan),
                         "CFARDetectorOp.cfar_power");
  }
  if (mask_loan) {
    detection_mask_out_.emit(detection_mask, context, std::move(*mask_loan),
                             "CFARDetectorOp.detection_mask");
  }
  if (score_loan) {
    detection_score_out_.emit(detection_score, context, std::move(*score_loan),
                              "CFARDetectorOp.detection_score");
  }
  if (noise_loan) {
    noise_estimate_out_.emit(noise_estimate, context, std::move(*noise_loan),
                             "CFARDetectorOp.noise_estimate");
  }

  // Retain the input until the next tick has enqueued.
  retained_input_ = std::move(input);
  return {};
}

template class CFARDetectorOp<float>;
template class CFARDetectorOp<double>;

}  // namespace holoscan::radar
