// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "fmcw/tdm_sparse_compensation_op/tdm_sparse_compensation_op.hpp"

#include <cuda_runtime_api.h>

#include <array>
#include <chrono>
#include <complex>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/expected_macros.hpp>
#include <holoscan/core/payload_options.hpp>

#include "fmcw/tdm_sparse_compensation_op/tdm_sparse_compensation_stage.hpp"
#include "radar_common/dsp/mimo_coherent_compensation_stage.hpp"
#include "radar_common/sdk/override_param.hpp"
#include "radar_common/sdk/tensor_io.hpp"
#include "radar_common/tensor/dlpack_buffer.hpp"
#include "radar_common/tensor/max_byte_span.hpp"
#include "radar_common/tensor/tensor_utils.hpp"

namespace holoscan::radar {

namespace {

constexpr const char* kOpName = "TDMSparseCompensationOp";
constexpr DLDataType kInt64Dtype{kDLInt, 64, 1};
constexpr DLDataType kInt32Dtype{kDLInt, 32, 1};
constexpr DLDataType kMaskDtype{kDLUInt, 8, 1};

auto fail_invalid(const std::string& message) {
  return make_unexpected(Error{ErrorCode::kInvalidArgument, message});
}

auto fail_runtime(const std::string& message) {
  return make_unexpected(Error{ErrorCode::kFailure, message});
}

std::array<int64_t, 2> normalize_window(
    const std::vector<int64_t>& window) {
  std::array<int64_t, 2> result{};
  if (window.empty()) {
    result = {0, 0};
  } else if (window.size() == 1) {
    result = {window[0], window[0]};
  } else if (window.size() == 2) {
    result = {window[0], window[1]};
  } else {
    throw std::invalid_argument(
        "TDMSparseCompensationOp: snapshot_window must have 2 entries");
  }
  if (result[0] < 0 || result[1] < 0) {
    throw std::invalid_argument(
        "TDMSparseCompensationOp: snapshot_window entries must be >= 0");
  }
  return result;
}

template <typename T>
std::vector<std::complex<T>> cast_correction(
    const std::vector<std::complex<double>>& input) {
  std::vector<std::complex<T>> result;
  result.reserve(input.size());
  for (const auto& value : input) {
    result.emplace_back(static_cast<T>(value.real()),
                        static_cast<T>(value.imag()));
  }
  return result;
}

struct SparseStageConfig {
  FMCWWaveformDescriptor waveform;
  TXSchedule tx_schedule;
  std::vector<double> tx_positions;
  std::vector<double> rx_positions;
  OverlapPolicy overlap_policy{OverlapPolicy::kDedup};
  int64_t doppler_half{0};
  int64_t range_half{0};
  SnapshotEdgeMode doppler_edge{SnapshotEdgeMode::kWrap};
  SnapshotEdgeMode range_edge{SnapshotEdgeMode::kShift};
};

struct SparseStagePlan {
  SparseStageConfig stage;
  detail::TdmRdMapsLayout layout;
  detail::VirtualAssemblyPlan virtual_plan;
  std::vector<int64_t> slot_by_tx;
  int64_t num_detections{0};
  int64_t num_snapshots{0};
};

expected<SparseStageConfig, Error> resolve_params(
    Input<wire::FMCWWaveformT>& waveform_in,
    Input<TDMSparseWindowOverride>& window_in,
    const TDMSparseCompensationParams& pinned) {
  HOLOSCAN_TRY(
      auto waveform,
      resolve_live_override_via(
          waveform_in, pinned.waveform,
          [](const wire::FMCWWaveformT& value) { return from_wire(value); }));
  if (!waveform.is_valid()) {
    return fail_invalid(
        "TDMSparseCompensationOp: waveform must be valid");
  }

  std::array<int64_t, 2> seed_window{};
  try {
    seed_window = normalize_window(pinned.snapshot_window);
  } catch (const std::invalid_argument& error) {
    return fail_invalid(error.what());
  }
  HOLOSCAN_TRY(
      auto live_window,
      resolve_live_override(
          window_in,
          TDMSparseWindowOverride{seed_window[0], seed_window[1]}));
  std::array<int64_t, 2> window{};
  try {
    window = normalize_window({live_window.doppler, live_window.range});
  } catch (const std::invalid_argument& error) {
    return fail_invalid(error.what());
  }

  return SparseStageConfig{
      .waveform = std::move(waveform),
      .tx_schedule = pinned.tx_schedule,
      .tx_positions = pinned.tx_positions,
      .rx_positions = pinned.rx_positions,
      .overlap_policy = pinned.overlap_policy,
      .doppler_half = window[0],
      .range_half = window[1],
      .doppler_edge = pinned.doppler_edge,
      .range_edge = pinned.range_edge};
}

expected<SparseStagePlan, Error> build_stage_plan(
    SparseStageConfig stage, const Tensor& rd_maps,
    const Tensor& detection_idx, const MaxTensorExtents& extents) {
  SparseStagePlan plan;
  plan.stage = std::move(stage);
  try {
    plan.layout = detail::parse_tdm_rd_maps_layout(rd_maps, "rd_maps");
    if (!plan.layout.leading.empty()) {
      return fail_invalid(
          "TDMSparseCompensationOp: rd_maps must have rank 4 "
          "(M_tx,N_rx,D,R)");
    }
    plan.virtual_plan = detail::build_virtual_plan(
        plan.stage.tx_positions, plan.stage.rx_positions,
        plan.layout.num_tx, plan.layout.num_rx,
        plan.stage.overlap_policy, kOpName);
    plan.slot_by_tx = detail::tdm_firing_slots(
        plan.stage.tx_schedule, plan.layout.num_tx);
  } catch (const std::invalid_argument& error) {
    return fail_invalid(error.what());
  }

  const auto& idx_shape = detection_idx.shape();
  if (idx_shape.size() != 2U || idx_shape[1] != 2) {
    return fail_invalid(
        "TDMSparseCompensationOp: detection_idx must have shape "
        "(num_detections, 2)");
  }
  plan.num_detections = idx_shape[0];
  if (plan.num_detections < 0) {
    return fail_invalid(
        "TDMSparseCompensationOp: negative detection count");
  }

  const int64_t max_detections =
      extents.max_detections.has_value()
          ? *extents.max_detections
          : *extents.max_items;
  const int64_t max_virtual =
      extents.max_virtual_channels.has_value()
          ? *extents.max_virtual_channels
          : (*extents.max_tx) * (*extents.max_rx);
  if (plan.layout.num_tx > *extents.max_tx ||
      plan.layout.num_rx > *extents.max_rx ||
      plan.layout.num_doppler > *extents.max_doppler_bins ||
      plan.layout.num_range > *extents.max_range_bins ||
      plan.num_detections > max_detections ||
      plan.virtual_plan.num_virtual > max_virtual) {
    return fail_invalid(
        "TDMSparseCompensationOp: input exceeds MaxTensorExtents");
  }

  constexpr int64_t kMaxHalf =
      (std::numeric_limits<int64_t>::max() - 1) / 2;
  if (plan.stage.doppler_half > kMaxHalf ||
      plan.stage.range_half > kMaxHalf) {
    return fail_invalid(
        "TDMSparseCompensationOp: snapshot_window is too large");
  }
  const int64_t doppler_len = 2 * plan.stage.doppler_half + 1;
  const int64_t range_len = 2 * plan.stage.range_half + 1;
  if ((plan.stage.doppler_edge == SnapshotEdgeMode::kShift &&
       doppler_len > plan.layout.num_doppler) ||
      (plan.stage.range_edge == SnapshotEdgeMode::kShift &&
       range_len > plan.layout.num_range)) {
    return fail_invalid(
        "TDMSparseCompensationOp: shift window exceeds an RD axis");
  }
  if (doppler_len >
      std::numeric_limits<int64_t>::max() / range_len) {
    return fail_invalid(
        "TDMSparseCompensationOp: snapshot_window product is too large");
  }
  plan.num_snapshots = doppler_len * range_len;
  if (plan.num_snapshots > *extents.max_snapshots) {
    return fail_invalid(
        "TDMSparseCompensationOp: snapshot count exceeds "
        "MaxTensorExtents.max_snapshots");
  }
  return plan;
}

expected<void*, Error> writable_ptr(Tensor& tensor, const char* name) {
  auto result = tensor.mutable_data();
  if (!result) {
    return fail_runtime(
        std::string(kOpName) + ": " + name + " is not writable");
  }
  return *result;
}

expected<void, Error> copy_table(
    const std::shared_ptr<Tensor>& destination, std::size_t capacity,
    const std::vector<int64_t>& source, const char* name,
    cudaStream_t stream) {
  if (source.size() > capacity) {
    return fail_invalid(
        std::string(kOpName) + ": " + name + " exceeds ctor capacity");
  }
  if (source.empty()) {
    return {};
  }
  HOLOSCAN_TRY(auto pointer, writable_ptr(*destination, name));
  if (cudaMemcpyAsync(pointer, source.data(),
                      source.size() * sizeof(int64_t),
                      cudaMemcpyHostToDevice, stream) != cudaSuccess) {
    return fail_runtime(
        std::string(kOpName) + ": " + name + " H2D failed");
  }
  return {};
}

}  // namespace

template <typename T>
TDMSparseCompensationOp<T>::TDMSparseCompensationOp(
    const MaxTensorExtents& extents) {
  const int64_t max_tx =
      require_extent(extents.max_tx, "max_tx", kOpName);
  const int64_t max_rx =
      require_extent(extents.max_rx, "max_rx", kOpName);
  (void)require_extent(
      extents.max_doppler_bins, "max_doppler_bins", kOpName);
  (void)require_extent(
      extents.max_range_bins, "max_range_bins", kOpName);
  const int64_t max_detections =
      extents.max_detections.has_value()
          ? require_extent(
                extents.max_detections, "max_detections", kOpName)
          : require_extent(extents.max_items, "max_items", kOpName);
  const int64_t max_snapshots =
      require_extent(extents.max_snapshots, "max_snapshots", kOpName);
  const int64_t max_virtual =
      extents.max_virtual_channels.has_value()
          ? require_extent(
                extents.max_virtual_channels,
                "max_virtual_channels", kOpName)
          : max_tx * max_rx;
  extents_ = extents;
  snapshot_max_byte_span_ = byte_span_complex_cube<T>(
      {max_detections, max_virtual, max_snapshots}, kOpName);
  is_valid_max_byte_span_ = byte_span_elems(
      product_extents({max_detections, max_snapshots}, kOpName),
      1U, kOpName);
  geometry_max_byte_span_ =
      static_cast<std::size_t>((max_virtual + 1) * 3) * sizeof(double);

  src_capacity_ = static_cast<std::size_t>(max_virtual);
  offsets_capacity_ = static_cast<std::size_t>(max_virtual + 1);
  members_capacity_ =
      static_cast<std::size_t>(max_tx * max_rx);
  src_channel_ = make_device_tensor(
      {max_virtual}, kInt64Dtype);
  member_offsets_ = make_device_tensor(
      {max_virtual + 1}, kInt64Dtype);
  member_indices_ = make_device_tensor(
      {max_tx * max_rx}, kInt64Dtype);
  error_flag_ = make_device_tensor({1}, kInt32Dtype);
}

template <typename T>
void TDMSparseCompensationOp<T>::setup(OperatorSpec& spec) {
  constexpr TensorRepresentation kDevice{
      .memory_kind = MemoryKind::kCudaDevice,
      .dtype = std::nullopt,
      .rank = std::nullopt};
  constexpr TensorRepresentation kHostGeometry{
      .memory_kind = MemoryKind::kHost,
      .dtype = DLDataType{kDLFloat, 64U, 1U},
      .rank = 2U};
  spec.input(rd_maps)
      .queue_depth(2U)
      .expects_tensor(tensor_input(kDevice));
  spec.input(detection_idx)
      .queue_depth(2U)
      .expects_tensor(tensor_input(kDevice));
  spec.input(waveform_in).queue_depth(2U);
  spec.input(snapshot_window_in).queue_depth(2U);
  spec.output(snapshot).produces_tensor(
      pooled_tensor_output(kDevice, 3U, snapshot_max_byte_span_));
  spec.output(is_valid).produces_tensor(
      pooled_tensor_output(kDevice, 2U, is_valid_max_byte_span_));
  spec.output(array_geometry)
      .max_emits_per_compute(1U)
      .produces_tensor(pooled_tensor_output(
          kHostGeometry, 2U, geometry_max_byte_span_));
}

template <typename T>
Contract TDMSparseCompensationOp<T>::contract() const {
  Contract result;
  result
      .trigger(OnAligned{
          .ports = ports(rd_maps, detection_idx),
          .window = std::chrono::nanoseconds{0}})
      .input(rd_maps, Exact{})
      .input(detection_idx, Exact{});
  add_live_override_to_contract(result, waveform_in);
  add_live_override_to_contract(result, snapshot_window_in);
  return result;
}

template <typename T>
expected<void, Error>
TDMSparseCompensationOp<T>::ensure_virtual_tables(
    const detail::VirtualAssemblyPlan& plan,
    const std::vector<double>& tx_positions,
    const std::vector<double>& rx_positions, cudaStream_t stream) {
  if (cached_num_tx_ == plan.num_tx &&
      cached_num_rx_ == plan.num_rx &&
      cached_policy_ == plan.policy &&
      cached_tx_positions_ == tx_positions &&
      cached_rx_positions_ == rx_positions) {
    return {};
  }
  HOLOSCAN_TRY(copy_table(
      src_channel_, src_capacity_, plan.src_channel,
      "src_channel", stream));
  HOLOSCAN_TRY(copy_table(
      member_offsets_, offsets_capacity_, plan.member_offsets,
      "member_offsets", stream));
  HOLOSCAN_TRY(copy_table(
      member_indices_, members_capacity_, plan.member_indices,
      "member_indices", stream));
  cached_num_tx_ = plan.num_tx;
  cached_num_rx_ = plan.num_rx;
  cached_policy_ = plan.policy;
  cached_tx_positions_ = tx_positions;
  cached_rx_positions_ = rx_positions;
  return {};
}

template <typename T>
expected<void, Error> TDMSparseCompensationOp<T>::compute(
    ExecutionContext& context) {
  HOLOSCAN_TRY(auto rd, receive_tensor(rd_maps, kOpName));
  HOLOSCAN_TRY(auto idx, receive_tensor(detection_idx, kOpName));
  try {
    require_complex<T>(rd, "rd_maps");
    require_cuda_accessible(rd, "rd_maps", kOpName);
    require_contiguous(rd, "rd_maps", kOpName);
    require_cuda_accessible(idx, "detection_idx", kOpName);
    require_contiguous(idx, "detection_idx", kOpName);
    require_dtype(idx, kInt64Dtype, "detection_idx", kOpName);
  } catch (const std::invalid_argument& error) {
    return fail_invalid(error.what());
  }

  const TDMSparseCompensationParams& pinned =
      context.params<TDMSparseCompensationParams>();
  HOLOSCAN_TRY(
      auto stage,
      resolve_params(
          waveform_in, snapshot_window_in, pinned));
  HOLOSCAN_TRY(
      auto plan,
      build_stage_plan(
          std::move(stage), rd, idx, extents_));

  cudaStream_t stream = context.cuda_stream().get();
  if (stream == nullptr) {
    return make_unexpected(
        Error{ErrorCode::kNotSupported,
              "TDMSparseCompensationOp: no CUDA compute stream"});
  }
  HOLOSCAN_TRY(ensure_virtual_tables(
      plan.virtual_plan, plan.stage.tx_positions,
      plan.stage.rx_positions, stream));

  const std::string correction_key =
      detail::tdm_correction_cache_key(
          plan.stage.waveform, plan.layout.num_tx,
          plan.layout.num_doppler, plan.slot_by_tx);
  const CachedCorrection& correction = correction_cache_.get_or_build(
      correction_key, "sparse TDM correction table", [&] {
        CachedCorrection result;
        const auto host = detail::build_tdm_correction_table(
            plan.stage.waveform, plan.layout.num_tx,
            plan.layout.num_doppler, plan.slot_by_tx);
        const auto typed = cast_correction<T>(host);
        result.device_table = make_device_tensor(
            {plan.layout.num_tx, plan.layout.num_doppler},
            complex_dtype<T>());
        auto pointer = result.device_table->mutable_data();
        if (!pointer) {
          throw std::runtime_error(
              "TDMSparseCompensationOp: correction table not writable");
        }
        if (cudaMemcpyAsync(
                *pointer, typed.data(),
                typed.size() * sizeof(std::complex<T>),
                cudaMemcpyHostToDevice, stream) != cudaSuccess) {
          throw std::runtime_error(
              "TDMSparseCompensationOp: correction-table H2D failed");
        }
        return result;
      });

  const std::vector<int64_t> snapshot_shape{
      plan.num_detections, plan.virtual_plan.num_virtual,
      plan.num_snapshots};
  const std::vector<int64_t> valid_shape{
      plan.num_detections, plan.num_snapshots};
  const ArrayGeometry geometry = detail::make_array_geometry(
      plan.virtual_plan, plan.stage.waveform.wavelength());
  HOLOSCAN_TRY(
      auto geometry_prepared,
      prepare_wire_array_geometry(array_geometry, geometry));

  auto snapshot_loan = snapshot.allocate_tensor(
      TensorLoanRequest{
          .shape = snapshot_shape,
          .dtype = complex_dtype<T>()});
  if (!snapshot_loan) {
    return make_unexpected(std::move(snapshot_loan).error());
  }
  auto valid_loan = is_valid.allocate_tensor(
      TensorLoanRequest{
          .shape = valid_shape,
          .dtype = kMaskDtype});
  if (!valid_loan) {
    return make_unexpected(std::move(valid_loan).error());
  }
  auto snapshot_writer =
      snapshot_loan->write(context.cuda_stream());
  if (!snapshot_writer) {
    return make_unexpected(std::move(snapshot_writer).error());
  }
  auto valid_writer =
      valid_loan->write(context.cuda_stream());
  if (!valid_writer) {
    return make_unexpected(std::move(valid_writer).error());
  }

  if (plan.num_detections > 0) {
    HOLOSCAN_TRY(auto error_pointer,
                 writable_ptr(*error_flag_, "error_flag"));
    if (cudaMemsetAsync(
            error_pointer, 0, sizeof(int), stream) != cudaSuccess) {
      return fail_runtime(
          "TDMSparseCompensationOp: error-flag clear failed");
    }
    try {
      detail::run_tdm_sparse_compensation_stage<T>(
          rd.data(), static_cast<const int64_t*>(idx.data()),
          plan.num_detections, plan.layout, plan.num_snapshots,
          plan.stage.doppler_half, plan.stage.range_half,
          plan.stage.doppler_edge, plan.stage.range_edge,
          plan.virtual_plan, correction.device_table->data(),
          static_cast<const int64_t*>(src_channel_->data()),
          static_cast<const int64_t*>(member_offsets_->data()),
          static_cast<const int64_t*>(member_indices_->data()),
          snapshot_writer->data(),
          static_cast<uint8_t*>(valid_writer->data()),
          static_cast<int*>(error_pointer), stream);
    } catch (const std::runtime_error& error) {
      return fail_runtime(error.what());
    }
    int host_error = 0;
    if (cudaMemcpyAsync(
            &host_error, error_flag_->data(), sizeof(int),
            cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
        cudaStreamSynchronize(stream) != cudaSuccess) {
      return fail_runtime(
          "TDMSparseCompensationOp: index validation D2H failed");
    }
    if (host_error == detail::kSparseTdmErrCellCoord) {
      return fail_invalid(
          "TDMSparseCompensationOp: detection_idx coordinate "
          "out of range");
    }
  }

  auto snapshot_committed =
      std::move(*snapshot_writer).commit();
  if (!snapshot_committed) {
    return make_unexpected(
        std::move(snapshot_committed).error());
  }
  auto valid_committed = std::move(*valid_writer).commit();
  if (!valid_committed) {
    return make_unexpected(
        std::move(valid_committed).error());
  }

  HOLOSCAN_TRY(emit_prepared_wire_array_geometry(
      array_geometry, context, std::move(geometry_prepared)));
  (void)emit_prepared_tensor(
      is_valid, context, std::move(*valid_loan));
  HOLOSCAN_TRY(emit_prepared_tensor(
      snapshot, context, std::move(*snapshot_loan)));

  retained_rd_maps_ = std::move(rd);
  retained_detection_idx_ = std::move(idx);
  return {};
}

template class TDMSparseCompensationOp<float>;
template class TDMSparseCompensationOp<double>;

}  // namespace holoscan::radar
