// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "fmcw/ddm_sparse_demultiplex_op/ddm_sparse_demultiplex_op.hpp"

#include <cuda_runtime_api.h>

#include <array>
#include <chrono>
#include <cmath>
#include <complex>
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

#include "fmcw/ddm_sparse_demultiplex_op/ddm_sparse_demultiplex_stage.hpp"
#include "radar_common/dsp/mimo_coherent_compensation_stage.hpp"
#include "radar_common/sdk/override_param.hpp"
#include "radar_common/sdk/tensor_io.hpp"
#include "radar_common/tensor/dlpack_buffer.hpp"
#include "radar_common/tensor/max_byte_span.hpp"
#include "radar_common/tensor/tensor_utils.hpp"

namespace holoscan::radar {
namespace {

constexpr const char* kOpName = "DDMSparseDemultiplexOp";
constexpr DLDataType kInt64Dtype{kDLInt, 64, 1};
constexpr DLDataType kInt32Dtype{kDLInt, 32, 1};
constexpr DLDataType kMaskDtype{kDLUInt, 8, 1};

auto fail_invalid(const std::string& message) {
  return make_unexpected(Error{ErrorCode::kInvalidArgument, message});
}

auto fail_runtime(const std::string& message) {
  return make_unexpected(Error{ErrorCode::kFailure, message});
}

std::array<int64_t, 2> normalize_window(const std::vector<int64_t>& value) {
  std::array<int64_t, 2> result{};
  if (value.empty()) {
    result = {0, 0};
  } else if (value.size() == 1U) {
    result = {value[0], value[0]};
  } else if (value.size() == 2U) {
    result = {value[0], value[1]};
  } else {
    throw std::invalid_argument(
        "DDMSparseDemultiplexOp: snapshot_window must have 2 entries");
  }
  if (result[0] < 0 || result[1] < 0) {
    throw std::invalid_argument(
        "DDMSparseDemultiplexOp: snapshot_window entries must be >= 0");
  }
  return result;
}

template <typename T>
std::vector<std::complex<T>> cast_complex(
    const std::vector<std::complex<double>>& input) {
  std::vector<std::complex<T>> result;
  result.reserve(input.size());
  for (const auto& value : input) {
    result.emplace_back(static_cast<T>(value.real()),
                        static_cast<T>(value.imag()));
  }
  return result;
}

expected<void*, Error> writable_ptr(Tensor& tensor, const char* name) {
  auto pointer = tensor.mutable_data();
  if (!pointer) {
    return fail_runtime(std::string(kOpName) + ": " + name +
                        " is not writable");
  }
  return *pointer;
}

expected<void, Error> copy_i64(
    const std::shared_ptr<Tensor>& destination, std::size_t capacity,
    const std::vector<int64_t>& source, const char* name,
    cudaStream_t stream) {
  if (source.size() > capacity) {
    return fail_invalid(std::string(kOpName) + ": " + name +
                        " exceeds constructor capacity");
  }
  if (source.empty()) {
    return {};
  }
  HOLOSCAN_TRY(auto pointer, writable_ptr(*destination, name));
  if (cudaMemcpyAsync(pointer, source.data(), source.size() * sizeof(int64_t),
                      cudaMemcpyHostToDevice, stream) != cudaSuccess) {
    return fail_runtime(std::string(kOpName) + ": " + name +
                        " H2D failed");
  }
  return {};
}

struct SparsePlan {
  detail::DdmRdMapsLayout layout;
  detail::VirtualAssemblyPlan virtual_plan;
  int64_t num_detections{0};
  int64_t width{0};
  int64_t doppler_half{0};
  int64_t range_half{0};
  int64_t num_snapshots{0};
  double wavelength{0.0};
};

expected<SparsePlan, Error> build_plan(
    const DDMSparseDemultiplexParams& params, double wavelength,
    const std::array<int64_t, 2>& window, const Tensor& rd_maps,
    const Tensor& detection_idx, const std::optional<Tensor>& fold_idx,
    const MaxTensorExtents& extents) {
  SparsePlan plan;
  plan.wavelength = wavelength;
  plan.doppler_half = window[0];
  plan.range_half = window[1];
  if (!(wavelength > 0.0) || !std::isfinite(wavelength)) {
    return fail_invalid(
        "DDMSparseDemultiplexOp: wavelength must be finite and > 0");
  }
  try {
    plan.layout = detail::parse_ddm_rd_maps_layout(rd_maps, "rd_maps");
    if (!plan.layout.leading.empty()) {
      return fail_invalid(
          "DDMSparseDemultiplexOp: rd_maps must have rank 3 (N_rx,D,R)");
    }
    (void)detail::ddm_exact_band_indices(
        params.ddm_code, plan.layout.num_doppler);
    plan.width = detail::ddm_num_doppler_per_tx(
        plan.layout.num_doppler, params.ddm_code.num_tx,
        params.ddm_code.guard_bands);
    plan.virtual_plan = detail::build_virtual_plan(
        params.tx_positions, params.rx_positions, params.ddm_code.num_tx,
        plan.layout.num_rx, params.overlap_policy, kOpName);
  } catch (const std::invalid_argument& error) {
    return fail_invalid(error.what());
  }

  const auto& idx_shape = detection_idx.shape();
  if (idx_shape.size() != 2U || idx_shape[1] != 2) {
    return fail_invalid(
        "DDMSparseDemultiplexOp: detection_idx must have shape (N,2)");
  }
  plan.num_detections = idx_shape[0];
  if (fold_idx.has_value()) {
    const auto& shape = fold_idx->shape();
    if (shape.size() != 1U || shape[0] != plan.num_detections) {
      return fail_invalid(
          "DDMSparseDemultiplexOp: fold_idx must have shape (N)");
    }
  }
  const int64_t max_detections = extents.max_detections.has_value()
      ? *extents.max_detections
      : *extents.max_items;
  const int64_t max_virtual = extents.max_virtual_channels.has_value()
      ? *extents.max_virtual_channels
      : (*extents.max_tx) * (*extents.max_rx);
  if (params.ddm_code.num_tx > *extents.max_tx ||
      plan.layout.num_rx > *extents.max_rx ||
      plan.layout.num_doppler > *extents.max_doppler_bins ||
      plan.layout.num_range > *extents.max_range_bins ||
      plan.num_detections > max_detections ||
      plan.virtual_plan.num_virtual > max_virtual) {
    return fail_invalid(
        "DDMSparseDemultiplexOp: input exceeds MaxTensorExtents");
  }
  constexpr int64_t kMaxHalf =
      (std::numeric_limits<int64_t>::max() - 1) / 2;
  if (plan.doppler_half > kMaxHalf || plan.range_half > kMaxHalf) {
    return fail_invalid(
        "DDMSparseDemultiplexOp: snapshot_window is too large");
  }
  const int64_t doppler_len = 2 * plan.doppler_half + 1;
  const int64_t range_len = 2 * plan.range_half + 1;
  if ((params.doppler_edge == SnapshotEdgeMode::kShift &&
       doppler_len > plan.width) ||
      (params.range_edge == SnapshotEdgeMode::kShift &&
       range_len > plan.layout.num_range)) {
    return fail_invalid(
        "DDMSparseDemultiplexOp: shift window exceeds an RD axis");
  }
  if (doppler_len > std::numeric_limits<int64_t>::max() / range_len) {
    return fail_invalid(
        "DDMSparseDemultiplexOp: snapshot_window product is too large");
  }
  plan.num_snapshots = doppler_len * range_len;
  if (plan.num_snapshots > *extents.max_snapshots) {
    return fail_invalid(
        "DDMSparseDemultiplexOp: snapshots exceed MaxTensorExtents");
  }
  return plan;
}

}  // namespace

template <typename T>
DDMSparseDemultiplexOp<T>::DDMSparseDemultiplexOp(
    const MaxTensorExtents& extents) {
  const int64_t max_tx = require_extent(extents.max_tx, "max_tx", kOpName);
  const int64_t max_rx = require_extent(extents.max_rx, "max_rx", kOpName);
  (void)require_extent(
      extents.max_doppler_bins, "max_doppler_bins", kOpName);
  (void)require_extent(extents.max_range_bins, "max_range_bins", kOpName);
  const int64_t max_detections = extents.max_detections.has_value()
      ? require_extent(extents.max_detections, "max_detections", kOpName)
      : require_extent(extents.max_items, "max_items", kOpName);
  const int64_t max_snapshots =
      require_extent(extents.max_snapshots, "max_snapshots", kOpName);
  const int64_t max_virtual = extents.max_virtual_channels.has_value()
      ? require_extent(extents.max_virtual_channels, "max_virtual_channels",
                       kOpName)
      : max_tx * max_rx;
  extents_ = extents;
  snapshot_max_byte_span_ = byte_span_complex_cube<T>(
      {max_detections, max_virtual, max_snapshots}, kOpName);
  is_valid_max_byte_span_ = byte_span_elems(
      product_extents({max_detections, max_snapshots}, kOpName), 1U, kOpName);
  geometry_max_byte_span_ =
      static_cast<std::size_t>((max_virtual + 1) * 3) * sizeof(double);
  src_capacity_ = static_cast<std::size_t>(max_virtual);
  offsets_capacity_ = static_cast<std::size_t>(max_virtual + 1);
  members_capacity_ = static_cast<std::size_t>(max_tx * max_rx);
  src_channel_ = make_device_tensor({max_virtual}, kInt64Dtype);
  member_offsets_ = make_device_tensor({max_virtual + 1}, kInt64Dtype);
  member_indices_ = make_device_tensor({max_tx * max_rx}, kInt64Dtype);
  error_flag_ = make_device_tensor({1}, kInt32Dtype);
}

template <typename T>
void DDMSparseDemultiplexOp<T>::setup(OperatorSpec& spec) {
  constexpr TensorRepresentation kDevice{
      .memory_kind = MemoryKind::kCudaDevice,
      .dtype = std::nullopt,
      .rank = std::nullopt};
  constexpr TensorRepresentation kHostGeometry{
      .memory_kind = MemoryKind::kHost,
      .dtype = DLDataType{kDLFloat, 64U, 1U},
      .rank = 2U};
  spec.input(rd_maps).queue_depth(2U).expects_tensor(tensor_input(kDevice));
  spec.input(detection_idx).queue_depth(2U).expects_tensor(tensor_input(kDevice));
  spec.input(fold_idx)
      .optional()
      .queue_depth(2U)
      .expects_tensor(tensor_input(kDevice));
  spec.input(wavelength_in).optional().queue_depth(2U);
  spec.input(snapshot_window_in).optional().queue_depth(2U);
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
Contract DDMSparseDemultiplexOp<T>::contract() const {
  Contract result;
  result
      .trigger(OnAligned{
          .ports = ports(rd_maps, detection_idx),
          .window = std::chrono::nanoseconds{0}})
      .input(rd_maps, Exact{})
      .input(detection_idx, Exact{})
      .input(fold_idx, Exact{})
      .on_missing(fold_idx, SafeDefault{});
  add_live_override_to_contract(result, wavelength_in);
  add_live_override_to_contract(result, snapshot_window_in);
  return result;
}

template <typename T>
expected<void, Error> DDMSparseDemultiplexOp<T>::ensure_virtual_tables(
    const detail::VirtualAssemblyPlan& plan,
    const std::vector<double>& tx_positions,
    const std::vector<double>& rx_positions, cudaStream_t stream) {
  if (cached_num_tx_ == plan.num_tx && cached_num_rx_ == plan.num_rx &&
      cached_policy_ == plan.policy &&
      cached_tx_positions_ == tx_positions &&
      cached_rx_positions_ == rx_positions) {
    return {};
  }
  HOLOSCAN_TRY(copy_i64(src_channel_, src_capacity_, plan.src_channel,
                        "src_channel", stream));
  HOLOSCAN_TRY(copy_i64(member_offsets_, offsets_capacity_, plan.member_offsets,
                        "member_offsets", stream));
  HOLOSCAN_TRY(copy_i64(member_indices_, members_capacity_, plan.member_indices,
                        "member_indices", stream));
  cached_num_tx_ = plan.num_tx;
  cached_num_rx_ = plan.num_rx;
  cached_policy_ = plan.policy;
  cached_tx_positions_ = tx_positions;
  cached_rx_positions_ = rx_positions;
  return {};
}

template <typename T>
expected<void, Error> DDMSparseDemultiplexOp<T>::compute(
    ExecutionContext& context) {
  HOLOSCAN_TRY(auto rd, receive_tensor(rd_maps, kOpName));
  HOLOSCAN_TRY(auto idx, receive_tensor(detection_idx, kOpName));
  HOLOSCAN_TRY(auto maybe_fold, receive_optional_value(fold_idx));
  try {
    require_complex<T>(rd, "rd_maps");
    require_cuda_accessible(rd, "rd_maps", kOpName);
    require_contiguous(rd, "rd_maps", kOpName);
    require_cuda_accessible(idx, "detection_idx", kOpName);
    require_contiguous(idx, "detection_idx", kOpName);
    require_dtype(idx, kInt64Dtype, "detection_idx", kOpName);
    if (maybe_fold.has_value()) {
      require_cuda_accessible(*maybe_fold, "fold_idx", kOpName);
      require_contiguous(*maybe_fold, "fold_idx", kOpName);
      require_dtype(*maybe_fold, kInt64Dtype, "fold_idx", kOpName);
    }
  } catch (const std::invalid_argument& error) {
    return fail_invalid(error.what());
  }
  const auto& params = context.params<DDMSparseDemultiplexParams>();
  HOLOSCAN_TRY(auto wavelength,
               resolve_live_override(wavelength_in, params.wavelength));
  std::array<int64_t, 2> seed{};
  try {
    seed = normalize_window(params.snapshot_window);
  } catch (const std::invalid_argument& error) {
    return fail_invalid(error.what());
  }
  HOLOSCAN_TRY(auto live_window,
               resolve_live_override(
                   snapshot_window_in,
                   DDMSparseWindowOverride{seed[0], seed[1]}));
  std::array<int64_t, 2> window{};
  try {
    window = normalize_window({live_window.doppler, live_window.range});
  } catch (const std::invalid_argument& error) {
    return fail_invalid(error.what());
  }
  HOLOSCAN_TRY(auto plan,
               build_plan(params, wavelength, window, rd, idx, maybe_fold,
                          extents_));

  cudaStream_t stream = context.cuda_stream().get();
  if (stream == nullptr) {
    return make_unexpected(
        Error{ErrorCode::kNotSupported,
              "DDMSparseDemultiplexOp: no CUDA compute stream"});
  }
  HOLOSCAN_TRY(ensure_virtual_tables(
      plan.virtual_plan, params.tx_positions, params.rx_positions, stream));
  const auto key =
      detail::ddm_demux_cache_key(params.ddm_code, plan.layout.num_doppler);
  const CachedDdmTables& tables =
      ddm_cache_.get_or_build(key, "sparse DDM tables", [&] {
        CachedDdmTables result;
        const auto bands = detail::ddm_exact_band_indices(
            params.ddm_code, plan.layout.num_doppler);
        std::vector<int64_t> band_and_inverse(
            static_cast<size_t>(2 * params.ddm_code.num_tx), -1);
        for (int64_t tx = 0; tx < params.ddm_code.num_tx; ++tx) {
          const int64_t band = bands[static_cast<size_t>(tx)];
          band_and_inverse[static_cast<size_t>(tx)] = band;
          band_and_inverse[static_cast<size_t>(
              params.ddm_code.num_tx + band)] = tx;
        }
        const auto demux = detail::build_ddm_demux_table(
            detail::ddm_resolve_offsets(params.ddm_code),
            params.ddm_code.num_tx, plan.layout.num_doppler, plan.width);
        const auto phase = cast_complex<T>(
            detail::ddm_initial_phase_weights(params.ddm_code));
        result.demux = make_device_tensor(
            {params.ddm_code.num_tx, plan.width}, kInt64Dtype);
        result.tx_by_band =
            make_device_tensor({2 * params.ddm_code.num_tx}, kInt64Dtype);
        result.initial_phase = make_device_tensor(
            {params.ddm_code.num_tx}, complex_dtype<T>());
        auto demux_p = result.demux->mutable_data();
        auto bands_p = result.tx_by_band->mutable_data();
        auto phase_p = result.initial_phase->mutable_data();
        if (!demux_p || !bands_p || !phase_p ||
            cudaMemcpyAsync(*demux_p, demux.data(),
                            demux.size() * sizeof(int64_t),
                            cudaMemcpyHostToDevice, stream) != cudaSuccess ||
            cudaMemcpyAsync(*bands_p, band_and_inverse.data(),
                            band_and_inverse.size() * sizeof(int64_t),
                            cudaMemcpyHostToDevice, stream) != cudaSuccess ||
            cudaMemcpyAsync(*phase_p, phase.data(),
                            phase.size() * sizeof(std::complex<T>),
                            cudaMemcpyHostToDevice, stream) != cudaSuccess) {
          throw std::runtime_error(
              "DDMSparseDemultiplexOp: DDM table H2D failed");
        }
        return result;
      });

  const ArrayGeometry geometry =
      detail::make_array_geometry(plan.virtual_plan, plan.wavelength);
  HOLOSCAN_TRY(auto geometry_prepared,
               prepare_wire_array_geometry(array_geometry, geometry));
  const std::vector<int64_t> snapshot_shape{
      plan.num_detections, plan.virtual_plan.num_virtual, plan.num_snapshots};
  const std::vector<int64_t> valid_shape{
      plan.num_detections, plan.num_snapshots};
  auto snapshot_loan = snapshot.allocate_tensor(TensorLoanRequest{
      .shape = snapshot_shape,
      .dtype = complex_dtype<T>()});
  if (!snapshot_loan) {
    return make_unexpected(std::move(snapshot_loan).error());
  }
  auto valid_loan = is_valid.allocate_tensor(TensorLoanRequest{
      .shape = valid_shape,
      .dtype = kMaskDtype});
  if (!valid_loan) {
    return make_unexpected(std::move(valid_loan).error());
  }
  auto snapshot_writer = snapshot_loan->write(context.cuda_stream());
  auto valid_writer = valid_loan->write(context.cuda_stream());
  if (!snapshot_writer) {
    return make_unexpected(std::move(snapshot_writer).error());
  }
  if (!valid_writer) {
    return make_unexpected(std::move(valid_writer).error());
  }
  if (plan.num_detections > 0) {
    HOLOSCAN_TRY(auto error_pointer, writable_ptr(*error_flag_, "error_flag"));
    if (cudaMemsetAsync(error_pointer, 0, sizeof(int), stream) != cudaSuccess) {
      return fail_runtime("DDMSparseDemultiplexOp: error clear failed");
    }
    detail::run_ddm_sparse_demultiplex_stage<T>(
        rd.data(), static_cast<const int64_t*>(idx.data()),
        maybe_fold.has_value()
            ? static_cast<const int64_t*>(maybe_fold->data())
            : nullptr,
        plan.num_detections, plan.layout, plan.width, params.ddm_code.num_tx,
        plan.num_snapshots, plan.doppler_half, plan.range_half,
        params.doppler_edge, params.range_edge, plan.virtual_plan,
        static_cast<const int64_t*>(tables.demux->data()),
        static_cast<const int64_t*>(tables.tx_by_band->data()),
        tables.initial_phase->data(),
        static_cast<const int64_t*>(src_channel_->data()),
        static_cast<const int64_t*>(member_offsets_->data()),
        static_cast<const int64_t*>(member_indices_->data()),
        snapshot_writer->data(), static_cast<uint8_t*>(valid_writer->data()),
        static_cast<int*>(error_pointer), stream);
    int host_error = 0;
    if (cudaMemcpyAsync(&host_error, error_flag_->data(), sizeof(int),
                        cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
        cudaStreamSynchronize(stream) != cudaSuccess) {
      return fail_runtime(
          "DDMSparseDemultiplexOp: index validation D2H failed");
    }
    if (host_error == detail::kSparseDdmErrFold) {
      return fail_invalid(
          "DDMSparseDemultiplexOp: fold_idx must be in [0,M_tx)");
    }
    if (host_error == detail::kSparseDdmErrCellCoord) {
      return fail_invalid(
          "DDMSparseDemultiplexOp: detection_idx coordinate out of range");
    }
  }
  auto snapshot_committed = std::move(*snapshot_writer).commit();
  if (!snapshot_committed) {
    return make_unexpected(std::move(snapshot_committed).error());
  }
  auto valid_committed = std::move(*valid_writer).commit();
  if (!valid_committed) {
    return make_unexpected(std::move(valid_committed).error());
  }
  HOLOSCAN_TRY(emit_prepared_wire_array_geometry(
      array_geometry, context, std::move(geometry_prepared)));
  (void)emit_prepared_tensor(is_valid, context, std::move(*valid_loan));
  HOLOSCAN_TRY(
      emit_prepared_tensor(snapshot, context, std::move(*snapshot_loan)));
  retained_rd_maps_ = std::move(rd);
  retained_detection_idx_ = std::move(idx);
  if (maybe_fold.has_value()) {
    retained_fold_idx_ = std::move(*maybe_fold);
  }
  return {};
}

template class DDMSparseDemultiplexOp<float>;
template class DDMSparseDemultiplexOp<double>;

}  // namespace holoscan::radar
