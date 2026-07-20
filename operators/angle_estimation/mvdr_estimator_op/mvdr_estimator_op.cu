// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "angle_estimation/mvdr_estimator_op/mvdr_estimator_op.hpp"

#include <cuda/std/complex>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "angle_estimation/mvdr_estimator_op/mvdr_stage.hpp"
#include "radar_common/dsp/spatial_smoothing.hpp"
#include "radar_common/dsp/spatial_smoothing_device.hpp"
#include "radar_common/dsp/steering_weights_device.hpp"
#include "radar_common/sdk/arg_setters.hpp"
#include "radar_common/sdk/override_param.hpp"
#include "radar_common/sdk/tensor_io.hpp"
#include "radar_common/tensor/cuda_check.hpp"
#include "radar_common/tensor/dlpack_buffer.hpp"
#include "radar_common/tensor/tensor_utils.hpp"

namespace holoscan::radar {

namespace {

constexpr const char* kOpName = "MVDREstimatorOp";

// Upper bound on max_sources: bounds the per-block shared-memory selected-beam
// list in peak_extract_kernel. Generous for "number of Capon peaks".
constexpr int64_t kMaxSourcesCap = 64;

template <typename T>
std::vector<T> cast_vector(const std::vector<double>& in) {
  std::vector<T> out(in.size());
  for (size_t i = 0; i < in.size(); ++i) {
    out[i] = static_cast<T>(in[i]);
  }
  return out;
}

}  // namespace

template <typename T>
void MVDREstimatorOp<T>::initialize() {
  ensure_radar_arg_setters();
  Operator::initialize();
}

template <typename T>
void MVDREstimatorOp<T>::setup(OperatorSpec& spec) {
  spec.input<holoscan::TensorMap>("snapshot");
  spec.input<holoscan::TensorMap>("is_valid")
      .condition(holoscan::ConditionType::kNone);

  spec.output<holoscan::TensorMap>("aoa_estimates");
  // Optional grid-scan / multi-source / diagnostic outputs. kNone so an
  // unconnected port never gates scheduling (they are always emitted; a
  // consumer that only wants aoa_estimates simply leaves them unwired).
  spec.output<holoscan::TensorMap>("angle_spectrum")
      .condition(holoscan::ConditionType::kNone);
  spec.output<holoscan::TensorMap>("doa_multi")
      .condition(holoscan::ConditionType::kNone);
  spec.output<holoscan::TensorMap>("source_count")
      .condition(holoscan::ConditionType::kNone);
  spec.output<holoscan::TensorMap>("eigenvalues")
      .condition(holoscan::ConditionType::kNone);

  spec.param(array_geometry_, "array_geometry", "Array geometry",
             "Element positions and wavelength", ArrayGeometry{});
  add_override_port<ArrayGeometry>(spec, "array_geometry");

  spec.param(angle_convention_, "angle_convention", "Angle convention",
             "Boundary AER labeling for scan grid and DOA output",
             AngleConvention{});
  spec.param(az_angles_, "az_angles", "Azimuth scan grid",
             "Azimuth grid (or u grid under direction_cosine)",
             std::vector<double>{});
  add_override_port<std::vector<double>>(spec, "az_angles");
  spec.param(el_angles_, "el_angles", "Elevation scan grid",
             "Elevation grid (or v grid under direction_cosine)",
             std::vector<double>{});
  add_override_port<std::vector<double>>(spec, "el_angles");
  spec.param(diagonal_loading_, "diagonal_loading", "Diagonal loading",
             "R + delta*I regularization; negative selects auto", -1.0);
  add_override_port<double>(spec, "diagonal_loading");
  spec.param(max_sources_, "max_sources", "Max sources",
             "Capacity of doa_multi/source_count — number of Capon peaks to "
             "report (aoa_estimates is the strongest)",
             int64_t{1});
  add_override_port<int64_t>(spec, "max_sources");
  spec.param(spatial_smoothing_, "spatial_smoothing", "Spatial smoothing",
             "Per-aperture-axis sub-array counts on a filled ULA/URA "
             "(empty = off; [P] ULA; [P_az, P_el] URA)",
             std::vector<int64_t>{});
  add_override_port<std::vector<int64_t>>(spec, "spatial_smoothing");
  spec.param(forward_backward_, "forward_backward",
             "Forward-backward smoothing",
             "Include conjugate-reversed looks (needs a centro-symmetric "
             "ULA/URA); only applies when spatial_smoothing is set",
             true);
}

template <typename T>
void MVDREstimatorOp<T>::compute(InputContext& op_input,
                                 OutputContext& op_output, ExecutionContext&) {
  if (!seeded_) {
    effective_geometry_ = array_geometry_.get();
    effective_geometry_.validate();
    effective_az_ = az_angles_.get();
    effective_el_ = el_angles_.get();
    effective_loading_ = diagonal_loading_.get();
    effective_max_sources_ = max_sources_.get();
    effective_smoothing_ = spatial_smoothing_.get();
    effective_forward_backward_ = forward_backward_.get();
    seeded_ = true;
  }

  bool geom_changed = false;
  bool grid_changed = false;
  bool loading_changed = false;
  bool sources_changed = false;
  effective_geometry_ = resolve_override_param(
      op_input, "array_geometry", effective_geometry_, &geom_changed);
  effective_geometry_.validate();
  effective_az_ = resolve_override_param(op_input, "az_angles", effective_az_,
                                         &grid_changed);
  effective_el_ = resolve_override_param(op_input, "el_angles", effective_el_,
                                         &grid_changed);
  effective_loading_ = resolve_override_param(
      op_input, "diagonal_loading", effective_loading_, &loading_changed);
  (void)loading_changed;
  effective_max_sources_ = resolve_override_param(
      op_input, "max_sources", effective_max_sources_, &sources_changed);
  (void)sources_changed;
  bool smoothing_changed = false;
  effective_smoothing_ = resolve_override_param(
      op_input, "spatial_smoothing", effective_smoothing_, &smoothing_changed);

  if (effective_az_.empty() || effective_el_.empty()) {
    throw std::invalid_argument(
        "MVDREstimatorOp: az_angles and el_angles must be non-empty");
  }
  if (effective_max_sources_ < 1 ||
      effective_max_sources_ > kMaxSourcesCap) {
    throw std::invalid_argument(
        "MVDREstimatorOp: max_sources must be in [1, 64]");
  }

  auto snapshot = receive_single(op_input, "snapshot", kOpName);
  require_cuda_accessible(*snapshot, "snapshot", kOpName);
  require_contiguous(*snapshot, "snapshot", kOpName);
  require_dtype(*snapshot, complex_dtype<T>(), "snapshot", kOpName);

  const auto& shape = snapshot->shape();
  if (shape.size() != 2 && shape.size() != 3) {
    throw std::invalid_argument(
        "MVDREstimatorOp: snapshot must have shape "
        "(num_items, num_channels) or "
        "(num_items, num_channels, num_snapshots)");
  }
  const int64_t num_items = shape[0];
  const int64_t num_channels = shape[1];
  const int64_t num_snapshots = shape.size() == 3 ? shape[2] : 1;
  if (num_channels != effective_geometry_.num_elements()) {
    throw std::invalid_argument(
        "MVDREstimatorOp: snapshot num_channels must match "
        "array_geometry.num_elements");
  }

  auto is_valid = receive_optional(op_input, "is_valid", kOpName);
  const unsigned char* mask_ptr = nullptr;
  if (is_valid) {
    const DLDataType dt = is_valid->dtype();
    const bool ok_dtype = (dt.code == kDLBool || dt.code == kDLUInt) &&
                          dt.bits == 8 && dt.lanes == 1;
    if (!ok_dtype) {
      throw std::invalid_argument(
          "MVDREstimatorOp: is_valid must have dtype bool or uint8");
    }
    const auto& vshape = is_valid->shape();
    if (vshape.size() != 2 || vshape[0] != num_items ||
        vshape[1] != num_snapshots) {
      throw std::invalid_argument(
          "MVDREstimatorOp: is_valid must have shape "
          "(num_items, num_snapshots) matching snapshot");
    }
    require_cuda_accessible(*is_valid, "is_valid", kOpName);
    require_contiguous(*is_valid, "is_valid", kOpName);
    mask_ptr = static_cast<const unsigned char*>(is_valid->data());
  }

  auto stream = op_input.receive_cuda_stream("snapshot");

  // Spatial-smoothing front-end. When enabled, gather the snapshot into
  // overlapping sub-array looks (forward / forward-backward) and estimate on
  // the reduced sub-array geometry; the covariance/eig/inverse/scan path
  // below is unchanged, just on `est_channels` reduced channels. Off (empty
  // param) → estimate on the full array.
  const ArrayGeometry* manifold_geom = &effective_geometry_;
  int64_t est_channels = num_channels;
  int64_t est_snapshots = num_snapshots;
  const cuda::std::complex<T>* est_snapshot =
      static_cast<const cuda::std::complex<T>*>(snapshot->data());
  const unsigned char* est_mask = mask_ptr;
  std::shared_ptr<holoscan::Tensor> ss_snapshot;  // keep-alive
  std::shared_ptr<holoscan::Tensor> ss_mask_t;    // keep-alive

  if (!effective_smoothing_.empty()) {
    // (Re)build the plan + upload index tables only when the config changes.
    std::ostringstream ks;
    ks << effective_geometry_.wavelength << '|';
    for (double v : effective_geometry_.element_positions) {
      ks << v << ',';
    }
    ks << '|';
    for (int64_t v : effective_smoothing_) {
      ks << v << ',';
    }
    ks << '|' << (effective_forward_backward_ ? 1 : 0);
    const std::string key = ks.str();
    if (key != smoothing_key_) {
      smoothing_plan_ = build_spatial_smoothing_plan(
          effective_geometry_, effective_smoothing_,
          effective_forward_backward_);
      smoothing_key_ = key;
      if (smoothing_plan_.enabled) {
        const int64_t p = smoothing_plan_.num_subarrays;
        const int64_t l = smoothing_plan_.subarray_size;
        d_subarray_index_ =
            make_device_tensor({p * l}, DLDataType{kDLInt, 32, 1});
        if (cudaMemcpyAsync(
                d_subarray_index_->data(),
                smoothing_plan_.subarray_index.data(),
                smoothing_plan_.subarray_index.size() * sizeof(int32_t),
                cudaMemcpyHostToDevice, stream) != cudaSuccess) {
          throw std::runtime_error(
              "MVDREstimatorOp: spatial-smoothing index H2D failed");
        }
        if (smoothing_plan_.forward_backward) {
          d_fb_flip_ = make_device_tensor({l}, DLDataType{kDLInt, 32, 1});
          if (cudaMemcpyAsync(d_fb_flip_->data(),
                              smoothing_plan_.fb_flip.data(),
                              smoothing_plan_.fb_flip.size() * sizeof(int32_t),
                              cudaMemcpyHostToDevice, stream) != cudaSuccess) {
            throw std::runtime_error(
                "MVDREstimatorOp: spatial-smoothing fb-flip H2D failed");
          }
        } else {
          d_fb_flip_.reset();
        }
        HOLOSCAN_LOG_WARN(
            "MVDREstimatorOp: spatial smoothing on — aperture {}->{} channels, "
            "{} sub-arrays{}",
            num_channels, l, p,
            smoothing_plan_.forward_backward ? " (forward-backward)" : "");
      }
    }

    if (smoothing_plan_.enabled) {
      const int64_t l = smoothing_plan_.subarray_size;
      const int64_t m = smoothing_plan_.looks_per_snapshot() * num_snapshots;
      ss_snapshot = make_device_tensor({num_items, l, m}, complex_dtype<T>());
      if (mask_ptr != nullptr) {
        ss_mask_t =
            make_device_tensor({num_items, m}, DLDataType{kDLUInt, 8, 1});
      }
      run_spatial_smoothing_gather<T>(
          static_cast<const cuda::std::complex<T>*>(snapshot->data()), mask_ptr,
          num_items, num_channels, num_snapshots,
          static_cast<const int32_t*>(d_subarray_index_->data()),
          d_fb_flip_ ? static_cast<const int32_t*>(d_fb_flip_->data())
                     : nullptr,
          smoothing_plan_.num_subarrays, l, smoothing_plan_.forward_backward,
          static_cast<cuda::std::complex<T>*>(ss_snapshot->data()),
          ss_mask_t ? static_cast<unsigned char*>(ss_mask_t->data()) : nullptr,
          stream);
      check_cuda("spatial smoothing gather", kOpName);

      manifold_geom = &smoothing_plan_.reduced_geometry;
      est_channels = l;
      est_snapshots = m;
      est_snapshot =
          static_cast<const cuda::std::complex<T>*>(ss_snapshot->data());
      est_mask = ss_mask_t
                     ? static_cast<const unsigned char*>(ss_mask_t->data())
                     : nullptr;
    }
  }

  const bool warn_on_build = geom_changed || grid_changed || smoothing_changed;
  // Untapered manifold (rectangular window): MVDR shapes the beam adaptively.
  // Built against the reduced sub-array geometry when smoothing is on.
  const std::string cache_key =
      steering_cache_key(*manifold_geom, angle_convention_.get(), effective_az_,
                         effective_el_, WindowSpec{});
  const std::shared_ptr<holoscan::Tensor>& manifold =
      steering_cache_.get_or_build(
          cache_key, warn_on_build, "steering manifold", [&] {
            return build_steering_weights_device<T>(
                *manifold_geom, angle_convention_.get(), effective_az_,
                effective_el_, WindowSpec{}, BeamOrder::kElMajor,
                SteeringLayout::kBeamMajor, /*conjugate=*/false, stream);
          });

  const int64_t num_az = static_cast<int64_t>(effective_az_.size());
  const int64_t num_el = static_cast<int64_t>(effective_el_.size());

  const auto az_host = cast_vector<T>(effective_az_);
  const auto el_host = cast_vector<T>(effective_el_);
  auto az_grid = make_device_tensor({num_az}, real_dtype<T>());
  auto el_grid = make_device_tensor({num_el}, real_dtype<T>());
  if (cudaMemcpyAsync(az_grid->data(), az_host.data(),
                      az_host.size() * sizeof(T), cudaMemcpyHostToDevice,
                      stream) != cudaSuccess ||
      cudaMemcpyAsync(el_grid->data(), el_host.data(),
                      el_host.size() * sizeof(T), cudaMemcpyHostToDevice,
                      stream) != cudaSuccess) {
    throw std::runtime_error("MVDREstimatorOp: scan-grid H2D failed");
  }

  auto spectrum =
      make_device_tensor({num_items, num_az, num_el}, real_dtype<T>());
  auto aoa = make_device_tensor({num_items, 2}, real_dtype<T>());
  auto doa_multi = make_device_tensor(
      {num_items, effective_max_sources_, 2}, real_dtype<T>());
  auto source_count =
      make_device_tensor({num_items}, DLDataType{kDLInt, 32, 1});
  // eigenvalues + covariance scratch are sized to the estimation channel count
  // (reduced sub-array size L when smoothing is on, else the full array).
  auto eigenvalues =
      make_device_tensor({num_items, est_channels}, real_dtype<T>());
  auto cov = make_device_tensor({num_items, est_channels, est_channels},
                                complex_dtype<T>());
  auto rinv = make_device_tensor({num_items, est_channels, est_channels},
                                 complex_dtype<T>());
  auto k_counts =
      make_device_tensor({num_items}, DLDataType{kDLInt, 64, 1});

  if (!manifold) {
    throw std::runtime_error(
        "MVDREstimatorOp: internal error: steering manifold missing");
  }

  detail::run_mvdr_stage<T>(
      est_snapshot, est_mask, num_items, est_channels, est_snapshots,
      effective_loading_,
      static_cast<const cuda::std::complex<T>*>(manifold->data()), num_az,
      num_el, static_cast<const T*>(az_grid->data()),
      static_cast<const T*>(el_grid->data()), effective_max_sources_,
      static_cast<cuda::std::complex<T>*>(cov->data()),
      static_cast<cuda::std::complex<T>*>(rinv->data()),
      static_cast<int64_t*>(k_counts->data()),
      static_cast<T*>(spectrum->data()), static_cast<T*>(aoa->data()),
      static_cast<T*>(eigenvalues->data()),
      static_cast<T*>(doa_multi->data()),
      static_cast<int32_t*>(source_count->data()), stream);
  check_cuda("mvdr stage", kOpName);

  if (stream != cudaStreamDefault) {
    op_output.set_cuda_stream(stream, "aoa_estimates");
    op_output.set_cuda_stream(stream, "angle_spectrum");
    op_output.set_cuda_stream(stream, "doa_multi");
    op_output.set_cuda_stream(stream, "source_count");
    op_output.set_cuda_stream(stream, "eigenvalues");
  }
  emit_single(op_output, "aoa_estimates", aoa);
  emit_single(op_output, "angle_spectrum", spectrum);
  emit_single(op_output, "doa_multi", doa_multi);
  emit_single(op_output, "source_count", source_count);
  emit_single(op_output, "eigenvalues", eigenvalues);
}

template class MVDREstimatorOp<float>;
template class MVDREstimatorOp<double>;

}  // namespace holoscan::radar
