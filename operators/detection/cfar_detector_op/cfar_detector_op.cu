// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "detection/cfar_detector_op/cfar_detector_op.hpp"

#include <cuda_runtime_api.h>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "detection/cfar_detector_op/cfar_alpha.hpp"
#include "detection/cfar_detector_op/cfar_stage.hpp"
#include "detection/cfar_detector_op/nci_plan.hpp"
#include "radar_common/sdk/arg_setters.hpp"
#include "radar_common/sdk/override_param.hpp"
#include "radar_common/tensor/dlpack_buffer.hpp"
#include "radar_common/tensor/tensor_utils.hpp"

namespace holoscan::radar {

namespace {

constexpr DLDataType kBoolDtype{kDLBool, 8, 1};
constexpr DLDataType kInt64Dtype{kDLInt, 64, 1};

std::shared_ptr<holoscan::Tensor> receive_single(
    holoscan::InputContext& op_input, const char* port) {
  auto maybe = op_input.receive<holoscan::TensorMap>(port);
  if (!maybe) {
    throw std::runtime_error(std::string("CFARDetectorOp: no message on '") +
                             port + "'");
  }
  holoscan::TensorMap& map = maybe.value();
  if (map.size() != 1) {
    throw std::invalid_argument(
        std::string("CFARDetectorOp: input port '") + port +
        "' expects a single-entry TensorMap, got " +
        std::to_string(map.size()) + " entries");
  }
  return map.begin()->second;
}

void require_pair(const std::vector<int64_t>& v, const char* name) {
  if (v.size() != 2) {
    throw std::invalid_argument(
        std::string("CFARDetectorOp: '") + name +
        "' must have exactly 2 entries (doppler, range), got " +
        std::to_string(v.size()));
  }
  if (v[0] < 0 || v[1] < 0) {
    throw std::invalid_argument(std::string("CFARDetectorOp: '") + name +
                                "' entries must be >= 0");
  }
}

void emit_single(holoscan::OutputContext& op_output, const char* port,
                 std::shared_ptr<holoscan::Tensor> tensor) {
  holoscan::TensorMap map;
  map.insert({port, std::move(tensor)});
  op_output.emit(map, port);
}

}  // namespace

template <typename T>
CFARDetectorOp<T>::~CFARDetectorOp() = default;

template <typename T>
void CFARDetectorOp<T>::initialize() {
  ensure_radar_arg_setters();
  register_any_argument_setter<CfarVariant>("CfarVariant");
  Operator::initialize();
}

template <typename T>
void CFARDetectorOp<T>::setup(OperatorSpec& spec) {
  spec.input<holoscan::TensorMap>("rd_maps");
  spec.output<holoscan::TensorMap>("detection_mask");
  spec.output<holoscan::TensorMap>("detection_score");
  spec.output<holoscan::TensorMap>("noise_estimate");
  spec.output<holoscan::TensorMap>("detection_idx");

  spec.param(cfar_variant_, "cfar_variant", "CFAR variant",
             "Noise-estimation rule {CA, GO, SO, OS}. Static: selects the "
             "kernel; instantiate separate detectors to vary it.",
             CfarVariant::kCA);
  spec.param(os_rank_, "os_rank", "Order-statistic rank",
             "Rank into the sorted training cells; used only for OS.",
             int64_t{0});
  spec.param(guard_cells_, "guard_cells", "Guard cells (doppler, range)",
             "Guard extent each side of the CUT, excluded from the noise "
             "estimate.",
             std::vector<int64_t>{});
  spec.param(training_cells_, "training_cells",
             "Training cells (doppler, range)",
             "Training/reference extent each side, for the noise estimate.",
             std::vector<int64_t>{});
  spec.param(pfa_, "pfa", "Probability of false alarm",
             "Sets the per-variant threshold multiplier alpha.", 1e-6);
  spec.param(threshold_scale_, "threshold_scale", "Threshold scale",
             "Empirical multiplier on top of the Pfa-derived threshold; "
             "1.0 = pure CFAR.",
             1.0);
  spec.param(nci_axes_, "nci_axes", "Non-coherent integration axes",
             "Leading-axis indices to non-coherently integrate before "
             "thresholding.",
             std::vector<int64_t>{});
  spec.param(max_detections_, "max_detections", "Max detections",
             "Cap on emitted detections (<= 0 → uncapped).", int64_t{0});
  spec.param(max_cached_kernels_, "max_cached_kernels", "Max cached kernels",
             "Distinct window footprints retained before eviction.",
             int64_t{4});
  spec.param(blocking_alpha_lut_rebuild_, "blocking_alpha_lut_rebuild",
             "Blocking alpha LUT rebuild",
             "When false, serve the previous LUT while a new one calibrates "
             "after a [live] config change; when true, block compute() until "
             "the new LUT is ready.",
             false);

  add_override_port<int64_t>(spec, "os_rank");
  add_override_port<std::vector<int64_t>>(spec, "guard_cells");
  add_override_port<std::vector<int64_t>>(spec, "training_cells");
  add_override_port<double>(spec, "pfa");
  add_override_port<double>(spec, "threshold_scale");
  add_override_port<std::vector<int64_t>>(spec, "nci_axes");
  add_override_port<int64_t>(spec, "max_detections");
}

template <typename T>
void CFARDetectorOp<T>::compute(InputContext& op_input,
                                OutputContext& op_output,
                                [[maybe_unused]] ExecutionContext& context) {
  std::shared_ptr<holoscan::Tensor> input = receive_single(op_input, "rd_maps");

  if (!seeded_) {
    effective_guard_cells_ = guard_cells_.get();
    effective_training_cells_ = training_cells_.get();
    effective_pfa_ = pfa_.get();
    effective_threshold_scale_ = threshold_scale_.get();
    effective_os_rank_ = os_rank_.get();
    effective_nci_axes_ = nci_axes_.get();
    effective_max_detections_ = max_detections_.get();
    seeded_ = true;
  }
  effective_guard_cells_ = resolve_override_param<std::vector<int64_t>>(
      op_input, "guard_cells", effective_guard_cells_, nullptr);
  effective_training_cells_ = resolve_override_param<std::vector<int64_t>>(
      op_input, "training_cells", effective_training_cells_, nullptr);
  effective_pfa_ = resolve_override_param<double>(op_input, "pfa",
                                                  effective_pfa_, nullptr);
  effective_threshold_scale_ = resolve_override_param<double>(
      op_input, "threshold_scale", effective_threshold_scale_, nullptr);
  effective_os_rank_ = resolve_override_param<int64_t>(
      op_input, "os_rank", effective_os_rank_, nullptr);
  effective_nci_axes_ = resolve_override_param<std::vector<int64_t>>(
      op_input, "nci_axes", effective_nci_axes_, nullptr);
  effective_max_detections_ = resolve_override_param<int64_t>(
      op_input, "max_detections", effective_max_detections_, nullptr);

  require_pair(effective_guard_cells_, "guard_cells");
  require_pair(effective_training_cells_, "training_cells");

  const InputPrecision precision =
      classify_input_dtype<T>(*input, "rd_maps", /*allow_real=*/true);
  const bool input_is_complex = (precision == InputPrecision::kComplex);
  if (!is_cuda_accessible(*input)) {
    throw std::invalid_argument(
        "CFARDetectorOp: input port 'rd_maps' requires CUDA-accessible "
        "memory");
  }
  if (!input->is_contiguous()) {
    throw std::invalid_argument(
        "CFARDetectorOp: input port 'rd_maps' must be a compact row-major "
        "tensor");
  }

  const detail::NciPlan plan =
      detail::NciPlan::from_shape(input->shape(), effective_nci_axes_,
                                  "rd_maps");
  const int64_t num_doppler = input->shape()[input->shape().size() - 2];
  const int64_t num_range = input->shape()[input->shape().size() - 1];
  const std::vector<int64_t> output_shape =
      plan.output_shape(num_doppler, num_range);

  cudaStream_t stream = op_input.receive_cuda_stream("rd_maps");

  auto score = make_device_tensor(output_shape, real_dtype<T>());
  auto noise = make_device_tensor(output_shape, real_dtype<T>());
  auto mask = make_device_tensor(output_shape, kBoolDtype);
  auto power = make_device_tensor(
      {plan.batch_in, num_doppler, num_range}, real_dtype<T>());
  auto integrated_power = make_device_tensor(
      {plan.out_batch, num_doppler, num_range}, real_dtype<T>());

  const int guard_d = static_cast<int>(effective_guard_cells_[0]);
  const int guard_r = static_cast<int>(effective_guard_cells_[1]);
  const int train_d = static_cast<int>(effective_training_cells_[0]);
  const int train_r = static_cast<int>(effective_training_cells_[1]);

  const CfarVariant variant = cfar_variant_.get();
  if (variant == CfarVariant::kOS) {
    if (effective_os_rank_ < 0) {
      throw std::invalid_argument(
          "CFARDetectorOp: os_rank must be non-negative for OS variant");
    }
    const int64_t max_train = detail::max_training_cell_count(
        static_cast<int>(num_doppler), static_cast<int>(num_range), guard_d,
        guard_r, train_d, train_r);
    if (max_train > detail::kMaxOsTrainingCells) {
      throw std::invalid_argument(
          "CFARDetectorOp: OS window footprint exceeds the supported "
          "training-cell limit");
    }
    if (effective_os_rank_ >= max_train) {
      throw std::invalid_argument(
          "CFARDetectorOp: os_rank must be less than the maximum training-"
          "cell count for the configured window footprint");
    }
  }

  AlphaLutKey lut_key{variant,
                      effective_pfa_,
                      plan.fold,
                      effective_os_rank_,
                      guard_d,
                      guard_r,
                      train_d,
                      train_r,
                      static_cast<int>(num_doppler),
                      static_cast<int>(num_range)};

  auto upload_alpha_lut = [&](const std::vector<double>& host_lut) {
    std::vector<T> lut_t(host_lut.size());
    for (size_t i = 0; i < host_lut.size(); ++i) {
      lut_t[i] = static_cast<T>(host_lut[i]);
    }
    alpha_lut_dev_ = make_device_tensor(
        {static_cast<int64_t>(lut_t.size())}, real_dtype<T>());
    if (cudaMemcpyAsync(alpha_lut_dev_->data(), lut_t.data(),
                        lut_t.size() * sizeof(T), cudaMemcpyHostToDevice,
                        stream) != cudaSuccess) {
      throw std::runtime_error("CFARDetectorOp: alpha LUT H2D copy failed");
    }
  };

  const bool blocking_lut = blocking_alpha_lut_rebuild_.get();
  if (lut_key != alpha_lut_key_) {
    if (blocking_lut) {
      alpha_lut_host_cache_ = detail::get_cached_alpha_lut(lut_key);
      alpha_lut_key_ = lut_key;
      pending_lut_key_ = {};
      pending_lut_active_ = false;
      stale_lut_warned_ = false;
      upload_alpha_lut(alpha_lut_host_cache_);
    } else {
      detail::AlphaLutSnapshot snap = detail::poll_cached_alpha_lut(lut_key);
      if (snap.ready) {
        alpha_lut_host_cache_ = std::move(snap.lut);
        alpha_lut_key_ = lut_key;
        pending_lut_key_ = {};
        pending_lut_active_ = false;
        stale_lut_warned_ = false;
        upload_alpha_lut(alpha_lut_host_cache_);
      } else if (!alpha_lut_host_cache_.empty()) {
        detail::ensure_alpha_lut_build(lut_key);
        pending_lut_key_ = lut_key;
        pending_lut_active_ = true;
        if (!stale_lut_warned_) {
          HOLOSCAN_LOG_WARN(
              "CFARDetectorOp: alpha LUT for the new configuration is still "
              "calibrating; using the previous LUT for this tick");
          stale_lut_warned_ = true;
        }
      } else {
        alpha_lut_host_cache_ = detail::get_cached_alpha_lut(lut_key);
        alpha_lut_key_ = lut_key;
        pending_lut_key_ = {};
        pending_lut_active_ = false;
        stale_lut_warned_ = false;
        upload_alpha_lut(alpha_lut_host_cache_);
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
      upload_alpha_lut(alpha_lut_host_cache_);
    }
  }

  // Upload the NCI lookup table only when the integration layout changes;
  // otherwise reuse the cached device buffer (no per-tick alloc/free/upload).
  const int64_t* nci_table_dev_ptr = nullptr;
  if (plan.fold > 1) {
    if (plan.batch_in_table != nci_table_host_cache_) {
      nci_table_dev_ = make_device_tensor(
          {static_cast<int64_t>(plan.batch_in_table.size())}, kInt64Dtype);
      if (cudaMemcpyAsync(nci_table_dev_->data(), plan.batch_in_table.data(),
                          plan.batch_in_table.size() * sizeof(int64_t),
                          cudaMemcpyHostToDevice, stream) != cudaSuccess) {
        throw std::runtime_error("CFARDetectorOp: NCI table H2D copy failed");
      }
      nci_table_host_cache_ = plan.batch_in_table;
    }
    nci_table_dev_ptr = static_cast<const int64_t*>(nci_table_dev_->data());
  }

  detail::run_cfar_stage<T>(
      variant, input->data(), input_is_complex, plan.batch_in, plan.out_batch,
      num_doppler, num_range, guard_d, guard_r, train_d, train_r,
      effective_os_rank_, effective_threshold_scale_, plan.fold,
      nci_table_dev_ptr, alpha_lut_dev_->data(),
      static_cast<int64_t>(alpha_lut_host_cache_.size()), power->data(),
      integrated_power->data(), score->data(), noise->data(), mask->data(),
      stream);

  const int64_t plane = num_doppler * num_range;
  const int64_t out_elems = plan.out_batch * plane;
  std::vector<uint8_t> host_mask(static_cast<size_t>(out_elems));
  if (cudaMemcpyAsync(host_mask.data(), mask->data(),
                      static_cast<size_t>(out_elems) * sizeof(uint8_t),
                      cudaMemcpyDeviceToHost, stream) != cudaSuccess) {
    throw std::runtime_error("CFARDetectorOp: mask D2H copy failed");
  }
  cudaStreamSynchronize(stream);

  const int64_t idx_cols = 2 + plan.n_preserved;
  std::vector<int64_t> idx_rows;
  idx_rows.reserve(64);
  std::vector<int64_t> preserved_coords;
  const bool capped = effective_max_detections_ > 0;
  int64_t num_detections = 0;
  bool dropped = false;
  for (int64_t b_out = 0; b_out < plan.out_batch; ++b_out) {
    detail::decompose_row_major(b_out, plan.preserved_dims, preserved_coords);
    for (int64_t d = 0; d < num_doppler; ++d) {
      for (int64_t r = 0; r < num_range; ++r) {
        const size_t mask_idx =
            static_cast<size_t>(b_out * plane + d * num_range + r);
        if (host_mask[mask_idx] == 0) {
          continue;
        }
        if (capped && num_detections >= effective_max_detections_) {
          dropped = true;
          break;
        }
        for (int64_t coord : preserved_coords) {
          idx_rows.push_back(coord);
        }
        idx_rows.push_back(d);
        idx_rows.push_back(r);
        ++num_detections;
      }
      if (dropped) {
        break;
      }
    }
    if (dropped) {
      break;
    }
  }
  if (dropped) {
    HOLOSCAN_LOG_WARN(
        "CFARDetectorOp: detections exceeded max_detections ({}); surplus "
        "dropped",
        effective_max_detections_);
  }

  auto detection_idx =
      make_device_tensor({num_detections, idx_cols}, kInt64Dtype);
  if (num_detections > 0 &&
      cudaMemcpy(detection_idx->data(), idx_rows.data(),
                 idx_rows.size() * sizeof(int64_t),
                 cudaMemcpyHostToDevice) != cudaSuccess) {
    throw std::runtime_error("CFARDetectorOp: detection_idx H2D copy failed");
  }

  if (stream != cudaStreamDefault) {
    op_output.set_cuda_stream(stream, "detection_mask");
    op_output.set_cuda_stream(stream, "detection_score");
    op_output.set_cuda_stream(stream, "noise_estimate");
    op_output.set_cuda_stream(stream, "detection_idx");
  }
  emit_single(op_output, "detection_mask", mask);
  emit_single(op_output, "detection_score", score);
  emit_single(op_output, "noise_estimate", noise);
  emit_single(op_output, "detection_idx", detection_idx);

  retained_input_ = std::move(input);
}

template class CFARDetectorOp<float>;
template class CFARDetectorOp<double>;

}  // namespace holoscan::radar
