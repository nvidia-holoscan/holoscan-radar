// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
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

#include "detection/cfar_detector_op/cfar_alpha.hpp"
#include "detection/cfar_detector_op/cfar_compact.cuh"
#include "detection/cfar_detector_op/cfar_stage.hpp"
#include "detection/cfar_detector_op/nci_plan.hpp"
#include "radar_common/sdk/arg_setters.hpp"
#include "radar_common/sdk/override_param.hpp"
#include "radar_common/tensor/dlpack_buffer.hpp"
#include "radar_common/tensor/tensor_utils.hpp"

// YAML::convert for the operator-local CfarVariant enum, mirroring the shared
// converters in radar_common/sdk/yaml_converters.hpp (reusing
// detail::parse_enum) so ArgumentSetter::ensure_type<CfarVariant>() supports
// from_config(). Kept in this TU because CfarVariant is cfar-local; common
// must not depend on it.
namespace YAML {
template <>
struct convert<holoscan::radar::CfarVariant> {
  static Node encode(const holoscan::radar::CfarVariant& value) {
    return Node(std::string(holoscan::radar::to_string(value)));
  }
  static bool decode(const Node& node, holoscan::radar::CfarVariant& value) {
    if (!node.IsScalar()) {
      return false;
    }
    static constexpr holoscan::radar::CfarVariant kAll[] = {
        holoscan::radar::CfarVariant::kCA, holoscan::radar::CfarVariant::kGO,
        holoscan::radar::CfarVariant::kSO, holoscan::radar::CfarVariant::kOS};
    value = holoscan::radar::detail::parse_enum(node.as<std::string>(),
                                                "CfarVariant", kAll);
    return true;
  }
};
}  // namespace YAML

namespace holoscan::radar {

namespace {

constexpr DLDataType kBoolDtype{kDLBool, 8, 1};
constexpr DLDataType kInt64Dtype{kDLInt, 64, 1};

// GO/SO always calibrate alpha from Pfa via Monte Carlo; OS uses MC when K>1
// (closed-form when K=1). Warn once so callers know to supply alpha_lut for
// low-latency live updates.
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

void validate_external_alpha_lut(CfarVariant variant,
                                 const std::vector<double>& lut,
                                 int64_t max_lut_index, int num_doppler,
                                 int num_range, int guard_d, int guard_r,
                                 int train_d, int train_r) {
  if (max_lut_index <= 0) {
    return;
  }
  if (lut.size() <= static_cast<size_t>(max_lut_index)) {
    throw std::invalid_argument(
        "CFARDetectorOp: alpha_lut length must exceed the maximum LUT index "
        "for the configured variant, footprint, and map size");
  }
  // Every slot in 1..max must be finite and non-negative. Zeros are only legal
  // for GO/SO unreachable indices (sparse offline tables); CA/OS use every
  // index in range as a possible training-cell count.
  for (int64_t idx = 1; idx <= max_lut_index; ++idx) {
    const double alpha = lut[static_cast<size_t>(idx)];
    if (!std::isfinite(alpha) || alpha < 0.0) {
      throw std::invalid_argument(
          "CFARDetectorOp: alpha_lut entries for used indices must be finite "
          "and non-negative");
    }
  }
  if (variant == CfarVariant::kGO || variant == CfarVariant::kSO) {
    const std::vector<int64_t> reachable = detail::reachable_go_lut_indices(
        num_doppler, num_range, guard_d, guard_r, train_d, train_r);
    for (int64_t idx : reachable) {
      if (idx <= 0 || idx > max_lut_index) {
        continue;
      }
      if (lut[static_cast<size_t>(idx)] <= 0.0) {
        throw std::invalid_argument(
            "CFARDetectorOp: alpha_lut entries for reachable GO/SO LUT "
            "indices must be finite and positive");
      }
    }
    return;
  }
  for (int64_t idx = 1; idx <= max_lut_index; ++idx) {
    if (lut[static_cast<size_t>(idx)] <= 0.0) {
      throw std::invalid_argument(
          "CFARDetectorOp: alpha_lut entries for used indices must be finite "
          "and positive");
    }
  }
}

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
  ArgumentSetter::ensure_type<CfarVariant>();
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
             "Sets the per-variant threshold multiplier alpha when alpha_lut "
             "is unset.",
             1e-6);
  spec.param(alpha_lut_, "alpha_lut", "Precomputed alpha LUT",
             "Optional threshold multiplier table; when non-empty, used "
             "directly and pfa is ignored for alpha calibration.",
             std::vector<double>{});
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
  add_override_port<std::vector<double>>(spec, "alpha_lut");
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
    effective_alpha_lut_ = alpha_lut_.get();
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
  bool pfa_override_changed = false;
  effective_pfa_ = resolve_override_param<double>(op_input, "pfa",
                                                  effective_pfa_,
                                                  &pfa_override_changed);
  effective_alpha_lut_ = resolve_override_param<std::vector<double>>(
      op_input, "alpha_lut", effective_alpha_lut_, nullptr);
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

  // Reuse a ping-ponged dense workspace when the CFAR geometry is unchanged.
  dense_workspace_idx_ ^= 1;
  DenseWorkspace& ws = dense_workspace_[dense_workspace_idx_];
  const bool ws_hit = ws.score && ws.batch_in == plan.batch_in &&
                      ws.out_batch == plan.out_batch &&
                      ws.num_doppler == num_doppler &&
                      ws.num_range == num_range;
  if (!ws_hit) {
    ws.score = make_device_tensor(output_shape, real_dtype<T>());
    ws.noise = make_device_tensor(output_shape, real_dtype<T>());
    ws.mask = make_device_tensor(output_shape, kBoolDtype);
    ws.power = make_device_tensor({plan.batch_in, num_doppler, num_range},
                                 real_dtype<T>());
    ws.integrated_power = make_device_tensor(
        {plan.out_batch, num_doppler, num_range}, real_dtype<T>());
    ws.batch_in = plan.batch_in;
    ws.out_batch = plan.out_batch;
    ws.num_doppler = num_doppler;
    ws.num_range = num_range;
  }
  auto& score = ws.score;
  auto& noise = ws.noise;
  auto& mask = ws.mask;
  auto& power = ws.power;
  auto& integrated_power = ws.integrated_power;

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

  const bool had_external_alpha_lut = using_external_alpha_lut_;
  const bool has_external_alpha_lut = !effective_alpha_lut_.empty();

  if (had_external_alpha_lut && !has_external_alpha_lut) {
    if (!alpha_lut_cleared_pfa_active_warned_) {
      HOLOSCAN_LOG_WARN(
          "CFARDetectorOp: alpha_lut cleared; pfa-derived alpha calibration is "
          "now active and may use the slow Monte Carlo path for GO, SO, or OS "
          "with K > 1");
      alpha_lut_cleared_pfa_active_warned_ = true;
    }
    using_external_alpha_lut_ = false;
    // Drop the external table from the host cache so the non-blocking stale
    // path cannot keep applying it while pfa-derived calibration runs.
    alpha_lut_host_cache_.clear();
    alpha_lut_dev_.reset();
    alpha_lut_key_ = {};
    pending_lut_key_ = {};
    pending_lut_active_ = false;
    stale_lut_warned_ = false;
  }

  if (pfa_override_changed && has_external_alpha_lut &&
      !pfa_ignored_while_alpha_lut_warned_) {
    HOLOSCAN_LOG_WARN(
        "CFARDetectorOp: pfa override has no effect while alpha_lut is "
        "supplied");
    pfa_ignored_while_alpha_lut_warned_ = true;
  }

  if (has_external_alpha_lut) {
    // max_alpha_lut_index is closed-form for maps that fit the window (the
    // common case); only needed to validate an externally supplied LUT.
    const int64_t max_lut_index = max_alpha_lut_index(
        variant, static_cast<int>(num_doppler), static_cast<int>(num_range),
        guard_d, guard_r, train_d, train_r);
    validate_external_alpha_lut(variant, effective_alpha_lut_, max_lut_index,
                                static_cast<int>(num_doppler),
                                static_cast<int>(num_range), guard_d, guard_r,
                                train_d, train_r);
    using_external_alpha_lut_ = true;
    if (effective_alpha_lut_ != alpha_lut_host_cache_) {
      alpha_lut_host_cache_ = effective_alpha_lut_;
      upload_alpha_lut(alpha_lut_host_cache_);
    }
  } else {
    using_external_alpha_lut_ = false;
    const bool blocking_lut = blocking_alpha_lut_rebuild_.get();
    if (lut_key != alpha_lut_key_) {
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
      if (blocking_lut) {
        warn_slow_pfa_lut();
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
        } else {
          warn_slow_pfa_lut();
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
  const int64_t idx_cols = 2 + plan.n_preserved;

  if (!detection_idx_compact_) {
    detection_idx_compact_ =
        std::make_shared<detail::DetectionIdxCompactState>();
  }
  const int64_t total_detections = detail::count_mask_detections(
      static_cast<const bool*>(mask->data()), out_elems,
      detection_idx_compact_.get(), stream);

  const bool capped = effective_max_detections_ > 0;
  const int64_t num_detections =
      capped ? std::min(total_detections, effective_max_detections_)
             : total_detections;
  if (capped && total_detections > effective_max_detections_) {
    HOLOSCAN_LOG_WARN(
        "CFARDetectorOp: detections exceeded max_detections ({}); surplus "
        "dropped",
        effective_max_detections_);
  }

  auto detection_idx =
      make_device_tensor({num_detections, idx_cols}, kInt64Dtype);
  detail::compact_detection_idx(
      static_cast<const bool*>(mask->data()), out_elems, num_doppler,
      num_range, plan.preserved_dims, num_detections,
      static_cast<int64_t*>(detection_idx->data()),
      detection_idx_compact_.get(), stream);

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
