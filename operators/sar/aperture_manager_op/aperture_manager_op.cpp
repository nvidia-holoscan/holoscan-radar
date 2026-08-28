// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "sar/aperture_manager_op/aperture_manager_op.hpp"

#include <cuda_runtime_api.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "radar_common/dsp/window.hpp"
#include "radar_common/sdk/arg_setters.hpp"
#include "radar_common/sdk/override_param.hpp"
#include "radar_common/sdk/port_connected.hpp"
#include "radar_common/sdk/stream_boundary.hpp"
#include "radar_common/sdk/tensor_io.hpp"
#include "radar_common/tensor/dlpack_buffer.hpp"
#include "radar_common/tensor/tensor_utils.hpp"

namespace holoscan::radar {

namespace {

template <typename T>
constexpr const char* precision_name() {
  return std::is_same_v<T, float> ? "float" : "double";
}

template <typename T>
const std::string& op_name() {
  static const std::string name =
      std::string("ApertureManagerOp<") + precision_name<T>() + ">";
  return name;
}

bool is_resetting(FormationControl control) {
  return control == FormationControl::kEmitAndReset ||
         control == FormationControl::kResetWithoutEmit ||
         control == FormationControl::kResetThenEmit;
}

bool is_pre_reset(FormationControl control) {
  return control == FormationControl::kResetWithoutEmit ||
         control == FormationControl::kResetThenEmit;
}

uint8_t read_uint8(const holoscan::Tensor& tensor, const std::string& who,
                   const char* port, cudaStream_t stream) {
  if (tensor.size() != 1) {
    throw std::invalid_argument(who + ": " + port +
                                " must have shape (1,)");
  }
  uint8_t value = 0;
  if (cudaMemcpyAsync(&value, tensor.data(), sizeof(value),
                      cudaMemcpyDeviceToHost, stream) != cudaSuccess) {
    throw std::runtime_error(who + ": failed to read " + port);
  }
  if (cudaStreamSynchronize(stream) != cudaSuccess) {
    throw std::runtime_error(who + ": failed to sync " + port);
  }
  return value;
}

uint32_t read_uint32(const holoscan::Tensor& tensor, const std::string& who,
                     const char* port, cudaStream_t stream) {
  if (tensor.size() != 1) {
    throw std::invalid_argument(who + ": " + port +
                                " must have shape (1,)");
  }
  uint32_t value = 0;
  if (cudaMemcpyAsync(&value, tensor.data(), sizeof(value),
                      cudaMemcpyDeviceToHost, stream) != cudaSuccess) {
    throw std::runtime_error(who + ": failed to read " + port);
  }
  if (cudaStreamSynchronize(stream) != cudaSuccess) {
    throw std::runtime_error(who + ": failed to sync " + port);
  }
  return value;
}

FormationControl read_control(const holoscan::Tensor& tensor,
                              const std::string& who,
                              cudaStream_t stream) {
  const uint8_t value = read_uint8(tensor, who, "control", stream);
  if (value > static_cast<uint8_t>(FormationControl::kResetThenEmit)) {
    throw std::invalid_argument(
        who + ": control value " + std::to_string(value) +
        " is outside the FormationControl table");
  }
  return static_cast<FormationControl>(value);
}

std::shared_ptr<holoscan::Tensor> make_control_tensor(
    FormationControl value) {
  auto tensor =
      make_device_tensor({1}, DLDataType{kDLUInt, 8, 1});
  const uint8_t raw = static_cast<uint8_t>(value);
  if (cudaMemcpy(tensor->data(), &raw, sizeof(raw),
                 cudaMemcpyHostToDevice) != cudaSuccess) {
    throw std::runtime_error(
        "ApertureManagerOp: control H2D copy failed");
  }
  return tensor;
}

template <typename T>
std::shared_ptr<holoscan::Tensor> upload_weights(
    const std::vector<T>& host, const std::vector<int64_t>& shape,
    const std::string& who) {
  auto tensor = make_device_tensor(shape, real_dtype<T>());
  const size_t bytes = host.size() * sizeof(T);
  if (bytes > 0 &&
      cudaMemcpy(tensor->data(), host.data(), bytes,
                 cudaMemcpyHostToDevice) != cudaSuccess) {
    throw std::runtime_error(who + ": weight H2D copy failed");
  }
  return tensor;
}

void keep_group(std::vector<std::shared_ptr<holoscan::Tensor>>* keep,
                const PulseGroup& group) {
  if (group.profiles) {
    keep->push_back(group.profiles);
  }
  if (group.positions) {
    keep->push_back(group.positions);
  }
  if (group.reference) {
    keep->push_back(group.reference);
  }
}

}  // namespace

template <typename T>
void ApertureManagerOp<T>::initialize() {
  ensure_radar_arg_setters();
  Operator::initialize();
  metadata_policy(MetadataPolicy::kUpdate);

  const std::string& who = op_name<T>();
  max_ = max_aperture_pulses_.get();
  emit_interval_ = emit_interval_pulses_.get();
  emit_mode_value_ = emit_mode_.get();
  window_ = aperture_window_.get();
  windowed_ = window_.type != WindowType::kNone;
  evict_connected_ = input_port_connected(this, "evict");
  geometry_in_connected_ = input_port_connected(this, "sample_geometry");
  points_in_connected_ = input_port_connected(this, "sample_points");

  if (max_ < 0) {
    throw std::invalid_argument(
        who + ": max_aperture_pulses must be >= 0");
  }
  if (emit_interval_ < 0) {
    throw std::invalid_argument(
        who + ": emit_interval_pulses must be >= 0");
  }
  if (emit_mode_value_ == ApertureEmitMode::kReform && max_ == 0) {
    throw std::invalid_argument(
        who + ": emit_mode 'reform' requires max_aperture_pulses > 0");
  }
  if (evict_connected_ && max_ == 0) {
    throw std::invalid_argument(
        who + ": connecting 'evict' requires max_aperture_pulses > 0");
  }
  if (geometry_in_connected_ && points_in_connected_) {
    throw std::invalid_argument(
        who + ": connect sample_geometry or sample_points, not both");
  }
  const bool stamped = max_ == 0 && emit_interval_ > 0 &&
                       emit_mode_value_ == ApertureEmitMode::kReset;
  const bool re_form = emit_mode_value_ == ApertureEmitMode::kReform &&
                       max_ > 0;
  if (windowed_ && !stamped && !re_form) {
    throw std::invalid_argument(
        who + ": aperture_window is only legal on a block aperture "
              "(max_aperture_pulses = 0, emit_interval_pulses > 0, "
              "emit_mode = reset) or with emit_mode = reform");
  }
  if (!evict_connected_ && max_ > 0 && emit_interval_ >= max_) {
    HOLOSCAN_LOG_WARN(
        "{}: emit_interval_pulses ({}) >= max_aperture_pulses ({}); "
        "consecutive images share no pulses — "
        "max_aperture_pulses = 0 is cheaper",
        who, emit_interval_, max_);
  }
}

template <typename T>
void ApertureManagerOp<T>::setup(OperatorSpec& spec) {
  spec.input<holoscan::TensorMap>("range_profiles");
  spec.input<holoscan::TensorMap>("positions");
  spec.input<holoscan::TensorMap>("reference_position");
  add_override_port<SampleGeometry>(spec, "sample_geometry");
  spec.input<holoscan::TensorMap>("sample_points")
      .condition(ConditionType::kNone);
  spec.input<holoscan::TensorMap>("control")
      .condition(ConditionType::kNone);
  spec.input<holoscan::TensorMap>("evict")
      .condition(ConditionType::kNone);

  spec.output<holoscan::TensorMap>("range_profiles")
      .condition(ConditionType::kNone);
  spec.output<holoscan::TensorMap>("positions")
      .condition(ConditionType::kNone);
  spec.output<holoscan::TensorMap>("reference_position")
      .condition(ConditionType::kNone);
  spec.output<holoscan::TensorMap>("weight")
      .condition(ConditionType::kNone);
  spec.output<SampleGeometry>("sample_geometry")
      .condition(ConditionType::kNone);
  spec.output<holoscan::TensorMap>("sample_points")
      .condition(ConditionType::kNone);
  spec.output<holoscan::TensorMap>("control")
      .condition(ConditionType::kNone);

  spec.param(max_aperture_pulses_, "max_aperture_pulses",
             "Retention capacity (pulses)",
             "0 = no retention (pass-through). Static. Default: 0.",
             static_cast<int64_t>(0));
  spec.param(emit_interval_pulses_, "emit_interval_pulses",
             "Emit after this many admissions",
             "0 = never emit on a count. Static. Default: 0.",
             static_cast<int64_t>(0));
  spec.param(emit_mode_, "emit_mode", "Emit control variant",
             "continue, reset, or reform. Static. Default: continue.",
             ApertureEmitMode::kContinue);
  spec.param(aperture_window_, "aperture_window", "Azimuth taper",
             "WindowSpec across the aperture. Default: none.",
             WindowSpec{});
}

template <typename T>
void ApertureManagerOp<T>::start() {
  buffer_ = ApertureBuffer(max_);
  ctrl_continue_ =
      make_control_tensor(FormationControl::kEmitAndContinue);
  ctrl_reset_ = make_control_tensor(FormationControl::kEmitAndReset);
  ctrl_pre_emit_ =
      make_control_tensor(FormationControl::kResetThenEmit);
  ctrl_no_emit_ =
      make_control_tensor(FormationControl::kResetWithoutEmit);
  if (windowed_ && emit_interval_ > 0 &&
      emit_mode_value_ == ApertureEmitMode::kReset && max_ == 0) {
    stamp_window_ = generate_window(window_, emit_interval_);
  }
}

template <typename T>
void ApertureManagerOp<T>::compute(InputContext& op_input,
                                   OutputContext& op_output,
                                   [[maybe_unused]] ExecutionContext&
                                       context) {
  const std::string& who = op_name<T>();
  auto profiles = receive_single(op_input, "range_profiles", who);
  auto positions = receive_single(op_input, "positions", who);
  auto reference =
      receive_single(op_input, "reference_position", who);
  cudaStream_t stream = op_input.receive_cuda_stream("range_profiles");
  auto stamp_stream = [&](const char* port) {
    if (stream != cudaStreamDefault) {
      op_output.set_cuda_stream(stream, port);
    }
  };

  const std::string boundary = metadata()->get<std::string>(
      kStreamBoundaryKey, std::string());
  const bool d22_collection =
      boundary == kStreamBoundaryCollection;
  const bool d22_end = boundary == kStreamBoundaryEndOfStream;
  const bool d22 = d22_collection || d22_end;

  auto request_tensor = receive_optional(op_input, "control", who);
  std::optional<FormationControl> request;
  if (request_tensor) {
    request = read_control(
        *request_tensor, who,
        op_input.receive_cuda_stream("control"));
  }
  auto geom = receive_override<SampleGeometry>(op_input,
                                               "sample_geometry");
  if (geom.has_value()) {
    pending_geometry_ = std::move(*geom);
  }
  auto points = receive_optional(op_input, "sample_points", who);
  if (points) {
    pending_points_ = points;
  }
  int64_t evict_count = 0;
  auto evict_tensor = receive_optional(op_input, "evict", who);
  if (evict_connected_ && evict_tensor) {
    evict_count = static_cast<int64_t>(read_uint32(
        *evict_tensor, who, "evict",
        op_input.receive_cuda_stream("evict")));
  }

  classify_input_dtype<T>(*profiles, "range_profiles",
                          /*allow_real=*/false);
  require_dtype(*positions, real_dtype<T>(), "positions", who);
  require_dtype(*reference, real_dtype<T>(), "reference_position",
                who);
  require_contiguous(*profiles, "range_profiles", who);
  require_contiguous(*positions, "positions", who);
  require_contiguous(*reference, "reference_position", who);
  require_cuda_accessible(*profiles, "range_profiles", who);
  require_cuda_accessible(*positions, "positions", who);
  require_cuda_accessible(*reference, "reference_position", who);

  const BatchedShape cube = collapse_batch(*profiles, "range_profiles");
  const BatchedShape pos = collapse_batch(*positions, "positions");
  const BatchedShape ref = collapse_batch(*reference,
                                          "reference_position");
  if (pos.cols != 3 || ref.cols != 3) {
    throw std::invalid_argument(
        who + ": positions and reference_position last axis must be 3");
  }
  if (pos.rows != cube.rows || ref.rows != cube.rows ||
      pos.batch != cube.batch || ref.batch != cube.batch) {
    throw std::invalid_argument(
        who + ": positions / reference_position pulse and batch axes "
              "must match range_profiles");
  }
  const int64_t np = cube.rows;
  if (max_ > 0 && np > max_) {
    throw std::invalid_argument(
        who + ": incoming pulse count (" + std::to_string(np) +
        ") exceeds max_aperture_pulses (" + std::to_string(max_) + ")");
  }

  const bool reform = emit_mode_value_ == ApertureEmitMode::kReform;
  const bool request_pre =
      request.has_value() && is_pre_reset(*request) && !reform;

  if (request_pre) {
    dropped_ += buffer_.occupancy();
    buffer_.clear();
    admitted_since_emit_ = 0;
    slot_ = 0;
  }

  std::vector<PulseGroup> departing;
  if (!reform && evict_connected_ && evict_count > 0) {
    if (evict_count > buffer_.occupancy()) {
      throw std::invalid_argument(
          who + ": evict count (" + std::to_string(evict_count) +
          ") exceeds occupancy (" +
          std::to_string(buffer_.occupancy()) + ")");
    }
    departing = buffer_.evict(evict_count);
    evicted_ += pulse_group_total(departing);
  }
  if (!reform && !evict_connected_ && max_ > 0) {
    auto extra = buffer_.evict_for_admission(np);
    evicted_ += pulse_group_total(extra);
    departing.insert(departing.end(), extra.begin(), extra.end());
  }
  if (evict_connected_ && max_ > 0 &&
      buffer_.occupancy() + np > max_) {
    throw std::invalid_argument(
        who + ": admitting " + std::to_string(np) +
        " pulses would exceed max_aperture_pulses (" +
        std::to_string(max_) + ") without an evict");
  }

  PulseGroup incoming{profiles, positions, reference, 0, np};
  if (max_ > 0) {
    buffer_.admit(incoming);
  }
  const int64_t slot0 = slot_;
  admitted_ += np;
  admitted_since_emit_ += np;
  slot_ += np;

  const bool cadence = emit_interval_ > 0 &&
                       admitted_since_emit_ >= emit_interval_;
  const bool request_continue =
      request.has_value() &&
      *request == FormationControl::kEmitAndContinue;
  const bool request_reset =
      request.has_value() && is_resetting(*request);
  const bool end_aperture =
      d22 || request_reset ||
      (cadence && emit_mode_value_ != ApertureEmitMode::kContinue);
  const bool emit_now =
      cadence || request.has_value() || d22;

  FormationControl out_ctrl = FormationControl::kEmitAndContinue;
  if (reform && emit_now) {
    out_ctrl = FormationControl::kResetThenEmit;
  } else if (d22 || (cadence &&
                     emit_mode_value_ == ApertureEmitMode::kReset)) {
    out_ctrl = FormationControl::kEmitAndReset;
  } else if (request.has_value() && !reform) {
    out_ctrl = *request;
  } else if (cadence) {
    out_ctrl = FormationControl::kEmitAndContinue;
  }

  const bool drop_after =
      (end_aperture && !request_pre) ||
      (reform && end_aperture);
  const bool replay = reform && emit_now;

  auto forward_geometry = [&]() {
    if (pending_geometry_.has_value()) {
      op_output.emit(*pending_geometry_, "sample_geometry");
      pending_geometry_.reset();
    }
    if (pending_points_) {
      stamp_stream("sample_points");
      emit_single(op_output, "sample_points", pending_points_);
      pending_points_.reset();
    }
  };

  keep_.clear();
  keep_group(&keep_, incoming);
  for (const PulseGroup& group : departing) {
    keep_group(&keep_, group);
  }

  if (replay) {
    std::vector<PulseGroup> replayed = buffer_.retained();
    for (const PulseGroup& group : replayed) {
      keep_group(&keep_, group);
    }
    PulseGroup batch = concat_pulse_groups(replayed, stream, who.c_str());
    const int64_t n_out = batch.count;
    std::vector<T> weights(static_cast<size_t>(n_out), T{1});
    if (windowed_ && n_out > 0) {
      const auto coeff = generate_window(window_, n_out);
      if (static_cast<int64_t>(coeff.size()) == n_out) {
        for (int64_t i = 0; i < n_out; ++i) {
          weights[static_cast<size_t>(i)] =
              static_cast<T>(coeff[static_cast<size_t>(i)]);
        }
      }
    }
    stamp_stream("range_profiles");
    stamp_stream("positions");
    stamp_stream("reference_position");
    stamp_stream("weight");
    emit_single(op_output, "range_profiles", batch.profiles);
    emit_single(op_output, "positions", batch.positions);
    emit_single(op_output, "reference_position", batch.reference);
    emit_single(op_output, "weight",
                upload_weights<T>(weights, weight_shape_for(cube, n_out),
                                  who));
  } else if (!reform) {
    std::vector<PulseGroup> admitted_groups;
    admitted_groups.push_back(incoming);
    std::vector<PulseGroup> all = admitted_groups;
    all.insert(all.end(), departing.begin(), departing.end());
    PulseGroup batch = concat_pulse_groups(all, stream, who.c_str());
    const int64_t n_add = np;
    const int64_t n_dep = pulse_group_total(departing);
    const int64_t n_out = n_add + n_dep;
    std::vector<T> weights(static_cast<size_t>(n_out), T{1});
    if (windowed_ && !stamp_window_.empty()) {
      for (int64_t i = 0; i < n_add; ++i) {
        const int64_t slot = slot0 + i;
        if (slot < static_cast<int64_t>(stamp_window_.size())) {
          weights[static_cast<size_t>(i)] = static_cast<T>(
              stamp_window_[static_cast<size_t>(slot)]);
        }
      }
    }
    for (int64_t i = 0; i < n_dep; ++i) {
      weights[static_cast<size_t>(n_add + i)] = static_cast<T>(-1);
    }
    stamp_stream("range_profiles");
    stamp_stream("positions");
    stamp_stream("reference_position");
    stamp_stream("weight");
    emit_single(op_output, "range_profiles", batch.profiles);
    emit_single(op_output, "positions", batch.positions);
    emit_single(op_output, "reference_position", batch.reference);
    emit_single(op_output, "weight",
                upload_weights<T>(weights, weight_shape_for(cube, n_out),
                                  who));
  }

  if (emit_now) {
    std::shared_ptr<holoscan::Tensor> ctrl = ctrl_continue_;
    if (out_ctrl == FormationControl::kEmitAndReset) {
      ctrl = ctrl_reset_;
    } else if (out_ctrl == FormationControl::kResetThenEmit) {
      ctrl = ctrl_pre_emit_;
    } else if (out_ctrl == FormationControl::kResetWithoutEmit) {
      ctrl = ctrl_no_emit_;
    }
    stamp_stream("control");
    emit_single(op_output, "control", ctrl);
    if (end_aperture || request_pre) {
      forward_geometry();
    }
    if (windowed_ && !stamp_window_.empty() && end_aperture &&
        slot_ < emit_interval_) {
      HOLOSCAN_LOG_WARN(
          "{}: aperture_window stamped for {} slots but flush "
          "arrived at {}; sidelobes follow the truncated window",
          who, emit_interval_, slot_);
    }
    if (end_aperture || request_continue || cadence) {
      admitted_since_emit_ = 0;
    }
  }

  if (drop_after) {
    dropped_ += buffer_.occupancy();
    buffer_.clear();
    slot_ = 0;
    admitted_since_emit_ = 0;
  }

  prev_keep_.swap(keep_);
  keep_.clear();
}

template class ApertureManagerOp<float>;
template class ApertureManagerOp<double>;

}  // namespace holoscan::radar
