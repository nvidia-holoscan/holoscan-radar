// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "sar/phase_correction_op/phase_correction_op.hpp"

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include "radar_common/sdk/arg_setters.hpp"
#include "radar_common/sdk/tensor_io.hpp"
#include "radar_common/tensor/dlpack_buffer.hpp"
#include "radar_common/tensor/tensor_utils.hpp"
#include "sar/phase_correction_op/phase_correction_stage.hpp"

namespace holoscan::radar {

namespace {

template <typename T>
constexpr const char* precision_name() {
  return std::is_same_v<T, float> ? "float" : "double";
}

template <typename T>
const std::string& op_name() {
  static const std::string name =
      std::string("PhaseCorrectionOp<") + precision_name<T>() + ">";
  return name;
}

// phase_correction is range_profiles without the last (range) axis.
void require_matching_pulse_axis(const holoscan::Tensor& profiles,
                                 const holoscan::Tensor& phase,
                                 const std::string& who) {
  const auto pshape = profiles.shape();
  const auto zshape = phase.shape();
  if (pshape.size() < 2) {
    throw std::invalid_argument(
        who + ": range_profiles must have rank >= 2");
  }
  if (zshape.size() != pshape.size() - 1) {
    throw std::invalid_argument(
        who + ": phase_correction rank (" +
        std::to_string(zshape.size()) +
        ") must be range_profiles rank minus one (" +
        std::to_string(pshape.size() - 1) + ")");
  }
  for (size_t i = 0; i < zshape.size(); ++i) {
    if (zshape[i] != pshape[i]) {
      throw std::invalid_argument(
          who + ": phase_correction axis " + std::to_string(i) +
          " is " + std::to_string(zshape[i]) +
          ", range_profiles has " + std::to_string(pshape[i]));
    }
  }
}

}  // namespace

template <typename T>
void PhaseCorrectionOp<T>::initialize() {
  ensure_radar_arg_setters();
  Operator::initialize();
  metadata_policy(MetadataPolicy::kUpdate);

  const std::string& who = op_name<T>();
  sense_value_ = sense_.get();
  if (sense_value_ != PhaseCorrectionSense::kSubtract &&
      sense_value_ != PhaseCorrectionSense::kAdd) {
    throw std::invalid_argument(
        who + ": sense is required (subtract or add)");
  }
}

template <typename T>
void PhaseCorrectionOp<T>::setup(OperatorSpec& spec) {
  spec.input<holoscan::TensorMap>("range_profiles");
  spec.input<holoscan::TensorMap>("phase_correction");
  spec.output<holoscan::TensorMap>("range_profiles");

  spec.param(sense_, "sense", "Correction sense",
             "subtract multiplies by exp(-i phi); add by exp(+i phi). "
             "Static. Required.",
             PhaseCorrectionSense::kUnspecified);
}

template <typename T>
void PhaseCorrectionOp<T>::compute(InputContext& op_input,
                                   OutputContext& op_output,
                                   [[maybe_unused]] ExecutionContext&
                                       context) {
  const std::string& who = op_name<T>();

  auto profiles = receive_single(op_input, "range_profiles", who);
  auto phase = receive_single(op_input, "phase_correction", who);

  classify_input_dtype<T>(*profiles, "range_profiles",
                          /*allow_real=*/false);
  require_dtype(*phase, real_dtype<T>(), "phase_correction", who);
  require_cuda_accessible(*profiles, "range_profiles", who);
  require_cuda_accessible(*phase, "phase_correction", who);
  require_contiguous(*profiles, "range_profiles", who);
  require_contiguous(*phase, "phase_correction", who);
  require_matching_pulse_axis(*profiles, *phase, who);

  const BatchedShape cube = collapse_batch(*profiles, "range_profiles");
  cudaStream_t stream = op_input.receive_cuda_stream("range_profiles");

  auto output = make_device_tensor(
      full_rank_shape(cube, cube.rows, cube.cols), complex_dtype<T>());
  detail::run_phase_correction<T>(
      sense_value_, profiles->data(), phase->data(), cube.batch,
      cube.rows, cube.cols, output->data(), stream, who);

  if (stream != cudaStreamDefault) {
    op_output.set_cuda_stream(stream, "range_profiles");
  }
  emit_single(op_output, "range_profiles", output);

  retained_profiles_ = std::move(profiles);
  retained_phase_ = std::move(phase);
}

template class PhaseCorrectionOp<float>;
template class PhaseCorrectionOp<double>;

}  // namespace holoscan::radar
