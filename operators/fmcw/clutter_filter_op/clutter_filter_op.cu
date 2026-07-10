// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "fmcw/clutter_filter_op/clutter_filter_op.hpp"

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include "fmcw/clutter_filter_op/clutter_stage.hpp"
#include "radar_common/sdk/arg_setters.hpp"
#include "radar_common/tensor/dlpack_buffer.hpp"
#include "radar_common/sdk/override_param.hpp"
#include "radar_common/tensor/tensor_utils.hpp"

namespace holoscan::radar {

template <typename T>
void ClutterFilterOp<T>::initialize() {
  // Custom parameter types (the method enum) need registered argument
  // setters before Operator::initialize() processes the arguments.
  ensure_radar_arg_setters();
  Operator::initialize();
}

template <typename T>
void ClutterFilterOp<T>::setup(OperatorSpec& spec) {
  spec.input<holoscan::TensorMap>("range_profiles");
  spec.output<holoscan::TensorMap>("range_profiles");

  spec.param(method_, "method", "Clutter suppression method",
             "Slow-time clutter suppression {mean_subtraction, "
             "single_delay, double_delay}: mean_subtraction nulls "
             "zero-Doppler with no chirp loss; the delay cancellers are "
             "1st/2nd-order MTI high-pass filters and drop 1-2 chirps.",
             ClutterFilterMethod::kMeanSubtraction);
  add_override_port<ClutterFilterMethod>(spec, "method");
}

template <typename T>
void ClutterFilterOp<T>::compute(InputContext& op_input,
                                 OutputContext& op_output,
                                 [[maybe_unused]] ExecutionContext& context) {
  // Single-entry TensorMap, accepted regardless of its key (module
  // convention: the port name is the contract, not the map key).
  auto maybe_map = op_input.receive<holoscan::TensorMap>("range_profiles");
  if (!maybe_map) {
    throw std::runtime_error(
        "ClutterFilterOp: no message received on input port "
        "'range_profiles'");
  }
  holoscan::TensorMap& in_map = maybe_map.value();
  if (in_map.size() != 1) {
    throw std::invalid_argument(
        "ClutterFilterOp: input port 'range_profiles' expects a "
        "single-entry TensorMap, got " +
        std::to_string(in_map.size()) + " entries");
  }
  std::shared_ptr<holoscan::Tensor> input = in_map.begin()->second;
  if (!input) {
    throw std::runtime_error(
        "ClutterFilterOp: received a null tensor on input port "
        "'range_profiles'");
  }

  // method for this tick; no setup cost, so no warn-once machinery.
  const ClutterFilterMethod method =
      resolve_override_param<ClutterFilterMethod>(op_input, "method",
                                                  method_.get(), nullptr);
  if (method == ClutterFilterMethod::kNone) {
    throw std::invalid_argument(
        "ClutterFilterOp: method 'none' is not supported by the standalone "
        "operator (it exists only for the fused operator's inline clutter "
        "stage); use one of {mean_subtraction, single_delay, double_delay}");
  }

  // Complex-only input per the locked contract; collapse leading batch axes.
  classify_input_dtype<T>(*input, "range_profiles", /*allow_real=*/false);
  require_cuda_accessible(*input, "range_profiles", "ClutterFilterOp");
  const BatchedShape batched = collapse_batch(*input, "range_profiles");
  const int64_t num_chirps = batched.rows;
  if (method == ClutterFilterMethod::kSingleDelay && num_chirps < 2) {
    throw std::invalid_argument(
        "ClutterFilterOp: method 'single_delay' needs num_chirps >= 2, "
        "got " + std::to_string(num_chirps));
  }
  if (method == ClutterFilterMethod::kDoubleDelay && num_chirps < 3) {
    throw std::invalid_argument(
        "ClutterFilterOp: method 'double_delay' needs num_chirps >= 3, "
        "got " + std::to_string(num_chirps));
  }

  cudaStream_t stream = op_input.receive_cuda_stream("range_profiles");

  const int64_t num_chirps_out =
      detail::clutter_num_chirps_out(method, num_chirps);
  auto output = make_device_tensor(
      full_rank_shape(batched, num_chirps_out, batched.cols),
      complex_dtype<T>());

  detail::run_clutter_stage<T>(method, input->data(), batched.batch,
                               num_chirps, batched.cols, output->data(),
                               stream);

  // receive_cuda_stream already configures outputs to carry the internal
  // stream; set_cuda_stream additionally covers manual-stream setups. It
  // only accepts Holoscan-managed streams, so skip the default stream
  // (no stream pool and no upstream stream this tick).
  if (stream != cudaStreamDefault) {
    op_output.set_cuda_stream(stream, "range_profiles");
  }
  holoscan::TensorMap out_map;
  out_map.insert({"range_profiles", output});
  op_output.emit(out_map, "range_profiles");

  // Stream-ordered lifetime guard: keep this tick's input alive until the
  // next tick so async reads on `stream` cannot outlive the buffer.
  retained_input_ = std::move(input);
}

template class ClutterFilterOp<float>;
template class ClutterFilterOp<double>;

}  // namespace holoscan::radar
