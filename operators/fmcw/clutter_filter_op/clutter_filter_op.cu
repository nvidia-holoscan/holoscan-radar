// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "fmcw/clutter_filter_op/clutter_filter_op.hpp"

#include <stdexcept>

#include "fmcw/clutter_filter_op/clutter_stage.hpp"
#include "radar_common/sdk/override_param.hpp"

namespace holoscan::radar {

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
                                 ExecutionContext& context) {
  // Implemented in a subsequent commit.
  throw std::runtime_error("ClutterFilterOp: not implemented yet");
}

template class ClutterFilterOp<float>;
template class ClutterFilterOp<double>;

}  // namespace holoscan::radar
