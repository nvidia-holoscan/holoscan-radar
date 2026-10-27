// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// App-owned consumer for AutofocusOp.sharpness. Without one the operator sees
// no wired consumer, skips the work behind the port for the rest of the run,
// and the iteration trace -- the one metric that says whether PGA converged
// rather than merely produced a number -- is unobservable.

#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/expected_macros.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/logger/logger.hpp>

#include "radar_common/sdk/tensor_io.hpp"
#include "radar_common/tensor/tensor_utils.hpp"

namespace holoscan::radar::app::sar {

using holoscan::Contract;
using holoscan::Error;
using holoscan::ErrorCode;
using holoscan::ExecutionContext;
using holoscan::expected;
using holoscan::Input;
using holoscan::make_unexpected;
using holoscan::OnEach;
using holoscan::OperatorSpec;
using holoscan::Tensor;

// Logs the sharpness trace and the ratio across it. Sharpness is intensity
// squared summed over the image, so it is scale-dependent and only the ratio
// between iterations means anything; a trace that is flat after iteration 0 is
// PGA declining to improve the image, which reads identically in the image
// metrics to PGA having nothing to improve.
class AutofocusSharpnessReportOp : public holoscan::Operator<> {
 public:
  AutofocusSharpnessReportOp() = default;

  void setup(OperatorSpec& spec) override { spec.input(sharpness).queue_depth(2U); }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnEach{sharpness});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    constexpr const char* kWho = "AutofocusSharpnessReportOp";
    HOLOSCAN_TRY(auto trace, receive_tensor(sharpness, kWho));
    try {
      require_contiguous(trace, "sharpness", kWho);
      require_cuda_accessible(trace, "sharpness", kWho);
    } catch (const std::invalid_argument& e) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument, e.what()});
    }
    const auto shape = trace.shape();
    if (shape.size() != 1U || shape[0] < 1) {
      return make_unexpected(
          Error{ErrorCode::kInvalidArgument, "AutofocusSharpnessReportOp: expected a 1-D trace"});
    }
    const bool f64 = trace.dtype().code == kDLFloat && trace.dtype().bits == 64;
    const bool f32 = trace.dtype().code == kDLFloat && trace.dtype().bits == 32;
    if (!f64 && !f32) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument,
                                   "AutofocusSharpnessReportOp: expected float32/float64"});
    }
    cudaStream_t stream = context.cuda_stream().get();
    if (stream == nullptr) {
      return make_unexpected(
          Error{ErrorCode::kNotSupported, "AutofocusSharpnessReportOp: no CUDA stream"});
    }
    const auto n = static_cast<std::size_t>(shape[0]);
    std::vector<double> values(n);
    if (f64) {
      if (cudaMemcpyAsync(values.data(), trace.data(), n * sizeof(double), cudaMemcpyDeviceToHost,
                          stream) != cudaSuccess ||
          cudaStreamSynchronize(stream) != cudaSuccess) {
        return make_unexpected(
            Error{ErrorCode::kFailure, "AutofocusSharpnessReportOp: D2H failed"});
      }
    } else {
      std::vector<float> tmp(n);
      if (cudaMemcpyAsync(tmp.data(), trace.data(), n * sizeof(float), cudaMemcpyDeviceToHost,
                          stream) != cudaSuccess ||
          cudaStreamSynchronize(stream) != cudaSuccess) {
        return make_unexpected(
            Error{ErrorCode::kFailure, "AutofocusSharpnessReportOp: D2H failed"});
      }
      for (std::size_t i = 0; i < n; ++i) {
        values[i] = static_cast<double>(tmp[i]);
      }
    }

    std::string trace_text;
    bool monotone = true;
    for (std::size_t i = 0; i < n; ++i) {
      if (i > 0) {
        trace_text += " -> ";
        if (values[i] < values[i - 1]) {
          monotone = false;
        }
      }
      trace_text += fmt::format("{:.4g}", values[i]);
    }
    const double gain = values[0] > 0.0 ? values[n - 1] / values[0] : 0.0;
    HOLOSCAN_LOG_INFO("AutofocusSharpnessReportOp: sharpness {} ({:.3f}x over {} iteration(s))",
                      trace_text, gain, n - 1U);
    if (!monotone) {
      HOLOSCAN_LOG_WARN(
          "AutofocusSharpnessReportOp: sharpness did not improve monotonically; an iteration was "
          "rejected by the acceptance gate");
    }
    return {};
  }

  Input<Tensor> sharpness{"sharpness"};
};

}  // namespace holoscan::radar::app::sar
