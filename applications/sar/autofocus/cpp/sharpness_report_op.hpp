// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// App-owned consumer for AutofocusOp.sharpness. Without one the operator sees
// no wired consumer, skips the work behind the port for the rest of the run,
// and the iteration trace -- the one metric that says whether PGA converged
// rather than merely produced a number -- is unobservable.

#pragma once

#include <cuda_runtime_api.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <optional>
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

#include "radar_common/descriptors/enums.hpp"
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
//
// `termination` and `iterations` resolve exactly that ambiguity, which is why
// they are taken here: the trace says what happened to the metric, and those
// two say what the operator did about it.
//
// All three are required and aligned rather than optional joins. The `[live]`
// spelling (`Exact{}` + `on_missing SafeDefault{}`) does not work for them:
// SafeDefault means "fall back to the Params seed", and a tensor port has no
// seed, so the join never matches and the ports read as permanently absent.
// AutofocusOp emits all four outputs from one compute() with
// `capture_time = activation_time()`, so they share a stamp exactly and a
// zero-window alignment is the right join.
class AutofocusSharpnessReportOp : public holoscan::Operator<> {
 public:
  AutofocusSharpnessReportOp() = default;

  void setup(OperatorSpec& spec) override {
    spec.input(sharpness).queue_depth(2U);
    spec.input(termination).queue_depth(2U);
    spec.input(iterations).queue_depth(2U);
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result
        .trigger(holoscan::OnAligned{.ports = holoscan::ports(sharpness, termination, iterations),
                                     .window = std::chrono::nanoseconds{0}})
        .input(sharpness, holoscan::Exact{})
        .input(termination, holoscan::Exact{})
        .input(iterations, holoscan::Exact{});
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

    // The provenance half, when the graph wires it. A gain of 1.000x above is
    // the ambiguous case this resolves: it reads the same whether PGA refused
    // or had nothing to do, and only these say which.
    const auto reason = read_status(termination, stream);
    const auto accepted = read_status(iterations, stream);
    if (reason.has_value()) {
      const auto code = static_cast<AutofocusTermination>(*reason);
      if (autofocus_refused(code)) {
        HOLOSCAN_LOG_WARN(
            "AutofocusSharpnessReportOp: autofocus REFUSED ({}); the image was returned "
            "uncorrected and phase_correction is zero",
            to_string(code));
      } else {
        HOLOSCAN_LOG_INFO("AutofocusSharpnessReportOp: termination {}, {} iteration(s) accepted",
                          to_string(code), accepted.value_or(-1));
      }
    }
    return {};
  }

  Input<Tensor> sharpness{"sharpness"};
  Input<Tensor> termination{"termination"};
  Input<Tensor> iterations{"iterations"};

 private:
  // One int32 off the device, or nothing when the port is unwired or had no
  // match this tick. A read failure is reported as absent rather than failing
  // the tick: this operator is a report, and losing the log line must not take
  // the run down.
  static std::optional<std::int32_t> read_status(Input<Tensor>& port, cudaStream_t stream) {
    auto received = port.receive_data();
    if (!received) {
      return std::nullopt;
    }
    const Tensor& t = *received;
    if (t.size() < 1 || t.dtype().code != kDLInt || t.dtype().bits != 32) {
      return std::nullopt;
    }
    std::int32_t value = 0;
    if (cudaMemcpyAsync(&value, t.data(), sizeof(value), cudaMemcpyDeviceToHost, stream) !=
            cudaSuccess ||
        cudaStreamSynchronize(stream) != cudaSuccess) {
      return std::nullopt;
    }
    return value;
  }
};

}  // namespace holoscan::radar::app::sar
