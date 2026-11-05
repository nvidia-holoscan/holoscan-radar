// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// App-owned map from AutofocusOp's azimuth-bin phase onto pulses. Not a
// contract operator. PGA's phase_correction is slow-time / azimuth-FFT
// indexed, length = the named azimuth axis. Figure 2's PhaseCorrectionOp is
// pulse-indexed. When the aux is Nyquist-sampled (n_u != n_pulses) this is the
// known transform-length resampling the AutofocusOp notes already allow. An
// oversampled helper packs the slow-time history into the low-frequency DFT
// bins (zeros in the extra high-frequency slots), so downsampling takes those
// bins rather than stretching the whole vector. A shorter helper still
// linear-interpolates up onto the aperture.

#pragma once

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
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

#include "radar_common/sdk/tensor_io.hpp"
#include "radar_common/tensor/max_byte_span.hpp"
#include "radar_common/tensor/tensor_utils.hpp"

namespace holoscan::radar::app::sar {

class PhaseResampleOp : public holoscan::Operator<> {
 public:
  explicit PhaseResampleOp(int64_t num_pulses) : num_pulses_(num_pulses) {
    if (num_pulses_ < 2) {
      throw std::invalid_argument("PhaseResampleOp: num_pulses must be >= 2");
    }
    // Either precision may arrive, so the pool is sized for the wider one.
    max_byte_span_ = byte_span_real_cube<double>({num_pulses_}, "PhaseResampleOp");
  }

  void setup(OperatorSpec& spec) override {
    constexpr TensorRepresentation device{
        .memory_kind = MemoryKind::kCudaDevice, .dtype = std::nullopt, .rank = std::nullopt};
    spec.input(phase_correction).queue_depth(2U).expects_tensor(tensor_input(device));
    spec.output(phase_correction_out)
        .produces_tensor(pooled_tensor_output(device, 1U, max_byte_span_));
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnEach{phase_correction});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    constexpr const char* kOpName = "PhaseResampleOp";
    HOLOSCAN_TRY(auto phase, receive_tensor(phase_correction, kOpName));
    try {
      require_contiguous(phase, "phase_correction", kOpName);
      require_cuda_accessible(phase, "phase_correction", kOpName);
    } catch (const std::invalid_argument& e) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument, e.what()});
    }
    const auto shape = phase.shape();
    if (shape.size() != 1U || shape[0] < 2) {
      return make_unexpected(
          Error{ErrorCode::kInvalidArgument, "PhaseResampleOp: expected (n) with n >= 2"});
    }
    const int64_t n_az = shape[0];
    const auto dtype = phase.dtype();
    const bool f64 = dtype.code == kDLFloat && dtype.bits == 64;
    const bool f32 = dtype.code == kDLFloat && dtype.bits == 32;
    if (!f64 && !f32) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument,
                                   "PhaseResampleOp: expected float32/float64 phase_correction"});
    }
    cudaStream_t stream = context.cuda_stream().get();
    if (stream == nullptr) {
      return make_unexpected(Error{ErrorCode::kNotSupported, "PhaseResampleOp: no CUDA stream"});
    }

    auto emitted = emit_required_tensor(
        phase_correction_out, context, "PhaseResampleOp.phase_correction_out", false, [&] {
          return prepare_device_tensor(
              phase_correction_out, context, std::vector<int64_t>{num_pulses_}, dtype,
              [&](void* destination, std::size_t bytes) -> expected<void, Error> {
                if (n_az == num_pulses_) {
                  // Straight copy: AutofocusOp.phase_correction is
                  // pulse-indexed (OPERATOR_API.md), so no reordering belongs
                  // here. Checked rather than assumed -- rolling this vector
                  // by n/2 either way costs peak recovery (0.55 -> 0.33 on
                  // Kourou), which is what an unnecessary fftshift looks like.
                  if (cudaMemcpyAsync(destination, phase.data(), bytes, cudaMemcpyDeviceToDevice,
                                      stream) != cudaSuccess) {
                    return make_unexpected(
                        Error{ErrorCode::kFailure, "PhaseResampleOp: D2D copy failed"});
                  }
                  return {};
                }
                return resample(phase, n_az, f64, destination, stream);
              });
        });

    // The identity branch's device-to-device copy is issued async on the
    // compute stream and nothing synchronizes after it, so the source has to
    // outlive this call -- and it is the branch the default polar auxiliary
    // always takes. `resample` synchronizes internally and would not need
    // this, but holding the tensor unconditionally is cheaper to reason about
    // than holding it on one branch only.
    retained_phase_ = std::move(phase);
    return emitted;
  }

  Input<Tensor> phase_correction{"phase_correction"};
  Output<Tensor> phase_correction_out{"phase_correction_out"};

 private:
  Tensor retained_phase_;

  expected<void, Error> resample(const Tensor& phase, int64_t n_az, bool f64, void* destination,
                                 cudaStream_t stream) const {
    std::vector<double> phi(static_cast<std::size_t>(n_az));
    if (f64) {
      if (cudaMemcpyAsync(phi.data(), phase.data(), static_cast<std::size_t>(n_az) * sizeof(double),
                          cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
          cudaStreamSynchronize(stream) != cudaSuccess) {
        return make_unexpected(Error{ErrorCode::kFailure, "PhaseResampleOp: phase D2H failed"});
      }
    } else {
      std::vector<float> tmp(static_cast<std::size_t>(n_az));
      if (cudaMemcpyAsync(tmp.data(), phase.data(), tmp.size() * sizeof(float),
                          cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
          cudaStreamSynchronize(stream) != cudaSuccess) {
        return make_unexpected(Error{ErrorCode::kFailure, "PhaseResampleOp: phase D2H failed"});
      }
      for (int64_t i = 0; i < n_az; ++i) {
        phi[static_cast<std::size_t>(i)] = static_cast<double>(tmp[static_cast<std::size_t>(i)]);
      }
    }

    std::vector<double> out(static_cast<std::size_t>(num_pulses_));
    if (n_az > num_pulses_) {
      // Oversampled aux: the estimate arrives pulse-ordered over an azimuth
      // extent wider than the aperture, and only the central `num_pulses`
      // samples carry real pulses -- the rest is the empty bandwidth the extra
      // columns bought. Take the centre.
      const int64_t start = (n_az - num_pulses_) / 2;
      for (int64_t p = 0; p < num_pulses_; ++p) {
        out[static_cast<std::size_t>(p)] = phi[static_cast<std::size_t>(start + p)];
      }
    } else {
      const double tmax = static_cast<double>(n_az - 1);
      const double scale = tmax / static_cast<double>(num_pulses_ - 1);
      for (int64_t p = 0; p < num_pulses_; ++p) {
        const double t = std::min(tmax, std::max(0.0, static_cast<double>(p) * scale));
        const int64_t i0 = static_cast<int64_t>(std::floor(t));
        const int64_t i1 = std::min(i0 + 1, n_az - 1);
        const double f = t - static_cast<double>(i0);
        const double a = phi[static_cast<std::size_t>(i0)];
        const double b = phi[static_cast<std::size_t>(i1)];
        out[static_cast<std::size_t>(p)] = a + f * (b - a);
      }
    }

    if (f64) {
      if (cudaMemcpyAsync(destination, out.data(), out.size() * sizeof(double),
                          cudaMemcpyHostToDevice, stream) != cudaSuccess) {
        return make_unexpected(Error{ErrorCode::kFailure, "PhaseResampleOp: H2D failed"});
      }
    } else {
      std::vector<float> tmp(static_cast<std::size_t>(num_pulses_));
      for (int64_t p = 0; p < num_pulses_; ++p) {
        tmp[static_cast<std::size_t>(p)] = static_cast<float>(out[static_cast<std::size_t>(p)]);
      }
      if (cudaMemcpyAsync(destination, tmp.data(), tmp.size() * sizeof(float),
                          cudaMemcpyHostToDevice, stream) != cudaSuccess) {
        return make_unexpected(Error{ErrorCode::kFailure, "PhaseResampleOp: H2D failed"});
      }
    }
    if (cudaStreamSynchronize(stream) != cudaSuccess) {
      return make_unexpected(Error{ErrorCode::kFailure, "PhaseResampleOp: H2D sync failed"});
    }
    return {};
  }

  int64_t num_pulses_{0};
  std::size_t max_byte_span_{0U};
};

}  // namespace holoscan::radar::app::sar
