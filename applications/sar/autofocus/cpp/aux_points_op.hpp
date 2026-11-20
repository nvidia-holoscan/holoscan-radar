// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// App-owned source for the Figure 2 range-bearing aux lattice. Not a contract
// operator: BackprojectionOp.sample_points is the contract port, and this is
// the app supplying it, which D20 says is the application's job.
//
// It sits in series on the replay edge between the manager and the auxiliary
// formation, and emits the lattice straight to that formation rather than
// through the manager. One producer and one tick is what gives the lattice and
// the profiles it belongs to the same capture time, which is the co-arrival the
// aux formation needs.
//
// Do not reduce this to a trigger-only tap off the manager's replay edge.
// `BackprojectionOp.sample_points` is a `[live]` override, so a lattice that
// has not arrived takes `on_missing SafeDefault{}` -- the Params lattice, which
// for a polar aux is unset -- rather than waiting for one. Exact{} decides
// which sample matches, not whether the consumer blocks, so a tap races the
// formation and loses: measured on Kourou, aux_bp computed first and failed
// with "sample_geometry is required". A blocking rule in series cannot lose
// that race, which is why the lattice rides the profiles it belongs to.
//
// The lattice is built once in start() into operator-owned device memory, in
// row chunks rather than one host-side copy: 23k columns x a couple of
// thousand rows is a gigabyte-scale tensor. It is published once, on the first
// tick -- see compute() for why the consumer needs it only then.

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
#include <holoscan/logger/logger.hpp>

#include "radar_common/sdk/tensor_io.hpp"
#include "radar_common/tensor/max_byte_span.hpp"
#include "radar_common/tensor/tensor_utils.hpp"
#include "config.hpp"  // NOLINT(build/include_subdir)

namespace holoscan::radar::app::sar {

class AuxPointsSourceOp : public holoscan::Operator<> {
 public:
  explicit AuxPointsSourceOp(AuxPolarLattice lattice) : lattice_(lattice) {
    constexpr const char* kWho = "AuxPointsSourceOp";
    if (lattice_.num_v < 2 || lattice_.num_u < 2) {
      throw std::invalid_argument("AuxPointsSourceOp: lattice was not derived");
    }
    max_byte_span_ = byte_span_real_cube<double>({lattice_.num_v, lattice_.num_u, 3}, kWho);
  }

  ~AuxPointsSourceOp() override { cudaFree(points_); }

  void setup(OperatorSpec& spec) override {
    constexpr TensorRepresentation device{
        .memory_kind = MemoryKind::kCudaDevice, .dtype = std::nullopt, .rank = std::nullopt};
    spec.input(range_profiles).queue_depth(2U).expects_tensor(tensor_input(device));
    // Pass-through, not a pool. This operator is only here to put the lattice
    // and the profiles behind one producer; it does not touch the profiles, so
    // it forwards the sample it was handed rather than allocating a second
    // copy of the replay. On Kourou that pool was 11.4 GiB of reservation and
    // an aperture-sized device copy per image, both for a tensor it never read.
    spec.output(range_profiles_out).produces_tensor(tensor_output(device));
    // Depth 3 even though the lattice is emitted exactly once, on the first
    // tick. On the Kourou polar aux that is ~1.4 GiB per slot, so trimming it
    // looks attractive -- but depth 1 makes the first `aux_points` compute
    // fail with kInvalidArgument (measured, `Sar_autofocus_polar`): the
    // consumer still holds the loan when the pool is asked for the slot. If
    // this is worth reclaiming, it needs the holding side understood first,
    // not a smaller number.
    spec.output(sample_points).produces_tensor(pooled_tensor_output(device, 3U, max_byte_span_));
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnEach{range_profiles});
    return result;
  }

  void start() override {
    const int64_t nv = lattice_.num_v;
    const int64_t nu = lattice_.num_u;
    const std::size_t count = static_cast<std::size_t>(nv) * static_cast<std::size_t>(nu) * 3U;
    if (cudaMalloc(&points_, count * sizeof(double)) != cudaSuccess) {
      throw std::runtime_error("AuxPointsSourceOp: sample_points allocation failed");
    }

    // P(i, j) = apc + R_i * (cos t_j look + sin t_j uhat), so the per-column
    // unit direction is computed once and each row is a scale of it. Row
    // nv / 2 is the anchor range and column nu / 2 the anchor bearing.
    std::vector<double> dir(static_cast<std::size_t>(nu) * 3U);
    const double col0 = static_cast<double>(nu / 2);
    for (int64_t j = 0; j < nu; ++j) {
      const double t = (static_cast<double>(j) - col0) * lattice_.bearing_spacing;
      const double ct = std::cos(t);
      const double st = std::sin(t);
      for (int64_t k = 0; k < 3; ++k) {
        dir[static_cast<std::size_t>(j * 3 + k)] =
            ct * lattice_.look_hat[static_cast<std::size_t>(k)] +
            st * lattice_.uhat[static_cast<std::size_t>(k)];
      }
    }

    const int64_t chunk = std::max<int64_t>(1, std::min<int64_t>(nv, (int64_t{1} << 20) / nu + 1));
    std::vector<double> host(static_cast<std::size_t>(chunk * nu * 3));
    const double row0 = static_cast<double>(nv / 2);
    for (int64_t v0 = 0; v0 < nv; v0 += chunk) {
      const int64_t rows = std::min(chunk, nv - v0);
      for (int64_t i = 0; i < rows; ++i) {
        const double r =
            lattice_.range0 + (static_cast<double>(v0 + i) - row0) * lattice_.range_spacing;
        double* out = host.data() + static_cast<std::size_t>(i * nu * 3);
        for (int64_t j = 0; j < nu; ++j) {
          const double* d = dir.data() + static_cast<std::size_t>(j * 3);
          double* p = out + static_cast<std::size_t>(j * 3);
          p[0] = lattice_.apc[0] + r * d[0];
          p[1] = lattice_.apc[1] + r * d[1];
          p[2] = lattice_.apc[2] + r * d[2];
        }
      }
      const std::size_t bytes = static_cast<std::size_t>(rows * nu * 3) * sizeof(double);
      if (cudaMemcpy(points_ + v0 * nu * 3, host.data(), bytes, cudaMemcpyHostToDevice) !=
          cudaSuccess) {
        throw std::runtime_error("AuxPointsSourceOp: sample_points H2D failed");
      }
    }
    HOLOSCAN_LOG_INFO("AuxPointsSourceOp: staged {}x{} range-bearing points ({:.2f} GiB)", nv, nu,
                      static_cast<double>(count) * 8.0 / (1024.0 * 1024.0 * 1024.0));
  }

  void stop() override {
    cudaFree(points_);
    points_ = nullptr;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    // The whole sample, not just the tensor: forwarding it keeps the replay's
    // capture time, which is what the lattice below is stamped to match.
    auto profiles = range_profiles.receive();
    if (!profiles) {
      return make_unexpected(std::move(profiles).error());
    }
    cudaStream_t stream = context.cuda_stream().get();
    if (stream == nullptr) {
      return make_unexpected(Error{ErrorCode::kNotSupported, "AuxPointsSourceOp: no CUDA stream"});
    }
    // Emit the lattice before the profiles. The profiles are a trigger port on
    // the consuming BackprojectionOp and `sample_points` is a `[live]` override
    // joined with `on_missing SafeDefault{}`, so the activation does not wait
    // for the lattice: whatever has not been emitted by the time the trigger
    // set completes is simply missing for that tick.
    //
    // Once, not every tick. The lattice is fixed for the run and the consumer
    // caches the first one it is given, holding it across accumulator resets
    // (it only drops a cached lattice when a `sample_geometry` override
    // arrives, which a polar aux never sends). Re-sending it cost a
    // gigabyte-scale device comparison on every tick to conclude nothing had
    // changed, and under a sliced replay it would also publish one lattice per
    // batch into a queue sized for far fewer.
    if (!lattice_sent_) {
      const std::vector<int64_t> shape{lattice_.num_v, lattice_.num_u, 3};
      HOLOSCAN_TRY(emit_required_tensor(
          sample_points, context, "AuxPointsSourceOp.sample_points", false, [&] {
            return prepare_device_tensor(
                sample_points, context, shape, real_dtype<double>(),
                [&](void* destination, std::size_t bytes) -> expected<void, Error> {
                  if (cudaMemcpyAsync(destination, points_, bytes, cudaMemcpyDeviceToDevice,
                                      stream) != cudaSuccess) {
                    return make_unexpected(
                        Error{ErrorCode::kFailure, "AuxPointsSourceOp: sample_points copy failed"});
                  }
                  return {};
                });
          }));
      lattice_sent_ = true;
      // The staging buffer is dead once the copy above has run: the consumer
      // takes its own device copy of the lattice, and nothing re-sends. At
      // Kourou polar extents that is ~1.4 GiB held for the rest of the run.
      // Sync first -- the copy is async on `stream`, so freeing on enqueue
      // would be a use-after-free.
      if (cudaStreamSynchronize(stream) != cudaSuccess) {
        return make_unexpected(
            Error{ErrorCode::kFailure, "AuxPointsSourceOp: sample_points sync failed"});
      }
      cudaFree(points_);
      points_ = nullptr;
    }
    return emit_required_result(context, "AuxPointsSourceOp.range_profiles_out", false, [&] {
      return range_profiles_out.emit(*profiles);
    });
  }

  Input<Tensor> range_profiles{"range_profiles"};
  Output<Tensor> range_profiles_out{"range_profiles_out"};
  Output<Tensor> sample_points{"sample_points"};

 private:
  AuxPolarLattice lattice_;
  std::size_t max_byte_span_{0U};
  double* points_{nullptr};
  bool lattice_sent_{false};
};

}  // namespace holoscan::radar::app::sar
