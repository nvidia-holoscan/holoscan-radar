// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
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

#include "radar_app/sar/mosaic_sample.hpp"
#include "radar_common/descriptors/sample_geometry.hpp"
#include "radar_common/sdk/tensor_io.hpp"
#include "radar_schemas/sample_geometry_bridge.hpp"

namespace holoscan::radar::app::sar {
// Holds tile or scroll windows on the same parent lattice. Geometry is
// published with every BackprojectionOp image, so no tick-local app state
// needs to guess which image a geometry describes.
//
// The accumulator is resident and always full resolution; `image_out` carries
// a view of it. `crop_nu` narrows that view to a parent-u window (the walking
// range-window edge is not scene), and `display_max_side` decimates it to a
// pixel budget. Both default to off, which republishes the whole parent.
// Sizing the view matters because the pool holds several slots of it and a
// strip is large: at 1.03 m on the Imperial collect the parent is 100618 x
// 15077, so one slot is 22.6 GiB and no pool of them fits on a 96 GiB card.
// Decimating to a display budget is what makes a high-resolution strip
// viewable at all; the full-resolution pixels never leave this operator.
// The decimation multi-looks (see `sample_mosaic_rect`) and so publishes
// amplitude with zero phase, because a strip's decimation is steep enough
// that subsampled speckle is noise. A view at factor 1 is a plain copy and
// keeps its phase, cropped or not.
// `T` sizes the declared output pool only: on the Imperial parent that span is
// 3.62 GiB at complex128 against 1.81 at complex64, and six declared slots make
// the difference 10.8 GiB. The accumulator and the publication stay dtype-
// agnostic at runtime -- the incoming dtype is latched and honoured either way,
// so `T` being wrong costs a rejected tile rather than a silent misread. It is
// deliberately not defaulted: pass the app's sample precision, which is single
// unless the collect asked for double, so the pool tracks what actually arrives.
template <typename T>
class SarMosaicOp : public Operator<> {
 public:
  SarMosaicOp(SampleGeometry parent, bool loop, int64_t crop_u0 = 0, int64_t crop_nu = 0,
              int64_t display_max_side = 0)
  : parent_(std::move(parent)),
    loop_(loop),
    crop_u0_(crop_u0),
    crop_nu_(crop_nu),
    display_max_side_(display_max_side) {
    if (!parent_.is_valid())
      throw std::invalid_argument("SarMosaicOp: invalid parent geometry");
    if (crop_u0_ < 0 || crop_nu_ < 0 || display_max_side_ < 0 ||
        crop_u0_ + crop_nu_ > parent_.num_u())
      throw std::invalid_argument("SarMosaicOp: invalid parent-u crop or display budget");
    if (crop_nu_ == 0) {
      crop_u0_ = 0;
      crop_nu_ = parent_.num_u();
    }
    factor_ = 1;
    if (display_max_side_ > 0) {
      const int64_t longest = std::max(parent_.num_v(), crop_nu_);
      if (longest > display_max_side_)
        factor_ = (longest + display_max_side_ - 1) / display_max_side_;
      // A factor wider than the shorter axis leaves that axis under one whole
      // box. Capping keeps every box inside the canvas; clamping the output
      // extent up to 1 instead does not, because the kernel still reads
      // `factor_` rows and columns for the sample it was told to write, off
      // the end of the short axis. The budget then binds on the long axis
      // only, which is the honest outcome for a view that degenerate.
      factor_ = std::min(factor_, std::min(parent_.num_v(), crop_nu_));
    }
    out_v_ = parent_.num_v() / factor_;
    out_u_ = crop_nu_ / factor_;
  }
  ~SarMosaicOp() override {
    if (canvas_ != nullptr)
      (void)cudaFree(canvas_);
  }
  void setup(OperatorSpec& spec) override {
    constexpr TensorRepresentation device{
        .memory_kind = MemoryKind::kCudaDevice, .dtype = std::nullopt, .rank = std::nullopt};
    spec.input(image).queue_depth(8U).expects_tensor(tensor_input(device));
    spec.input(sample_geometry).queue_depth(8U);
    spec.output(image_out).produces_tensor(pooled_tensor_output(
        device, 2U,
        static_cast<uint64_t>(out_v_) * static_cast<uint64_t>(out_u_) * 2U * sizeof(T)));
  }
  [[nodiscard]] Contract contract() const override {
    Contract result;
    result
        .trigger(OnAligned{.ports = ports(image, sample_geometry),
                           .window = std::chrono::nanoseconds{0}})
        .input(image, Exact{})
        .input(sample_geometry, Exact{});
    return result;
  }
  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    HOLOSCAN_TRY(auto tile, receive_tensor(image, "SarMosaicOp"));
    HOLOSCAN_TRY(auto wire, sample_geometry.receive_data());
    HOLOSCAN_TRY(auto window, from_wire(wire));
    const auto shape = tile.shape();
    if (shape.size() != 2U || shape[0] != window.num_v() || shape[1] != window.num_u())
      return make_unexpected(
          Error{ErrorCode::kInvalidArgument, "SarMosaicOp: image/window shape mismatch"});
    const auto dtype = tile.dtype();
    if (dtype.code != kDLComplex || (dtype.bits != 64U && dtype.bits != 128U) || dtype.lanes != 1U)
      return make_unexpected(
          Error{ErrorCode::kInvalidArgument, "SarMosaicOp: expected complex64 or complex128"});
    const int64_t pv = parent_.num_v(), pu = parent_.num_u();
    const std::size_t elem = static_cast<std::size_t>(dtype.bits / 8U);
    const std::size_t bytes = static_cast<std::size_t>(pv * pu) * elem;
    const auto stream = context.cuda_stream().get();
    if (canvas_ == nullptr) {
      if (cudaMalloc(&canvas_, bytes) != cudaSuccess)
        return make_unexpected(
            Error{ErrorCode::kResourceExhausted, "SarMosaicOp: canvas allocation failed"});
      dtype_ = dtype;
      if (cudaMemsetAsync(canvas_, 0, bytes, stream) != cudaSuccess)
        return make_unexpected(Error{ErrorCode::kFailure, "SarMosaicOp: canvas clear failed"});
    } else if (dtype.code != dtype_.code || dtype.bits != dtype_.bits ||
               dtype.lanes != dtype_.lanes) {
      return make_unexpected(
          Error{ErrorCode::kInvalidArgument, "SarMosaicOp: image dtype changed"});
    }
    const auto& po = parent_.origin();
    const auto& wo = window.origin();
    const double dx = wo[0] - po[0], dy = wo[1] - po[1], dz = wo[2] - po[2];
    const auto& av = parent_.axis_v();
    const auto& au = parent_.axis_u();
    const double v = (dx * av[0] + dy * av[1] + dz * av[2]) / parent_.spacing()[0];
    const double u = (dx * au[0] + dy * au[1] + dz * au[2]) / parent_.spacing()[1];
    const int64_t v0 = static_cast<int64_t>(std::llround(v));
    const int64_t u0 = static_cast<int64_t>(std::llround(u));
    if (std::abs(v - static_cast<double>(v0)) > 0.25 ||
        std::abs(u - static_cast<double>(u0)) > 0.25 || v0 < 0 || u0 < 0 || v0 + shape[0] > pv ||
        u0 + shape[1] > pu) {
      return make_unexpected(
          Error{ErrorCode::kInvalidArgument, "SarMosaicOp: window outside parent lattice"});
    }
    if (loop_ && last_v0_ >= 0 && v0 < last_v0_ &&
        cudaMemsetAsync(canvas_, 0, bytes, stream) != cudaSuccess) {
      return make_unexpected(Error{ErrorCode::kFailure, "SarMosaicOp: wrap clear failed"});
    }
    auto* dst = static_cast<char*>(canvas_) + static_cast<std::size_t>(v0 * pu + u0) * elem;
    if (cudaMemcpy2DAsync(
            dst, static_cast<std::size_t>(pu) * elem, tile.data(),
            static_cast<std::size_t>(shape[1]) * elem, static_cast<std::size_t>(shape[1]) * elem,
            static_cast<std::size_t>(shape[0]), cudaMemcpyDeviceToDevice, stream) != cudaSuccess) {
      return make_unexpected(Error{ErrorCode::kFailure, "SarMosaicOp: blit failed"});
    }
    last_v0_ = v0;
    return emit_required_tensor(image_out, context, "SarMosaicOp.image_out", false, [&] {
      return prepare_device_tensor(
          image_out, context, std::vector<int64_t>{out_v_, out_u_}, dtype,
          [&](void* output, std::size_t size) -> expected<void, Error> {
            if (size != static_cast<std::size_t>(out_v_ * out_u_) * elem)
              return make_unexpected(Error{ErrorCode::kFailure, "SarMosaicOp: view size mismatch"});
            // Keyed on the factor alone. `sample_mosaic_rect` multi-looks
            // unconditionally, so it publishes amplitude with a zeroed
            // imaginary part even for the single sample a factor-1 box holds;
            // requiring `crop_nu_ == pu` here as well sent an undecimated crop
            // down that path and discarded its phase on the way to the dump.
            // A crop is a strided copy, and the whole parent is the same copy
            // with `crop_u0_ == 0` and matching pitches.
            if (factor_ == 1) {
              const std::size_t row = static_cast<std::size_t>(out_u_) * elem;
              if (cudaMemcpy2DAsync(
                      output, row,
                      static_cast<const char*>(canvas_) + static_cast<std::size_t>(crop_u0_) * elem,
                      static_cast<std::size_t>(pu) * elem, row, static_cast<std::size_t>(out_v_),
                      cudaMemcpyDeviceToDevice, stream) != cudaSuccess)
                return make_unexpected(
                    Error{ErrorCode::kFailure, "SarMosaicOp: parent copy failed"});
              return {};
            }
            if (sample_mosaic_rect(canvas_, pu, 0, crop_u0_, output, out_v_, out_u_, factor_,
                                   dtype.bits, stream) != cudaSuccess)
              return make_unexpected(Error{ErrorCode::kFailure, "SarMosaicOp: parent view failed"});
            return {};
          });
    });
  }
  Input<Tensor> image{"image"};
  Input<WireSampleGeometry> sample_geometry{"sample_geometry"};
  Output<Tensor> image_out{"image_out"};

 private:
  SampleGeometry parent_;
  bool loop_{false};
  int64_t crop_u0_{0}, crop_nu_{0}, display_max_side_{0};
  int64_t factor_{1}, out_v_{0}, out_u_{0};
  void* canvas_{nullptr};
  DLDataType dtype_{};
  int64_t last_v0_{-1};
};
}  // namespace holoscan::radar::app::sar
