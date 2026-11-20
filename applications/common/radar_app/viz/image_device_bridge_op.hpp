// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cuda_runtime_api.h>

#include <array>
#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/core/tensor_contract.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/sensor_io/provenance.hpp>
#include <holoscan/sensor_io/schemas/image_schema_traits.hpp>

#include "radar_app/viz/frame_viewport.hpp"

namespace holoscan::radar::app::viz {

// An explicit host boundary joins GPUs even when CUDA peer access is unavailable.
// Each operator has only one CUDA device obligation, as required by HSDK 5.
// Instantiated only at image connections whose CUDA devices differ.
template <bool Upload>
class ImageDeviceBridgeOp final : public Operator<> {
 public:
  ImageDeviceBridgeOp(std::int32_t width, std::int32_t height,
                      std::shared_ptr<FrameViewport> viewport = {})
  : width_(width), height_(height), viewport_(std::move(viewport)) {}

  void setup(OperatorSpec& spec) override {
    const TensorBounds bounds{.max_rank = 3U,
                              .max_byte_span = static_cast<std::uint64_t>(width_) * height_ * 4U};
    const auto format = [](MemoryKind kind) {
      return TensorRepresentation{.memory_kind = kind,
                                  .dtype = DLDataType{kDLUInt, 8U, 1U},
                                  .rank = 3U,
                                  .strides = TensorStridePolicy::kContiguousOnly};
    };
    spec.input(image_in)
        .tensor_field(schema::ImageTensorFields::data)
        .expects_tensor(TensorInputSpec{
            .representation = format(Upload ? MemoryKind::kHost : MemoryKind::kCudaDevice),
            .bounds = bounds});
    spec.output(image_out)
        .tensor_field(schema::ImageTensorFields::data)
        .produces_tensor(TensorOutputSpec{
            .representation = format(Upload ? MemoryKind::kCudaDevice : MemoryKind::kHost),
            .bounds = bounds,
            .storage = TensorOutputStorage::kRuntimePool});
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnEach{image_in});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    auto sample = image_in.receive();
    if (!sample)
      return make_unexpected(std::move(sample).error());
    const auto& image = sample->data;
    if (image.width < 1 || image.width > width_ || image.height < 1 || image.height > height_ ||
        !image.data || image.data->ndim() != 3 || image.data->shape()[0] != image.height ||
        image.data->shape()[1] != image.width || image.data->shape()[2] != 4) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument, "image transfer shape mismatch"});
    }
    auto descriptor = image;
    if (viewport_) {
      const auto drawn = viewport_->drawn(sample->metadata.capture_time().timestamp_ns);
      // A display stall may outlive the bounded layout history. Drop that
      // stale display frame rather than guessing at a newer frame's extent.
      if (!drawn)
        return {};
      const auto [width, height] = *drawn;
      descriptor.width = std::min(width, image.width);
      descriptor.height = std::min(height, image.height);
    }
    const std::array<std::int64_t, 3U> shape{descriptor.height, descriptor.width, 4};
    auto loan = image_out.allocate_tensor(
        TensorLoanRequest{.shape = shape, .dtype = DLDataType{kDLUInt, 8U, 1U}});
    if (!loan)
      return make_unexpected(std::move(loan).error());
    const auto stream = context.cuda_stream();
    auto writer = [&] {
      if constexpr (Upload)
        return loan->write(stream);
      else
        return loan->write_host();
    }();
    if (!writer)
      return make_unexpected(std::move(writer).error());
    const auto status =
        cudaMemcpy2DAsync(writer->data(), descriptor.width * 4U, image.data->data(),
                          image.width * 4U, descriptor.width * 4U, descriptor.height,
                          Upload ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost, stream.get());
    // Publish host bytes only after D2H completes, and keep the H2D source
    // loan alive until its read completes. This cold topology choice adds
    // synchronization only to the optional cross-GPU image boundary.
    if (status != cudaSuccess || cudaStreamSynchronize(stream.get()) != cudaSuccess) {
      return make_unexpected(Error{ErrorCode::kFailure, "image device transfer failed"});
    }
    auto committed = std::move(*writer).commit();
    if (!committed)
      return make_unexpected(std::move(committed).error());
    return image_out.emit_tensor(std::move(*loan), descriptor, sensor_io::derive_from(*sample));
  }

  Input<schema::ImageT> image_in{"image_in"};
  Output<schema::ImageT> image_out{"image_out"};

 private:
  std::int32_t width_;
  std::int32_t height_;
  std::shared_ptr<FrameViewport> viewport_;
};

using ImageDownloadOp = ImageDeviceBridgeOp<false>;
using ImageUploadOp = ImageDeviceBridgeOp<true>;

// Transfer images without giving either operator CUDA obligations on two GPUs.
inline void connect_image_devices(Graph& graph, CompileOptions& options,
                                  PartitionHandle& source_partition,
                                  PartitionHandle& target_partition,
                                  const Output<schema::ImageT>& source,
                                  const Input<schema::ImageT>& target, const std::string& prefix,
                                  std::int32_t width, std::int32_t height, DeviceId source_device,
                                  DeviceId target_device, const ConnectionOptions& queue) {
  if (source_device == target_device) {
    graph.add_flow(source, target, queue);
    return;
  }
  auto download = graph.op<ImageDownloadOp>(prefix + "_download", width, height);
  auto upload = graph.op<ImageUploadOp>(prefix + "_upload", width, height);
  source_partition.add(download);
  target_partition.add(upload);
  options.deployment.bind_tensor_output_device(TensorOutputDevicePlacement{
      .operator_path = prefix + "_upload", .output_port = "image_out", .device = target_device});
  graph.add_flow(source, download->image_in, queue);
  graph.add_flow(download->image_out, upload->image_in, queue);
  graph.add_flow(upload->image_out, target, queue);
}

}  // namespace holoscan::radar::app::viz
