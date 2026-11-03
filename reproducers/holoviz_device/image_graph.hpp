// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cuda_runtime_api.h>
#include <dlpack/dlpack.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/core/tensor_contract.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/sensor_io/schemas/image_schema_traits.hpp>
#include <holoscan/time/realtime_clock.hpp>
#include <holoscan/visualization/holoviz_op.hpp>

namespace repro {
namespace hs = holoscan;
namespace viz = holoscan::visualization;
constexpr std::uint32_t kSize = 64;
inline std::atomic<int> rendered{0};
inline std::atomic<int> emitted{0};

// One bounded GPU image pool; no radar code, CUDA kernels, or constructor-time GPU allocation.
class ImageSource final : public hs::Operator<> {
 public:
  void setup(hs::OperatorSpec& spec) override {
    spec.output(image)
        .tensor_field(hs::schema::ImageTensorFields::data)
        .produces_tensor(hs::TensorOutputSpec{
            .representation = {.memory_kind = hs::MemoryKind::kCudaDevice,
                               .dtype = DLDataType{kDLUInt, 8U, 1U},
                               .rank = 3U},
            .bounds = hs::TensorBounds{.max_rank = 3U, .max_byte_span = kSize * kSize * 4U},
            .storage = hs::TensorOutputStorage::kRuntimePool});
  }

  [[nodiscard]] hs::Contract contract() const override {
    hs::Contract result;
    result.trigger(hs::OnClock{.period = std::chrono::milliseconds{100}});
    return result;
  }

  [[nodiscard]] hs::expected<void, hs::Error> compute(hs::ExecutionContext& context) override {
    const std::array<std::int64_t, 3> shape{kSize, kSize, 4};
    HOLOSCAN_TRY(auto loan, image.allocate_tensor(hs::TensorLoanRequest{
                                .shape = shape, .dtype = DLDataType{kDLUInt, 8U, 1U}}));
    HOLOSCAN_TRY(auto writer, loan.write(context.cuda_stream()));
    if (cudaMemsetAsync(writer.data(), 255, kSize * kSize * 4U, context.cuda_stream().get()) !=
        cudaSuccess) {
      return hs::make_unexpected(hs::Error{hs::ErrorCode::kFailure, "image fill failed"});
    }
    HOLOSCAN_TRY(std::move(writer).commit());
    hs::schema::ImageT descriptor;
    descriptor.width = kSize;
    descriptor.height = kSize;
    descriptor.encoding = hs::schema::ImageEncoding_RGBA8;
    HOLOSCAN_TRY(image.emit_tensor(std::move(loan), descriptor,
                                   hs::EmitOptions{.capture_time = context.activation_time()}));
    ++emitted;
    return {};
  }

  hs::Output<hs::schema::ImageT> image{"image"};
};

// A real output consumer proves that the offscreen renderer executed, rather than just starting.
class CaptureSink final : public hs::Operator<> {
 public:
  void setup(hs::OperatorSpec& spec) override { spec.input(image); }
  [[nodiscard]] hs::Contract contract() const override {
    hs::Contract result;
    result.trigger(hs::OnEach{image});
    return result;
  }
  [[nodiscard]] hs::expected<void, hs::Error> compute(hs::ExecutionContext&) override {
    HOLOSCAN_TRY(auto sample, image.receive());
    if (!sample.data.data)
      return hs::make_unexpected(hs::Error{hs::ErrorCode::kFailure, "capture has no tensor"});
    ++rendered;
    return {};
  }
  hs::Input<hs::schema::ImageT> image{"image"};
};

inline int run(bool window, int device) {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess || device < 0 || device >= count)
    throw std::invalid_argument("DEVICE is outside the process-visible CUDA device range");
  if (cudaSetDevice(device) != cudaSuccess)
    throw std::runtime_error("cudaSetDevice failed");
  hs::Graph graph{"holoviz-device-repro"};
  auto source = graph.op<ImageSource>("source");
  viz::LayerStack layers;
  layers.add_image_layer("image", viz::ImageLayerDesc{.format = viz::ImageFormat::kR8G8B8A8Unorm,
                                                      .max_width = kSize,
                                                      .max_height = kSize});
  const viz::SinkDesc sink =
      window ? viz::SinkDesc{viz::WindowSinkDesc{
                   .title = "HoloViz device repro", .width = kSize, .height = kSize}}
             : viz::SinkDesc{viz::CudaSinkDesc{.width = kSize, .height = kSize}};
  auto renderer = graph.op<viz::HolovizOp>("renderer", std::move(layers), sink);
  graph.partition("producer").add(source);
  graph.partition("render").add(renderer);
  graph.add_flow(
      source->image, renderer->image_layer_input(0),
      hs::ConnectionOptions{.queue_policy = hs::QueuePolicy::kDropOldest, .queue_depth = 1U});
  hs::CompileOptions options;
  options.deployment.bind_tensor_output_device(hs::TensorOutputDevicePlacement{
      .operator_path = "source", .output_port = "image", .device = hs::DeviceId{device}});
  if (!window) {
    auto capture = graph.op<CaptureSink>("capture");
    graph.partition("consumer").add(capture);
    graph.add_flow(renderer->output(), capture->image);
    // This binding controls the pool/activation device; GA does not use it to select the renderer.
    options.deployment.bind_tensor_output_device(hs::TensorOutputDevicePlacement{
        .operator_path = "renderer", .output_port = "output", .device = hs::DeviceId{device}});
  }
  graph.set_default_clock(graph.add_clock<hs::RealtimeClock>("clock"));
  std::cout << "Before compile: sink=" << (window ? "window" : "cuda")
            << ", tensor device=" << device << '\n'
            << std::flush;
  const auto plan = hs::compile(graph, std::move(options));
  if (!plan.ok()) {
    std::cerr << plan.json() << '\n';
    return 1;
  }
  auto session = hs::run_async(plan);
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{2};
  while (session.termination() == hs::RunTermination::kRunning && session.diagnostics().empty() &&
         std::chrono::steady_clock::now() < deadline)
    std::this_thread::sleep_for(std::chrono::milliseconds{20});
  session.request_stop();
  session.wait();
  for (const auto& diagnostic : session.diagnostics())
    std::cerr << hs::format_diagnostic(diagnostic, plan.graph()) << '\n';
  std::cout << "Emitted=" << emitted << ", captured=" << rendered << '\n';
  return session.diagnostics().empty() && session.termination() == hs::RunTermination::kStopped &&
                 emitted > 0 && (window || rendered > 0)
             ? 0
             : 1;
}
}  // namespace repro
