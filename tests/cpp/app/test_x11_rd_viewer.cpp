// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// [EA HSDK5 ONLY] Graph smoke: synthetic RD → RdMapToImageOp → X11RdViewerOp.
// Skips when DISPLAY / XOpenDisplay is unavailable.

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <complex>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/expected_macros.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "radar_app/viz/rd_map_to_image_op.hpp"
#include "radar_app/viz/x11_display.hpp"
#include "radar_app/viz/x11_rd_viewer_op.hpp"
#include "radar_common/descriptors/max_tensor_extents.hpp"
#include "radar_common/sdk/tensor_io.hpp"
#include "radar_common/tensor/dlpack_buffer.hpp"
#include "radar_common/tensor/tensor_utils.hpp"
#include "utils/tensor_fixtures.hpp"

namespace holoscan::radar::app {
namespace {

constexpr auto kTick = std::chrono::milliseconds{20};
constexpr int kFrames = 8;
constexpr int64_t kDoppler = 64;
constexpr int64_t kRange = 128;

struct SmokeBank {
  static void reset(std::shared_ptr<holoscan::Tensor> frame, int count) {
    const std::lock_guard lock(mutex);
    input = std::move(frame);
    remaining = count;
    presented = 0;
  }

  static inline std::mutex mutex;
  static inline std::condition_variable ready;
  static inline std::shared_ptr<holoscan::Tensor> input;
  static inline int remaining{0};
  static inline int presented{0};
};

class SmokeSourceOp final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    constexpr holoscan::TensorRepresentation kDevice{
        .memory_kind = holoscan::MemoryKind::kCudaDevice,
        .dtype = std::nullopt,
        .rank = std::nullopt,
    };
    const std::size_t span = tensor_byte_span(
        std::vector<int64_t>{kDoppler, kRange}, complex_dtype<float>());
    spec.output(out, "out")
        .produces_tensor(holoscan::radar::pooled_tensor_output(
            kDevice, 2U, span));
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = kTick});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    std::shared_ptr<holoscan::Tensor> frame;
    {
      const std::lock_guard lock(SmokeBank::mutex);
      if (SmokeBank::remaining <= 0 || SmokeBank::input == nullptr) {
        return {};
      }
      --SmokeBank::remaining;
      frame = SmokeBank::input;
    }
    auto loan = out.allocate_tensor(holoscan::TensorLoanRequest{
        .shape = frame->shape(), .dtype = frame->dtype()});
    if (!loan) {
      return holoscan::make_unexpected(std::move(loan).error());
    }
    auto writer = loan->write(context.cuda_stream());
    if (!writer) {
      return holoscan::make_unexpected(std::move(writer).error());
    }
    if (cudaMemcpyAsync(writer->data(), frame->data(), writer->byte_size(),
                        cudaMemcpyDeviceToDevice,
                        context.cuda_stream().get()) != cudaSuccess) {
      return holoscan::make_unexpected(holoscan::Error{
          holoscan::ErrorCode::kFailure, "SmokeSourceOp: D2D failed"});
    }
    auto committed = std::move(*writer).commit();
    if (!committed) {
      return holoscan::make_unexpected(std::move(committed).error());
    }
    return out.emit(
        std::move(*loan),
        holoscan::EmitOptions{.capture_time = context.activation_time()});
  }

  holoscan::Output<holoscan::Tensor> out{"out"};
};

class PresentCountOp final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    constexpr holoscan::TensorRepresentation kDevice{
        .memory_kind = holoscan::MemoryKind::kCudaDevice,
        .dtype = std::nullopt,
        .rank = std::nullopt,
    };
    spec.input(in, "in").queue_depth(8U).expects_tensor(holoscan::radar::tensor_input(kDevice));
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnEach{in});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext&) override {
    auto sample = in.receive_data();
    if (!sample) {
      return holoscan::make_unexpected(std::move(sample).error());
    }
    {
      const std::lock_guard lock(SmokeBank::mutex);
      ++SmokeBank::presented;
    }
    SmokeBank::ready.notify_all();
    return {};
  }

  holoscan::Input<holoscan::Tensor> in{"in"};
};

std::shared_ptr<holoscan::Tensor> make_tone_rd() {
  std::vector<std::complex<float>> host(
      static_cast<size_t>(kDoppler * kRange));
  for (int64_t d = 0; d < kDoppler; ++d) {
    for (int64_t r = 0; r < kRange; ++r) {
      const float phase =
          0.15F * static_cast<float>(r) + 0.07F * static_cast<float>(d);
      host[static_cast<size_t>(d * kRange + r)] = {
          std::cos(phase), std::sin(phase)};
    }
  }
  return testing::to_device_tensor(host, {kDoppler, kRange},
                                   complex_dtype<float>());
}

TEST(X11RdViewer, LiveRdMapSmoke) {
  if (!x11_display_available()) {
    GTEST_SKIP() << "no X11 display (DISPLAY / XOpenDisplay)";
  }

  auto cube = make_tone_rd();
  SmokeBank::reset(cube, kFrames);
  X11RdViewerOp::reset_presented_count();

  MaxTensorExtents extents;
  extents.max_doppler_bins = kDoppler;
  extents.max_range_bins = kRange;

  holoscan::Graph graph{"x11-rd-smoke"};
  const auto src = graph.op<SmokeSourceOp>("src");
  const auto to_image = graph.op<RdMapToImageOp>("to_image", extents);
  const auto viewer = graph.op<X11RdViewerOp>("viewer");
  const auto counter = graph.op<PresentCountOp>("counter");
  const holoscan::ConnectionOptions q{.queue_depth = 2U};
  graph.add_flow(src->out, to_image->rd_maps, q);
  graph.add_flow(to_image->rd_image, viewer->rd_image, q);
  graph.add_flow(to_image->rd_image, counter->in, q);
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

  holoscan::CompileOptions options;
  options.deployment.bind_tensor_output_device(
      holoscan::TensorOutputDevicePlacement{
          .operator_path = "src",
          .output_port = "out",
          .device = holoscan::DeviceId{0},
      });
  options.deployment.bind_tensor_output_device(
      holoscan::TensorOutputDevicePlacement{
          .operator_path = "to_image",
          .output_port = "rd_image",
          .device = holoscan::DeviceId{0},
      });

  const holoscan::ExecutionPlan plan =
      holoscan::compile(graph, std::move(options));
  ASSERT_TRUE(plan.ok()) << plan.json();

  holoscan::RunSession session = holoscan::run_async(plan);
  {
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds{10};
    std::unique_lock lock(SmokeBank::mutex);
    while ((SmokeBank::presented < kFrames ||
            X11RdViewerOp::presented_count() < kFrames) &&
           session.diagnostics().empty() &&
           std::chrono::steady_clock::now() < deadline) {
      SmokeBank::ready.wait_for(lock, std::chrono::milliseconds{25});
    }
  }
  session.request_stop();
  session.wait();

  ASSERT_TRUE(session.diagnostics().empty());
  {
    const std::lock_guard lock(SmokeBank::mutex);
    EXPECT_GE(SmokeBank::presented, kFrames);
  }
  EXPECT_GE(X11RdViewerOp::presented_count(), kFrames);
}

// ---- Overlay wiring smoke -------------------------------------------------
// The viewer's overlay ports are wired per config, and on EA a wired port must
// appear in contract() while an unwired one must not. This exercises the full
// eight-port OnAligned/Exact path (rd_image + detections + bins + cluster
// labels/representatives + the boresight panel's aoa_estimates + the header
// variant label) that the SIMO / MIMO / aoa_estimators apps build when
// clustering is on.

constexpr int64_t kDetections = 4;
constexpr int64_t kClusters = 2;

struct OverlayBank {
  static inline std::shared_ptr<holoscan::Tensor> detection_idx;
  static inline std::shared_ptr<holoscan::Tensor> labels;
  static inline std::shared_ptr<holoscan::Tensor> rep_idx;
  static inline std::shared_ptr<holoscan::Tensor> range_bins;
  static inline std::shared_ptr<holoscan::Tensor> velocity_bins;
  static inline std::shared_ptr<holoscan::Tensor> aoa_estimates;
  static inline std::shared_ptr<holoscan::Tensor> label_idx;
};

class OverlaySourceOp final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    declare(spec, rd_out, "rd_out", {kDoppler, kRange}, complex_dtype<float>());
    declare(spec, idx_out, "idx_out", {kDetections, 2}, kIndexDtype);
    declare(spec, labels_out, "labels_out", {kDetections}, kIndexDtype);
    declare(spec, rep_out, "rep_out", {kClusters, 2}, kIndexDtype);
    declare(spec, range_out, "range_out", {kRange}, real_dtype<float>());
    declare(spec, velocity_out, "velocity_out", {kDoppler},
            real_dtype<float>());
    declare(spec, aoa_out, "aoa_out", {kDetections, 2}, real_dtype<float>());
    declare(spec, label_out, "label_out", {1}, kIndexDtype);
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = kTick});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    std::shared_ptr<holoscan::Tensor> frame;
    {
      const std::lock_guard lock(SmokeBank::mutex);
      if (SmokeBank::remaining <= 0 || SmokeBank::input == nullptr) {
        return {};
      }
      --SmokeBank::remaining;
      frame = SmokeBank::input;
    }
    HOLOSCAN_RETURN_IF_ERROR(emit_copy(rd_out, frame, context));
    HOLOSCAN_RETURN_IF_ERROR(
        emit_copy(idx_out, OverlayBank::detection_idx, context));
    HOLOSCAN_RETURN_IF_ERROR(emit_copy(labels_out, OverlayBank::labels,
                                       context));
    HOLOSCAN_RETURN_IF_ERROR(emit_copy(rep_out, OverlayBank::rep_idx, context));
    HOLOSCAN_RETURN_IF_ERROR(
        emit_copy(range_out, OverlayBank::range_bins, context));
    HOLOSCAN_RETURN_IF_ERROR(
        emit_copy(velocity_out, OverlayBank::velocity_bins, context));
    HOLOSCAN_RETURN_IF_ERROR(
        emit_copy(aoa_out, OverlayBank::aoa_estimates, context));
    HOLOSCAN_RETURN_IF_ERROR(
        emit_copy(label_out, OverlayBank::label_idx, context));
    return {};
  }

  holoscan::Output<holoscan::Tensor> rd_out{"rd_out"};
  holoscan::Output<holoscan::Tensor> idx_out{"idx_out"};
  holoscan::Output<holoscan::Tensor> labels_out{"labels_out"};
  holoscan::Output<holoscan::Tensor> rep_out{"rep_out"};
  holoscan::Output<holoscan::Tensor> range_out{"range_out"};
  holoscan::Output<holoscan::Tensor> velocity_out{"velocity_out"};
  holoscan::Output<holoscan::Tensor> aoa_out{"aoa_out"};
  holoscan::Output<holoscan::Tensor> label_out{"label_out"};

 private:
  static constexpr DLDataType kIndexDtype{kDLInt, 64, 1};

  static void declare(holoscan::OperatorSpec& spec,
                      holoscan::Output<holoscan::Tensor>& port,
                      const char* name, const std::vector<int64_t>& shape,
                      DLDataType dtype) {
    constexpr holoscan::TensorRepresentation kDevice{
        .memory_kind = holoscan::MemoryKind::kCudaDevice,
        .dtype = std::nullopt,
        .rank = std::nullopt,
    };
    spec.output(port, name)
        .produces_tensor(holoscan::radar::pooled_tensor_output(
            kDevice,
            static_cast<uint32_t>(shape.size()),
            tensor_byte_span(shape, dtype)));
  }

  static holoscan::expected<void, holoscan::Error> emit_copy(
      holoscan::Output<holoscan::Tensor>& port,
      const std::shared_ptr<holoscan::Tensor>& frame,
      holoscan::ExecutionContext& context) {
    auto loan = port.allocate_tensor(holoscan::TensorLoanRequest{
        .shape = frame->shape(), .dtype = frame->dtype()});
    if (!loan) {
      return holoscan::make_unexpected(std::move(loan).error());
    }
    auto writer = loan->write(context.cuda_stream());
    if (!writer) {
      return holoscan::make_unexpected(std::move(writer).error());
    }
    if (cudaMemcpyAsync(writer->data(), frame->data(), writer->byte_size(),
                        cudaMemcpyDeviceToDevice,
                        context.cuda_stream().get()) != cudaSuccess) {
      return holoscan::make_unexpected(holoscan::Error{
          holoscan::ErrorCode::kFailure, "OverlaySourceOp: D2D failed"});
    }
    auto committed = std::move(*writer).commit();
    if (!committed) {
      return holoscan::make_unexpected(std::move(committed).error());
    }
    return port.emit(
        std::move(*loan),
        holoscan::EmitOptions{.capture_time = context.activation_time()});
  }
};

void bind_device(holoscan::CompileOptions* options, const char* op,
                 const char* port) {
  options->deployment.bind_tensor_output_device(
      holoscan::TensorOutputDevicePlacement{
          .operator_path = op,
          .output_port = port,
          .device = holoscan::DeviceId{0},
      });
}

TEST(X11RdViewer, OverlayPortsSmoke) {
  if (!x11_display_available()) {
    GTEST_SKIP() << "no X11 display (DISPLAY / XOpenDisplay)";
  }

  SmokeBank::reset(make_tone_rd(), kFrames);
  X11RdViewerOp::reset_presented_count();
  // Two clusters of two cells each; representatives are their first cells.
  OverlayBank::detection_idx = testing::to_device_tensor(
      std::vector<int64_t>{8, 20, 9, 21, 40, 90, 41, 91}, {kDetections, 2},
      DLDataType{kDLInt, 64, 1});
  OverlayBank::labels = testing::to_device_tensor(
      std::vector<int64_t>{0, 0, 1, 1}, {kDetections},
      DLDataType{kDLInt, 64, 1});
  OverlayBank::rep_idx = testing::to_device_tensor(
      std::vector<int64_t>{8, 20, 40, 90}, {kClusters, 2},
      DLDataType{kDLInt, 64, 1});
  std::vector<float> ranges(static_cast<size_t>(kRange));
  for (size_t i = 0; i < ranges.size(); ++i) {
    ranges[i] = static_cast<float>(i);
  }
  std::vector<float> velocities(static_cast<size_t>(kDoppler));
  for (size_t i = 0; i < velocities.size(); ++i) {
    velocities[i] = 10.0F - 0.5F * static_cast<float>(i);
  }
  OverlayBank::range_bins =
      testing::to_device_tensor(ranges, {kRange}, real_dtype<float>());
  OverlayBank::velocity_bins =
      testing::to_device_tensor(velocities, {kDoppler}, real_dtype<float>());
  // (az, el) radians per detection, inside the configured panel extents.
  OverlayBank::aoa_estimates = testing::to_device_tensor(
      std::vector<float>{-0.10F, 0.02F, -0.08F, 0.01F, 0.12F, -0.03F, 0.14F,
                         -0.02F},
      {kDetections, 2}, real_dtype<float>());
  // Active-variant index for the header label.
  OverlayBank::label_idx = testing::to_device_tensor(
      std::vector<int64_t>{1}, {1}, DLDataType{kDLInt, 64, 1});

  MaxTensorExtents extents;
  extents.max_doppler_bins = kDoppler;
  extents.max_range_bins = kRange;

  X11RdViewerConfig viz_cfg;
  viz_cfg.title = "X11RdViewer overlay smoke";
  viz_cfg.width = 640;
  viz_cfg.height = 480;
  viz_cfg.overlay_detections = true;
  viz_cfg.overlay_clusters = true;
  viz_cfg.plots_enabled = true;
  viz_cfg.plot_window = kFrames;
  viz_cfg.frame_period = 0.02;
  viz_cfg.overlay_boresight = true;
  viz_cfg.az_lo = -0.3;
  viz_cfg.az_hi = 0.3;
  viz_cfg.el_lo = -0.1;
  viz_cfg.el_hi = 0.1;
  viz_cfg.overlay_label = true;
  viz_cfg.label_caption = "variant";
  viz_cfg.label_names = {"alpha", "beta", "gamma"};

  holoscan::Graph graph{"x11-rd-overlay-smoke"};
  const auto src = graph.op<OverlaySourceOp>("src");
  const auto to_image = graph.op<RdMapToImageOp>("to_image", extents);
  const auto viewer = graph.op<X11RdViewerOp>("viewer", viz_cfg);
  const auto counter = graph.op<PresentCountOp>("counter");
  const holoscan::ConnectionOptions q{.queue_depth = 2U};
  graph.add_flow(src->rd_out, to_image->rd_maps, q);
  graph.add_flow(to_image->rd_image, viewer->rd_image, q);
  graph.add_flow(to_image->rd_image, counter->in, q);
  graph.add_flow(src->idx_out, viewer->detection_idx, q);
  graph.add_flow(src->labels_out, viewer->labels, q);
  graph.add_flow(src->rep_out, viewer->rep_idx, q);
  graph.add_flow(src->range_out, viewer->range_bins, q);
  graph.add_flow(src->velocity_out, viewer->velocity_bins, q);
  graph.add_flow(src->aoa_out, viewer->aoa_estimates, q);
  graph.add_flow(src->label_out, viewer->label_idx, q);
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

  holoscan::CompileOptions options;
  for (const char* port : {"rd_out", "idx_out", "labels_out", "rep_out",
                           "range_out", "velocity_out", "aoa_out",
                           "label_out"}) {
    bind_device(&options, "src", port);
  }
  bind_device(&options, "to_image", "rd_image");

  const holoscan::ExecutionPlan plan =
      holoscan::compile(graph, std::move(options));
  ASSERT_TRUE(plan.ok()) << plan.json();

  holoscan::RunSession session = holoscan::run_async(plan);
  {
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds{10};
    std::unique_lock lock(SmokeBank::mutex);
    while ((SmokeBank::presented < kFrames ||
            X11RdViewerOp::presented_count() < kFrames) &&
           session.diagnostics().empty() &&
           std::chrono::steady_clock::now() < deadline) {
      SmokeBank::ready.wait_for(lock, std::chrono::milliseconds{25});
    }
  }
  session.request_stop();
  session.wait();

  ASSERT_TRUE(session.diagnostics().empty());
  {
    const std::lock_guard lock(SmokeBank::mutex);
    EXPECT_GE(SmokeBank::presented, kFrames);
  }
  EXPECT_GE(X11RdViewerOp::presented_count(), kFrames);
}

}  // namespace
}  // namespace holoscan::radar::app
