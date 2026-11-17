// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include <holoscan/threading/thread_placement.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/compile.hpp>
#include <holoscan/core/expected_macros.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/sample.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/sensor_io/provenance.hpp>

#include "radar_app/app/gpu_placement.hpp"
#include "radar_app/app/gui_lifecycle.hpp"
#include "radar_app/app/visualization_config.hpp"
#include "radar_app/viz/image_device_bridge_op.hpp"
#include "radar_app/viz/image_fit.hpp"
#include "radar_app/viz/window_extent.hpp"
#include "radar_app/viz/native_image_presenter_op.hpp"
#if HOLOSCAN_RADAR_HAS_X11
#include "radar_app/viz/x11_image_presenter_op.hpp"
#endif

namespace holoscan::radar::app::viz {

class CompletedFrameSinkOp final : public Operator<> {
 public:
  explicit CompletedFrameSinkOp(
      std::shared_ptr<FrameProgress> progress = authoring_frame_progress())
  : progress_(std::move(progress)) {}
  void setup(OperatorSpec& spec) override { spec.input(image); }
  Contract contract() const override {
    Contract result;
    result.trigger(OnEach{image});
    return result;
  }
  expected<void, Error> compute(ExecutionContext&) override {
    HOLOSCAN_TRY(auto frame, image.receive());
    if (!frame.data.data)
      return make_unexpected(Error{ErrorCode::kInvalidArgument, "Missing completed image"});
    if (progress_)
      progress_->capture_ns.store(frame.metadata.capture_time().timestamp_ns);
    return {};
  }
  Input<schema::ImageT> image{"image"};

 private:
  std::shared_ptr<FrameProgress> progress_;
};

// Only complete display frames are retained. Periodic replay keeps SDK window
// events alive even before the first formation and throughout final-image hold.
class CompletedFrameReplayOp final : public Operator<> {
 public:
  CompletedFrameReplayOp(std::int32_t width, std::int32_t height, double rate = 60.0,
                         std::shared_ptr<FrameProgress> progress = authoring_frame_progress(),
                         bool fit = false, std::string title = {})
  : width_(width),
    height_(height),
    progress_(std::move(progress)),
    fit_(fit),
    extent_(std::move(title)),
    period_(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::duration<double>(1.0 / std::max(20.0, rate)))) {}
  void setup(OperatorSpec& spec) override {
    const TensorRepresentation format{.memory_kind = MemoryKind::kCudaDevice,
                                      .dtype = DLDataType{kDLUInt, 8U, 1U},
                                      .rank = 3U,
                                      .strides = TensorStridePolicy::kContiguousOnly};
    spec.input(image)
        .tensor_field(schema::ImageTensorFields::data)
        .expects_tensor(TensorInputSpec{.representation = format});
    spec.output(output)
        .tensor_field(schema::ImageTensorFields::data)
        .produces_tensor(TensorOutputSpec{
            .representation = format,
            .bounds =
                TensorBounds{.max_rank = 3U,
                             .max_byte_span = static_cast<std::uint64_t>(width_) * height_ * 4U},
            .storage = TensorOutputStorage::kRuntimePool});
  }
  Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = period_}).input(image, DrainAll{});
    return result;
  }
  expected<void, Error> compute(ExecutionContext& context) override {
    // A completed SAR frame can carry an old acquisition timestamp after a
    // long formation. Consume bounded queue survivors by arrival, not age.
    // Required image/text/geometry coherence was already established upstream.
    HOLOSCAN_TRY(auto frames, image.selected_batch());
    if (!frames.empty()) {
      held_ = frames.back();
      if (progress_)
        progress_->capture_ns.store(held_->metadata.capture_time().timestamp_ns);
    }
    const std::array<std::int64_t, 3> shape{height_, width_, 4};
    HOLOSCAN_TRY(auto loan, output.allocate_tensor(TensorLoanRequest{
                                .shape = shape, .dtype = DLDataType{kDLUInt, 8U, 1U}}));
    HOLOSCAN_TRY(auto writer, loan.write(context.cuda_stream()));
    const auto stream = context.cuda_stream().get();
    cudaError_t status;
    if (held_) {
      const auto& frame = held_->data;
      if (frame.width != width_ || frame.height != height_ || !frame.data)
        return make_unexpected(Error{ErrorCode::kInvalidArgument, "Replay image shape mismatch"});
      if (fit_) {
        const auto [ww, wh] = extent_.query(width_, height_);
        status = fit_image(frame.data->data(), writer.data(), width_, height_, ww, wh, stream);
      } else {
        status = cudaMemcpyAsync(writer.data(), frame.data->data(), writer.byte_size(),
                                 cudaMemcpyDeviceToDevice, stream);
      }
    } else {
      status = cudaMemsetAsync(writer.data(), 0, writer.byte_size(), stream);
    }
    if (status != cudaSuccess)
      return make_unexpected(Error{ErrorCode::kFailure, "Completed frame replay failed"});
    HOLOSCAN_RETURN_IF_ERROR(std::move(writer).commit());
    schema::ImageT descriptor{};
    descriptor.width = width_;
    descriptor.height = height_;
    descriptor.encoding = schema::ImageEncoding_RGBA8;
    // Repaint does not change the capture timestamp of the retained content.
    return output.emit_tensor(std::move(loan), descriptor,
                              held_ ? sensor_io::derive_from(*held_)
                                    : EmitOptions{.capture_time = context.activation_time()});
  }
  void stop() override {
    held_.reset();
    extent_.close();
  }
  Input<schema::ImageT> image{"image"};
  Output<schema::ImageT> output{"output"};

 private:
  std::int32_t width_, height_;
  std::shared_ptr<FrameProgress> progress_;
  bool fit_;
  WindowExtent extent_;
  std::chrono::nanoseconds period_;
  std::optional<Sample<schema::ImageT>> held_;
};

inline void add_image_presentation(
    Graph& graph, CompileOptions& options, PartitionHandle& composition,
    const Output<schema::ImageT>& frame, const std::string& prefix, std::int32_t width,
    std::int32_t height, const std::string& title, const VisualizationConfig& config,
    std::shared_ptr<FrameProgress> progress = authoring_frame_progress(),
    const std::string& position = {}) {
  const ConnectionOptions queue{.queue_policy = QueuePolicy::kDropOldest, .queue_depth = 1U};
  auto presentation = graph.partition(prefix + "_presentation");
  options.deployment.bind_thread_placement(prefix + "_presentation", ThreadPlacement{});
  if (config.headless) {
    auto sink = graph.op<CompletedFrameSinkOp>(prefix + "_sink", std::move(progress));
    presentation.add(sink);
    graph.add_flow(frame, sink->image, queue);
    return;
  }
#if HOLOSCAN_RADAR_HAS_X11
  if (config.x11_presentation.value_or(true)) {
    auto download = graph.op<ImageDownloadOp>(prefix + "_download", width, height);
    auto viewer =
        graph.op<X11ImagePresenterOp>(prefix + "_viewer", width, height, title,
                                      config.resize_mode == "fit", std::move(progress), position);
    composition.add(download);
    presentation.add(viewer);
    graph.add_flow(frame, download->image_in, queue);
    graph.add_flow(download->image_out, viewer->image, queue);
    return;
  }
#endif
  auto replay =
      graph.op<CompletedFrameReplayOp>(prefix + "_replay", width, height, config.display_rate_hz,
                                       std::move(progress), config.resize_mode == "fit", title);
  auto viewer = graph.op<NativeImagePresenterOp>(prefix + "_viewer", width, height, title);
  auto repaint = graph.partition(prefix + "_repaint");
  repaint.add(replay);
  options.deployment.bind_thread_placement(prefix + "_repaint", ThreadPlacement{});
  presentation.add(viewer);
  options.deployment.bind_tensor_output_device(
      TensorOutputDevicePlacement{.operator_path = prefix + "_replay",
                                  .output_port = "output",
                                  .device = DeviceId{application_composer_device()}});
  graph.add_flow(frame, replay->image, queue);
  connect_image_devices(graph, options, repaint, presentation, replay->output,
                        viewer->image_layer_input(0), prefix + "_display", width, height,
                        DeviceId{application_composer_device()},
                        DeviceId{application_display_device()}, queue);
}

}  // namespace holoscan::radar::app::viz
