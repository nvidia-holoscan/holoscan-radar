// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <array>
#include <initializer_list>
#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include <holoscan/core/expected_macros.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/sample.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/sensor_io/provenance.hpp>

#include "radar_app/app/gui_lifecycle.hpp"
#include "radar_app/viz/image_fit.hpp"
#include "radar_app/viz/window_extent.hpp"

namespace holoscan::radar::app::viz {

// Native presentation must redraw the window. Retain only finished GPU panels;
// never trigger SAR rasterization, HUD generation or host download from this clock.
class DockedFrameOp final : public Operator<> {
 public:
  DockedFrameOp(int width, int image_height, int footer_height, double rate, bool fit,
                std::string title, std::shared_ptr<FrameProgress> progress, int status_width = 0)
  : width_(width),
    image_height_(image_height),
    footer_height_(footer_height),
    status_width_(status_width),
    fit_(fit),
    extent_(std::move(title)),
    progress_(std::move(progress)),
    period_(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::duration<double>(1.0 / std::max(20.0, rate)))) {}
  void setup(OperatorSpec& spec) override {
    const TensorRepresentation format{.memory_kind = MemoryKind::kCudaDevice,
                                      .dtype = DLDataType{kDLUInt, 8U, 1U},
                                      .rank = 3U,
                                      .strides = TensorStridePolicy::kContiguousOnly};
    for (auto* port : {&image, &footer})
      spec.input(*port)
          .tensor_field(schema::ImageTensorFields::data)
          .expects_tensor(TensorInputSpec{.representation = format});
    spec.output(output)
        .tensor_field(schema::ImageTensorFields::data)
        .produces_tensor(TensorOutputSpec{
            .representation = format,
            .bounds = TensorBounds{.max_rank = 3U,
                                   .max_byte_span = static_cast<std::uint64_t>(width_) *
                                                    (image_height_ + footer_height_) * 4U},
            .storage = TensorOutputStorage::kRuntimePool});
  }
  Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = period_}).input(image, DrainAll{}).input(footer, DrainAll{});
    return result;
  }
  expected<void, Error> compute(ExecutionContext& context) override {
    HOLOSCAN_TRY(auto images, image.selected_batch());
    HOLOSCAN_TRY(auto footers, footer.selected_batch());
    if (!images.empty())
      held_image_ = images.back();
    if (!footers.empty())
      held_footer_ = footers.back();
    if ((held_image_ &&
         (!held_image_->data.data || held_image_->data.width != width_ ||
          held_image_->data.height != image_height_ + (status_width_ > 0 ? footer_height_ : 0))) ||
        (held_footer_ &&
         (!held_footer_->data.data || held_footer_->data.width != width_ - status_width_ ||
          held_footer_->data.height != footer_height_)))
      return make_unexpected(Error{ErrorCode::kInvalidArgument, "Docked panel shape mismatch"});
    const int height = image_height_ + footer_height_;
    const std::array<std::int64_t, 3> shape{height, width_, 4};
    HOLOSCAN_TRY(auto loan, output.allocate_tensor(TensorLoanRequest{
                                .shape = shape, .dtype = DLDataType{kDLUInt, 8U, 1U}}));
    HOLOSCAN_TRY(auto writer, loan.write(context.cuda_stream()));
    const auto [ww, wh] = extent_.query(width_, height);
    const auto status = compose_docked_panels(
        held_image_ ? held_image_->data.data->data() : nullptr,
        held_footer_ ? held_footer_->data.data->data() : nullptr, writer.data(), width_,
        image_height_, footer_height_, ww, wh, fit_, context.cuda_stream().get(), status_width_);
    if (status != cudaSuccess)
      return make_unexpected(Error{ErrorCode::kFailure, "Docked GPU presentation failed"});
    HOLOSCAN_RETURN_IF_ERROR(std::move(writer).commit());
    schema::ImageT descriptor{};
    descriptor.width = width_;
    descriptor.height = height;
    descriptor.encoding = schema::ImageEncoding_RGBA8;
    HOLOSCAN_RETURN_IF_ERROR(
        output.emit_tensor(std::move(loan), descriptor,
                           held_image_ ? sensor_io::derive_from(*held_image_)
                                       : EmitOptions{.capture_time = context.activation_time()}));
    if (held_image_ && progress_)
      progress_->capture_ns.store(held_image_->metadata.capture_time().timestamp_ns);
    return {};
  }
  void stop() override {
    held_image_.reset();
    held_footer_.reset();
    extent_.close();
  }
  Input<schema::ImageT> image{"image"}, footer{"footer"};
  Output<schema::ImageT> output{"output"};

 private:
  int width_, image_height_, footer_height_, status_width_;
  bool fit_;
  WindowExtent extent_;
  std::shared_ptr<FrameProgress> progress_;
  std::chrono::nanoseconds period_;
  std::optional<Sample<schema::ImageT>> held_image_, held_footer_;
};
}  // namespace holoscan::radar::app::viz
