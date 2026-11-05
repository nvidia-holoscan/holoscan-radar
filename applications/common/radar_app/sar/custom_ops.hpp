// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>

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

#include "radar_app/sar/hud_format.hpp"
#include "radar_common/sar/formation_hud.hpp"
#include "radar_common/sdk/tensor_io.hpp"

#ifdef HOLOSCAN_RADAR_SAR_HOLOVIZ
#include <holoscan/sensor_io/sensor_schema_package.hpp>
#endif

namespace holoscan::radar::app::sar {
#ifdef HOLOSCAN_RADAR_SAR_HOLOVIZ
class SarImageSchemaSinkOp : public holoscan::Operator<> {
 public:
  void setup(OperatorSpec& spec) override { spec.input(frame).queue_depth(4U); }
  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnEach{frame});
    return result;
  }
  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    HOLOSCAN_TRY(auto image, frame.receive_data());
    if (!image.data)
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    return {};
  }
  Input<schema::ImageT> frame{"frame"};
};
#endif

struct ImageCount {
  std::atomic<int64_t> count{0};
  std::atomic<int64_t> reported_capture_ns{0};
};

class SarImageReportOp : public holoscan::Operator<> {
 public:
  // `product_extent` is the formed product's size when that is not the size of
  // the tensor on `image`. A mosaic publishes a decimated view of a resident
  // accumulator, and the strip is what was formed; the view is a transport
  // detail nobody reading the log cares about. {0, 0} reports the tensor.
  explicit SarImageReportOp(std::shared_ptr<ImageCount> count,
                            std::shared_ptr<FormationHud> hud = {},
                            std::array<int64_t, 2> product_extent = {0, 0})
  : count_(std::move(count)), hud_(std::move(hud)), product_extent_(product_extent) {}
  void setup(OperatorSpec& spec) override { spec.input(image).queue_depth(4U); }
  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnEach{image});
    return result;
  }
  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    HOLOSCAN_TRY(auto received, receive_tensor(image, "SarImageReportOp"));
    const auto shape = received.shape();
    if (shape.size() != 2U) {
      return make_unexpected(
          Error{ErrorCode::kInvalidArgument, "SarImageReportOp: expected (num_v,num_u)"});
    }
    const auto index = count_->count.fetch_add(1) + 1;
    const bool shape_changed = !last_num_v_.has_value() || *last_num_v_ != shape[0] ||
                               !last_num_u_.has_value() || *last_num_u_ != shape[1];
    const int64_t report_v = product_extent_[0] > 0 ? product_extent_[0] : shape[0];
    const int64_t report_u = product_extent_[1] > 0 ? product_extent_[1] : shape[1];
    log_formed(index, report_v, report_u, /*info=*/false);
    const auto now = std::chrono::steady_clock::now();
    if (shape_changed || !last_info_at_.has_value() || now - *last_info_at_ >= kHeartbeatPeriod) {
      log_formed(index, report_v, report_u, /*info=*/true);
      last_info_at_ = now;
    }
    last_num_v_ = shape[0];
    last_num_u_ = shape[1];
    count_->reported_capture_ns.store(context.activation_time().timestamp_ns,
                                      std::memory_order_release);
    return {};
  }
  Input<Tensor> image{"image"};

 private:
  static constexpr auto kHeartbeatPeriod = std::chrono::seconds(1);

  void log_formed(int64_t index, int64_t num_v, int64_t num_u, bool info) const {
    std::string extra;
    if (hud_) {
      const auto snap = hud_->snapshot();
      const auto& bp = snap.bp;
      const auto& pb = bp.playback;
      extra =
          format_bp_rate_line(bp.compute_s, bp.pulses, bp.pixels, bp.num_v, bp.num_u, num_v, num_u);
      if (info && !extra.empty())
        hud_->publish_rate_line(extra);
      if (!extra.empty())
        extra = "  " + extra;
      if (pb.valid) {
        extra += "  " + format_lag_line(pb.behind_s, pb.target_s(), pb.wall_s);
      }
    }
    if (info) {
      HOLOSCAN_LOG_INFO("SAR: image {} formed, {}x{}{}", index, num_v, num_u, extra);
    } else {
      HOLOSCAN_LOG_DEBUG("SAR: image {} formed, {}x{}{}", index, num_v, num_u, extra);
    }
  }

  std::shared_ptr<ImageCount> count_;
  std::shared_ptr<FormationHud> hud_{};
  std::array<int64_t, 2> product_extent_{0, 0};
  std::optional<std::chrono::steady_clock::time_point> last_info_at_{};
  std::optional<int64_t> last_num_v_{};
  std::optional<int64_t> last_num_u_{};
};
}  // namespace holoscan::radar::app::sar
