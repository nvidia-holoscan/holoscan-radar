// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
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
};

class SarImageReportOp : public holoscan::Operator<> {
 public:
  explicit SarImageReportOp(std::shared_ptr<ImageCount> count,
                            std::shared_ptr<FormationHud> hud = {})
  : count_(std::move(count)), hud_(std::move(hud)) {}
  void setup(OperatorSpec& spec) override { spec.input(image).queue_depth(4U); }
  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnEach{image});
    return result;
  }
  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    HOLOSCAN_TRY(auto received, receive_tensor(image, "SarImageReportOp"));
    const auto shape = received.shape();
    if (shape.size() != 2U) {
      return make_unexpected(
          Error{ErrorCode::kInvalidArgument, "SarImageReportOp: expected (num_v,num_u)"});
    }
    const auto index = count_->count.fetch_add(1) + 1;
    if (hud_) {
      const auto& pb = hud_->snapshot().bp.playback;
      if (pb.valid) {
        HOLOSCAN_LOG_INFO(
            "SAR: image {} formed, {}x{}  collect={:.3f}s wall={:.3f}s behind={:.3f}s", index,
            shape[0], shape[1], pb.collect_s, pb.wall_s, pb.behind_s);
      } else {
        HOLOSCAN_LOG_INFO("SAR: image {} formed, {}x{}", index, shape[0], shape[1]);
      }
    } else {
      HOLOSCAN_LOG_INFO("SAR: image {} formed, {}x{}", index, shape[0], shape[1]);
    }
    return {};
  }
  Input<Tensor> image{"image"};

 private:
  std::shared_ptr<ImageCount> count_;
  std::shared_ptr<FormationHud> hud_{};
};
}  // namespace holoscan::radar::app::sar
