// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <vector>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/parameter_schema.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>

#include "io/cphd_reader_op/collection_paths.hpp"
#include "radar_common/descriptors/coordinate_frame.hpp"
#include "radar_common/descriptors/max_tensor_extents.hpp"
#include "radar_common/sdk/tensor_io.hpp"
#include "radar_schemas/coordinate_frame_bridge.hpp"

namespace holoscan::radar {
namespace cphd {
class CphdFile;
}

enum class SarStreamBoundary : uint8_t {
  kNone = 0,
  kCollectionBoundary = 1,
  kEndOfStream = 2,
};

struct CphdReaderParams {
  CollectionPaths path{};
  CoordinateFrame coordinate_frame{};
  int64_t pulses_per_tick{1};
  int64_t channel{0};
  std::vector<int64_t> pulse_range{};
  bool loop{false};
  double playback_rate{0.0};
  double duty_cycle{1.0};
};

HOLOSCAN_PARAM_SCHEMA(CphdReaderParams,
                      holoscan::param_schema("com.nvidia.holoscan.radar.CphdReaderParams", 1),
                      holoscan::field<&CphdReaderParams::path>("path"),
                      holoscan::field<&CphdReaderParams::coordinate_frame>("coordinate_frame"),
                      holoscan::field<&CphdReaderParams::pulses_per_tick>("pulses_per_tick"),
                      holoscan::field<&CphdReaderParams::channel>("channel"),
                      holoscan::field<&CphdReaderParams::pulse_range>("pulse_range"),
                      holoscan::field<&CphdReaderParams::loop>("loop"),
                      holoscan::field<&CphdReaderParams::playback_rate>("playback_rate"),
                      holoscan::field<&CphdReaderParams::duty_cycle>("duty_cycle"));

struct CollectionSpan {
  int file_index{0};
  std::chrono::steady_clock::time_point dwell_start{};
  std::optional<std::chrono::steady_clock::time_point> dwell_end;
  std::optional<std::chrono::steady_clock::time_point> quiet_start;
  std::optional<std::chrono::steady_clock::time_point> quiet_end;
};
struct PlaybackStatus {
  bool valid{false};
  bool paced{false};
  double collect_s{0.0};
  double wall_s{0.0};
  double behind_s{0.0};
  double last_slack_s{0.0};
};
struct LastMeanApc {
  bool valid{false};
  int64_t file_index{0};
  std::array<double, 3> ecef{};
};

template <typename T>
class CPHDReaderOp : public holoscan::Operator<CphdReaderParams> {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>);

 public:
  explicit CPHDReaderOp(const MaxTensorExtents& extents);
  ~CPHDReaderOp() override;

  void setup(OperatorSpec& spec) override;
  [[nodiscard]] Contract contract() const override;
  void start() override;
  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override;

  std::optional<std::chrono::steady_clock::time_point> last_packet_at() const {
    return last_packet_at_;
  }
  std::vector<CollectionSpan> timeline_spans() const { return spans_; }
  PlaybackStatus playback_status() const { return playback_; }
  LastMeanApc last_mean_apc() const { return last_apc_; }

  Output<Tensor> phase_history{"phase_history"};
  Output<Tensor> positions{"positions"};
  Output<Tensor> reference_position{"reference_position"};
  Output<Tensor> sample_axis{"sample_axis"};
  Output<Tensor> pulse_times{"pulse_times"};
  Output<Tensor> velocities{"velocities"};
  Output<WireCoordinateFrame> coordinate_frame{"coordinate_frame"};
  Output<uint8_t> stream_boundary{"stream_boundary"};

 private:
  void open_file();
  MaxTensorExtents extents_;
  std::vector<std::string> files_;
  std::unique_ptr<cphd::CphdFile> file_;
  size_t file_index_{0};
  int64_t first_pulse_{0}, last_pulse_{0}, next_pulse_{0};
  bool stopped_{false};
  bool nonfinite_warned_{false};
  double pending_cooldown_s_{0.0};
  double collection_first_time_{0.0}, collection_last_time_{0.0};
  std::chrono::steady_clock::time_point collection_wall_start_{};
  std::optional<std::chrono::steady_clock::time_point> last_packet_at_;
  std::vector<CollectionSpan> spans_;
  PlaybackStatus playback_{};
  LastMeanApc last_apc_{};
  LatchedOutput axis_out_, times_out_, velocities_out_, frame_out_;
};

extern template class CPHDReaderOp<float>;
extern template class CPHDReaderOp<double>;
}  // namespace holoscan::radar
