// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <chrono>
#include <cstddef>
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
#include "radar_common/sar/formation_hud.hpp"
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

template <typename T>
class CPHDReaderOp : public holoscan::Operator<CphdReaderParams> {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>);

 public:
  explicit CPHDReaderOp(const MaxTensorExtents& extents, std::shared_ptr<FormationHud> hud = {});
  ~CPHDReaderOp() override;

  void setup(OperatorSpec& spec) override;
  [[nodiscard]] Contract contract() const override;
  void start() override;
  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override;

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
  void publish_hud();
  MaxTensorExtents extents_;
  std::shared_ptr<FormationHud> hud_{};
  std::vector<std::string> files_;
  std::unique_ptr<cphd::CphdFile> file_;
  size_t file_index_{0};
  int64_t first_pulse_{0}, last_pulse_{0}, next_pulse_{0};
  bool stopped_{false};
  bool nonfinite_warned_{false};
  double pending_cooldown_s_{0.0};
  double collection_t_min_{0.0};
  double collection_t_max_{0.0};
  bool collection_time_valid_{false};
  std::chrono::steady_clock::time_point collection_wall_start_{};
  std::optional<std::chrono::steady_clock::time_point> last_packet_at_;
  std::vector<CollectionSpan> spans_;
  PlaybackStatus playback_{};
  LastMeanApc last_apc_{};
  LatchedOutput axis_out_, times_out_, velocities_out_, frame_out_;
  // Private H2D stream. Synchronizing the runtime compute stream after upload
  // waits for every kernel already recorded on that stream — including
  // backprojection if the compiler fused ingest onto the formation stream.
  std::uintptr_t h2d_stream_{0};
  std::uintptr_t h2d_done_{0};
  // Reused CF8 ingest: pinned stored-bytes (or host-decoded integer samples),
  // a device CF8 scratch for the double widen, and per-pulse AmpSF.
  std::uintptr_t pinned_signal_{0};
  std::uintptr_t device_cf8_{0};
  std::uintptr_t device_amp_{0};
  std::size_t pinned_signal_bytes_{0};
};

extern template class CPHDReaderOp<float>;
extern template class CPHDReaderOp<double>;
}  // namespace holoscan::radar
