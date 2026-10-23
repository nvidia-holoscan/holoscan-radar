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
#include <holoscan/core/lifecycle.hpp>
#include <holoscan/core/readiness_source.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/parameter_schema.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>

#include "io/cphd_reader_op/collection_paths.hpp"
#include "io/cphd_reader_op/playback_scheduler.hpp"
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
  bool wait_for_downstream{false};
};

HOLOSCAN_PARAM_SCHEMA(CphdReaderParams,
                      holoscan::param_schema("com.nvidia.holoscan.radar.CphdReaderParams", 2),
                      holoscan::field<&CphdReaderParams::path>("path"),
                      holoscan::field<&CphdReaderParams::coordinate_frame>("coordinate_frame"),
                      holoscan::field<&CphdReaderParams::pulses_per_tick>("pulses_per_tick"),
                      holoscan::field<&CphdReaderParams::channel>("channel"),
                      holoscan::field<&CphdReaderParams::pulse_range>("pulse_range"),
                      holoscan::field<&CphdReaderParams::loop>("loop"),
                      holoscan::field<&CphdReaderParams::playback_rate>("playback_rate"),
                      holoscan::field<&CphdReaderParams::duty_cycle>("duty_cycle"),
                      holoscan::field<&CphdReaderParams::wait_for_downstream>("wait_for_downstream")
                          .description("When true, required Tensor emits wait on kBackpressured / "
                                       "kResourceExhausted instead of failing compute(). Unmetered "
                                       "playback_rate 0 always waits. Static. Default false."));

// Unmetered ingest (playback_rate 0) waits on a full queue. A positive rate
// fail-fasts unless wait_for_downstream is true.
[[nodiscard]] inline bool cphd_reader_waits_on_full(const CphdReaderParams& params) {
  return params.wait_for_downstream || !(params.playback_rate > 0.0);
}

template <typename T>
class CPHDReaderOp : public holoscan::Operator<CphdReaderParams> {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>);

 public:
  explicit CPHDReaderOp(const MaxTensorExtents& extents, std::shared_ptr<FormationHud> hud = {});
  ~CPHDReaderOp() override;

  void setup(OperatorSpec& spec) override;
  [[nodiscard]] Contract contract() const override;
  void start() override;
  void stop() override;
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
  LifecycleStatus on_arm(LifecycleContext& context) noexcept;
  LifecycleStatus on_release(LifecycleContext& context) noexcept;
  void release_resources() noexcept;
  struct Batch;
  enum class Stage { kRead, kPublish, kPacing, kQuiet, kStopped };
  NotificationSource wake_;
  cphd::PlaybackScheduler scheduler_;
  Stage stage_{Stage::kRead};
  std::unique_ptr<Batch> batch_;
  std::optional<int64_t> capture_time_ns_;
  void open_file();
  void publish_hud();
  void maybe_warn_behind();
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
  bool behind_episode_{false};
  std::optional<std::chrono::steady_clock::time_point> last_behind_warn_at_{};
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
