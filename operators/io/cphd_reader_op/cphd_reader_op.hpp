// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <type_traits>
#include <vector>

#include <holoscan/holoscan.hpp>

#include "io/cphd_reader_op/collection_paths.hpp"
#include "radar_common/descriptors/coordinate_frame.hpp"

namespace holoscan::radar {

namespace cphd {
class CphdFile;  // host-side file reading; defined in cphd_file.hpp
}  // namespace cphd

// Pinned staging for CPHDReaderOp. Defined in cphd_reader_op.cu.
struct CphdReaderResources;

// One collection on the wall-clock timeline the formation app
// paints. Not a port (R10). Open dwell: dwell_end empty (bar runs
// to now). Open quiet: quiet_end empty (cooldown sleep in flight).
struct CollectionSpan {
  int file_index{0};
  std::chrono::steady_clock::time_point dwell_start{};
  std::optional<std::chrono::steady_clock::time_point> dwell_end;
  std::optional<std::chrono::steady_clock::time_point> quiet_start;
  std::optional<std::chrono::steady_clock::time_point> quiet_end;
};

// Reads a CPHD collection and emits phase history with the per-pulse
// geometry stored alongside it. A source operator: no data inputs. This
// header is the contract reference for the operator's ports, shapes,
// dtypes, and parameters; see docs/planning/SAR/OPERATOR_API.md.
//
// Out: phase_history      — complex64/complex128, (num_pulses, num_samples).
// Out: positions          — float32/float64, (num_pulses, 3). Per-pulse
//                           aperture phase-centre positions.
// Out: reference_position — float32/float64, (num_pulses, 3). Per-pulse
//                           Stabilization Reference Point position.
// Out: sample_axis        — optional, float32/float64, (num_samples,).
// Out: pulse_times        — optional, float64 always, (num_pulses,).
// Out: velocities         — optional, float32/float64, (num_pulses, 3).
// Out: coordinate_frame   — optional, CoordinateFrame; emitted once, on the
//                           first tick of the collection.
//
// num_pulses is pulses_per_tick except on the final tick, where it is the
// remainder — downstream must read the pulse count off the tensor.
//
// Precision is fixed at compile time by the template instantiation
// (T = float → complex64/float32, T = double → complex128/float64).
// pulse_times is float64 in both, since single precision cannot hold a
// collection-relative timestamp to useful accuracy.
//
// Parameters:
//   path             — a single CPHD file, a directory of them, or an
//                      explicit ordered list. A directory is enumerated
//                      lexicographically rather than numerically, so indexed
//                      names want zero-padding; an explicit list keeps the
//                      order given. Static, required.
//   coordinate_frame — frame the emitted positions are expressed in. A frame
//                      disagreeing with the one the file declares is an
//                      error. Static, required.
//   pulses_per_tick  — batch size along the pulse axis. Static, required.
//   channel          — which CPHD channel to read. Static, default 0.
//   pulse_range      — half-open [first, last) pulse range, applied per file.
//                      Static, default the whole collection.
//   loop             — wrap to the first file instead of ending the stream.
//                      Static, default false.
//   playback_rate    — 0 = unconstrained; 1 = collect-time pacing (video
//                      convention: 2 = twice as fast). Static, default 0.
//   duty_cycle       — (0, 1]; 1 = no gap. Cooldown between collections
//                      is T_c*(1/D - 1). Static, default 1.
//
// **No parameter is [live]**, so none has an override input port: a reader's
// file, frame, batching and pacing are build-time configuration.
//
// One channel per instance is deliberate, not a gap: CPHD channels may carry
// different pulse and sample counts, so they do not stack into one tensor.
// See OPERATOR_API.md.
//
// A `path` naming several files reads them in sequence as separate
// collections, announcing `collection_boundary` at each seam and
// `end_of_stream` at the last unless `loop` is true, in which case the
// last seam wraps and is also `collection_boundary`. `pulse_range`
// applies to each file independently, each being its own collection.
template <typename T>
class CPHDReaderOp : public holoscan::Operator {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                "CPHDReaderOp supports float and double precision");

 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(CPHDReaderOp)

  CPHDReaderOp() = default;
  ~CPHDReaderOp() override;

  // Registers the argument setters for the custom parameter types and
  // attaches the boolean scheduling term the operator stops itself with.
  void initialize() override;

  void setup(OperatorSpec& spec) override;
  void start() override;
  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override;

  // Wall time of the most recent last-batch emit (every data port
  // of that tick). Not a port (R10); the formation app subtracts
  // this from BackprojectionOp's emit time.
  std::optional<std::chrono::steady_clock::time_point> last_packet_at()
      const {
    if (!last_packet_valid_.load(std::memory_order_acquire)) {
      return std::nullopt;
    }
    const auto ns =
        last_packet_ns_.load(std::memory_order_relaxed);
    return std::chrono::steady_clock::time_point{
        std::chrono::nanoseconds{ns}};
  }

  // Closed and in-progress dwell/quiet intervals, oldest first.
  // The formation timeline window copies this every tick.
  std::vector<CollectionSpan> timeline_spans() const;

 private:
  // Parameters per the contract. All static: a reader's file, frame,
  // batching and pacing are build-time configuration, not in-flight
  // overrides.
  Parameter<CollectionPaths> path_;
  Parameter<CoordinateFrame> coordinate_frame_;
  Parameter<int64_t> pulses_per_tick_;
  Parameter<int64_t> channel_;
  Parameter<std::vector<int64_t>> pulse_range_;
  Parameter<bool> loop_;
  Parameter<double> playback_rate_;
  Parameter<double> duty_cycle_;

  // Lets the operator stop itself once the selected range is exhausted. Held
  // directly rather than as a parameter: a condition passed to add_arg is
  // registered as a scheduling term, not as a parameter value, so exposing
  // it as one would add a contract-surface parameter that never binds.
  std::shared_ptr<holoscan::BooleanCondition> stop_condition_;

  std::unique_ptr<cphd::CphdFile> file_;

  // The collections to read, in order: the parameter with any directory
  // expanded. Resolved and validated in full at start() so that a bad file
  // late in the list fails before the first pulse rather than an hour in.
  std::vector<std::string> files_;
  size_t file_index_{0};

  // Opens files_[file_index_] and resolves the pulse range within it.
  void open_current_file();
  // Sleep so this tick's wall time is at least batch_span / playback_rate.
  void pace_tick(double batch_span);
  // Queue a platform cooldown after a collection when another
  // follows; slept at the next tick's start so the closing image
  // is already on screen.
  void cooldown_if_needed();
  // Publish last-batch emit time for the formation app.
  void publish_last_packet_at(std::chrono::steady_clock::time_point at);
  void timeline_open_dwell();
  void timeline_close_dwell();
  void timeline_quiet_begin();
  void timeline_quiet_end();

  // Resolved per file from pulse_range_ and that file's vector count.
  int64_t first_pulse_{0};
  int64_t last_pulse_{0};  // exclusive
  int64_t next_pulse_{0};
  // TxTime span of the pulses actually read from the current collection.
  double collection_t_min_{0.0};
  double collection_t_max_{0.0};
  bool collection_time_valid_{false};
  std::chrono::steady_clock::time_point tick_start_{};
  // Last collection's final pulse-batch emit, as nanoseconds of
  // steady_clock. Atomically published so EventBased can read it
  // from SarImageToLutOp during the reader's last pace_tick.
  std::atomic<int64_t> last_packet_ns_{0};
  std::atomic<bool> last_packet_valid_{false};
  // Wall-clock dwell/quiet ring for the app timeline. shared_ptr so
  // FORWARD_ARGS stay movable (mutex is not).
  static constexpr size_t kTimelineCap = 32;
  struct TimelineLog {
    mutable std::mutex mu;
    std::deque<CollectionSpan> spans;
  };
  std::shared_ptr<TimelineLog> timeline_{
      std::make_shared<TimelineLog>()};
  // Sum of compute() over the current collection; logged at the
  // seam. Host read + H2D + this tick's stream sync; not GPU FFT/BP.
  double collection_compute_s_{0.0};
  // Duty-cycle gap, taken at the first tick of the following
  // collection so the closing image is already up.
  double pending_cooldown_s_{0.0};
  // Reset at each collection: the descriptor is emitted once per collection,
  // not once per instance, because a new file may bring a new frame.
  bool frame_emitted_{false};
  // Real collections carry occasional non-finite ephemeris; warned about
  // once per instance rather than per tick.
  bool nonfinite_warned_{false};

  // Pinned staging so a tick does not value-initialize tens of MiB of
  // phase history. The H2D stream comes from the operator's GXF
  // CudaStreamPool (allocate_cuda_stream), not a homemade cudaStream_t:
  // set_cuda_stream only accepts GXF-mapped streams. shared_ptr, not
  // unique_ptr: FORWARD_ARGS constructors live in this header, and
  // unique_ptr's destructor needs a complete type in those TUs (see
  // CFARDetectorOp). make_shared is in the .cu.
  std::shared_ptr<CphdReaderResources> resources_;
};

// Explicit instantiations live in cphd_reader_op.cu.
extern template class CPHDReaderOp<float>;
extern template class CPHDReaderOp<double>;

}  // namespace holoscan::radar
