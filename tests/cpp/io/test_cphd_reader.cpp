// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <mutex>
#include <thread>
#include <utility>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <filesystem>  // NOLINT(build/c++17)
#include <fstream>
#include <future>
#include <stdexcept>
#include <exception>
#include <memory>
#include <string>
#include <vector>

#include <holoscan/testing/operator_test_harness.hpp>
#include <holoscan/core/compile.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/expected_macros.hpp>
#include <holoscan/threading/thread_placement.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "io/cphd_reader_op/playback_scheduler.hpp"

#include "utils/fixture_completion.hpp"
#include "io/cphd_reader_op/cphd_reader_op.hpp"
#include "io/cphd_reader_op/host_read_ahead.hpp"
#include "radar_common/tensor/tensor_utils.hpp"
#include "radar_common/sar/formation_hud.hpp"
#include "utils/tensor_fixtures.hpp"
#include "io/cphd_reader_op/cphd_file.hpp"
#include "io/cphd_fixture.hpp"  // NOLINT(build/include_subdir)

namespace {
namespace fs = std::filesystem;
using holoscan::radar::cphd::CphdFile;
using holoscan::radar::testing::cphd_fixture::make_fixture;

TEST(CphdReaderOp, AbsolutePlaybackAndQuietDeadlines) {
  using holoscan::radar::cphd::PlaybackClock;
  using holoscan::radar::cphd::after_seconds;
  using holoscan::radar::cphd::quiet_seconds;
  const PlaybackClock::time_point origin{std::chrono::seconds{100}};
  EXPECT_EQ(after_seconds(origin, 0.04 / 2.0), origin + std::chrono::milliseconds{20});
  EXPECT_EQ(after_seconds(origin, 0.04 / 0.5), origin + std::chrono::milliseconds{80});
  EXPECT_DOUBLE_EQ(quiet_seconds(4.0, 0.5, 2.0), 2.0);
  EXPECT_DOUBLE_EQ(quiet_seconds(4.0, 0.5, 0.5), 8.0);
  EXPECT_DOUBLE_EQ(quiet_seconds(4.0, 0.5, 0.0), 4.0);
  EXPECT_DOUBLE_EQ(quiet_seconds(4.0, 1.0, 2.0), 0.0);
  // Deadlines are rooted at collection start, not at a late callback's time.
  EXPECT_LT(after_seconds(origin, 0.04), origin + std::chrono::seconds{1});
}

TEST(CphdReaderOp, HostReadAheadOverlapsCurrentBatchAndRemainsBounded) {
  using holoscan::radar::cphd::HostReadAhead;
  using holoscan::radar::cphd::PlaybackScheduler;
  PlaybackScheduler worker;
  HostReadAhead<int> ahead;
  std::promise<void> read_started, allow_read, completed;
  const auto allowed = allow_read.get_future().share();
  const auto started = read_started.get_future();
  auto done = completed.get_future();
  worker.start();
  // Model the current slot being retained for GPU upload/publication retries.
  auto current = std::make_unique<int>(7);
  ahead.request(worker, [&] {
    read_started.set_value();
    allowed.wait();
    return std::make_unique<int>(8);
  });
  const auto observed = started.wait_for(std::chrono::seconds{1});
  // Always unblock cleanup even if the worker failed to start.
  EXPECT_EQ(observed, std::future_status::ready);
  EXPECT_EQ(*current, 7);
  // Neither a third slot nor a second outstanding read may be admitted.
  EXPECT_THROW(ahead.request(worker,
                             [] {
                               return std::make_unique<int>(9);
                             }),
               std::logic_error);
  worker.submit([&] {
    completed.set_value();
  });
  allow_read.set_value();
  EXPECT_EQ(done.wait_for(std::chrono::seconds{1}), std::future_status::ready);
  worker.stop();
  auto ready = ahead.take();
  ASSERT_TRUE(ready.has_value());
  EXPECT_FALSE(ready->error);
  EXPECT_EQ(*ready->batch, 8);
  EXPECT_EQ(*current, 7);  // The prefetched read did not overwrite the current slot.
  ahead.clear();
}

TEST(CphdReaderOp, HostReadAheadDefersErrorsAndJoinsAnInProgressRead) {
  using holoscan::radar::cphd::HostReadAhead;
  using holoscan::radar::cphd::PlaybackScheduler;
  PlaybackScheduler worker;
  HostReadAhead<int> ahead;
  std::promise<void> read_started, allow_read;
  auto started = read_started.get_future();
  const auto allowed = allow_read.get_future().share();
  worker.start();
  ahead.request(worker, [&]() -> std::unique_ptr<int> {
    read_started.set_value();
    allowed.wait();
    throw std::runtime_error("deferred CPHD read error");
  });
  EXPECT_EQ(started.wait_for(std::chrono::seconds{1}), std::future_status::ready);
  auto stopped = std::async(std::launch::async, [&] {
    worker.stop();
  });
  EXPECT_EQ(stopped.wait_for(std::chrono::milliseconds{10}), std::future_status::timeout);
  // Shutdown cannot preempt a synchronous filesystem read; it joins it.
  allow_read.set_value();
  EXPECT_EQ(stopped.wait_for(std::chrono::seconds{1}), std::future_status::ready);
  stopped.get();
  auto result = ahead.take();
  ASSERT_TRUE(result.has_value());
  EXPECT_FALSE(result->batch);
  ASSERT_TRUE(result->error);
  try {
    std::rethrow_exception(result->error);
    FAIL() << "Expected a deferred read exception";
  } catch (const std::runtime_error& error) {
    EXPECT_STREQ(error.what(), "deferred CPHD read error");
  }
  ahead.clear();
}

TEST(CphdReaderOp, EofDrainWaitsForClosingImageAndReport) {
  holoscan::radar::FormationHudSnapshot snapshot;
  snapshot.reader_stopped = true;
  snapshot.reader_capture_time_ns = 100;
  snapshot.last_packet_at = std::chrono::steady_clock::now();
  snapshot.bp.emit_at = *snapshot.last_packet_at + std::chrono::seconds{1};
  snapshot.bp.capture_time_ns = 99;
  snapshot.bp.closes_aperture = true;
  EXPECT_FALSE(holoscan::radar::sar_eof_drained(snapshot, 99));
  snapshot.bp.capture_time_ns = 100;
  snapshot.bp.closes_aperture = false;
  EXPECT_FALSE(holoscan::radar::sar_eof_drained(snapshot, 100));
  snapshot.bp.capture_time_ns = 101;  // final fan-out slice
  snapshot.bp.closes_aperture = true;
  EXPECT_FALSE(holoscan::radar::sar_eof_drained(snapshot, 100));
  EXPECT_TRUE(holoscan::radar::sar_eof_drained(snapshot, 101));
  snapshot.reader_stopped = false;
  EXPECT_FALSE(holoscan::radar::sar_eof_drained(snapshot, 101));
}

struct ReaderObservations {
  std::mutex mutex;
  std::array<std::vector<holoscan::TimePoint>, 4> stamps;
  std::vector<uint8_t> boundaries;
  std::atomic<unsigned> received{0};
};

// Delaying only positions forces a partial batch: phase_history can already
// have been accepted when positions runs out of queue/pool capacity.
class ReaderTensorSink : public holoscan::Operator<> {
 public:
  ReaderTensorSink(std::shared_ptr<ReaderObservations> observations, unsigned index, bool stall)
  : observations_(std::move(observations)), index_(index), stall_(stall) {}
  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(in).expects_tensor(holoscan::radar::tensor_input(
        holoscan::TensorRepresentation{.memory_kind = holoscan::MemoryKind::kCudaDevice}));
  }
  holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnEach{in});
    return result;
  }
  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext& context) override {
    HOLOSCAN_TRY(auto tensor, in.receive_data());
    {
      const std::lock_guard lock(observations_->mutex);
      observations_->stamps[index_].push_back(context.activation_time());
    }
    if (stall_) {
      while (!context.cancellation().stop_requested())
        std::this_thread::sleep_for(std::chrono::milliseconds{1});
    } else if (index_ == 1U) {
      std::this_thread::sleep_for(std::chrono::milliseconds{30});
    }
    return {};
  }
  holoscan::Input<holoscan::Tensor> in{"in"};

 private:
  std::shared_ptr<ReaderObservations> observations_;
  unsigned index_;
  bool stall_;
};

class ReaderBoundarySink : public holoscan::Operator<> {
 public:
  explicit ReaderBoundarySink(std::shared_ptr<ReaderObservations> observations)
  : observations_(std::move(observations)) {}
  void setup(holoscan::OperatorSpec& spec) override { spec.input(in); }
  holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnEach{in});
    return result;
  }
  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext& context) override {
    HOLOSCAN_TRY(auto value, in.receive_data());
    const std::lock_guard lock(observations_->mutex);
    observations_->stamps[3].push_back(context.activation_time());
    observations_->boundaries.push_back(value);
    ++observations_->received;
    return {};
  }
  holoscan::Input<uint8_t> in{"in"};

 private:
  std::shared_ptr<ReaderObservations> observations_;
};

TEST(CphdReaderOp, PartialPublicationRetriesPreserveOrderAndStampAndCancel) {
  using holoscan::Graph;
  using holoscan::ConnectionOptions;
  using holoscan::QueuePolicy;
  using holoscan::RealtimeClock;
  using holoscan::CompileOptions;
  using holoscan::ThreadPlacement;
  using holoscan::TensorOutputDevicePlacement;
  using holoscan::DeviceId;
  using holoscan::compile;
  using holoscan::run_async;
  using holoscan::radar::CPHDReaderOp;
  using holoscan::radar::CphdReaderParams;
  using holoscan::radar::CollectionPaths;
  using holoscan::radar::FormationHud;
  using holoscan::radar::MaxTensorExtents;
  using holoscan::radar::CoordinateFrame;
  using holoscan::radar::CoordinateFrameSpec;
  using holoscan::radar::CoordinateFrameType;
  const auto path = fs::temp_directory_path() / "holoscan_reader_retry.cphd";
  make_fixture(path, 0.0, 0.0, 6378000.0, false, 20);
  for (bool stall : {false, true}) {
    SCOPED_TRACE(stall);
    auto observed = std::make_shared<ReaderObservations>();
    auto hud = std::make_shared<FormationHud>();
    Graph graph{"reader-retry"};
    MaxTensorExtents extents;
    extents.max_pulses = 1;
    extents.max_samples_per_pulse = 8;
    auto reader = graph.op<CPHDReaderOp<double>>("reader", extents, hud);
    CphdReaderParams p;
    p.path = CollectionPaths(path.string());
    p.coordinate_frame = CoordinateFrame(CoordinateFrameSpec{.frame = CoordinateFrameType::kEcef});
    reader.set_params(p);
    const auto history = graph.op<ReaderTensorSink>("history", observed, 0U, false);
    const auto positions = graph.op<ReaderTensorSink>("positions", observed, 1U, stall);
    const auto reference = graph.op<ReaderTensorSink>("reference", observed, 2U, false);
    const auto boundary = graph.op<ReaderBoundarySink>("boundary", observed);
    const ConnectionOptions queue{.queue_policy = QueuePolicy::kBlocking, .queue_depth = 1U};
    graph.add_flow(reader->phase_history, history->in, queue);
    graph.add_flow(reader->positions, positions->in, queue);
    graph.add_flow(reader->reference_position, reference->in, queue);
    graph.add_flow(reader->stream_boundary, boundary->in, queue);
    graph.partition("reader").add(reader);
    graph.partition("history").add(history);
    graph.partition("positions").add(positions);
    graph.partition("reference").add(reference);
    graph.partition("boundary").add(boundary);
    graph.set_default_clock(graph.add_clock<RealtimeClock>("clock"));
    CompileOptions options;
    for (const char* lane : {"reader", "history", "positions", "reference", "boundary"})
      options.deployment.bind_thread_placement(lane, ThreadPlacement{});
    for (const char* port : {"phase_history", "positions", "reference_position", "sample_axis",
                             "pulse_times", "velocities"})
      options.deployment.bind_tensor_output_device(TensorOutputDevicePlacement{
          .operator_path = "reader", .output_port = port, .device = DeviceId{0}});
    const auto plan = compile(graph, std::move(options));
    ASSERT_TRUE(plan.ok()) << plan.json();
    auto session = run_async(plan);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{5};
    while (std::chrono::steady_clock::now() < deadline) {
      if (stall ? observed->received.load() >= 1U
                : observed->received.load() == 20U && hud->snapshot().reader_stopped)
        break;
      std::this_thread::sleep_for(std::chrono::milliseconds{5});
    }
    if (stall)
      std::this_thread::sleep_for(std::chrono::milliseconds{100});
    session.request_stop();
    EXPECT_TRUE(session.wait_for(std::chrono::seconds{1}));
    session.wait();
    EXPECT_TRUE(session.diagnostics().empty());
    const std::lock_guard lock(observed->mutex);
    if (stall) {
      // History has advanced beyond positions: this actually exercised partial publication.
      EXPECT_GT(observed->stamps[0].size(), observed->stamps[1].size());
      continue;
    }
    EXPECT_TRUE(hud->snapshot().reader_stopped);
    for (const auto& stamps : observed->stamps) {
      ASSERT_EQ(stamps.size(), 20U);
      EXPECT_EQ(stamps, observed->stamps[0]);
      for (std::size_t i = 1; i < stamps.size(); ++i)
        EXPECT_GT(stamps[i].timestamp_ns, stamps[i - 1].timestamp_ns);
    }
    ASSERT_EQ(observed->boundaries.size(), 20U);
    EXPECT_EQ(observed->boundaries.back(), 2U);
  }
  fs::remove(path);
}

TEST(CphdReaderOp, UnmeteredWaitsOnFullQueueUnlessPaced) {
  using holoscan::radar::CphdReaderParams;
  using holoscan::radar::cphd_reader_waits_on_full;
  CphdReaderParams params;
  EXPECT_TRUE(cphd_reader_waits_on_full(params));
  params.playback_rate = 1.0;
  EXPECT_FALSE(cphd_reader_waits_on_full(params));
  params.wait_for_downstream = true;
  EXPECT_TRUE(cphd_reader_waits_on_full(params));
  params.wait_for_downstream = false;
  params.playback_rate = 0.0;
  EXPECT_TRUE(cphd_reader_waits_on_full(params));
}

TEST(CphdReaderOp, DutyCycleQuietTimeScalesWithPlaybackRate) {
  using holoscan::radar::CPHDReaderOp;
  using holoscan::radar::CphdReaderParams;
  using holoscan::radar::CoordinateFrame;
  using holoscan::radar::CoordinateFrameSpec;
  using holoscan::radar::CoordinateFrameType;
  using holoscan::radar::FormationHud;
  using holoscan::radar::MaxTensorExtents;
  const fs::path path = fs::temp_directory_path() / "holoscan_sar_reader_cooldown.cphd";
  make_fixture(path);
  struct Case {
    double rate;
    double duty;
    double quiet_s;
  };
  // Five pulses at 10 ms spacing span 40 ms. A 1/26 duty cycle adds
  // 1000 ms of radar quiet time, scaled along with playback (except rate 0).
  const std::array<Case, 5> cases{{{2.0, 1.0 / 26.0, 0.5},
                                   {0.5, 1.0 / 26.0, 2.0},
                                   {1.0, 1.0 / 26.0, 1.0},
                                   {0.0, 1.0 / 26.0, 1.0},
                                   {2.0, 1.0, 0.0}}};
  for (const auto& sample : cases) {
    SCOPED_TRACE(sample.rate);
    SCOPED_TRACE(sample.duty);
    MaxTensorExtents extents;
    extents.max_pulses = 5;
    extents.max_samples_per_pulse = 8;
    auto hud = std::make_shared<FormationHud>();
    auto fixture = holoscan::testing::create_operator_test<CPHDReaderOp<double>>(extents, hud);
    CphdReaderParams params;
    params.path = holoscan::radar::CollectionPaths(path.string());
    params.coordinate_frame =
        CoordinateFrame(CoordinateFrameSpec{.frame = CoordinateFrameType::kEcef});
    params.pulses_per_tick = 5;
    params.loop = true;
    params.playback_rate = sample.rate;
    params.duty_cycle = sample.duty;
    fixture.set_params(params);
    fixture.capture_snapshot<holoscan::Tensor>("phase_history", 2U,
                                               5U * 8U * sizeof(std::complex<double>));
    fixture.capture_snapshot<holoscan::Tensor>("positions", 2U, 5U * 3U * sizeof(double));
    fixture.capture_snapshot<holoscan::Tensor>("reference_position", 2U, 5U * 3U * sizeof(double));
    fixture.capture<uint8_t>("stream_boundary", 2U);
    for (const char* port : {"phase_history", "positions", "reference_position", "sample_axis",
                             "pulse_times", "velocities"}) {
      fixture.output_device(port, holoscan::DeviceId{0});
    }
    auto running = fixture.run_async({.expect_stop = true});
    const bool observed = running.wait_for_output("stream_boundary", 2U, std::chrono::seconds{5});
    running.request_stop();
    const auto result = running.finish();
    ASSERT_TRUE(observed) << result.describe_failure();
    ASSERT_TRUE(result.ok()) << result.describe_failure();
    const auto snapshot = hud->snapshot();
    ASSERT_GE(snapshot.spans.size(), 2U);
    const auto& first = snapshot.spans.front();
    if (sample.quiet_s == 0.0) {
      EXPECT_FALSE(first.quiet_start.has_value());
      EXPECT_FALSE(first.quiet_end.has_value());
    } else {
      ASSERT_TRUE(first.quiet_start.has_value());
      ASSERT_TRUE(first.quiet_end.has_value());
      const double actual =
          std::chrono::duration<double>(*first.quiet_end - *first.quiet_start).count();
      // Observe the actual sleep interval, with slack for host scheduling jitter.
      EXPECT_NEAR(actual, sample.quiet_s, 0.15);
    }
  }
  fs::remove(path);
}

TEST(CphdReaderOp, FirstBatchAsapAndStopInterruptsPacingQuietAndEof) {
  using holoscan::radar::CPHDReaderOp;
  using holoscan::radar::CphdReaderParams;
  using holoscan::radar::CoordinateFrame;
  using holoscan::radar::CoordinateFrameSpec;
  using holoscan::radar::CoordinateFrameType;
  using holoscan::radar::FormationHud;
  using holoscan::radar::MaxTensorExtents;
  const auto path = fs::temp_directory_path() / "holoscan_reader_cancel.cphd";
  make_fixture(path);
  for (int state = 0; state != 3; ++state) {
    SCOPED_TRACE(state);
    MaxTensorExtents extents;
    extents.max_pulses = 5;
    extents.max_samples_per_pulse = 8;
    auto hud = std::make_shared<FormationHud>();
    auto fixture = holoscan::testing::create_operator_test<CPHDReaderOp<double>>(extents, hud);
    CphdReaderParams p;
    p.path = holoscan::radar::CollectionPaths(path.string());
    p.coordinate_frame = CoordinateFrame(CoordinateFrameSpec{.frame = CoordinateFrameType::kEcef});
    p.pulses_per_tick = 5;
    p.playback_rate = state == 0 ? 0.0001 : 0.0;
    p.loop = state == 1;
    p.duty_cycle = state == 1 ? 0.0001 : 1.0;
    fixture.set_params(p);
    fixture.capture_snapshot<holoscan::Tensor>("phase_history", 1U, 5U * 8U * 16U);
    fixture.capture_snapshot<holoscan::Tensor>("positions", 1U, 5U * 3U * 8U);
    fixture.capture_snapshot<holoscan::Tensor>("reference_position", 1U, 5U * 3U * 8U);
    fixture.capture<uint8_t>("stream_boundary", 1U);
    for (const char* port : {"phase_history", "positions", "reference_position", "sample_axis",
                             "pulse_times", "velocities"})
      fixture.output_device(port, holoscan::DeviceId{0});
    auto running = fixture.run_async({.expect_stop = true});
    // Even the 400-second playback case must publish its first batch ASAP.
    EXPECT_TRUE(running.wait_for_output("stream_boundary", 1U, std::chrono::seconds{2}));
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{1};
    while (state != 0 && std::chrono::steady_clock::now() < deadline) {
      const auto snapshot = hud->snapshot();
      if (state == 1 ? !snapshot.spans.empty() && snapshot.spans.back().quiet_start.has_value()
                     : snapshot.reader_stopped)
        break;
      std::this_thread::sleep_for(std::chrono::milliseconds{1});
    }
    const auto before = std::chrono::steady_clock::now();
    running.request_stop();
    const auto result = running.finish();
    EXPECT_LT(std::chrono::steady_clock::now() - before, std::chrono::milliseconds{500});
    EXPECT_TRUE(result.ok()) << result.describe_failure();
    const auto snapshot = hud->snapshot();
    if (state == 0) {
      EXPECT_FALSE(snapshot.reader_stopped);
      EXPECT_FALSE(snapshot.spans.back().dwell_end.has_value());
    } else if (state == 1) {
      EXPECT_TRUE(snapshot.spans.back().quiet_start.has_value());
      EXPECT_FALSE(snapshot.spans.back().quiet_end.has_value());
    } else {
      EXPECT_TRUE(snapshot.reader_stopped);
    }
  }
  fs::remove(path);
}

TEST(CphdFile, CppFixtureDecodesHeaderPvpAndSignal) {
  const fs::path path = fs::temp_directory_path() / "holoscan_sar_cphd_cpp_fixture.cphd";
  make_fixture(path);
  const CphdFile file(path.string());
  ASSERT_EQ(file.num_channels(), 1);
  EXPECT_EQ(file.channel(0).num_vectors, 5);
  EXPECT_EQ(file.channel(0).num_samples, 8);
  EXPECT_EQ(file.sample_format(), holoscan::radar::cphd::SampleFormat::kCF8);
  const auto pvp = file.read_pvp_block(0, 1, 3);
  const auto pos = pvp.field("TxPos", 3);
  ASSERT_EQ(pos.size(), 9U);
  EXPECT_DOUBLE_EQ(pos[1], 1.0);
  EXPECT_DOUBLE_EQ(pos[7], 3.0);
  const auto sc0 = pvp.field("SC0", 1);
  ASSERT_EQ(sc0.size(), 3U);
  EXPECT_DOUBLE_EQ(sc0[0], 9.0e9);
  const auto signal = file.read_signal(0, 1, 3);
  ASSERT_EQ(signal.size(), 24U);
  EXPECT_NEAR(signal[0].real(), std::cos(0.1), 1e-6);
  EXPECT_NEAR(signal[0].imag(), std::sin(0.1), 1e-6);
  EXPECT_NEAR(signal[23].real(), std::cos(1.7), 1e-6);
  fs::remove(path);
}
TEST(CphdReaderOp, AcceptsRecordedGridRampButRejectsFrequencyStep) {
  using holoscan::radar::CPHDReaderOp;
  using holoscan::radar::CphdReaderParams;
  using holoscan::radar::CoordinateFrame;
  using holoscan::radar::CoordinateFrameSpec;
  using holoscan::radar::CoordinateFrameType;
  using holoscan::radar::MaxTensorExtents;
  const fs::path ramp = fs::temp_directory_path() / "holoscan_sar_reader_ramp.cphd";
  const fs::path step = fs::temp_directory_path() / "holoscan_sar_reader_step.cphd";
  make_fixture(ramp, 5.0, 0.03);
  make_fixture(step, 110000.0);
  const auto run_reader = [](const fs::path& path) {
    MaxTensorExtents extents;
    extents.max_pulses = 5;
    extents.max_samples_per_pulse = 8;
    auto fixture = holoscan::testing::create_operator_test<CPHDReaderOp<double>>(extents);
    CphdReaderParams params;
    params.path = holoscan::radar::CollectionPaths(path.string());
    params.coordinate_frame =
        CoordinateFrame(CoordinateFrameSpec{.frame = CoordinateFrameType::kEcef});
    params.pulses_per_tick = 5;
    fixture.set_params(params);
    fixture.capture_snapshot<holoscan::Tensor>("phase_history", 1U,
                                               5U * 8U * sizeof(std::complex<double>));
    fixture.capture_snapshot<holoscan::Tensor>("positions", 1U, 5U * 3U * sizeof(double));
    fixture.capture_snapshot<holoscan::Tensor>("reference_position", 1U, 5U * 3U * sizeof(double));
    fixture.capture<uint8_t>("stream_boundary", 1U);
    for (const char* port : {"phase_history", "positions", "reference_position", "sample_axis",
                             "pulse_times", "velocities"}) {
      fixture.output_device(port, holoscan::DeviceId{0});
    }
    // The reader becomes idle after D22 so a downstream aperture can drain;
    // isolated operator tests must explicitly stop after observing the boundary.
    holoscan::radar::testing::FixtureCompletion completion;
    auto running = fixture.run_async(completion.run_options());
    (void)completion.wait_for_output(running, "stream_boundary", 1U);
    running.request_stop();
    auto fixture_result = running.finish();
    completion.check(fixture_result);
    return fixture_result;
  };
  const auto accepted = run_reader(ramp);
  ASSERT_TRUE(accepted.ok()) << accepted.describe_failure();
  const auto boundaries = accepted.samples<uint8_t>("stream_boundary");
  ASSERT_EQ(boundaries.size(), 1U);
  EXPECT_EQ(boundaries[0].data, 2U);
  const auto rejected = run_reader(step);
  EXPECT_FALSE(rejected.ok());
  EXPECT_NE(rejected.describe_failure().find("SC0 varies across a pulse batch"), std::string::npos);
  fs::remove(ramp);
  fs::remove(step);
}

TEST(CphdReaderOp, PrefetchedReadFailureFollowsTheValidOutputPrefix) {
  using holoscan::radar::CPHDReaderOp;
  using holoscan::radar::CphdReaderParams;
  using holoscan::radar::CoordinateFrame;
  using holoscan::radar::CoordinateFrameSpec;
  using holoscan::radar::CoordinateFrameType;
  using holoscan::radar::MaxTensorExtents;
  const auto path = fs::temp_directory_path() / "holoscan_reader_prefetch_error.cphd";
  make_fixture(path);
  // Preserve the header/PVPs and first two pulses, but truncate the next signal block.
  fs::resize_file(path, 8192 + 2 * 8 * 8);
  MaxTensorExtents extents;
  extents.max_pulses = 2;
  extents.max_samples_per_pulse = 8;
  auto fixture = holoscan::testing::create_operator_test<CPHDReaderOp<double>>(extents);
  CphdReaderParams p;
  p.path = holoscan::radar::CollectionPaths(path.string());
  p.coordinate_frame = CoordinateFrame(CoordinateFrameSpec{.frame = CoordinateFrameType::kEcef});
  p.pulses_per_tick = 2;
  // Leave enough pacing time for the next batch's failure to be prefetched.
  p.playback_rate = 0.1;
  p.wait_for_downstream = true;
  fixture.set_params(p);
  fixture.capture_snapshot<holoscan::Tensor>("phase_history", 3U, 2U * 8U * 16U);
  fixture.capture_snapshot<holoscan::Tensor>("positions", 3U, 2U * 3U * 8U);
  fixture.capture_snapshot<holoscan::Tensor>("reference_position", 3U, 2U * 3U * 8U);
  fixture.capture<uint8_t>("stream_boundary", 3U);
  for (const char* port : {"phase_history", "positions", "reference_position", "sample_axis",
                           "pulse_times", "velocities"})
    fixture.output_device(port, holoscan::DeviceId{0});
  auto running = fixture.run_async({.expect_stop = true});
  EXPECT_TRUE(running.wait_for_output("stream_boundary", 1U, std::chrono::seconds{2}));
  // Wait for the failure rather than externally cancelling during pacing.
  EXPECT_FALSE(running.wait_for_output("stream_boundary", 2U, std::chrono::seconds{2}));
  const auto result = running.finish();
  EXPECT_FALSE(result.ok());
  EXPECT_NE(result.describe_failure().find("signal block"), std::string::npos);
  EXPECT_EQ(result.snapshots<holoscan::Tensor>("phase_history").size(), 1U);
  const auto boundaries = result.samples<uint8_t>("stream_boundary");
  ASSERT_EQ(boundaries.size(), 1U);
  EXPECT_EQ(boundaries[0].data, 0U);
  fs::remove(path);
}

TEST(CphdReaderOp, EmitsEachBatchAndTypedEndBoundary) {
  using holoscan::radar::CPHDReaderOp;
  using holoscan::radar::CphdReaderParams;
  using holoscan::radar::CoordinateFrame;
  using holoscan::radar::CoordinateFrameSpec;
  using holoscan::radar::CoordinateFrameType;
  using holoscan::radar::MaxTensorExtents;
  const fs::path path = fs::temp_directory_path() / "holoscan_sar_reader_native.cphd";
  make_fixture(path);
  MaxTensorExtents extents;
  extents.max_pulses = 2;
  extents.max_samples_per_pulse = 8;
  auto fixture = holoscan::testing::create_operator_test<CPHDReaderOp<float>>(extents);
  CphdReaderParams params;
  params.path = holoscan::radar::CollectionPaths(path.string());
  params.coordinate_frame =
      CoordinateFrame(CoordinateFrameSpec{.frame = CoordinateFrameType::kEcef});
  params.pulses_per_tick = 2;
  fixture.set_params(params);
  fixture.capture_snapshot<holoscan::Tensor>("phase_history", 3U,
                                             2U * 8U * sizeof(std::complex<float>));
  fixture.capture_snapshot<holoscan::Tensor>("positions", 3U, 2U * 3U * sizeof(float));
  fixture.capture_snapshot<holoscan::Tensor>("reference_position", 3U, 2U * 3U * sizeof(float));
  fixture.capture<uint8_t>("stream_boundary", 3U);
  for (const char* port : {"phase_history", "positions", "reference_position", "sample_axis",
                           "pulse_times", "velocities"}) {
    fixture.output_device(port, holoscan::DeviceId{0});
  }
  holoscan::radar::testing::FixtureCompletion completion;
  auto running = fixture.run_async(completion.run_options());
  const bool observed = completion.wait_for_output(running, "stream_boundary", 3U);
  running.request_stop();
  const auto result = running.finish();
  completion.check(result);
  ASSERT_TRUE(observed) << result.describe_failure();
  ASSERT_TRUE(result.ok()) << result.describe_failure();
  const auto boundaries = result.samples<uint8_t>("stream_boundary");
  ASSERT_EQ(boundaries.size(), 3U);
  EXPECT_EQ(boundaries[0].data, 0U);
  EXPECT_EQ(boundaries[1].data, 0U);
  EXPECT_EQ(boundaries[2].data, 2U);
  const auto outputs = result.snapshots<holoscan::Tensor>("phase_history");
  ASSERT_EQ(outputs.size(), 3U);
  EXPECT_EQ(outputs[0].shape(), (std::vector<int64_t>{2, 8}));
  EXPECT_EQ(outputs[2].shape(), (std::vector<int64_t>{1, 8}));
  for (std::size_t batch = 0; batch < outputs.size(); ++batch) {
    const int64_t first = static_cast<int64_t>(batch) * 2;
    const auto expected_signal =
        CphdFile(path.string()).read_signal(0, first, std::min<int64_t>(2, 5 - first));
    const auto host = holoscan::radar::testing::to_host(outputs[batch]);
    ASSERT_EQ(host.size(), static_cast<int64_t>(expected_signal.size()));
    const auto* samples = host.as<std::complex<float>>();
    for (std::size_t i = 0; i < expected_signal.size(); ++i) {
      EXPECT_NEAR(samples[i].real(), expected_signal[i].real(), 1e-5F) << batch << ":" << i;
      EXPECT_NEAR(samples[i].imag(), expected_signal[i].imag(), 1e-5F) << batch << ":" << i;
    }
  }
  fs::remove(path);
}
TEST(CphdReaderOp, MarksFinalBatchOfEachCollection) {
  using holoscan::radar::CPHDReaderOp;
  using holoscan::radar::CphdReaderParams;
  using holoscan::radar::CoordinateFrame;
  using holoscan::radar::CoordinateFrameSpec;
  using holoscan::radar::CoordinateFrameType;
  using holoscan::radar::MaxTensorExtents;
  const fs::path first = fs::temp_directory_path() / "holoscan_sar_reader_first.cphd";
  const fs::path second = fs::temp_directory_path() / "holoscan_sar_reader_second.cphd";
  make_fixture(first);
  make_fixture(second);
  {
    std::ofstream out(second, std::ios::binary | std::ios::in | std::ios::out);
    out.seekp(8192);
    holoscan::radar::testing::cphd_fixture::write_be32(out, 0.25F);
    holoscan::radar::testing::cphd_fixture::write_be32(out, 0.75F);
  }

  MaxTensorExtents extents;
  extents.max_pulses = 2;
  extents.max_samples_per_pulse = 8;
  auto fixture = holoscan::testing::create_operator_test<CPHDReaderOp<double>>(extents);
  CphdReaderParams params;
  params.path =
      holoscan::radar::CollectionPaths(std::vector<std::string>{first.string(), second.string()});
  params.coordinate_frame =
      CoordinateFrame(CoordinateFrameSpec{.frame = CoordinateFrameType::kEcef});
  params.pulses_per_tick = 2;
  fixture.set_params(params);
  fixture.capture_snapshot<holoscan::Tensor>("phase_history", 6U,
                                             2U * 8U * sizeof(std::complex<double>));
  fixture.capture_snapshot<holoscan::Tensor>("positions", 6U, 2U * 3U * sizeof(double));
  fixture.capture_snapshot<holoscan::Tensor>("reference_position", 6U, 2U * 3U * sizeof(double));
  fixture.capture<uint8_t>("stream_boundary", 6U);
  for (const char* port : {"phase_history", "positions", "reference_position", "sample_axis",
                           "pulse_times", "velocities"}) {
    fixture.output_device(port, holoscan::DeviceId{0});
  }
  holoscan::radar::testing::FixtureCompletion completion;
  auto running = fixture.run_async(completion.run_options());
  const bool observed = completion.wait_for_output(running, "stream_boundary", 6U);
  running.request_stop();
  const auto result = running.finish();
  completion.check(result);
  ASSERT_TRUE(observed) << result.describe_failure();
  ASSERT_TRUE(result.ok()) << result.describe_failure();
  const auto boundaries = result.samples<uint8_t>("stream_boundary");
  ASSERT_EQ(boundaries.size(), 6U);
  const std::vector<uint8_t> expected{0U, 0U, 1U, 0U, 0U, 2U};
  for (std::size_t i = 0; i < expected.size(); ++i)
    EXPECT_EQ(boundaries[i].data, expected[i]) << "batch " << i;
  const auto outputs = result.snapshots<holoscan::Tensor>("phase_history");
  ASSERT_EQ(outputs.size(), 6U);
  for (std::size_t batch = 0; batch < outputs.size(); ++batch) {
    const auto& path = batch < 3U ? first : second;
    const int64_t pulse = static_cast<int64_t>(batch % 3U) * 2;
    const auto expected_signal =
        CphdFile(path.string()).read_signal(0, pulse, std::min<int64_t>(2, 5 - pulse));
    const auto host = holoscan::radar::testing::to_host(outputs[batch]);
    ASSERT_EQ(host.size(), static_cast<int64_t>(expected_signal.size()));
    const auto* samples = host.as<std::complex<double>>();
    for (std::size_t i = 0; i < expected_signal.size(); ++i) {
      EXPECT_NEAR(samples[i].real(), expected_signal[i].real(), 1e-5) << batch << ":" << i;
      EXPECT_NEAR(samples[i].imag(), expected_signal[i].imag(), 1e-5) << batch << ":" << i;
    }
  }
  fs::remove(first);
  fs::remove(second);
}

}  // namespace
