// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Performance characterization for ClutterFilterOp — slow-time static-clutter
// suppression (mean subtraction / 1st- / 2nd-order delay-line MTI). Sweeps the
// (chirps x range_bins) size ladder, the suppression method, and a
// multi-channel batch.
//
// [EA HSDK5 ONLY] Graph-based rewrite of the HSDK 4.x Application harness.
// Mid-session [live] method override measurement is deferred until
// docs/planning/HSDK5_PORT.md §1 (EA partition / device binding). Params
// still select the method under test. Measurement tool only: no timing
// assertions. See docs/planning/OPERATOR_DEVELOPMENT.md.

#include <cuda_runtime_api.h>

#include <chrono>
#include <complex>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "fmcw/clutter_filter_op/clutter_filter_op.hpp"
#include "radar_common/descriptors/enums.hpp"
#include "utils/perf_app.hpp"
#include "utils/perf_driver.hpp"
#include "utils/perf_platform.hpp"
#include "utils/perf_process_warmup.hpp"
#include "utils/perf_report.hpp"
#include "utils/perf_runner.hpp"
#include "utils/tensor_factory.hpp"

namespace hr = holoscan::radar;
namespace perf = holoscan::radar::perf;

namespace {

using PerfPrecision = float;  // v1 characterizes single precision (complex64).

inline constexpr auto kPerfTick = std::chrono::milliseconds{1};

struct SizeSpec {
  std::string id;
  int64_t num_chirps;
  int64_t num_range_bins;
  std::string role;
};

struct ClutterScenario {
  std::string id;
  std::string size_id;
  int64_t num_chirps{0};
  int64_t num_range_bins{0};
  int64_t channels{1};
  hr::ClutterFilterMethod method{hr::ClutterFilterMethod::kMeanSubtraction};
};

const std::vector<SizeSpec>& size_ladder() {
  static const std::vector<SizeSpec> ladder = {
      {"xs", 16, 128, "small map; fast smoke"},
      {"sm", 32, 256, "compact sensor / few chirps"},
      {"md", 64, 512, "common dev size"},
      {"lg", 128, 1024, "moderate production-scale"},
      {"xl", 256, 2048, "range-heavy moderate (v1 upper bound)"}};
  return ladder;
}

const SizeSpec* find_size(const std::string& id) {
  for (const auto& s : size_ladder()) {
    if (s.id == id) {
      return &s;
    }
  }
  return nullptr;
}

const SizeSpec& must_find_size(const std::string& id) {
  const SizeSpec* s = find_size(id);
  if (s == nullptr) {
    std::cerr << "internal error: unknown size id '" << id << "'\n";
    std::abort();
  }
  return *s;
}

std::string method_tag(hr::ClutterFilterMethod m) {
  switch (m) {
    case hr::ClutterFilterMethod::kMeanSubtraction: return "msub";
    case hr::ClutterFilterMethod::kSingleDelay: return "sdel";
    case hr::ClutterFilterMethod::kDoubleDelay: return "ddel";
    case hr::ClutterFilterMethod::kNone: return "none";
  }
  return "unk";
}

std::vector<int64_t> profiles_shape(const ClutterScenario& scen) {
  if (scen.channels > 1) {
    return {scen.channels, scen.num_chirps, scen.num_range_bins};
  }
  return {scen.num_chirps, scen.num_range_bins};
}

// ---------------------------------------------------------------------------
// Graph source / terminal sink with TimingCollector hooks.
// ---------------------------------------------------------------------------
struct PerfFrameBank {
  static void reset(std::shared_ptr<holoscan::Tensor> frame, int count,
                    perf::TimingCollector* collector) {
    const std::lock_guard lock(mutex);
    input_frame = std::move(frame);
    frames_remaining = count;
    captured = 0;
    collector_ = collector;
  }

  static inline std::mutex mutex;
  static inline std::condition_variable ready;
  static inline std::shared_ptr<holoscan::Tensor> input_frame;
  static inline int frames_remaining{0};
  static inline int captured{0};
  static inline perf::TimingCollector* collector_{nullptr};
};

class PerfTensorSourceOp final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    constexpr holoscan::TensorPortLayout kDevice{
        .memory_kind = holoscan::MemoryKind::kCudaDevice,
        .dtype = std::nullopt,
        .rank = std::nullopt,
    };
    spec.output(out, "out")
        .produces_tensor(kDevice)
        .tensor_allocation(holoscan::TensorAllocationBounds{
            .max_rank = 8U, .max_byte_span = 256ULL * 1024ULL * 1024ULL});
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = kPerfTick});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    std::shared_ptr<holoscan::Tensor> frame;
    {
      const std::lock_guard lock(PerfFrameBank::mutex);
      if (PerfFrameBank::frames_remaining <= 0 ||
          PerfFrameBank::input_frame == nullptr) {
        return {};
      }
      --PerfFrameBank::frames_remaining;
      frame = PerfFrameBank::input_frame;
      if (PerfFrameBank::collector_ != nullptr) {
        if (!PerfFrameBank::collector_->first_tick_set) {
          PerfFrameBank::collector_->t_first_tick = perf::Clock::now();
          PerfFrameBank::collector_->first_tick_set = true;
        }
        PerfFrameBank::collector_->starts.push_back(perf::Clock::now());
      }
    }

    const auto shape = frame->shape();
    const DLDataType dtype = frame->dtype();
    auto loan = out.allocate_tensor(
        holoscan::TensorLoanRequest{.shape = shape, .dtype = dtype});
    if (!loan) {
      return holoscan::make_unexpected(std::move(loan).error());
    }
    auto writer = loan->write(context.cuda_stream());
    if (!writer) {
      return holoscan::make_unexpected(std::move(writer).error());
    }
    cudaStream_t stream = context.cuda_stream().get();
    if (cudaMemcpyAsync(writer->data(), frame->data(), writer->byte_size(),
                        cudaMemcpyDeviceToDevice, stream) != cudaSuccess) {
      return holoscan::make_unexpected(
          holoscan::Error{holoscan::ErrorCode::kFailure,
                          "PerfTensorSourceOp: cudaMemcpyAsync failed"});
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

class PerfTerminalSinkOp final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(in, "in").queue_depth(8U);
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
    // Include all GPU work the operator enqueued for this frame.
    cudaDeviceSynchronize();
    {
      const std::lock_guard lock(PerfFrameBank::mutex);
      if (PerfFrameBank::collector_ != nullptr) {
        PerfFrameBank::collector_->ends.push_back(perf::Clock::now());
      }
      ++PerfFrameBank::captured;
    }
    PerfFrameBank::ready.notify_all();
    return {};
  }

  holoscan::Input<holoscan::Tensor> in{"in"};
};

void bind_device_pools(holoscan::CompileOptions& options) {
  options.deployment.bind_tensor_output_device(
      holoscan::TensorOutputDevicePlacement{
          .operator_path = "src",
          .output_port = "out",
          .device = holoscan::DeviceId{0},
      });
  options.deployment.bind_tensor_output_device(
      holoscan::TensorOutputDevicePlacement{
          .operator_path = "clutter",
          .output_port = "range_profiles_out",
          .device = holoscan::DeviceId{0},
      });
}

perf::TimingCollector run_clutter_graph(
    const ClutterScenario& scen, int total_frames) {
  perf::TimingCollector collector;
  collector.t_construct = perf::Clock::now();

  auto cube =
      perf::make_iq_cube_device<PerfPrecision>(profiles_shape(scen));
  PerfFrameBank::reset(std::move(cube), total_frames, &collector);

  holoscan::Graph graph{"clutter-perf"};
  const auto src = graph.op<PerfTensorSourceOp>("src");
  auto op = graph.op<hr::ClutterFilterOp<PerfPrecision>>("clutter");
  hr::ClutterFilterParams params;
  params.method = scen.method;
  op.set_params(std::move(params));
  const auto sink = graph.op<PerfTerminalSinkOp>("out_sink");
  graph.add_flow(src->out, op->range_profiles,
                 holoscan::ConnectionOptions{.queue_depth = 4U});
  graph.add_flow(op->range_profiles_out, sink->in,
                 holoscan::ConnectionOptions{.queue_depth = 4U});
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

  holoscan::CompileOptions options;
  bind_device_pools(options);
  const holoscan::ExecutionPlan plan =
      holoscan::compile(graph, std::move(options));
  if (!plan.ok()) {
    throw std::runtime_error(
        std::string("clutter perf compile failed: ") +
        std::string(plan.json()));
  }

  holoscan::RunSession session = holoscan::run_async(plan);
  {
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds{60};
    std::unique_lock lock(PerfFrameBank::mutex);
    while (PerfFrameBank::captured < total_frames &&
           session.diagnostics().empty() &&
           std::chrono::steady_clock::now() < deadline) {
      PerfFrameBank::ready.wait_for(lock, std::chrono::milliseconds{25});
    }
  }
  session.request_stop();
  session.wait();
  if (!session.diagnostics().empty()) {
    std::string msg = "clutter perf run diagnostics:";
    for (const auto& d : session.diagnostics()) {
      msg += " ";
      msg += std::string(d.token);
    }
    throw std::runtime_error(msg);
  }
  if (collector.ends.size() < static_cast<size_t>(total_frames)) {
    throw std::runtime_error(
        "clutter perf: expected " + std::to_string(total_frames) +
        " frames, got " + std::to_string(collector.ends.size()));
  }
  return collector;
}

void fill_throughput(perf::ResultRow* row, const ClutterScenario& scen,
                     double median_ms) {
  if (median_ms <= 0.0) {
    return;
  }
  const double sec = median_ms / 1000.0;
  const double batch = static_cast<double>(scen.channels);
  const double chirps = static_cast<double>(scen.num_chirps);
  const double bins = static_cast<double>(scen.num_range_bins);
  row->cells_per_second = batch * chirps * bins / sec;
  row->slices_per_second = batch * bins / sec;

  const double csz = sizeof(std::complex<PerfPrecision>);
  const double bytes = 2.0 * batch * chirps * bins * csz;
  row->estimated_effective_gbps = bytes / sec / 1e9;
}

perf::ResultRow base_row(const ClutterScenario& scen) {
  perf::ResultRow row;
  row.scenario_id = scen.id;
  row.op_name = "ClutterFilterOp";
  row.variant = method_tag(scen.method);
  row.size_id = scen.size_id;
  row.map_shape = {scen.num_chirps, scen.num_range_bins};
  row.input_shape = profiles_shape(scen);
  row.out_batch = scen.channels;
  row.k = scen.num_chirps;
  // [EA HSDK5 ONLY] live override timing deferred — HSDK5_PORT.md §1.
  row.notes = "live_override=deferred_ea";
  return row;
}

ClutterScenario make_scenario(hr::ClutterFilterMethod method, const SizeSpec& s,
                              int64_t channels = 1) {
  ClutterScenario sc;
  sc.size_id = s.id;
  sc.num_chirps = s.num_chirps;
  sc.num_range_bins = s.num_range_bins;
  sc.channels = channels;
  sc.method = method;
  std::ostringstream id;
  id << "clutter_" << method_tag(method) << "_" << s.id;
  if (channels > 1) {
    id << "_ch" << channels;
  }
  id << "_compute";
  sc.id = id.str();
  return sc;
}

struct ClutterTraits : perf::PerfDriverDefaults {
  using Scenario = ClutterScenario;

  static const char* tool_name() { return "perf_clutter_filter"; }
  static const char* tool_description() {
    return "ClutterFilterOp performance characterization (HSDK 5 Graph)";
  }
  static const char* sizes_all_help() {
    return "Mean-subtraction size sweep across the full ladder";
  }
  static const char* sizes_subset_help() {
    return "Mean-subtraction size sweep across an explicit subset";
  }
  static const char* full_help() {
    return "Full matrix (methods x sizes x channel batch); live-override "
           "timing deferred on HSDK 5 EA (HSDK5_PORT.md §1)";
  }

  static std::vector<perf::SizeDef> size_ladder_defs() {
    std::vector<perf::SizeDef> out;
    for (const auto& s : size_ladder()) {
      perf::SizeDef def;
      def.id = s.id;
      def.shape = {s.num_chirps, s.num_range_bins};
      def.cells = s.num_chirps * s.num_range_bins;
      def.role = s.role;
      out.push_back(def);
    }
    return out;
  }

  static std::vector<ClutterScenario> build_default_matrix() {
    std::vector<ClutterScenario> out;
    for (const char* sid : {"xs", "md", "lg", "xl"}) {
      out.push_back(make_scenario(hr::ClutterFilterMethod::kMeanSubtraction,
                                  must_find_size(sid)));
    }
    const SizeSpec& md = must_find_size("md");
    out.push_back(make_scenario(hr::ClutterFilterMethod::kSingleDelay, md));
    out.push_back(make_scenario(hr::ClutterFilterMethod::kDoubleDelay, md));
    out.push_back(make_scenario(hr::ClutterFilterMethod::kMeanSubtraction, md,
                                4));
    return out;
  }

  static std::vector<ClutterScenario> build_size_sweep(
      const std::vector<std::string>& sizes) {
    std::vector<ClutterScenario> out;
    for (const auto& sid : sizes) {
      const SizeSpec* s = find_size(sid);
      if (s == nullptr) {
        std::cerr << "warning: unknown size id '" << sid << "', skipping\n";
        continue;
      }
      out.push_back(
          make_scenario(hr::ClutterFilterMethod::kMeanSubtraction, *s));
    }
    return out;
  }

  static std::vector<ClutterScenario> build_full_matrix() {
    std::vector<ClutterScenario> out;
    const auto methods = {hr::ClutterFilterMethod::kMeanSubtraction,
                          hr::ClutterFilterMethod::kSingleDelay,
                          hr::ClutterFilterMethod::kDoubleDelay};
    for (const auto& s : size_ladder()) {
      for (auto m : methods) {
        out.push_back(make_scenario(m, s));
      }
    }
    for (const char* sid : {"md", "lg"}) {
      const SizeSpec& s = must_find_size(sid);
      for (int64_t ch : {4, 16}) {
        out.push_back(
            make_scenario(hr::ClutterFilterMethod::kMeanSubtraction, s, ch));
      }
    }
    return out;
  }

  static void run_scenario(const ClutterScenario& scen,
                           const perf::RunSettings& rs,
                           perf::PerfReport* report) {
    // No mid-session live override until HSDK5_PORT.md §1 is closed.
    const perf::RunPlan plan = perf::make_run_plan(rs, /*has_override=*/false);
    const auto samples = perf::collect_scenario_samples(
        rs, plan, [&](int /*rep*/, const perf::RunPlan& p) {
          return run_clutter_graph(scen, p.total_frames);
        });

    perf::append_setup_row(base_row(scen), samples, report);

    perf::ResultRow row = base_row(scen);
    perf::fill_compute_row(&row, samples);
    fill_throughput(&row, scen, row.latency.median_ms);
    row.notes =
        "cold_ms is first-frame latency; median/p90 are post-warmup steady "
        "state; live method-override timing deferred (HSDK5_PORT.md §1); "
        "cells/s is complex cells/s";
    report->results.push_back(row);
  }

  static void process_warmup(const perf::RunSettings&) {
    perf::run_process_warmup([] {
      perf::PerfReport discarded;
      perf::RunSettings warm;
      warm.warmup_frames = 0;
      warm.measured_frames = 2;
      warm.repeats = 1;
      run_scenario(make_scenario(hr::ClutterFilterMethod::kMeanSubtraction,
                                 must_find_size("xs")),
                   warm, &discarded);
    });
  }
};

}  // namespace

int main(int argc, char** argv) {
  return holoscan::radar::perf::perf_main<ClutterTraits>(argc, argv);
}
