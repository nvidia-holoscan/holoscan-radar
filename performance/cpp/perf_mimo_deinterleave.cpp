// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Performance characterization for MIMODeinterleaveOp — TDM slow-time gather
// into an explicit TX axis. Sweeps (M_tx, N_rx, L, samples) and a leading
// batch axis. No [live] parameters. Measurement tool only.

#include <cuda_runtime_api.h>

#include <complex>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/holoscan.hpp>

#include "fmcw/mimo_deinterleave_op/mimo_deinterleave_op.hpp"
#include "radar_common/descriptors/tx_schedule.hpp"
#include "utils/perf_app.hpp"
#include "utils/perf_driver.hpp"
#include "utils/perf_platform.hpp"
#include "utils/perf_process_warmup.hpp"
#include "utils/perf_report.hpp"
#include "utils/perf_runner.hpp"
#include "utils/perf_sink.hpp"
#include "utils/perf_source.hpp"
#include "utils/tensor_factory.hpp"

namespace hr = holoscan::radar;
namespace perf = holoscan::radar::perf;

namespace {

using PerfPrecision = float;

struct SizeSpec {
  std::string id;
  int64_t num_tx;
  int64_t num_rx;
  int64_t chirps_per_tx;
  int64_t samples;
  std::string role;
};

struct DeinterleaveScenario {
  std::string id;
  std::string size_id;
  int64_t num_tx{0};
  int64_t num_rx{0};
  int64_t chirps_per_tx{0};
  int64_t samples{0};
  int64_t batch{1};
};

const std::vector<SizeSpec>& size_ladder() {
  static const std::vector<SizeSpec> ladder = {
      {"xs", 2, 4, 8, 64, "small TDM; fast smoke"},
      {"sm", 4, 8, 16, 128, "compact MIMO"},
      {"md", 4, 16, 32, 256, "common dev size"},
      {"lg", 8, 32, 64, 512, "moderate MIMO"},
      {"xl", 8, 64, 128, 1024, "upper characterization bound"}};
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

std::vector<int64_t> interleaved_shape(const DeinterleaveScenario& scen) {
  const int64_t slow = scen.num_tx * scen.chirps_per_tx;
  if (scen.batch > 1) {
    return {scen.batch, scen.num_rx, slow, scen.samples};
  }
  return {scen.num_rx, slow, scen.samples};
}

template <typename T>
class DeinterleavePerfApp : public holoscan::Application {
 public:
  const DeinterleaveScenario* scen{nullptr};
  perf::TimingCollector* collector{nullptr};
  holoscan::TensorMap frame;
  int total_frames{25};

  void compose() override {
    using holoscan::Arg;
    hr::TXSchedule schedule;
    schedule.num_tx = scen->num_tx;
    auto op = make_operator<hr::MIMODeinterleaveOp<T>>(
        "deinterleave", Arg("tx_schedule") = schedule);

    auto src = make_operator<perf::PerfTensorSourceOp<perf::AdcCubePort>>(
        "src", make_condition<holoscan::CountCondition>(total_frames));
    src->configure(collector, frame);
    add_flow(src, op, {{"adc_cube", "adc_cube"}});

    auto sink = make_operator<perf::PerfTerminalSinkOp>("sink");
    sink->configure(collector);
    add_flow(op, sink, {{"adc_cube", "input"}});
  }
};

void fill_throughput(perf::ResultRow* row, const DeinterleaveScenario& scen,
                     double median_ms) {
  if (median_ms <= 0.0) {
    return;
  }
  const double sec = median_ms / 1000.0;
  const double batch = static_cast<double>(scen.batch);
  const double elems = batch * static_cast<double>(scen.num_tx) *
                       static_cast<double>(scen.num_rx) *
                       static_cast<double>(scen.chirps_per_tx) *
                       static_cast<double>(scen.samples);
  row->cells_per_second = elems / sec;
  const double csz = sizeof(std::complex<PerfPrecision>);
  const double slow = static_cast<double>(scen.num_tx * scen.chirps_per_tx);
  const double in_bytes =
      batch * static_cast<double>(scen.num_rx) * slow *
      static_cast<double>(scen.samples) * csz;
  const double out_bytes = elems * csz;
  row->estimated_effective_gbps = (in_bytes + out_bytes) / sec / 1e9;
}

perf::ResultRow base_row(const DeinterleaveScenario& scen) {
  perf::ResultRow row;
  row.scenario_id = scen.id;
  row.op_name = "MIMODeinterleaveOp";
  row.variant = scen.batch > 1 ? "batch" : "rr";
  row.size_id = scen.size_id;
  row.input_shape = interleaved_shape(scen);
  row.out_batch = scen.batch;
  row.k = scen.chirps_per_tx;
  return row;
}

DeinterleaveScenario make_scenario(const SizeSpec& s, int64_t batch = 1) {
  DeinterleaveScenario sc;
  sc.size_id = s.id;
  sc.num_tx = s.num_tx;
  sc.num_rx = s.num_rx;
  sc.chirps_per_tx = s.chirps_per_tx;
  sc.samples = s.samples;
  sc.batch = batch;
  std::ostringstream id;
  id << "deinterleave_rr_" << s.id;
  if (batch > 1) {
    id << "_b" << batch;
  }
  sc.id = id.str();
  return sc;
}

struct DeinterleaveTraits : perf::PerfDriverDefaults {
  using Scenario = DeinterleaveScenario;

  static const char* tool_name() { return "perf_mimo_deinterleave"; }
  static const char* tool_description() {
    return "MIMODeinterleaveOp performance characterization";
  }
  static const char* sizes_all_help() {
    return "Round-robin TDM gather across the full ladder";
  }
  static const char* sizes_subset_help() {
    return "Round-robin TDM gather across an explicit subset";
  }
  static const char* full_help() {
    return "Full matrix (sizes + batch axis on md/lg)";
  }

  static std::vector<perf::SizeDef> size_ladder_defs() {
    std::vector<perf::SizeDef> out;
    for (const auto& s : size_ladder()) {
      perf::SizeDef def;
      def.id = s.id;
      const int64_t slow = s.num_tx * s.chirps_per_tx;
      def.shape = {s.num_rx, slow, s.samples};
      def.cells = s.num_rx * slow * s.samples;
      def.role = s.role;
      out.push_back(def);
    }
    return out;
  }

  static std::vector<DeinterleaveScenario> build_default_matrix() {
    std::vector<DeinterleaveScenario> out;
    for (const char* sid : {"xs", "md", "lg", "xl"}) {
      out.push_back(make_scenario(must_find_size(sid)));
    }
    out.push_back(make_scenario(must_find_size("md"), 4));
    return out;
  }

  static std::vector<DeinterleaveScenario> build_size_sweep(
      const std::vector<std::string>& sizes) {
    std::vector<DeinterleaveScenario> out;
    for (const auto& sid : sizes) {
      const SizeSpec* s = find_size(sid);
      if (s == nullptr) {
        std::cerr << "warning: unknown size id '" << sid << "', skipping\n";
        continue;
      }
      out.push_back(make_scenario(*s));
    }
    return out;
  }

  static std::vector<DeinterleaveScenario> build_full_matrix() {
    std::vector<DeinterleaveScenario> out;
    for (const auto& s : size_ladder()) {
      out.push_back(make_scenario(s));
    }
    for (const char* sid : {"md", "lg"}) {
      out.push_back(make_scenario(must_find_size(sid), 4));
    }
    return out;
  }

  static void run_scenario(const DeinterleaveScenario& scen,
                           const perf::RunSettings& rs,
                           perf::PerfReport* report) {
    const perf::RunPlan plan = perf::make_run_plan(rs, /*has_override=*/false);
    const auto samples = perf::collect_scenario_samples(
        rs, plan, [&](int /*rep*/, const perf::RunPlan& p) {
          perf::TimingCollector collector;
          auto cube = perf::single_map(
              perf::make_iq_cube_device<PerfPrecision>(interleaved_shape(scen)),
              "adc_cube");

          collector.t_construct = perf::Clock::now();
          auto app =
              holoscan::make_application<DeinterleavePerfApp<PerfPrecision>>();
          app->scen = &scen;
          app->collector = &collector;
          app->frame = std::move(cube);
          app->total_frames = p.total_frames;
          app->run();
          return collector;
        });

    perf::append_setup_row(base_row(scen), samples, report);

    perf::ResultRow row = base_row(scen);
    perf::fill_compute_row(&row, samples);
    fill_throughput(&row, scen, row.latency.median_ms);
    row.notes =
        "pure gather marshaling; cold_ms is first-frame latency; "
        "median/p90 are steady state; cells/s counts complex samples moved";
    report->results.push_back(row);
  }

  static void process_warmup(const perf::RunSettings&) {
    perf::run_process_warmup([] {
      perf::PerfReport discarded;
      perf::RunSettings warm;
      warm.warmup_frames = 0;
      warm.measured_frames = 2;
      warm.repeats = 1;
      run_scenario(make_scenario(must_find_size("xs")), warm, &discarded);
    });
  }
};

}  // namespace

int main(int argc, char** argv) {
  return holoscan::radar::perf::perf_main<DeinterleaveTraits>(argc, argv);
}
