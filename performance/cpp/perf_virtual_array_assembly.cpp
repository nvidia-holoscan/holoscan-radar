// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Performance characterization for VirtualArrayAssemblyOp — flatten separated
// (tx, rx) RD maps into a virtual channel axis and emit runtime array_geometry.
// Sweeps (M_tx, N_rx, num_doppler, num_range_bins), a leading batch axis, and
// inlines a mid-session [live] wavelength override. Measurement tool only.

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

#include "fmcw/virtual_array_assembly_op/virtual_array_assembly_op.hpp"
#include "radar_common/descriptors/enums.hpp"
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

constexpr double kDefaultWavelength = 0.03;
constexpr double kOverrideWavelength = 0.025;

struct SizeSpec {
  std::string id;
  int64_t num_tx;
  int64_t num_rx;
  int64_t num_doppler;
  int64_t num_range_bins;
  std::string role;
};

struct AssemblyScenario {
  std::string id;
  std::string size_id;
  int64_t num_tx{0};
  int64_t num_rx{0};
  int64_t num_doppler{0};
  int64_t num_range_bins{0};
  int64_t batch{1};
  hr::OverlapPolicy policy{hr::OverlapPolicy::kDedup};
  int64_t override_at{0};
};

const std::vector<SizeSpec>& size_ladder() {
  static const std::vector<SizeSpec> ladder = {
      {"xs", 2, 4, 32, 128, "small MIMO; fast smoke"},
      {"sm", 4, 8, 64, 256, "compact MIMO"},
      {"md", 4, 16, 128, 512, "common dev size"},
      {"lg", 8, 32, 256, 1024, "moderate MIMO"},
      {"xl", 8, 64, 512, 2048, "upper characterization bound"}};
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

std::vector<double> linear_positions(int64_t count, double spacing) {
  std::vector<double> out(static_cast<size_t>(count * 3));
  for (int64_t i = 0; i < count; ++i) {
    const size_t base = static_cast<size_t>(i * 3);
    out[base] = 0.0;
    out[base + 1] = static_cast<double>(i) * spacing;
    out[base + 2] = 0.0;
  }
  return out;
}

std::string policy_tag(hr::OverlapPolicy policy) {
  switch (policy) {
    case hr::OverlapPolicy::kDedup: return "dedup";
    case hr::OverlapPolicy::kAverage: return "avg";
    case hr::OverlapPolicy::kKeepAll: return "keep";
  }
  return "unk";
}

std::vector<int64_t> separated_shape(const AssemblyScenario& scen) {
  if (scen.batch > 1) {
    return {scen.batch, scen.num_tx, scen.num_rx, scen.num_doppler,
            scen.num_range_bins};
  }
  return {scen.num_tx, scen.num_rx, scen.num_doppler, scen.num_range_bins};
}

template <typename T>
class VirtualAssemblyPerfApp : public holoscan::Application {
 public:
  const AssemblyScenario* scen{nullptr};
  perf::TimingCollector* collector{nullptr};
  holoscan::TensorMap frame;
  std::vector<double> tx_positions;
  std::vector<double> rx_positions;
  double override_wavelength{kOverrideWavelength};
  int64_t override_at{0};
  int total_frames{25};

  void compose() override {
    using holoscan::Arg;
    auto op = make_operator<hr::VirtualArrayAssemblyOp<T>>(
        "virtual_array", Arg("tx_positions") = tx_positions,
        Arg("rx_positions") = rx_positions,
        Arg("wavelength") = kDefaultWavelength,
        Arg("overlap_policy") = scen->policy);

    auto src = make_operator<perf::PerfTensorSourceOp<perf::RdMapsPort>>(
        "src", make_condition<holoscan::CountCondition>(total_frames));
    src->configure(collector, frame);
    add_flow(src, op, {{"rd_maps", "rd_maps"}});

    auto rd_sink = make_operator<perf::PerfTerminalSinkOp>("rd_sink");
    rd_sink->configure(collector);
    add_flow(op, rd_sink, {{"rd_maps", "input"}});

    auto geom_drain = make_operator<perf::PerfNullSinkOp>("geom_drain");
    add_flow(op, geom_drain, {{"array_geometry", "input"}});

    auto ovr = make_operator<perf::PerfOverrideSourceOp<double>>(
        "ovr", make_condition<holoscan::CountCondition>(total_frames));
    ovr->configure(override_at, override_wavelength);
    add_flow(ovr, op, {{"output", "wavelength"}});
  }
};

void fill_throughput(perf::ResultRow* row, const AssemblyScenario& scen,
                     double median_ms) {
  if (median_ms <= 0.0) {
    return;
  }
  const double sec = median_ms / 1000.0;
  const double batch = static_cast<double>(scen.batch);
  const double elems = batch * static_cast<double>(scen.num_tx) *
                       static_cast<double>(scen.num_rx) *
                       static_cast<double>(scen.num_doppler) *
                       static_cast<double>(scen.num_range_bins);
  row->cells_per_second = elems / sec;
  const double csz = sizeof(std::complex<PerfPrecision>);
  const double out_virtual =
      static_cast<double>(scen.num_tx * scen.num_rx);  // dedup upper bound
  const double in_bytes = elems * csz;
  const double out_bytes =
      batch * out_virtual * static_cast<double>(scen.num_doppler) *
      static_cast<double>(scen.num_range_bins) * csz;
  row->estimated_effective_gbps = (in_bytes + out_bytes) / sec / 1e9;
}

perf::ResultRow base_row(const AssemblyScenario& scen) {
  perf::ResultRow row;
  row.scenario_id = scen.id;
  row.op_name = "VirtualArrayAssemblyOp";
  row.variant = policy_tag(scen.policy);
  row.size_id = scen.size_id;
  row.input_shape = separated_shape(scen);
  row.out_batch = scen.batch;
  row.k = scen.num_doppler;
  row.notes = "live_override=wavelength";
  return row;
}

AssemblyScenario make_scenario(const SizeSpec& s, int64_t batch = 1,
                               hr::OverlapPolicy policy =
                                   hr::OverlapPolicy::kDedup) {
  AssemblyScenario sc;
  sc.size_id = s.id;
  sc.num_tx = s.num_tx;
  sc.num_rx = s.num_rx;
  sc.num_doppler = s.num_doppler;
  sc.num_range_bins = s.num_range_bins;
  sc.batch = batch;
  sc.policy = policy;
  std::ostringstream id;
  id << "virtual_" << policy_tag(policy) << "_" << s.id;
  if (batch > 1) {
    id << "_b" << batch;
  }
  sc.id = id.str();
  return sc;
}

struct VirtualAssemblyTraits : perf::PerfDriverDefaults {
  using Scenario = AssemblyScenario;

  static const char* tool_name() { return "perf_virtual_array_assembly"; }
  static const char* tool_description() {
    return "VirtualArrayAssemblyOp performance characterization";
  }
  static const char* sizes_all_help() {
    return "Dedup policy size sweep across the full ladder";
  }
  static const char* sizes_subset_help() {
    return "Dedup policy size sweep across an explicit subset";
  }
  static const char* full_help() {
    return "Full matrix (sizes + overlap policies + batch axis on md/lg)";
  }

  static std::vector<perf::SizeDef> size_ladder_defs() {
    std::vector<perf::SizeDef> out;
    for (const auto& s : size_ladder()) {
      perf::SizeDef def;
      def.id = s.id;
      def.shape = {s.num_tx, s.num_rx, s.num_doppler, s.num_range_bins};
      def.cells = s.num_tx * s.num_rx * s.num_doppler * s.num_range_bins;
      def.role = s.role;
      out.push_back(def);
    }
    return out;
  }

  static std::vector<AssemblyScenario> build_default_matrix() {
    std::vector<AssemblyScenario> out;
    for (const char* sid : {"xs", "md", "lg", "xl"}) {
      out.push_back(make_scenario(must_find_size(sid)));
    }
    out.push_back(make_scenario(must_find_size("md"), 4));
    return out;
  }

  static std::vector<AssemblyScenario> build_size_sweep(
      const std::vector<std::string>& sizes) {
    std::vector<AssemblyScenario> out;
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

  static std::vector<AssemblyScenario> build_full_matrix() {
    std::vector<AssemblyScenario> out;
    for (const auto& s : size_ladder()) {
      out.push_back(make_scenario(s));
    }
    const SizeSpec& md = must_find_size("md");
    out.push_back(make_scenario(md, 1, hr::OverlapPolicy::kAverage));
    out.push_back(make_scenario(md, 1, hr::OverlapPolicy::kKeepAll));
    for (const char* sid : {"md", "lg"}) {
      out.push_back(make_scenario(must_find_size(sid), 4));
    }
    return out;
  }

  static void run_scenario(const AssemblyScenario& scen,
                           const perf::RunSettings& rs,
                           perf::PerfReport* report) {
    const perf::RunPlan plan = perf::make_run_plan(rs, /*has_override=*/true);
    const auto samples = perf::collect_scenario_samples(
        rs, plan, [&](int /*rep*/, const perf::RunPlan& p) {
          perf::TimingCollector collector;
          auto cube = perf::single_map(
              perf::make_iq_cube_device<PerfPrecision>(separated_shape(scen)),
              "rd_maps");

          collector.t_construct = perf::Clock::now();
          auto app = holoscan::make_application<VirtualAssemblyPerfApp<
              PerfPrecision>>();
          app->scen = &scen;
          app->collector = &collector;
          app->frame = std::move(cube);
          app->tx_positions = linear_positions(scen.num_tx, 0.5);
          app->rx_positions = linear_positions(scen.num_rx, 0.5);
          app->override_at = p.override_at;
          app->total_frames = p.total_frames;
          app->run();
          return collector;
        });

    perf::append_setup_row(base_row(scen), samples, report);

    perf::ResultRow row = base_row(scen);
    perf::fill_compute_row(&row, samples);
    fill_throughput(&row, scen, row.latency.median_ms);
    row.notes =
        "virtual channel flatten + geometry emit; cold_ms is first-frame "
        "latency; median/p90 are pre-override steady state; override_ms is "
        "peak latency over the 3-frame live wavelength switch; cells/s counts "
        "input complex samples";
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
  return holoscan::radar::perf::perf_main<VirtualAssemblyTraits>(argc, argv);
}
