// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Performance characterization for TDMDenseCompensationOp.
// Measurement only; no timing assertions.

#include <complex>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/graph.hpp>

#include "fmcw/tdm_dense_compensation_op/tdm_dense_compensation_op.hpp"
#include "radar_common/descriptors/ddm_code_descriptor.hpp"
#include "radar_common/descriptors/fmcw_waveform_descriptor.hpp"
#include "radar_common/descriptors/max_tensor_extents.hpp"
#include "utils/perf_driver.hpp"
#include "utils/perf_graph.hpp"
#include "utils/perf_platform.hpp"
#include "utils/perf_process_warmup.hpp"
#include "utils/perf_report.hpp"
#include "utils/perf_runner.hpp"
#include "utils/tensor_factory.hpp"

namespace hr = holoscan::radar;
namespace perf = holoscan::radar::perf;

namespace {

using PerfPrecision = float;

constexpr double kDefaultPri = 25.0e-6;

struct SizeSpec {
  std::string id;
  int64_t num_tx;
  int64_t num_rx;
  int64_t num_doppler;
  int64_t num_range_bins;
  std::string role;
};

struct CompensationScenario {
  std::string id;
  std::string size_id;
  std::string variant{"tdm"};  // tdm | ddm | ddm_nl
  int64_t num_tx{0};
  int64_t num_rx{0};
  int64_t num_doppler{0};
  int64_t num_range_bins{0};
  bool non_linear_ddm{false};
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

hr::FMCWWaveformDescriptor make_waveform(double pri) {
  hr::FMCWWaveformSpec spec;
  spec.carrier_frequency = 77.0e9;
  spec.bandwidth = 150.0e6;
  spec.chirp_duration = 20.0e-6;
  spec.sample_rate = 12.8e6;
  spec.chirp_repetition_interval = pri;
  return hr::FMCWWaveformDescriptor(spec);
}

hr::DDMCodeDescriptor make_ddm_code(int64_t num_tx) {
  hr::DDMCodeDescriptor code;
  code.num_tx = num_tx;
  code.doppler_offsets.resize(static_cast<size_t>(num_tx));
  for (int64_t m = 0; m < num_tx; ++m) {
    code.doppler_offsets[static_cast<size_t>(m)] =
        static_cast<double>(m) / static_cast<double>(num_tx);
  }
  code.guard_bands = 0;
  return code;
}

hr::DDMCodeDescriptor make_ddm_nl_code(int64_t num_tx, int64_t num_doppler) {
  hr::DDMCodeDescriptor code;
  code.num_tx = num_tx;
  code.guard_bands = 0;
  code.phase_codes.resize(static_cast<size_t>(num_tx * num_doppler));
  for (int64_t m = 0; m < num_tx; ++m) {
    for (int64_t k = 0; k < num_doppler; ++k) {
      const double phase = std::sin(0.7 * static_cast<double>(k) + static_cast<double>(m)) * 2.0;
      code.phase_codes[static_cast<size_t>(m * num_doppler + k)] =
          std::complex<double>(std::cos(phase), std::sin(phase));
    }
  }
  return code;
}

std::vector<int64_t> input_shape(const CompensationScenario& scen) {
  return {scen.num_tx, scen.num_rx, scen.num_doppler, scen.num_range_bins};
}

perf::TimingCollector run_compensation_graph(const CompensationScenario& scen, int total_frames) {
  auto cube = perf::make_iq_cube_device<PerfPrecision>(input_shape(scen));
  return perf::run_perf_graph(
      std::move(cube), total_frames, "compensation-perf",
      [&](holoscan::Graph& graph, const auto& src, holoscan::CompileOptions& options) {
        hr::MaxTensorExtents extents;
        extents.max_tx = scen.num_tx;
        extents.max_rx = scen.num_rx;
        extents.max_doppler_bins = scen.num_doppler;
        extents.max_range_bins = scen.num_range_bins;
        auto op = graph.op<hr::TDMDenseCompensationOp<PerfPrecision>>("compensation", extents);
        hr::TDMDenseCompensationParams params;
        params.waveform = make_waveform(kDefaultPri);
        op.set_params(std::move(params));
        const auto sink = graph.op<perf::PerfTerminalSinkOp>("out_sink");
        const holoscan::ConnectionOptions q{.queue_depth = 4U};
        graph.add_flow(src->out, op->rd_maps, q);
        graph.add_flow(op->rd_maps_out, sink->in, q);
        perf::bind_tensor_out(options, "compensation", "rd_maps_out");
      });
}

void fill_throughput(perf::ResultRow* row, const CompensationScenario& scen, double median_ms) {
  if (median_ms <= 0.0) {
    return;
  }
  const double sec = median_ms / 1000.0;
  const double tx_factor = static_cast<double>(scen.num_tx);
  const double elems = tx_factor * static_cast<double>(scen.num_rx) *
                       static_cast<double>(scen.num_doppler) *
                       static_cast<double>(scen.num_range_bins);
  row->cells_per_second = elems / sec;
  const double csz = sizeof(std::complex<PerfPrecision>);
  const double bytes = elems * csz;
  row->estimated_effective_gbps = 2.0 * bytes / sec / 1e9;
}

perf::ResultRow base_row(const CompensationScenario& scen) {
  perf::ResultRow row;
  row.scenario_id = scen.id;
  row.op_name = "TDMDenseCompensationOp";
  row.variant = scen.variant;
  row.size_id = scen.size_id;
  row.input_shape = input_shape(scen);
  row.k = scen.num_doppler;
  // This scenario measures the Params-selected waveform configuration.
  row.notes = "live_override=not_measured";
  return row;
}

CompensationScenario make_scenario(const SizeSpec& size) {
  CompensationScenario scenario;
  scenario.size_id = size.id;
  scenario.variant = "tdm";
  scenario.num_tx = size.num_tx;
  scenario.num_rx = size.num_rx;
  scenario.num_doppler = size.num_doppler;
  scenario.num_range_bins = size.num_range_bins;
  scenario.id = "tdm_dense_compensation_" + size.id;
  return scenario;
}

struct CompensationTraits : perf::PerfDriverDefaults {
  using Scenario = CompensationScenario;

  static const char* tool_name() { return "perf_tdm_dense_compensation"; }
  static const char* tool_description() {
    return "TDMDenseCompensationOp performance "
           "characterization (HSDK 5 Graph)";
  }
  static const char* sizes_all_help() {
    return "TDM dense compensation size sweep across the full ladder";
  }
  static const char* sizes_subset_help() {
    return "TDM dense compensation sweep across an explicit subset";
  }
  static const char* full_help() { return "Full TDM dense compensation size matrix"; }

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

  static std::vector<CompensationScenario> build_default_matrix() {
    std::vector<CompensationScenario> out;
    for (const char* sid : {"xs", "md", "lg", "xl"}) {
      out.push_back(make_scenario(must_find_size(sid)));
    }
    return out;
  }

  static std::vector<CompensationScenario> build_size_sweep(const std::vector<std::string>& sizes) {
    std::vector<CompensationScenario> out;
    for (const auto& sid : sizes) {
      const SizeSpec* size = find_size(sid);
      if (size != nullptr)
        out.push_back(make_scenario(*size));
    }
    return out;
  }

  static std::vector<CompensationScenario> build_full_matrix() {
    std::vector<CompensationScenario> out;
    for (const auto& size : size_ladder())
      out.push_back(make_scenario(size));
    return out;
  }

  static void run_scenario(const CompensationScenario& scen, const perf::RunSettings& rs,
                           perf::PerfReport* report) {
    const perf::RunPlan plan = perf::make_run_plan(rs, /*has_override=*/false);
    const auto samples =
        perf::collect_scenario_samples(rs, plan, [&](int /*rep*/, const perf::RunPlan& p) {
          return run_compensation_graph(scen, p.total_frames);
        });

    perf::append_setup_row(base_row(scen), samples, report);

    perf::ResultRow row = base_row(scen);
    perf::fill_compute_row(&row, samples);
    fill_throughput(&row, scen, row.latency.median_ms);
    row.notes =
        "TDM per-bin complex multiply; cold_ms includes the initial "
        "correction-table build; median/p90 are steady state; cells/s "
        "counts complex samples touched";
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
  return holoscan::radar::perf::perf_main<CompensationTraits>(argc, argv);
}
