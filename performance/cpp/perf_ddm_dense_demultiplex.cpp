// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Performance characterization for DDMDenseDemultiplexOp.
// Measurement only; no timing assertions.

#include <complex>
#include <algorithm>
#include <filesystem>  // NOLINT(build/c++17): repo builds with C++17
#include <stdexcept>
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

#include "fmcw/ddm_dense_demultiplex_op/ddm_dense_demultiplex_op.hpp"
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
  bool velocity_axis{false};
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
  return {scen.num_rx, scen.num_doppler, scen.num_range_bins};
}

perf::TimingCollector run_compensation_graph(const CompensationScenario& scen, int total_frames) {
  auto cube = perf::make_iq_cube_device<PerfPrecision>(input_shape(scen));
  const auto configure = [&](holoscan::Graph& graph, holoscan::CompileOptions& options) {
    hr::MaxTensorExtents extents;
    extents.max_tx = scen.num_tx;
    extents.max_rx = scen.num_rx;
    extents.max_doppler_bins = scen.num_doppler;
    extents.max_range_bins = scen.num_range_bins;
    auto op = graph.op<hr::DDMDenseDemultiplexOp<PerfPrecision>>("compensation", extents);
    hr::DDMDenseDemultiplexParams params;
    params.ddm_code = scen.non_linear_ddm ? make_ddm_nl_code(scen.num_tx, scen.num_doppler)
                                          : make_ddm_code(scen.num_tx);
    op.set_params(std::move(params));
    const auto sink = graph.op<perf::PerfTerminalSinkOp>("out_sink");
    const holoscan::ConnectionOptions q{.queue_depth = 4U};
    graph.add_flow(op->rd_maps_out, sink->in, q);
    perf::bind_tensor_out(options, "compensation", "velocity_bins_out");
    perf::bind_tensor_out(options, "compensation", "rd_maps_out");
    return op;
  };
  if (scen.velocity_axis) {
    auto axis = perf::make_power_map_device<PerfPrecision>({scen.num_doppler});
    return perf::run_perf_graph_dual(
        std::move(axis), std::move(cube), total_frames, "compensation-perf",
        [&](holoscan::Graph& graph, const auto& src, holoscan::CompileOptions& options) {
          auto op = configure(graph, options);
          const holoscan::ConnectionOptions q{.queue_depth = 4U};
          graph.add_flow(src->out_a, op->velocity_bins, q);
          graph.add_flow(src->out_b, op->rd_maps, q);
          const auto axis_sink = graph.op<perf::PerfNullSinkOp>("velocity_axis_sink");
          graph.add_flow(op->velocity_bins_out, axis_sink->in, q);
        });
  }
  return perf::run_perf_graph(
      std::move(cube), total_frames, "compensation-perf",
      [&](holoscan::Graph& graph, const auto& src, holoscan::CompileOptions& options) {
        auto op = configure(graph, options);
        const holoscan::ConnectionOptions q{.queue_depth = 4U};
        graph.add_flow(src->out, op->rd_maps, q);
      });
}

void fill_throughput(perf::ResultRow* row, const CompensationScenario& scen, double median_ms) {
  if (median_ms <= 0.0) {
    return;
  }
  const double sec = median_ms / 1000.0;
  const double tx_factor = 1.0;
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
  row.op_name = "DDMDenseDemultiplexOp";
  row.variant = scen.variant + (scen.velocity_axis ? "_axis" : "");
  row.size_id = scen.size_id;
  row.input_shape = input_shape(scen);
  row.k = scen.num_doppler;
  // This scenario measures the Params-selected waveform configuration.
  row.notes = "live_override=not_measured";
  return row;
}

CompensationScenario make_scenario(const SizeSpec& size, bool non_linear = false,
                                   bool velocity_axis = false) {
  CompensationScenario scenario;
  scenario.size_id = size.id;
  scenario.variant = non_linear ? "ddm_nl" : "ddm";
  scenario.num_tx = size.num_tx;
  scenario.num_rx = size.num_rx;
  scenario.num_doppler = size.num_doppler;
  scenario.num_range_bins = size.num_range_bins;
  scenario.non_linear_ddm = non_linear;
  scenario.velocity_axis = velocity_axis;
  scenario.id =
      "ddm_dense_demultiplex_" + scenario.variant + "_" + size.id + (velocity_axis ? "_axis" : "");
  return scenario;
}

struct CompensationTraits : perf::PerfDriverDefaults {
  using Scenario = CompensationScenario;

  static const char* tool_name() { return "perf_ddm_dense_demultiplex"; }
  static const char* tool_description() {
    return "DDMDenseDemultiplexOp performance "
           "characterization (HSDK 5 Graph)";
  }
  static const char* sizes_all_help() {
    return "Linear and nonlinear DDM size sweep across the full ladder";
  }
  static const char* sizes_subset_help() {
    return "Linear and nonlinear DDM sweep across an explicit subset";
  }
  static const char* full_help() { return "Full linear and nonlinear DDM size matrix"; }

  static perf::ProfileSuite suite() {
    return perf::load_profile_suite(
        std::filesystem::path(HOLOSCAN_RADAR_PERF_PROFILE_DIR) / "ddm_dense_demultiplex.yaml",
        "ddm_dense_demultiplex");
  }

  static std::vector<perf::SizeDef> size_ladder_defs() {
    const auto profiles = suite();
    std::vector<perf::SizeDef> result;
    for (size_t i = 0; i < perf::kProfiles.size(); ++i) {
      const auto& shape = profiles.shapes[i];
      if (shape.size() != 4)
        throw std::invalid_argument("DDMDenseDemultiplex profile rank mismatch");
      result.push_back({std::string(perf::kProfiles[i]), shape,
                        shape[0] * shape[1] * shape[2] * shape[3],
                        std::string(perf::kProfiles[i])});
    }
    return result;
  }

  static std::vector<CompensationScenario> build_profile_matrix(const std::string& scope) {
    const auto profiles = suite();
    const auto selected = profiles.select(scope);
    std::vector<CompensationScenario> result;
    for (const auto& c : profiles.cases) {
      size_t i = 0;
      for (; i < perf::kProfiles.size(); ++i) {
        if (c.profile == perf::kProfiles[i])
          break;
      }
      const auto& shape = profiles.shapes.at(i);
      if (shape.size() != 4 || shape[0] < 2 || shape[2] % shape[0] != 0)
        throw std::invalid_argument("DDMDenseDemultiplex invalid profile shape");
      perf::require_keys(c.parameters, {"code", "velocity_axis"}, "parameters");
      if (!c.parameters["code"] || !c.parameters["velocity_axis"] || c.override_value)
        throw std::invalid_argument("DDMDenseDemultiplex case needs static code and axis mode");
      const auto code = c.parameters["code"].as<std::string>();
      if (code != "linear" && code != "nonlinear")
        throw std::invalid_argument("DDMDenseDemultiplex code must be linear or nonlinear");
      const bool axis = c.parameters["velocity_axis"].as<bool>();
      const bool chosen =
          std::any_of(selected.begin(), selected.end(), [&](const perf::ProfileCase& item) {
            return item.id == c.id;
          });
      if (!chosen)
        continue;
      const SizeSpec size{c.profile, shape[0], shape[1], shape[2], shape[3], c.profile};
      auto scenario = make_scenario(size, code == "nonlinear", axis);
      scenario.id = c.id;
      scenario.size_id = c.profile;
      result.push_back(std::move(scenario));
    }
    return result;
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
    if (scen.non_linear_ddm) {
      row.notes =
          "DDM nonlinear phase-code correlation; cold_ms includes weights "
          "and FFT setup; output keeps the full Doppler axis";
    } else {
      row.notes =
          "DDM linear per-band gather; cold_ms includes the initial "
          "demultiplex table build";
    }
    row.notes += scen.velocity_axis ? "; velocity_axis=wired" : "; velocity_axis=absent";
    report->results.push_back(row);
  }

  static void process_warmup(const perf::RunSettings&) {
    perf::run_process_warmup([] {
      perf::PerfReport discarded;
      perf::RunSettings warm;
      warm.warmup_frames = 0;
      warm.measured_frames = 2;
      warm.repeats = 1;
      run_scenario(build_profile_matrix("minimal").front(), warm, &discarded);
    });
  }
};

}  // namespace

int main(int argc, char** argv) {
  return holoscan::radar::perf::perf_main<CompensationTraits>(argc, argv);
}
