// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Performance characterization for TDMDenseCompensationOp.
// Measurement only; no timing assertions.

#include <complex>
#include <set>
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

#include "fmcw/tdm_dense_compensation_op/tdm_dense_compensation_op.hpp"
#include "radar_common/descriptors/ddm_code_descriptor.hpp"
#include "radar_common/descriptors/fmcw_waveform_descriptor.hpp"
#include "radar_common/descriptors/max_tensor_extents.hpp"
#include "radar_schemas/fmcw_waveform_bridge.hpp"
#include "utils/perf_driver.hpp"
#include "utils/perf_graph.hpp"
#include "utils/perf_platform.hpp"
#include "utils/perf_report.hpp"
#include "utils/perf_recipe.hpp"
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
  double pri{kDefaultPri};
  bool live_waveform{false};
  double changed_pri{kDefaultPri * 1.1};
};

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

perf::TimingCollector run_graph(const CompensationScenario& scen, const perf::RunPlan& plan) {
  auto cube = perf::make_iq_cube_device<PerfPrecision>(input_shape(scen));
  const auto configure = [&](holoscan::Graph& graph, const auto& src,
                             holoscan::CompileOptions& options, auto connect) {
    hr::MaxTensorExtents extents;
    extents.max_tx = scen.num_tx;
    extents.max_rx = scen.num_rx;
    extents.max_doppler_bins = scen.num_doppler;
    extents.max_range_bins = scen.num_range_bins;
    auto op = graph.op<hr::TDMDenseCompensationOp<PerfPrecision>>("compensation", extents);
    hr::TDMDenseCompensationParams params;
    params.waveform = make_waveform(scen.pri);
    op.set_params(std::move(params));
    const holoscan::ConnectionOptions q{.queue_depth = 4U};
    graph.add_flow(src->out, op->rd_maps, q);
    connect(graph, src, op, q);
    perf::PerfRecipeOutputs outputs(graph, options, "compensation");
    outputs.timed(op->rd_maps_out, "rd_maps_out");
    outputs.finish(src, op);
  };
  if (scen.live_waveform) {
    const auto selected =
        perf::live_case("waveform_in", hr::to_wire(make_waveform(scen.pri)),
                        hr::to_wire(make_waveform(scen.changed_pri)),
                        [](auto& graph, const auto& src, const auto& op, const auto& q) {
                          graph.add_flow(src->override_out, op->waveform_in, q);
                        });
    return perf::run_selected_live_recipe(std::array{std::move(cube)}, plan, "compensation-perf",
                                          selected, configure, perf::PerfPublishMode::kCopy);
  }
  return perf::run_data_recipe(
      std::array{std::move(cube)}, plan, "compensation-perf",
      [&](auto& graph, const auto& src, auto& options) {
        configure(graph, src, options, [](auto&, const auto&, const auto&, const auto&) {});
      },
      perf::PerfPublishMode::kCopy);
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
  row.op_name = "TDMDenseCompensationOp";
  row.variant = scen.variant;
  row.input_shape = input_shape(scen);
  row.k = scen.num_doppler;
  row.live_port = scen.live_waveform ? "waveform" : "";
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

void decorate_row(perf::ResultRow& row, const CompensationScenario& scen) {
  fill_throughput(&row, scen, row.latency.median_ms);
  row.notes =
      "TDM per-bin complex multiply; cold_ms includes the initial "
      "correction-table build; median/p90 are steady state; cells/s "
      "counts complex samples touched";
}

struct CompensationTraits : perf::PerfDriverDefaults {
  using Scenario = CompensationScenario;

  static const char* tool_name() { return "perf_tdm_dense_compensation"; }
  static const char* tool_description() {
    return "TDMDenseCompensationOp performance "
           "characterization (HSDK 5 Graph)";
  }

  static perf::SizeDef size_def(const std::string& id, const std::vector<int64_t>& shape) {
    if (shape.size() != 4)
      throw std::invalid_argument("TDMDenseCompensation profile rank mismatch");
    return {id, shape, shape[0] * shape[1] * shape[2] * shape[3], id};
  }
  static CompensationScenario parse_case(const perf::ProfileCase& c,
                                         const std::vector<int64_t>& shape) {
    if (shape.size() != 4 || shape[0] < 2)
      throw std::invalid_argument("TDMDenseCompensation invalid profile shape");
    perf::require_keys(c.parameters, {"pri"}, "parameters");
    if (!c.parameters["pri"] || c.parameters["pri"].as<double>() <= 20.0e-6)
      throw std::invalid_argument("TDMDenseCompensation requires valid PRI");
    const SizeSpec size{c.profile, shape[0], shape[1], shape[2], shape[3], c.profile};
    auto scenario = make_scenario(size);
    scenario.pri = c.parameters["pri"].as<double>();
    scenario.id = c.id;
    scenario.size_id = c.profile;
    if (c.override_value) {
      if ((*c.override_value)["port"].as<std::string>() != "waveform")
        throw std::invalid_argument("TDMDenseCompensation only waveform is live");
      scenario.changed_pri = (*c.override_value)["value"].as<double>();
      if (scenario.changed_pri <= 20.0e-6)
        throw std::invalid_argument("TDMDenseCompensation override PRI too short");
      scenario.live_waveform = true;
    }
    return scenario;
  }
};

}  // namespace

int main(int argc, char** argv) {
  return holoscan::radar::perf::perf_main<CompensationTraits>(argc, argv);
}
