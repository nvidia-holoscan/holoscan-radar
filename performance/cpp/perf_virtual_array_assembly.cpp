// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Performance characterization for VirtualArrayAssemblyOp — flatten separated
// (tx, rx) RD maps into a virtual channel axis and emit runtime
// array_geometry. Sweeps (M_tx, N_rx, num_doppler, num_range_bins) and
// overlap policies. Measurement tool only: no timing
// assertions. See docs/planning/OPERATOR_DEVELOPMENT.md.
//
// Params select the overlap policy under test. The always-emitted geometry
// output is drained by PerfNullSinkOp.

#include <complex>
#include <filesystem>  // NOLINT(build/c++17): repo builds with C++17
#include <set>
#include <stdexcept>
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

#include "fmcw/virtual_array_assembly_op/virtual_array_assembly_op.hpp"
#include "radar_common/descriptors/enums.hpp"
#include "radar_common/descriptors/max_tensor_extents.hpp"
#include "radar_schemas/array_geometry_bridge.hpp"
#include "utils/perf_driver.hpp"
#include "utils/perf_graph.hpp"
#include "utils/perf_platform.hpp"
#include "utils/perf_report.hpp"
#include "utils/perf_runner.hpp"
#include "utils/tensor_factory.hpp"

namespace hr = holoscan::radar;
namespace perf = holoscan::radar::perf;

namespace {

using PerfPrecision = float;

constexpr double kDefaultWavelength = 0.03;

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
  hr::OverlapPolicy policy{hr::OverlapPolicy::kDedup};
  bool live_wavelength{false};
  double override_wavelength{kDefaultWavelength * 1.1};
};

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
    case hr::OverlapPolicy::kDedup:
      return "dedup";
    case hr::OverlapPolicy::kAverage:
      return "avg";
    case hr::OverlapPolicy::kKeepAll:
      return "keep";
  }
  return "unk";
}

// The contract input is exactly (M_tx, N_rx, num_doppler, num_range_bins).
// There is no leading batch rung: an extra leading dim is opaque batch, which
// the operator rejects (see VirtualArrayAssemblyOp.RejectsOpaqueLeadingBatch).
std::vector<int64_t> separated_shape(const AssemblyScenario& scen) {
  return {scen.num_tx, scen.num_rx, scen.num_doppler, scen.num_range_bins};
}

perf::TimingCollector run_assembly_graph(const AssemblyScenario& scen, const perf::RunPlan& plan) {
  auto cube = perf::make_iq_cube_device<PerfPrecision>(separated_shape(scen));
  // Value-agnostic gather: zero-fill loan is valid.
  const auto build = [&](holoscan::Graph& graph, const auto& src,
                         holoscan::CompileOptions& options) {
    hr::MaxTensorExtents extents;
    extents.max_tx = scen.num_tx;
    extents.max_rx = scen.num_rx;
    extents.max_doppler_bins = scen.num_doppler;
    extents.max_range_bins = scen.num_range_bins;
    auto op = graph.op<hr::VirtualArrayAssemblyOp<PerfPrecision>>("virtual_array", extents);
    hr::VirtualArrayAssemblyParams params;
    params.tx_positions = linear_positions(scen.num_tx, 0.5);
    params.rx_positions = linear_positions(scen.num_rx, 0.5);
    params.wavelength = kDefaultWavelength;
    params.overlap_policy = scen.policy;
    op.set_params(std::move(params));
    const auto sink = graph.op<perf::PerfTerminalSinkOp>("out_sink");
    const auto geom_drain = graph.op<perf::PerfPayloadSinkOp<hr::WireArrayGeometry>>("geom_drain");
    const holoscan::ConnectionOptions q{.queue_depth = 4U};
    graph.add_flow(src->out, op->rd_maps, q);
    if constexpr (requires { src->override_out; }) {
      graph.add_flow(src->override_out, op->wavelength, q);
    }
    graph.add_flow(op->rd_maps_out, sink->in, q);
    graph.add_flow(op->array_geometry, geom_drain->in, q);
    graph.partition("perf").add(src).add(op).add(sink).add(geom_drain);
    perf::bind_tensor_out(options, "virtual_array", "rd_maps_out");
  };
  if (scen.live_wavelength) {
    return perf::run_perf_graph_with_override(std::move(cube), plan.total_frames, plan.override_at,
                                              kDefaultWavelength, scen.override_wavelength,
                                              "assembly-perf", build,
                                              perf::PerfPublishMode::kFillInPlace, true);
  }
  return perf::run_perf_graph(std::move(cube), plan.total_frames, "assembly-perf", build,
                              perf::PerfPublishMode::kFillInPlace);
}

void fill_throughput(perf::ResultRow* row, const AssemblyScenario& scen, double median_ms) {
  if (median_ms <= 0.0) {
    return;
  }
  const double sec = median_ms / 1000.0;
  const double elems = static_cast<double>(scen.num_tx) * static_cast<double>(scen.num_rx) *
                       static_cast<double>(scen.num_doppler) *
                       static_cast<double>(scen.num_range_bins);
  row->cells_per_second = elems / sec;
  const double csz = sizeof(std::complex<PerfPrecision>);
  const double out_virtual = static_cast<double>(scen.num_tx * scen.num_rx);
  const double in_bytes = elems * csz;
  const double out_bytes = out_virtual * static_cast<double>(scen.num_doppler) *
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
  row.k = scen.num_doppler;
  row.live_port = scen.live_wavelength ? "wavelength" : "";
  return row;
}

AssemblyScenario make_scenario(const SizeSpec& s,
                               hr::OverlapPolicy policy = hr::OverlapPolicy::kDedup) {
  AssemblyScenario sc;
  sc.size_id = s.id;
  sc.num_tx = s.num_tx;
  sc.num_rx = s.num_rx;
  sc.num_doppler = s.num_doppler;
  sc.num_range_bins = s.num_range_bins;
  sc.policy = policy;
  std::ostringstream id;
  id << "virtual_" << policy_tag(policy) << "_" << s.id;
  sc.id = id.str();
  return sc;
}

struct VirtualAssemblyTraits : perf::PerfDriverDefaults {
  using Scenario = AssemblyScenario;

  static const char* tool_name() { return "perf_virtual_array_assembly"; }
  static const char* tool_description() {
    return "VirtualArrayAssemblyOp performance "
           "characterization (HSDK 5 Graph)";
  }

  static perf::ProfileSuite suite() {
    return perf::load_profile_suite(
        std::filesystem::path(HOLOSCAN_RADAR_PERF_PROFILE_DIR) / "virtual_array_assembly.yaml",
        "virtual_array_assembly");
  }

  static std::vector<perf::SizeDef> size_ladder_defs() {
    const auto profiles = suite();
    std::vector<perf::SizeDef> result;
    for (size_t i = 0; i < perf::kProfiles.size(); ++i) {
      const auto& shape = profiles.shapes[i];
      if (shape.size() != 4)
        throw std::invalid_argument("VirtualArrayAssembly profile rank mismatch");
      result.push_back({std::string(perf::kProfiles[i]), shape,
                        shape[0] * shape[1] * shape[2] * shape[3],
                        std::string(perf::kProfiles[i])});
    }
    return result;
  }

  static AssemblyScenario parse_case(const perf::ProfileCase& c,
                                     const std::vector<int64_t>& shape) {
    if (shape.size() != 4 || shape[0] < 1 || shape[1] < 1)
      throw std::invalid_argument("VirtualArrayAssembly invalid profile shape");
    perf::require_keys(c.parameters, {"overlap_policy"}, "parameters");
    if (!c.parameters["overlap_policy"])
      throw std::invalid_argument("VirtualArrayAssembly requires overlap_policy");
    const auto policy = c.parameters["overlap_policy"].as<std::string>();
    hr::OverlapPolicy mode;
    if (policy == "dedup")
      mode = hr::OverlapPolicy::kDedup;
    else if (policy == "average")
      mode = hr::OverlapPolicy::kAverage;
    else if (policy == "keep_all")
      mode = hr::OverlapPolicy::kKeepAll;
    else
      throw std::invalid_argument("VirtualArrayAssembly invalid overlap_policy");
    const SizeSpec size{c.profile, shape[0], shape[1], shape[2], shape[3], c.profile};
    auto scenario = make_scenario(size, mode);
    scenario.id = c.id;
    scenario.size_id = c.profile;
    if (c.override_value) {
      if ((*c.override_value)["port"].as<std::string>() != "wavelength")
        throw std::invalid_argument("VirtualArrayAssembly only wavelength is live");
      scenario.live_wavelength = true;
      scenario.override_wavelength = (*c.override_value)["value"].as<double>();
      if (scenario.override_wavelength <= 0.0)
        throw std::invalid_argument("VirtualArrayAssembly wavelength must be positive");
    }
    return scenario;
  }

  static void run_scenario(const AssemblyScenario& scen, const perf::RunSettings& rs,
                           perf::PerfReport* report) {
    perf::measure_scenario(
        rs, scen.live_wavelength,
        [&](int /*rep*/, const perf::RunPlan& p) {
          return run_assembly_graph(scen, p);
        },
        base_row(scen), report,
        [&](perf::ResultRow& row) {
          fill_throughput(&row, scen, row.latency.median_ms);
          row.notes =
              "virtual channel flatten + geometry emit; cells/s counts input complex samples";
          if (scen.live_wavelength)
            row.notes += "; one-frame wavelength change";
        });
  }
};

}  // namespace

int main(int argc, char** argv) {
  return holoscan::radar::perf::perf_main<VirtualAssemblyTraits>(argc, argv);
}
