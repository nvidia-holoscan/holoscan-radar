// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Detection-count characterization for sparse TDM compensation.

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>  // NOLINT(build/c++17): repo builds with C++17
#include <set>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/graph.hpp>

#include "fmcw/tdm_sparse_compensation_op/tdm_sparse_compensation_op.hpp"
#include "radar_common/descriptors/fmcw_waveform_descriptor.hpp"
#include "radar_common/descriptors/max_tensor_extents.hpp"
#include "radar_common/tensor/dlpack_buffer.hpp"
#include "radar_common/tensor/tensor_utils.hpp"
#include "utils/perf_driver.hpp"
#include "utils/perf_graph.hpp"
#include "utils/perf_process_warmup.hpp"
#include "utils/perf_report.hpp"
#include "utils/perf_runner.hpp"
#include "utils/tensor_factory.hpp"

namespace hr = holoscan::radar;
namespace perf = holoscan::radar::perf;

namespace {

using Precision = float;
constexpr DLDataType kInt64Dtype{kDLInt, 64, 1};

struct SizeSpec {
  std::string id;
  int64_t tx;
  int64_t rx;
  int64_t doppler;
  int64_t range;
  int64_t detections;
  std::string role;
};

struct Scenario {
  std::string id;
  std::string size_id;
  SizeSpec size;
  std::string live_port;
  double pri{1.0e-3};
  double changed_pri{1.1e-3};
  int64_t window_doppler{0};
  int64_t window_range{0};
  int64_t changed_doppler{1};
  int64_t changed_range{1};
  hr::OverlapPolicy policy{hr::OverlapPolicy::kKeepAll};
};

const std::vector<SizeSpec>& size_ladder() {
  static const std::vector<SizeSpec> result{{"xs", 2, 4, 64, 256, 16, "smoke"},
                                            {"sm", 3, 4, 128, 512, 64, "sparse automotive"},
                                            {"md", 4, 8, 256, 512, 256, "common MIMO"},
                                            {"lg", 4, 16, 512, 1024, 1024, "large array"},
                                            {"xl", 8, 16, 512, 2048, 4096, "upper bound"}};
  return result;
}

const SizeSpec* find_size(const std::string& id) {
  for (const auto& size : size_ladder()) {
    if (size.id == id) {
      return &size;
    }
  }
  return nullptr;
}

hr::FMCWWaveformDescriptor waveform(double pri = 1.0e-3) {
  hr::FMCWWaveformSpec spec;
  spec.carrier_frequency = 77.0e9;
  spec.bandwidth = 1.0e9;
  spec.chirp_duration = 40.0e-6;
  spec.sample_rate = 20.0e6;
  spec.chirp_repetition_interval = pri;
  return hr::FMCWWaveformDescriptor(spec);
}

std::vector<double> ula_positions(int64_t count, double spacing) {
  std::vector<double> result(static_cast<std::size_t>(count * 3), 0.0);
  for (int64_t index = 0; index < count; ++index) {
    result[static_cast<std::size_t>(index * 3)] = spacing * static_cast<double>(index);
  }
  return result;
}

std::shared_ptr<holoscan::Tensor> make_detections(int64_t count, int64_t doppler, int64_t range) {
  std::vector<int64_t> host(static_cast<std::size_t>(count * 2));
  for (int64_t index = 0; index < count; ++index) {
    const int64_t cell = (index * 997) % (doppler * range);
    host[static_cast<std::size_t>(index * 2)] = cell / range;
    host[static_cast<std::size_t>(index * 2 + 1)] = cell % range;
  }
  auto result = hr::make_device_tensor({count, 2}, kInt64Dtype);
  if (!host.empty()) {
    auto pointer = result->mutable_data();
    if (!pointer || cudaMemcpy(*pointer, host.data(), host.size() * sizeof(int64_t),
                               cudaMemcpyHostToDevice) != cudaSuccess) {
      throw std::runtime_error("detection-index H2D failed");
    }
  }
  return result;
}

perf::TimingCollector run_graph(const Scenario& scenario, const perf::RunPlan& plan) {
  const auto& size = scenario.size;
  auto rd = perf::make_iq_cube_device<Precision>({size.tx, size.rx, size.doppler, size.range});
  auto idx = make_detections(size.detections, size.doppler, size.range);
  const auto configure = [&](holoscan::Graph& graph, const auto& source,
                             holoscan::CompileOptions& options, auto connect) {
    hr::MaxTensorExtents extents;
    extents.max_tx = size.tx;
    extents.max_rx = size.rx;
    extents.max_virtual_channels = size.tx * size.rx;
    extents.max_doppler_bins = size.doppler;
    extents.max_range_bins = size.range;
    extents.max_detections = size.detections;
    extents.max_snapshots = std::max<int64_t>(
        (2 * scenario.window_doppler + 1) * (2 * scenario.window_range + 1),
        scenario.live_port == "snapshot_window"
            ? (2 * scenario.changed_doppler + 1) * (2 * scenario.changed_range + 1)
            : 1);
    auto op = graph.op<hr::TDMSparseCompensationOp<Precision>>("sparse", extents);
    hr::TDMSparseCompensationParams params;
    params.waveform = waveform(scenario.pri);
    params.tx_positions = ula_positions(size.tx, 0.02);
    params.rx_positions = ula_positions(size.rx, 0.002);
    params.overlap_policy = scenario.policy;
    params.snapshot_window = {scenario.window_doppler, scenario.window_range};
    op.set_params(std::move(params));
    const auto terminal = graph.op<perf::PerfTerminalSinkOp>("terminal");
    const auto mask = graph.op<perf::PerfNullSinkOp>("mask");
    const auto geometry = graph.op<perf::PerfPayloadSinkOp<hr::WireArrayGeometry>>("geometry");
    const holoscan::ConnectionOptions queue{.queue_depth = 4U};
    graph.add_flow(source->out_a, op->rd_maps, queue);
    graph.add_flow(source->out_b, op->detection_idx, queue);
    connect(graph, source, op, queue);
    graph.add_flow(op->snapshot, terminal->in, queue);
    graph.add_flow(op->is_valid, mask->in, queue);
    graph.add_flow(op->array_geometry, geometry->in, queue);
    graph.partition("perf").add(source).add(op).add(terminal).add(mask).add(geometry);
    perf::bind_tensor_out(options, "sparse", "snapshot");
    perf::bind_tensor_out(options, "sparse", "is_valid");
  };
  if (scenario.live_port == "waveform") {
    return perf::run_perf_graph_dual_with_override(
        std::move(rd), std::move(idx), plan.total_frames, plan.override_at,
        hr::to_wire(waveform(scenario.pri)), hr::to_wire(waveform(scenario.changed_pri)),
        "tdm-sparse-compensation-perf",
        [&](auto& graph, const auto& source, auto& options) {
          configure(graph, source, options,
                    [](auto& g, const auto& src, const auto& op, const auto& q) {
                      g.add_flow(src->override_out, op->waveform_in, q);
                    });
        },
        perf::PerfPublishMode::kCopy, true);
  }
  if (scenario.live_port == "snapshot_window") {
    const hr::TDMSparseWindowOverride initial{.doppler = scenario.window_doppler,
                                              .range = scenario.window_range};
    const hr::TDMSparseWindowOverride changed{.doppler = scenario.changed_doppler,
                                              .range = scenario.changed_range};
    return perf::run_perf_graph_dual_with_override(
        std::move(rd), std::move(idx), plan.total_frames, plan.override_at, initial, changed,
        "tdm-sparse-compensation-perf",
        [&](auto& graph, const auto& source, auto& options) {
          configure(graph, source, options,
                    [](auto& g, const auto& src, const auto& op, const auto& q) {
                      g.add_flow(src->override_out, op->snapshot_window_in, q);
                    });
        },
        perf::PerfPublishMode::kCopy, true);
  }
  return perf::run_perf_graph_dual(
      std::move(rd), std::move(idx), plan.total_frames, "tdm-sparse-compensation-perf",
      [&](auto& graph, const auto& source, auto& options) {
        configure(graph, source, options, [](auto&, const auto&, const auto&, const auto&) {});
      });
}

perf::ResultRow base_row(const Scenario& scenario) {
  const auto& size = scenario.size;
  perf::ResultRow row;
  row.scenario_id = scenario.id;
  row.op_name = "TDMSparseCompensationOp";
  row.variant = "sparse_tdm";
  row.live_port = scenario.live_port;
  row.size_id = size.id;
  row.map_shape = {size.doppler, size.range};
  row.input_shape = {size.tx, size.rx, size.doppler, size.range};
  row.out_batch = size.detections;
  row.k = size.tx * size.rx;
  row.notes = "end-to-end Graph latency; cells/s is detections/s";
  return row;
}

struct Traits : perf::PerfDriverDefaults {
  using Scenario = ::Scenario;

  static const char* tool_name() { return "perf_tdm_sparse_compensation"; }
  static const char* tool_description() {
    return "Sparse TDM compensation detection-count characterization";
  }
  static const char* sizes_all_help() { return "Run the full sparse TDM size ladder"; }
  static const char* sizes_subset_help() { return "Run selected sparse TDM sizes"; }
  static const char* full_help() {
    return "Run all sizes; compare dense operator reports separately";
  }

  static perf::ProfileSuite suite() {
    return perf::load_profile_suite(
        std::filesystem::path(HOLOSCAN_RADAR_PERF_PROFILE_DIR) / "tdm_sparse_compensation.yaml",
        "tdm_sparse_compensation");
  }
  static std::vector<perf::SizeDef> size_ladder_defs() {
    const auto profiles = suite();
    std::vector<perf::SizeDef> result;
    for (size_t i = 0; i < perf::kProfiles.size(); ++i) {
      const auto& shape = profiles.shapes[i];
      if (shape.size() != 5)
        throw std::invalid_argument("TDMSparseCompensation profile rank mismatch");
      result.push_back({std::string(perf::kProfiles[i]), shape,
                        shape[0] * shape[1] * shape[2] * shape[3],
                        std::string(perf::kProfiles[i])});
    }
    return result;
  }
  static std::vector<Scenario> build_profile_matrix(const std::string& scope) {
    const auto profiles = suite();
    std::set<std::string> selected_ids;
    for (const auto& c : profiles.select(scope))
      selected_ids.insert(c.id);
    std::vector<Scenario> result;
    for (const auto& c : profiles.cases) {
      size_t i = 0;
      for (; i < perf::kProfiles.size(); ++i) {
        if (c.profile == perf::kProfiles[i])
          break;
      }
      const auto& shape = profiles.shapes.at(i);
      if (shape.size() != 5 || shape[0] < 2 || shape[4] < 1)
        throw std::invalid_argument("TDMSparseCompensation invalid profile shape");
      perf::require_keys(c.parameters, {"snapshot_window", "overlap_policy", "pri"}, "parameters");
      if (!c.parameters["snapshot_window"] || !c.parameters["overlap_policy"])
        throw std::invalid_argument("TDMSparseCompensation missing parameter");
      const auto window = c.parameters["snapshot_window"];
      if (!window.IsSequence() || window.size() != 2)
        throw std::invalid_argument("TDMSparseCompensation window must have two extents");
      const auto policy = c.parameters["overlap_policy"].as<std::string>();
      hr::OverlapPolicy overlap;
      if (policy == "keep_all")
        overlap = hr::OverlapPolicy::kKeepAll;
      else if (policy == "dedup")
        overlap = hr::OverlapPolicy::kDedup;
      else if (policy == "average")
        overlap = hr::OverlapPolicy::kAverage;
      else
        throw std::invalid_argument("TDMSparseCompensation invalid overlap policy");
      SizeSpec size{c.profile, shape[0], shape[1], shape[2], shape[3], shape[4], c.profile};
      Scenario scenario{c.id, c.profile, size};
      scenario.policy = overlap;
      if (c.parameters["pri"])
        scenario.pri = c.parameters["pri"].as<double>();
      if (scenario.pri <= 40.0e-6)
        throw std::invalid_argument("TDMSparseCompensation PRI too short");
      scenario.window_doppler = window[0].as<int64_t>();
      scenario.window_range = window[1].as<int64_t>();
      if (scenario.window_doppler < 0 || scenario.window_range < 0 || scenario.window_doppler > 2 ||
          scenario.window_range > 2)
        throw std::invalid_argument("TDMSparseCompensation window out of range");
      if (c.override_value) {
        scenario.live_port = (*c.override_value)["port"].as<std::string>();
        if (scenario.live_port == "waveform") {
          scenario.changed_pri = (*c.override_value)["value"].as<double>();
          if (scenario.changed_pri <= 40.0e-6)
            throw std::invalid_argument("TDMSparseCompensation invalid waveform PRI");
        } else if (scenario.live_port == "snapshot_window") {
          const auto changed = (*c.override_value)["value"];
          if (!changed.IsSequence() || changed.size() != 2)
            throw std::invalid_argument("TDMSparseCompensation invalid changed window");
          scenario.changed_doppler = changed[0].as<int64_t>();
          scenario.changed_range = changed[1].as<int64_t>();
          if (scenario.changed_doppler < 0 || scenario.changed_range < 0 ||
              scenario.changed_doppler > 2 || scenario.changed_range > 2)
            throw std::invalid_argument("TDMSparseCompensation changed window out of range");
        } else {
          throw std::invalid_argument("unknown TDMSparseCompensation live port");
        }
      }
      if (selected_ids.contains(c.id))
        result.push_back(std::move(scenario));
    }
    return result;
  }
  static void run_scenario(const Scenario& scenario, const perf::RunSettings& settings,
                           perf::PerfReport* report) {
    const perf::RunPlan plan = perf::make_run_plan(settings, !scenario.live_port.empty());
    const auto samples =
        perf::collect_scenario_samples(settings, plan, [&](int, const perf::RunPlan& current) {
          return run_graph(scenario, current);
        });
    perf::append_setup_row(base_row(scenario), samples, report);
    auto row = base_row(scenario);
    perf::fill_compute_row(&row, samples);
    if (row.latency.median_ms > 0.0) {
      row.cells_per_second =
          static_cast<double>(scenario.size.detections) / (row.latency.median_ms / 1000.0);
    }
    report->results.push_back(std::move(row));
  }

  static void process_warmup(const perf::RunSettings&) {
    perf::run_process_warmup([&] {
      perf::PerfReport report;
      perf::RunSettings settings;
      settings.warmup_frames = 0;
      settings.measured_frames = 2;
      settings.repeats = 1;
      run_scenario(build_profile_matrix("minimal").front(), settings, &report);
    });
  }
};

}  // namespace

int main(int argc, char** argv) {
  return perf::perf_main<Traits>(argc, argv);
}
