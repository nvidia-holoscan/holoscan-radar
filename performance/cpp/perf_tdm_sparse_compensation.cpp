// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Detection-count characterization for sparse TDM compensation.

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdint>
#include <set>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
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
#include "utils/perf_report.hpp"
#include "utils/perf_recipe.hpp"
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
    const holoscan::ConnectionOptions queue{.queue_depth = 4U};
    graph.add_flow(source->out_a, op->rd_maps, queue);
    graph.add_flow(source->out_b, op->detection_idx, queue);
    connect(graph, source, op, queue);
    perf::PerfRecipeOutputs outputs(graph, options, "sparse");
    outputs.timed(op->snapshot, "snapshot");
    outputs.drain(op->is_valid, "is_valid");
    outputs.drain(op->array_geometry, "array_geometry");
    outputs.finish(source, op);
  };
  if (scenario.live_port.empty()) {
    return perf::run_data_recipe(
        std::array{std::move(rd), std::move(idx)}, plan, "tdm-sparse-compensation-perf",
        [&](auto& graph, const auto& source, auto& options) {
          configure(graph, source, options, [](auto&, const auto&, const auto&, const auto&) {});
        });
  }
  const auto cases = std::make_tuple(
      perf::live_option("waveform",
                        [&] {
                          return perf::live_case(
                              "waveform", hr::to_wire(waveform(scenario.pri)),
                              hr::to_wire(waveform(scenario.changed_pri)),
                              [](auto& graph, const auto& source, const auto& op, const auto& q) {
                                graph.add_flow(source->override_out, op->waveform_in, q);
                              });
                        }),
      perf::live_option("snapshot_window", [&] {
        return perf::live_case("snapshot_window",
                               hr::TDMSparseWindowOverride{.doppler = scenario.window_doppler,
                                                           .range = scenario.window_range},
                               hr::TDMSparseWindowOverride{.doppler = scenario.changed_doppler,
                                                           .range = scenario.changed_range},
                               [](auto& graph, const auto& source, const auto& op, const auto& q) {
                                 graph.add_flow(source->override_out, op->snapshot_window_in, q);
                               });
      }));
  return perf::dispatch_live_case(scenario.live_port, cases, [&](const auto& selected) {
    return perf::run_selected_live_recipe(std::array{std::move(rd), std::move(idx)}, plan,
                                          "tdm-sparse-compensation-perf", selected, configure);
  });
}

perf::ResultRow base_row(const Scenario& scenario) {
  const auto& size = scenario.size;
  perf::ResultRow row;
  row.op_name = "TDMSparseCompensationOp";
  row.variant = "sparse_tdm";
  row.live_port = scenario.live_port;
  perf::set_input_and_trailing_map_shape(row, {size.tx, size.rx, size.doppler, size.range});
  row.out_batch = size.detections;
  row.k = size.tx * size.rx;
  row.notes = "end-to-end Graph latency; cells/s is detections/s";
  return row;
}

void decorate_row(perf::ResultRow& row, const Scenario& scenario) {
  if (row.latency.median_ms > 0.0) {
    row.cells_per_second =
        static_cast<double>(scenario.size.detections) / (row.latency.median_ms / 1000.0);
  }
}

struct Traits : perf::PerfDriverDefaults {
  using Scenario = ::Scenario;

  static const char* tool_name() { return "perf_tdm_sparse_compensation"; }
  static const char* tool_description() {
    return "Sparse TDM compensation detection-count characterization";
  }

  static perf::SizeDef size_def(const std::string& id, const std::vector<int64_t>& shape) {
    if (shape.size() != 5)
      throw std::invalid_argument("TDMSparseCompensation profile rank mismatch");
    return {id, shape, shape[0] * shape[1] * shape[2] * shape[3], id};
  }
  static Scenario parse_case(const perf::ProfileCase& c, const std::vector<int64_t>& shape) {
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
    return scenario;
  }
};

}  // namespace

int main(int argc, char** argv) {
  return perf::perf_main<Traits>(argc, argv);
}
