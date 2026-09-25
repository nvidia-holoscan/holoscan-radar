// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cuda_runtime_api.h>

#include <algorithm>
#include <set>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/graph.hpp>

#include "fmcw/ddm_sparse_demultiplex_op/ddm_sparse_demultiplex_op.hpp"
#include "radar_common/tensor/dlpack_buffer.hpp"
#include "utils/perf_driver.hpp"
#include "utils/perf_graph.hpp"
#include "utils/perf_report.hpp"
#include "utils/perf_runner.hpp"
#include "utils/tensor_factory.hpp"

namespace hr = holoscan::radar;
namespace perf = holoscan::radar::perf;
namespace {

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
  double changed_wavelength{0.0041};
  int64_t window_doppler{0};
  int64_t window_range{0};
  int64_t changed_doppler{1};
  int64_t changed_range{1};
  hr::OverlapPolicy policy{hr::OverlapPolicy::kKeepAll};
};

const std::vector<SizeSpec>& sizes() {
  static const std::vector<SizeSpec> result{{"xs", 2, 4, 64, 256, 16, "smoke"},
                                            {"sm", 4, 4, 128, 512, 64, "sparse automotive"},
                                            {"md", 4, 8, 256, 512, 256, "common MIMO"},
                                            {"lg", 4, 16, 512, 1024, 1024, "large array"},
                                            {"xl", 8, 16, 512, 2048, 4096, "upper bound"}};
  return result;
}

std::vector<double> positions(int64_t count, double spacing) {
  std::vector<double> result(static_cast<size_t>(count * 3));
  for (int64_t index = 0; index < count; ++index) {
    result[static_cast<size_t>(index * 3)] = index * spacing;
  }
  return result;
}

hr::DDMCodeDescriptor code(int64_t tx) {
  hr::DDMCodeDescriptor result;
  result.num_tx = tx;
  for (int64_t index = 0; index < tx; ++index) {
    result.doppler_offsets.push_back(static_cast<double>(index) / static_cast<double>(tx));
  }
  return result;
}

std::shared_ptr<holoscan::Tensor> detections(int64_t count, int64_t width, int64_t range) {
  std::vector<int64_t> host(static_cast<size_t>(2 * count));
  for (int64_t index = 0; index < count; ++index) {
    const int64_t cell = (index * 997) % (width * range);
    host[static_cast<size_t>(2 * index)] = cell / range;
    host[static_cast<size_t>(2 * index + 1)] = cell % range;
  }
  auto result = hr::make_device_tensor({count, 2}, kInt64Dtype);
  auto pointer = result->mutable_data();
  if (!pointer || cudaMemcpy(*pointer, host.data(), host.size() * sizeof(int64_t),
                             cudaMemcpyHostToDevice) != cudaSuccess) {
    throw std::runtime_error("DDM sparse detection H2D failed");
  }
  return result;
}

perf::TimingCollector run_graph(const Scenario& scenario, const perf::RunPlan& plan) {
  const auto& size = scenario.size;
  auto rd = perf::make_iq_cube_device<float>({size.rx, size.doppler, size.range});
  auto idx = detections(size.detections, size.doppler / size.tx, size.range);
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
    auto op = graph.op<hr::DDMSparseDemultiplexOp<float>>("sparse", extents);
    hr::DDMSparseDemultiplexParams params;
    params.ddm_code = code(size.tx);
    params.tx_positions = positions(size.tx, 0.02);
    params.rx_positions = positions(size.rx, 0.002);
    params.overlap_policy = scenario.policy;
    params.wavelength = 0.0039;
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
  if (scenario.live_port == "wavelength") {
    return perf::run_perf_graph_dual_with_override(
        std::move(rd), std::move(idx), plan.total_frames, plan.override_at, 0.0039,
        scenario.changed_wavelength, "ddm-sparse-perf",
        [&](auto& graph, const auto& source, auto& options) {
          configure(graph, source, options,
                    [](auto& g, const auto& src, const auto& op, const auto& q) {
                      g.add_flow(src->override_out, op->wavelength_in, q);
                    });
        },
        perf::PerfPublishMode::kCopy, true);
  }
  if (scenario.live_port == "snapshot_window") {
    const hr::DDMSparseWindowOverride initial{.doppler = scenario.window_doppler,
                                              .range = scenario.window_range};
    const hr::DDMSparseWindowOverride changed{.doppler = scenario.changed_doppler,
                                              .range = scenario.changed_range};
    return perf::run_perf_graph_dual_with_override(
        std::move(rd), std::move(idx), plan.total_frames, plan.override_at, initial, changed,
        "ddm-sparse-perf",
        [&](auto& graph, const auto& source, auto& options) {
          configure(graph, source, options,
                    [](auto& g, const auto& src, const auto& op, const auto& q) {
                      g.add_flow(src->override_out, op->snapshot_window_in, q);
                    });
        },
        perf::PerfPublishMode::kCopy, true);
  }
  return perf::run_perf_graph_dual(
      std::move(rd), std::move(idx), plan.total_frames, "ddm-sparse-perf",
      [&](auto& graph, const auto& source, auto& options) {
        configure(graph, source, options, [](auto&, const auto&, const auto&, const auto&) {});
      });
}

perf::ResultRow base_row(const Scenario& scenario) {
  perf::ResultRow row;
  row.op_name = "DDMSparseDemultiplexOp";
  row.variant = "sparse_ddm";
  row.live_port = scenario.live_port;
  row.map_shape = {scenario.size.doppler, scenario.size.range};
  row.input_shape = {scenario.size.rx, scenario.size.doppler, scenario.size.range};
  row.out_batch = scenario.size.detections;
  row.k = scenario.size.tx * scenario.size.rx;
  row.notes = "honest pre-demux input and detection-sized snapshot output";
  return row;
}

struct Traits : perf::PerfDriverDefaults {
  using Scenario = ::Scenario;
  static const char* tool_name() { return "perf_ddm_sparse_demultiplex"; }
  static const char* tool_description() { return "Sparse DDM detection-count characterization"; }

  static perf::SizeDef size_def(const std::string& id, const std::vector<int64_t>& shape) {
    if (shape.size() != 5)
      throw std::invalid_argument("DDMSparseDemultiplex profile rank mismatch");
    return {id, shape, shape[1] * shape[2] * shape[3], id};
  }
  static Scenario parse_case(const perf::ProfileCase& c, const std::vector<int64_t>& shape) {
    if (shape.size() != 5 || shape[0] < 2 || shape[2] % shape[0] != 0 || shape[4] < 1)
      throw std::invalid_argument("DDMSparseDemultiplex invalid profile shape");
    perf::require_keys(c.parameters, {"snapshot_window", "overlap_policy"}, "parameters");
    if (!c.parameters["snapshot_window"] || !c.parameters["overlap_policy"])
      throw std::invalid_argument("DDMSparseDemultiplex missing parameter");
    const auto window = c.parameters["snapshot_window"];
    if (!window.IsSequence() || window.size() != 2)
      throw std::invalid_argument("DDMSparseDemultiplex window must have two extents");
    const auto policy = c.parameters["overlap_policy"].as<std::string>();
    hr::OverlapPolicy overlap;
    if (policy == "keep_all")
      overlap = hr::OverlapPolicy::kKeepAll;
    else if (policy == "dedup")
      overlap = hr::OverlapPolicy::kDedup;
    else if (policy == "average")
      overlap = hr::OverlapPolicy::kAverage;
    else
      throw std::invalid_argument("DDMSparseDemultiplex invalid overlap policy");
    SizeSpec size{c.profile, shape[0], shape[1], shape[2], shape[3], shape[4], c.profile};
    Scenario scenario{c.id, c.profile, size};
    scenario.policy = overlap;
    scenario.window_doppler = window[0].as<int64_t>();
    scenario.window_range = window[1].as<int64_t>();
    if (scenario.window_doppler < 0 || scenario.window_range < 0 || scenario.window_doppler > 2 ||
        scenario.window_range > 2)
      throw std::invalid_argument("DDMSparseDemultiplex window out of range");
    if (c.override_value) {
      scenario.live_port = (*c.override_value)["port"].as<std::string>();
      if (scenario.live_port == "wavelength") {
        scenario.changed_wavelength = (*c.override_value)["value"].as<double>();
        if (scenario.changed_wavelength <= 0.0)
          throw std::invalid_argument("DDMSparseDemultiplex invalid wavelength");
      } else if (scenario.live_port == "snapshot_window") {
        const auto changed = (*c.override_value)["value"];
        if (!changed.IsSequence() || changed.size() != 2)
          throw std::invalid_argument("DDMSparseDemultiplex invalid changed window");
        scenario.changed_doppler = changed[0].as<int64_t>();
        scenario.changed_range = changed[1].as<int64_t>();
        if (scenario.changed_doppler < 0 || scenario.changed_range < 0 ||
            scenario.changed_doppler > 2 || scenario.changed_range > 2)
          throw std::invalid_argument("DDMSparseDemultiplex changed window out of range");
      } else {
        throw std::invalid_argument("unknown DDMSparseDemultiplex live port");
      }
    }
    return scenario;
  }
};

}  // namespace

int main(int argc, char** argv) {
  return perf::perf_main<Traits>(argc, argv);
}
