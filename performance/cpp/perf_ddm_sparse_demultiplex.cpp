// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cuda_runtime_api.h>

#include <algorithm>
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
#include "utils/perf_process_warmup.hpp"
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
  return perf::run_perf_graph_dual(
      std::move(rd), std::move(idx), plan.total_frames, "ddm-sparse-perf",
      [&](holoscan::Graph& graph, const auto& source, holoscan::CompileOptions& options) {
        hr::MaxTensorExtents extents;
        extents.max_tx = size.tx;
        extents.max_rx = size.rx;
        extents.max_virtual_channels = size.tx * size.rx;
        extents.max_doppler_bins = size.doppler;
        extents.max_range_bins = size.range;
        extents.max_detections = size.detections;
        extents.max_snapshots = 1;
        auto op = graph.op<hr::DDMSparseDemultiplexOp<float>>("sparse", extents);
        hr::DDMSparseDemultiplexParams params;
        params.ddm_code = code(size.tx);
        params.tx_positions = positions(size.tx, 0.02);
        params.rx_positions = positions(size.rx, 0.002);
        params.overlap_policy = hr::OverlapPolicy::kKeepAll;
        params.wavelength = 0.0039;
        op.set_params(std::move(params));
        const auto terminal = graph.op<perf::PerfTerminalSinkOp>("terminal");
        const auto mask = graph.op<perf::PerfNullSinkOp>("mask");
        const auto geometry = graph.op<perf::PerfPayloadSinkOp<hr::WireArrayGeometry>>("geometry");
        const holoscan::ConnectionOptions queue{.queue_depth = 4U};
        graph.add_flow(source->out_a, op->rd_maps, queue);
        graph.add_flow(source->out_b, op->detection_idx, queue);
        graph.add_flow(op->snapshot, terminal->in, queue);
        graph.add_flow(op->is_valid, mask->in, queue);
        graph.add_flow(op->array_geometry, geometry->in, queue);
        graph.partition("perf").add(source).add(op).add(terminal).add(mask).add(geometry);
        perf::bind_tensor_out(options, "sparse", "snapshot");
        perf::bind_tensor_out(options, "sparse", "is_valid");
      });
}

perf::ResultRow base_row(const Scenario& scenario) {
  perf::ResultRow row;
  row.scenario_id = scenario.id;
  row.op_name = "DDMSparseDemultiplexOp";
  row.variant = "keep_all_k1";
  row.size_id = scenario.size_id;
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
  static const char* sizes_all_help() { return "Run full size ladder"; }
  static const char* sizes_subset_help() { return "Run selected sizes"; }
  static const char* full_help() { return "Run all sparse DDM sizes"; }
  static std::vector<perf::SizeDef> size_ladder_defs() {
    std::vector<perf::SizeDef> result;
    for (const auto& size : sizes()) {
      result.push_back({size.id, {size.doppler, size.range}, size.doppler * size.range, size.role});
    }
    return result;
  }
  static std::vector<Scenario> build_size_sweep(const std::vector<std::string>& ids) {
    std::vector<Scenario> result;
    for (const auto& size : sizes()) {
      if (std::find(ids.begin(), ids.end(), size.id) != ids.end()) {
        result.push_back({"ddm_sparse_" + size.id, size.id, size});
      }
    }
    return result;
  }
  static std::vector<Scenario> build_default_matrix() {
    return build_size_sweep({"xs", "sm", "md"});
  }
  static std::vector<Scenario> build_full_matrix() {
    return build_size_sweep({"xs", "sm", "md", "lg", "xl"});
  }
  static void run_scenario(const Scenario& scenario, const perf::RunSettings& settings,
                           perf::PerfReport* report) {
    const auto plan = perf::make_run_plan(settings, false);
    const auto samples =
        perf::collect_scenario_samples(settings, plan, [&](int, const perf::RunPlan& current) {
          return run_graph(scenario, current);
        });
    perf::append_setup_row(base_row(scenario), samples, report);
    auto row = base_row(scenario);
    perf::fill_compute_row(&row, samples);
    report->results.push_back(std::move(row));
  }
  static void process_warmup(const perf::RunSettings&) {
    perf::run_process_warmup([&] {
      perf::PerfReport report;
      perf::RunSettings settings;
      settings.warmup_frames = 0;
      settings.measured_frames = 2;
      settings.repeats = 1;
      run_scenario(build_size_sweep({"xs"}).front(), settings, &report);
    });
  }
};

}  // namespace

int main(int argc, char** argv) {
  return perf::perf_main<Traits>(argc, argv);
}
