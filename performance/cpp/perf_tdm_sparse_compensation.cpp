// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Detection-count characterization for sparse TDM compensation.

#include <cuda_runtime_api.h>

#include <cstdint>
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
};

const std::vector<SizeSpec>& size_ladder() {
  static const std::vector<SizeSpec> result{
      {"xs", 2, 4, 64, 256, 16, "smoke"},
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

hr::FMCWWaveformDescriptor waveform() {
  hr::FMCWWaveformSpec spec;
  spec.carrier_frequency = 77.0e9;
  spec.bandwidth = 1.0e9;
  spec.chirp_duration = 40.0e-6;
  spec.sample_rate = 20.0e6;
  spec.chirp_repetition_interval = 1.0e-3;
  return hr::FMCWWaveformDescriptor(spec);
}

std::vector<double> ula_positions(int64_t count, double spacing) {
  std::vector<double> result(static_cast<std::size_t>(count * 3), 0.0);
  for (int64_t index = 0; index < count; ++index) {
    result[static_cast<std::size_t>(index * 3)] =
        spacing * static_cast<double>(index);
  }
  return result;
}

std::shared_ptr<holoscan::Tensor> make_detections(
    int64_t count, int64_t doppler, int64_t range) {
  std::vector<int64_t> host(static_cast<std::size_t>(count * 2));
  for (int64_t index = 0; index < count; ++index) {
    const int64_t cell = (index * 997) % (doppler * range);
    host[static_cast<std::size_t>(index * 2)] = cell / range;
    host[static_cast<std::size_t>(index * 2 + 1)] = cell % range;
  }
  auto result = hr::make_device_tensor({count, 2}, kInt64Dtype);
  if (!host.empty()) {
    auto pointer = result->mutable_data();
    if (!pointer ||
        cudaMemcpy(*pointer, host.data(), host.size() * sizeof(int64_t),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
      throw std::runtime_error("detection-index H2D failed");
    }
  }
  return result;
}

perf::TimingCollector run_graph(
    const Scenario& scenario, const perf::RunPlan& plan) {
  const auto& size = scenario.size;
  auto rd_maps = perf::make_iq_cube_device<Precision>(
      {size.tx, size.rx, size.doppler, size.range});
  auto index = make_detections(
      size.detections, size.doppler, size.range);
  return perf::run_perf_graph_dual(
      std::move(rd_maps), std::move(index), plan.total_frames,
      "tdm-sparse-compensation-perf",
      [&](holoscan::Graph& graph, const auto& source,
          holoscan::CompileOptions& options) {
        hr::MaxTensorExtents extents;
        extents.max_tx = size.tx;
        extents.max_rx = size.rx;
        extents.max_virtual_channels = size.tx * size.rx;
        extents.max_doppler_bins = size.doppler;
        extents.max_range_bins = size.range;
        extents.max_detections = size.detections;
        extents.max_snapshots = 1;
        auto op = graph.op<hr::TDMSparseCompensationOp<Precision>>(
            "sparse", extents);
        hr::TDMSparseCompensationParams params;
        params.waveform = waveform();
        params.tx_positions = ula_positions(size.tx, 0.02);
        params.rx_positions = ula_positions(size.rx, 0.002);
        params.overlap_policy = hr::OverlapPolicy::kKeepAll;
        op.set_params(std::move(params));

        const auto terminal =
            graph.op<perf::PerfTerminalSinkOp>("terminal");
        const auto mask_sink =
            graph.op<perf::PerfNullSinkOp>("mask_sink");
        const auto geometry_sink =
            graph.op<perf::PerfPayloadSinkOp<hr::WireArrayGeometry>>(
                "geometry_sink");
        const holoscan::ConnectionOptions queue{.queue_depth = 4U};
        graph.add_flow(source->out_a, op->rd_maps, queue);
        graph.add_flow(source->out_b, op->detection_idx, queue);
        graph.add_flow(op->snapshot, terminal->in, queue);
        graph.add_flow(op->is_valid, mask_sink->in, queue);
        graph.add_flow(op->array_geometry, geometry_sink->in, queue);
        graph.partition("perf")
            .add(source)
            .add(op)
            .add(terminal)
            .add(mask_sink)
            .add(geometry_sink);
        perf::bind_tensor_out(options, "sparse", "snapshot");
        perf::bind_tensor_out(options, "sparse", "is_valid");
      });
}

perf::ResultRow base_row(const Scenario& scenario) {
  const auto& size = scenario.size;
  perf::ResultRow row;
  row.scenario_id = scenario.id;
  row.op_name = "TDMSparseCompensationOp";
  row.variant = "keep_all_k1";
  row.size_id = size.id;
  row.map_shape = {size.doppler, size.range};
  row.input_shape = {size.tx, size.rx, size.doppler, size.range};
  row.out_batch = size.detections;
  row.k = size.tx * size.rx;
  row.notes =
      "compare with perf_tdm_dense_compensation and "
      "perf_virtual_array_assembly at the same cube shape";
  return row;
}

struct Traits : perf::PerfDriverDefaults {
  using Scenario = ::Scenario;

  static const char* tool_name() {
    return "perf_tdm_sparse_compensation";
  }
  static const char* tool_description() {
    return "Sparse TDM compensation detection-count characterization";
  }
  static const char* sizes_all_help() {
    return "Run the full sparse TDM size ladder";
  }
  static const char* sizes_subset_help() {
    return "Run selected sparse TDM sizes";
  }
  static const char* full_help() {
    return "Run all sizes; compare dense operator reports separately";
  }

  static std::vector<perf::SizeDef> size_ladder_defs() {
    std::vector<perf::SizeDef> result;
    for (const auto& size : size_ladder()) {
      result.push_back(perf::SizeDef{
          .id = size.id,
          .shape = {size.doppler, size.range},
          .cells = size.doppler * size.range,
          .role = size.role});
    }
    return result;
  }

  static std::vector<Scenario> build_default_matrix() {
    return build_size_sweep({"xs", "sm", "md"});
  }

  static std::vector<Scenario> build_size_sweep(
      const std::vector<std::string>& sizes) {
    std::vector<Scenario> result;
    for (const auto& id : sizes) {
      const SizeSpec* size = find_size(id);
      if (size != nullptr) {
        result.push_back(
            Scenario{
                .id = "tdm_sparse_" + id,
                .size_id = id,
                .size = *size});
      }
    }
    return result;
  }

  static std::vector<Scenario> build_full_matrix() {
    return build_size_sweep({"xs", "sm", "md", "lg", "xl"});
  }

  static void run_scenario(
      const Scenario& scenario, const perf::RunSettings& settings,
      perf::PerfReport* report) {
    const perf::RunPlan plan =
        perf::make_run_plan(settings, false);
    const auto samples = perf::collect_scenario_samples(
        settings, plan, [&](int, const perf::RunPlan& current) {
          return run_graph(scenario, current);
        });
    perf::append_setup_row(base_row(scenario), samples, report);
    auto row = base_row(scenario);
    perf::fill_compute_row(&row, samples);
    if (row.latency.median_ms > 0.0) {
      row.cells_per_second =
          static_cast<double>(scenario.size.detections) /
          (row.latency.median_ms / 1000.0);
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
      run_scenario(
          build_size_sweep({"xs"}).front(), settings, &report);
    });
  }
};

}  // namespace

int main(int argc, char** argv) {
  return perf::perf_main<Traits>(argc, argv);
}
