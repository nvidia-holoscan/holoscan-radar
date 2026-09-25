// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Performance characterization for FeatureAssemblyOp — sweeps detection count
// and map size. Measurement tool only: no timing assertions.
//
// Graph rewrite via run_perf_graph_triple. The driver times the physical
// {range, velocity} pack with exact-aligned axis inputs. No [live] parameters.

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>  // NOLINT(build/c++17): repo builds with C++17
#include <stdexcept>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/graph.hpp>

#include "detection/feature_assembly_op/feature_assembly_op.hpp"
#include "radar_common/descriptors/max_tensor_extents.hpp"
#include "radar_common/tensor/dlpack_buffer.hpp"
#include "radar_common/tensor/tensor_utils.hpp"
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

constexpr DLDataType kInt64Dtype{kDLInt, 64, 1};

struct SizeSpec {
  std::string id;
  int64_t doppler;
  int64_t range;
  std::string role;
};

struct FeatScenario {
  std::string id;
  std::string size_id;
  int64_t num_detections{0};
  int64_t doppler{0};
  int64_t range{0};
};

std::vector<hr::AssemblyFeature> pack_features() {
  return {hr::AssemblyFeature::kRange, hr::AssemblyFeature::kVelocity};
}

template <typename T>
std::shared_ptr<holoscan::Tensor> make_device_from_host(const std::vector<T>& host,
                                                        const std::vector<int64_t>& shape,
                                                        DLDataType dtype) {
  auto tensor = hr::make_device_tensor(shape, dtype);
  if (!host.empty()) {
    auto mut = tensor->mutable_data();
    if (!mut) {
      throw std::runtime_error("perf_feature_assembly: tensor not writable");
    }
    if (cudaMemcpy(mut.value(), host.data(), sizeof(T) * host.size(), cudaMemcpyHostToDevice) !=
        cudaSuccess) {
      throw std::runtime_error("perf_feature_assembly: cudaMemcpy H2D failed");
    }
  }
  return tensor;
}

std::shared_ptr<holoscan::Tensor> make_detection_idx(int64_t num_detections, int64_t doppler,
                                                     int64_t range) {
  std::vector<int64_t> host(static_cast<size_t>(std::max<int64_t>(num_detections, 0) * 2), 0);
  const int64_t cells = std::max<int64_t>(doppler * range, 1);
  for (int64_t i = 0; i < num_detections; ++i) {
    const int64_t cell = (i * 997) % cells;
    host[static_cast<size_t>(i * 2)] = cell / range;
    host[static_cast<size_t>(i * 2 + 1)] = cell % range;
  }
  return make_device_from_host(host, {num_detections, 2}, kInt64Dtype);
}

perf::TimingCollector run_feat_graph(const FeatScenario& scen, int total_frames) {
  auto idx_t = make_detection_idx(scen.num_detections, scen.doppler, scen.range);
  std::vector<PerfPrecision> range_host(static_cast<std::size_t>(scen.range));
  std::vector<PerfPrecision> velocity_host(static_cast<std::size_t>(scen.doppler));
  for (int64_t i = 0; i < scen.range; ++i) {
    range_host[static_cast<std::size_t>(i)] = static_cast<PerfPrecision>(i);
  }
  for (int64_t i = 0; i < scen.doppler; ++i) {
    velocity_host[static_cast<std::size_t>(i)] = static_cast<PerfPrecision>(i);
  }
  auto range_t = make_device_from_host(range_host, {scen.range}, hr::real_dtype<PerfPrecision>());
  auto velocity_t =
      make_device_from_host(velocity_host, {scen.doppler}, hr::real_dtype<PerfPrecision>());
  return perf::run_perf_graph_triple(
      std::move(idx_t), std::move(range_t), std::move(velocity_t), total_frames, "feat-asm-perf",
      [&](holoscan::Graph& graph, const auto& src, holoscan::CompileOptions& options) {
        hr::MaxTensorExtents extents;
        extents.max_points = std::max<int64_t>(scen.num_detections, 1);
        extents.max_features = static_cast<int64_t>(pack_features().size());
        auto features = pack_features();
        auto op =
            graph.op<hr::FeatureAssemblyOp<PerfPrecision>>("feat", extents, std::move(features));

        const auto pts_sink = graph.op<perf::PerfTerminalSinkOp>("pts_sink");
        const holoscan::ConnectionOptions q{.queue_depth = 4U};

        graph.add_flow(src->out_a, op->detection_idx, q);
        graph.add_flow(src->out_b, op->range_bins, q);
        graph.add_flow(src->out_c, op->velocity_bins, q);
        graph.add_flow(op->points, pts_sink->in, q);

        perf::bind_tensor_out(options, "feat", "points");
      });
}

void fill_throughput(perf::ResultRow* row, const FeatScenario& scen, double median_ms) {
  if (median_ms <= 0.0) {
    return;
  }
  const double sec = median_ms / 1000.0;
  const double n = static_cast<double>(scen.num_detections);
  row->cells_per_second = n / sec;

  const double tsz = sizeof(PerfPrecision);
  const double idx_bytes = n * 2.0 * static_cast<double>(sizeof(int64_t));
  const double axis_bytes = static_cast<double>(scen.doppler + scen.range) * tsz;
  const int64_t num_features = static_cast<int64_t>(pack_features().size());
  const double points_bytes = n * static_cast<double>(num_features) * tsz;
  row->estimated_effective_gbps = (idx_bytes + axis_bytes + points_bytes) / sec / 1e9;
}

perf::ResultRow base_row(const FeatScenario& scen) {
  perf::ResultRow row;
  row.scenario_id = scen.id;
  row.op_name = "FeatureAssemblyOp";
  row.variant = "physical_range_velocity";
  row.size_id = scen.size_id;
  row.map_shape = {scen.doppler, scen.range};
  row.input_shape = {scen.num_detections, 2};
  row.out_batch = scen.num_detections;
  row.k = static_cast<int64_t>(pack_features().size());
  row.notes = "join=exact_aligned_axes";
  return row;
}

FeatScenario make_scenario(const SizeSpec& s, int64_t num_detections) {
  FeatScenario sc;
  sc.size_id = s.id;
  sc.doppler = s.doppler;
  sc.range = s.range;
  sc.num_detections = num_detections;
  std::ostringstream id;
  id << "feat_asm_physical_" << s.id << "_n" << num_detections << "_compute";
  sc.id = id.str();
  return sc;
}

struct FeatTraits : perf::PerfDriverDefaults {
  using Scenario = FeatScenario;

  static const char* tool_name() { return "perf_feature_assembly"; }
  static const char* tool_description() {
    return "FeatureAssemblyOp performance characterization "
           "(HSDK 5 Graph)";
  }

  static perf::ProfileSuite suite() {
    return perf::load_profile_suite(
        std::filesystem::path(HOLOSCAN_RADAR_PERF_PROFILE_DIR) / "feature_assembly.yaml",
        "feature_assembly");
  }

  static std::vector<perf::SizeDef> size_ladder_defs() {
    const auto profiles = suite();
    std::vector<perf::SizeDef> result;
    for (size_t i = 0; i < perf::kProfiles.size(); ++i) {
      const auto& shape = profiles.shapes[i];
      if (shape.size() != 2)
        throw std::invalid_argument("feature_assembly profile rank mismatch");
      result.push_back({std::string(perf::kProfiles[i]), shape, shape[0] * shape[1],
                        std::string(perf::kProfiles[i])});
    }
    return result;
  }

  static FeatScenario parse_case(const perf::ProfileCase& c, const std::vector<int64_t>& shape) {
    if (shape.size() != 2)
      throw std::invalid_argument("feature_assembly profile rank mismatch");
    perf::require_keys(c.parameters, {"detections"}, "parameters");
    if (c.override_value.has_value())
      throw std::invalid_argument("FeatureAssembly has no live input");
    const int64_t count = c.parameters["detections"].as<int64_t>();
    if (count < 0 || count > 65536)
      throw std::invalid_argument("FeatureAssembly detections out of range");
    const SizeSpec size{c.profile, shape[0], shape[1], c.profile};
    auto scenario = make_scenario(size, count);
    scenario.id = c.id;
    scenario.size_id = c.profile;
    return scenario;
  }

  static void run_scenario(const FeatScenario& scen, const perf::RunSettings& rs,
                           perf::PerfReport* report) {
    perf::measure_scenario(
        rs, false,
        [&](int /*rep*/, const perf::RunPlan& p) {
          return run_feat_graph(scen, p.total_frames);
        },
        base_row(scen), report,
        [&](perf::ResultRow& row) {
          fill_throughput(&row, scen, row.latency.median_ms);
          row.notes =
              "bins-only (range+velocity) — score/aoa optional ports "
              "omitted from this Graph benchmark (covered by stage tests); "
              "cold_ms is first-frame latency; median/p90 are "
              "steady state; cells/s is detections/s";
        });
  }
};

}  // namespace

int main(int argc, char** argv) {
  return holoscan::radar::perf::perf_main<FeatTraits>(argc, argv);
}
