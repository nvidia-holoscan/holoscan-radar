// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "detection/feature_assembly_op/feature_assembly_op.hpp"
#include "utils/perf_driver.hpp"
#include "utils/perf_yaml_pilot.hpp"

namespace hr = holoscan::radar;
namespace perf = holoscan::radar::perf;

namespace holoscan::radar::perf {

template <>
struct PerfOperatorFactory<FeatureAssemblyOp<float>> {
  inline static thread_local std::vector<AssemblyFeature> features;
  static holoscan::OperatorHandle<FeatureAssemblyOp<float>> create(
      holoscan::Graph& graph, std::string_view name, const MaxTensorExtents& extents) {
    return graph.op<FeatureAssemblyOp<float>>(name, extents, features);
  }
};

}  // namespace holoscan::radar::perf

namespace {

using Op = hr::FeatureAssemblyOp<float>;
struct Scenario : perf::RecipeScenario {};

std::vector<hr::AssemblyFeature> constructor_features(const Scenario& scen) {
  const auto features = scen.recipe.constructor["features"];
  if (!features.IsSequence() || features.size() != 2 || features[0].as<std::string>() != "range" ||
      features[1].as<std::string>() != "velocity")
    throw std::invalid_argument("FeatureAssembly recipe requires [range, velocity]");
  return {hr::AssemblyFeature::kRange, hr::AssemblyFeature::kVelocity};
}

perf::TimingCollector run_graph(const Scenario& scen, const perf::RunPlan& plan) {
  hr::MaxTensorExtents extents;
  extents.max_points = std::max<int64_t>(scen.axis("detections"), 1);
  extents.max_features = 2;
  perf::PerfOperatorFactory<Op>::features = constructor_features(scen);
  return perf::run_recipe_case<Op, 3>(scen, plan, extents, Op::Params{});
}

perf::ResultRow base_row(const Scenario& scen) {
  perf::ResultRow row;
  row.op_name = "FeatureAssemblyOp";
  row.variant = "physical_range_velocity";
  row.map_shape = {scen.axis("doppler"), scen.axis("range")};
  row.input_shape = {scen.axis("detections"), 2};
  row.out_batch = scen.axis("detections");
  row.k = 2;
  row.notes = "join=exact_aligned_axes";
  return row;
}

void decorate_row(perf::ResultRow& row, const Scenario& scen) {
  const double count = static_cast<double>(scen.axis("detections"));
  const double bytes =
      count * (2.0 * sizeof(int64_t) + 2.0 * sizeof(float)) +
      static_cast<double>(scen.axis("doppler") + scen.axis("range")) * sizeof(float);
  perf::set_recipe_throughput(row, count, std::nullopt, bytes);
  row.notes =
      "bins-only (range+velocity) — score/aoa optional ports "
      "omitted from this Graph benchmark (covered by stage tests); "
      "cold_ms is first-frame latency; median/p90 are "
      "steady state; cells/s is detections/s";
}

struct Traits : perf::PerfDriverDefaults {
  using Scenario = ::Scenario;
  static const char* tool_name() { return "perf_feature_assembly"; }
  static const char* tool_description() {
    return "FeatureAssemblyOp performance characterization (HSDK 5 Graph)";
  }
  static perf::SizeDef size_def(const std::string& id, const std::vector<int64_t>& shape) {
    return perf::recipe_size_def(id, shape, 2, 2);
  }
  static void validate_suite(const perf::ProfileSuite& suite) {
    (void)perf::validate_pilot_suite<Op>(
        suite, "radar.FeatureAssemblyOp.float", "feature_assembly_bounds",
        {{"detection_idx", {"detections", "two"}, "int64", "device", "in_bounds_rd_indices", {}},
         {"range_bins", {"range"}, "float32", "device", "axis_ramp", {}},
         {"velocity_bins", {"doppler"}, "float32", "device", "axis_ramp", {}}},
        {"points", {}, {}, 4U}, {}, {});
  }
  static Scenario parse_case(const perf::ProfileCase& c, const std::vector<int64_t>& shape,
                             const perf::ProfileSuite& suite) {
    auto scen = perf::pilot_scenario<Scenario>(suite, c, shape, 2);
    if (scen.axis("detections") > 65536)
      throw std::invalid_argument("FeatureAssembly detections out of range");
    perf::require_recipe_params(scen.params, {});
    (void)constructor_features(scen);
    return scen;
  }
};

}  // namespace

int main(int argc, char** argv) {
  return perf::perf_main<Traits>(argc, argv);
}
