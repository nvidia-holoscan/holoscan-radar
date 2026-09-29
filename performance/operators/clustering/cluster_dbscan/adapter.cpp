// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "clustering/cluster_dbscan_op/cluster_dbscan_op.hpp"
#include "utils/perf_driver.hpp"
#include "utils/perf_yaml_pilot.hpp"

namespace hr = holoscan::radar;
namespace perf = holoscan::radar::perf;

namespace {

using Op = hr::ClusterDBSCANOp<float>;
struct Scenario : perf::RecipeScenario {};

hr::ClusterDBSCANParams parameters(const Scenario& scen) {
  hr::ClusterDBSCANParams params;
  params.epsilon = scen.params["epsilon"].as<std::vector<double>>();
  const auto minimum = scen.params["min_points"];
  params.min_points = minimum.IsMap()
                          ? perf::scaled_recipe_axis(minimum, scen.axis("features"), "min_points")
                          : minimum.as<int64_t>();
  params.max_clusters = scen.params["max_clusters"].as<int64_t>();
  return params;
}

perf::TimingCollector run_graph(const Scenario& scen, const perf::RunPlan& plan) {
  return perf::run_recipe_selected_case<Op, 1>(
      scen, plan, perf::direct_recipe_extents(scen.recipe, scen.values), parameters(scen));
}

perf::ResultRow base_row(const Scenario& scen) {
  perf::ResultRow row;
  row.op_name = "ClusterDBSCANOp";
  row.variant = "dbscan";
  row.map_shape = {scen.axis("points"), scen.axis("features")};
  row.input_shape = row.map_shape;
  row.out_batch = scen.axis("clusters");
  row.k = scen.axis("features");
  row.live_port = scen.live_port;
  return row;
}

void decorate_row(perf::ResultRow& row, const Scenario& scen) {
  const double n = static_cast<double>(scen.axis("points"));
  const double bytes =
      n * static_cast<double>(scen.axis("features")) * sizeof(float) + n * sizeof(int64_t);
  perf::set_recipe_throughput(row, n, static_cast<double>(scen.axis("clusters")), bytes);
  row.notes = "end-to-end Graph latency; cells/s is points/s";
  if (!scen.live_port.empty())
    row.notes += "; one-frame " + scen.live_port + " change";
}

struct Traits : perf::PerfDriverDefaults {
  using Scenario = ::Scenario;
  static const char* tool_name() { return "perf_cluster_dbscan"; }
  static const char* tool_description() {
    return "ClusterDBSCANOp performance characterization (HSDK 5 Graph)";
  }
  static perf::SizeDef size_def(const std::string& id, const std::vector<int64_t>& shape) {
    return perf::recipe_size_def(id, shape, 3, 1);
  }
  static void validate_suite(const perf::ProfileSuite& suite) {
    (void)perf::validate_pilot_suite<Op>(
        suite, "radar.ClusterDBSCANOp.float", "",
        {{"points", {"points", "features"}, "float32", "device", "cluster_blobs", {}}},
        {"labels", {"sizes"}, {}, 4U},
        {{"epsilon", "epsilon"}, {"min_points", "min_points"}, {"max_clusters", "max_clusters"}},
        {{"epsilon", "host_float64_vector"}, {"min_points", "int64"}, {"max_clusters", "int64"}});
  }
  static Scenario parse_case(const perf::ProfileCase& c, const std::vector<int64_t>& shape,
                             const perf::ProfileSuite& suite) {
    auto scen = perf::pilot_scenario<Scenario>(suite, c, shape, 3);
    if (scen.axis("points") < 1 || scen.axis("features") < 2 || scen.axis("features") > 5 ||
        scen.axis("clusters") < 1)
      throw std::invalid_argument("ClusterDBSCAN invalid profile shape");
    perf::require_recipe_params(scen.params, {"epsilon", "min_points", "max_clusters"});
    const auto params = parameters(scen);
    if (params.epsilon.size() != 1 || params.epsilon[0] <= 0 || params.min_points < 0 ||
        params.max_clusters < 0)
      throw std::invalid_argument("ClusterDBSCAN invalid Params");
    if (scen.live_port == "epsilon") {
      const auto value = scen.live_value.as<std::vector<double>>();
      if (value.size() != 1 || value[0] <= 0)
        throw std::invalid_argument("ClusterDBSCAN invalid live epsilon");
    } else if (!scen.live_port.empty() && scen.live_value.as<int64_t>() < 0) {
      throw std::invalid_argument("ClusterDBSCAN live count must be nonnegative");
    }
    return scen;
  }
};

}  // namespace

int main(int argc, char** argv) {
  return perf::perf_main<Traits>(argc, argv);
}
