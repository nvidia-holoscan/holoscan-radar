// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "detection/detection_reduction_op/detection_reduction_op.hpp"
#include "utils/perf_driver.hpp"
#include "utils/perf_yaml_pilot.hpp"

namespace hr = holoscan::radar;
namespace perf = holoscan::radar::perf;

namespace {

using Op = hr::DetectionReductionOp<float>;
struct Scenario : perf::RecipeScenario {};

hr::RepresentativePolicy policy(std::string_view name) {
  if (name == "peak")
    return hr::RepresentativePolicy::kPeak;
  if (name == "first")
    return hr::RepresentativePolicy::kFirst;
  if (name == "medoid")
    return hr::RepresentativePolicy::kMedoid;
  if (name == "weighted_mean")
    return hr::RepresentativePolicy::kWeightedMean;
  throw std::invalid_argument("unknown DetectionReduction representative policy");
}

hr::ScoreMethod score(std::string_view name) {
  if (name == "weighted_sum")
    return hr::ScoreMethod::kWeightedSum;
  if (name == "weighted_l2")
    return hr::ScoreMethod::kWeightedL2;
  if (name == "weighted_max")
    return hr::ScoreMethod::kWeightedMax;
  throw std::invalid_argument("unknown DetectionReduction score method");
}

hr::DetectionReductionParams parameters(const Scenario& scen) {
  hr::DetectionReductionParams params;
  params.representative = policy(scen.params["representative"].as<std::string>());
  params.score_method = score(scen.params["score_method"].as<std::string>());
  const auto weights = scen.params["feature_weights"];
  if (weights["factory"].as<std::string>() != "ones" ||
      weights["count"].as<std::string>() != "features")
    throw std::invalid_argument("DetectionReduction feature_weights needs ones(features)");
  params.feature_weights.assign(static_cast<std::size_t>(scen.axis("features")), 1.0);
  return params;
}

perf::TimingCollector run_graph(const Scenario& scen, const perf::RunPlan& plan) {
  const auto extents = perf::direct_recipe_extents(scen.recipe, scen.values);
  const auto params = parameters(scen);
  return perf::run_recipe_selected_case<Op, 3>(
      scen, plan, extents, params, [&](const std::string& codec) -> perf::TimingCollector {
        if (codec == "representative_policy") {
          const auto value = policy(scen.live_value.as<std::string>());
          return perf::run_recipe_live_case_with_connector<Op, 3>(
              scen, plan, extents, params, value, value,
              [](auto& graph, const auto& source, const auto& op, const auto& queue) {
                graph.add_flow(source->override_out, op->representative, queue);
              });
        }
        if (codec == "score_method") {
          const auto value = score(scen.live_value.as<std::string>());
          return perf::run_recipe_live_case_with_connector<Op, 3>(
              scen, plan, extents, params, value, value,
              [](auto& graph, const auto& source, const auto& op, const auto& queue) {
                graph.add_flow(source->override_out, op->score_method, queue);
              });
        }
        if (codec == "first_double") {
          auto weights = params.feature_weights;
          weights[0] = 2.0;
          return perf::run_recipe_live_case<Op, 3>(scen, plan, extents, params, weights, weights);
        }
        throw std::invalid_argument("unknown DetectionReduction live codec: " + codec);
      });
}

perf::ResultRow base_row(const Scenario& scen) {
  perf::ResultRow row;
  row.op_name = "DetectionReductionOp";
  row.variant = std::string(hr::to_string(policy(scen.params["representative"].as<std::string>())));
  row.map_shape = {scen.axis("points"), scen.axis("features")};
  row.input_shape = {scen.axis("points"), 2};
  row.out_batch = scen.axis("clusters");
  row.k = scen.axis("features");
  row.live_port = scen.live_port;
  return row;
}

void decorate_row(perf::ResultRow& row, const Scenario& scen) {
  const double points = static_cast<double>(scen.axis("points"));
  const double clusters = static_cast<double>(scen.axis("clusters"));
  const double features = static_cast<double>(scen.axis("features"));
  const double bytes = points * (3.0 * sizeof(int64_t) + features * sizeof(float)) +
                       clusters * (2.0 * sizeof(int64_t) + features * sizeof(float));
  perf::set_recipe_throughput(row, points, clusters, bytes);
  row.notes = row.live_port.empty()
                  ? "default parameters; no live override"
                  : "cold_ms is first-frame latency; override_ms is one changed frame for " +
                        scen.live_port + "; cells/s is points/s";
}

struct Traits : perf::PerfDriverDefaults {
  using Scenario = ::Scenario;
  static const char* tool_name() { return "perf_detection_reduction"; }
  static const char* tool_description() {
    return "DetectionReductionOp performance characterization (HSDK 5 Graph)";
  }
  static perf::SizeDef size_def(const std::string& id, const std::vector<int64_t>& shape) {
    return perf::recipe_size_def(id, shape, 3, 1);
  }
  static void validate_suite(const perf::ProfileSuite& suite) {
    (void)perf::validate_pilot_suite<Op>(
        suite, "radar.DetectionReductionOp.float", "",
        {{"labels", {"points"}, "int64", "device", "cluster_labels", {}},
         {"detection_idx", {"points", "two"}, "int64", "device", "reduction_indices", {}},
         {"points", {"points", "features"}, "float32", "device", "reduction_points", {}}},
        {"rep_idx", {}, {}, 4U, "reduced_points"},
        {{"representative", "representative"},
         {"score_method", "score_method"},
         {"feature_weights", "feature_weights"}},
        {{"representative", "representative_policy"},
         {"score_method", "score_method"},
         {"feature_weights", "first_double"}});
  }
  static Scenario parse_case(const perf::ProfileCase& c, const std::vector<int64_t>& shape,
                             const perf::ProfileSuite& suite) {
    auto scen = perf::pilot_scenario<Scenario>(suite, c, shape, 3);
    if (scen.axis("points") < 1 || scen.axis("clusters") < 1 ||
        scen.axis("clusters") > scen.axis("points") || scen.axis("features") < 1)
      throw std::invalid_argument("DetectionReduction invalid profile shape");
    perf::require_recipe_params(scen.params, {"representative", "score_method", "feature_weights"});
    (void)parameters(scen);
    if (scen.live_port == "representative")
      (void)policy(scen.live_value.as<std::string>());
    else if (scen.live_port == "score_method")
      (void)score(scen.live_value.as<std::string>());
    else if (scen.live_port == "feature_weights" &&
             scen.live_value.as<std::string>() != "first_double")
      throw std::invalid_argument("unknown DetectionReduction weights fixture");
    return scen;
  }
};

}  // namespace

int main(int argc, char** argv) {
  return perf::perf_main<Traits>(argc, argv);
}
