// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <complex>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "angle_estimation/subarray_selection_op/subarray_selection_op.hpp"
#include "utils/perf_driver.hpp"
#include "utils/perf_yaml_pilot.hpp"

namespace hr = holoscan::radar;
namespace perf = holoscan::radar::perf;

namespace {

using Op = hr::SubarraySelectionOp<perf::Scalar>;
struct Scenario : perf::RecipeScenario {};

hr::ArrayGeometry holey_geometry(const Scenario& scen, double wavelength) {
  hr::ArrayGeometry geometry;
  geometry.wavelength = wavelength;
  geometry.geometry_type = hr::ArrayGeometryType::kUla;
  const auto channels = scen.axis("channels");
  const auto selected = channels - channels / 4;
  geometry.element_positions.reserve(static_cast<std::size_t>(channels * 3));
  for (int64_t channel = 0; channel < channels; ++channel) {
    const int64_t lattice = channel < selected ? channel : channel + 1;
    geometry.element_positions.insert(geometry.element_positions.end(),
                                      {0.0, 0.5 * static_cast<double>(lattice), 0.0});
  }
  return geometry;
}

hr::SubarraySelectionParams parameters(const Scenario& scen) {
  hr::SubarraySelectionParams params;
  params.array_geometry =
      holey_geometry(scen, scen.params["array_geometry"]["wavelength"].as<double>());
  params.min_subarray_size = scen.params["min_subarray_size"].as<int64_t>();
  return params;
}

perf::TimingCollector run_graph(const Scenario& scen, const perf::RunPlan& plan) {
  const auto extents = perf::direct_recipe_extents(scen.recipe, scen.values);
  const auto params = parameters(scen);
  const auto run = [&]<std::size_t N>() -> perf::TimingCollector {
    return perf::run_recipe_selected_case<Op, N>(
        scen, plan, extents, params, [&](const std::string& codec) -> perf::TimingCollector {
          if (codec == "holey_ula_patch") {
            const auto wavelength = scen.live_value["wavelength"].as<double>();
            const auto value = hr::to_wire(holey_geometry(scen, wavelength));
            return perf::run_recipe_live_case<Op, N>(scen, plan, extents, params, value, value);
          }
          if (codec == "contiguous_subarray") {
            std::vector<int64_t> indices;
            for (int64_t i = 0; i < scen.axis("channels") - scen.axis("channels") / 4; ++i)
              indices.push_back(i);
            return perf::run_recipe_live_case<Op, N>(scen, plan, extents, params, indices, indices);
          }
          throw std::invalid_argument("unsupported SubarraySelection live codec: " + codec);
        });
  };
  if (scen.recipe.inputs.size() == 1U)
    return run.template operator()<1>();
  return run.template operator()<2>();
}

perf::ResultRow base_row(const Scenario& scen) {
  perf::ResultRow row;
  row.op_name = "SubarraySelectionOp";
  row.variant = "holey_ula";
  row.input_shape = {scen.axis("items"), scen.axis("channels"), scen.axis("snapshots")};
  row.out_batch = scen.axis("items");
  row.k = scen.axis("channels") - scen.axis("channels") / 4;
  row.live_port = scen.live_port;
  return row;
}

void decorate_row(perf::ResultRow& row, const Scenario& scen) {
  const auto values = static_cast<double>(scen.axis("items")) *
                      static_cast<double>(scen.axis("channels") - scen.axis("channels") / 4) *
                      static_cast<double>(scen.axis("snapshots"));
  perf::set_recipe_throughput(row, static_cast<double>(scen.axis("items")), values,
                              4.0 * values * sizeof(std::complex<perf::Scalar>));
  row.notes = row.live_port.empty()
                  ? "default parameters; no live override"
                  : "cold_ms includes first selection plan; override_ms is one changed frame for " +
                        row.live_port + "; cells/s is items/s";
}

struct Traits : perf::PerfDriverDefaults {
  using Scenario = ::Scenario;
  static const char* tool_name() { return "perf_subarray_selection"; }
  static const char* tool_description() {
    return "SubarraySelectionOp performance characterization (HSDK 5 Graph)";
  }
  static perf::SizeDef size_def(const std::string& id, const std::vector<int64_t>& shape) {
    return perf::recipe_size_def(id, shape, 3, 3);
  }
  static void validate_suite(const perf::ProfileSuite& suite) {
    (void)perf::validate_pilot_suite<Op>(
        suite, "radar.SubarraySelectionOp.float", "",
        {{"is_valid", {"items", "snapshots"}, "bool", "device", "valid_mask", {}},
         {"snapshot", {"items", "channels", "snapshots"}, "complex64", "device", "iq_cube", {}}},
        {"snapshot_out",
         {"array_geometry"},
         {"is_valid_out"},
         4U,
         "",
         {{"array_geometry", "array_geometry"}}},
        {{"array_geometry", "array_geometry_in"}, {"subarray", "subarray"}},
        {{"array_geometry", "holey_ula_patch"}, {"subarray", "contiguous_subarray"}});
  }
  static Scenario parse_case(const perf::ProfileCase& c, const std::vector<int64_t>& shape,
                             const perf::ProfileSuite& suite) {
    auto scen = perf::pilot_scenario<Scenario>(suite, c, shape, 3);
    if (scen.axis("items") < 1 || scen.axis("channels") < 4 || scen.axis("snapshots") < 1)
      throw std::invalid_argument("SubarraySelection invalid shape");
    perf::require_recipe_params(scen.params, {"array_geometry", "min_subarray_size"});
    const auto selected = scen.axis("channels") - scen.axis("channels") / 4;
    if (scen.params["min_subarray_size"].as<int64_t>() < 2 ||
        scen.params["min_subarray_size"].as<int64_t>() > selected)
      throw std::invalid_argument("SubarraySelection invalid minimum size");
    (void)parameters(scen);
    if (scen.live_port == "array_geometry" && scen.live_value["wavelength"].as<double>() <= 0)
      throw std::invalid_argument("SubarraySelection invalid live wavelength");
    if (scen.live_port == "subarray" && scen.live_value.as<std::string>() != "contiguous")
      throw std::invalid_argument("SubarraySelection live subarray must be contiguous");
    const auto with_mask = scen.axis("with_mask");
    const auto mask_dtype = scen.axis("mask_dtype");
    if ((with_mask != 0 && with_mask != 1) || (mask_dtype != 0 && mask_dtype != 1))
      throw std::invalid_argument("SubarraySelection invalid mask fixture");
    if (with_mask == 0)
      scen.recipe.inputs.erase(scen.recipe.inputs.begin());
    else if (mask_dtype == 1)
      scen.recipe.inputs.front().dtype = "uint8";
    return scen;
  }
};

}  // namespace

int main(int argc, char** argv) {
  return perf::perf_main<Traits>(argc, argv);
}
