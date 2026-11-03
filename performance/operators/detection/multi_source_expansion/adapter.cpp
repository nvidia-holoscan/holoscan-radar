// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cuda_runtime_api.h>

#include <array>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "detection/multi_source_expansion_op/multi_source_expansion_op.hpp"
#include "radar_common/tensor/dlpack_buffer.hpp"
#include "radar_common/tensor/tensor_utils.hpp"
#include "utils/perf_driver.hpp"
#include "utils/perf_yaml_pilot.hpp"

namespace hr = holoscan::radar;
namespace perf = holoscan::radar::perf;

namespace {

using Op = hr::MultiSourceExpansionOp<perf::Scalar>;
struct Scenario : perf::RecipeScenario {};

template <typename Element>
std::shared_ptr<holoscan::Tensor> upload(const std::vector<Element>& host,
                                         const std::vector<int64_t>& shape, DLDataType dtype) {
  auto tensor = hr::make_device_tensor(shape, dtype);
  auto writable = tensor->mutable_data();
  if (!host.empty() &&
      (!writable || cudaMemcpy(writable.value(), host.data(), host.size() * sizeof(Element),
                               cudaMemcpyHostToDevice) != cudaSuccess))
    throw std::runtime_error("MultiSourceExpansion fixture upload failed");
  return tensor;
}

std::array<std::shared_ptr<holoscan::Tensor>, 6> stage_inputs(const Scenario& scen) {
  const auto rows = scen.axis("rows");
  const auto selected = scen.axis("selected");
  const auto sources = scen.axis("sources");
  const auto columns = scen.axis("index_columns");
  std::vector<int64_t> indices(static_cast<std::size_t>(rows * columns));
  std::vector<perf::Scalar> baseline(static_cast<std::size_t>(rows * 2));
  for (int64_t row = 0; row < rows; ++row) {
    for (int64_t column = 0; column < columns; ++column)
      indices[static_cast<std::size_t>(row * columns + column)] = row * 11 + column;
    baseline[static_cast<std::size_t>(row * 2)] = static_cast<perf::Scalar>((row % 101) * 0.001);
  }
  std::vector<int64_t> selected_idx(static_cast<std::size_t>(selected));
  std::vector<perf::Scalar> resolved(static_cast<std::size_t>(selected * 2));
  std::vector<perf::Scalar> multi(static_cast<std::size_t>(selected * sources * 2));
  std::vector<int32_t> counts(static_cast<std::size_t>(selected), static_cast<int32_t>(sources));
  for (int64_t row = 0; row < selected; ++row) {
    selected_idx[static_cast<std::size_t>(row)] = row;
    resolved[static_cast<std::size_t>(row * 2)] = -0.1F;
    for (int64_t source = 0; source < sources; ++source)
      multi[static_cast<std::size_t>((row * sources + source) * 2)] =
          -0.1F + 0.05F * static_cast<perf::Scalar>(source);
  }
  return {upload(indices, {rows, columns}, DLDataType{kDLInt, 64, 1}),
          upload(baseline, {rows, 2}, hr::real_dtype<perf::Scalar>()),
          upload(selected_idx, {selected}, DLDataType{kDLInt, 64, 1}),
          upload(resolved, {selected, 2}, hr::real_dtype<perf::Scalar>()),
          upload(multi, {selected, sources, 2}, hr::real_dtype<perf::Scalar>()),
          upload(counts, {selected}, DLDataType{kDLInt, 32, 1})};
}

perf::TimingCollector run_graph(const Scenario& scen, const perf::RunPlan& plan) {
  hr::MaxTensorExtents extents;
  extents.max_items = scen.axis("rows");
  extents.max_sources = scen.axis("sources");
  extents.max_points = scen.axis("rows") + scen.axis("selected") * (scen.axis("sources") - 1);
  return perf::run_recipe_staged_case<Op, 6>(scen, plan, extents, Op::Params{}, stage_inputs(scen));
}

perf::ResultRow base_row(const Scenario& scen) {
  perf::ResultRow row;
  row.op_name = "MultiSourceExpansionOp";
  row.variant = "confirmed_sources";
  row.input_shape = {scen.axis("rows"), scen.axis("index_columns")};
  row.out_batch = scen.axis("rows") + scen.axis("selected") * (scen.axis("sources") - 1);
  row.k = scen.axis("sources");
  row.notes = "both outputs drained";
  return row;
}

void decorate_row(perf::ResultRow& row, const Scenario& scen) {
  perf::set_recipe_throughput(row, static_cast<double>(scen.axis("rows")),
                              static_cast<double>(row.out_batch));
  row.notes =
      "cold_ms is first frame; median/p90 are steady state; both outputs are drained; cells/s is "
      "original rows/s; slices/s is emitted rows/s";
}

struct Traits : perf::PerfDriverDefaults {
  using Scenario = ::Scenario;
  static const char* tool_name() { return "perf_multi_source_expansion"; }
  static const char* tool_description() {
    return "MultiSourceExpansionOp HSDK 5 Graph characterization";
  }
  static perf::SizeDef size_def(const std::string& id, const std::vector<int64_t>& shape) {
    return perf::recipe_size_def(id, shape, 1, 1);
  }
  static void validate_suite(const perf::ProfileSuite& suite) {
    (void)perf::validate_pilot_suite<Op>(
        suite, "radar.MultiSourceExpansionOp.float", "multi_source_expansion_bounds",
        {{"detection_idx", {"rows", "index_columns"}, "int64", "device", "typed", {}},
         {"baseline_aoa", {"rows", "two"}, "float32", "device", "typed", {}},
         {"selected_idx", {"selected"}, "int64", "device", "typed", {}},
         {"resolved_aoa", {"selected", "two"}, "float32", "device", "typed", {}},
         {"doa_multi", {"selected", "sources", "two"}, "float32", "device", "typed", {}},
         {"source_count", {"selected"}, "int32", "device", "typed", {}}},
        {"detection_idx_out", {"aoa_estimates_out"}, {}, 4U}, {}, {});
  }
  static Scenario parse_case(const perf::ProfileCase& c, const std::vector<int64_t>& shape,
                             const perf::ProfileSuite& suite) {
    auto scen = perf::pilot_scenario<Scenario>(suite, c, shape, 1);
    if (scen.axis("rows") < 1 || scen.axis("selected_percent") < 1 ||
        scen.axis("selected_percent") > 100 || scen.axis("sources") < 1 ||
        scen.axis("sources") > 4 || scen.axis("index_columns") < 2 ||
        scen.axis("index_columns") > 10 || scen.axis("selected") < 1)
      throw std::invalid_argument("MultiSourceExpansion invalid fixture");
    perf::require_recipe_params(scen.params, {});
    return scen;
  }
};

}  // namespace

int main(int argc, char** argv) {
  return perf::perf_main<Traits>(argc, argv);
}
