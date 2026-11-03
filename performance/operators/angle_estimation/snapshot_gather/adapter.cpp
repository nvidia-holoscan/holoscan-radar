// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Performance characterization for SnapshotGatherOp — the sparse-path bridge
// that gathers a per-detection channel-vector neighborhood out of per-channel
// RD maps. Sweeps channel count, RD-map size, detection count, and snapshot
// window. Measurement tool only: no timing assertions.
//
//  Graph-based rewrite: rd_maps, detection_idx, and any
// snapshot_window override are co-emitted with the same capture time and
// consumed through the production OnAligned + Exact contract.

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <set>
#include <iostream>
#include <memory>
#include <map>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/graph.hpp>

#include "angle_estimation/snapshot_gather_op/snapshot_gather_op.hpp"
#include "radar_common/descriptors/enums.hpp"
#include "radar_common/descriptors/max_tensor_extents.hpp"
#include "radar_common/tensor/dlpack_buffer.hpp"
#include "radar_common/tensor/tensor_utils.hpp"
#include "utils/perf_driver.hpp"
#include "utils/perf_graph.hpp"
#include "utils/perf_platform.hpp"
#include "utils/perf_report.hpp"
#include "utils/perf_recipe.hpp"
#include "utils/perf_registered_recipe.hpp"
#include "utils/perf_yaml_pilot.hpp"
#include "utils/perf_yaml_codecs.hpp"
#include "utils/perf_runner.hpp"
#include "utils/tensor_factory.hpp"

namespace hr = holoscan::radar;
namespace perf = holoscan::radar::perf;

namespace perf_snapshot_gather_adapter {

using PerfPrecision = perf::Scalar;

constexpr DLDataType kInt64Dtype{kDLInt, 64, 1};

struct Window {
  int64_t wd{0};
  int64_t wr{0};
};

struct SnapScenario : perf::RecipeScenario {
  Window window;
  Window override_window;
};

int64_t snapshots_for(const Window& w) {
  return (2 * w.wd + 1) * (2 * w.wr + 1);
}

void validate_window(const Window& window, const perf::RecipeScenario& scenario) {
  if (window.wd < 0 || window.wr < 0 || window.wd > (scenario.axis("doppler") - 1) / 2 ||
      window.wr > (scenario.axis("range") - 1) / 2)
    throw std::invalid_argument("SnapshotGather window exceeds profile plane");
}

std::string window_tag(const Window& w) {
  std::ostringstream os;
  os << "w" << w.wd << "x" << w.wr;
  return os.str();
}

perf::TimingCollector run_graph(const SnapScenario& scen, const perf::RunPlan& plan) {
  hr::MaxTensorExtents extents;
  extents.max_detections = scen.axis("detections");
  extents.max_channels = scen.axis("channels");
  extents.max_snapshots = std::max(
      snapshots_for(scen.window), scen.live_port.empty() ? 1 : snapshots_for(scen.override_window));
  hr::SnapshotGatherParams params;
  params.snapshot_window = {scen.window.wd, scen.window.wr};
  params.doppler_edge = perf::recipe_snapshot_edge(scen.params["doppler_edge"].as<std::string>());
  params.range_edge = perf::recipe_snapshot_edge(scen.params["range_edge"].as<std::string>());
  using Op = hr::SnapshotGatherOp<PerfPrecision>;
  return perf::run_recipe_selected_case<Op, 2>(
      scen, plan, extents, params, [&](const std::string& codec) -> perf::TimingCollector {
        if (codec != "snapshot_window")
          throw std::invalid_argument("unsupported SnapshotGather live codec: " + codec);
        const hr::SnapshotWindowOverride changed{.doppler = scen.override_window.wd,
                                                 .range = scen.override_window.wr};
        return perf::run_recipe_live_case<Op, 2>(scen, plan, extents, params, changed, changed);
      });
}

void fill_throughput(perf::ResultRow* row, const SnapScenario& scen) {
  const double n = static_cast<double>(scen.axis("detections"));
  const double snaps = static_cast<double>(snapshots_for(scen.window));
  const double gathered = n * static_cast<double>(scen.axis("channels")) * snaps;

  const double complex_sz = 2.0 * static_cast<double>(sizeof(PerfPrecision));
  const double idx_bytes = n * 2.0 * static_cast<double>(sizeof(int64_t));
  const double valid_bytes = n * snaps;
  perf::set_recipe_throughput(*row, n, n * snaps,
                              2.0 * gathered * complex_sz + valid_bytes + idx_bytes);
}

perf::ResultRow base_row(const SnapScenario& scen) {
  perf::ResultRow row;
  row.op_name = "SnapshotGatherOp";
  row.variant = window_tag(scen.window);
  row.map_shape = {scen.axis("doppler"), scen.axis("range")};
  row.input_shape = {scen.axis("detections"), 2};
  row.out_batch = scen.axis("detections");
  row.k = scen.axis("channels");
  row.live_port = scen.live_port;
  return row;
}

void decorate_row(perf::ResultRow& row, const SnapScenario& scen) {
  fill_throughput(&row, scen);
  row.notes = "end-to-end Graph latency; cells/s is detections/s";
  if (!scen.live_port.empty())
    row.notes += "; one-frame snapshot_window change";
}

struct SnapTraits : perf::PerfDriverDefaults {
  using Scenario = SnapScenario;

  static const char* tool_name() { return "perf_snapshot_gather"; }
  static const char* tool_description() {
    return "SnapshotGatherOp performance characterization "
           "(HSDK 5 Graph)";
  }

  static perf::SizeDef size_def(const std::string& id, const std::vector<int64_t>& shape) {
    return perf::recipe_size_def(id, shape, 4, 3);
  }

  static void validate_suite(const perf::ProfileSuite& suite) {
    (void)perf::validate_pilot_suite<hr::SnapshotGatherOp<PerfPrecision>>(
        suite, "radar.SnapshotGatherOp.float", "snapshot_gather_bounds",
        {{"rd_maps", {"channels", "doppler", "range"}, "complex64", "device", "iq_cube", {}},
         {"detection_idx", {"detections", "two"}, "int64", "device", "in_bounds_rd_indices", {}}},
        {"snapshot", {"is_valid"}, {}, 4U}, {{"snapshot_window", "snapshot_window"}},
        {{"snapshot_window", "snapshot_window"}});
  }
  static SnapScenario parse_case(const perf::ProfileCase& c, const std::vector<int64_t>& shape,
                                 const perf::ProfileSuite& suite) {
    auto scenario = perf::pilot_scenario<SnapScenario>(suite, c, shape, 4);
    const auto& params = scenario.params;
    perf::require_recipe_params(params, {"snapshot_window", "doppler_edge", "range_edge"});
    const auto window = perf::recipe_pair<int64_t>(params["snapshot_window"], "snapshot_window");
    (void)perf::recipe_snapshot_edge(params["doppler_edge"].as<std::string>());
    (void)perf::recipe_snapshot_edge(params["range_edge"].as<std::string>());
    const Window w{window[0], window[1]};
    validate_window(w, scenario);
    scenario.window = w;
    if (c.override_value) {
      const auto changed =
          perf::recipe_pair<int64_t>(scenario.live_value, "snapshot_window override");
      scenario.override_window = {changed[0], changed[1]};
      validate_window(scenario.override_window, scenario);
    }
    return scenario;
  }
};

}  // namespace perf_snapshot_gather_adapter

int main(int argc, char** argv) {
  return holoscan::radar::perf::perf_main<perf_snapshot_gather_adapter::SnapTraits>(argc, argv);
}
