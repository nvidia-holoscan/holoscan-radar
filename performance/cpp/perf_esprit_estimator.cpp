// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Performance characterization for ESPRITEstimatorOp. Measures the shared
// covariance/Hermitian-eig/EFT front-end plus the device LS rotation solve.
// Includes a mid-session live bundle for geometry, model order, Pfa, capacity,
// and spatial smoothing. Measurement only: no timing assertions.

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/holoscan.hpp>

#include "angle_estimation/esprit_estimator_op/esprit_estimator_op.hpp"
#include "radar_common/descriptors/angle_convention.hpp"
#include "radar_common/descriptors/array_geometry.hpp"
#include "utils/perf_driver.hpp"
#include "utils/perf_process_warmup.hpp"
#include "utils/perf_report.hpp"
#include "utils/perf_runner.hpp"
#include "utils/perf_sink.hpp"
#include "utils/perf_source.hpp"
#include "utils/tensor_factory.hpp"

namespace hr = holoscan::radar;
namespace perf = holoscan::radar::perf;

namespace {

struct SizeSpec {
  std::string id;
  int64_t channels;
  int64_t detections;
  int64_t snapshots;
  std::string role;
};

struct Scenario {
  std::string id;
  std::string size_id;
  int64_t channels{0};
  int64_t detections{0};
  int64_t snapshots{0};
  int64_t override_at{0};
};

struct SnapshotPort {
  static constexpr const char* kName = "snapshot";
};

const std::vector<SizeSpec>& size_ladder() {
  static const std::vector<SizeSpec> sizes = {
      {"xs", 4, 64, 16, "small array; fast smoke"},
      {"sm", 8, 128, 32, "compact SIMO array"},
      {"md", 16, 256, 48, "common dev size"},
      {"lg", 32, 512, 64, "wide aperture, many looks"},
      {"xl", 32, 1024, 96, "upper characterization bound"}};
  return sizes;
}

const SizeSpec* find_size(const std::string& id) {
  for (const auto& size : size_ladder()) {
    if (size.id == id) {
      return &size;
    }
  }
  return nullptr;
}

const SizeSpec& must_find_size(const std::string& id) {
  const SizeSpec* size = find_size(id);
  if (size == nullptr) {
    std::cerr << "unknown size '" << id << "'\n";
    std::abort();
  }
  return *size;
}

hr::ArrayGeometry make_ula(int64_t channels, double spacing) {
  hr::ArrayGeometry geometry;
  geometry.wavelength = 1.0;
  geometry.geometry_type = hr::ArrayGeometryType::kUla;
  for (int64_t channel = 0; channel < channels; ++channel) {
    geometry.element_positions.insert(
        geometry.element_positions.end(),
        {0.0, spacing * static_cast<double>(channel), 0.0});
  }
  return geometry;
}

Scenario make_scenario(const SizeSpec& size) {
  Scenario scenario;
  scenario.size_id = size.id;
  scenario.channels = size.channels;
  scenario.detections = size.detections;
  scenario.snapshots = size.snapshots;
  std::ostringstream id;
  id << "esprit_" << size.id << "_c" << size.channels << "_n"
     << size.detections << "_s" << size.snapshots;
  scenario.id = id.str();
  return scenario;
}

template <typename T>
class ESPRITPerfApp : public holoscan::Application {
 public:
  const Scenario* scenario{nullptr};
  perf::TimingCollector* collector{nullptr};
  holoscan::TensorMap snapshot;
  int total_frames{0};

  void compose() override {
    auto estimator = make_operator<hr::ESPRITEstimatorOp<T>>(
        "esprit", holoscan::Arg("array_geometry") =
                      make_ula(scenario->channels, 0.5),
        holoscan::Arg("angle_convention") = [] {
          hr::AngleConvention convention;
          convention.preset = hr::AngleConventionPreset::kRadarBoresight;
          return convention;
        }(),
        holoscan::Arg("num_sources") = int64_t{1},
        holoscan::Arg("source_count_pfa") = 1e-4,
        holoscan::Arg("max_sources") = int64_t{4});

    auto source = make_operator<perf::PerfTensorSourceOp<SnapshotPort>>(
        "source", make_condition<holoscan::CountCondition>(total_frames));
    source->configure(collector, snapshot);
    add_flow(source, estimator, {{"snapshot", "snapshot"}});

    auto sink = make_operator<perf::PerfTerminalSinkOp>("sink");
    sink->configure(collector);
    add_flow(estimator, sink, {{"aoa_estimates", "input"}});
    for (const char* port : {"doa_multi", "source_count", "eigenvalues"}) {
      auto drain = make_operator<perf::PerfNullSinkOp>(
          std::string(port) + "_drain");
      add_flow(estimator, drain, {{port, "input"}});
    }

    auto geometry = make_operator<
        perf::PerfOverrideSourceOp<hr::ArrayGeometry>>(
        "geometry_override",
        make_condition<holoscan::CountCondition>(total_frames));
    geometry->configure(scenario->override_at,
                        make_ula(scenario->channels, 0.55));
    add_flow(geometry, estimator, {{"output", "array_geometry"}});

    auto sources = make_operator<perf::PerfOverrideSourceOp<int64_t>>(
        "sources_override",
        make_condition<holoscan::CountCondition>(total_frames));
    sources->configure(scenario->override_at, int64_t{-1});
    add_flow(sources, estimator, {{"output", "num_sources"}});

    auto pfa = make_operator<perf::PerfOverrideSourceOp<double>>(
        "pfa_override",
        make_condition<holoscan::CountCondition>(total_frames));
    pfa->configure(scenario->override_at, 0.5);
    add_flow(pfa, estimator, {{"output", "source_count_pfa"}});

    auto capacity = make_operator<perf::PerfOverrideSourceOp<int64_t>>(
        "capacity_override",
        make_condition<holoscan::CountCondition>(total_frames));
    capacity->configure(scenario->override_at, int64_t{3});
    add_flow(capacity, estimator, {{"output", "max_sources"}});

    auto smoothing = make_operator<
        perf::PerfOverrideSourceOp<std::vector<int64_t>>>(
        "smoothing_override",
        make_condition<holoscan::CountCondition>(total_frames));
    smoothing->configure(scenario->override_at, std::vector<int64_t>{2});
    add_flow(smoothing, estimator, {{"output", "spatial_smoothing"}});
  }
};

perf::ResultRow base_row(const Scenario& scenario) {
  perf::ResultRow row;
  row.scenario_id = scenario.id;
  row.op_name = "ESPRITEstimatorOp";
  row.variant = "esprit_ls";
  row.size_id = scenario.size_id;
  row.input_shape = {scenario.detections, scenario.channels,
                     scenario.snapshots};
  row.out_batch = scenario.detections;
  row.k = scenario.channels;
  return row;
}

struct Traits : perf::PerfDriverDefaults {
  using Scenario = ::Scenario;

  static const char* tool_name() { return "perf_esprit_estimator"; }
  static const char* tool_description() {
    return "ESPRITEstimatorOp performance characterization";
  }
  static const char* sizes_all_help() { return "Default ESPRIT size sweep"; }
  static const char* sizes_subset_help() {
    return "ESPRIT sweep across an explicit subset";
  }
  static const char* full_help() { return "Full ESPRIT size ladder"; }

  static std::vector<perf::SizeDef> size_ladder_defs() {
    std::vector<perf::SizeDef> result;
    for (const auto& size : size_ladder()) {
      perf::SizeDef definition;
      definition.id = size.id;
      definition.shape = {size.channels, size.detections, size.snapshots};
      definition.cells = size.detections;
      definition.role = size.role;
      result.push_back(definition);
    }
    return result;
  }

  static std::vector<Scenario> build_default_matrix() {
    return {make_scenario(must_find_size("xs")),
            make_scenario(must_find_size("sm")),
            make_scenario(must_find_size("md"))};
  }

  static std::vector<Scenario> build_size_sweep(
      const std::vector<std::string>& ids) {
    std::vector<Scenario> result;
    for (const auto& id : ids) {
      if (const SizeSpec* size = find_size(id)) {
        result.push_back(make_scenario(*size));
      }
    }
    return result;
  }

  static std::vector<Scenario> build_full_matrix() {
    std::vector<Scenario> result;
    for (const auto& size : size_ladder()) {
      result.push_back(make_scenario(size));
    }
    return result;
  }

  static void run_scenario(const Scenario& scenario,
                           const perf::RunSettings& settings,
                           perf::PerfReport* report) {
    const auto plan = perf::make_run_plan(settings, true);
    const auto samples = perf::collect_scenario_samples(
        settings, plan, [&](int, const perf::RunPlan& run_plan) {
          Scenario run = scenario;
          run.override_at = run_plan.override_at;
          perf::TimingCollector collector;
          auto tensor = perf::make_iq_cube_device<float>(
              {scenario.detections, scenario.channels, scenario.snapshots});
          collector.t_construct = perf::Clock::now();
          auto app = holoscan::make_application<ESPRITPerfApp<float>>();
          app->scenario = &run;
          app->collector = &collector;
          app->snapshot = perf::single_map(std::move(tensor), "snapshot");
          app->total_frames = run_plan.total_frames;
          app->run();
          return collector;
        });
    perf::append_setup_row(base_row(scenario), samples, report);
    perf::ResultRow row = base_row(scenario);
    perf::fill_compute_row(&row, samples);
    if (row.latency.median_ms > 0.0) {
      row.cells_per_second =
          scenario.detections / (row.latency.median_ms / 1000.0);
    }
    row.notes =
        "fixed one-source steady path; live bundle switches geometry, EFT, "
        "Pfa, capacity, and smoothing";
    report->results.push_back(row);
  }

  static void process_warmup(const perf::RunSettings&) {
    perf::run_process_warmup([] {
      perf::PerfReport discarded;
      perf::RunSettings settings;
      settings.warmup_frames = 0;
      settings.measured_frames = 2;
      settings.repeats = 1;
      run_scenario(make_scenario(must_find_size("xs")), settings, &discarded);
    });
  }
};

}  // namespace

int main(int argc, char** argv) {
  return perf::perf_main<Traits>(argc, argv);
}
