// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Performance characterization for BeamscanEstimatorOp — conventional
// sparse-path Bartlett DOA on a scan grid. Sweeps detection count, channel
// count, snapshot count, and scan-grid size. Measurement tool only: no timing
// assertions.
// See docs/planning/OPERATOR_DEVELOPMENT.md.
//
//  Graph-based rewrite of the HSDK 4.x Application harness.

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <filesystem>  // NOLINT(build/c++17): repo builds with C++17
#include <set>
#include <iostream>
#include <memory>
#include <mutex>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/tensor_output_loan.hpp>

#include "angle_estimation/beamscan_estimator_op/beamscan_estimator_op.hpp"
#include "radar_common/descriptors/angle_convention.hpp"
#include "radar_common/descriptors/array_geometry.hpp"
#include "radar_common/descriptors/array_geometry_tensor.hpp"
#include "radar_common/descriptors/enums.hpp"
#include "radar_common/descriptors/max_tensor_extents.hpp"
#include "radar_common/descriptors/window_spec.hpp"
#include "radar_common/sdk/tensor_io.hpp"
#include "radar_schemas/array_geometry_bridge.hpp"
#include "radar_schemas/window_spec_bridge.hpp"
#include "utils/perf_driver.hpp"
#include "utils/perf_graph.hpp"
#include "utils/perf_payloads.hpp"
#include "utils/perf_platform.hpp"
#include "utils/perf_report.hpp"
#include "utils/perf_runner.hpp"
#include "utils/tensor_factory.hpp"

namespace hr = holoscan::radar;
namespace perf = holoscan::radar::perf;

namespace {

using perf::linspace;
using perf::make_ula_geometry;
using perf::emit_host_vector;
using perf::emit_wire_doubles;

using PerfPrecision = float;

struct SizeSpec {
  std::string id;
  int64_t channels;
  int64_t detections;
  int64_t snapshots;
  int64_t num_az;
  int64_t num_el;
  std::string role;
};

struct BeamScenario {
  std::string id;
  std::string size_id;
  int64_t channels{0};
  int64_t detections{0};
  int64_t snapshots{1};
  int64_t num_az{0};
  int64_t num_el{0};
  enum class OverrideKind {
    kNone,
    kArrayGeometry,
    kAzAngles,
    kElAngles,
    kArrayTaper,
    kPeakInterpolation,
  } override_kind{OverrideKind::kNone};
  hr::WindowType static_taper{hr::WindowType::kNone};
};

const char* override_name(BeamScenario::OverrideKind kind) {
  switch (kind) {
    case BeamScenario::OverrideKind::kNone:
      return "";
    case BeamScenario::OverrideKind::kArrayGeometry:
      return "array_geometry";
    case BeamScenario::OverrideKind::kAzAngles:
      return "az_coordinates";
    case BeamScenario::OverrideKind::kElAngles:
      return "el_coordinates";
    case BeamScenario::OverrideKind::kArrayTaper:
      return "array_taper";
    case BeamScenario::OverrideKind::kPeakInterpolation:
      return "peak_interpolation";
  }
  return "unknown";
}

struct LivePorts {
 public:
  LivePorts(hr::BeamscanParams override, BeamScenario::OverrideKind kind)
  : override_(std::move(override)), kind_(kind) {}

  void setup(holoscan::OperatorSpec& spec) {
    constexpr holoscan::TensorRepresentation kDevice{
        .memory_kind = holoscan::MemoryKind::kCudaDevice,
        .dtype = std::nullopt,
        .rank = std::nullopt,
    };
    constexpr holoscan::TensorRepresentation kHostVector{
        .memory_kind = holoscan::MemoryKind::kHost,
        .dtype = DLDataType{kDLFloat, 64U, 1U},
        .rank = 1U,
    };
    constexpr holoscan::TensorRepresentation kHostMatrix{
        .memory_kind = holoscan::MemoryKind::kHost,
        .dtype = DLDataType{kDLFloat, 64U, 1U},
        .rank = 2U,
    };
    constexpr std::size_t kHostPayloadSpan = 4096U * sizeof(double);
    spec.output(snapshot).produces_tensor(
        hr::pooled_tensor_output(kDevice, 8U, perf::PerfFrameBank::span_a()));
    spec.output(array_geometry)
        .max_emits_per_compute(1U)
        .produces_tensor(hr::pooled_tensor_output(kHostMatrix, 2U, kHostPayloadSpan));
    spec.output(az_coordinates)
        .max_emits_per_compute(1U)
        .produces_tensor(hr::pooled_tensor_output(kHostVector, 1U, kHostPayloadSpan));
    spec.output(el_coordinates)
        .max_emits_per_compute(1U)
        .produces_tensor(hr::pooled_tensor_output(kHostVector, 1U, kHostPayloadSpan));
    spec.output(array_taper)
        .max_emits_per_compute(1U)
        .produces_tensor(hr::pooled_tensor_output(kHostVector, 1U, kHostPayloadSpan));
    spec.output(peak_interpolation).max_emits_per_compute(1U);
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> emit_override(
      holoscan::ExecutionContext& context) {
    const holoscan::TimePoint time = context.activation_time();
    holoscan::expected<void, holoscan::Error> emitted;
    switch (kind_) {
      case BeamScenario::OverrideKind::kNone:
        break;
      case BeamScenario::OverrideKind::kArrayGeometry: {
        const auto packed = hr::pack_array_geometry(override_.array_geometry);
        const std::array<int64_t, 2> shape{override_.array_geometry.num_elements() + 1, 3};
        emitted = emit_wire_doubles(array_geometry, hr::to_wire(override_.array_geometry), packed,
                                    shape, time);
        break;
      }
      case BeamScenario::OverrideKind::kAzAngles:
        emitted = emit_host_vector(az_coordinates, override_.az_coordinates, time);
        break;
      case BeamScenario::OverrideKind::kElAngles:
        emitted = emit_host_vector(el_coordinates, override_.el_coordinates, time);
        break;
      case BeamScenario::OverrideKind::kArrayTaper: {
        const std::array<int64_t, 1> shape{
            static_cast<int64_t>(override_.array_taper.coefficients.size())};
        emitted = emit_wire_doubles(array_taper, hr::to_wire(override_.array_taper),
                                    override_.array_taper.coefficients, shape, time);
        break;
      }
      case BeamScenario::OverrideKind::kPeakInterpolation:
        emitted = peak_interpolation.emit(override_.peak_interpolation,
                                          holoscan::EmitOptions{.capture_time = time});
        break;
    }
    if (!emitted)
      return emitted;
    return {};
  }

  holoscan::Output<holoscan::Tensor> snapshot{"snapshot"};
  holoscan::Output<hr::WireArrayGeometry> array_geometry{"array_geometry"};
  holoscan::Output<holoscan::Tensor> az_coordinates{"az_coordinates"};
  holoscan::Output<holoscan::Tensor> el_coordinates{"el_coordinates"};
  holoscan::Output<hr::WireWindowSpec> array_taper{"array_taper"};
  holoscan::Output<hr::PeakInterpolation> peak_interpolation{"peak_interpolation"};

 private:
  hr::BeamscanParams override_;
  BeamScenario::OverrideKind kind_{BeamScenario::OverrideKind::kNone};
};

using LiveSource = perf::PerfLiveSourceOp<LivePorts, &LivePorts::snapshot>;

perf::TimingCollector run_beamscan_graph(const BeamScenario& scen, const perf::RunPlan& plan) {
  auto snapshot =
      perf::make_iq_cube_device<PerfPrecision>({scen.detections, scen.channels, scen.snapshots});
  hr::BeamscanParams initial;
  initial.array_geometry = make_ula_geometry(scen.channels, 0.5, 1.0);
  initial.az_coordinates = linspace(-0.5, 0.5, scen.num_az);
  initial.el_coordinates = linspace(-0.3, 0.3, scen.num_el);
  initial.peak_interpolation = hr::PeakInterpolation::kNone;
  initial.array_taper.type = scen.static_taper;
  hr::BeamscanParams override = initial;
  switch (scen.override_kind) {
    case BeamScenario::OverrideKind::kNone:
      break;
    case BeamScenario::OverrideKind::kArrayGeometry:
      override.array_geometry = make_ula_geometry(scen.channels, 0.45, 1.0);
      break;
    case BeamScenario::OverrideKind::kAzAngles:
      override.az_coordinates = linspace(-0.45, 0.55, scen.num_az);
      break;
    case BeamScenario::OverrideKind::kElAngles:
      override.el_coordinates = linspace(-0.25, 0.35, scen.num_el);
      break;
    case BeamScenario::OverrideKind::kArrayTaper:
      override.array_taper.type = hr::WindowType::kHann;
      break;
    case BeamScenario::OverrideKind::kPeakInterpolation:
      override.peak_interpolation = hr::PeakInterpolation::kParabolic;
      break;
  }
  hr::MaxTensorExtents extents;
  extents.max_items = scen.detections;
  extents.max_snapshots = scen.snapshots;
  extents.max_az = scen.num_az;
  extents.max_el = scen.num_el;
  perf::TimingCollector collector;
  collector.t_construct = perf::Clock::now();
  perf::PerfFrameBank::reset(std::move(snapshot), plan.total_frames, &collector);

  holoscan::Graph graph{"beamscan-perf"};
  const auto source = graph.op<LiveSource>("src", plan.override_at, override, scen.override_kind);
  auto op = graph.op<hr::BeamscanEstimatorOp<PerfPrecision>>("beamscan", extents);
  op.set_params(std::move(initial));
  const auto sink = graph.op<perf::PerfTerminalSinkOp>("aoa_sink");
  const auto spectrum = graph.op<perf::PerfNullSinkOp>("spectrum_drain");
  const holoscan::ConnectionOptions q{.queue_depth = 8U};
  graph.add_flow(source->snapshot, op->snapshot, q);
  if (scen.override_kind == BeamScenario::OverrideKind::kArrayGeometry)
    graph.add_flow(source->array_geometry, op->array_geometry, q);
  if (scen.override_kind == BeamScenario::OverrideKind::kAzAngles)
    graph.add_flow(source->az_coordinates, op->az_coordinates, q);
  if (scen.override_kind == BeamScenario::OverrideKind::kElAngles)
    graph.add_flow(source->el_coordinates, op->el_coordinates, q);
  if (scen.override_kind == BeamScenario::OverrideKind::kArrayTaper)
    graph.add_flow(source->array_taper, op->array_taper, q);
  if (scen.override_kind == BeamScenario::OverrideKind::kPeakInterpolation)
    graph.add_flow(source->peak_interpolation, op->peak_interpolation, q);
  graph.add_flow(op->aoa_estimates, sink->in, q);
  graph.add_flow(op->angle_spectrum, spectrum->in, q);

  holoscan::CompileOptions options;
  perf::bind_tensor_out(options, "src", "snapshot");
  perf::bind_tensor_out(options, "beamscan", "aoa_estimates");
  perf::bind_tensor_out(options, "beamscan", "angle_spectrum");
  perf::execute_perf_graph(graph, std::move(options), plan.total_frames, "beamscan-perf",
                           &collector);
  return collector;
}

void fill_throughput(perf::ResultRow* row, const BeamScenario& scen, double median_ms) {
  if (median_ms <= 0.0) {
    return;
  }
  const double sec = median_ms / 1000.0;
  const double n = static_cast<double>(scen.detections);
  const double beams = static_cast<double>(scen.num_az) * static_cast<double>(scen.num_el);
  const double ch = static_cast<double>(scen.channels);
  row->cells_per_second = n / sec;
  row->slices_per_second = n * beams / sec;

  const double csz = 2.0 * static_cast<double>(sizeof(PerfPrecision));
  const double snap_bytes = n * ch * static_cast<double>(scen.snapshots) * csz;
  const double weight_bytes = beams * ch * csz;
  const double spec_bytes = n * beams * static_cast<double>(sizeof(PerfPrecision));
  const double aoa_bytes = n * 2.0 * static_cast<double>(sizeof(PerfPrecision));
  row->estimated_effective_gbps = (snap_bytes + weight_bytes + spec_bytes + aoa_bytes) / sec / 1e9;
}

perf::ResultRow base_row(const BeamScenario& scen) {
  perf::ResultRow row;
  row.scenario_id = scen.id;
  row.op_name = "BeamscanEstimatorOp";
  row.variant = "bartlett";
  row.size_id = scen.size_id;
  row.map_shape = {scen.num_az, scen.num_el};
  row.input_shape = {scen.detections, scen.channels, scen.snapshots};
  row.out_batch = scen.detections;
  row.k = scen.channels;
  row.live_port = override_name(scen.override_kind);
  return row;
}

BeamScenario make_scenario(const SizeSpec& s,
                           BeamScenario::OverrideKind kind = BeamScenario::OverrideKind::kNone) {
  BeamScenario sc;
  sc.size_id = s.id;
  sc.channels = s.channels;
  sc.detections = s.detections;
  sc.snapshots = s.snapshots;
  sc.num_az = s.num_az;
  sc.num_el = s.num_el;
  sc.override_kind = kind;
  std::ostringstream id;
  id << "beamscan_" << s.id << "_c" << s.channels << "_n" << s.detections << "_g" << s.num_az << "x"
     << s.num_el << "_s" << s.snapshots << "_" << override_name(kind);
  sc.id = id.str();
  return sc;
}

struct BeamTraits : perf::PerfDriverDefaults {
  using Scenario = BeamScenario;

  static const char* tool_name() { return "perf_beamscan_estimator"; }
  static const char* tool_description() {
    return "BeamscanEstimatorOp performance "
           "characterization (HSDK 5 Graph)";
  }

  static perf::ProfileSuite suite() {
    return perf::load_profile_suite(
        std::filesystem::path(HOLOSCAN_RADAR_PERF_PROFILE_DIR) / "beamscan_estimator.yaml",
        "beamscan_estimator");
  }
  static std::vector<perf::SizeDef> size_ladder_defs() {
    const auto profiles = suite();
    std::vector<perf::SizeDef> result;
    for (size_t i = 0; i < perf::kProfiles.size(); ++i) {
      const auto& shape = profiles.shapes[i];
      if (shape.size() != 5)
        throw std::invalid_argument("Beamscan profile rank mismatch");
      result.push_back({std::string(perf::kProfiles[i]), shape, shape[0] * shape[1] * shape[2],
                        std::string(perf::kProfiles[i])});
    }
    return result;
  }
  static Scenario parse_case(const perf::ProfileCase& c, const std::vector<int64_t>& shape) {
    if (shape.size() != 5 || shape[0] < 2 || shape[1] < 1 || shape[2] < 1 || shape[3] < 1 ||
        shape[4] < 1)
      throw std::invalid_argument("Beamscan invalid shape");
    perf::require_keys(c.parameters, {"array_taper"}, "parameters");
    const auto taper = c.parameters["array_taper"].as<std::string>();
    if (taper != "none" && taper != "hann")
      throw std::invalid_argument("Beamscan invalid taper");
    const SizeSpec size{c.profile, shape[0], shape[1], shape[2], shape[3], shape[4], c.profile};
    auto scenario = make_scenario(size);
    scenario.id = c.id;
    scenario.static_taper = taper == "hann" ? hr::WindowType::kHann : hr::WindowType::kNone;
    if (c.override_value) {
      const auto port = (*c.override_value)["port"].as<std::string>();
      const auto value = (*c.override_value)["value"].as<std::string>();
      if (port == "array_geometry" && value == "spacing_045")
        scenario.override_kind = Scenario::OverrideKind::kArrayGeometry;
      else if (port == "az_coordinates" && value == "shift_005")
        scenario.override_kind = Scenario::OverrideKind::kAzAngles;
      else if (port == "el_coordinates" && value == "shift_005")
        scenario.override_kind = Scenario::OverrideKind::kElAngles;
      else if (port == "array_taper" && value == "hann")
        scenario.override_kind = Scenario::OverrideKind::kArrayTaper;
      else if (port == "peak_interpolation" && value == "parabolic")
        scenario.override_kind = Scenario::OverrideKind::kPeakInterpolation;
      else
        throw std::invalid_argument("Beamscan invalid live override");
    }
    return scenario;
  }

  static void run_scenario(const BeamScenario& scen, const perf::RunSettings& rs,
                           perf::PerfReport* report) {
    perf::measure_scenario(
        rs, scen.override_kind != BeamScenario::OverrideKind::kNone,
        [&](int /*rep*/, const perf::RunPlan& p) {
          return run_beamscan_graph(scen, p);
        },
        base_row(scen), report,
        [&](perf::ResultRow& row) {
          fill_throughput(&row, scen, row.latency.median_ms);
          row.notes =
              "cold_ms is first-frame latency; median/p90 "
              "are post-warmup steady state; override_ms "
              "isolates " +
              std::string(override_name(scen.override_kind)) + "; cells/s is detections/s";
          if (row.live_port.empty())
            row.notes = "default parameters; no live override";
        });
  }
};

}  // namespace

int main(int argc, char** argv) {
  return holoscan::radar::perf::perf_main<BeamTraits>(argc, argv);
}
