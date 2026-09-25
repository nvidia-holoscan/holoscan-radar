// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Performance characterization for PhaseShiftBeamformerOp — conventional
// (Bartlett) dense-path beamforming forming a beam for every range-Doppler
// cell (the Range-Doppler-Angle cube). Sweeps channel count, scan-grid size,
// and RD extent. Measurement tool only: no timing assertions.
// See docs/planning/OPERATOR_DEVELOPMENT.md.
//
//  Graph-based rewrite of the HSDK 4.x Application harness.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
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

#include "angle_estimation/phase_shift_beamformer_op/phase_shift_beamformer_op.hpp"
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

using PerfPrecision = float;  // v1 characterizes single precision.

struct SizeSpec {
  std::string id;
  int64_t channels;
  int64_t num_az;
  int64_t num_el;
  int64_t num_doppler;
  int64_t num_range;
  std::string role;
};

struct BeamScenario {
  std::string id;
  std::string size_id;
  int64_t channels{0};
  int64_t num_az{0};
  int64_t num_el{0};
  int64_t num_doppler{0};
  int64_t num_range{0};
  enum class OverrideKind {
    kNone,
    kArrayGeometry,
    kAzAngles,
    kElAngles,
    kArrayTaper,
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
  }
  return "unknown";
}

struct LivePorts {
 public:
  LivePorts(hr::PhaseShiftBeamformerParams override, BeamScenario::OverrideKind kind)
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
    spec.output(rd_maps).produces_tensor(
        hr::pooled_tensor_output(kDevice, 8U, perf::PerfFrameBank::span_a()));
    constexpr std::size_t kHostPayloadSpan = 4096U * sizeof(double);
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
    }
    if (!emitted)
      return emitted;
    return {};
  }

  holoscan::Output<holoscan::Tensor> rd_maps{"rd_maps"};
  holoscan::Output<hr::WireArrayGeometry> array_geometry{"array_geometry"};
  holoscan::Output<holoscan::Tensor> az_coordinates{"az_coordinates"};
  holoscan::Output<holoscan::Tensor> el_coordinates{"el_coordinates"};
  holoscan::Output<hr::WireWindowSpec> array_taper{"array_taper"};

 private:
  hr::PhaseShiftBeamformerParams override_;
  BeamScenario::OverrideKind kind_{BeamScenario::OverrideKind::kNone};
};

using LiveSource = perf::PerfLiveSourceOp<LivePorts, &LivePorts::rd_maps>;

perf::TimingCollector run_graph(const BeamScenario& scen, const perf::RunPlan& plan) {
  auto cube =
      perf::make_iq_cube_device<PerfPrecision>({scen.channels, scen.num_doppler, scen.num_range});
  hr::PhaseShiftBeamformerParams initial;
  initial.array_geometry = make_ula_geometry(scen.channels, 0.5, 1.0);
  initial.az_coordinates = linspace(-0.5, 0.5, scen.num_az);
  initial.el_coordinates = linspace(-0.3, 0.3, scen.num_el);
  initial.array_taper.type = scen.static_taper;
  hr::PhaseShiftBeamformerParams override = initial;
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
  }

  hr::MaxTensorExtents extents;
  extents.max_az = scen.num_az;
  extents.max_el = scen.num_el;
  extents.max_doppler_bins = scen.num_doppler;
  extents.max_range_bins = scen.num_range;
  perf::TimingCollector collector;
  collector.t_construct = perf::Clock::now();
  perf::PerfFrameBank::reset(std::move(cube), plan.total_frames, &collector);

  holoscan::Graph graph{"beamformer-perf"};
  const auto src = graph.op<LiveSource>("src", plan.override_at, override, scen.override_kind);
  auto op = graph.op<hr::PhaseShiftBeamformerOp<PerfPrecision>>("beamformer", extents);
  op.set_params(std::move(initial));
  const auto sink = graph.op<perf::PerfTerminalSinkOp>("out_sink");
  const holoscan::ConnectionOptions q{.queue_depth = 8U};
  auto add_flow = [&](auto& output, auto& input, const char* edge) {
    try {
      graph.add_flow(output, input, q);
    } catch (const holoscan::RuntimeError& error) {
      throw std::runtime_error(std::string("beamformer-perf edge ") + edge + ": " + error.what());
    }
  };
  add_flow(src->rd_maps, op->rd_maps, "rd_maps");
  if (scen.override_kind == BeamScenario::OverrideKind::kArrayGeometry)
    add_flow(src->array_geometry, op->array_geometry, "array_geometry");
  if (scen.override_kind == BeamScenario::OverrideKind::kAzAngles)
    add_flow(src->az_coordinates, op->az_coordinates, "az_coordinates");
  if (scen.override_kind == BeamScenario::OverrideKind::kElAngles)
    add_flow(src->el_coordinates, op->el_coordinates, "el_coordinates");
  if (scen.override_kind == BeamScenario::OverrideKind::kArrayTaper)
    add_flow(src->array_taper, op->array_taper, "array_taper");
  add_flow(op->rda_cube, sink->in, "rda_cube");

  holoscan::CompileOptions options;
  perf::bind_tensor_out(options, "src", "rd_maps");
  perf::bind_tensor_out(options, "beamformer", "rda_cube");
  perf::bind_tensor_out(options, "beamformer", "az_grid");
  perf::bind_tensor_out(options, "beamformer", "el_grid");
  perf::execute_perf_graph(graph, std::move(options), plan.total_frames, "beamformer-perf",
                           &collector);
  return collector;
}

void fill_throughput(perf::ResultRow* row, const BeamScenario& scen, double median_ms) {
  if (median_ms <= 0.0) {
    return;
  }
  const double sec = median_ms / 1000.0;
  const double cells = static_cast<double>(scen.num_doppler) * static_cast<double>(scen.num_range);
  const double beams = static_cast<double>(scen.num_az) * static_cast<double>(scen.num_el);
  const double ch = static_cast<double>(scen.channels);
  row->cells_per_second = cells / sec;
  row->slices_per_second = cells * beams / sec;

  const double complex_sz = 2.0 * static_cast<double>(sizeof(PerfPrecision));
  const double in_bytes = cells * ch * complex_sz;
  const double weight_bytes = beams * ch * complex_sz;
  const double out_bytes = cells * beams * complex_sz;
  row->estimated_effective_gbps = (in_bytes + weight_bytes + out_bytes) / sec / 1e9;
}

perf::ResultRow base_row(const BeamScenario& scen) {
  perf::ResultRow row;
  row.op_name = "PhaseShiftBeamformerOp";
  row.variant = "bartlett_dense";
  row.live_port = override_name(scen.override_kind);
  row.map_shape = {scen.num_az, scen.num_el};
  row.input_shape = {scen.channels, scen.num_doppler, scen.num_range};
  row.out_batch = scen.num_doppler * scen.num_range;
  row.k = scen.channels;

  return row;
}

BeamScenario make_scenario(const SizeSpec& s,
                           BeamScenario::OverrideKind kind = BeamScenario::OverrideKind::kNone) {
  BeamScenario sc;
  sc.size_id = s.id;
  sc.channels = s.channels;
  sc.num_az = s.num_az;
  sc.num_el = s.num_el;
  sc.num_doppler = s.num_doppler;
  sc.num_range = s.num_range;
  sc.override_kind = kind;
  std::ostringstream id;
  id << "beamform_" << s.id << "_c" << s.channels << "_g" << s.num_az << "x" << s.num_el << "_rd"
     << s.num_doppler << "x" << s.num_range << "_live_" << override_name(kind);
  sc.id = id.str();
  return sc;
}

void decorate_row(perf::ResultRow& row, const BeamScenario& scen) {
  fill_throughput(&row, scen, row.latency.median_ms);
  row.notes =
      "cold_ms is first-frame latency; median/p90 are "
      "pre-change steady state; override_ms is the one changed frame for " +
      std::string(override_name(scen.override_kind)) +
      " changes the steering configuration; cells/s is "
      "RD-cells/s (beams/s = slices/s)";
  if (row.live_port.empty())
    row.notes = "default parameters; no live override";
}

struct BeamTraits : perf::PerfDriverDefaults {
  using Scenario = BeamScenario;

  static const char* tool_name() { return "perf_phase_shift_beamformer"; }
  static const char* tool_description() {
    return "PhaseShiftBeamformerOp performance characterization "
           "(HSDK 5 Graph)";
  }

  static perf::SizeDef size_def(const std::string& id, const std::vector<int64_t>& shape) {
    if (shape.size() != 5)
      throw std::invalid_argument("PhaseShiftBeamformer profile rank mismatch");
    return {id, shape, shape[0] * shape[3] * shape[4], id};
  }
  static Scenario parse_case(const perf::ProfileCase& c, const std::vector<int64_t>& shape) {
    if (shape.size() != 5 || shape[0] < 2 || shape[1] < 1 || shape[2] < 1 || shape[3] < 1 ||
        shape[4] < 1)
      throw std::invalid_argument("PhaseShiftBeamformer invalid shape");
    perf::require_keys(c.parameters, {"array_taper"}, "parameters");
    const auto taper = c.parameters["array_taper"].as<std::string>();
    if (taper != "none" && taper != "hann")
      throw std::invalid_argument("PhaseShiftBeamformer invalid taper");
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
      else
        throw std::invalid_argument("PhaseShiftBeamformer invalid live override");
    }
    return scenario;
  }
};

}  // namespace

int main(int argc, char** argv) {
  return holoscan::radar::perf::perf_main<BeamTraits>(argc, argv);
}
