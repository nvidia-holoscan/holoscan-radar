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
#include "utils/perf_platform.hpp"
#include "utils/perf_process_warmup.hpp"
#include "utils/perf_report.hpp"
#include "utils/perf_runner.hpp"
#include "utils/tensor_factory.hpp"

namespace hr = holoscan::radar;
namespace perf = holoscan::radar::perf;

namespace {

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
    kArrayGeometry,
    kAzAngles,
    kElAngles,
    kArrayTaper,
    kPeakInterpolation,
  } override_kind{OverrideKind::kAzAngles};
};

const char* override_name(BeamScenario::OverrideKind kind) {
  switch (kind) {
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

const std::vector<SizeSpec>& size_ladder() {
  static const std::vector<SizeSpec> ladder = {
      {"xs", 4, 64, 1, 21, 11, "small array; one-look compatibility"},
      {"sm", 8, 256, 3, 41, 21, "compact multi-look array"},
      {"md", 16, 1024, 9, 61, 31, "3x3 sparse-window look count"},
      {"lg", 32, 4096, 9, 81, 41, "dense virtual array"},
      {"xl", 64, 16384, 9, 101, 51,
       "upper characterization bound"}};
  return ladder;
}

const SizeSpec* find_size(const std::string& id) {
  for (const auto& s : size_ladder()) {
    if (s.id == id) {
      return &s;
    }
  }
  return nullptr;
}

const SizeSpec& must_find_size(const std::string& id) {
  const SizeSpec* s = find_size(id);
  if (s == nullptr) {
    std::cerr << "internal error: unknown size id '"
              << id << "'\n";
    std::abort();
  }
  return *s;
}

std::vector<double> linspace(double start, double stop,
                             int64_t n) {
  std::vector<double> out(static_cast<size_t>(n));
  if (n == 1) {
    out[0] = start;
    return out;
  }
  const double step =
      (stop - start) / static_cast<double>(n - 1);
  for (int64_t i = 0; i < n; ++i) {
    out[static_cast<size_t>(i)] =
        start + step * static_cast<double>(i);
  }
  return out;
}

hr::ArrayGeometry make_ula_geometry(int64_t num_elements,
                                    double spacing,
                                    double wavelength) {
  hr::ArrayGeometry geom;
  geom.wavelength = wavelength;
  geom.geometry_type = hr::ArrayGeometryType::kUla;
  geom.element_positions.reserve(
      static_cast<size_t>(num_elements * 3));
  for (int64_t m = 0; m < num_elements; ++m) {
    geom.element_positions.push_back(0.0);
    geom.element_positions.push_back(
        static_cast<double>(m) * spacing);
    geom.element_positions.push_back(0.0);
  }
  return geom;
}

holoscan::expected<void, holoscan::Error> emit_host_vector(
    holoscan::Output<holoscan::Tensor>& output,
    std::span<const double> values, holoscan::TimePoint capture_time) {
  const std::array<int64_t, 1> shape{
      static_cast<int64_t>(values.size())};
  constexpr DLDataType dtype{kDLFloat, 64U, 1U};
  auto loan = output.allocate_tensor(
      holoscan::TensorLoanRequest{.shape = shape, .dtype = dtype});
  if (!loan) {
    return holoscan::make_unexpected(std::move(loan).error());
  }
  auto written = loan->write_host_and_commit(
      [values](const holoscan::TensorOutputWriteGuard& writer) {
        std::copy(values.begin(), values.end(), writer.data_as<double>());
      });
  if (!written) {
    return holoscan::make_unexpected(std::move(written).error());
  }
  return output.emit(
      std::move(*loan),
      holoscan::EmitOptions{.capture_time = capture_time});
}

template <typename Wire>
holoscan::expected<void, holoscan::Error> emit_wire_doubles(
    holoscan::Output<Wire>& output, Wire wire,
    std::span<const double> values, std::span<const int64_t> shape,
    holoscan::TimePoint capture_time) {
  constexpr DLDataType dtype{kDLFloat, 64U, 1U};
  auto loan = output.allocate_tensor(
      holoscan::TensorLoanRequest{.shape = shape, .dtype = dtype});
  if (!loan) {
    return holoscan::make_unexpected(std::move(loan).error());
  }
  auto written = loan->write_host_and_commit(
      [values](const holoscan::TensorOutputWriteGuard& writer) {
        std::copy(values.begin(), values.end(), writer.data_as<double>());
      });
  if (!written) {
    return holoscan::make_unexpected(std::move(written).error());
  }
  return output.emit_tensor(
      std::move(*loan), std::move(wire),
      holoscan::EmitOptions{.capture_time = capture_time});
}

class PerfBeamscanOverrideSourceOp final : public holoscan::Operator<> {
 public:
  PerfBeamscanOverrideSourceOp(int64_t override_at,
                              hr::BeamscanParams initial,
                              hr::BeamscanParams override)
      : override_at_(override_at),
        initial_(std::move(initial)),
        override_(std::move(override)) {}

  void setup(holoscan::OperatorSpec& spec) override {
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
    spec.output(snapshot)
        .produces_tensor(hr::pooled_tensor_output(
            kDevice, 8U, perf::PerfFrameBank::span_a()));
    spec.output(array_geometry)
        .max_emits_per_compute(1U)
        .produces_tensor(
            hr::pooled_tensor_output(kHostMatrix, 2U, kHostPayloadSpan));
    spec.output(az_coordinates)
        .max_emits_per_compute(1U)
        .produces_tensor(
            hr::pooled_tensor_output(kHostVector, 1U, kHostPayloadSpan));
    spec.output(el_coordinates)
        .max_emits_per_compute(1U)
        .produces_tensor(
            hr::pooled_tensor_output(kHostVector, 1U, kHostPayloadSpan));
    spec.output(array_taper)
        .max_emits_per_compute(1U)
        .produces_tensor(
            hr::pooled_tensor_output(kHostVector, 1U, kHostPayloadSpan));
    spec.output(peak_interpolation).max_emits_per_compute(1U);
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = perf::kPerfTick});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    std::shared_ptr<holoscan::Tensor> frame;
    perf::PerfPublishMode mode = perf::PerfPublishMode::kCopy;
    {
      const std::lock_guard lock(perf::PerfFrameBank::mutex);
      if (perf::PerfFrameBank::frames_remaining <= 0 ||
          perf::PerfFrameBank::input_frame == nullptr) {
        return {};
      }
      --perf::PerfFrameBank::frames_remaining;
      frame = perf::PerfFrameBank::input_frame;
      mode = perf::PerfFrameBank::publish_mode;
    }

    const auto& values =
        frame_index_ < override_at_ ? initial_ : override_;
    const holoscan::TimePoint time = context.activation_time();
    const auto packed = hr::pack_array_geometry(values.array_geometry);
    auto geometry_wire = hr::to_wire(values.array_geometry);
    const std::array<int64_t, 2> geometry_shape{
        values.array_geometry.num_elements() + 1, 3};
    auto emitted = emit_wire_doubles(
        array_geometry, std::move(geometry_wire), packed, geometry_shape, time);
    if (!emitted) {
      return emitted;
    }
    emitted = emit_host_vector(az_coordinates, values.az_coordinates, time);
    if (!emitted) {
      return emitted;
    }
    emitted = emit_host_vector(el_coordinates, values.el_coordinates, time);
    if (!emitted) {
      return emitted;
    }
    auto taper_wire = hr::to_wire(values.array_taper);
    const std::array<int64_t, 1> taper_shape{
        static_cast<int64_t>(values.array_taper.coefficients.size())};
    emitted = emit_wire_doubles(
        array_taper, std::move(taper_wire),
        values.array_taper.coefficients, taper_shape, time);
    if (!emitted) {
      return emitted;
    }
    emitted = peak_interpolation.emit(
        values.peak_interpolation,
        holoscan::EmitOptions{.capture_time = time});
    if (!emitted) {
      return emitted;
    }
    {
      const std::lock_guard lock(perf::PerfFrameBank::mutex);
      perf::detail::record_start_locked();
    }
    ++frame_index_;
    return perf::detail::emit_device_frame(snapshot, context, frame, mode);
  }

  holoscan::Output<holoscan::Tensor> snapshot{"snapshot"};
  holoscan::Output<hr::WireArrayGeometry>
      array_geometry{"array_geometry"};
  holoscan::Output<holoscan::Tensor> az_coordinates{"az_coordinates"};
  holoscan::Output<holoscan::Tensor> el_coordinates{"el_coordinates"};
  holoscan::Output<hr::WireWindowSpec> array_taper{"array_taper"};
  holoscan::Output<hr::PeakInterpolation>
      peak_interpolation{"peak_interpolation"};

 private:
  int64_t override_at_{-1};
  hr::BeamscanParams initial_;
  hr::BeamscanParams override_;
  int64_t frame_index_{0};
};

perf::TimingCollector run_beamscan_graph(
    const BeamScenario& scen, const perf::RunPlan& plan) {
  auto snapshot = perf::make_iq_cube_device<PerfPrecision>(
      {scen.detections, scen.channels, scen.snapshots});
  hr::BeamscanParams initial;
  initial.array_geometry =
      make_ula_geometry(scen.channels, 0.5, 1.0);
  initial.az_coordinates = linspace(-0.5, 0.5, scen.num_az);
  initial.el_coordinates = linspace(-0.3, 0.3, scen.num_el);
  initial.peak_interpolation = hr::PeakInterpolation::kNone;
  hr::BeamscanParams override = initial;
  switch (scen.override_kind) {
    case BeamScenario::OverrideKind::kArrayGeometry:
      override.array_geometry =
          make_ula_geometry(scen.channels, 0.45, 1.0);
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
  perf::PerfFrameBank::reset(
      std::move(snapshot), plan.total_frames, &collector);

  holoscan::Graph graph{"beamscan-perf"};
  const auto source = graph.op<PerfBeamscanOverrideSourceOp>(
      "src", plan.override_at, initial, override);
  auto op = graph.op<hr::BeamscanEstimatorOp<PerfPrecision>>(
      "beamscan", extents);
  op.set_params(std::move(initial));
  const auto sink = graph.op<perf::PerfTerminalSinkOp>("aoa_sink");
  const auto spectrum =
      graph.op<perf::PerfNullSinkOp>("spectrum_drain");
  const holoscan::ConnectionOptions q{.queue_depth = 8U};
  graph.add_flow(source->snapshot, op->snapshot, q);
  graph.add_flow(source->array_geometry, op->array_geometry, q);
  graph.add_flow(source->az_coordinates, op->az_coordinates, q);
  graph.add_flow(source->el_coordinates, op->el_coordinates, q);
  graph.add_flow(source->array_taper, op->array_taper, q);
  graph.add_flow(
      source->peak_interpolation, op->peak_interpolation, q);
  graph.add_flow(op->aoa_estimates, sink->in, q);
  graph.add_flow(op->angle_spectrum, spectrum->in, q);

  holoscan::CompileOptions options;
  perf::bind_tensor_out(options, "src", "snapshot");
  perf::bind_tensor_out(options, "beamscan", "aoa_estimates");
  perf::bind_tensor_out(options, "beamscan", "angle_spectrum");
  graph.set_default_clock(
      graph.add_clock<holoscan::RealtimeClock>("clock"));
  const holoscan::ExecutionPlan compiled =
      holoscan::compile(graph, std::move(options));
  if (!compiled.ok()) {
    throw std::runtime_error(
        "beamscan-perf compile failed: " +
        std::string(compiled.json()));
  }
  holoscan::RunSession session = holoscan::run_async(compiled);
  perf::detail::wait_for_frames(session, plan.total_frames);
  session.request_stop();
  session.wait();
  if (!session.diagnostics().empty()) {
    std::string message = "beamscan-perf run diagnostics:";
    for (const auto& diagnostic : session.diagnostics()) {
      message += " ";
      message += std::string(diagnostic.token);
    }
    throw std::runtime_error(message);
  }
  if (collector.ends.size() <
      static_cast<std::size_t>(plan.total_frames)) {
    throw std::runtime_error(perf::detail::stall_message(
        "beamscan-perf", collector.ends.size(), plan.total_frames));
  }
  return collector;
}

void fill_throughput(perf::ResultRow* row,
                     const BeamScenario& scen,
                     double median_ms) {
  if (median_ms <= 0.0) {
    return;
  }
  const double sec = median_ms / 1000.0;
  const double n = static_cast<double>(scen.detections);
  const double beams =
      static_cast<double>(scen.num_az) *
      static_cast<double>(scen.num_el);
  const double ch = static_cast<double>(scen.channels);
  row->cells_per_second = n / sec;
  row->slices_per_second = n * beams / sec;

  const double csz =
      2.0 * static_cast<double>(sizeof(PerfPrecision));
  const double snap_bytes = n * ch * static_cast<double>(scen.snapshots) * csz;
  const double weight_bytes = beams * ch * csz;
  const double spec_bytes =
      n * beams *
      static_cast<double>(sizeof(PerfPrecision));
  const double aoa_bytes =
      n * 2.0 *
      static_cast<double>(sizeof(PerfPrecision));
  row->estimated_effective_gbps =
      (snap_bytes + weight_bytes + spec_bytes +
       aoa_bytes) /
      sec / 1e9;
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
  return row;
}

BeamScenario make_scenario(const SizeSpec& s,
                           BeamScenario::OverrideKind kind) {
  BeamScenario sc;
  sc.size_id = s.id;
  sc.channels = s.channels;
  sc.detections = s.detections;
  sc.snapshots = s.snapshots;
  sc.num_az = s.num_az;
  sc.num_el = s.num_el;
  sc.override_kind = kind;
  std::ostringstream id;
  id << "beamscan_" << s.id << "_c" << s.channels
     << "_n" << s.detections << "_g" << s.num_az
     << "x" << s.num_el << "_s" << s.snapshots << "_"
     << override_name(kind);
  sc.id = id.str();
  return sc;
}

void append_override_scenarios(std::vector<BeamScenario>* out,
                               const SizeSpec& size) {
  for (const auto kind :
       {BeamScenario::OverrideKind::kArrayGeometry,
        BeamScenario::OverrideKind::kAzAngles,
        BeamScenario::OverrideKind::kElAngles,
        BeamScenario::OverrideKind::kArrayTaper,
        BeamScenario::OverrideKind::kPeakInterpolation}) {
    out->push_back(make_scenario(size, kind));
  }
}

struct BeamTraits : perf::PerfDriverDefaults {
  using Scenario = BeamScenario;

  static const char* tool_name() {
    return "perf_beamscan_estimator";
  }
  static const char* tool_description() {
    return "BeamscanEstimatorOp performance "
           "characterization (HSDK 5 Graph)";
  }
  static const char* sizes_all_help() {
    return "Default size sweep with all live overrides";
  }
  static const char* sizes_subset_help() {
    return "Size sweep across an explicit subset";
  }
  static const char* full_help() {
    return "Full size ladder with all live overrides";
  }

  static std::vector<perf::SizeDef> size_ladder_defs() {
    std::vector<perf::SizeDef> out;
    for (const auto& s : size_ladder()) {
      perf::SizeDef def;
      def.id = s.id;
      def.shape = {s.num_az, s.num_el};
      def.cells = s.num_az * s.num_el;
      def.role = s.role;
      out.push_back(def);
    }
    return out;
  }

  static std::vector<BeamScenario> build_default_matrix() {
    std::vector<BeamScenario> out;
    for (const char* sid : {"xs", "sm", "md"}) {
      append_override_scenarios(&out, must_find_size(sid));
    }
    return out;
  }

  static std::vector<BeamScenario> build_size_sweep(
      const std::vector<std::string>& sizes) {
    std::vector<BeamScenario> out;
    for (const auto& sid : sizes) {
      const SizeSpec* s = find_size(sid);
      if (s == nullptr) {
        std::cerr << "warning: unknown size id '"
                  << sid << "', skipping\n";
        continue;
      }
      append_override_scenarios(&out, *s);
    }
    return out;
  }

  static std::vector<BeamScenario> build_full_matrix() {
    std::vector<BeamScenario> out;
    for (const auto& s : size_ladder()) {
      append_override_scenarios(&out, s);
    }
    return out;
  }

  static void run_scenario(const BeamScenario& scen,
                           const perf::RunSettings& rs,
                           perf::PerfReport* report) {
    const perf::RunPlan plan =
        perf::make_run_plan(rs, /*has_override=*/true);
    const auto samples = perf::collect_scenario_samples(
        rs, plan,
        [&](int /*rep*/, const perf::RunPlan& p) {
          return run_beamscan_graph(scen, p);
        });

    perf::append_setup_row(
        base_row(scen), samples, report);

    perf::ResultRow row = base_row(scen);
    perf::fill_compute_row(&row, samples);
    fill_throughput(&row, scen, row.latency.median_ms);
    row.notes =
        "cold_ms is first-frame latency; median/p90 "
        "are post-warmup steady state; override_ms "
        "isolates " + std::string(override_name(scen.override_kind)) +
        "; cells/s is detections/s";
    report->results.push_back(row);
  }

  static void process_warmup(const perf::RunSettings&) {
    perf::run_process_warmup([] {
      perf::PerfReport discarded;
      perf::RunSettings warm;
      warm.warmup_frames = 0;
      warm.measured_frames = 2;
      warm.repeats = 1;
      run_scenario(
          make_scenario(
              must_find_size("xs"),
              BeamScenario::OverrideKind::kAzAngles),
          warm, &discarded);
    });
  }
};

}  // namespace

int main(int argc, char** argv) {
  return holoscan::radar::perf::perf_main<BeamTraits>(
      argc, argv);
}
