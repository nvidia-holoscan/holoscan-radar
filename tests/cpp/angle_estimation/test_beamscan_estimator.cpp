// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// BeamscanEstimatorOp contract tests (HSDK 5 Graph harness): known-DOA
// recovery, scan-grid / peak-interp Params coverage, optional is_valid,
// float/double agreement, and error paths.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "angle_estimation/beamscan_estimator_op/beamscan_estimator_op.hpp"
#include "angle_estimation/beamscan_estimator_op/is_valid_mask.hpp"
#include "radar_common/descriptors/angle_convention.hpp"
#include "radar_common/descriptors/array_geometry.hpp"
#include "radar_common/descriptors/enums.hpp"
#include "radar_common/dsp/angle_convention.hpp"
#include "radar_common/math/constants.hpp"
#include "radar_common/sdk/tensor_io.hpp"
#include "radar_common/tensor/tensor_utils.hpp"
#include "radar_schemas/array_geometry_bridge.hpp"
#include "radar_schemas/window_spec_bridge.hpp"
#include "utils/sdk_harness.hpp"
#include "utils/tensor_fixtures.hpp"

namespace {

using holoscan::radar::AngleConvention;
using holoscan::radar::AngleConventionPreset;
using holoscan::radar::ArrayGeometry;
using holoscan::radar::BeamscanEstimatorOp;
using holoscan::radar::BeamscanParams;
using holoscan::radar::PeakInterpolation;
using holoscan::radar::WindowSpec;
using holoscan::radar::WindowType;
using holoscan::radar::complex_dtype;
using holoscan::radar::detail::check_is_valid_mask;
using holoscan::radar::kPi;
using holoscan::radar::testing::DualIOGraphRunResult;
using holoscan::radar::testing::GraphRunResult;
using holoscan::radar::testing::HostTensor;
using holoscan::radar::testing::run_dual_io_operator_graph_result;
using holoscan::radar::testing::run_single_io_operator_graph_result;
using holoscan::radar::testing::run_single_in_dual_out_operator_graph_result;
using holoscan::radar::testing::to_device_bool_tensor;
using holoscan::radar::testing::to_device_tensor;
using holoscan::radar::testing::to_pageable_host_tensor;

ArrayGeometry make_ula_geometry(int64_t num_elements, double spacing,
                                double wavelength) {
  ArrayGeometry geom;
  geom.wavelength = wavelength;
  geom.geometry_type = holoscan::radar::ArrayGeometryType::kUla;
  geom.element_positions.reserve(static_cast<size_t>(num_elements * 3));
  for (int64_t m = 0; m < num_elements; ++m) {
    geom.element_positions.push_back(0.0);
    geom.element_positions.push_back(static_cast<double>(m) * spacing);
    geom.element_positions.push_back(0.0);
  }
  return geom;
}

ArrayGeometry make_planar_ura_geometry(int64_t ny, int64_t nz, double spacing,
                                       double wavelength) {
  ArrayGeometry geom;
  geom.wavelength = wavelength;
  geom.geometry_type = holoscan::radar::ArrayGeometryType::kUra;
  geom.element_positions.reserve(static_cast<size_t>(ny * nz * 3));
  for (int64_t iz = 0; iz < nz; ++iz) {
    for (int64_t iy = 0; iy < ny; ++iy) {
      geom.element_positions.push_back(0.0);
      geom.element_positions.push_back(static_cast<double>(iy) * spacing);
      geom.element_positions.push_back(static_cast<double>(iz) * spacing);
    }
  }
  return geom;
}

std::vector<std::complex<double>> make_steering_snapshot(
    const ArrayGeometry& geom, const AngleConvention& conv, double az,
    double el) {
  const auto dir = holoscan::radar::boundary_to_canonical(conv, az, el);
  const double k = 2.0 * kPi / geom.wavelength;
  std::vector<std::complex<double>> snap(
      static_cast<size_t>(geom.num_elements()));
  for (int64_t m = 0; m < geom.num_elements(); ++m) {
    const size_t base = static_cast<size_t>(m * 3);
    const double phase = k * (geom.element_positions[base] * dir.x +
                              geom.element_positions[base + 1] * dir.y +
                              geom.element_positions[base + 2] * dir.z);
    snap[static_cast<size_t>(m)] = {std::cos(phase), std::sin(phase)};
  }
  return snap;
}

std::vector<double> linspace(double start, double stop, int64_t n) {
  std::vector<double> out(static_cast<size_t>(n));
  if (n == 1) {
    out[0] = start;
    return out;
  }
  const double step = (stop - start) / static_cast<double>(n - 1);
  for (int64_t i = 0; i < n; ++i) {
    out[static_cast<size_t>(i)] = start + step * static_cast<double>(i);
  }
  return out;
}

template <typename T>
std::shared_ptr<holoscan::Tensor> snapshot_to_device(
    const std::vector<std::complex<double>>& host,
    const std::vector<int64_t>& shape) {
  std::vector<std::complex<T>> cast(host.size());
  for (size_t i = 0; i < host.size(); ++i) {
    cast[i] = {static_cast<T>(host[i].real()), static_cast<T>(host[i].imag())};
  }
  return to_device_tensor(cast, shape, complex_dtype<T>());
}

double spectrum_max(const HostTensor& spec) {
  const double* values = spec.as<double>();
  const int64_t n = spec.size();
  double best = 0.0;
  for (int64_t i = 0; i < n; ++i) {
    best = std::max(best, values[static_cast<size_t>(i)]);
  }
  return best;
}

struct BeamscanOutputs {
  HostTensor aoa;
  HostTensor spectrum;
};

template <typename Source, typename OpRef, typename Sink0, typename Sink1>
void wire_beamscan(holoscan::Graph& graph, const Source& source, OpRef& op,
                   const Sink0& sink0, const Sink1& sink1) {
  const holoscan::ConnectionOptions q{.queue_depth = 4U};
  graph.add_flow(source->out, op->snapshot, q);
  graph.add_flow(op->aoa_estimates, sink0->in, q);
  graph.add_flow(op->angle_spectrum, sink1->in, q);
}

// Dual-input wiring for optional is_valid (DualTensorSourceOp out0/out1).
template <typename Source, typename OpRef, typename Sink0, typename Sink1>
void wire_beamscan_with_is_valid(holoscan::Graph& graph, const Source& source,
                                 OpRef& op, const Sink0& sink0,
                                 const Sink1& sink1) {
  const holoscan::ConnectionOptions q{.queue_depth = 4U};
  // Publish is_valid (out0) before snapshot (out1), matching production
  // causal order for the Exact optional sideband.
  graph.add_flow(source->out0, op->is_valid, q);
  graph.add_flow(source->out1, op->snapshot, q);
  graph.add_flow(op->aoa_estimates, sink0->in, q);
  graph.add_flow(op->angle_spectrum, sink1->in, q);
}

BeamscanParams make_params(ArrayGeometry geom, AngleConvention conv,
                           std::vector<double> az, std::vector<double> el,
                           PeakInterpolation peak = PeakInterpolation::kNone,
                           WindowSpec taper = {}) {
  BeamscanParams params{};
  params.array_geometry = std::move(geom);
  params.angle_convention = std::move(conv);
  params.az_angles = std::move(az);
  params.el_angles = std::move(el);
  params.peak_interpolation = peak;
  params.array_taper = std::move(taper);
  return params;
}

template <typename T>
DualIOGraphRunResult run_beamscan_result(
    std::shared_ptr<holoscan::Tensor> snapshot, BeamscanParams params) {
  const holoscan::radar::MaxTensorExtents sized =
      holoscan::radar::testing::extents_for_scan_grid(
          params.az_angles.size(), params.el_angles.size());
  return run_single_in_dual_out_operator_graph_result<BeamscanEstimatorOp<T>>(
      {std::move(snapshot)}, std::move(params), "aoa_estimates",
      "angle_spectrum",
      [](holoscan::Graph& graph, const auto& source, auto& op,
         const auto& sink0, const auto& sink1) {
        wire_beamscan(graph, source, op, sink0, sink1);
      },
      {}, sized);
}

template <typename T>
BeamscanOutputs run_beamscan(std::shared_ptr<holoscan::Tensor> snapshot,
                             BeamscanParams params) {
  DualIOGraphRunResult result =
      run_beamscan_result<T>(std::move(snapshot), std::move(params));
  if (!result.plan_ok) {
    throw std::runtime_error(std::string("compile failed: ") +
                             result.plan_json);
  }
  if (!result.diagnostics.empty()) {
    std::string msg = "run diagnostics:";
    for (const auto& d : result.diagnostics) {
      msg += " ";
      msg += d;
    }
    throw std::runtime_error(msg);
  }
  if (result.frames0.size() != 1U || result.frames1.size() != 1U) {
    throw std::runtime_error(
        "BeamscanEstimatorOp test: expected one output frame pair, got " +
        std::to_string(result.frames0.size()) + "/" +
        std::to_string(result.frames1.size()));
  }
  return {std::move(result.frames0.front()),
          std::move(result.frames1.front())};
}

template <typename T>
DualIOGraphRunResult run_beamscan_with_mask_result(
    std::shared_ptr<holoscan::Tensor> snapshot,
    std::shared_ptr<holoscan::Tensor> is_valid, BeamscanParams params) {
  return run_dual_io_operator_graph_result<BeamscanEstimatorOp<T>>(
      {std::move(is_valid)}, {std::move(snapshot)}, std::move(params),
      "aoa_estimates", "angle_spectrum",
      [](holoscan::Graph& graph, const auto& source, auto& op,
         const auto& sink0, const auto& sink1) {
        wire_beamscan_with_is_valid(graph, source, op, sink0, sink1);
      });
}

struct LiveBeamscanTick {
  std::shared_ptr<holoscan::Tensor> snapshot;
  std::optional<ArrayGeometry> geometry;
  std::optional<std::vector<double>> az;
  std::optional<std::vector<double>> el;
  std::optional<WindowSpec> taper;
  bool malformed_geometry{false};
  std::optional<PeakInterpolation> peak_interpolation;
};

struct LiveBeamscanBank {
  static void reset(std::vector<LiveBeamscanTick> new_ticks) {
    ticks = std::move(new_ticks);
    index = 0;
  }
  static inline std::vector<LiveBeamscanTick> ticks;
  static inline std::size_t index{0};
};

holoscan::expected<void, holoscan::Error> emit_beamscan_host_vector(
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
holoscan::expected<void, holoscan::Error> emit_beamscan_wire_doubles(
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

class LiveBeamscanSourceOp final : public holoscan::Operator<> {
 public:
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
    spec.output(snapshot)
        .produces_tensor(holoscan::radar::pooled_tensor_output(
            kDevice, 8U, 256ULL * 1024ULL * 1024ULL));
    spec.output(array_geometry)
        .max_emits_per_compute(1U)
        .produces_tensor(holoscan::radar::pooled_tensor_output(
            kHostMatrix, 2U, 4096U * sizeof(double)));
    for (auto* output : {&az_angles, &el_angles}) {
      spec.output(*output)
          .max_emits_per_compute(1U)
          .produces_tensor(holoscan::radar::pooled_tensor_output(
              kHostVector, 1U, 4096U * sizeof(double)));
    }
    spec.output(array_taper)
        .max_emits_per_compute(1U)
        .produces_tensor(holoscan::radar::pooled_tensor_output(
            kHostVector, 1U, 4096U * sizeof(double)));
    spec.output(peak_interpolation).max_emits_per_compute(1U);
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{
        .period = holoscan::radar::testing::detail::kHarnessTick});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    if (LiveBeamscanBank::index >= LiveBeamscanBank::ticks.size()) {
      return {};
    }
    const LiveBeamscanTick& tick =
        LiveBeamscanBank::ticks[LiveBeamscanBank::index++];
    const holoscan::TimePoint time = context.activation_time();
    if (tick.geometry.has_value()) {
      auto packed = holoscan::radar::pack_array_geometry(*tick.geometry);
      if (tick.malformed_geometry) {
        packed[packed.size() - 3U] = 0.0;
      }
      auto wire = holoscan::radar::to_wire(*tick.geometry);
      const std::array<int64_t, 2> shape{
          tick.geometry->num_elements() + 1, 3};
      auto emitted = emit_beamscan_wire_doubles(
          array_geometry, std::move(wire), packed, shape, time);
      if (!emitted) {
        return emitted;
      }
    }
    if (tick.az.has_value()) {
      auto emitted =
          emit_beamscan_host_vector(az_angles, *tick.az, time);
      if (!emitted) {
        return emitted;
      }
    }
    if (tick.el.has_value()) {
      auto emitted =
          emit_beamscan_host_vector(el_angles, *tick.el, time);
      if (!emitted) {
        return emitted;
      }
    }
    if (tick.taper.has_value()) {
      auto wire = holoscan::radar::to_wire(*tick.taper);
      const std::array<int64_t, 1> shape{
          static_cast<int64_t>(tick.taper->coefficients.size())};
      auto emitted = emit_beamscan_wire_doubles(
          array_taper, std::move(wire), tick.taper->coefficients, shape, time);
      if (!emitted) {
        return emitted;
      }
    }
    if (tick.peak_interpolation.has_value()) {
      auto emitted = peak_interpolation.emit(
          *tick.peak_interpolation,
          holoscan::EmitOptions{.capture_time = time});
      if (!emitted) {
        return emitted;
      }
    }
    return holoscan::radar::testing::detail::emit_bank_frame(
        snapshot, context, tick.snapshot, "LiveBeamscanSourceOp");
  }

  holoscan::Output<holoscan::Tensor> snapshot{"snapshot"};
  holoscan::Output<holoscan::radar::WireArrayGeometry>
      array_geometry{"array_geometry"};
  holoscan::Output<holoscan::Tensor> az_angles{"az_angles"};
  holoscan::Output<holoscan::Tensor> el_angles{"el_angles"};
  holoscan::Output<holoscan::radar::WireWindowSpec>
      array_taper{"array_taper"};
  holoscan::Output<PeakInterpolation>
      peak_interpolation{"peak_interpolation"};
};

DualIOGraphRunResult run_live_beamscan_graph(
    std::vector<LiveBeamscanTick> ticks, BeamscanParams params,
    holoscan::radar::MaxTensorExtents extents =
        holoscan::radar::testing::default_test_extents()) {
  DualIOGraphRunResult result;
  holoscan::radar::testing::detail::FrameBank::reset();
  holoscan::radar::testing::detail::FrameBankB::reset();
  LiveBeamscanBank::reset(std::move(ticks));
  const std::size_t expected = LiveBeamscanBank::ticks.size();
  holoscan::Graph graph{"live-beamscan-overrides"};
  const auto source = graph.op<LiveBeamscanSourceOp>("source");
  auto op = holoscan::radar::testing::emplace_harness_op<
      BeamscanEstimatorOp<double>>(graph, extents);
  op.set_params(std::move(params));
  const auto sink0 =
      graph.op<holoscan::radar::testing::detail::TensorCaptureOp>("sink0");
  const auto sink1 =
      graph.op<holoscan::radar::testing::detail::TensorCaptureBOp>("sink1");
  const holoscan::ConnectionOptions q{.queue_depth = 8U};
  graph.add_flow(source->snapshot, op->snapshot, q);
  graph.add_flow(source->array_geometry, op->array_geometry, q);
  graph.add_flow(source->az_angles, op->az_angles, q);
  graph.add_flow(source->el_angles, op->el_angles, q);
  graph.add_flow(source->array_taper, op->array_taper, q);
  graph.add_flow(source->peak_interpolation, op->peak_interpolation, q);
  graph.add_flow(op->aoa_estimates, sink0->in, q);
  graph.add_flow(op->angle_spectrum, sink1->in, q);
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));
  holoscan::CompileOptions options;
  options.deployment.bind_tensor_output_device(
      holoscan::TensorOutputDevicePlacement{
          .operator_path = "source",
          .output_port = "snapshot",
          .device = holoscan::DeviceId{0}});
  holoscan::radar::testing::detail::bind_op_device_output(
      options, "aoa_estimates");
  holoscan::radar::testing::detail::bind_op_device_output(
      options, "angle_spectrum");
  try {
    const holoscan::ExecutionPlan plan =
        holoscan::compile(graph, std::move(options));
    result.plan_ok = plan.ok();
    result.plan_json = plan.json();
    if (!plan.ok()) {
      return result;
    }
    holoscan::RunSession session = holoscan::run_async(plan);
    const auto deadline = std::chrono::steady_clock::now() +
                          holoscan::radar::testing::detail::kHarnessDeadline;
    while (session.diagnostics().empty() &&
           std::chrono::steady_clock::now() < deadline) {
      std::size_t count0 = 0;
      std::size_t count1 = 0;
      {
        const std::lock_guard lock(
            holoscan::radar::testing::detail::FrameBank::mutex);
        count0 =
            holoscan::radar::testing::detail::FrameBank::captured.size();
      }
      {
        const std::lock_guard lock(
            holoscan::radar::testing::detail::FrameBankB::mutex);
        count1 =
            holoscan::radar::testing::detail::FrameBankB::captured.size();
      }
      if (count0 >= expected && count1 >= expected) {
        break;
      }
      std::this_thread::sleep_for(
          holoscan::radar::testing::detail::kHarnessPoll);
    }
    session.request_stop();
    session.wait();
    result.termination = session.termination();
    for (const auto& diagnostic : session.diagnostics()) {
      result.diagnostics.emplace_back(diagnostic.token);
    }
  } catch (const holoscan::RuntimeError& error) {
    result.plan_ok = false;
    result.plan_json = std::string("RuntimeError: ") + error.what();
  }
  {
    const std::lock_guard lock(
        holoscan::radar::testing::detail::FrameBank::mutex);
    result.frames0 = std::move(
        holoscan::radar::testing::detail::FrameBank::captured);
  }
  {
    const std::lock_guard lock(
        holoscan::radar::testing::detail::FrameBankB::mutex);
    result.frames1 = std::move(
        holoscan::radar::testing::detail::FrameBankB::captured);
  }
  return result;
}

TEST(BeamscanEstimatorOpTest, DefaultsToRadarBoresightConvention) {
  EXPECT_EQ(BeamscanParams{}.angle_convention.preset,
            AngleConventionPreset::kRadarBoresight);
}

TEST(BeamscanEstimatorOpTest, RecoversKnownDoa2D) {
  constexpr int64_t kNy = 4;
  constexpr int64_t kNz = 4;
  constexpr double kTruthAz = 0.12;
  constexpr double kTruthEl = 0.05;

  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto geom = make_planar_ura_geometry(kNy, kNz, 0.5, 1.0);
  const auto snap_host =
      make_steering_snapshot(geom, conv, kTruthAz, kTruthEl);
  auto snap_t = to_device_tensor(snap_host, {1, geom.num_elements()},
                                 complex_dtype<double>());

  const auto out = run_beamscan<double>(
      snap_t,
      make_params(geom, conv, linspace(-0.5, 0.5, 101), linspace(-0.3, 0.3, 61),
                  PeakInterpolation::kParabolic));

  ASSERT_EQ(out.aoa.shape, (std::vector<int64_t>{1, 2}));
  const double* aoa = out.aoa.as<double>();
  EXPECT_NEAR(aoa[0], kTruthAz, 0.01);
  EXPECT_NEAR(aoa[1], kTruthEl, 0.01);
}

TEST(BeamscanEstimatorOpTest, Accepts3DSnapshotWithOneLook) {
  constexpr int64_t kN = 4;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto snap_host = make_steering_snapshot(geom, conv, 0.0, 0.0);
  auto snap_t =
      to_device_tensor(snap_host, {1, kN, 1}, complex_dtype<double>());

  const auto out = run_beamscan<double>(
      snap_t, make_params(geom, conv, linspace(-0.2, 0.2, 21), {0.0}));
  EXPECT_EQ(out.spectrum.shape[0], 1);
}

TEST(BeamscanEstimatorOpTest, RejectsMultiSnapshotInput) {
  constexpr int64_t kN = 4;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  std::vector<std::complex<double>> host(static_cast<size_t>(kN * 3),
                                         {1.0, 0.0});
  auto snap_t = to_device_tensor(host, {1, kN, 3}, complex_dtype<double>());
  const DualIOGraphRunResult result = run_beamscan_result<double>(
      snap_t, make_params(geom, AngleConvention{}, {0.0}, {0.0}));
  EXPECT_TRUE(result.plan_ok);
  EXPECT_TRUE(result.frames0.empty());
  EXPECT_FALSE(result.diagnostics.empty());
}

TEST(BeamscanEstimatorOpTest, DirectionCosineMasksInvisibleRegion) {
  constexpr int64_t kN = 4;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kDirectionCosine;
  const auto snap_host = make_steering_snapshot(geom, conv, 0.0, 0.0);
  auto snap_t = to_device_tensor(snap_host, {1, kN}, complex_dtype<double>());

  const auto out = run_beamscan<double>(
      snap_t, make_params(geom, conv, linspace(-1.0, 1.0, 5),
                          linspace(-1.0, 1.0, 5)));

  const double* spec = out.spectrum.as<double>();
  EXPECT_NEAR(spec[0], 0.0, 1e-6);
}

TEST(BeamscanEstimatorOpTest, RejectsInvalidScanCoordinates) {
  constexpr int64_t kN = 4;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  const std::vector<std::complex<double>> host(static_cast<size_t>(kN),
                                               {1.0, 0.0});
  AngleConvention direction_cosine;
  direction_cosine.preset = AngleConventionPreset::kDirectionCosine;
  auto out_of_range = run_beamscan_result<double>(
      to_device_tensor(host, {1, kN}, complex_dtype<double>()),
      make_params(geom, direction_cosine, {0.0}, {-1.01}));
  EXPECT_FALSE(out_of_range.diagnostics.empty());

  auto non_finite = run_beamscan_result<double>(
      to_device_tensor(host, {1, kN}, complex_dtype<double>()),
      make_params(geom, AngleConvention{},
                  {std::numeric_limits<double>::quiet_NaN()}, {0.0}));
  EXPECT_FALSE(non_finite.diagnostics.empty());
}

TEST(BeamscanEstimatorOpTest, FloatInstantiationSmoke) {
  constexpr int64_t kN = 4;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto snap_host = make_steering_snapshot(geom, conv, 0.05, 0.0);
  auto snap_t = snapshot_to_device<float>(snap_host, {1, kN});

  const auto out = run_beamscan<float>(
      snap_t, make_params(geom, conv, linspace(-0.2, 0.2, 21), {0.0}));
  ASSERT_EQ(out.aoa.shape, (std::vector<int64_t>{1, 2}));
  const float* aoa = out.aoa.as<float>();
  EXPECT_NEAR(aoa[0], 0.05f, 0.02f);
}

TEST(BeamscanEstimatorOpTest, RejectsChannelMismatch) {
  constexpr int64_t kN = 4;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  std::vector<std::complex<double>> host(static_cast<size_t>(kN - 1),
                                         {1.0, 0.0});
  auto snap_t = to_device_tensor(host, {1, kN - 1}, complex_dtype<double>());
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const DualIOGraphRunResult result = run_beamscan_result<double>(
      snap_t, make_params(geom, conv, {0.0}, {0.0}));
  EXPECT_TRUE(result.plan_ok);
  EXPECT_TRUE(result.frames0.empty());
  EXPECT_FALSE(result.diagnostics.empty());
}

TEST(BeamscanEstimatorOpTest, RejectsEmptyScanGrid) {
  constexpr int64_t kN = 4;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  std::vector<std::complex<double>> host(static_cast<size_t>(kN), {1.0, 0.0});
  auto snap_t = to_device_tensor(host, {1, kN}, complex_dtype<double>());
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const DualIOGraphRunResult result = run_beamscan_result<double>(
      snap_t, make_params(geom, conv, {}, {0.0}));
  EXPECT_TRUE(result.plan_ok);
  EXPECT_TRUE(result.frames0.empty());
  EXPECT_FALSE(result.diagnostics.empty());
}

TEST(BeamscanEstimatorOpTest, PeakInterpolationConfigurableFromParams) {
  // Authoring Params must select peak_interpolation (live Graph override
  // deferred — HSDK5_PORT §1).
  constexpr int64_t kN = 8;
  constexpr double kTruthAz = 0.17;

  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  const auto snap_host = make_steering_snapshot(geom, conv, kTruthAz, 0.0);
  auto snap_none =
      to_device_tensor(snap_host, {1, kN}, complex_dtype<double>());
  auto snap_para =
      to_device_tensor(snap_host, {1, kN}, complex_dtype<double>());

  const auto out_none = run_beamscan<double>(
      snap_none,
      make_params(geom, conv, linspace(-0.2, 0.2, 11), linspace(-0.1, 0.1, 3),
                  PeakInterpolation::kNone));
  const auto out_para = run_beamscan<double>(
      snap_para,
      make_params(geom, conv, linspace(-0.2, 0.2, 11), linspace(-0.1, 0.1, 3),
                  PeakInterpolation::kParabolic));

  const double* none_az = out_none.aoa.as<double>();
  const double* para_az = out_para.aoa.as<double>();
  EXPECT_NEAR(none_az[0], 0.16, 1e-6);
  EXPECT_NEAR(para_az[0], kTruthAz, 0.02);
  EXPECT_GT(std::abs(para_az[0] - none_az[0]), 5e-3);
}

// Truth sits off-grid, so the interpolation mode is readable in the reported
// angle: kNone can only name the peak grid cell (0.16) while kParabolic
// refines toward the truth (0.17).
LiveBeamscanTick make_interp_tick(
    std::shared_ptr<holoscan::Tensor> snapshot,
    std::optional<PeakInterpolation> interpolation) {
  LiveBeamscanTick tick{std::move(snapshot)};
  tick.peak_interpolation = interpolation;
  return tick;
}

TEST(BeamscanEstimatorOpTest, LivePeakInterpolationAppliesForOneTick) {
  constexpr int64_t kN = 8;
  constexpr double kTruthAz = 0.17;

  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  auto snapshot = snapshot_to_device<double>(
      make_steering_snapshot(geom, conv, kTruthAz, 0.0), {1, kN});
  auto params = make_params(geom, conv, linspace(-0.2, 0.2, 11),
                            linspace(-0.1, 0.1, 3), PeakInterpolation::kNone);

  const DualIOGraphRunResult out = run_live_beamscan_graph(
      {make_interp_tick(snapshot, std::nullopt),
       make_interp_tick(snapshot, PeakInterpolation::kParabolic),
       make_interp_tick(snapshot, std::nullopt)},
      std::move(params));

  ASSERT_TRUE(out.plan_ok) << out.plan_json;
  ASSERT_TRUE(out.diagnostics.empty());
  ASSERT_EQ(out.frames0.size(), 3U);
  const double params_az = out.frames0[0].as<double>()[0];
  const double override_az = out.frames0[1].as<double>()[0];
  const double reverted_az = out.frames0[2].as<double>()[0];
  EXPECT_NEAR(params_az, 0.16, 1e-6);
  EXPECT_NEAR(override_az, kTruthAz, 0.02);
  EXPECT_GT(std::abs(override_az - params_az), 5e-3);
  // The third tick publishes nothing, so it is back on Params `none`.
  EXPECT_EQ(reverted_az, params_az);
}

TEST(BeamscanEstimatorOpTest, LiveAzGridAppliesForOneTick) {
  constexpr int64_t kN = 4;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  auto snapshot = snapshot_to_device<double>(
      make_steering_snapshot(geom, AngleConvention{}, 0.0, 0.0),
      {1, kN});
  auto params = make_params(geom, AngleConvention{}, {-0.1, 0.0, 0.1}, {0.0});
  const DualIOGraphRunResult out = run_live_beamscan_graph(
      {{snapshot},
       {snapshot, std::nullopt, linspace(-0.2, 0.2, 5)},
       {snapshot}},
      std::move(params));
  ASSERT_TRUE(out.plan_ok) << out.plan_json;
  ASSERT_TRUE(out.diagnostics.empty());
  ASSERT_EQ(out.frames1.size(), 3U);
  EXPECT_EQ(out.frames1[0].shape[1], 3);
  EXPECT_EQ(out.frames1[1].shape[1], 5);
  EXPECT_EQ(out.frames1[2].shape[1], 3);
}

TEST(BeamscanEstimatorOpTest, LiveElGridAppliesForOneTick) {
  constexpr int64_t kN = 4;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  auto snapshot = snapshot_to_device<double>(
      make_steering_snapshot(geom, AngleConvention{}, 0.0, 0.0),
      {1, kN});
  auto params = make_params(geom, AngleConvention{}, {0.0}, {0.0});
  const DualIOGraphRunResult out = run_live_beamscan_graph(
      {{snapshot},
       {snapshot, std::nullopt, std::nullopt,
        std::vector<double>{-0.1, 0.0, 0.1}},
       {snapshot}},
      std::move(params));
  ASSERT_TRUE(out.plan_ok) << out.plan_json;
  ASSERT_TRUE(out.diagnostics.empty());
  ASSERT_EQ(out.frames1.size(), 3U);
  EXPECT_EQ(out.frames1[0].shape[2], 1);
  EXPECT_EQ(out.frames1[1].shape[2], 3);
  EXPECT_EQ(out.frames1[2].shape[2], 1);
}

TEST(BeamscanEstimatorOpTest, LiveArrayGeometryAppliesForOneTick) {
  constexpr int64_t kN = 4;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  const auto changed = make_ula_geometry(kN, 0.25, 1.0);
  auto snapshot = snapshot_to_device<double>(
      make_steering_snapshot(geom, AngleConvention{}, 0.2, 0.0),
      {1, kN});
  auto params = make_params(
      geom, AngleConvention{}, linspace(-0.4, 0.4, 9), {0.0});
  const DualIOGraphRunResult out = run_live_beamscan_graph(
      {{snapshot}, {snapshot, changed}, {snapshot}}, std::move(params));
  ASSERT_TRUE(out.plan_ok) << out.plan_json;
  ASSERT_TRUE(out.diagnostics.empty());
  ASSERT_EQ(out.frames1.size(), 3U);
  EXPECT_NE(out.frames1[0].data, out.frames1[1].data);
  EXPECT_EQ(out.frames1[0].data, out.frames1[2].data);
}

TEST(BeamscanEstimatorOpTest, LiveArrayTaperAppliesForOneTick) {
  constexpr int64_t kN = 8;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  auto snapshot = snapshot_to_device<double>(
      make_steering_snapshot(geom, AngleConvention{}, 0.15, 0.0),
      {1, kN});
  WindowSpec hann;
  hann.type = WindowType::kHann;
  auto params = make_params(
      geom, AngleConvention{}, linspace(-0.4, 0.4, 17), {0.0});
  const DualIOGraphRunResult out = run_live_beamscan_graph(
      {{snapshot},
       {snapshot, std::nullopt, std::nullopt, std::nullopt, hann},
       {snapshot}},
      std::move(params));
  ASSERT_TRUE(out.plan_ok) << out.plan_json;
  ASSERT_TRUE(out.diagnostics.empty());
  ASSERT_EQ(out.frames1.size(), 3U);
  EXPECT_NE(out.frames1[0].data, out.frames1[1].data);
  EXPECT_EQ(out.frames1[0].data, out.frames1[2].data);
}

TEST(BeamscanEstimatorOpTest, LiveOverridesRejectMalformedAndBounds) {
  constexpr int64_t kN = 4;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  auto snapshot = snapshot_to_device<double>(
      make_steering_snapshot(geom, AngleConvention{}, 0.0, 0.0),
      {1, kN});
  auto params = make_params(geom, AngleConvention{}, {0.0}, {0.0});
  const DualIOGraphRunResult malformed = run_live_beamscan_graph(
      {{snapshot, geom, std::nullopt, std::nullopt, std::nullopt, true}},
      params);
  ASSERT_TRUE(malformed.plan_ok) << malformed.plan_json;
  EXPECT_TRUE(malformed.frames0.empty());
  EXPECT_FALSE(malformed.diagnostics.empty());

  auto extents = holoscan::radar::testing::default_test_extents();
  extents.max_el = 2;
  const DualIOGraphRunResult bounded = run_live_beamscan_graph(
      {{snapshot, std::nullopt, std::nullopt,
        std::vector<double>{-0.1, 0.0, 0.1}}},
      std::move(params), extents);
  ASSERT_TRUE(bounded.plan_ok) << bounded.plan_json;
  EXPECT_TRUE(bounded.frames0.empty());
  EXPECT_FALSE(bounded.diagnostics.empty());
}

TEST(BeamscanEstimatorOpTest, ArrayTaperConfigurableFromParams) {
  constexpr int64_t kN = 8;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto snap_host = make_steering_snapshot(geom, conv, 0.12, 0.0);
  auto snap_rect =
      to_device_tensor(snap_host, {1, kN}, complex_dtype<double>());
  auto snap_hann =
      to_device_tensor(snap_host, {1, kN}, complex_dtype<double>());

  WindowSpec hann;
  hann.type = WindowType::kHann;

  const auto out_rect = run_beamscan<double>(
      snap_rect,
      make_params(geom, conv, linspace(-0.3, 0.3, 31), {0.0}));
  const auto out_hann = run_beamscan<double>(
      snap_hann,
      make_params(geom, conv, linspace(-0.3, 0.3, 31), {0.0},
                  PeakInterpolation::kNone, hann));

  EXPECT_GT(spectrum_max(out_rect.spectrum), 0.0);
  EXPECT_GT(spectrum_max(out_hann.spectrum), 0.0);
  EXPECT_NE(out_rect.spectrum.data, out_hann.spectrum.data);
}

TEST(BeamscanEstimatorOpTest, UraArrayTaperLowersBothAxisSidelobes) {
  constexpr int64_t kNy = 8;
  constexpr int64_t kNz = 8;
  constexpr int64_t kGrid = 201;
  const auto geom = make_planar_ura_geometry(kNy, kNz, 0.5, 1.0);
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto snapshot =
      make_steering_snapshot(geom, conv, 0.0, 0.0);
  const auto angles = linspace(-1.0, 1.0, kGrid);

  const auto sidelobe_ratio = [&](const WindowSpec& taper, bool scan_azimuth) {
    auto snapshot_t =
        to_device_tensor(snapshot, {1, kNy * kNz}, complex_dtype<double>());
    const std::vector<double> az =
        scan_azimuth ? angles : std::vector<double>{0.0};
    const std::vector<double> el =
        scan_azimuth ? std::vector<double>{0.0} : angles;
    const auto output = run_beamscan<double>(
        snapshot_t,
        make_params(geom, conv, az, el, PeakInterpolation::kNone, taper));
    const double* spectrum = output.spectrum.as<double>();
    double peak = 0.0;
    double sidelobe = 0.0;
    for (int64_t i = 0; i < kGrid; ++i) {
      const double power = spectrum[static_cast<size_t>(i)];
      peak = std::max(peak, power);
      if (std::abs(angles[static_cast<size_t>(i)]) > 0.65) {
        sidelobe = std::max(sidelobe, power);
      }
    }
    return sidelobe / peak;
  };

  WindowSpec hann;
  hann.type = WindowType::kHann;
  const double rect_az = sidelobe_ratio(WindowSpec{}, true);
  const double rect_el = sidelobe_ratio(WindowSpec{}, false);
  const double tapered_az = sidelobe_ratio(hann, true);
  const double tapered_el = sidelobe_ratio(hann, false);
  EXPECT_GT(rect_az, 0.0);
  EXPECT_GT(rect_el, 0.0);
  EXPECT_LT(tapered_az, rect_az);
  EXPECT_LT(tapered_el, rect_el);
}

TEST(BeamscanEstimatorOpTest, RecoversKnownDoaAzOnly1DScan) {
  constexpr int64_t kN = 8;
  constexpr double kTruthAz = 0.083;

  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  const auto snap_host = make_steering_snapshot(geom, conv, kTruthAz, 0.0);
  auto snap_none =
      to_device_tensor(snap_host, {1, kN}, complex_dtype<double>());
  auto snap_parabolic =
      to_device_tensor(snap_host, {1, kN}, complex_dtype<double>());

  const auto out_none = run_beamscan<double>(
      snap_none,
      make_params(geom, conv, linspace(-0.3, 0.3, 61), {0.0},
                  PeakInterpolation::kNone));
  const auto out_parabolic = run_beamscan<double>(
      snap_parabolic,
      make_params(geom, conv, linspace(-0.3, 0.3, 61), {0.0},
                  PeakInterpolation::kParabolic));

  ASSERT_EQ(out_parabolic.spectrum.shape,
            (std::vector<int64_t>{1, 61, 1}));
  const double none_az = out_none.aoa.as<double>()[0];
  const double parabolic_az = out_parabolic.aoa.as<double>()[0];
  EXPECT_NEAR(none_az, 0.08, 1e-6);
  EXPECT_NEAR(parabolic_az, kTruthAz, 0.003);
  EXPECT_LT(std::abs(parabolic_az - kTruthAz),
            std::abs(none_az - kTruthAz));
  EXPECT_NEAR(out_parabolic.aoa.as<double>()[1], 0.0, 1e-6);
}

TEST(BeamscanEstimatorOpTest, RecoversKnownDoaPhysicsSpherical) {
  constexpr int64_t kNy = 4;
  constexpr int64_t kNz = 4;
  constexpr double kTruthAz = 0.1;
  constexpr double kHorizonEl = 0.04;
  constexpr double kPolarEl = kPi / 2.0 - kHorizonEl;

  AngleConvention conv;
  conv.preset = AngleConventionPreset::kPhysicsSpherical;
  const auto geom = make_planar_ura_geometry(kNy, kNz, 0.5, 1.0);
  const auto snap_host =
      make_steering_snapshot(geom, conv, kTruthAz, kPolarEl);
  auto snap_t = to_device_tensor(snap_host, {1, geom.num_elements()},
                                 complex_dtype<double>());

  const auto out = run_beamscan<double>(
      snap_t,
      make_params(geom, conv, linspace(-0.5, 0.5, 101), linspace(1.2, 1.7, 61),
                  PeakInterpolation::kParabolic));

  const double* aoa = out.aoa.as<double>();
  EXPECT_NEAR(aoa[0], kTruthAz, 0.01);
  EXPECT_NEAR(aoa[1], kPolarEl, 0.02);
}

TEST(BeamscanEstimatorOpTest, RecoversKnownDoaAerospaceNed) {
  constexpr int64_t kNy = 4;
  constexpr int64_t kNz = 4;
  constexpr double kTruthAz = -0.1;
  constexpr double kTruthEl = -0.03;

  AngleConvention conv;
  conv.preset = AngleConventionPreset::kAerospaceNed;
  const auto geom = make_planar_ura_geometry(kNy, kNz, 0.5, 1.0);
  const auto snap_host =
      make_steering_snapshot(geom, conv, kTruthAz, kTruthEl);
  auto snap_t = to_device_tensor(snap_host, {1, geom.num_elements()},
                                 complex_dtype<double>());

  const auto out = run_beamscan<double>(
      snap_t,
      make_params(geom, conv, linspace(-0.5, 0.5, 101), linspace(-0.3, 0.3, 61),
                  PeakInterpolation::kParabolic));

  const double* aoa = out.aoa.as<double>();
  EXPECT_NEAR(aoa[0], kTruthAz, 0.01);
  EXPECT_NEAR(aoa[1], kTruthEl, 0.01);
}

TEST(BeamscanEstimatorOpTest, FloatDoubleAgreeOnDoa) {
  constexpr int64_t kN = 8;
  constexpr double kTruthAz = 0.09;

  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  const auto snap_host = make_steering_snapshot(geom, conv, kTruthAz, 0.0);
  auto snap_f = snapshot_to_device<float>(snap_host, {1, kN});
  auto snap_d = snapshot_to_device<double>(snap_host, {1, kN});

  BeamscanParams params = make_params(
      geom, conv, linspace(-0.3, 0.3, 51), {0.0},
      PeakInterpolation::kParabolic);

  const auto out_f = run_beamscan<float>(snap_f, params);
  const auto out_d = run_beamscan<double>(snap_d, params);

  const float* aoa_f = out_f.aoa.as<float>();
  const double* aoa_d = out_d.aoa.as<double>();
  EXPECT_NEAR(aoa_f[0], static_cast<float>(aoa_d[0]), 2e-3f);
  EXPECT_NEAR(aoa_f[1], static_cast<float>(aoa_d[1]), 2e-3f);
}

TEST(BeamscanEstimatorOpTest, PeakInterpolationNoneReturnsGridCell) {
  constexpr int64_t kN = 8;
  const double kTruthAz = 0.1;

  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  const auto snap_host = make_steering_snapshot(geom, conv, kTruthAz, 0.0);
  auto snap_t = to_device_tensor(snap_host, {1, kN}, complex_dtype<double>());

  const auto out = run_beamscan<double>(
      snap_t,
      make_params(geom, conv, linspace(-0.2, 0.2, 41), {0.0},
                  PeakInterpolation::kNone));

  const double* aoa = out.aoa.as<double>();
  EXPECT_DOUBLE_EQ(aoa[0], kTruthAz);
  EXPECT_DOUBLE_EQ(aoa[1], 0.0);
}

TEST(BeamscanEstimatorOpTest, EmptyItemsProducesEmptyOutputs) {
  constexpr int64_t kN = 4;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  std::vector<std::complex<double>> host;
  auto snap_t = to_device_tensor(host, {0, kN}, complex_dtype<double>());

  const auto out = run_beamscan<double>(
      snap_t, make_params(geom, conv, {-0.1, 0.0, 0.1}, {0.0}));

  EXPECT_EQ(out.aoa.shape, (std::vector<int64_t>{0, 2}));
  EXPECT_EQ(out.spectrum.shape, (std::vector<int64_t>{0, 3, 1}));
}

TEST(BeamscanEstimatorOpTest, OptionalSpectrumMayBeLeftUnwired) {
  constexpr int64_t kN = 4;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  auto snapshot = snapshot_to_device<double>(
      make_steering_snapshot(geom, AngleConvention{}, 0.0, 0.0), {1, kN});
  auto params = make_params(geom, AngleConvention{}, {-0.1, 0.0, 0.1}, {0.0});
  const auto extents =
      holoscan::radar::testing::extents_for_scan_grid(3, 1);
  const GraphRunResult result =
      run_single_io_operator_graph_result<BeamscanEstimatorOp<double>>(
          {std::move(snapshot)}, std::move(params), "aoa_estimates",
          [](holoscan::Graph& graph, const auto& source, auto& op,
             const auto& sink) {
            const holoscan::ConnectionOptions q{.queue_depth = 4U};
            graph.add_flow(source->out, op->snapshot, q);
            graph.add_flow(op->aoa_estimates, sink->in, q);
          },
          {"angle_spectrum"}, extents);
  ASSERT_TRUE(result.plan_ok) << result.plan_json;
  ASSERT_TRUE(result.diagnostics.empty());
  ASSERT_EQ(result.frames.size(), 1U);
  EXPECT_EQ(result.frames[0].shape, (std::vector<int64_t>{1, 2}));
}

TEST(BeamscanEstimatorOpTest, RejectsInvalidSnapshotRank) {
  constexpr int64_t kN = 4;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  std::vector<std::complex<double>> host(static_cast<size_t>(kN), {1.0, 0.0});
  auto snap_t = to_device_tensor(host, {kN}, complex_dtype<double>());
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const DualIOGraphRunResult result = run_beamscan_result<double>(
      snap_t, make_params(geom, conv, {0.0}, {0.0}));
  EXPECT_TRUE(result.plan_ok);
  EXPECT_TRUE(result.frames0.empty());
  EXPECT_FALSE(result.diagnostics.empty());
}

TEST(BeamscanEstimatorOpTest, RejectsSnapshotDtypeMismatch) {
  constexpr int64_t kN = 4;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  const auto snap_host = make_steering_snapshot(
      geom, AngleConvention{AngleConventionPreset::kRadarBoresight}, 0.0, 0.0);
  auto snap_f = snapshot_to_device<float>(snap_host, {1, kN});
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const DualIOGraphRunResult result = run_beamscan_result<double>(
      snap_f, make_params(geom, conv, {0.0}, {0.0}));
  EXPECT_TRUE(result.plan_ok);
  EXPECT_TRUE(result.frames0.empty());
  EXPECT_FALSE(result.diagnostics.empty());
}

TEST(BeamscanEstimatorOpTest, RejectsItemsBeyondCapturedExtent) {
  constexpr int64_t kN = 4;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  std::vector<std::complex<double>> host(static_cast<size_t>(2 * kN),
                                         {1.0, 0.0});
  auto snapshot =
      to_device_tensor(host, {2, kN}, complex_dtype<double>());
  auto extents = holoscan::radar::testing::extents_for_scan_grid(1, 1);
  extents.max_items = 1;
  const DualIOGraphRunResult result =
      run_single_in_dual_out_operator_graph_result<BeamscanEstimatorOp<double>>(
          {std::move(snapshot)},
          make_params(geom, AngleConvention{}, {0.0}, {0.0}),
          "aoa_estimates", "angle_spectrum",
          [](holoscan::Graph& graph, const auto& source, auto& op,
             const auto& sink0, const auto& sink1) {
            wire_beamscan(graph, source, op, sink0, sink1);
          },
          {}, extents);
  ASSERT_TRUE(result.plan_ok) << result.plan_json;
  EXPECT_TRUE(result.frames0.empty());
  EXPECT_FALSE(result.diagnostics.empty());
}

TEST(BeamscanEstimatorOpTest, SpectrumShapeMatchesScanGrid) {
  constexpr int64_t kN = 4;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto snap_host = make_steering_snapshot(geom, conv, 0.05, 0.0);
  std::vector<std::complex<double>> batch;
  batch.insert(batch.end(), snap_host.begin(), snap_host.end());
  batch.insert(batch.end(), snap_host.begin(), snap_host.end());
  auto snap_t = to_device_tensor(batch, {2, kN}, complex_dtype<double>());

  const auto out = run_beamscan<double>(
      snap_t,
      make_params(geom, conv, linspace(-0.2, 0.2, 17), linspace(-0.1, 0.1, 9)));

  EXPECT_EQ(out.aoa.shape, (std::vector<int64_t>{2, 2}));
  EXPECT_EQ(out.spectrum.shape, (std::vector<int64_t>{2, 17, 9}));
  EXPECT_GT(spectrum_max(out.spectrum), 0.0);
}

TEST(BeamscanEstimatorOpTest, MultipleItemsRecoverIndependentDoa) {
  constexpr int64_t kNy = 4;
  constexpr int64_t kNz = 4;
  constexpr double kAz0 = 0.11;
  constexpr double kEl0 = 0.04;
  constexpr double kAz1 = -0.07;
  constexpr double kEl1 = -0.02;

  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto geom = make_planar_ura_geometry(kNy, kNz, 0.5, 1.0);
  const auto snap0 = make_steering_snapshot(geom, conv, kAz0, kEl0);
  const auto snap1 = make_steering_snapshot(geom, conv, kAz1, kEl1);
  std::vector<std::complex<double>> batch;
  batch.insert(batch.end(), snap0.begin(), snap0.end());
  batch.insert(batch.end(), snap1.begin(), snap1.end());
  auto snap_t = to_device_tensor(batch, {2, geom.num_elements()},
                                 complex_dtype<double>());

  const auto out = run_beamscan<double>(
      snap_t,
      make_params(geom, conv, linspace(-0.4, 0.4, 81), linspace(-0.2, 0.2, 41),
                  PeakInterpolation::kParabolic));

  const double* aoa = out.aoa.as<double>();
  EXPECT_NEAR(aoa[0], kAz0, 0.01);
  EXPECT_NEAR(aoa[1], kEl0, 0.01);
  EXPECT_NEAR(aoa[2], kAz1, 0.01);
  EXPECT_NEAR(aoa[3], kEl1, 0.01);
}

TEST(BeamscanEstimatorOpTest, AcceptsValidIsValidMaskOn3DSnapshot) {
  constexpr int64_t kN = 4;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  auto snapshot = snapshot_to_device<double>(
      make_steering_snapshot(geom, AngleConvention{}, 0.0, 0.0),
      {1, kN, 1});
  auto mask = to_device_bool_tensor({0}, {1, 1});
  const DualIOGraphRunResult result = run_beamscan_with_mask_result<double>(
      std::move(snapshot), std::move(mask),
      make_params(geom, AngleConvention{}, {-0.1, 0.0, 0.1}, {0.0}));
  ASSERT_TRUE(result.plan_ok) << result.plan_json;
  ASSERT_TRUE(result.diagnostics.empty());
  ASSERT_EQ(result.frames0.size(), 1U);
  ASSERT_EQ(result.frames1.size(), 1U);
  EXPECT_TRUE(std::isnan(result.frames0[0].as<double>()[0]));
  EXPECT_TRUE(std::isnan(result.frames0[0].as<double>()[1]));
  for (int64_t i = 0; i < result.frames1[0].size(); ++i) {
    EXPECT_EQ(result.frames1[0].as<double>()[i], 0.0);
  }
}

TEST(BeamscanEstimatorOpTest, RejectsInvalidIsValidMaskOn3DSnapshot) {
  constexpr int64_t kN = 4;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  auto snapshot = snapshot_to_device<double>(
      make_steering_snapshot(geom, AngleConvention{}, 0.0, 0.0),
      {1, kN, 1});
  auto mask = to_device_bool_tensor({1, 1}, {1, 2});
  const DualIOGraphRunResult result = run_beamscan_with_mask_result<double>(
      std::move(snapshot), std::move(mask),
      make_params(geom, AngleConvention{}, {0.0}, {0.0}));
  ASSERT_TRUE(result.plan_ok) << result.plan_json;
  EXPECT_TRUE(result.frames0.empty());
  EXPECT_FALSE(result.diagnostics.empty());
}

TEST(BeamscanEstimatorOpTest, CheckIsValidMaskAcceptsTrueBoolDevice) {
  auto mask = to_device_bool_tensor({1}, {1, 1});
  EXPECT_NO_THROW(check_is_valid_mask(*mask, 1, 1, "test"));
}

TEST(BeamscanEstimatorOpTest, CheckIsValidMaskAcceptsFalseBoolDevice) {
  // Zero slots are legal (unused snapshot); the estimator blanks those items.
  auto mask = to_device_bool_tensor({0}, {1, 1});
  EXPECT_NO_THROW(check_is_valid_mask(*mask, 1, 1, "test"));
}

TEST(BeamscanEstimatorOpTest, CheckIsValidMaskRejectsPageableHost) {
  auto mask =
      to_pageable_host_tensor<uint8_t>({1}, {1, 1}, DLDataType{kDLBool, 8, 1});
  EXPECT_THROW(check_is_valid_mask(*mask, 1, 1, "test"), std::invalid_argument);
}

TEST(BeamscanEstimatorOpTest, CheckIsValidMaskRejectsShapeMismatch) {
  auto mask = to_pageable_host_tensor<uint8_t>(
      {1, 1}, {2, 1}, DLDataType{kDLBool, 8, 1});
  EXPECT_THROW(check_is_valid_mask(*mask, 1, 1, "test"), std::invalid_argument);
}

}  // namespace
