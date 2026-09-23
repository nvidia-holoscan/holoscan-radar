// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// MVDREstimatorOp contract tests (HSDK 5 Graph harness): known-DOA recovery,
// multi-source / eigenvalues / spatial smoothing Params coverage, float/
// double agreement, optional-mask wiring, live overrides, and error paths.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/expected_macros.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "angle_estimation/mvdr_estimator_op/mvdr_estimator_op.hpp"
#include "radar_common/descriptors/angle_convention.hpp"
#include "radar_common/descriptors/array_geometry.hpp"
#include "radar_common/dsp/angle_convention.hpp"
#include "radar_common/math/constants.hpp"
#include "radar_common/sdk/tensor_io.hpp"
#include "radar_common/tensor/tensor_utils.hpp"
#include "radar_schemas/array_geometry_bridge.hpp"
#include "utils/sdk_harness.hpp"
#include "utils/tensor_fixtures.hpp"

namespace {

using holoscan::radar::AngleConvention;
using holoscan::radar::AngleConventionPreset;
using holoscan::radar::ArrayGeometry;
using holoscan::radar::MVDREstimatorOp;
using holoscan::radar::MVDRParams;
using holoscan::radar::SpatialSmoothingOverride;
using holoscan::radar::WireArrayGeometry;
using holoscan::radar::complex_dtype;
using holoscan::radar::kPi;
using holoscan::radar::testing::FiveOutGraphRunResult;
using holoscan::radar::testing::HostTensor;
using holoscan::radar::testing::run_dual_io_operator_graph_result;
using holoscan::radar::testing::run_single_in_five_out_operator_graph_result;
using holoscan::radar::testing::to_device_tensor;

ArrayGeometry make_ula_geometry(int64_t num_elements, double spacing, double wavelength) {
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

ArrayGeometry make_planar_ura_geometry(int64_t ny, int64_t nz, double spacing, double wavelength) {
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

std::vector<std::complex<double>> steering_vector(const ArrayGeometry& geom,
                                                  const AngleConvention& conv, double az,
                                                  double el) {
  const auto dir = holoscan::radar::boundary_to_canonical(conv, az, el);
  const double k = 2.0 * kPi / geom.wavelength;
  std::vector<std::complex<double>> a(static_cast<size_t>(geom.num_elements()));
  for (int64_t m = 0; m < geom.num_elements(); ++m) {
    const size_t base = static_cast<size_t>(m * 3);
    const double phase =
        k * (geom.element_positions[base] * dir.x + geom.element_positions[base + 1] * dir.y +
             geom.element_positions[base + 2] * dir.z);
    a[static_cast<size_t>(m)] = {std::cos(phase), std::sin(phase)};
  }
  return a;
}

std::vector<std::complex<double>> make_block(const ArrayGeometry& geom, const AngleConvention& conv,
                                             double az, double el, int64_t S, double noise,
                                             unsigned seed) {
  const auto a = steering_vector(geom, conv, az, el);
  const int64_t C = geom.num_elements();
  std::vector<std::complex<double>> x(static_cast<size_t>(C * S));
  std::mt19937 rng(seed);
  std::normal_distribution<double> n(0.0, noise);
  for (int64_t s = 0; s < S; ++s) {
    const double gp = 2.0 * kPi * static_cast<double>(s) / static_cast<double>(S);
    const std::complex<double> g{std::cos(gp), std::sin(gp)};
    for (int64_t c = 0; c < C; ++c) {
      std::complex<double> v = a[static_cast<size_t>(c)] * g;
      v += std::complex<double>(n(rng), n(rng));
      x[static_cast<size_t>(c * S + s)] = v;
    }
  }
  return x;
}

std::vector<std::complex<double>> make_block_two(const ArrayGeometry& geom,
                                                 const AngleConvention& conv, double az0,
                                                 double el0, double az1, double el1, int64_t S,
                                                 double noise, unsigned seed) {
  const auto a0 = steering_vector(geom, conv, az0, el0);
  const auto a1 = steering_vector(geom, conv, az1, el1);
  const int64_t C = geom.num_elements();
  std::vector<std::complex<double>> x(static_cast<size_t>(C * S));
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> phase(0.0, 2.0 * kPi);
  std::normal_distribution<double> n(0.0, noise);
  for (int64_t s = 0; s < S; ++s) {
    const double p0 = phase(rng);
    const double p1 = phase(rng);
    const std::complex<double> g0{std::cos(p0), std::sin(p0)};
    const std::complex<double> g1{std::cos(p1), std::sin(p1)};
    for (int64_t c = 0; c < C; ++c) {
      std::complex<double> v = a0[static_cast<size_t>(c)] * g0 + a1[static_cast<size_t>(c)] * g1;
      v += std::complex<double>(n(rng), n(rng));
      x[static_cast<size_t>(c * S + s)] = v;
    }
  }
  return x;
}

std::vector<std::complex<double>> make_block_two_coherent(const ArrayGeometry& geom,
                                                          const AngleConvention& conv, double az0,
                                                          double el0, double az1, double el1,
                                                          int64_t S, double noise, unsigned seed) {
  const auto a0 = steering_vector(geom, conv, az0, el0);
  const auto a1 = steering_vector(geom, conv, az1, el1);
  const int64_t C = geom.num_elements();
  std::vector<std::complex<double>> x(static_cast<size_t>(C * S));
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> phase(0.0, 2.0 * kPi);
  std::normal_distribution<double> n(0.0, noise);
  for (int64_t s = 0; s < S; ++s) {
    const double ps = phase(rng);
    const std::complex<double> g{std::cos(ps), std::sin(ps)};
    for (int64_t c = 0; c < C; ++c) {
      std::complex<double> v = (a0[static_cast<size_t>(c)] + a1[static_cast<size_t>(c)]) * g;
      v += std::complex<double>(n(rng), n(rng));
      x[static_cast<size_t>(c * S + s)] = v;
    }
  }
  return x;
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
std::shared_ptr<holoscan::Tensor> block_to_device(const std::vector<std::complex<double>>& host,
                                                  const std::vector<int64_t>& shape) {
  std::vector<std::complex<T>> cast(host.size());
  for (size_t i = 0; i < host.size(); ++i) {
    cast[i] = {static_cast<T>(host[i].real()), static_cast<T>(host[i].imag())};
  }
  return to_device_tensor(cast, shape, complex_dtype<T>());
}

struct MvdrOutputs {
  HostTensor aoa;
  HostTensor spectrum;
  HostTensor doa_multi;
  HostTensor source_count;
  HostTensor eigenvalues;
};

// cpplint misreads the wrapped template header as an indented declaration
// inside the anonymous namespace; the parameter list cannot fit on one line.
template <typename Source, typename OpRef, typename S0, typename S1, typename S2, typename S3,
          typename S4>  // NOLINT(whitespace/indent_namespace)
void wire_mvdr(holoscan::Graph& graph, const Source& source, OpRef& op, const S0& sink0,
               const S1& sink1, const S2& sink2, const S3& sink3, const S4& sink4) {
  const holoscan::ConnectionOptions q{.queue_depth = 4U};
  graph.add_flow(source->out, op->snapshot, q);
  graph.add_flow(op->aoa_estimates, sink0->in, q);
  graph.add_flow(op->angle_spectrum, sink1->in, q);
  graph.add_flow(op->doa_multi, sink2->in, q);
  graph.add_flow(op->source_count, sink3->in, q);
  graph.add_flow(op->eigenvalues, sink4->in, q);
}

MVDRParams make_params(ArrayGeometry geom, AngleConvention conv, std::vector<double> az,
                       std::vector<double> el, double loading = -1.0, int64_t max_sources = 1,
                       std::vector<int64_t> smoothing = {}, bool forward_backward = true) {
  MVDRParams params{};
  params.array_geometry = std::move(geom);
  params.angle_convention = std::move(conv);
  params.az_coordinates = std::move(az);
  params.el_coordinates = std::move(el);
  params.diagonal_loading = loading;
  params.max_sources = max_sources;
  params.spatial_smoothing = std::move(smoothing);
  params.forward_backward = forward_backward;
  return params;
}

template <typename T>
FiveOutGraphRunResult run_mvdr_result(std::shared_ptr<holoscan::Tensor> snapshot,
                                      MVDRParams params) {
  const holoscan::radar::MaxTensorExtents sized = holoscan::radar::testing::extents_for_scan_grid(
      params.az_coordinates.size(), params.el_coordinates.size());
  return run_single_in_five_out_operator_graph_result<MVDREstimatorOp<T>>(
      {std::move(snapshot)}, std::move(params), "aoa_estimates", "angle_spectrum", "doa_multi",
      "source_count", "eigenvalues",
      [](holoscan::Graph& graph, const auto& source, auto& op, const auto& sink0, const auto& sink1,
         const auto& sink2, const auto& sink3, const auto& sink4) {
        wire_mvdr(graph, source, op, sink0, sink1, sink2, sink3, sink4);
      },
      sized);
}

template <typename T>
MvdrOutputs run_mvdr(std::shared_ptr<holoscan::Tensor> snapshot, MVDRParams params) {
  FiveOutGraphRunResult result = run_mvdr_result<T>(std::move(snapshot), std::move(params));
  if (!result.plan_ok) {
    throw std::runtime_error(std::string("compile failed: ") + result.plan_json);
  }
  if (!result.diagnostics.empty()) {
    std::string msg = "run diagnostics:";
    for (const auto& d : result.diagnostics) {
      msg += " ";
      msg += d;
    }
    throw std::runtime_error(msg);
  }
  if (result.frames0.size() != 1U || result.frames1.size() != 1U || result.frames2.size() != 1U ||
      result.frames3.size() != 1U || result.frames4.size() != 1U) {
    throw std::runtime_error("MVDREstimatorOp test: expected one output frame set");
  }
  return {std::move(result.frames0.front()), std::move(result.frames1.front()),
          std::move(result.frames2.front()), std::move(result.frames3.front()),
          std::move(result.frames4.front())};
}

struct LiveMvdrTick {
  std::shared_ptr<holoscan::Tensor> snapshot;
  std::optional<ArrayGeometry> geometry;
  std::optional<std::vector<double>> az;
  std::optional<std::vector<double>> el;
  std::optional<SpatialSmoothingOverride> smoothing;
  int64_t skew_ns{0};
  std::optional<double> diagonal_loading;
  std::optional<int64_t> max_sources;
};

struct LiveMvdrBank {
  static void reset(std::vector<LiveMvdrTick> new_ticks) {
    const std::lock_guard lock(mutex);
    ticks = std::move(new_ticks);
    index = 0U;
  }
  static inline std::mutex mutex;
  static inline std::vector<LiveMvdrTick> ticks;
  static inline std::size_t index{0U};
};

class LiveMvdrSourceOp final : public holoscan::Operator<> {
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
    constexpr holoscan::TensorRepresentation kHostGeometry{
        .memory_kind = holoscan::MemoryKind::kHost,
        .dtype = DLDataType{kDLFloat, 64U, 1U},
        .rank = 2U,
    };
    spec.output(snapshot).produces_tensor(
        holoscan::radar::pooled_tensor_output(kDevice, 3U, 1024U * 1024U));
    spec.output(geometry).max_emits_per_compute(1U).produces_tensor(
        holoscan::radar::pooled_tensor_output(kHostGeometry, 2U, 65U * 3U * sizeof(double)));
    spec.output(az).max_emits_per_compute(1U).produces_tensor(
        holoscan::radar::pooled_tensor_output(kHostVector, 1U, 65U * sizeof(double)));
    spec.output(el).max_emits_per_compute(1U).produces_tensor(
        holoscan::radar::pooled_tensor_output(kHostVector, 1U, 33U * sizeof(double)));
    spec.output(smoothing);
    spec.output(diagonal_loading);
    spec.output(max_sources);
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = holoscan::radar::testing::detail::kHarnessTick});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> emit_vector(
      holoscan::Output<holoscan::Tensor>& output, const std::vector<double>& values,
      holoscan::TimePoint capture_time) {
    const std::array<int64_t, 1> shape{static_cast<int64_t>(values.size())};
    auto loan = output.allocate_tensor(
        holoscan::TensorLoanRequest{.shape = shape, .dtype = DLDataType{kDLFloat, 64U, 1U}});
    if (!loan) {
      return holoscan::make_unexpected(std::move(loan).error());
    }
    HOLOSCAN_RETURN_IF_ERROR(
        loan->write_host_and_commit([&values](const holoscan::TensorOutputWriteGuard& writer) {
          std::copy(values.begin(), values.end(), writer.data_as<double>());
        }));
    return output.emit(std::move(*loan), holoscan::EmitOptions{.capture_time = capture_time});
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> emit_geometry(
      const ArrayGeometry& value, holoscan::TimePoint capture_time) {
    const std::vector<double> packed = holoscan::radar::pack_array_geometry(value);
    const std::array<int64_t, 2> shape{value.num_elements() + 1, 3};
    auto loan = geometry.allocate_tensor(
        holoscan::TensorLoanRequest{.shape = shape, .dtype = DLDataType{kDLFloat, 64U, 1U}});
    if (!loan) {
      return holoscan::make_unexpected(std::move(loan).error());
    }
    HOLOSCAN_RETURN_IF_ERROR(
        loan->write_host_and_commit([&packed](const holoscan::TensorOutputWriteGuard& writer) {
          std::copy(packed.begin(), packed.end(), writer.data_as<double>());
        }));
    return geometry.emit_tensor(std::move(*loan), holoscan::radar::to_wire(value),
                                holoscan::EmitOptions{.capture_time = capture_time});
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    LiveMvdrTick tick;
    {
      const std::lock_guard lock(LiveMvdrBank::mutex);
      if (LiveMvdrBank::index >= LiveMvdrBank::ticks.size()) {
        return {};
      }
      tick = LiveMvdrBank::ticks[LiveMvdrBank::index++];
    }
    holoscan::TimePoint override_time = context.activation_time();
    override_time.timestamp_ns += tick.skew_ns;
    if (tick.geometry.has_value()) {
      HOLOSCAN_RETURN_IF_ERROR(emit_geometry(*tick.geometry, override_time));
    }
    if (tick.az.has_value()) {
      HOLOSCAN_RETURN_IF_ERROR(emit_vector(az, *tick.az, override_time));
    }
    if (tick.el.has_value()) {
      HOLOSCAN_RETURN_IF_ERROR(emit_vector(el, *tick.el, override_time));
    }
    if (tick.smoothing.has_value()) {
      HOLOSCAN_RETURN_IF_ERROR(
          smoothing.emit(*tick.smoothing, holoscan::EmitOptions{.capture_time = override_time}));
    }
    if (tick.diagonal_loading.has_value()) {
      HOLOSCAN_RETURN_IF_ERROR(diagonal_loading.emit(
          *tick.diagonal_loading, holoscan::EmitOptions{.capture_time = override_time}));
    }
    if (tick.max_sources.has_value()) {
      HOLOSCAN_RETURN_IF_ERROR(max_sources.emit(
          *tick.max_sources, holoscan::EmitOptions{.capture_time = override_time}));
    }
    return holoscan::radar::testing::detail::emit_bank_frame(snapshot, context, tick.snapshot,
                                                             "LiveMvdrSourceOp");
  }

  holoscan::Output<holoscan::Tensor> snapshot{"snapshot"};
  holoscan::Output<WireArrayGeometry> geometry{"array_geometry"};
  holoscan::Output<holoscan::Tensor> az{"az_coordinates"};
  holoscan::Output<holoscan::Tensor> el{"el_coordinates"};
  holoscan::Output<SpatialSmoothingOverride> smoothing{"spatial_smoothing"};
  holoscan::Output<double> diagonal_loading{"diagonal_loading"};
  holoscan::Output<int64_t> max_sources{"max_sources"};
};

// Which override ports the graph wires. The operator declares every entry
// unconditionally, so this is the application's choice alone.
struct LiveMvdrWiring {
  bool geometry{false};
  bool az{false};
  bool el{false};
  bool smoothing{false};
  bool diagonal_loading{false};
  bool max_sources{false};
};

FiveOutGraphRunResult run_live_mvdr(std::vector<LiveMvdrTick> ticks, MVDRParams params,
                                    LiveMvdrWiring wiring,
                                    const holoscan::radar::MaxTensorExtents& extents =
                                        holoscan::radar::testing::default_test_extents()) {
  FiveOutGraphRunResult result;
  holoscan::radar::testing::detail::FrameBank::reset();
  holoscan::radar::testing::detail::FrameBankB::reset();
  holoscan::radar::testing::detail::FrameBankC::reset();
  holoscan::radar::testing::detail::FrameBankD::reset();
  holoscan::radar::testing::detail::FrameBankE::reset();
  const std::size_t expected = ticks.size();
  LiveMvdrBank::reset(std::move(ticks));

  holoscan::Graph graph{"mvdr-live-container-overrides"};
  const auto source = graph.op<LiveMvdrSourceOp>("source");
  auto op = holoscan::radar::testing::emplace_harness_op<MVDREstimatorOp<double>>(graph, extents);
  op.set_params(std::move(params));
  const auto sink0 = graph.op<holoscan::radar::testing::detail::TensorCaptureOp>("sink0");
  const auto sink1 = graph.op<holoscan::radar::testing::detail::TensorCaptureBOp>("sink1");
  const auto sink2 = graph.op<holoscan::radar::testing::detail::TensorCaptureCOp>("sink2");
  const auto sink3 = graph.op<holoscan::radar::testing::detail::TensorCaptureDOp>("sink3");
  const auto sink4 = graph.op<holoscan::radar::testing::detail::TensorCaptureEOp>("sink4");
  const holoscan::ConnectionOptions q{.queue_depth = 8U};
  graph.add_flow(source->snapshot, op->snapshot, q);
  if (wiring.geometry) {
    graph.add_flow(source->geometry, op->array_geometry, q);
  }
  if (wiring.az) {
    graph.add_flow(source->az, op->az_coordinates, q);
  }
  if (wiring.el) {
    graph.add_flow(source->el, op->el_coordinates, q);
  }
  if (wiring.smoothing) {
    graph.add_flow(source->smoothing, op->spatial_smoothing, q);
  }
  if (wiring.diagonal_loading) {
    graph.add_flow(source->diagonal_loading, op->diagonal_loading, q);
  }
  if (wiring.max_sources) {
    graph.add_flow(source->max_sources, op->max_sources, q);
  }
  graph.add_flow(op->aoa_estimates, sink0->in, q);
  graph.add_flow(op->angle_spectrum, sink1->in, q);
  graph.add_flow(op->doa_multi, sink2->in, q);
  graph.add_flow(op->source_count, sink3->in, q);
  graph.add_flow(op->eigenvalues, sink4->in, q);
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

  holoscan::CompileOptions options;
  for (const char* port :
       {"aoa_estimates", "angle_spectrum", "doa_multi", "source_count", "eigenvalues"}) {
    holoscan::radar::testing::detail::bind_op_device_output(options, port);
  }
  options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
      .operator_path = "source", .output_port = "snapshot", .device = holoscan::DeviceId{0}});
  try {
    const holoscan::ExecutionPlan plan = holoscan::compile(graph, std::move(options));
    result.plan_ok = plan.ok();
    result.plan_json = plan.json();
    if (!plan.ok()) {
      return result;
    }
    holoscan::RunSession session = holoscan::run_async(plan);
    const auto deadline =
        std::chrono::steady_clock::now() + holoscan::radar::testing::detail::kHarnessDeadline;
    while (session.diagnostics().empty() && std::chrono::steady_clock::now() < deadline) {
      std::size_t spectrum_count = 0U;
      std::size_t eigenvalue_count = 0U;
      {
        const std::lock_guard lock(holoscan::radar::testing::detail::FrameBankB::mutex);
        spectrum_count = holoscan::radar::testing::detail::FrameBankB::captured.size();
      }
      {
        const std::lock_guard lock(holoscan::radar::testing::detail::FrameBankE::mutex);
        eigenvalue_count = holoscan::radar::testing::detail::FrameBankE::captured.size();
      }
      if (spectrum_count >= expected && eigenvalue_count >= expected) {
        break;
      }
      std::this_thread::sleep_for(holoscan::radar::testing::detail::kHarnessPoll);
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
    const std::lock_guard lock(holoscan::radar::testing::detail::FrameBankB::mutex);
    result.frames1 = std::move(holoscan::radar::testing::detail::FrameBankB::captured);
  }
  {
    const std::lock_guard lock(holoscan::radar::testing::detail::FrameBankC::mutex);
    result.frames2 = std::move(holoscan::radar::testing::detail::FrameBankC::captured);
  }
  {
    const std::lock_guard lock(holoscan::radar::testing::detail::FrameBankE::mutex);
    result.frames4 = std::move(holoscan::radar::testing::detail::FrameBankE::captured);
  }
  return result;
}

TEST(MVDREstimatorOpTest, DefaultsToRadarBoresightConvention) {
  EXPECT_EQ(MVDRParams{}.angle_convention.preset, AngleConventionPreset::kRadarBoresight);
}

TEST(MVDREstimatorOpTest, SpatialSmoothingResolvesCoherentSourcesUla) {
  constexpr int64_t kN = 10;
  constexpr int64_t kS = 64;
  constexpr double kAz0 = 0.30;
  constexpr double kAz1 = -0.30;

  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  const auto block = make_block_two_coherent(geom, conv, kAz0, 0.0, kAz1, 0.0, kS, 0.02, 71u);
  auto snap = block_to_device<double>(block, {1, kN, kS});

  const auto out = run_mvdr<double>(
      snap, make_params(geom, conv, linspace(-0.6, 0.6, 241), {0.0}, -1.0, 2, {4}, true));

  ASSERT_EQ(out.eigenvalues.shape, (std::vector<int64_t>{1, 7}));
  EXPECT_EQ(out.source_count.as<int32_t>()[0], 2);

  const double* dm = out.doa_multi.as<double>();
  const double a = dm[0];
  const double b = dm[2];
  const bool matched = (std::abs(a - kAz0) < 0.04 && std::abs(b - kAz1) < 0.04) ||
                       (std::abs(a - kAz1) < 0.04 && std::abs(b - kAz0) < 0.04);
  EXPECT_TRUE(matched) << "a=" << a << " b=" << b;
}

TEST(MVDREstimatorOpTest, SpatialSmoothing2DUraRecoversDoa) {
  constexpr int64_t kNy = 4;
  constexpr int64_t kNz = 4;
  constexpr int64_t kS = 32;
  constexpr double kTruthAz = 0.10;
  constexpr double kTruthEl = 0.06;

  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto geom = make_planar_ura_geometry(kNy, kNz, 0.5, 1.0);
  const auto block = make_block(geom, conv, kTruthAz, kTruthEl, kS, 0.05, 44u);
  auto snap = block_to_device<double>(block, {1, geom.num_elements(), kS});

  const auto out =
      run_mvdr<double>(snap, make_params(geom, conv, linspace(-0.4, 0.4, 81),
                                         linspace(-0.3, 0.3, 61), -1.0, 1, {2, 2}, true));

  const double* aoa = out.aoa.as<double>();
  EXPECT_NEAR(aoa[0], kTruthAz, 0.03);
  EXPECT_NEAR(aoa[1], kTruthEl, 0.03);
}

TEST(MVDREstimatorOpTest, SpatialSmoothingReducesEigenvalueChannels) {
  constexpr int64_t kN = 8;
  constexpr int64_t kS = 32;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto block = make_block(geom, conv, 0.1, 0.0, kS, 0.05, 45u);
  auto snap = block_to_device<double>(block, {1, kN, kS});

  const auto out = run_mvdr<double>(
      snap, make_params(geom, conv, linspace(-0.3, 0.3, 61), {0.0}, -1.0, 1, {3}, false));

  ASSERT_EQ(out.eigenvalues.shape, (std::vector<int64_t>{1, 6}));
  const double* aoa = out.aoa.as<double>();
  EXPECT_NEAR(aoa[0], 0.1, 0.03);
}

TEST(MVDREstimatorOpTest, SpatialSmoothingRejectsNonUniformGeometry) {
  ArrayGeometry geom;
  geom.wavelength = 1.0;
  const double ys[4] = {0.0, 0.5, 1.0, 2.0};
  for (double y : ys) {
    geom.element_positions.insert(geom.element_positions.end(), {0.0, y, 0.0});
  }
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto a = steering_vector(geom, conv, 0.0, 0.0);
  auto snap = block_to_device<double>(a, {1, geom.num_elements()});

  const FiveOutGraphRunResult result =
      run_mvdr_result<double>(snap, make_params(geom, conv, {0.0}, {0.0}, -1.0, 1, {2}));
  EXPECT_TRUE(result.plan_ok);
  EXPECT_TRUE(result.frames0.empty());
  EXPECT_FALSE(result.diagnostics.empty());
}

TEST(MVDREstimatorOpTest, RecoversKnownDoaUra) {
  constexpr int64_t kNy = 4;
  constexpr int64_t kNz = 4;
  constexpr int64_t kS = 32;
  constexpr double kTruthAz = 0.12;
  constexpr double kTruthEl = 0.05;

  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto geom = make_planar_ura_geometry(kNy, kNz, 0.5, 1.0);
  const auto block = make_block(geom, conv, kTruthAz, kTruthEl, kS, 0.05, 7u);
  auto snap = block_to_device<double>(block, {1, geom.num_elements(), kS});

  const auto out = run_mvdr<double>(
      snap, make_params(geom, conv, linspace(-0.5, 0.5, 101), linspace(-0.3, 0.3, 61)));

  ASSERT_EQ(out.aoa.shape, (std::vector<int64_t>{1, 2}));
  const double* aoa = out.aoa.as<double>();
  EXPECT_NEAR(aoa[0], kTruthAz, 0.02);
  EXPECT_NEAR(aoa[1], kTruthEl, 0.02);
}

// MVDR documents four optional outputs, but every one of them used to be
// emitted under HOLOSCAN_TRY, so an application that declined any of them died
// on its first tick with RUNTIME_OPERATOR_CALLBACK_FAILED — the contract
// promised optional and the code required wired. Nothing exposes an output's
// wiring to the operator, so it cannot ask first; it emits and absorbs the
// failure instead. This pins that wiring aoa_estimates alone is legal.
//
// Several frames rather than one: a latched port still takes a pool loan on
// the tick it discovers it is unwired, so a slot the runtime failed to recycle
// would starve the operator a few frames in rather than immediately.
// See OPERATOR_API.md, "Optional output convention".
TEST(MVDREstimatorOpTest, UnwiredOptionalOutputsDoNotFailTheRun) {
  constexpr int64_t kN = 8;
  constexpr int64_t kS = 32;
  constexpr std::size_t kFrames = 16;

  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  const auto block = make_block(geom, conv, 0.08, 0.0, kS, 0.05, 11u);

  std::vector<std::shared_ptr<holoscan::Tensor>> snapshots;
  snapshots.reserve(kFrames);
  for (std::size_t i = 0; i < kFrames; ++i) {
    snapshots.push_back(block_to_device<double>(block, {1, kN, kS}));
  }

  auto params = make_params(geom, conv, linspace(-0.5, 0.5, 101), {0.0});
  const holoscan::radar::MaxTensorExtents sized = holoscan::radar::testing::extents_for_scan_grid(
      params.az_coordinates.size(), params.el_coordinates.size());

  const holoscan::radar::testing::GraphRunResult out =
      holoscan::radar::testing::run_single_io_operator_graph_result<MVDREstimatorOp<double>>(
          std::move(snapshots), std::move(params), "aoa_estimates",
          [](holoscan::Graph& graph, const auto& source, auto& op, const auto& sink) {
            const holoscan::ConnectionOptions q{.queue_depth = 4U};
            graph.add_flow(source->out, op->snapshot, q);
            graph.add_flow(op->aoa_estimates, sink->in, q);
          },
          {"angle_spectrum", "doa_multi", "source_count", "eigenvalues"}, sized);

  ASSERT_TRUE(out.plan_ok) << out.plan_json;
  EXPECT_TRUE(out.diagnostics.empty());
  // Every tick's wired output is delivered, start to finish.
  ASSERT_EQ(out.frames.size(), kFrames);
  for (const HostTensor& frame : out.frames) {
    EXPECT_EQ(frame.shape, (std::vector<int64_t>{1, 2}));
  }
}

TEST(MVDREstimatorOpTest, RecoversKnownDoaUlaAzOnly) {
  constexpr int64_t kN = 8;
  constexpr int64_t kS = 32;
  constexpr double kTruthAz = 0.08;

  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  const auto block = make_block(geom, conv, kTruthAz, 0.0, kS, 0.05, 3u);
  auto snap = block_to_device<double>(block, {1, kN, kS});

  const auto out = run_mvdr<double>(snap, make_params(geom, conv, linspace(-0.3, 0.3, 61), {0.0}));

  ASSERT_EQ(out.spectrum.shape, (std::vector<int64_t>{1, 61, 1}));
  const double* aoa = out.aoa.as<double>();
  EXPECT_NEAR(aoa[0], kTruthAz, 0.02);
  EXPECT_NEAR(aoa[1], 0.0, 1e-9);
}

TEST(MVDREstimatorOpTest, FixedDiagonalLoadingRecoversDoa) {
  constexpr int64_t kN = 8;
  constexpr int64_t kS = 32;
  constexpr double kTruthAz = 0.1;

  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  const auto block = make_block(geom, conv, kTruthAz, 0.0, kS, 0.05, 9u);
  auto snap = block_to_device<double>(block, {1, kN, kS});

  const auto out =
      run_mvdr<double>(snap, make_params(geom, conv, linspace(-0.3, 0.3, 61), {0.0}, 1e-2));

  const double* aoa = out.aoa.as<double>();
  EXPECT_NEAR(aoa[0], kTruthAz, 0.02);
}

TEST(MVDREstimatorOpTest, DiagonalLoadingConfigurableFromParams) {
  // Exercise the authoring-time Params value for diagonal_loading.
  constexpr int64_t kN = 8;
  constexpr int64_t kS = 32;
  constexpr double kTruthAz = 0.1;

  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  const auto block = make_block(geom, conv, kTruthAz, 0.0, kS, 0.05, 51u);
  auto snap_a = block_to_device<double>(block, {1, kN, kS});
  auto snap_b = block_to_device<double>(block, {1, kN, kS});

  const auto out_a =
      run_mvdr<double>(snap_a, make_params(geom, conv, linspace(-0.3, 0.3, 31), {0.0}, 1e-3));
  const auto out_b =
      run_mvdr<double>(snap_b, make_params(geom, conv, linspace(-0.3, 0.3, 31), {0.0}, 1.0));

  EXPECT_EQ(out_a.spectrum.shape, out_b.spectrum.shape);
  EXPECT_NE(out_a.spectrum.data, out_b.spectrum.data);
}

TEST(MVDREstimatorOpTest, MaxSourcesConfigurableFromParams) {
  constexpr int64_t kN = 8;
  constexpr int64_t kS = 32;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto block = make_block(geom, conv, 0.1, 0.0, kS, 0.05, 101u);
  auto snap1 = block_to_device<double>(block, {1, kN, kS});
  auto snap3 = block_to_device<double>(block, {1, kN, kS});

  const auto out1 =
      run_mvdr<double>(snap1, make_params(geom, conv, linspace(-0.3, 0.3, 61), {0.0}, -1.0, 1));
  const auto out3 =
      run_mvdr<double>(snap3, make_params(geom, conv, linspace(-0.3, 0.3, 61), {0.0}, -1.0, 3));

  EXPECT_EQ(out1.doa_multi.shape, (std::vector<int64_t>{1, 1, 2}));
  EXPECT_EQ(out3.doa_multi.shape, (std::vector<int64_t>{1, 3, 2}));
}

TEST(MVDREstimatorOpTest, Rejects2DSingleSnapshot) {
  constexpr int64_t kN = 4;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto a = steering_vector(geom, conv, 0.05, 0.0);
  auto snap = block_to_device<double>(a, {1, kN});

  const auto result =
      run_mvdr_result<double>(snap, make_params(geom, conv, linspace(-0.2, 0.2, 21), {0.0}));
  EXPECT_TRUE(result.plan_ok);
  EXPECT_TRUE(result.frames0.empty());
  EXPECT_FALSE(result.diagnostics.empty());
}

TEST(MVDREstimatorOpTest, FloatInstantiationSmoke) {
  constexpr int64_t kN = 8;
  constexpr int64_t kS = 32;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto block = make_block(geom, conv, 0.06, 0.0, kS, 0.05, 11u);
  auto snap = block_to_device<float>(block, {1, kN, kS});

  const auto out = run_mvdr<float>(snap, make_params(geom, conv, linspace(-0.3, 0.3, 61), {0.0}));
  ASSERT_EQ(out.aoa.shape, (std::vector<int64_t>{1, 2}));
  const float* aoa = out.aoa.as<float>();
  EXPECT_NEAR(aoa[0], 0.06f, 0.03f);
}

TEST(MVDREstimatorOpTest, FloatDoubleAgreeOnDoa) {
  constexpr int64_t kN = 8;
  constexpr int64_t kS = 32;
  constexpr double kTruthAz = 0.09;

  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  const auto block = make_block(geom, conv, kTruthAz, 0.0, kS, 0.05, 5u);
  auto snap_f = block_to_device<float>(block, {1, kN, kS});
  auto snap_d = block_to_device<double>(block, {1, kN, kS});

  MVDRParams params = make_params(geom, conv, linspace(-0.3, 0.3, 61), {0.0});

  const auto out_f = run_mvdr<float>(snap_f, params);
  const auto out_d = run_mvdr<double>(snap_d, params);

  const float* aoa_f = out_f.aoa.as<float>();
  const double* aoa_d = out_d.aoa.as<double>();
  EXPECT_NEAR(aoa_f[0], static_cast<float>(aoa_d[0]), 1e-2f);
}

TEST(MVDREstimatorOpTest, DeterministicAcrossTicks) {
  constexpr int64_t kN = 8;
  constexpr int64_t kS = 64;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto block = make_block(geom, conv, 0.09, 0.0, kS, 0.02, 171u);
  auto snap = block_to_device<double>(block, {1, kN, kS});
  MVDRParams params = make_params(geom, conv, linspace(-0.3, 0.3, 61), {0.0}, 1e-2, 2);
  auto extents = holoscan::radar::testing::default_test_extents();
  extents.max_items = 1;

  const FiveOutGraphRunResult result =
      run_live_mvdr({{snap}, {snap}}, std::move(params), {}, extents);

  ASSERT_TRUE(result.plan_ok) << result.plan_json;
  ASSERT_TRUE(result.diagnostics.empty());
  ASSERT_EQ(result.frames1.size(), 2U);
  ASSERT_EQ(result.frames2.size(), 2U);
  ASSERT_EQ(result.frames4.size(), 2U);
  EXPECT_EQ(result.frames1[0].data, result.frames1[1].data);
  EXPECT_EQ(result.frames2[0].data, result.frames2[1].data);
  EXPECT_EQ(result.frames4[0].data, result.frames4[1].data);
}

TEST(MVDREstimatorOpTest, DeterministicAcrossTicksAboveMatInvThreshold) {
  constexpr int64_t kN = 40;
  constexpr int64_t kS = 64;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto block = make_block(geom, conv, 0.09, 0.0, kS, 0.02, 172u);
  auto snap = block_to_device<double>(block, {1, kN, kS});
  MVDRParams params = make_params(geom, conv, linspace(-0.3, 0.3, 31), {0.0}, 1e-2);
  auto extents = holoscan::radar::testing::default_test_extents();
  extents.max_items = 1;

  const FiveOutGraphRunResult result =
      run_live_mvdr({{snap}, {snap}}, std::move(params), {}, extents);

  ASSERT_TRUE(result.plan_ok) << result.plan_json;
  ASSERT_TRUE(result.diagnostics.empty());
  ASSERT_EQ(result.frames1.size(), 2U);
  ASSERT_EQ(result.frames2.size(), 2U);
  ASSERT_EQ(result.frames4.size(), 2U);
  EXPECT_EQ(result.frames1[0].data, result.frames1[1].data);
  EXPECT_EQ(result.frames2[0].data, result.frames2[1].data);
  EXPECT_EQ(result.frames4[0].data, result.frames4[1].data);
}

TEST(MVDREstimatorOpTest, DirectionCosineMasksInvisibleRegion) {
  constexpr int64_t kN = 4;
  constexpr int64_t kS = 16;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kDirectionCosine;
  const auto block = make_block(geom, conv, 0.0, 0.0, kS, 0.05, 2u);
  auto snap = block_to_device<double>(block, {1, kN, kS});

  const auto out = run_mvdr<double>(
      snap, make_params(geom, conv, linspace(-1.0, 1.0, 5), linspace(-1.0, 1.0, 5)));

  const double* spec = out.spectrum.as<double>();
  EXPECT_NEAR(spec[0], 0.0, 1e-9);
}

TEST(MVDREstimatorOpTest, RejectsInvalidScanCoordinates) {
  constexpr int64_t kN = 4;
  constexpr int64_t kS = 16;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  const auto block = make_block(geom, AngleConvention{}, 0.0, 0.0, kS, 0.05, 2u);
  AngleConvention direction_cosine;
  direction_cosine.preset = AngleConventionPreset::kDirectionCosine;
  auto out_of_range = run_mvdr_result<double>(block_to_device<double>(block, {1, kN, kS}),
                                              make_params(geom, direction_cosine, {-1.01}, {0.0}));
  EXPECT_FALSE(out_of_range.diagnostics.empty());

  auto non_finite = run_mvdr_result<double>(
      block_to_device<double>(block, {1, kN, kS}),
      make_params(geom, AngleConvention{}, {0.0}, {std::numeric_limits<double>::quiet_NaN()}));
  EXPECT_FALSE(non_finite.diagnostics.empty());
}

TEST(MVDREstimatorOpTest, MultipleItemsRecoverIndependentDoa) {
  constexpr int64_t kNy = 4;
  constexpr int64_t kNz = 4;
  constexpr int64_t kS = 24;
  constexpr double kAz0 = 0.11;
  constexpr double kEl0 = 0.04;
  constexpr double kAz1 = -0.07;
  constexpr double kEl1 = -0.02;

  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto geom = make_planar_ura_geometry(kNy, kNz, 0.5, 1.0);
  const auto b0 = make_block(geom, conv, kAz0, kEl0, kS, 0.05, 21u);
  const auto b1 = make_block(geom, conv, kAz1, kEl1, kS, 0.05, 22u);
  std::vector<std::complex<double>> batch;
  batch.insert(batch.end(), b0.begin(), b0.end());
  batch.insert(batch.end(), b1.begin(), b1.end());
  auto snap = block_to_device<double>(batch, {2, geom.num_elements(), kS});

  const auto out = run_mvdr<double>(
      snap, make_params(geom, conv, linspace(-0.4, 0.4, 81), linspace(-0.2, 0.2, 41)));

  ASSERT_EQ(out.aoa.shape, (std::vector<int64_t>{2, 2}));
  ASSERT_EQ(out.spectrum.shape, (std::vector<int64_t>{2, 81, 41}));
  const double* aoa = out.aoa.as<double>();
  EXPECT_NEAR(aoa[0], kAz0, 0.02);
  EXPECT_NEAR(aoa[1], kEl0, 0.02);
  EXPECT_NEAR(aoa[2], kAz1, 0.02);
  EXPECT_NEAR(aoa[3], kEl1, 0.02);
}

TEST(MVDREstimatorOpTest, MultiSourceRecoversTwoDoa) {
  constexpr int64_t kN = 12;
  constexpr int64_t kS = 128;
  constexpr double kAz0 = 0.20;
  constexpr double kAz1 = -0.15;

  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  const auto block = make_block_two(geom, conv, kAz0, 0.0, kAz1, 0.0, kS, 0.02, 71u);
  auto snap = block_to_device<double>(block, {1, kN, kS});

  const auto out =
      run_mvdr<double>(snap, make_params(geom, conv, linspace(-0.5, 0.5, 201), {0.0}, -1.0, 2));

  ASSERT_EQ(out.doa_multi.shape, (std::vector<int64_t>{1, 2, 2}));
  ASSERT_EQ(out.source_count.shape, (std::vector<int64_t>{1}));
  EXPECT_EQ(out.source_count.as<int32_t>()[0], 2);

  const double* dm = out.doa_multi.as<double>();
  const double az_a = dm[0];
  const double az_b = dm[2];
  const bool matched = (std::abs(az_a - kAz0) < 0.03 && std::abs(az_b - kAz1) < 0.03) ||
                       (std::abs(az_a - kAz1) < 0.03 && std::abs(az_b - kAz0) < 0.03);
  EXPECT_TRUE(matched) << "az_a=" << az_a << " az_b=" << az_b;

  const double* aoa = out.aoa.as<double>();
  EXPECT_NEAR(dm[0], aoa[0], 1e-9);
  EXPECT_NEAR(dm[1], aoa[1], 1e-9);
}

TEST(MVDREstimatorOpTest, MaxSourcesCapacityNanFillsUnusedRows) {
  constexpr int64_t kN = 8;
  constexpr int64_t kS = 64;
  constexpr double kTruthAz = 0.1;

  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  const auto block = make_block(geom, conv, kTruthAz, 0.0, kS, 0.02, 81u);
  auto snap = block_to_device<double>(block, {1, kN, kS});

  const auto out =
      run_mvdr<double>(snap, make_params(geom, conv, linspace(-0.3, 0.3, 61), {0.0}, -1.0, 4));

  ASSERT_EQ(out.doa_multi.shape, (std::vector<int64_t>{1, 4, 2}));
  const int32_t count = out.source_count.as<int32_t>()[0];
  EXPECT_GE(count, 1);
  EXPECT_LE(count, 4);

  const double* dm = out.doa_multi.as<double>();
  EXPECT_NEAR(dm[0], kTruthAz, 0.02);
  for (int64_t k = 0; k < 4; ++k) {
    if (k < count) {
      EXPECT_FALSE(std::isnan(dm[k * 2]));
    } else {
      EXPECT_TRUE(std::isnan(dm[k * 2]));
      EXPECT_TRUE(std::isnan(dm[k * 2 + 1]));
    }
  }
}

TEST(MVDREstimatorOpTest, EigenvaluesRankOneSourceDominant) {
  constexpr int64_t kN = 8;
  constexpr int64_t kS = 128;

  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  const auto block = make_block(geom, conv, 0.1, 0.0, kS, 0.02, 91u);
  auto snap = block_to_device<double>(block, {1, kN, kS});

  const auto out = run_mvdr<double>(snap, make_params(geom, conv, linspace(-0.3, 0.3, 31), {0.0}));

  ASSERT_EQ(out.eigenvalues.shape, (std::vector<int64_t>{1, kN}));
  const double* ev = out.eigenvalues.as<double>();
  for (int64_t c = 0; c < kN; ++c) {
    EXPECT_GE(ev[c], -1e-6);
    if (c > 0) {
      EXPECT_GE(ev[c], ev[c - 1] - 1e-9);
    }
  }
  EXPECT_GT(ev[kN - 1], 10.0 * ev[kN - 2]);
}

TEST(MVDREstimatorOpTest, RejectsInvalidMaxSources) {
  constexpr int64_t kN = 4;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto a = steering_vector(geom, conv, 0.0, 0.0);
  auto snap = block_to_device<double>(a, {1, kN, 1});
  const FiveOutGraphRunResult result =
      run_mvdr_result<double>(snap, make_params(geom, conv, {0.0}, {0.0}, -1.0, 0));
  EXPECT_TRUE(result.plan_ok);
  EXPECT_TRUE(result.frames0.empty());
  EXPECT_FALSE(result.diagnostics.empty());
}

TEST(MVDREstimatorOpTest, EmptyItemsProducesEmptyOutputs) {
  constexpr int64_t kN = 4;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  std::vector<std::complex<double>> host;
  auto snap = to_device_tensor(host, {0, kN, 4}, complex_dtype<double>());

  const auto out = run_mvdr<double>(snap, make_params(geom, conv, {-0.1, 0.0, 0.1}, {0.0}));

  EXPECT_EQ(out.aoa.shape, (std::vector<int64_t>{0, 2}));
  EXPECT_EQ(out.spectrum.shape, (std::vector<int64_t>{0, 3, 1}));
  EXPECT_EQ(out.doa_multi.shape, (std::vector<int64_t>{0, 1, 2}));
  EXPECT_EQ(out.source_count.shape, (std::vector<int64_t>{0}));
  EXPECT_EQ(out.eigenvalues.shape, (std::vector<int64_t>{0, kN}));
}

TEST(MVDREstimatorOpTest, RejectsChannelMismatch) {
  constexpr int64_t kN = 4;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  std::vector<std::complex<double>> host(static_cast<size_t>((kN - 1) * 4), {1.0, 0.0});
  auto snap = to_device_tensor(host, {1, kN - 1, 4}, complex_dtype<double>());
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const FiveOutGraphRunResult result =
      run_mvdr_result<double>(snap, make_params(geom, conv, {0.0}, {0.0}));
  EXPECT_TRUE(result.plan_ok);
  EXPECT_TRUE(result.frames0.empty());
  EXPECT_FALSE(result.diagnostics.empty());
}

TEST(MVDREstimatorOpTest, RejectsEmptyScanGrid) {
  constexpr int64_t kN = 4;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  std::vector<std::complex<double>> host(static_cast<size_t>(kN * 4), {1.0, 0.0});
  auto snap = to_device_tensor(host, {1, kN, 4}, complex_dtype<double>());
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const FiveOutGraphRunResult result =
      run_mvdr_result<double>(snap, make_params(geom, conv, {}, {0.0}));
  EXPECT_TRUE(result.plan_ok);
  EXPECT_TRUE(result.frames0.empty());
  EXPECT_FALSE(result.diagnostics.empty());
}

TEST(MVDREstimatorOpTest, RejectsInvalidSnapshotRank) {
  constexpr int64_t kN = 4;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  std::vector<std::complex<double>> host(static_cast<size_t>(kN), {1.0, 0.0});
  auto snap = to_device_tensor(host, {kN}, complex_dtype<double>());
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const FiveOutGraphRunResult result =
      run_mvdr_result<double>(snap, make_params(geom, conv, {0.0}, {0.0}));
  EXPECT_TRUE(result.plan_ok);
  EXPECT_TRUE(result.frames0.empty());
  EXPECT_FALSE(result.diagnostics.empty());
}

TEST(MVDREstimatorOpTest, RejectsSnapshotDtypeMismatch) {
  constexpr int64_t kN = 4;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto a = steering_vector(geom, conv, 0.0, 0.0);
  auto snap_f = block_to_device<float>(a, {1, kN, 1});
  const FiveOutGraphRunResult result =
      run_mvdr_result<double>(snap_f, make_params(geom, conv, {0.0}, {0.0}));
  EXPECT_TRUE(result.plan_ok);
  EXPECT_TRUE(result.frames0.empty());
  EXPECT_FALSE(result.diagnostics.empty());
}

TEST(MVDREstimatorOpTest, IsValidMaskExcludesWrongDoaSnapshots) {
  constexpr int64_t kN = 8;
  constexpr int64_t kS = 32;
  constexpr double kTruthAz = 0.14;
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  const auto truth = make_block(geom, conv, kTruthAz, 0.0, kS, 0.01, 701u);
  const auto wrong = make_block(geom, conv, -0.24, 0.0, kS, 0.01, 702u);
  std::vector<std::complex<double>> mixed(static_cast<size_t>(kN * 2 * kS));
  for (int64_t channel = 0; channel < kN; ++channel) {
    for (int64_t look = 0; look < kS; ++look) {
      mixed[static_cast<size_t>(channel * 2 * kS + look)] =
          truth[static_cast<size_t>(channel * kS + look)];
      mixed[static_cast<size_t>(channel * 2 * kS + kS + look)] =
          wrong[static_cast<size_t>(channel * kS + look)];
    }
  }
  auto snap = block_to_device<double>(mixed, {1, kN, 2 * kS});
  std::vector<uint8_t> mask_host(static_cast<size_t>(2 * kS), uint8_t{0});
  std::fill_n(mask_host.begin(), static_cast<size_t>(kS), uint8_t{1});
  auto mask = to_device_tensor(mask_host, {1, 2 * kS}, DLDataType{kDLUInt, 8, 1});
  auto params = make_params(geom, conv, linspace(-0.4, 0.4, 81), {0.0});
  const auto extents = holoscan::radar::testing::extents_for_scan_grid(
      params.az_coordinates.size(), params.el_coordinates.size());

  const auto result = run_dual_io_operator_graph_result<MVDREstimatorOp<double>>(
      {mask}, {snap}, std::move(params), "aoa_estimates", "angle_spectrum",
      [](holoscan::Graph& graph, const auto& source, auto& op, const auto& sink0,
         const auto& sink1) {
        const holoscan::ConnectionOptions q{.queue_depth = 4U};
        graph.add_flow(source->out0, op->is_valid, q);
        graph.add_flow(source->out1, op->snapshot, q);
        graph.add_flow(op->aoa_estimates, sink0->in, q);
        graph.add_flow(op->angle_spectrum, sink1->in, q);
      },
      extents, {"doa_multi", "source_count", "eigenvalues"});
  ASSERT_TRUE(result.plan_ok) << result.plan_json;
  ASSERT_TRUE(result.diagnostics.empty());
  ASSERT_EQ(result.frames0.size(), 1U);
  EXPECT_NEAR(result.frames0[0].as<double>()[0], kTruthAz, 0.03);
}

TEST(MVDREstimatorOpTest, AllInvalidMaskBlanksDoaAndSpectrum) {
  constexpr int64_t kN = 4;
  constexpr int64_t kS = 8;
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  const auto block = make_block(geom, conv, 0.1, 0.0, kS, 0.01, 703u);
  auto snap = block_to_device<double>(block, {1, kN, kS});
  auto mask =
      to_device_tensor(std::vector<uint8_t>(kS, uint8_t{0}), {1, kS}, DLDataType{kDLUInt, 8, 1});

  const auto result = run_dual_io_operator_graph_result<MVDREstimatorOp<double>>(
      {mask}, {snap}, make_params(geom, conv, {-0.1, 0.0, 0.1}, {0.0}), "aoa_estimates",
      "angle_spectrum",
      [](holoscan::Graph& graph, const auto& source, auto& op, const auto& sink0,
         const auto& sink1) {
        const holoscan::ConnectionOptions q{.queue_depth = 4U};
        graph.add_flow(source->out0, op->is_valid, q);
        graph.add_flow(source->out1, op->snapshot, q);
        graph.add_flow(op->aoa_estimates, sink0->in, q);
        graph.add_flow(op->angle_spectrum, sink1->in, q);
      },
      holoscan::radar::testing::default_test_extents(),
      {"doa_multi", "source_count", "eigenvalues"});
  ASSERT_TRUE(result.plan_ok) << result.plan_json;
  ASSERT_TRUE(result.diagnostics.empty());
  ASSERT_EQ(result.frames0.size(), 1U);
  ASSERT_EQ(result.frames1.size(), 1U);
  EXPECT_TRUE(std::isnan(result.frames0[0].as<double>()[0]));
  const double* spectrum = result.frames1[0].as<double>();
  for (size_t index = 0; index < result.frames1[0].data.size() / sizeof(double); ++index) {
    EXPECT_EQ(spectrum[index], 0.0);
  }
}

TEST(MVDREstimatorOpTest, LiveAzCoordinatesAppliesForOneTick) {
  constexpr int64_t kN = 4;
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  const auto block = make_block(geom, conv, 0.0, 0.0, 8, 0.02, 301u);
  auto snap = block_to_device<double>(block, {1, kN, 8});
  MVDRParams params = make_params(geom, conv, {-0.2, 0.2}, {0.0});

  const FiveOutGraphRunResult result = run_live_mvdr(
      {{snap, std::nullopt, std::vector<double>{-0.1, 0.0, 0.1}, std::nullopt, std::nullopt, 0},
       {snap, std::nullopt, std::vector<double>{-0.2, -0.1, 0.0, 0.1, 0.2}, std::nullopt,
        std::nullopt, 1}},
      std::move(params), {.az = true});

  ASSERT_TRUE(result.plan_ok) << result.plan_json;
  ASSERT_TRUE(result.diagnostics.empty());
  ASSERT_EQ(result.frames1.size(), 2U);
  EXPECT_EQ(result.frames1[0].shape, (std::vector<int64_t>{1, 3, 1}));
  // Tick 1's override is skewed, so it misses Exact and the grid is the
  // two-angle Params az again.
  EXPECT_EQ(result.frames1[1].shape, (std::vector<int64_t>{1, 2, 1}));
}

TEST(MVDREstimatorOpTest, LiveDiagonalLoadingAppliesForOneTick) {
  constexpr int64_t kN = 8;
  constexpr int64_t kS = 32;
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  const auto block = make_block(geom, conv, 0.1, 0.0, kS, 0.02, 617u);
  auto snap = block_to_device<double>(block, {1, kN, kS});
  MVDRParams params = make_params(geom, conv, linspace(-0.3, 0.3, 31), {0.0}, 1e-3);

  const FiveOutGraphRunResult result = run_live_mvdr(
      {{snap, std::nullopt, std::nullopt, std::nullopt, std::nullopt, 0, std::nullopt,
        std::nullopt},
       {snap, std::nullopt, std::nullopt, std::nullopt, std::nullopt, 0, 1.0, std::nullopt},
       {snap, std::nullopt, std::nullopt, std::nullopt, std::nullopt, 0, std::nullopt,
        std::nullopt}},
      std::move(params), {.diagonal_loading = true});

  ASSERT_TRUE(result.plan_ok) << result.plan_json;
  ASSERT_TRUE(result.diagnostics.empty());
  ASSERT_EQ(result.frames1.size(), 3U);
  EXPECT_EQ(result.frames1[0].data, result.frames1[2].data);
  EXPECT_NE(result.frames1[0].data, result.frames1[1].data);
}

// max_sources widens doa_multi to {items, max_sources, 2}, so the override is
// observable in the shape alone, which is why this one can span ticks where
// the diagonal_loading test cannot.
TEST(MVDREstimatorOpTest, LiveMaxSourcesAppliesForOneTickAndReallocates) {
  constexpr int64_t kN = 4;
  constexpr int64_t kLiveMaxSources = 3;
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  const auto block = make_block(geom, conv, 0.1, 0.0, 16, 0.02, 618u);
  auto snap = block_to_device<double>(block, {1, kN, 16});

  MVDRParams params = make_params(geom, conv, linspace(-0.3, 0.3, 31), {0.0});
  params.max_sources = 1;

  // Tick 1 sends nothing, so the allocation narrows back to the Params cap.
  const FiveOutGraphRunResult result =
      run_live_mvdr({{snap, std::nullopt, std::nullopt, std::nullopt, std::nullopt, 0, std::nullopt,
                      kLiveMaxSources},
                     {snap, std::nullopt, std::nullopt, std::nullopt, std::nullopt, 0, std::nullopt,
                      std::nullopt}},
                    std::move(params), {.max_sources = true});

  ASSERT_TRUE(result.plan_ok) << result.plan_json;
  ASSERT_TRUE(result.diagnostics.empty());
  ASSERT_EQ(result.frames2.size(), 2U);
  EXPECT_EQ(result.frames2[0].shape, (std::vector<int64_t>{1, kLiveMaxSources, 2}));
  EXPECT_EQ(result.frames2[1].shape, (std::vector<int64_t>{1, 1, 2}));
}

TEST(MVDREstimatorOpTest, LiveSpatialSmoothingAppliesForOneTick) {
  constexpr int64_t kN = 8;
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  const auto block = make_block(geom, conv, 0.1, 0.0, 16, 0.02, 303u);
  auto snap = block_to_device<double>(block, {1, kN, 16});
  MVDRParams params = make_params(geom, conv, linspace(-0.3, 0.3, 31), {0.0});

  const FiveOutGraphRunResult result =
      run_live_mvdr({{snap, std::nullopt, std::nullopt, std::nullopt, std::nullopt, 0},
                     {snap, std::nullopt, std::nullopt, std::nullopt,
                      SpatialSmoothingOverride{.count = 1, .first = 3}, 0},
                     {snap, std::nullopt, std::nullopt, std::nullopt, std::nullopt, 0}},
                    std::move(params), {.smoothing = true});

  ASSERT_TRUE(result.plan_ok) << result.plan_json;
  ASSERT_TRUE(result.diagnostics.empty());
  ASSERT_EQ(result.frames4.size(), 3U);
  // Smoothing changes the covariance order, so the eigenvalue count is the
  // observable. Ticks 0 and 2 publish nothing and share the Params order.
  EXPECT_NE(result.frames4[1].shape, result.frames4[0].shape);
  EXPECT_EQ(result.frames4[2].shape, result.frames4[0].shape);
}

TEST(MVDREstimatorOpTest, LiveSpatialSmoothingRejectsMalformedPod) {
  constexpr int64_t kN = 4;
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  const auto block = make_block(geom, conv, 0.0, 0.0, 8, 0.02, 304u);
  auto snap = block_to_device<double>(block, {1, kN, 8});
  MVDRParams params = make_params(geom, conv, {0.0}, {0.0});

  const FiveOutGraphRunResult result = run_live_mvdr(
      {{snap, std::nullopt, std::nullopt, std::nullopt, SpatialSmoothingOverride{.count = 3}, 0}},
      std::move(params), {.smoothing = true});

  ASSERT_TRUE(result.plan_ok) << result.plan_json;
  EXPECT_TRUE(result.frames1.empty());
  EXPECT_FALSE(result.diagnostics.empty());
}

TEST(MVDREstimatorOpTest, LiveAzCoordinatesRejectsExtentAboveMaxAz) {
  constexpr int64_t kN = 4;
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  const auto block = make_block(geom, conv, 0.0, 0.0, 8, 0.02, 305u);
  auto snap = block_to_device<double>(block, {1, kN, 8});
  MVDRParams params = make_params(geom, conv, {0.0}, {0.0});

  const FiveOutGraphRunResult result = run_live_mvdr(
      {{snap, std::nullopt, std::vector<double>(65U, 0.0), std::nullopt, std::nullopt, 0}},
      std::move(params), {.az = true});

  ASSERT_TRUE(result.plan_ok) << result.plan_json;
  EXPECT_TRUE(result.frames1.empty());
  EXPECT_FALSE(result.diagnostics.empty());
}

}  // namespace
