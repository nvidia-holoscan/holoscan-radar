// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Whole-Graph coverage for the autofocus estimate's journey off the operator
// that produced it: transfer onto a differently formed product through
// PhaseCorrectionOp, and the retain/reform path that forms an auxiliary image,
// estimates on it, and corrects the pulses the product is formed from.
//
// These cases cannot live in tests/cpp/sar/test_autofocus.cpp. Single-operator
// suites use holoscan::testing, which drives exactly one operator; what is
// under test here is the wiring between four of them. The scene is synthetic
// point targets and the sources are fixture operators, so there is no reader,
// no CPHD, and no application config.

#include <gtest/gtest.h>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <complex>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/diagnostic.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/expected_macros.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "radar_common/descriptors/coordinate_frame.hpp"
#include "radar_common/descriptors/max_tensor_extents.hpp"
#include "radar_common/descriptors/sample_geometry.hpp"
#include "radar_common/sdk/tensor_io.hpp"
#include "radar_common/sim/sar_phase_history_sim.hpp"
#include "radar_common/tensor/tensor_utils.hpp"
#include "sar/autofocus_phase_support.hpp"
#include "autofocus/cpp/phase_resample_op.hpp"
#include "io/cphd_reader_op/cphd_reader_op.hpp"
#include "sar/aperture_manager_op/aperture_manager_op.hpp"
#include "sar/autofocus_op/autofocus_op.hpp"
#include "sar/backprojection_op/backprojection_op.hpp"
#include "sar/phase_correction_op/phase_correction_op.hpp"
#include "utils/tensor_fixtures.hpp"

namespace {

namespace hr = holoscan::radar;
using holoscan::radar::sar_test::remove_affine;
using holoscan::radar::sar_test::to_bin_order;
namespace test = holoscan::radar::testing;

using hr::ApertureEmitMode;
using hr::ApertureManagerOp;
using hr::ApertureManagerParams;
using hr::AutofocusAzimuthAxis;
using hr::AutofocusOp;
using hr::AutofocusParams;
using hr::BackprojectionBackend;
using hr::BackprojectionOp;
using hr::BackprojectionParams;
using hr::CoordinateFrame;
using hr::CoordinateFrameSpec;
using hr::CoordinateFrameType;
using hr::FormationControl;
using hr::MaxTensorExtents;
using hr::PhaseCorrectionOp;
using hr::PhaseCorrectionParams;
using hr::PhaseCorrectionSense;
using hr::SampleGeometry;
using hr::SampleGeometrySpec;
using hr::sim::sar_differential_range;
using hr::sim::sar_matched_phase;
using test::HostTensor;

// --- shared scene ------------------------------------------------------

constexpr int64_t kPulses = 64;
constexpr int64_t kBins = 64;
constexpr double kSpacing = 0.25;

double centred_origin() {
  return -std::floor(static_cast<double>(kBins) / 2.0) * kSpacing;
}

CoordinateFrame scene_frame() {
  CoordinateFrameSpec spec;
  spec.frame = CoordinateFrameType::kSceneLocal;
  return CoordinateFrame(spec);
}

std::vector<double> as_double(const HostTensor& t) {
  std::vector<double> out(static_cast<size_t>(t.size()));
  if (t.dtype.bits == 64) {
    const auto* v = t.as<double>();
    for (int64_t i = 0; i < t.size(); ++i) {
      out[static_cast<size_t>(i)] = v[i];
    }
  } else {
    const auto* v = t.as<float>();
    for (int64_t i = 0; i < t.size(); ++i) {
      out[static_cast<size_t>(i)] = static_cast<double>(v[i]);
    }
  }
  return out;
}

// The recovery acceptance number, from docs/planning/SAR/TESTING.md: "the
// corrected product peak must reach >= 0.9 of the error-free peak". Every
// assertion that a correction *worked* uses this.
constexpr double kRecoveryFraction = 0.9;
// A separate and deliberately looser statement: the *uncorrected* product must
// be visibly worse than clean. Not a contract number, and not the same claim --
// sharing one constant for both let a run that recovered 71% read as a success.
constexpr double kDefocusFraction = 0.7;

// Returns NaN on a non-finite pixel rather than skipping it. `std::max` returns
// its first argument when the second is NaN, so folding NaNs into `best` would
// report a peak of exactly 0.0 for an all-NaN image -- which *satisfies* every
// `EXPECT_LT(defocused, ...)` below instead of failing it. That is how
// a NaN product reached a green suite once already; a NaN must now fail loudly
// at the first test that looks at the image.
double image_peak(const HostTensor& image) {
  constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
  if (image.dtype.code != kDLComplex || image.dtype.bits != 64) {
    ADD_FAILURE() << "image_peak: expected complex64, got code " << int{image.dtype.code}
                  << " bits " << int{image.dtype.bits};
    return kNaN;
  }
  double best = 0.0;
  const auto* z = image.as<std::complex<float>>();
  for (int64_t i = 0; i < image.size(); ++i) {
    const double magnitude = static_cast<double>(std::abs(z[i]));
    if (!std::isfinite(magnitude)) {
      ADD_FAILURE() << "image_peak: non-finite pixel at linear index " << i;
      return kNaN;
    }
    best = std::max(best, magnitude);
  }
  return best;
}

double phase_corr(const std::vector<double>& a, const std::vector<double>& b) {
  double dot = 0.0;
  double na = 0.0;
  double nb = 0.0;
  const size_t n = std::min(a.size(), b.size());
  for (size_t i = 0; i < n; ++i) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return (na > 0.0 && nb > 0.0) ? (dot / std::sqrt(na * nb)) : 0.0;
}

// --- fixture plumbing --------------------------------------------------

// One tick's worth of everything a formation chain is driven with. The source
// co-emits the whole struct so the OnAligned Exact joins downstream see one
// capture time; two independent clock sources would never align.
struct SceneTick {
  std::shared_ptr<holoscan::Tensor> range_profiles;
  std::shared_ptr<holoscan::Tensor> positions;
  std::shared_ptr<holoscan::Tensor> reference_position;
  std::shared_ptr<holoscan::Tensor> control;
  std::shared_ptr<holoscan::Tensor> inject_phase;
  std::shared_ptr<holoscan::Tensor> correct_phase;
  // Emitting onto a port no edge consumes fails the activation with
  // kNotReady, so every field is optional and each graph populates exactly
  // what it wires.
  std::optional<uint8_t> stream_boundary;
};

struct SceneBank {
  static void reset() {
    const std::lock_guard lock(mutex);
    ticks.clear();
    next = 0;
    images.clear();
    aux_images.clear();
    phases.clear();
    raw_phases.clear();
    sharpness.clear();
    controls.clear();
    profiles.clear();
  }

  static bool have(std::size_t n_images) {
    const std::lock_guard lock(mutex);
    return images.size() >= n_images;
  }

  static inline std::mutex mutex;
  static inline std::condition_variable ready;
  static inline std::vector<SceneTick> ticks;
  static inline std::size_t next{0};
  static inline std::vector<HostTensor> images;
  static inline std::vector<HostTensor> aux_images;
  static inline std::vector<HostTensor> phases;
  static inline std::vector<HostTensor> raw_phases;
  static inline std::vector<HostTensor> sharpness;
  static inline std::vector<HostTensor> controls;
  static inline std::vector<HostTensor> profiles;
};

holoscan::expected<void, holoscan::Error> emit_copy(holoscan::Output<holoscan::Tensor>& out,
                                                    holoscan::ExecutionContext& context,
                                                    const std::shared_ptr<holoscan::Tensor>& frame,
                                                    const char* who) {
  HOLOSCAN_TRY(auto loan, out.allocate_tensor(holoscan::TensorLoanRequest{
                              .shape = frame->shape(), .dtype = frame->dtype()}));
  HOLOSCAN_TRY(auto writer, loan.write(context.cuda_stream()));
  if (writer.byte_size() > 0 &&
      cudaMemcpyAsync(writer.data(), frame->data(), writer.byte_size(), cudaMemcpyDeviceToDevice,
                      context.cuda_stream().get()) != cudaSuccess) {
    return holoscan::make_unexpected(
        holoscan::Error{holoscan::ErrorCode::kFailure, std::string(who) + ": cudaMemcpyAsync"});
  }
  HOLOSCAN_RETURN_IF_ERROR(std::move(writer).commit());
  return out.emit(std::move(loan),
                  holoscan::EmitOptions{.capture_time = context.activation_time()});
}

// Publishes one SceneTick per activation and then goes quiet. `control` and the
// two phase vectors are emitted before the trigger ports on purpose: every one
// of them lands on a `[live]` override downstream, and an override that is not
// already queued when the OnAligned trigger set completes is simply missing for
// that tick -- the activation does not wait for it.
constexpr std::size_t kSceneMaxBytes = 64ULL * 1024ULL * 1024ULL;

class SceneSourceOp final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    constexpr holoscan::TensorRepresentation device{
        .memory_kind = holoscan::MemoryKind::kCudaDevice,
        .dtype = std::nullopt,
        .rank = std::nullopt};
    spec.output(range_profiles, "range_profiles")
        .produces_tensor(hr::pooled_tensor_output(device, 4U, kSceneMaxBytes));
    spec.output(positions, "positions")
        .produces_tensor(hr::pooled_tensor_output(device, 4U, kSceneMaxBytes));
    spec.output(reference_position, "reference_position")
        .produces_tensor(hr::pooled_tensor_output(device, 4U, kSceneMaxBytes));
    spec.output(control, "control")
        .produces_tensor(hr::pooled_tensor_output(device, 4U, kSceneMaxBytes));
    spec.output(inject_phase, "inject_phase")
        .produces_tensor(hr::pooled_tensor_output(device, 4U, kSceneMaxBytes));
    spec.output(correct_phase, "correct_phase")
        .produces_tensor(hr::pooled_tensor_output(device, 4U, kSceneMaxBytes));
    spec.output(stream_boundary, "stream_boundary");
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = std::chrono::milliseconds{5}});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    SceneTick tick;
    {
      const std::lock_guard lock(SceneBank::mutex);
      if (SceneBank::next >= SceneBank::ticks.size()) {
        return {};
      }
      tick = SceneBank::ticks[SceneBank::next++];
    }
    const holoscan::EmitOptions when{.capture_time = context.activation_time()};
    if (tick.control) {
      HOLOSCAN_RETURN_IF_ERROR(emit_copy(control, context, tick.control, "scene.control"));
    }
    if (tick.inject_phase) {
      HOLOSCAN_RETURN_IF_ERROR(emit_copy(inject_phase, context, tick.inject_phase, "scene.inject"));
    }
    if (tick.correct_phase) {
      HOLOSCAN_RETURN_IF_ERROR(
          emit_copy(correct_phase, context, tick.correct_phase, "scene.correct"));
    }
    if (tick.stream_boundary.has_value()) {
      HOLOSCAN_RETURN_IF_ERROR(stream_boundary.emit(*tick.stream_boundary, when));
    }
    if (tick.positions) {
      HOLOSCAN_RETURN_IF_ERROR(emit_copy(positions, context, tick.positions, "scene.positions"));
    }
    if (tick.reference_position) {
      HOLOSCAN_RETURN_IF_ERROR(
          emit_copy(reference_position, context, tick.reference_position, "scene.reference"));
    }
    return emit_copy(range_profiles, context, tick.range_profiles, "scene.profiles");
  }

  holoscan::Output<holoscan::Tensor> range_profiles{"range_profiles"};
  holoscan::Output<holoscan::Tensor> positions{"positions"};
  holoscan::Output<holoscan::Tensor> reference_position{"reference_position"};
  holoscan::Output<holoscan::Tensor> control{"control"};
  holoscan::Output<holoscan::Tensor> inject_phase{"inject_phase"};
  holoscan::Output<holoscan::Tensor> correct_phase{"correct_phase"};
  holoscan::Output<uint8_t> stream_boundary{"stream_boundary"};
};

// Collectors. Each writes one SceneBank vector; the graph wires only the ones
// a given case inspects, and drains the rest.
template <std::vector<HostTensor>* Sink>
class CollectOp final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override { spec.input(in, "in").queue_depth(4U); }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnEach{in});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext&) override {
    HOLOSCAN_TRY(auto sample, in.receive_data());
    HostTensor host = test::to_host(sample);
    {
      const std::lock_guard lock(SceneBank::mutex);
      Sink->push_back(std::move(host));
    }
    SceneBank::ready.notify_all();
    return {};
  }

  holoscan::Input<holoscan::Tensor> in{"in"};
};

using ImageCollectOp = CollectOp<&SceneBank::images>;
using AuxImageCollectOp = CollectOp<&SceneBank::aux_images>;
using PhaseCollectOp = CollectOp<&SceneBank::phases>;
using RawPhaseCollectOp = CollectOp<&SceneBank::raw_phases>;
using SharpnessCollectOp = CollectOp<&SceneBank::sharpness>;
using ControlCollectOp = CollectOp<&SceneBank::controls>;
using ProfileCollectOp = CollectOp<&SceneBank::profiles>;

class DrainOp final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override { spec.input(in, "in").queue_depth(4U); }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnEach{in});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext&) override {
    HOLOSCAN_TRY(auto sample, in.receive_data());
    (void)sample;
    return {};
  }

  holoscan::Input<holoscan::Tensor> in{"in"};
};

void bind_output(holoscan::CompileOptions* options, const char* op, const char* port) {
  options->deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
      .operator_path = op, .output_port = port, .device = holoscan::DeviceId{0}});
}

// Run a compiled graph until `done` holds. Every collector notifies, so a case
// that reads more than one output must name them all: stopping on the first
// capture can cut the session off before a sibling output arrives.
void run_until(holoscan::Graph* graph, holoscan::CompileOptions options,
               const std::function<bool()>& done, const char* what) {
  graph->set_default_clock(graph->template add_clock<holoscan::RealtimeClock>("clock"));
  const holoscan::ExecutionPlan plan = holoscan::compile(*graph, std::move(options));
  ASSERT_TRUE(plan.ok()) << plan.json();
  holoscan::RunSession session = holoscan::run_async(plan);
  {
    std::unique_lock lock(SceneBank::mutex);
    const bool observed = SceneBank::ready.wait_for(lock, std::chrono::seconds{30}, done);
    lock.unlock();
    if (!observed) {
      session.request_stop();
      session.wait();
      FAIL() << "timed out waiting for " << what << "\n"
             << holoscan::format_diagnostics(session.diagnostics(), plan.graph());
    }
  }
  session.request_stop();
  session.wait();
  // Drop the device tensors while CUDA is still up. Left in a static bank they
  // are freed during static destruction, after the driver has gone, and the
  // process segfaults on the way out with every test reported as passing.
  {
    const std::lock_guard lock(SceneBank::mutex);
    SceneBank::ticks.clear();
  }
  ASSERT_TRUE(session.diagnostics().empty());
}

// --- transfer onto a differently formed product ------------------------

// One target, one flat aperture: the product peak is a pure coherent sum over
// the pulses, so any per-pulse phase left on them shows up as a collapsed peak.
constexpr double kFormFreq = 9.6e9;

SampleGeometry grid_geometry(int64_t nv, int64_t nu) {
  SampleGeometrySpec spec;
  spec.origin = {-2.0, -2.0, 0.0};
  spec.axis_u = {1.0, 0.0, 0.0};
  spec.axis_v = {0.0, 1.0, 0.0};
  spec.spacing = {1.0, 1.0};
  spec.counts = {nv, nu};
  return SampleGeometry(spec);
}

std::shared_ptr<holoscan::Tensor> form_profiles() {
  const std::array<double, 3> apc{0.0, 0.0, 100.0};
  const std::array<double, 3> srp{0.0, 0.0, 0.0};
  const std::array<double, 3> target{0.0, 1.0, 0.0};
  const double delta_r = sar_differential_range(apc, srp, target);
  const auto matched = sar_matched_phase(kFormFreq, delta_r);
  const std::complex<double> amp{matched.real(), -matched.imag()};
  const int64_t bin = static_cast<int64_t>(std::llround((delta_r - centred_origin()) / kSpacing));
  std::vector<std::complex<double>> cube(static_cast<size_t>(kPulses * kBins), {0.0, 0.0});
  if (bin >= 0 && bin < kBins) {
    for (int64_t p = 0; p < kPulses; ++p) {
      cube[static_cast<size_t>(p * kBins + bin)] = amp;
    }
  }
  return test::complex_cube_to_device<float>(cube, {kPulses, kBins});
}

std::shared_ptr<holoscan::Tensor> xyz_tensor(const std::array<double, 3>& point) {
  std::vector<double> host(static_cast<size_t>(kPulses * 3));
  for (int64_t p = 0; p < kPulses; ++p) {
    host[static_cast<size_t>(p * 3 + 0)] = point[0];
    host[static_cast<size_t>(p * 3 + 1)] = point[1];
    host[static_cast<size_t>(p * 3 + 2)] = point[2];
  }
  return test::real_cube_to_device<float>(host, {kPulses, 3});
}

std::shared_ptr<holoscan::Tensor> control_tensor(FormationControl value) {
  return test::to_device_uint8_tensor({static_cast<uint8_t>(value)}, {1});
}

MaxTensorExtents form_extents(int64_t nv, int64_t nu) {
  MaxTensorExtents extents;
  extents.max_pulses = kPulses;
  extents.max_range_bins = kBins;
  extents.max_samples_per_pulse = kBins;
  extents.max_image_v = nv;
  extents.max_image_u = nu;
  return extents;
}

BackprojectionParams form_params(const SampleGeometry& geometry) {
  BackprojectionParams params;
  params.coordinate_frame = scene_frame();
  params.sample_geometry = geometry;
  params.range_bin_spacing = kSpacing;
  params.range_bin_origin = centred_origin();
  params.reference_frequency = kFormFreq;
  return params;
}

// Form one product image, optionally with `inject` added to the pulses by one
// PhaseCorrectionOp and `correct` subtracted back off by a second. Two
// instances in series is what the real chain does -- the error arrives on the
// pulses and the estimate is removed from them -- and it is the part no
// single-operator test can cover.
HostTensor run_form(const std::vector<double>& inject, const std::vector<double>& correct,
                    const SampleGeometry& geometry) {
  SceneBank::reset();
  SceneTick tick;
  tick.range_profiles = form_profiles();
  tick.positions = xyz_tensor({0.0, 0.0, 100.0});
  tick.reference_position = xyz_tensor({0.0, 0.0, 0.0});
  tick.control = control_tensor(FormationControl::kEmitAndReset);
  if (!inject.empty()) {
    tick.inject_phase = test::real_cube_to_device<float>(inject, {kPulses});
  }
  if (!correct.empty()) {
    tick.correct_phase = test::real_cube_to_device<float>(correct, {kPulses});
  }
  {
    const std::lock_guard lock(SceneBank::mutex);
    SceneBank::ticks.push_back(std::move(tick));
  }

  holoscan::Graph graph{"sar-form"};
  const auto source = graph.op<SceneSourceOp>("source");
  auto bp =
      graph.op<BackprojectionOp<float>>("bp", form_extents(geometry.num_v(), geometry.num_u()));
  bp.set_params(form_params(geometry));
  const auto image_sink = graph.op<ImageCollectOp>("image_sink");

  holoscan::CompileOptions options;
  for (const char* port : {"range_profiles", "positions", "reference_position", "control",
                           "inject_phase", "correct_phase"}) {
    bind_output(&options, "source", port);
  }
  bind_output(&options, "bp", "image");

  const holoscan::ConnectionOptions queue{.queue_depth = 4U};
  const holoscan::Output<holoscan::Tensor>* profiles = &source->range_profiles;
  if (!inject.empty()) {
    auto add =
        graph.op<PhaseCorrectionOp<float>>("add", form_extents(geometry.num_v(), geometry.num_u()));
    add.set_params(PhaseCorrectionParams{.sense = PhaseCorrectionSense::kAdd});
    bind_output(&options, "add", "range_profiles_out");
    graph.add_flow(*profiles, add->range_profiles_in, queue);
    graph.add_flow(source->inject_phase, add->phase_correction, queue);
    profiles = &add->range_profiles_out;
    if (!correct.empty()) {
      auto sub = graph.op<PhaseCorrectionOp<float>>(
          "sub", form_extents(geometry.num_v(), geometry.num_u()));
      sub.set_params(PhaseCorrectionParams{.sense = PhaseCorrectionSense::kSubtract});
      bind_output(&options, "sub", "range_profiles_out");
      graph.add_flow(*profiles, sub->range_profiles_in, queue);
      graph.add_flow(source->correct_phase, sub->phase_correction, queue);
      profiles = &sub->range_profiles_out;
    }
  }
  graph.add_flow(*profiles, bp->range_profiles, queue);
  graph.add_flow(source->positions, bp->positions, queue);
  graph.add_flow(source->reference_position, bp->reference_position, queue);
  graph.add_flow(source->control, bp->control, queue);
  graph.add_flow(bp->image, image_sink->in, queue);

  run_until(
      &graph, std::move(options),
      [] {
        return !SceneBank::images.empty();
      },
      "a product image");
  if (SceneBank::images.empty()) {
    throw std::runtime_error("run_form: no product image");
  }
  return SceneBank::images.front();
}

// --- the estimate under test -------------------------------------------

struct Estimate {
  HostTensor phase;
  HostTensor image;
  HostTensor sharpness;
};

// source -> AutofocusOp -> collectors. Every declared AutofocusOp output needs
// a real consumer: an unwired optional output makes the operator skip the work
// behind it for the whole run.
Estimate run_estimate(const std::vector<std::complex<float>>& img,
                      const std::vector<int64_t>& shape, AutofocusAzimuthAxis axis,
                      int64_t min_scatterers, double window_fraction = 0.5,
                      int64_t iterations = 4) {
  SceneBank::reset();
  SceneTick tick;
  tick.range_profiles = test::to_device_tensor(img, shape, hr::complex_dtype<float>());
  {
    const std::lock_guard lock(SceneBank::mutex);
    SceneBank::ticks.push_back(std::move(tick));
  }

  MaxTensorExtents extents = form_extents(shape[0], shape[1]);
  extents.max_pulses = std::max(shape[0], shape[1]);

  holoscan::Graph graph{"sar-estimate"};
  const auto source = graph.op<SceneSourceOp>("source");
  auto af = graph.op<AutofocusOp<float>>("af", extents);
  AutofocusParams params;
  params.azimuth_axis = axis;
  params.min_scatterers = min_scatterers;
  params.window_fraction = window_fraction;
  params.num_iterations = iterations;
  af.set_params(params);
  const auto image_sink = graph.op<ImageCollectOp>("image_sink");
  const auto phase_sink = graph.op<PhaseCollectOp>("phase_sink");
  const auto sharpness_sink = graph.op<SharpnessCollectOp>("sharpness_sink");

  holoscan::CompileOptions options;
  for (const char* port : {"range_profiles", "positions", "reference_position", "control",
                           "inject_phase", "correct_phase"}) {
    bind_output(&options, "source", port);
  }
  for (const char* port :
       {"image_out", "phase_correction", "sharpness", "termination", "iterations"}) {
    bind_output(&options, "af", port);
  }

  const holoscan::ConnectionOptions queue{.queue_depth = 4U};
  graph.add_flow(source->range_profiles, af->image, queue);
  graph.add_flow(af->phase_correction, phase_sink->in, queue);
  graph.add_flow(af->sharpness, sharpness_sink->in, queue);
  graph.add_flow(af->image_out, image_sink->in, queue);

  run_until(
      &graph, std::move(options),
      [] {
        return !SceneBank::images.empty() && !SceneBank::phases.empty() &&
               !SceneBank::sharpness.empty();
      },
      "the autofocus image, phase, and sharpness");
  if (SceneBank::phases.empty()) {
    throw std::runtime_error("run_estimate: no phase_correction");
  }
  return Estimate{
      .phase = SceneBank::phases.front(),
      .image = SceneBank::images.front(),
      .sharpness = SceneBank::sharpness.empty() ? HostTensor{} : SceneBank::sharpness.front()};
}

TEST(SarAutofocusGraph, TransferOntoDifferentGeometry) {
  // A stronger quadratic than the operator suite injects (5 rad at the edges
  // against 2.5): after PGA drops linear and constant, a 2.5 rad residual
  // barely moves the coherent sum on this single-position aperture.
  std::vector<double> phi(static_cast<size_t>(kPulses));
  const double mid = 0.5 * static_cast<double>(kPulses - 1);
  for (int64_t k = 0; k < kPulses; ++k) {
    const double x = (static_cast<double>(k) - mid) / std::max(mid, 1.0);
    phi[static_cast<size_t>(k)] = 5.0 * x * x + 0.12 * static_cast<double>(k) + 0.4;
  }
  // The estimation image is range-decimated (32 bins) against the 64-bin
  // product formation, which is the recommended auxiliary pattern.
  constexpr int64_t kEstRange = 32;
  // `phi` is the pulse-domain error, so the image carries it in bin order.
  const auto img = hr::sar_test::make_image<float>(to_bin_order(phi), 24, true, kPulses, kEstRange);
  const auto estimate = run_estimate(img, {kEstRange, kPulses}, AutofocusAzimuthAxis::kU, 16);
  auto est = as_double(estimate.phase);
  ASSERT_EQ(static_cast<int64_t>(est.size()), kPulses);

  // PGA cannot observe linear and constant terms, so forming with the raw
  // injected phi would leave those on the pulses and the coherent sum would
  // stay collapsed. Transfer the recoverable, detrended error -- which is what
  // PhaseCorrectionOp carries in the real chain.
  auto injected = phi;
  remove_affine(&injected);
  remove_affine(&est);

  const auto grid = grid_geometry(8, 8);
  const double clean = image_peak(run_form({}, {}, grid));
  const double defocused = image_peak(run_form(injected, {}, grid));
  const double restored = image_peak(run_form(injected, est, grid));
  ASSERT_GT(clean, 1.0);
  EXPECT_LT(defocused, kDefocusFraction * clean) << "clean=" << clean << " defocused=" << defocused;
  EXPECT_GT(restored, kRecoveryFraction * clean)
      << "clean=" << clean << " defocused=" << defocused << " restored=" << restored
      << " corr=" << phase_corr(est, injected);
}

TEST(SarAutofocusGraph, GroundGridIsAHazard) {
  // Form onto a square ground grid, then estimate on that image. The estimate
  // follows the image's azimuth axis, so it comes back 16 long against a
  // 64-pulse aperture: a plausible vector of the wrong length and the wrong
  // values rather than a rejection. Nothing downstream can detect this, which
  // is why the auxiliary lattice has to be conjugate to the aperture.
  std::vector<double> phi(static_cast<size_t>(kPulses));
  const double mid = 0.5 * static_cast<double>(kPulses - 1);
  for (int64_t k = 0; k < kPulses; ++k) {
    const double x = (static_cast<double>(k) - mid) / std::max(mid, 1.0);
    phi[static_cast<size_t>(k)] = 2.5 * x * x + 0.12 * static_cast<double>(k) + 0.4;
  }
  const HostTensor formed = run_form(phi, {}, grid_geometry(16, 16));
  ASSERT_EQ(formed.shape, (std::vector<int64_t>{16, 16}));
  std::vector<std::complex<float>> host(static_cast<size_t>(formed.size()));
  const auto* z = formed.as<std::complex<float>>();
  for (int64_t i = 0; i < formed.size(); ++i) {
    host[static_cast<size_t>(i)] = z[i];
  }

  // The floor has to be 1 for the hazard to appear at all: a 16x16 grid yields
  // a single strong scatterer, so the floor of four this test used to pass
  // made the operator refuse, and a refusal is the one outcome this test is
  // not about -- it produces a zero estimate, which scores a perfect
  // correlation and makes the whole test vacuous.
  const auto estimate = run_estimate(host, formed.shape, AutofocusAzimuthAxis::kV, 1, 0.5, 4);
  auto est = as_double(estimate.phase);
  EXPECT_EQ(static_cast<int64_t>(est.size()), 16);
  for (size_t i = 0; i < est.size(); ++i) {
    ASSERT_TRUE(std::isfinite(est[i])) << "non-finite estimate at " << i;
  }

  // TESTING.md requires showing the recovered phase does *not* match the
  // injected one. Correlation alone cannot carry that claim: `phase_corr`
  // returns 0.0 when either input has no energy, and `remove_affine` zeroes an
  // affine estimate, so a run that produced nothing at all scores a perfect
  // 0.0. Establish that there is a real estimate to judge before judging it.
  auto truth = phi;
  remove_affine(&est);
  remove_affine(&truth);

  double est_rms = 0.0;
  for (const double v : est) {
    est_rms += v * v;
  }
  est_rms = std::sqrt(est_rms / static_cast<double>(est.size()));
  ASSERT_GT(est_rms, 1e-6) << "the estimator returned a degenerate vector (rms=" << est_rms
                           << "); the correlation below cannot tell a wrong estimate from no "
                              "estimate, so this test would pass vacuously";

  // There is no correct way to line a 16-long estimate up against a 64-pulse
  // truth, which is the hazard; comparing the leading samples of each after
  // removing the affine part PGA cannot see is enough to show they are
  // unrelated.
  const double corr = phase_corr(est, truth);
  EXPECT_LT(std::abs(corr), 0.5) << "a ground-grid estimate tracked the injected phase (corr="
                                 << corr << "); the hazard this test documents may no longer hold";
}

// --- retain, reform, estimate on the auxiliary, correct the product ----

// A broadside strip with the auxiliary lattice conjugate to the aperture: the
// u axis runs along-track and the v axis along the slant range. 1 GHz puts the
// azimuth resolution on the order of the 0.5 m conjugate pixel, so a 5 rad
// quadratic is a several-pixel smear the LUE can see. At 9.6 GHz the same
// error is sub-pixel and PGA correctly reports nothing to fix.
constexpr int64_t kFigBatch = 32;
constexpr int64_t kFigTargets = 12;
constexpr int64_t kFigAuxV = 32;
constexpr double kFigDy = 0.5;
constexpr double kFigAlt = 100.0;
constexpr double kFigGndX = 50.0;
constexpr double kFigFreq = 1.0e9;

std::array<double, 3> fig_apc(int64_t pulse) {
  const double y = (static_cast<double>(pulse) - 0.5 * static_cast<double>(kPulses - 1)) * kFigDy;
  return {0.0, y, kFigAlt};
}

std::array<double, 3> fig_srp() {
  return {kFigGndX, 0.0, 0.0};
}

std::array<double, 3> fig_target(int64_t i) {
  return {kFigGndX + (static_cast<double>(i) - 5.5) * 0.3, 0.0, 0.0};
}

std::vector<double> fig_phi() {
  std::vector<double> phi(static_cast<size_t>(kPulses));
  const double mid = 0.5 * static_cast<double>(kPulses - 1);
  for (int64_t k = 0; k < kPulses; ++k) {
    const double x = (static_cast<double>(k) - mid) / std::max(mid, 1.0);
    phi[static_cast<size_t>(k)] = 5.0 * x * x;
  }
  return phi;
}

SampleGeometry fig_aux_geometry(int64_t num_u) {
  const double slant = std::sqrt(kFigGndX * kFigGndX + kFigAlt * kFigAlt);
  const double along_track = static_cast<double>(kPulses - 1) * kFigDy;
  SampleGeometrySpec spec;
  spec.axis_u = {0.0, 1.0, 0.0};
  spec.axis_v = {kFigGndX / slant, 0.0, -kFigAlt / slant};
  spec.spacing = {kSpacing, along_track / static_cast<double>(num_u - 1)};
  spec.counts = {kFigAuxV, num_u};
  const double half_v = 0.5 * static_cast<double>(kFigAuxV);
  const double half_u = 0.5 * static_cast<double>(num_u);
  spec.origin = {kFigGndX - half_v * spec.spacing[0] * spec.axis_v[0], -half_u * spec.spacing[1],
                 -half_v * spec.spacing[0] * spec.axis_v[2]};
  return SampleGeometry(spec);
}

SampleGeometry fig_product_geometry() {
  SampleGeometrySpec spec;
  spec.origin = {42.0, -8.0, 0.0};
  spec.axis_u = {1.0, 0.0, 0.0};
  spec.axis_v = {0.0, 1.0, 0.0};
  spec.spacing = {1.0, 1.0};
  spec.counts = {16, 16};
  return SampleGeometry(spec);
}

std::shared_ptr<holoscan::Tensor> fig_profiles(int64_t p0, int64_t p1, bool inject) {
  const auto srp = fig_srp();
  const auto phi = fig_phi();
  const double origin = centred_origin();
  std::vector<std::complex<double>> cube(static_cast<size_t>((p1 - p0) * kBins), {0.0, 0.0});
  for (int64_t p = p0; p < p1; ++p) {
    const auto apc = fig_apc(p);
    const double err = inject ? phi[static_cast<size_t>(p)] : 0.0;
    const std::complex<double> phasor(std::cos(err), std::sin(err));
    for (int64_t t = 0; t < kFigTargets; ++t) {
      const double delta_r = sar_differential_range(apc, srp, fig_target(t));
      const auto matched = sar_matched_phase(kFigFreq, delta_r);
      const std::complex<double> amp{matched.real(), -matched.imag()};
      const int64_t bin = static_cast<int64_t>(std::llround((delta_r - origin) / kSpacing));
      if (bin >= 0 && bin < kBins) {
        cube[static_cast<size_t>((p - p0) * kBins + bin)] += amp * phasor;
      }
    }
  }
  return test::complex_cube_to_device<float>(cube, {p1 - p0, kBins});
}

std::shared_ptr<holoscan::Tensor> fig_xyz(int64_t p0, int64_t p1, bool apc) {
  const int64_t n = p1 - p0;
  std::vector<double> host(static_cast<size_t>(n * 3));
  for (int64_t i = 0; i < n; ++i) {
    const auto point = apc ? fig_apc(p0 + i) : fig_srp();
    host[static_cast<size_t>(i * 3 + 0)] = point[0];
    host[static_cast<size_t>(i * 3 + 1)] = point[1];
    host[static_cast<size_t>(i * 3 + 2)] = point[2];
  }
  return test::real_cube_to_device<float>(host, {n, 3});
}

BackprojectionParams fig_params(const SampleGeometry& geometry) {
  BackprojectionParams params;
  params.coordinate_frame = scene_frame();
  params.sample_geometry = geometry;
  params.range_bin_spacing = kSpacing;
  params.range_bin_origin = centred_origin();
  params.reference_frequency = kFigFreq;
  // The reference backend keeps the two formations comparable at float input.
  params.backend = BackprojectionBackend::kNvradarFp64Reference;
  return params;
}

MaxTensorExtents fig_extents(int64_t aux_num_u) {
  MaxTensorExtents extents;
  extents.max_pulses = kPulses;
  extents.max_range_bins = kBins;
  extents.max_samples_per_pulse = kBins;
  extents.max_image_v = std::max<int64_t>(kFigAuxV, 16);
  extents.max_image_u = std::max<int64_t>(aux_num_u, 16);
  return extents;
}

struct Figure2Result {
  double peak{0.0};
  HostTensor aux_image;
  HostTensor phase;
  HostTensor raw_phase;
  HostTensor sharpness;
  HostTensor control;
  HostTensor replayed_profiles;
};

// `inject` puts the quadratic on the pulses; `autofocus` splices the auxiliary
// formation, the estimator, and the resampler in front of PhaseCorrectionOp;
// `apply` hands PhaseCorrectionOp a fixed vector instead, which is the oracle
// path. The aperture is admitted over two ticks and replayed once, at the
// end-of-stream boundary, which is what makes this a retain/reform test rather
// than a formation test.
Figure2Result run_fig2(bool inject, bool autofocus, const std::vector<double>& apply,
                       int64_t aux_num_u = kPulses, bool always_resample = false,
                       double window_fraction = -1.0) {
  SceneBank::reset();
  {
    const std::lock_guard lock(SceneBank::mutex);
    for (int tick_index = 0; tick_index < 2; ++tick_index) {
      const int64_t p0 = tick_index == 0 ? 0 : kFigBatch;
      const int64_t p1 = tick_index == 0 ? kFigBatch : kPulses;
      SceneTick tick;
      tick.range_profiles = fig_profiles(p0, p1, inject);
      tick.positions = fig_xyz(p0, p1, true);
      tick.reference_position = fig_xyz(p0, p1, false);
      tick.stream_boundary = static_cast<uint8_t>(
          tick_index == 1 ? hr::SarStreamBoundary::kEndOfStream : hr::SarStreamBoundary::kNone);
      if (!apply.empty()) {
        tick.correct_phase = test::real_cube_to_device<float>(apply, {kPulses});
      }
      SceneBank::ticks.push_back(std::move(tick));
    }
  }

  const MaxTensorExtents extents = fig_extents(aux_num_u);
  holoscan::Graph graph{"sar-figure2"};
  const auto source = graph.op<SceneSourceOp>("source");
  // Uniform `<float>` rather than the applications' `<float, double>`, and
  // deliberately so. The pair only varies the three geometry-carrying
  // operators; `AutofocusOp` and `PhaseCorrectionOp` -- the whole of what this
  // file tests -- take `<T>` alone and have no TGeom to get wrong. The mixed
  // pair is covered per-operator in `test_backprojection`, `test_aperture_
  // manager` and `test_cphd_reader`, and at graph level by the autofocus
  // application fixtures, whose driver defaults to `<float, double>`.
  // Templating this file on TGeom would add a scene source in double to
  // re-cover ground those tests already hold.
  auto manager = graph.op<ApertureManagerOp<float>>("manager", extents);
  manager.set_params(ApertureManagerParams{.max_aperture_pulses = kPulses,
                                           .emit_interval_pulses = 0,
                                           .emit_mode = ApertureEmitMode::kReform});
  auto product_bp = graph.op<BackprojectionOp<float>>("product_bp", extents);
  product_bp.set_params(fig_params(fig_product_geometry()));
  const auto image_sink = graph.op<ImageCollectOp>("image_sink");
  const auto control_sink = graph.op<ControlCollectOp>("control_sink");
  const auto profile_sink = graph.op<ProfileCollectOp>("profile_sink");

  holoscan::CompileOptions options;
  for (const char* port : {"range_profiles", "positions", "reference_position", "control",
                           "inject_phase", "correct_phase"}) {
    bind_output(&options, "source", port);
  }
  for (const char* port : {"range_profiles_out", "positions_out", "reference_position_out",
                           "weight", "sample_points_out", "control_out"}) {
    bind_output(&options, "manager", port);
  }
  bind_output(&options, "product_bp", "image");

  const holoscan::ConnectionOptions queue{.queue_depth = 4U};
  graph.add_flow(source->range_profiles, manager->range_profiles, queue);
  graph.add_flow(source->positions, manager->positions, queue);
  graph.add_flow(source->reference_position, manager->reference_position, queue);
  graph.add_flow(source->stream_boundary, manager->stream_boundary, queue);

  // PhaseCorrectionOp joins its two inputs with OnAligned, so `phase_correction`
  // is a required trigger port rather than an override: a run with nothing to
  // apply has to leave the operator out instead of wiring it half-way.
  bool wants_aux = autofocus;
  std::optional<holoscan::OperatorHandle<PhaseCorrectionOp<float>>> pc;
  if (autofocus || !apply.empty()) {
    pc.emplace(graph.op<PhaseCorrectionOp<float>>("pc", extents));
    pc->set_params(PhaseCorrectionParams{.sense = PhaseCorrectionSense::kSubtract});
    bind_output(&options, "pc", "range_profiles_out");
  }
  if (autofocus) {
    auto aux_bp = graph.op<BackprojectionOp<float>>("aux_bp", extents);
    aux_bp.set_params(fig_params(fig_aux_geometry(aux_num_u)));
    bind_output(&options, "aux_bp", "image");
    AutofocusParams af_params;
    af_params.azimuth_axis = AutofocusAzimuthAxis::kU;
    af_params.min_scatterers = 1;
    if (window_fraction > 0.0) {
      af_params.window_fraction = window_fraction;
    }
    auto af = graph.op<AutofocusOp<float>>("af", extents);
    af.set_params(af_params);
    for (const char* port :
         {"image_out", "phase_correction", "sharpness", "termination", "iterations"}) {
      bind_output(&options, "af", port);
    }
    const auto aux_sink = graph.op<AuxImageCollectOp>("aux_sink");
    const auto sharpness_sink = graph.op<SharpnessCollectOp>("sharpness_sink");

    graph.add_flow(manager->range_profiles_out, aux_bp->range_profiles, queue);
    graph.add_flow(manager->positions_out, aux_bp->positions, queue);
    graph.add_flow(manager->reference_position_out, aux_bp->reference_position, queue);
    graph.add_flow(manager->control_out, aux_bp->control, queue);
    graph.add_flow(aux_bp->image, af->image, queue);
    graph.add_flow(af->image_out, aux_sink->in, queue);
    graph.add_flow(af->sharpness, sharpness_sink->in, queue);
    // The pre-resample estimate as well as the post: the mapping case needs to
    // see the azimuth-indexed vector the resampler was handed.
    const auto raw_phase_sink = graph.op<RawPhaseCollectOp>("raw_phase_sink");
    graph.add_flow(af->phase_correction, raw_phase_sink->in, queue);
    const auto phase_sink = graph.op<PhaseCollectOp>("phase_sink");
    if (aux_num_u != kPulses || always_resample) {
      auto resample = graph.op<holoscan::radar::app::sar::PhaseResampleOp>("resample", kPulses);
      bind_output(&options, "resample", "phase_correction_out");
      graph.add_flow(af->phase_correction, resample->phase_correction, queue);
      graph.add_flow(resample->phase_correction_out, (*pc)->phase_correction, queue);
      graph.add_flow(resample->phase_correction_out, phase_sink->in, queue);
    } else {
      graph.add_flow(af->phase_correction, (*pc)->phase_correction, queue);
      graph.add_flow(af->phase_correction, phase_sink->in, queue);
    }
  } else if (!apply.empty()) {
    graph.add_flow(source->correct_phase, (*pc)->phase_correction, queue);
  }

  if (pc.has_value()) {
    graph.add_flow(manager->range_profiles_out, (*pc)->range_profiles_in, queue);
    graph.add_flow((*pc)->range_profiles_out, product_bp->range_profiles, queue);
  } else {
    graph.add_flow(manager->range_profiles_out, product_bp->range_profiles, queue);
  }
  graph.add_flow(manager->positions_out, product_bp->positions, queue);
  graph.add_flow(manager->reference_position_out, product_bp->reference_position, queue);
  graph.add_flow(manager->control_out, product_bp->control, queue);
  graph.add_flow(manager->control_out, control_sink->in, queue);
  graph.add_flow(manager->range_profiles_out, profile_sink->in, queue);
  graph.add_flow(product_bp->image, image_sink->in, queue);

  run_until(
      &graph, std::move(options),
      // Name every collector a caller reads. `control_sink` and
      // `profile_sink` are wired unconditionally and `raw_phase_sink`
      // alongside the aux chain, and tests assert on all three -- waiting
      // only for the product image lets the run return before they have
      // landed, which is a flake under load rather than a failure.
      [wants_aux] {
        return !SceneBank::images.empty() && !SceneBank::controls.empty() &&
               !SceneBank::profiles.empty() &&
               (!wants_aux || (!SceneBank::aux_images.empty() && !SceneBank::phases.empty() &&
                               !SceneBank::raw_phases.empty() && !SceneBank::sharpness.empty()));
      },
      "the product image");

  Figure2Result result;
  if (SceneBank::images.empty()) {
    throw std::runtime_error("run_fig2: no product image");
  }
  result.peak = image_peak(SceneBank::images.front());
  if (!SceneBank::aux_images.empty()) {
    result.aux_image = SceneBank::aux_images.front();
  }
  if (!SceneBank::phases.empty()) {
    result.phase = SceneBank::phases.front();
  }
  if (!SceneBank::raw_phases.empty()) {
    result.raw_phase = SceneBank::raw_phases.front();
  }
  if (!SceneBank::sharpness.empty()) {
    result.sharpness = SceneBank::sharpness.front();
  }
  if (!SceneBank::controls.empty()) {
    result.control = SceneBank::controls.front();
  }
  if (!SceneBank::profiles.empty()) {
    result.replayed_profiles = SceneBank::profiles.front();
  }
  return result;
}

TEST(SarAutofocusGraph, Figure2RetainAndReform) {
  const double clean = run_fig2(false, false, {}).peak;
  const double defocused = run_fig2(true, false, {}).peak;
  const double known = run_fig2(true, false, fig_phi()).peak;
  const auto restored = run_fig2(true, true, {});
  ASSERT_GT(clean, 1.0);

  ASSERT_FALSE(restored.phase.data.empty());
  ASSERT_FALSE(restored.sharpness.data.empty());
  ASSERT_FALSE(restored.aux_image.data.empty());
  auto truth = fig_phi();
  auto est = as_double(restored.phase);
  remove_affine(&truth);
  remove_affine(&est);

  EXPECT_LT(defocused, kDefocusFraction * clean) << "clean=" << clean << " defocused=" << defocused;
  EXPECT_GT(known, kRecoveryFraction * clean)
      << "the known-phase splice failed, so the correction path itself is broken: clean=" << clean
      << " known=" << known;
  EXPECT_GT(restored.peak, kRecoveryFraction * clean)
      << "clean=" << clean << " defocused=" << defocused << " restored=" << restored.peak
      << " known=" << known << " corr=" << phase_corr(est, truth);

  // The estimate itself, not just the peak it produced. `truth` and `est`
  // were computed above and then used only inside a failure message, so the
  // residual TESTING.md asks this case to report was never actually checked:
  // the whole numeric claim rested on the product peak.
  ASSERT_EQ(est.size(), truth.size());
  EXPECT_GT(phase_corr(est, truth), 0.9)
      << "the recovered phase does not track the injected one (corr=" << phase_corr(est, truth)
      << ")";

  // Reform holds the aperture until the boundary: two admission ticks, one
  // replay of all 64 pulses, and nothing in between.
  ASSERT_FALSE(restored.replayed_profiles.shape.empty());
  EXPECT_EQ(restored.replayed_profiles.shape[0], kPulses);
  EXPECT_EQ(SceneBank::profiles.size(), 1U)
      << "a pulse batch left the manager outside the boundary replay";
  ASSERT_FALSE(restored.control.data.empty());
  EXPECT_EQ(restored.control.as<uint8_t>()[0],
            static_cast<uint8_t>(FormationControl::kResetThenEmit));
}

TEST(SarAutofocusGraph, Figure2OversampledAuxTransfer) {
  // The same conjugate auxiliary with twice the along-track samples, so the
  // estimate is pulse-ordered at a length the pulses do not share and
  // PhaseResampleOp has to map it down. The 1:1 case never exercises that map.
  constexpr int64_t kOverU = 2 * kPulses;
  const double clean = run_fig2(false, false, {}).peak;
  const double defocused = run_fig2(true, false, {}).peak;
  const auto restored = run_fig2(true, true, {}, kOverU);
  ASSERT_GT(clean, 1.0);

  ASSERT_GE(restored.aux_image.shape.size(), 2U);
  EXPECT_EQ(restored.aux_image.shape[restored.aux_image.shape.size() - 2], kFigAuxV);
  EXPECT_EQ(restored.aux_image.shape.back(), kOverU);
  const auto est = as_double(restored.phase);
  EXPECT_EQ(static_cast<int64_t>(est.size()), kPulses)
      << "the estimate reached PhaseCorrectionOp still azimuth-indexed";

  EXPECT_LT(defocused, kDefocusFraction * clean) << "clean=" << clean << " defocused=" << defocused;
  EXPECT_GT(restored.peak, kRecoveryFraction * clean)
      << "clean=" << clean << " defocused=" << defocused << " restored=" << restored.peak;
}

// --- which mapping of an oversampled estimate reaches the pulses ---------

std::vector<double> map_linear(const std::vector<double>& in, int64_t n_out) {
  const int64_t n_in = static_cast<int64_t>(in.size());
  if (n_in == n_out) {
    return in;
  }
  const double scale = static_cast<double>(n_in - 1) / static_cast<double>(n_out - 1);
  const double tmax = static_cast<double>(n_in - 1);
  std::vector<double> out(static_cast<size_t>(n_out));
  for (int64_t p = 0; p < n_out; ++p) {
    const double t = std::min(tmax, std::max(0.0, static_cast<double>(p) * scale));
    const int64_t i0 = static_cast<int64_t>(std::floor(t));
    const int64_t i1 = std::min(i0 + 1, n_in - 1);
    const double f = t - static_cast<double>(i0);
    const double a = in[static_cast<size_t>(i0)];
    out[static_cast<size_t>(p)] = a + f * (in[static_cast<size_t>(i1)] - a);
  }
  return out;
}

// Take the low-frequency bins from each end, which is where an oversampled
// auxiliary puts the slow-time history, and drop the empty middle.
std::vector<double> map_low_freq(const std::vector<double>& in, int64_t n_out) {
  const int64_t n_in = static_cast<int64_t>(in.size());
  const int64_t half = n_out / 2;
  const int64_t tail = n_out - half;
  std::vector<double> out(static_cast<size_t>(n_out), 0.0);
  for (int64_t i = 0; i < half; ++i) {
    out[static_cast<size_t>(i)] = in[static_cast<size_t>(i)];
  }
  for (int64_t i = 0; i < tail; ++i) {
    out[static_cast<size_t>(half + i)] = in[static_cast<size_t>(n_in - tail + i)];
  }
  return out;
}

std::vector<double> map_center(const std::vector<double>& in, int64_t n_out) {
  const int64_t start = (static_cast<int64_t>(in.size()) - n_out) / 2;
  return {in.begin() + static_cast<std::ptrdiff_t>(start),
          in.begin() + static_cast<std::ptrdiff_t>(start + n_out)};
}

std::vector<double> map_decimate(const std::vector<double>& in, int64_t n_out) {
  const int64_t n_in = static_cast<int64_t>(in.size());
  std::vector<double> out(static_cast<size_t>(n_out));
  for (int64_t i = 0; i < n_out; ++i) {
    const double t =
        static_cast<double>(i) * static_cast<double>(n_in - 1) / static_cast<double>(n_out - 1);
    out[static_cast<size_t>(i)] =
        in[static_cast<size_t>(std::min<int64_t>(std::llround(t), n_in - 1))];
  }
  return out;
}

TEST(SarAutofocusGraph, Figure2OversampledDiagnostics) {
  // The shipped resampler is one choice out of a family of plausible maps from
  // an oversampled azimuth-bin estimate onto pulses, and several of them
  // produce a vector that looks right and forms a worse image. This compares
  // the candidates by what they actually recover, so a change to the mapping
  // has to beat the alternatives rather than merely run.
  constexpr int64_t kOverU = 2 * kPulses;
  const double clean = run_fig2(false, false, {}).peak;
  const double gate = kRecoveryFraction * clean;
  ASSERT_GT(clean, 1.0);
  const auto leftover = fig_phi();

  const auto oversampled = run_fig2(true, true, {}, kOverU);
  ASSERT_FALSE(oversampled.raw_phase.data.empty());
  const auto azimuth = as_double(oversampled.raw_phase);
  ASSERT_EQ(static_cast<int64_t>(azimuth.size()), kOverU);

  // The oracle: hand PhaseCorrectionOp the exact injected error. If this does
  // not recover the peak, the correction path is broken and nothing the
  // estimator produces could be judged.
  const double oracle = run_fig2(true, false, leftover).peak;

  auto negate = [](std::vector<double> v) {
    for (double& p : v) {
      p = -p;
    }
    return v;
  };
  auto reversed = azimuth;
  std::reverse(reversed.begin(), reversed.end());

  struct Candidate {
    const char* name;
    std::vector<double> phi;
  };
  // `center` is what PhaseResampleOp does. `low_freq` is what it used to do,
  // which is the right extraction for a bin-ordered vector and the wrong one
  // now; the negations are there because a sign error is the cheapest way to
  // get a plausible estimate that defocuses, and `linear` and `decimate`
  // stretch the whole azimuth extent across the pulses instead of taking the
  // part of it the aperture actually occupies.
  const std::vector<Candidate> candidates{
      {"center", map_center(azimuth, kPulses)},
      {"center_neg", negate(map_center(azimuth, kPulses))},
      {"low_freq", map_low_freq(azimuth, kPulses)},
      {"low_freq_neg", negate(map_low_freq(azimuth, kPulses))},
      {"linear", map_linear(azimuth, kPulses)},
      {"linear_neg", negate(map_linear(azimuth, kPulses))},
      {"decimate", map_decimate(azimuth, kPulses)},
      {"decimate_neg", negate(map_decimate(azimuth, kPulses))},
      {"linear_rev", map_linear(reversed, kPulses)},
      {"linear_rev_neg", negate(map_linear(reversed, kPulses))},
  };

  std::ostringstream report;
  report << "clean=" << clean << " gate=" << gate << " oracle=" << oracle
         << " shipped_resample=" << oversampled.peak << "\n";
  double best = 0.0;
  std::string best_name = "none";
  for (const auto& candidate : candidates) {
    const double peak = run_fig2(true, false, candidate.phi).peak;
    report << "  " << candidate.name << " peak=" << peak
           << " corr=" << phase_corr(candidate.phi, leftover) << (peak > gate ? " PASS" : " fail")
           << "\n";
    if (peak > best) {
      best = peak;
      best_name = candidate.name;
    }
  }
  report << "best=" << best_name << " peak=" << best << "\n";

  EXPECT_GT(oracle, gate) << "the oracle apply is broken\n" << report.str();
  EXPECT_EQ(best_name, "center") << "a different map of the oversampled estimate now beats the "
                                    "one PhaseResampleOp uses\n"
                                 << report.str();
  // The shipped path has to match the best candidate applied by hand, which is
  // what proves the estimate survives the resampler rather than merely that
  // some transform of it would have worked.
  EXPECT_NEAR(oversampled.peak, best, 1e-3 * clean) << report.str();
}

TEST(SarAutofocusGraph, Figure2IdentityResample) {
  // PhaseResampleOp in the chain at n_az == n_pulses must be a plain copy.
  // Rolling the vector by n/2 either way is the failure this guards: it still
  // produces a plausible estimate and only shows up as lost peak.
  const double clean = run_fig2(false, false, {}).peak;
  const double restored = run_fig2(true, true, {}, kPulses, true).peak;
  EXPECT_GT(restored, kRecoveryFraction * clean)
      << "the identity resample broke 1:1 transfer: clean=" << clean << " restored=" << restored;
}

}  // namespace
