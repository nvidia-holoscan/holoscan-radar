// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Tests for CFARDetectorOp (CA variant) on the HSDK 5 Graph harness.
// Scalar live mid-stream Graph coverage remains deferred (HSDK5_PORT.md §1).

#include <gtest/gtest.h>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/expected_macros.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "detection/cfar_detector_op/cfar_alpha.hpp"
#include "detection/cfar_detector_op/cfar_detector_op.hpp"
#include "radar_common/sdk/tensor_io.hpp"
#include "radar_common/tensor/tensor_utils.hpp"
#include "utils/sdk_harness.hpp"
#include "utils/tensor_fixtures.hpp"

namespace {

using holoscan::radar::CFARDetectorOp;
using holoscan::radar::CFARParams;
using holoscan::radar::CfarCellWindowOverride;
using holoscan::radar::CfarPeakNeighborhood;
using holoscan::radar::CfarVariant;
using holoscan::radar::complex_dtype;
using holoscan::radar::real_dtype;
using holoscan::radar::testing::FiveOutGraphRunResult;
using holoscan::radar::testing::HostTensor;
using holoscan::radar::testing::run_single_in_five_out_operator_graph_result;
using holoscan::radar::testing::to_device_tensor;

constexpr int64_t kDoppler = 16;
constexpr int64_t kRange = 32;
constexpr DLDataType kInt64ConfigDtype{kDLInt, 64, 1};
constexpr DLDataType kFloat64ConfigDtype{kDLFloat, 64, 1};

struct CfarLiveTick {
  std::shared_ptr<holoscan::Tensor> rd_maps;
  std::optional<CfarCellWindowOverride> guard_cells;
  std::optional<CfarCellWindowOverride> training_cells;
  std::optional<std::vector<int64_t>> nci_axes;
  std::optional<std::vector<double>> alpha_lut;
  std::vector<int64_t> nci_shape;
};

template <typename Tag>
struct CfarLiveBank {
  static void reset(const std::vector<CfarLiveTick>& new_ticks) {
    const std::lock_guard lock(mutex);
    ticks = new_ticks;
    index = 0;
  }

  static inline std::mutex mutex;
  static inline std::vector<CfarLiveTick> ticks;
  static inline std::size_t index{0};
};

struct CfarLiveDataTag {};
struct CfarLivePodControlsTag {};
struct CfarLiveTensorControlsTag {};

struct CfarLiveTimestampBank {
  static void reset() {
    const std::lock_guard lock(mutex);
    base_timestamp_ns = 0;
  }

  static inline std::mutex mutex;
  static inline int64_t base_timestamp_ns{0};
};

holoscan::TimePoint cfar_live_capture_time(
    std::size_t index, const holoscan::ExecutionContext& context) {
  holoscan::TimePoint capture_time = context.activation_time();
  {
    const std::lock_guard lock(CfarLiveTimestampBank::mutex);
    if (CfarLiveTimestampBank::base_timestamp_ns == 0) {
      CfarLiveTimestampBank::base_timestamp_ns = capture_time.timestamp_ns;
    }
    capture_time.timestamp_ns =
        CfarLiveTimestampBank::base_timestamp_ns +
        static_cast<int64_t>(index) *
            holoscan::radar::testing::detail::kHarnessTick.count() * 1000000;
  }
  return capture_time;
}

holoscan::expected<void, holoscan::Error> emit_live_device_frame(
    holoscan::Output<holoscan::Tensor>& out,
    holoscan::ExecutionContext& context,
    const std::shared_ptr<holoscan::Tensor>& frame,
    holoscan::TimePoint capture_time) {
  auto loan = out.allocate_tensor(holoscan::TensorLoanRequest{
      .shape = frame->shape(), .dtype = frame->dtype()});
  if (!loan) {
    return holoscan::make_unexpected(std::move(loan).error());
  }
  auto writer = loan->write(context.cuda_stream());
  if (!writer) {
    return holoscan::make_unexpected(std::move(writer).error());
  }
  if (writer->byte_size() > 0 &&
      cudaMemcpyAsync(writer->data(), frame->data(), writer->byte_size(),
                      cudaMemcpyDeviceToDevice,
                      context.cuda_stream().get()) != cudaSuccess) {
    return holoscan::make_unexpected(holoscan::Error{
        holoscan::ErrorCode::kFailure,
        "CfarLiveDataSourceOp: cudaMemcpyAsync failed"});
  }
  auto committed = std::move(*writer).commit();
  if (!committed) {
    return holoscan::make_unexpected(std::move(committed).error());
  }
  return out.emit(std::move(*loan),
                  holoscan::EmitOptions{.capture_time = capture_time});
}

class CfarLiveDataSourceOp final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    constexpr holoscan::TensorRepresentation kDevice{
        .memory_kind = holoscan::MemoryKind::kCudaDevice,
        .dtype = std::nullopt,
        .rank = std::nullopt,
    };
    spec.output(rd_maps)
        .produces_tensor(holoscan::radar::pooled_tensor_output(
            kDevice, 3U, 8U * kDoppler * kRange * 8U));
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{
        .period = holoscan::radar::testing::detail::kHarnessTick});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    // Controls for frame k are published one activation before frame k. This
    // gives Exact{} the causal ordering required for sibling OnClock sources.
    if (warmup_) {
      warmup_ = false;
      return {};
    }
    std::shared_ptr<holoscan::Tensor> frame;
    std::size_t index = 0;
    {
      const std::lock_guard lock(CfarLiveBank<CfarLiveDataTag>::mutex);
      auto& bank = CfarLiveBank<CfarLiveDataTag>::ticks;
      auto& bank_index = CfarLiveBank<CfarLiveDataTag>::index;
      if (bank_index >= bank.size()) {
        return {};
      }
      index = bank_index++;
      frame = bank[index].rd_maps;
    }
    return emit_live_device_frame(rd_maps, context, frame,
                                  cfar_live_capture_time(index, context));
  }

  holoscan::Output<holoscan::Tensor> rd_maps{"rd_maps"};

 private:
  bool warmup_{true};
};

struct CfarLiveControlConfig {
  static inline bool wire_guard_cells{false};
  static inline bool wire_training_cells{false};
  static inline bool wire_nci_axes{false};
  static inline bool wire_alpha_lut{false};
};

class CfarLivePodControlsSourceOp final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.output(guard_cells);
    spec.output(training_cells);
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{
        .period = holoscan::radar::testing::detail::kHarnessTick});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    CfarLiveTick tick;
    std::size_t index = 0;
    {
      const std::lock_guard lock(CfarLiveBank<CfarLivePodControlsTag>::mutex);
      auto& bank = CfarLiveBank<CfarLivePodControlsTag>::ticks;
      auto& bank_index = CfarLiveBank<CfarLivePodControlsTag>::index;
      if (bank_index >= bank.size()) {
        return {};
      }
      index = bank_index++;
      tick = bank[index];
    }
    const holoscan::EmitOptions options{
        .capture_time = cfar_live_capture_time(index, context)};
    if (CfarLiveControlConfig::wire_guard_cells &&
        tick.guard_cells.has_value()) {
      HOLOSCAN_RETURN_IF_ERROR(guard_cells.emit(*tick.guard_cells, options));
    }
    if (CfarLiveControlConfig::wire_training_cells &&
        tick.training_cells.has_value()) {
      HOLOSCAN_RETURN_IF_ERROR(
          training_cells.emit(*tick.training_cells, options));
    }
    return {};
  }

  holoscan::Output<CfarCellWindowOverride> guard_cells{"guard_cells"};
  holoscan::Output<CfarCellWindowOverride> training_cells{"training_cells"};
};

template <typename Value>
holoscan::expected<void, holoscan::Error> emit_live_host_tensor(
    holoscan::Output<holoscan::Tensor>& output,
    const std::vector<Value>& values, const std::vector<int64_t>& shape,
    DLDataType dtype, holoscan::TimePoint capture_time) {
  auto loan = output.allocate_tensor(
      holoscan::TensorLoanRequest{.shape = shape, .dtype = dtype});
  if (!loan) {
    return holoscan::make_unexpected(std::move(loan).error());
  }
  HOLOSCAN_RETURN_IF_ERROR(loan->write_host_and_commit(
      [&values](const holoscan::TensorOutputWriteGuard& writer) {
        std::copy(values.begin(), values.end(), writer.data_as<Value>());
      }));
  return output.emit(std::move(*loan),
                     holoscan::EmitOptions{.capture_time = capture_time});
}

class CfarLiveTensorControlsSourceOp final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    constexpr holoscan::TensorRepresentation kHostInt64{
        .memory_kind = holoscan::MemoryKind::kHost,
        .dtype = kInt64ConfigDtype,
        .rank = 1U,
    };
    constexpr holoscan::TensorRepresentation kHostFloat64{
        .memory_kind = holoscan::MemoryKind::kHost,
        .dtype = kFloat64ConfigDtype,
        .rank = 1U,
    };
    spec.output(nci_axes)
        .produces_tensor(holoscan::radar::pooled_tensor_output(
            kHostInt64, 1U, 2U * sizeof(int64_t)));
    spec.output(alpha_lut)
        .produces_tensor(holoscan::radar::pooled_tensor_output(
            kHostFloat64, 1U, kDoppler * kRange * sizeof(double)));
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{
        .period = holoscan::radar::testing::detail::kHarnessTick});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    CfarLiveTick tick;
    std::size_t index = 0;
    {
      const std::lock_guard lock(
          CfarLiveBank<CfarLiveTensorControlsTag>::mutex);
      auto& bank = CfarLiveBank<CfarLiveTensorControlsTag>::ticks;
      auto& bank_index = CfarLiveBank<CfarLiveTensorControlsTag>::index;
      if (bank_index >= bank.size()) {
        return {};
      }
      index = bank_index++;
      tick = bank[index];
    }
    const holoscan::TimePoint capture_time =
        cfar_live_capture_time(index, context);
    if (CfarLiveControlConfig::wire_nci_axes &&
        tick.nci_axes.has_value()) {
      const std::vector<int64_t> shape =
          tick.nci_shape.empty()
              ? std::vector<int64_t>{
                    static_cast<int64_t>(tick.nci_axes->size())}
              : tick.nci_shape;
      HOLOSCAN_RETURN_IF_ERROR(emit_live_host_tensor(
          nci_axes, *tick.nci_axes, shape, kInt64ConfigDtype, capture_time));
    }
    if (CfarLiveControlConfig::wire_alpha_lut &&
        tick.alpha_lut.has_value()) {
      HOLOSCAN_RETURN_IF_ERROR(emit_live_host_tensor(
          alpha_lut, *tick.alpha_lut,
          {static_cast<int64_t>(tick.alpha_lut->size())},
          kFloat64ConfigDtype, capture_time));
    }
    return {};
  }

  holoscan::Output<holoscan::Tensor> nci_axes{"nci_axes"};
  holoscan::Output<holoscan::Tensor> alpha_lut{"alpha_lut"};
};

struct CfarOutputs {
  HostTensor mask;
  HostTensor score;
  HostTensor noise;
  HostTensor idx;
  HostTensor power;
};

std::vector<double> power_map(int64_t num_doppler, int64_t num_range,
                              double floor,
                              const std::vector<std::pair<int64_t, int64_t>>&
                                  targets,
                              double peak) {
  std::vector<double> map(static_cast<size_t>(num_doppler * num_range), floor);
  for (const auto& [d, r] : targets) {
    map[static_cast<size_t>(d * num_range + r)] = peak;
  }
  return map;
}

std::vector<double> power_map(
    double floor,
    const std::vector<std::pair<int64_t, int64_t>>& targets, double peak) {
  return power_map(kDoppler, kRange, floor, targets, peak);
}

CFARParams ca_params(double pfa) {
  CFARParams p{};
  p.cfar_variant = CfarVariant::kCA;
  p.guard_cells = {1, 1};
  p.training_cells = {2, 2};
  p.pfa = pfa;
  return p;
}

// cpplint misreads the wrapped template header as an indented declaration
// inside the anonymous namespace; the parameter list cannot fit on one line.
template <typename Source, typename OpRef, typename S0, typename S1,
          typename S2, typename S3, typename S4>  // NOLINT(whitespace/indent_namespace)
void wire_cfar(holoscan::Graph& graph, const Source& source, OpRef& op,
               const S0& sink0, const S1& sink1, const S2& sink2,
               const S3& sink3, const S4& sink4) {
  const holoscan::ConnectionOptions q{.queue_depth = 4U};
  graph.add_flow(source->out, op->rd_maps, q);
  graph.add_flow(op->detection_mask, sink0->in, q);
  graph.add_flow(op->detection_score, sink1->in, q);
  graph.add_flow(op->noise_estimate, sink2->in, q);
  graph.add_flow(op->detection_idx, sink3->in, q);
  graph.add_flow(op->cfar_power, sink4->in, q);
}

template <typename T>
FiveOutGraphRunResult run_cfar_result(std::shared_ptr<holoscan::Tensor> input,
                                      CFARParams params) {
  return run_single_in_five_out_operator_graph_result<CFARDetectorOp<T>>(
      {std::move(input)}, std::move(params), "detection_mask",
      "detection_score", "noise_estimate", "detection_idx", "cfar_power",
      [](holoscan::Graph& graph, const auto& source, auto& op,
         const auto& sink0, const auto& sink1, const auto& sink2,
         const auto& sink3, const auto& sink4) {
        wire_cfar(graph, source, op, sink0, sink1, sink2, sink3, sink4);
      });
}

template <typename T>
CfarOutputs run_cfar(std::shared_ptr<holoscan::Tensor> input,
                     CFARParams params) {
  FiveOutGraphRunResult result =
      run_cfar_result<T>(std::move(input), std::move(params));
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
  if (result.frames0.size() != 1U || result.frames1.size() != 1U ||
      result.frames2.size() != 1U || result.frames3.size() != 1U ||
      result.frames4.size() != 1U) {
    throw std::runtime_error(
        "CFARDetectorOp test: expected one output frame set");
  }
  return {std::move(result.frames0.front()), std::move(result.frames1.front()),
          std::move(result.frames2.front()), std::move(result.frames3.front()),
          std::move(result.frames4.front())};
}

// Which override ports the graph wires, and so which controls its sources
// emit. The operator declares every entry unconditionally.
struct CfarLiveWiring {
  bool guard_cells{false};
  bool training_cells{false};
  bool nci_axes{false};
  bool alpha_lut{false};
};

FiveOutGraphRunResult run_cfar_live(std::vector<CfarLiveTick> ticks,
                                    CFARParams params,
                                    CfarLiveWiring wiring) {
  namespace harness = holoscan::radar::testing::detail;
  FiveOutGraphRunResult result;
  harness::FrameBank::reset();
  harness::FrameBankB::reset();
  harness::FrameBankC::reset();
  harness::FrameBankD::reset();
  harness::FrameBankE::reset();
  const std::size_t expected = ticks.size();
  CfarLiveBank<CfarLiveDataTag>::reset(ticks);
  CfarLiveBank<CfarLivePodControlsTag>::reset(ticks);
  CfarLiveBank<CfarLiveTensorControlsTag>::reset(ticks);
  CfarLiveTimestampBank::reset();
  const bool wire_guard_cells = wiring.guard_cells;
  const bool wire_training_cells = wiring.training_cells;
  const bool wire_nci_axes = wiring.nci_axes;
  const bool wire_alpha_lut = wiring.alpha_lut;
  CfarLiveControlConfig::wire_guard_cells = wire_guard_cells;
  CfarLiveControlConfig::wire_training_cells = wire_training_cells;
  CfarLiveControlConfig::wire_nci_axes = wire_nci_axes;
  CfarLiveControlConfig::wire_alpha_lut = wire_alpha_lut;

  holoscan::Graph graph{"cfar-live-config"};
  const auto source = graph.op<CfarLiveDataSourceOp>("data_source");
  auto op = graph.op<CFARDetectorOp<double>>(
      "op", holoscan::radar::testing::default_test_extents());
  op.set_params(std::move(params));
  const auto sink0 = graph.op<harness::TensorCaptureOp>("sink0");
  const auto sink1 = graph.op<harness::TensorCaptureBOp>("sink1");
  const auto sink2 = graph.op<harness::TensorCaptureCOp>("sink2");
  const auto sink3 = graph.op<harness::TensorCaptureDOp>("sink3");
  const auto sink4 = graph.op<harness::TensorCaptureEOp>("sink4");
  graph.partition("data")
      .add(source)
      .add(op)
      .add(sink0)
      .add(sink1)
      .add(sink2)
      .add(sink3)
      .add(sink4);
  const holoscan::ConnectionOptions q{.queue_depth = 4U};
  graph.add_flow(source->rd_maps, op->rd_maps, q);
  if (wire_guard_cells || wire_training_cells) {
    const auto control =
        graph.op<CfarLivePodControlsSourceOp>("pod_control_source");
    if (wire_guard_cells) {
      graph.add_flow(control->guard_cells, op->guard_cells, q);
    }
    if (wire_training_cells) {
      graph.add_flow(control->training_cells, op->training_cells, q);
    }
    graph.partition("pod_control").add(control);
  }
  if (wire_nci_axes || wire_alpha_lut) {
    const auto control =
        graph.op<CfarLiveTensorControlsSourceOp>("tensor_control_source");
    if (wire_nci_axes) {
      graph.add_flow(control->nci_axes, op->nci_axes, q);
    }
    if (wire_alpha_lut) {
      graph.add_flow(control->alpha_lut, op->alpha_lut, q);
    }
    graph.partition("tensor_control").add(control);
  }
  graph.add_flow(op->detection_mask, sink0->in, q);
  graph.add_flow(op->detection_score, sink1->in, q);
  graph.add_flow(op->noise_estimate, sink2->in, q);
  graph.add_flow(op->detection_idx, sink3->in, q);
  graph.add_flow(op->cfar_power, sink4->in, q);
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

  holoscan::CompileOptions options;
  options.deployment.bind_tensor_output_device(
      holoscan::TensorOutputDevicePlacement{
          .operator_path = "data_source",
          .output_port = "rd_maps",
          .device = holoscan::DeviceId{0},
      });
  harness::bind_op_device_output(options, "detection_mask");
  harness::bind_op_device_output(options, "detection_score");
  harness::bind_op_device_output(options, "noise_estimate");
  harness::bind_op_device_output(options, "detection_idx");
  harness::bind_op_device_output(options, "cfar_power");

  try {
    const holoscan::ExecutionPlan plan =
        holoscan::compile(graph, std::move(options));
    result.plan_ok = plan.ok();
    result.plan_json = plan.json();
    if (!plan.ok()) {
      return result;
    }
    holoscan::RunSession session = holoscan::run_async(plan);
    const auto deadline =
        std::chrono::steady_clock::now() + harness::kHarnessDeadline;
    while (session.diagnostics().empty() &&
           std::chrono::steady_clock::now() < deadline) {
      {
        const std::lock_guard lock(harness::FrameBank::mutex);
        if (harness::FrameBank::captured.size() >= expected) {
          break;
        }
      }
      std::this_thread::sleep_for(harness::kHarnessPoll);
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
    return result;
  }
  {
    const std::lock_guard lock(harness::FrameBank::mutex);
    result.frames0 = std::move(harness::FrameBank::captured);
  }
  {
    const std::lock_guard lock(harness::FrameBankB::mutex);
    result.frames1 = std::move(harness::FrameBankB::captured);
  }
  {
    const std::lock_guard lock(harness::FrameBankC::mutex);
    result.frames2 = std::move(harness::FrameBankC::captured);
  }
  {
    const std::lock_guard lock(harness::FrameBankD::mutex);
    result.frames3 = std::move(harness::FrameBankD::captured);
  }
  {
    const std::lock_guard lock(harness::FrameBankE::mutex);
    result.frames4 = std::move(harness::FrameBankE::captured);
  }
  return result;
}

int64_t num_detections(const HostTensor& idx) {
  return idx.shape.empty() ? 0 : idx.shape[0];
}

bool idx_contains(const HostTensor& idx, int64_t doppler, int64_t range,
                  const std::vector<int64_t>& batch = {}) {
  if (idx.shape.size() < 2) {
    return false;
  }
  const int64_t n_cols = idx.shape[1];
  const int64_t n_preserved = n_cols - 2;
  if (static_cast<int64_t>(batch.size()) != n_preserved) {
    return false;
  }
  const auto* rows = idx.as<int64_t>();
  for (int64_t i = 0; i < num_detections(idx); ++i) {
    const int64_t* row = rows + i * n_cols;
    bool batch_match = true;
    for (int64_t b = 0; b < n_preserved; ++b) {
      if (row[b] != batch[static_cast<size_t>(b)]) {
        batch_match = false;
        break;
      }
    }
    if (!batch_match) {
      continue;
    }
    if (row[n_preserved] == doppler && row[n_preserved + 1] == range) {
      return true;
    }
  }
  return false;
}

int64_t linear_index(const std::vector<int64_t>& shape,
                     const std::vector<int64_t>& coords) {
  int64_t idx = 0;
  int64_t stride = 1;
  for (int64_t i = static_cast<int64_t>(shape.size()) - 1; i >= 0; --i) {
    idx += coords[static_cast<size_t>(i)] * stride;
    stride *= shape[static_cast<size_t>(i)];
  }
  return idx;
}

std::vector<double> power_cube(
    const std::vector<int64_t>& shape, double floor,
    const std::vector<std::pair<std::vector<int64_t>, double>>& cells) {
  int64_t count = 1;
  for (int64_t dim : shape) {
    count *= dim;
  }
  std::vector<double> data(static_cast<size_t>(count), floor);
  for (const auto& [coords, value] : cells) {
    data[static_cast<size_t>(linear_index(shape, coords))] = value;
  }
  return data;
}

template <typename T>
void expect_mask_matches_score_interior(const HostTensor& mask,
                                        const HostTensor& score,
                                        int64_t num_doppler,
                                        int64_t num_range) {
  ASSERT_EQ(mask.shape, score.shape);
  const int64_t margin = 3;
  const auto* mask_bytes = mask.as<uint8_t>();
  const auto* scores = score.as<T>();
  for (int64_t d = margin; d < num_doppler - margin; ++d) {
    for (int64_t r = margin; r < num_range - margin; ++r) {
      const int64_t i = d * num_range + r;
      const bool detected = scores[i] > T{1};
      EXPECT_EQ(mask_bytes[i] != 0, detected)
          << "cell (" << d << ", " << r << ") score=" << scores[i];
    }
  }
}

double score_at(const HostTensor& score, int64_t d, int64_t r) {
  return score.as<double>()[d * kRange + r];
}

}  // namespace

TEST(CFARDetectorOpTest, DetectsSingleTargetRealPath) {
  const int64_t dt = 8;
  const int64_t rt = 16;
  const auto map = power_map(1.0, {{dt, rt}}, 100.0);
  auto input = to_device_tensor<double>(map, {kDoppler, kRange},
                                        real_dtype<double>());

  const auto out = run_cfar<double>(input, ca_params(1e-3));

  ASSERT_EQ(num_detections(out.idx), 1);
  const auto* rows = out.idx.as<int64_t>();
  EXPECT_EQ(rows[0], dt);
  EXPECT_EQ(rows[1], rt);

  const auto* mask = out.mask.as<uint8_t>();
  EXPECT_EQ(mask[dt * kRange + rt], 1);
  const auto* score = out.score.as<double>();
  EXPECT_GT(score[dt * kRange + rt], 1.0);
}

TEST(CFARDetectorOpTest, NoiseEstimateTracksFloor) {
  const double floor = 3.5;
  const auto map = power_map(floor, {}, floor);
  auto input = to_device_tensor<double>(map, {kDoppler, kRange},
                                        real_dtype<double>());

  const auto out = run_cfar<double>(input, ca_params(1e-3));

  const auto* noise = out.noise.as<double>();
  EXPECT_NEAR(noise[8 * kRange + 16], floor, 1e-9);
}

TEST(CFARDetectorOpTest, NoFalseAlarmsOnFlatMap) {
  const auto map = power_map(1.0, {}, 1.0);
  auto input = to_device_tensor<double>(map, {kDoppler, kRange},
                                        real_dtype<double>());

  const auto out = run_cfar<double>(input, ca_params(1e-6));

  EXPECT_EQ(num_detections(out.idx), 0);
}

TEST(CFARDetectorOpTest, ComplexInputFormsPower) {
  const int64_t dt = 5;
  const int64_t rt = 20;
  std::vector<std::complex<double>> cube(
      static_cast<size_t>(kDoppler * kRange), std::complex<double>(1.0, 0.0));
  cube[static_cast<size_t>(dt * kRange + rt)] = std::complex<double>(10.0, 0.0);
  auto input = to_device_tensor<std::complex<double>>(
      cube, {kDoppler, kRange}, complex_dtype<double>());

  const auto out = run_cfar<double>(input, ca_params(1e-3));

  ASSERT_EQ(num_detections(out.idx), 1);
  const auto* rows = out.idx.as<int64_t>();
  EXPECT_EQ(rows[0], dt);
  EXPECT_EQ(rows[1], rt);
  const auto* noise = out.noise.as<double>();
  EXPECT_NEAR(noise[0], 1.0, 1e-9);
}

TEST(CFARDetectorOpTest, CfarPowerIsSquareLawPowerWithoutNci) {
  const int64_t dt = 5;
  const int64_t rt = 20;
  std::vector<std::complex<double>> cube(
      static_cast<size_t>(kDoppler * kRange), std::complex<double>(1.0, 0.0));
  cube[static_cast<size_t>(dt * kRange + rt)] = std::complex<double>(3.0, 4.0);
  auto input = to_device_tensor<std::complex<double>>(
      cube, {kDoppler, kRange}, complex_dtype<double>());

  const auto out = run_cfar<double>(input, ca_params(1e-3));

  ASSERT_EQ(out.power.shape, (std::vector<int64_t>{kDoppler, kRange}));
  ASSERT_EQ(out.power.shape, out.score.shape);
  const auto* p = out.power.as<double>();
  for (int64_t i = 0; i < kDoppler * kRange; ++i) {
    EXPECT_NEAR(p[i], std::norm(cube[static_cast<size_t>(i)]), 1e-9) << i;
  }
  EXPECT_NEAR(p[dt * kRange + rt], 25.0, 1e-9);
}

TEST(CFARDetectorOpTest, CfarPowerIsNciSumOverFold) {
  constexpr int64_t kChannels = 4;
  const int64_t dt = 5;
  const int64_t rt = 20;
  std::vector<std::complex<double>> cube(
      static_cast<size_t>(kChannels * kDoppler * kRange));
  for (int64_t k = 0; k < kChannels; ++k) {
    for (int64_t i = 0; i < kDoppler * kRange; ++i) {
      cube[static_cast<size_t>(k * kDoppler * kRange + i)] =
          std::complex<double>(static_cast<double>(k + 1), 0.0);
    }
    cube[static_cast<size_t>(k * kDoppler * kRange + dt * kRange + rt)] =
        std::complex<double>(10.0, 0.0);
  }
  auto input = to_device_tensor<std::complex<double>>(
      cube, {kChannels, kDoppler, kRange}, complex_dtype<double>());

  CFARParams params = ca_params(1e-3);
  params.nci_axes = {0};
  params.nci_fold = kChannels;
  const auto out = run_cfar<double>(input, params);

  ASSERT_EQ(out.power.shape, (std::vector<int64_t>{kDoppler, kRange}));
  ASSERT_EQ(out.power.shape, out.score.shape);
  const auto* p = out.power.as<double>();
  EXPECT_NEAR(p[0], 30.0, 1e-9);
  EXPECT_NEAR(p[dt * kRange + rt], 400.0, 1e-9);

  ASSERT_EQ(num_detections(out.idx), 1);
  const int64_t peak =
      static_cast<int64_t>(std::max_element(p, p + kDoppler * kRange) - p);
  const auto* rows = out.idx.as<int64_t>();
  EXPECT_EQ(peak, rows[0] * kRange + rows[1]);
}

TEST(CFARDetectorOpTest, MaxDetectionsCap) {
  const auto map = power_map(1.0, {{4, 8}, {12, 24}}, 100.0);
  auto input = to_device_tensor<double>(map, {kDoppler, kRange},
                                        real_dtype<double>());

  CFARParams params = ca_params(1e-3);
  params.max_detections = 1;
  const auto out = run_cfar<double>(input, params);

  EXPECT_EQ(num_detections(out.idx), 1);
}

TEST(CFARDetectorOpTest, FloatDoubleAgree) {
  const int64_t dt = 8;
  const int64_t rt = 16;
  const auto map = power_map(1.0, {{dt, rt}}, 100.0);

  std::vector<float> mapf(map.begin(), map.end());
  auto in_f = to_device_tensor<float>(mapf, {kDoppler, kRange},
                                      real_dtype<float>());
  auto in_d = to_device_tensor<double>(map, {kDoppler, kRange},
                                       real_dtype<double>());

  const auto out_f = run_cfar<float>(in_f, ca_params(1e-3));
  const auto out_d = run_cfar<double>(in_d, ca_params(1e-3));

  ASSERT_EQ(num_detections(out_f.idx), num_detections(out_d.idx));
  const auto* rf = out_f.idx.as<int64_t>();
  const auto* rd = out_d.idx.as<int64_t>();
  EXPECT_EQ(rf[0], rd[0]);
  EXPECT_EQ(rf[1], rd[1]);
}

TEST(CFARDetectorOpTest, DetectsOnLargeMap) {
  constexpr int64_t kLargeDoppler = 256;
  constexpr int64_t kLargeRange = 512;
  const int64_t dt = 128;
  const int64_t rt = 256;
  const auto map =
      power_map(kLargeDoppler, kLargeRange, 1.0, {{dt, rt}}, 100.0);
  auto input = to_device_tensor<double>(map, {kLargeDoppler, kLargeRange},
                                        real_dtype<double>());

  const auto out = run_cfar<double>(input, ca_params(1e-3));

  ASSERT_EQ(num_detections(out.idx), 1);
  EXPECT_TRUE(idx_contains(out.idx, dt, rt));
  expect_mask_matches_score_interior<double>(out.mask, out.score,
                                             kLargeDoppler, kLargeRange);
}

TEST(CFARDetectorOpTest, MaskMatchesScoreInterior) {
  const auto map = power_map(1.0, {{8, 16}}, 100.0);
  auto input = to_device_tensor<double>(map, {kDoppler, kRange},
                                        real_dtype<double>());

  const auto out = run_cfar<double>(input, ca_params(1e-3));

  expect_mask_matches_score_interior<double>(out.mask, out.score, kDoppler,
                                             kRange);
  const auto* scores = out.score.as<double>();
  EXPECT_GT(scores[8 * kRange + 16], 1.0);
  const int64_t far = 4 * kRange + 4;
  EXPECT_LT(scores[far], 1.0);
}

TEST(CFARDetectorOpTest, RealAndComplexPathsAgree) {
  const int64_t dt = 10;
  const int64_t rt = 22;
  const double floor = 2.0;
  const double peak = 80.0;

  const auto real_map = power_map(floor, {{dt, rt}}, peak);
  auto real_input = to_device_tensor<double>(real_map, {kDoppler, kRange},
                                             real_dtype<double>());

  std::vector<std::complex<double>> complex_map(
      static_cast<size_t>(kDoppler * kRange),
      std::complex<double>(std::sqrt(floor), 0.0));
  complex_map[static_cast<size_t>(dt * kRange + rt)] =
      std::complex<double>(std::sqrt(peak), 0.0);
  auto complex_input = to_device_tensor<std::complex<double>>(
      complex_map, {kDoppler, kRange}, complex_dtype<double>());

  const auto out_real = run_cfar<double>(real_input, ca_params(1e-3));
  const auto out_complex = run_cfar<double>(complex_input, ca_params(1e-3));

  ASSERT_EQ(num_detections(out_real.idx), num_detections(out_complex.idx));
  EXPECT_TRUE(idx_contains(out_real.idx, dt, rt));
  EXPECT_TRUE(idx_contains(out_complex.idx, dt, rt));
}

TEST(CFARDetectorOpTest, BorderCellsUseInBoundsTrainingOnly) {
  const double floor = 4.0;
  const auto map = power_map(floor, {}, floor);
  auto input = to_device_tensor<double>(map, {kDoppler, kRange},
                                        real_dtype<double>());

  const auto out = run_cfar<double>(input, ca_params(1e-6));

  const auto* noise = out.noise.as<double>();
  const auto* scores = out.score.as<double>();
  EXPECT_NEAR(noise[0], floor, 1e-6);
  EXPECT_NEAR(noise[8 * kRange + 16], floor, 1e-9);
  EXPECT_LE(scores[0], 1.0);
  EXPECT_EQ(num_detections(out.idx), 0);
}

TEST(CFARDetectorOpTest, DetectsTwoTargets) {
  const auto map = power_map(1.0, {{4, 8}, {12, 24}}, 100.0);
  auto input = to_device_tensor<double>(map, {kDoppler, kRange},
                                        real_dtype<double>());

  const auto out = run_cfar<double>(input, ca_params(1e-3));

  ASSERT_EQ(num_detections(out.idx), 2);
  EXPECT_TRUE(idx_contains(out.idx, 4, 8));
  EXPECT_TRUE(idx_contains(out.idx, 12, 24));
}

TEST(CFARDetectorOpTest, DetectionIdxRasterOrder) {
  const auto map = power_map(1.0, {{4, 8}, {12, 24}}, 100.0);
  auto input = to_device_tensor<double>(map, {kDoppler, kRange},
                                        real_dtype<double>());

  const auto out = run_cfar<double>(input, ca_params(1e-3));

  ASSERT_EQ(num_detections(out.idx), 2);
  ASSERT_EQ(out.idx.shape.size(), 2u);
  ASSERT_EQ(out.idx.shape[1], 2);
  const auto* rows = out.idx.as<int64_t>();
  EXPECT_EQ(rows[0], 4);
  EXPECT_EQ(rows[1], 8);
  EXPECT_EQ(rows[2], 12);
  EXPECT_EQ(rows[3], 24);
}

TEST(CFARDetectorOpTest, LiveThresholdScaleOverrideMidStreamAndSticky) {
  GTEST_SKIP() << "[EA HSDK5 ONLY] Graph mid-stream sticky live harness "
                   "deferred (HSDK5_PORT.md §1)";
}

TEST(CFARDetectorOpTest, LivePfaOverrideMidStreamAndSticky) {
  GTEST_SKIP() << "[EA HSDK5 ONLY] Graph mid-stream sticky live harness "
                   "deferred (HSDK5_PORT.md §1)";
}

TEST(CFARDetectorOpTest, LiveMaxDetectionsOverrideMidStreamAndSticky) {
  GTEST_SKIP() << "[EA HSDK5 ONLY] Graph mid-stream sticky live harness "
                   "deferred (HSDK5_PORT.md §1)";
}

TEST(CFARDetectorOpTest, GuardCellsConfigurableFromParams) {
  constexpr int64_t dt = 8;
  constexpr int64_t rt = 16;
  auto map = power_map(1.0, {}, 1.0);
  map[static_cast<size_t>(dt * kRange + rt)] = 40.0;
  map[static_cast<size_t>(dt * kRange + rt + 2)] = 10000.0;
  auto input = to_device_tensor(map, {kDoppler, kRange}, real_dtype<double>());
  CFARParams narrow = ca_params(1e-3);
  narrow.blocking_alpha_lut_rebuild = true;
  CFARParams wide = narrow;
  wide.guard_cells = {1, 2};
  const auto n = run_cfar<double>(input, narrow);
  const auto w = run_cfar<double>(input, wide);
  EXPECT_LT(score_at(n.score, dt, rt), score_at(w.score, dt, rt));
}

TEST(CFARDetectorOpTest, TrainingCellsConfigurableFromParams) {
  constexpr int64_t dt = 8;
  constexpr int64_t rt = 16;
  auto map = power_map(1.0, {}, 1.0);
  map[static_cast<size_t>(6 * kRange + rt)] = 10000.0;
  map[static_cast<size_t>(dt * kRange + rt)] = 40.0;
  auto input = to_device_tensor(map, {kDoppler, kRange}, real_dtype<double>());
  CFARParams wide = ca_params(1e-3);
  wide.blocking_alpha_lut_rebuild = true;
  wide.training_cells = {3, 3};
  CFARParams narrow = ca_params(1e-3);
  narrow.blocking_alpha_lut_rebuild = true;
  narrow.training_cells = {1, 1};
  const auto w = run_cfar<double>(input, wide);
  const auto n = run_cfar<double>(input, narrow);
  EXPECT_GT(score_at(w.score, dt, rt), score_at(n.score, dt, rt));
}

TEST(CFARDetectorOpTest, LiveGuardCellsOverrideMidStreamAndSticky) {
  constexpr int64_t kChannels = 2;
  const auto cube = power_cube(
      {kChannels, kDoppler, kRange}, 1.0,
      {{{0, 8, 16}, 100.0}, {{1, 8, 16}, 100.0}});
  auto input = to_device_tensor<double>(
      cube, {kChannels, kDoppler, kRange}, real_dtype<double>());
  const auto lut = holoscan::radar::detail::build_ca_alpha_lut(
      1e-3, kChannels, static_cast<int>(kDoppler),
      static_cast<int>(kRange), 1, 2, 2, 3);

  CFARParams params = ca_params(1e-3);
  params.training_cells = {2, 3};
  params.nci_axes = {0};
  params.nci_fold = kChannels;
  params.alpha_lut = lut;
  const CfarLiveTick first{
      input, CfarCellWindowOverride{1, 2}, std::nullopt, std::nullopt,
      std::nullopt, {}};
  const CfarLiveTick missing{input, std::nullopt, std::nullopt, std::nullopt,
                             std::nullopt, {}};
  const auto result = run_cfar_live({first, missing}, params,
                                     {.guard_cells = true});

  ASSERT_TRUE(result.plan_ok) << result.plan_json;
  ASSERT_TRUE(result.diagnostics.empty());
  ASSERT_EQ(result.frames1.size(), 2U);
  EXPECT_EQ(result.frames1[0].shape,
            (std::vector<int64_t>{kDoppler, kRange}));
  EXPECT_EQ(result.frames1[1].shape, result.frames1[0].shape);
  const auto* score0 = result.frames1[0].as<double>();
  const auto* score1 = result.frames1[1].as<double>();
  EXPECT_DOUBLE_EQ(score0[8 * kRange + 16], score1[8 * kRange + 16]);
}

TEST(CFARDetectorOpTest, RejectsMalformedLiveCellWindow) {
  auto input = to_device_tensor<double>(
      power_map(1.0, {{8, 16}}, 100.0), {kDoppler, kRange},
      real_dtype<double>());
  CFARParams params = ca_params(1e-3);
  const CfarLiveTick malformed{
      input, CfarCellWindowOverride{-1, 1}, std::nullopt, std::nullopt,
      std::nullopt, {}};
  const auto result = run_cfar_live({malformed}, params,
                                     {.guard_cells = true});
  EXPECT_TRUE(result.plan_ok);
  EXPECT_TRUE(result.frames0.empty());
  EXPECT_FALSE(result.diagnostics.empty());
}

// A rank-2 nci_axes never reaches compute: the source's rank-1 pool refuses
// the loan, so the malformed shape surfaces as a run diagnostic rather than
// through the operator's own vector guard. Declaring the source rank-2 to
// smuggle it past is no longer possible either — HSDK 5 rejects a producer
// whose rank capacity exceeds the consumer's exact rank when the graph
// compiles (GRAPH_TENSOR_LAYOUT_EDGE_CONFLICT).
TEST(CFARDetectorOpTest, RejectsMalformedLiveNciAxesTensor) {
  constexpr int64_t kChannels = 2;
  const auto cube = power_cube({kChannels, kDoppler, kRange}, 1.0, {});
  auto input = to_device_tensor<double>(
      cube, {kChannels, kDoppler, kRange}, real_dtype<double>());
  CFARParams params = ca_params(1e-3);
  const CfarLiveTick malformed{
      input, std::nullopt, std::nullopt, std::vector<int64_t>{0},
      std::nullopt, {1, 1}};
  const auto result = run_cfar_live({malformed}, params,
                                     {.nci_axes = true});
  EXPECT_TRUE(result.plan_ok);
  EXPECT_TRUE(result.frames0.empty());
  EXPECT_FALSE(result.diagnostics.empty());
}

TEST(CFARDetectorOpTest, LiveAlphaLutOverrideStickyAndClearing) {
  auto input = to_device_tensor<double>(
      power_map(1.0, {{8, 16}}, 100.0), {kDoppler, kRange},
      real_dtype<double>());
  auto huge_lut = holoscan::radar::detail::build_ca_alpha_lut(
      1e-3, 1, static_cast<int>(kDoppler), static_cast<int>(kRange),
      1, 1, 2, 2);
  for (size_t i = 1; i < huge_lut.size(); ++i) {
    huge_lut[i] *= 1e6;
  }
  CFARParams params = ca_params(1e-3);
  params.blocking_alpha_lut_rebuild = true;
  const CfarLiveTick external{
      input, std::nullopt, std::nullopt, std::nullopt, huge_lut, {}};
  const CfarLiveTick clear{
      input, std::nullopt, std::nullopt, std::nullopt,
      std::vector<double>{}, {}};
  const CfarLiveTick missing{input, std::nullopt, std::nullopt, std::nullopt,
                             std::nullopt, {}};
  const auto result = run_cfar_live({external, clear, missing}, params,
                                     {.alpha_lut = true});

  ASSERT_TRUE(result.plan_ok) << result.plan_json;
  ASSERT_TRUE(result.diagnostics.empty());
  ASSERT_EQ(result.frames3.size(), 3U);
  EXPECT_EQ(num_detections(result.frames3[0]), 0);
  EXPECT_GE(num_detections(result.frames3[1]), 1);
  EXPECT_EQ(num_detections(result.frames3[2]),
            num_detections(result.frames3[1]));
}

TEST(CFARDetectorOpTest, RejectsMalformedLiveAlphaLut) {
  auto input = to_device_tensor<double>(
      power_map(1.0, {{8, 16}}, 100.0), {kDoppler, kRange},
      real_dtype<double>());
  auto lut = holoscan::radar::detail::build_ca_alpha_lut(
      1e-3, 1, static_cast<int>(kDoppler), static_cast<int>(kRange),
      1, 1, 2, 2);
  lut[1] = std::numeric_limits<double>::quiet_NaN();
  CFARParams params = ca_params(1e-3);
  const CfarLiveTick malformed{
      input, std::nullopt, std::nullopt, std::nullopt, lut, {}};
  const auto result = run_cfar_live({malformed}, params,
                                     {.alpha_lut = true});
  EXPECT_TRUE(result.plan_ok);
  EXPECT_TRUE(result.frames0.empty());
  EXPECT_FALSE(result.diagnostics.empty());
}

TEST(CFARDetectorOpTest, NciIntegratesChannelsForDetection) {
  constexpr int64_t kChannels = 4;
  const int64_t dt = 8;
  const int64_t rt = 16;
  const auto cube = power_cube(
      {kChannels, kDoppler, kRange}, 1.0,
      {{{0, dt, rt}, 5.0},
       {{1, dt, rt}, 5.0},
       {{2, dt, rt}, 5.0},
       {{3, dt, rt}, 5.0}});
  auto input = to_device_tensor<double>(cube, {kChannels, kDoppler, kRange},
                                        real_dtype<double>());

  CFARParams params = ca_params(1e-3);
  params.nci_axes = {0};
  params.blocking_alpha_lut_rebuild = true;
  const auto out = run_cfar<double>(input, params);

  ASSERT_EQ(num_detections(out.idx), 1);
  EXPECT_TRUE(idx_contains(out.idx, dt, rt));
}

TEST(CFARDetectorOpTest, NciSumsAllChannelPower) {
  constexpr int64_t kChannels = 4;
  const auto cube = power_cube({kChannels, kDoppler, kRange}, 1.0,
                               {{{0, 8, 16}, 40.0}});
  auto input = to_device_tensor<double>(cube, {kChannels, kDoppler, kRange},
                                        real_dtype<double>());

  CFARParams params = ca_params(1e-3);
  params.nci_axes = {0};
  const auto out = run_cfar<double>(input, params);

  ASSERT_EQ(out.power.shape, (std::vector<int64_t>{kDoppler, kRange}));
  const auto* p = out.power.as<double>();
  EXPECT_NEAR(p[8 * kRange + 16], 43.0, 1e-9);
}

TEST(CFARDetectorOpTest, NciPreservesBatchAxis) {
  constexpr int64_t kBatch = 2;
  constexpr int64_t kChannels = 3;
  const int64_t dt = 8;
  const int64_t rt = 16;
  const auto cube = power_cube(
      {kBatch, kChannels, kDoppler, kRange}, 1.0,
      {{{0, 0, dt, rt}, 20.0},
       {{0, 1, dt, rt}, 20.0},
       {{0, 2, dt, rt}, 20.0},
       {{1, 0, dt, rt}, 20.0},
       {{1, 1, dt, rt}, 20.0},
       {{1, 2, dt, rt}, 20.0}});
  auto input = to_device_tensor<double>(
      cube, {kBatch, kChannels, kDoppler, kRange}, real_dtype<double>());

  CFARParams params = ca_params(1e-3);
  params.nci_axes = {1};
  params.blocking_alpha_lut_rebuild = true;
  const auto out = run_cfar<double>(input, params);

  ASSERT_EQ(num_detections(out.idx), 2);
  EXPECT_TRUE(idx_contains(out.idx, dt, rt, {0}));
  EXPECT_TRUE(idx_contains(out.idx, dt, rt, {1}));
}

TEST(CFARDetectorOpTest, BatchIdxIncludesPreservedAxes) {
  constexpr int64_t kBatch = 2;
  const int64_t dt = 8;
  const int64_t rt = 16;
  const auto cube =
      power_cube({kBatch, kDoppler, kRange}, 1.0,
                 {{{0, dt, rt}, 100.0}, {{1, dt, rt}, 100.0}});
  auto input = to_device_tensor<double>(cube, {kBatch, kDoppler, kRange},
                                        real_dtype<double>());

  CFARParams params = ca_params(1e-3);
  params.blocking_alpha_lut_rebuild = true;
  const auto out = run_cfar<double>(input, params);

  ASSERT_EQ(num_detections(out.idx), 2);
  EXPECT_TRUE(idx_contains(out.idx, dt, rt, {0}));
  EXPECT_TRUE(idx_contains(out.idx, dt, rt, {1}));
}

TEST(CFARDetectorOpTest, NciMaskMatchesScoreOnIntegratedMap) {
  constexpr int64_t kChannels = 4;
  const auto cube = power_cube({kChannels, kDoppler, kRange}, 1.0,
                               {{{0, 8, 16}, 40.0}});
  auto input = to_device_tensor<double>(cube, {kChannels, kDoppler, kRange},
                                        real_dtype<double>());

  CFARParams params = ca_params(1e-3);
  params.nci_axes = {0};
  const auto out = run_cfar<double>(input, params);

  expect_mask_matches_score_interior<double>(out.mask, out.score, kDoppler,
                                             kRange);
}

TEST(CFARDetectorOpTest, RejectsMissingGuardCells) {
  const auto map = power_map(1.0, {}, 1.0);
  auto input = to_device_tensor<double>(map, {kDoppler, kRange},
                                        real_dtype<double>());
  CFARParams params{};
  params.cfar_variant = CfarVariant::kCA;
  params.training_cells = {2, 2};
  params.pfa = 1e-3;
  const FiveOutGraphRunResult result = run_cfar_result<double>(input, params);
  EXPECT_TRUE(result.plan_ok);
  EXPECT_TRUE(result.frames0.empty());
  EXPECT_FALSE(result.diagnostics.empty());
}

TEST(CFARDetectorOpTest, RejectsInvalidNciAxis) {
  const auto map = power_map(1.0, {}, 1.0);
  auto input = to_device_tensor<double>(map, {kDoppler, kRange},
                                        real_dtype<double>());
  CFARParams params = ca_params(1e-3);
  params.nci_axes = {0};
  const FiveOutGraphRunResult result = run_cfar_result<double>(input, params);
  EXPECT_TRUE(result.plan_ok);
  EXPECT_TRUE(result.frames0.empty());
  EXPECT_FALSE(result.diagnostics.empty());
}

TEST(CFARDetectorOpTest, CaAlphaK1MatchesContractClosedForm) {
  constexpr int64_t kN = 40;
  constexpr double kPfa = 1e-2;
  const double alpha = holoscan::radar::detail::ca_cfar_alpha(kPfa, kN, 1);
  const double closed = kN * (std::pow(kPfa, -1.0 / kN) - 1.0);
  EXPECT_NEAR(alpha, closed, 1e-6 * closed);
  EXPECT_NEAR(holoscan::radar::detail::ca_cfar_pfa_from_alpha(alpha, kN, 1),
              kPfa, 1e-9);
}

TEST(CFARDetectorOpTest, PeakOnlyOffMatchesRawMask) {
  const auto map = power_map(1.0, {{8, 15}, {8, 16}, {8, 17}}, 100.0);
  auto input = to_device_tensor<double>(map, {kDoppler, kRange},
                                        real_dtype<double>());

  const auto raw = run_cfar<double>(input, ca_params(1e-2));
  CFARParams off = ca_params(1e-2);
  off.peak_only = false;
  const auto out = run_cfar<double>(input, off);

  EXPECT_EQ(num_detections(raw.idx), 3);
  EXPECT_EQ(num_detections(out.idx), num_detections(raw.idx));
  for (const auto r : {15, 16, 17}) {
    EXPECT_TRUE(idx_contains(out.idx, 8, r));
  }
}

TEST(CFARDetectorOpTest, PeakOnlyKeepsSingleMax) {
  auto data = power_map(1.0, {}, 1.0);
  data[static_cast<size_t>(8 * kRange + 16)] = 100.0;
  data[static_cast<size_t>(8 * kRange + 17)] = 50.0;
  auto input = to_device_tensor<double>(data, {kDoppler, kRange},
                                        real_dtype<double>());

  CFARParams params = ca_params(1e-2);
  params.peak_only = true;
  const auto out = run_cfar<double>(input, params);

  ASSERT_EQ(num_detections(out.idx), 1);
  EXPECT_TRUE(idx_contains(out.idx, 8, 16));
  EXPECT_FALSE(idx_contains(out.idx, 8, 17));
  const auto* mask = out.mask.as<uint8_t>();
  EXPECT_EQ(mask[8 * kRange + 16], 1);
  EXPECT_EQ(mask[8 * kRange + 17], 0);
}

TEST(CFARDetectorOpTest, PeakOnlyCollapsesStripRasterTiebreak) {
  const auto map = power_map(1.0, {{8, 15}, {8, 16}, {8, 17}}, 100.0);
  auto input = to_device_tensor<double>(map, {kDoppler, kRange},
                                        real_dtype<double>());

  CFARParams params = ca_params(1e-2);
  params.peak_only = true;
  const auto out = run_cfar<double>(input, params);

  ASSERT_EQ(num_detections(out.idx), 1);
  EXPECT_TRUE(idx_contains(out.idx, 8, 15));
}

TEST(CFARDetectorOpTest, PeakOnlyNeighborhoodCross4VsSquare8) {
  auto data = power_map(1.0, {}, 1.0);
  data[static_cast<size_t>(8 * kRange + 16)] = 100.0;
  data[static_cast<size_t>(9 * kRange + 17)] = 90.0;
  auto input = to_device_tensor<double>(data, {kDoppler, kRange},
                                        real_dtype<double>());

  CFARParams cross = ca_params(1e-2);
  cross.peak_only = true;
  cross.peak_neighborhood = CfarPeakNeighborhood::kCross4;
  CFARParams square = cross;
  square.peak_neighborhood = CfarPeakNeighborhood::kSquare8;

  const auto out_cross = run_cfar<double>(input, cross);
  const auto out_square = run_cfar<double>(input, square);

  EXPECT_EQ(num_detections(out_cross.idx), 2);
  EXPECT_EQ(num_detections(out_square.idx), 1);
  EXPECT_TRUE(idx_contains(out_square.idx, 8, 16));
}

TEST(CFARDetectorOpTest, ThresholdScaleConfigurableFromParams) {
  const auto map = power_map(1.0, {{8, 16}}, 100.0);
  auto input = to_device_tensor<double>(map, {kDoppler, kRange},
                                        real_dtype<double>());
  CFARParams detect = ca_params(1e-3);
  detect.threshold_scale = 1.0;
  CFARParams suppress = ca_params(1e-3);
  suppress.threshold_scale = 1e4;
  EXPECT_EQ(num_detections(run_cfar<double>(input, detect).idx), 1);
  EXPECT_EQ(num_detections(run_cfar<double>(input, suppress).idx), 0);
}

TEST(CFARDetectorOpTest, MaxDetectionsConfigurableFromParams) {
  const auto map = power_map(1.0, {{4, 8}, {12, 24}}, 100.0);
  auto input = to_device_tensor<double>(map, {kDoppler, kRange},
                                        real_dtype<double>());
  CFARParams params = ca_params(1e-3);
  params.max_detections = 1;
  EXPECT_EQ(num_detections(run_cfar<double>(input, params).idx), 1);
}

TEST(CFARDetectorOpTest, PfaConfigurableFromParams) {
  constexpr int64_t dt = 8;
  constexpr int64_t rt = 16;
  const auto map = power_map(1.0, {{dt, rt}}, 26.0);
  auto input = to_device_tensor<double>(map, {kDoppler, kRange},
                                        real_dtype<double>());
  CFARParams loose = ca_params(1e-2);
  loose.blocking_alpha_lut_rebuild = true;
  CFARParams tight = ca_params(1e-8);
  tight.blocking_alpha_lut_rebuild = true;
  EXPECT_GT(score_at(run_cfar<double>(input, loose).score, dt, rt),
            score_at(run_cfar<double>(input, tight).score, dt, rt));
}
