// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// SubarraySelectionOp contract tests (HSDK 5 Graph harness): S3 select a
// uniform sub-array from a holey MIMO virtual array; emit reduced snapshot +
// WireArrayGeometry.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "angle_estimation/beamscan_estimator_op/beamscan_estimator_op.hpp"
#include "angle_estimation/subarray_selection_op/subarray_selection_op.hpp"
#include "radar_common/descriptors/array_geometry.hpp"
#include "radar_common/descriptors/array_geometry_tensor.hpp"
#include "radar_common/sdk/tensor_io.hpp"
#include "radar_common/tensor/tensor_utils.hpp"
#include "radar_schemas/array_geometry_bridge.hpp"
#include "utils/sdk_harness.hpp"
#include "utils/tensor_fixtures.hpp"

namespace {

using holoscan::radar::ArrayGeometry;
using holoscan::radar::BeamscanEstimatorOp;
using holoscan::radar::BeamscanParams;
using holoscan::radar::SubarraySelectionOp;
using holoscan::radar::SubarraySelectionParams;
using holoscan::radar::WireArrayGeometry;
using holoscan::radar::complex_dtype;
using holoscan::radar::from_wire;
using holoscan::radar::pack_array_geometry;
using holoscan::radar::testing::default_test_extents;
using holoscan::radar::testing::GraphRunResult;
using holoscan::radar::testing::HostTensor;
using holoscan::radar::testing::TensorWireGraphRunResult;
using holoscan::radar::testing::WireCaptureBank;
using holoscan::radar::testing::WireCaptureOp;
using holoscan::radar::testing::run_single_io_operator_graph_result;
using holoscan::radar::testing::
    run_single_in_tensor_wire_out_operator_graph_result;  // NOLINT
using holoscan::radar::testing::to_device_tensor;
using holoscan::radar::testing::to_pageable_host_tensor;
using holoscan::radar::to_wire;
using holoscan::radar::unpack_array_geometry;

constexpr double kD = 0.5;

ArrayGeometry make_linear(const std::vector<int>& iy) {
  ArrayGeometry g;
  g.wavelength = 1.0;
  for (int i : iy) {
    g.element_positions.insert(g.element_positions.end(), {0.0, i * kD, 0.0});
  }
  return g;
}

// snapshot (num_items, N, K) with value[item,c,s] = (c, s) encoded — the real
// part carries the channel index and the imag part the snapshot index.
std::shared_ptr<holoscan::Tensor> make_snapshot(int64_t items, int64_t n,
                                                int64_t k) {
  std::vector<std::complex<double>> x(
      static_cast<size_t>(items * n * k), {0.0, 0.0});
  for (int64_t it = 0; it < items; ++it) {
    for (int64_t c = 0; c < n; ++c) {
      for (int64_t s = 0; s < k; ++s) {
        x[static_cast<size_t>((it * n + c) * k + s)] = {
            static_cast<double>(c), static_cast<double>(s)};
      }
    }
  }
  return to_device_tensor(x, {items, n, k}, complex_dtype<double>());
}

struct Result {
  HostTensor snapshot;
  ArrayGeometry geometry;
};

template <typename Source, typename OpRef, typename Sink0, typename Sink1>
void wire_subarray(holoscan::Graph& graph, const Source& source, OpRef& op,
                   const Sink0& sink0, const Sink1& sink1) {
  const holoscan::ConnectionOptions q{.queue_depth = 4U};
  graph.add_flow(source->out, op->snapshot, q);
  graph.add_flow(op->snapshot_out, sink0->in, q);
  graph.add_flow(op->array_geometry, sink1->in, q);
}

TensorWireGraphRunResult<WireArrayGeometry> run_select_result(
    std::shared_ptr<holoscan::Tensor> snap, SubarraySelectionParams params,
    const holoscan::radar::MaxTensorExtents& extents =
        default_test_extents()) {
  return run_single_in_tensor_wire_out_operator_graph_result<
      WireArrayGeometry, SubarraySelectionOp<double>>(
      {std::move(snap)}, std::move(params), "snapshot_out",
      [](holoscan::Graph& graph, const auto& source, auto& op,
         const auto& sink0, const auto& sink1) {
        wire_subarray(graph, source, op, sink0, sink1);
      },
      // Optional is_valid_out is declared but unwired; still needs a device
      // bind (IMPLICIT_DEVICE_ZERO_REJECTED otherwise).
      {"is_valid_out"}, extents);
}

Result run_select(std::shared_ptr<holoscan::Tensor> snap,
                  SubarraySelectionParams params) {
  TensorWireGraphRunResult<WireArrayGeometry> result =
      run_select_result(std::move(snap), std::move(params));
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
        "SubarraySelectionOp test: expected one output frame pair, got " +
        std::to_string(result.frames0.size()) + "/" +
        std::to_string(result.frames1.size()));
  }
  auto geometry = from_wire(result.frames1.front());
  if (!geometry) {
    throw std::runtime_error(
        "SubarraySelectionOp test: invalid WireArrayGeometry");
  }
  return {std::move(result.frames0.front()), std::move(*geometry)};
}

GraphRunResult run_snapshot_only(
    std::vector<std::shared_ptr<holoscan::Tensor>> snapshots,
    SubarraySelectionParams params) {
  return run_single_io_operator_graph_result<SubarraySelectionOp<double>>(
      std::move(snapshots), std::move(params), "snapshot_out",
      [](holoscan::Graph& graph, const auto& source, auto& op,
         const auto& sink) {
        const holoscan::ConnectionOptions queue{.queue_depth = 4U};
        graph.add_flow(source->out, op->snapshot, queue);
        graph.add_flow(op->snapshot_out, sink->in, queue);
      },
      {"is_valid_out"});
}

std::set<double> y_set(const ArrayGeometry& g) {
  std::set<double> s;
  for (int64_t e = 0; e < g.num_elements(); ++e) {
    s.insert(g.element_positions[static_cast<size_t>(e * 3 + 1)]);
  }
  return s;
}

void expect_channels(const HostTensor& snapshot,
                     const std::vector<int64_t>& channels, int64_t num_snap) {
  const int64_t l = static_cast<int64_t>(channels.size());
  ASSERT_EQ(snapshot.shape, (std::vector<int64_t>{1, l, num_snap}));
  const auto* snap = snapshot.as<std::complex<double>>();
  for (int64_t j = 0; j < l; ++j) {
    for (int64_t s = 0; s < num_snap; ++s) {
      const auto& got = snap[static_cast<size_t>(j * num_snap + s)];
      EXPECT_NEAR(got.real(), static_cast<double>(channels[static_cast<size_t>(j)]),
                  1e-6)
          << "out channel " << j << " snapshot " << s;
      EXPECT_NEAR(got.imag(), static_cast<double>(s), 1e-6)
          << "out channel " << j << " snapshot " << s;
    }
  }
}

SubarraySelectionParams make_params(ArrayGeometry geom,
                                    std::vector<int64_t> subarray = {},
                                    int64_t min_size = 2) {
  SubarraySelectionParams params{};
  params.array_geometry = std::move(geom);
  params.subarray = std::move(subarray);
  params.min_subarray_size = min_size;
  return params;
}

struct LiveSubarrayTick {
  std::shared_ptr<holoscan::Tensor> snapshot;
  std::optional<ArrayGeometry> geometry;
  std::shared_ptr<holoscan::Tensor> subarray;
  int64_t geometry_skew_ns{0};
  int64_t subarray_skew_ns{0};
  bool malformed_geometry{false};
};

struct LiveSubarrayBank {
  static void reset(std::vector<LiveSubarrayTick> new_ticks) {
    ticks = std::move(new_ticks);
  }
  static inline std::vector<LiveSubarrayTick> ticks;
};

template <typename Wire, typename Element>
holoscan::expected<void, holoscan::Error> emit_descriptor_tensor(
    holoscan::Output<Wire>& output, Wire wire,
    std::span<const Element> values, holoscan::TimePoint capture_time) {
  const std::array<int64_t, 2> shape{
      static_cast<int64_t>(values.size() / 3U), 3};
  constexpr DLDataType dtype{kDLFloat, 64U, 1U};
  auto loan = output.allocate_tensor(
      holoscan::TensorLoanRequest{.shape = shape, .dtype = dtype});
  if (!loan) {
    return holoscan::make_unexpected(std::move(loan).error());
  }
  auto written = loan->write_host_and_commit(
      [values](const holoscan::TensorOutputWriteGuard& writer) {
        std::copy(values.begin(), values.end(),
                  writer.template data_as<Element>());
      });
  if (!written) {
    return holoscan::make_unexpected(std::move(written).error());
  }
  return output.emit_tensor(
      std::move(*loan), std::move(wire),
      holoscan::EmitOptions{.capture_time = capture_time});
}

holoscan::expected<void, holoscan::Error> emit_host_tensor(
    holoscan::Output<holoscan::Tensor>& output,
    const holoscan::Tensor& source, holoscan::TimePoint capture_time) {
  auto loan = output.allocate_tensor(
      holoscan::TensorLoanRequest{
          .shape = source.shape(), .dtype = source.dtype()});
  if (!loan) {
    return holoscan::make_unexpected(std::move(loan).error());
  }
  auto written = loan->write_host_and_commit(
      [&source](const holoscan::TensorOutputWriteGuard& writer) {
        if (source.nbytes() > 0) {
          std::memcpy(writer.template data_as<std::byte>(), source.data(),
                      static_cast<size_t>(source.nbytes()));
        }
      });
  if (!written) {
    return holoscan::make_unexpected(std::move(written).error());
  }
  return output.emit(
      std::move(*loan),
      holoscan::EmitOptions{.capture_time = capture_time});
}

class LiveTickSourceOp final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.output(tick, "tick");
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{
        .period = holoscan::radar::testing::detail::kHarnessTick});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    if (index_ >= LiveSubarrayBank::ticks.size()) {
      return {};
    }
    return tick.emit(
        static_cast<int64_t>(index_++),
        holoscan::EmitOptions{.capture_time = context.activation_time()});
  }

  holoscan::Output<int64_t> tick{"tick"};

 private:
  std::size_t index_{0};
};

class LiveSnapshotSourceOp final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    constexpr holoscan::TensorRepresentation kDevice{
        .memory_kind = holoscan::MemoryKind::kCudaDevice,
        .dtype = std::nullopt,
        .rank = std::nullopt,
    };
    spec.input(tick, "tick").queue_depth(8U);
    spec.output(snapshot, "snapshot")
        .produces_tensor(holoscan::radar::pooled_tensor_output(
            kDevice, 3U, 1024U * 1024U));
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnEach{tick});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    auto received = tick.receive_data();
    if (!received) {
      return holoscan::make_unexpected(std::move(received).error());
    }
    const LiveSubarrayTick& tick =
        LiveSubarrayBank::ticks[static_cast<std::size_t>(*received)];
    return holoscan::radar::testing::detail::emit_bank_frame(
        snapshot, context, tick.snapshot, "LiveSnapshotSourceOp");
  }

  holoscan::Input<int64_t> tick{"tick"};
  holoscan::Output<holoscan::Tensor> snapshot{"snapshot"};
};

class LiveGeometrySourceOp final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    constexpr holoscan::TensorRepresentation kGeometry{
        .memory_kind = holoscan::MemoryKind::kHost,
        .dtype = DLDataType{kDLFloat, 64U, 1U},
        .rank = 2U,
    };
    spec.input(tick, "tick").queue_depth(8U);
    spec.output(geometry, "geometry")
        .max_emits_per_compute(1U)
        .produces_tensor(holoscan::radar::pooled_tensor_output(
            kGeometry, 2U, 64U * 1024U));
    spec.output(next, "next");
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnEach{tick});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    auto received = tick.receive_data();
    if (!received) {
      return holoscan::make_unexpected(std::move(received).error());
    }
    const LiveSubarrayTick& value =
        LiveSubarrayBank::ticks[static_cast<std::size_t>(*received)];
    if (value.geometry.has_value()) {
      WireArrayGeometry wire = to_wire(*value.geometry);
      std::vector<double> packed = pack_array_geometry(*value.geometry);
      if (value.malformed_geometry) {
        packed[packed.size() - 3U] = -1.0;
      }
      holoscan::TimePoint capture_time = context.activation_time();
      capture_time.timestamp_ns += value.geometry_skew_ns;
      auto emitted = emit_descriptor_tensor(
          geometry, std::move(wire), std::span<const double>{packed},
          capture_time);
      if (!emitted) {
        return emitted;
      }
    }
    return next.emit(
        *received,
        holoscan::EmitOptions{.capture_time = context.activation_time()});
  }

  holoscan::Input<int64_t> tick{"tick"};
  holoscan::Output<WireArrayGeometry> geometry{"geometry"};
  holoscan::Output<int64_t> next{"next"};
};

class LiveSubarrayConfigSourceOp final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    constexpr holoscan::TensorRepresentation kHost{
        .memory_kind = holoscan::MemoryKind::kHost,
        .dtype = DLDataType{kDLInt, 64U, 1U},
        .rank = 1U,
    };
    spec.input(tick, "tick").queue_depth(8U);
    spec.output(subarray, "subarray")
        .max_emits_per_compute(1U)
        .produces_tensor(holoscan::radar::pooled_tensor_output(
            kHost, 1U, 1024U));
    spec.output(next, "next");
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnEach{tick});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    auto received = tick.receive_data();
    if (!received) {
      return holoscan::make_unexpected(std::move(received).error());
    }
    const LiveSubarrayTick& value =
        LiveSubarrayBank::ticks[static_cast<std::size_t>(*received)];
    if (value.subarray) {
      holoscan::TimePoint capture_time = context.activation_time();
      capture_time.timestamp_ns += value.subarray_skew_ns;
      auto emitted =
          emit_host_tensor(subarray, *value.subarray, capture_time);
      if (!emitted) {
        return emitted;
      }
    }
    return next.emit(
        *received,
        holoscan::EmitOptions{.capture_time = context.activation_time()});
  }

  holoscan::Input<int64_t> tick{"tick"};
  holoscan::Output<holoscan::Tensor> subarray{"subarray"};
  holoscan::Output<int64_t> next{"next"};
};

// Which override ports the graph wires, and so which control-plane sources
// it builds. The operator declares both entries unconditionally.
struct LiveSubarrayWiring {
  bool geometry{false};
  bool subarray{false};
};

TensorWireGraphRunResult<WireArrayGeometry> run_live_subarray_graph(
    std::vector<LiveSubarrayTick> ticks, SubarraySelectionParams params,
    LiveSubarrayWiring wiring,
    holoscan::radar::MaxTensorExtents extents = default_test_extents()) {
  TensorWireGraphRunResult<WireArrayGeometry> result;
  holoscan::radar::testing::detail::FrameBank::reset();
  WireCaptureBank<WireArrayGeometry>::reset();
  LiveSubarrayBank::reset(std::move(ticks));
  const std::size_t expected = LiveSubarrayBank::ticks.size();

  holoscan::Graph graph{"subarray-live-overrides"};
  const auto sequencer = graph.op<LiveTickSourceOp>("sequencer");
  const auto snapshot_source =
      graph.op<LiveSnapshotSourceOp>("snapshot_source");
  auto op = graph.op<SubarraySelectionOp<double>>("op", extents);
  op.set_params(std::move(params));
  const auto sink0 =
      graph.op<holoscan::radar::testing::detail::TensorCaptureOp>("sink0");
  const auto sink1 =
      graph.op<WireCaptureOp<WireArrayGeometry>>("sink1");
  const holoscan::ConnectionOptions q{.queue_depth = 8U};
  graph.add_flow(snapshot_source->snapshot, op->snapshot, q);
  auto control_partition = graph.partition("control");
  control_partition.add(sequencer);
  if (wiring.geometry && wiring.subarray) {
    const auto geometry_source =
        graph.op<LiveGeometrySourceOp>("geometry_source");
    const auto subarray_source =
        graph.op<LiveSubarrayConfigSourceOp>("subarray_source");
    control_partition.add(geometry_source).add(subarray_source);
    graph.add_flow(sequencer->tick, geometry_source->tick, q);
    graph.add_flow(geometry_source->geometry, op->array_geometry_in, q);
    graph.add_flow(geometry_source->next, subarray_source->tick, q);
    graph.add_flow(subarray_source->subarray, op->subarray, q);
    graph.add_flow(subarray_source->next, snapshot_source->tick, q);
  } else if (wiring.geometry) {
    const auto geometry_source =
        graph.op<LiveGeometrySourceOp>("geometry_source");
    control_partition.add(geometry_source);
    graph.add_flow(sequencer->tick, geometry_source->tick, q);
    graph.add_flow(geometry_source->geometry, op->array_geometry_in, q);
    graph.add_flow(geometry_source->next, snapshot_source->tick, q);
  } else if (wiring.subarray) {
    const auto subarray_source =
        graph.op<LiveSubarrayConfigSourceOp>("subarray_source");
    control_partition.add(subarray_source);
    graph.add_flow(sequencer->tick, subarray_source->tick, q);
    graph.add_flow(subarray_source->subarray, op->subarray, q);
    graph.add_flow(subarray_source->next, snapshot_source->tick, q);
  }
  graph.add_flow(op->snapshot_out, sink0->in, q);
  graph.add_flow(op->array_geometry, sink1->in, q);
  graph.partition("data").add(snapshot_source).add(op).add(sink0).add(sink1);
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

  holoscan::CompileOptions options;
  options.deployment.bind_tensor_output_device(
      holoscan::TensorOutputDevicePlacement{
          .operator_path = "snapshot_source",
          .output_port = "snapshot",
          .device = holoscan::DeviceId{0}});
  holoscan::radar::testing::detail::bind_op_device_output(
      options, "snapshot_out");
  holoscan::radar::testing::detail::bind_op_device_output(
      options, "is_valid_out");

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
            WireCaptureBank<WireArrayGeometry>::mutex);
        count1 = WireCaptureBank<WireArrayGeometry>::payloads.size();
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
        WireCaptureBank<WireArrayGeometry>::mutex);
    result.frames1 =
        std::move(WireCaptureBank<WireArrayGeometry>::payloads);
    result.frame1_capture_times =
        std::move(WireCaptureBank<WireArrayGeometry>::capture_times);
  }
  return result;
}

std::shared_ptr<holoscan::Tensor> host_subarray(
    const std::vector<int64_t>& values) {
  return to_pageable_host_tensor(
      values, {static_cast<int64_t>(values.size())},
      DLDataType{kDLInt, 64U, 1U});
}

ArrayGeometry unpack_output_geometry(const WireArrayGeometry& wire) {
  auto geometry = from_wire(wire);
  if (!geometry) {
    throw std::runtime_error("invalid WireArrayGeometry output");
  }
  return std::move(*geometry);
}

TEST(SubarraySelectionOpTest, HoleyUlaSelectsLongestRun) {
  const auto geom = make_linear({0, 1, 2, 3, 5, 6});  // longest run 0..3 → L=4
  auto snap = make_snapshot(1, 6, 8);
  const auto out =
      run_select(snap, make_params(geom, {}, /*min_size=*/2));
  EXPECT_EQ(out.snapshot.shape, (std::vector<int64_t>{1, 4, 8}));
  EXPECT_EQ(out.geometry.num_elements(), 4);
  EXPECT_EQ(y_set(out.geometry),
            (std::set<double>{0.0, kD, 2 * kD, 3 * kD}));
  expect_channels(out.snapshot, {0, 1, 2, 3}, 8);
}

TEST(SubarraySelectionOpTest, FilledUlaPassesThrough) {
  const auto geom = make_linear({0, 1, 2, 3});
  auto snap = make_snapshot(1, 4, 5);
  const auto out = run_select(snap, make_params(geom));
  EXPECT_EQ(out.snapshot.shape, (std::vector<int64_t>{1, 4, 5}));
  EXPECT_EQ(out.geometry.num_elements(), 4);
  EXPECT_EQ(out.geometry.wavelength, 1.0);
}

TEST(SubarraySelectionOpTest, FilledUnorderedUlaIsGatheredToLatticeOrder) {
  const auto geom = make_linear({2, 0, 1, 3});
  auto snap = make_snapshot(1, 4, 2);
  const auto out = run_select(snap, make_params(geom));

  EXPECT_EQ(out.snapshot.shape, (std::vector<int64_t>{1, 4, 2}));
  expect_channels(out.snapshot, {1, 2, 0, 3}, 2);
  EXPECT_EQ(y_set(out.geometry),
            (std::set<double>{0.0, kD, 2 * kD, 3 * kD}));
}

TEST(SubarraySelectionOpTest, OptionalGeometryMayBeLeftUnwired) {
  const auto geom = make_linear({0, 1, 2, 3, 5, 6});
  const GraphRunResult out = run_snapshot_only(
      {make_snapshot(1, 6, 2), make_snapshot(1, 6, 2)},
      make_params(geom));

  ASSERT_TRUE(out.plan_ok) << out.plan_json;
  EXPECT_TRUE(out.diagnostics.empty());
  ASSERT_EQ(out.frames.size(), 2U);
  expect_channels(out.frames[0], {0, 1, 2, 3}, 2);
  expect_channels(out.frames[1], {0, 1, 2, 3}, 2);
}

TEST(SubarraySelectionOpTest, ReducedGeometryDirectlyDrivesBeamscan) {
  namespace detail = holoscan::radar::testing::detail;
  detail::FrameBank::reset();
  detail::FrameBankB::reset();
  detail::FrameBank::input_frames = {
      to_device_tensor(
          std::vector<std::complex<double>>(6U, {1.0, 0.0}), {1, 6},
          complex_dtype<double>())};

  holoscan::Graph graph{"subarray-geometry-to-beamscan"};
  const auto source = graph.op<detail::TensorSourceOp>("source");
  auto extents = default_test_extents();
  auto selector =
      graph.op<SubarraySelectionOp<double>>("selector", extents);
  const ArrayGeometry full = make_linear({0, 1, 2, 3, 5, 6});
  selector.set_params(make_params(full, {0, 1, 2, 3}));

  auto beamscan =
      graph.op<BeamscanEstimatorOp<double>>("beamscan", extents);
  BeamscanParams beamscan_params;
  // Deliberately incompatible with the 4-channel selected snapshot.
  // Successful steering proves the same-tick wire geometry won over Params.
  beamscan_params.array_geometry = make_linear({0, 1, 2});
  beamscan_params.az_angles = {-0.2, 0.0, 0.2};
  beamscan_params.el_angles = {0.0};
  beamscan.set_params(std::move(beamscan_params));

  const auto aoa_sink = graph.op<detail::TensorCaptureOp>("aoa_sink");
  const auto spectrum_sink =
      graph.op<detail::TensorCaptureBOp>("spectrum_sink");
  const holoscan::ConnectionOptions queue{.queue_depth = 4U};
  graph.add_flow(source->out, selector->snapshot, queue);
  graph.add_flow(selector->snapshot_out, beamscan->snapshot, queue);
  graph.add_flow(
      selector->array_geometry, beamscan->array_geometry, queue);
  graph.add_flow(beamscan->aoa_estimates, aoa_sink->in, queue);
  graph.add_flow(beamscan->angle_spectrum, spectrum_sink->in, queue);
  graph.partition("integration")
      .add(source)
      .add(selector)
      .add(beamscan)
      .add(aoa_sink)
      .add(spectrum_sink);
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

  holoscan::CompileOptions options;
  auto bind = [&options](const char* op, const char* port) {
    options.deployment.bind_tensor_output_device(
        holoscan::TensorOutputDevicePlacement{
            .operator_path = op,
            .output_port = port,
            .device = holoscan::DeviceId{0}});
  };
  bind("source", "out");
  bind("selector", "snapshot_out");
  bind("selector", "is_valid_out");
  bind("beamscan", "aoa_estimates");
  bind("beamscan", "angle_spectrum");

  const holoscan::ExecutionPlan plan =
      holoscan::compile(graph, std::move(options));
  ASSERT_TRUE(plan.ok()) << plan.json();
  holoscan::RunSession session = holoscan::run_async(plan);
  const auto deadline =
      std::chrono::steady_clock::now() + detail::kHarnessDeadline;
  while (session.diagnostics().empty() &&
         std::chrono::steady_clock::now() < deadline) {
    bool captured = false;
    {
      const std::lock_guard lock(detail::FrameBank::mutex);
      captured = !detail::FrameBank::captured.empty();
    }
    if (captured) {
      break;
    }
    std::this_thread::sleep_for(detail::kHarnessPoll);
  }
  session.request_stop();
  session.wait();
  ASSERT_TRUE(session.diagnostics().empty());
  const std::lock_guard lock(detail::FrameBank::mutex);
  ASSERT_EQ(detail::FrameBank::captured.size(), 1U);
  const auto& aoa = detail::FrameBank::captured.front();
  ASSERT_EQ(aoa.shape, (std::vector<int64_t>{1, 2}));
  EXPECT_NEAR(aoa.as<double>()[0], 0.0, 1e-12);
}

TEST(SubarraySelectionOpTest, ExplicitSubarray) {
  const auto geom = make_linear({0, 1, 2, 3, 4, 5, 6, 7});
  auto snap = make_snapshot(1, 8, 3);
  const auto out =
      run_select(snap, make_params(geom, {2, 3, 4, 5}));
  EXPECT_EQ(out.snapshot.shape, (std::vector<int64_t>{1, 4, 3}));
  EXPECT_EQ(y_set(out.geometry),
            (std::set<double>{2 * kD, 3 * kD, 4 * kD, 5 * kD}));
  expect_channels(out.snapshot, {2, 3, 4, 5}, 3);
}

TEST(SubarraySelectionOpTest, SubarrayConfigurableFromParams) {
  const auto geom = make_linear({0, 1, 2, 3, 4, 5, 6, 7});
  auto snap = make_snapshot(1, 8, 2);
  SubarraySelectionParams params = make_params(geom, {1, 2, 3});
  const auto out = run_select(snap, params);
  EXPECT_EQ(out.snapshot.shape, (std::vector<int64_t>{1, 3, 2}));
  expect_channels(out.snapshot, {1, 2, 3}, 2);
}

TEST(SubarraySelectionOpTest, LiveOverridesRequireExactCaptureTime) {
  const auto base = make_linear({0, 1, 2, 3, 4, 5, 6, 7});
  auto changed = base;
  for (int64_t i = 0; i < changed.num_elements(); ++i) {
    changed.element_positions[static_cast<size_t>(i * 3 + 1)] += 10.0;
  }
  SubarraySelectionParams params = make_params(base, {0, 1});
  const TensorWireGraphRunResult<WireArrayGeometry> out =
      run_live_subarray_graph(
      {{make_snapshot(1, 8, 2), changed, host_subarray({2, 3, 4})},
       {make_snapshot(1, 8, 2), changed, host_subarray({2, 3, 4}),
        /*geometry_skew_ns=*/1, /*subarray_skew_ns=*/1}},
      std::move(params), {.geometry = true, .subarray = true});

  ASSERT_TRUE(out.plan_ok) << out.plan_json;
  ASSERT_TRUE(out.diagnostics.empty());
  ASSERT_EQ(out.frames0.size(), 2U);
  ASSERT_EQ(out.frames1.size(), 2U);
  expect_channels(out.frames0[0], {2, 3, 4}, 2);
  expect_channels(out.frames0[1], {0, 1}, 2);
  EXPECT_EQ(y_set(unpack_output_geometry(out.frames1[0])),
            (std::set<double>{11.0, 11.5, 12.0}));
  EXPECT_EQ(y_set(unpack_output_geometry(out.frames1[1])),
            (std::set<double>{0.0, 0.5}));
}

TEST(SubarraySelectionOpTest, LiveOverridesRevertToParamsAfterExactMiss) {
  const auto base = make_linear({0, 1, 2, 3, 4, 5, 6, 7});
  auto changed = base;
  for (int64_t i = 0; i < changed.num_elements(); ++i) {
    changed.element_positions[static_cast<size_t>(i * 3 + 1)] += 30.0;
  }
  SubarraySelectionParams params = make_params(base, {0, 1});
  const TensorWireGraphRunResult<WireArrayGeometry> out =
      run_live_subarray_graph(
      {{make_snapshot(1, 8, 2), changed, host_subarray({4, 5, 6})},
       {make_snapshot(1, 8, 2), std::nullopt, nullptr}},
      std::move(params), {.geometry = true, .subarray = true});

  ASSERT_TRUE(out.plan_ok) << out.plan_json;
  ASSERT_TRUE(out.diagnostics.empty());
  ASSERT_EQ(out.frames0.size(), 2U);
  ASSERT_EQ(out.frames1.size(), 2U);
  expect_channels(out.frames0[0], {4, 5, 6}, 2);
  expect_channels(out.frames0[1], {0, 1}, 2);
  EXPECT_EQ(y_set(unpack_output_geometry(out.frames1[1])),
            (std::set<double>{0.0, 0.5}));
}

TEST(SubarraySelectionOpTest, RejectsMalformedLivePayloads) {
  const auto geom = make_linear({0, 1, 2, 3});
  SubarraySelectionParams geometry_params = make_params(geom, {0, 1});
  const TensorWireGraphRunResult<WireArrayGeometry> bad_geometry =
      run_live_subarray_graph(
      {{make_snapshot(1, 4, 2), geom, nullptr, 0, 0,
        /*malformed_geometry=*/true}},
      std::move(geometry_params), {.geometry = true});
  EXPECT_TRUE(bad_geometry.plan_ok) << bad_geometry.plan_json;
  EXPECT_TRUE(bad_geometry.frames0.empty());
  EXPECT_FALSE(bad_geometry.diagnostics.empty());

  SubarraySelectionParams subarray_params = make_params(geom, {0, 1});
  const TensorWireGraphRunResult<WireArrayGeometry> bad_subarray =
      run_live_subarray_graph(
      {{make_snapshot(1, 4, 2), std::nullopt,
        host_subarray({0, 1, 3})}},
      std::move(subarray_params), {.subarray = true});
  EXPECT_TRUE(bad_subarray.plan_ok) << bad_subarray.plan_json;
  EXPECT_TRUE(bad_subarray.frames0.empty());
  EXPECT_FALSE(bad_subarray.diagnostics.empty());
}

TEST(SubarraySelectionOpTest, RejectsLiveSubarrayBeyondCapturedBound) {
  const auto geom = make_linear({0, 1, 2, 3});
  SubarraySelectionParams params = make_params(geom, {0, 1});
  auto extents = default_test_extents();
  extents.max_subarray_channels = 2;
  const TensorWireGraphRunResult<WireArrayGeometry> out =
      run_live_subarray_graph(
      {{make_snapshot(1, 4, 2), std::nullopt,
        host_subarray({0, 1, 2})}},
      std::move(params), {.subarray = true}, extents);

  EXPECT_TRUE(out.plan_ok) << out.plan_json;
  EXPECT_TRUE(out.frames0.empty());
  EXPECT_FALSE(out.diagnostics.empty());
}

TEST(SubarraySelectionOpTest, RejectsSelectionBeyondCapturedOutputBound) {
  const auto geom = make_linear({0, 1, 2, 3});
  auto extents = default_test_extents();
  extents.max_subarray_channels = 2;
  const TensorWireGraphRunResult<WireArrayGeometry> out =
      run_select_result(
      make_snapshot(1, 4, 2), make_params(geom), extents);

  EXPECT_TRUE(out.plan_ok) << out.plan_json;
  EXPECT_TRUE(out.frames0.empty());
  EXPECT_FALSE(out.diagnostics.empty());
}

TEST(SubarraySelectionOpTest, PackUnpackRoundTrip) {
  auto geom = make_linear({0, 2, 4});
  geom.validate();
  const auto packed = pack_array_geometry(geom);
  const auto back = unpack_array_geometry(
      packed.data(), geom.num_elements() + 1, 3);
  EXPECT_EQ(back, geom);
}

}  // namespace
