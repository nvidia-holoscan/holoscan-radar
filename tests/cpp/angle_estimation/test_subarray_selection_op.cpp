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
#include <holoscan/testing/operator_test_harness.hpp>

#include "utils/fixture_completion.hpp"
#include "angle_estimation/beamscan_estimator_op/beamscan_estimator_op.hpp"
#include "angle_estimation/subarray_selection_op/subarray_selection_op.hpp"
#include "radar_common/descriptors/array_geometry.hpp"
#include "radar_common/descriptors/array_geometry_tensor.hpp"
#include "radar_common/sdk/tensor_io.hpp"
#include "radar_common/tensor/tensor_utils.hpp"
#include "radar_schemas/array_geometry_bridge.hpp"
#include "utils/integration_graph_utils.hpp"
#include "utils/test_extents.hpp"
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
using holoscan::radar::testing::HostTensor;
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
std::shared_ptr<holoscan::Tensor> make_snapshot(int64_t items, int64_t n, int64_t k) {
  std::vector<std::complex<double>> x(static_cast<size_t>(items * n * k), {0.0, 0.0});
  for (int64_t it = 0; it < items; ++it) {
    for (int64_t c = 0; c < n; ++c) {
      for (int64_t s = 0; s < k; ++s) {
        x[static_cast<size_t>((it * n + c) * k + s)] = {static_cast<double>(c),
                                                        static_cast<double>(s)};
      }
    }
  }
  return to_device_tensor(x, {items, n, k}, complex_dtype<double>());
}

struct Result {
  HostTensor snapshot;
  ArrayGeometry geometry;
};

holoscan::testing::TestResult run_select_result(
    std::shared_ptr<holoscan::Tensor> snap, SubarraySelectionParams params,
    const holoscan::radar::MaxTensorExtents& extents = default_test_extents()) {
  using Op = SubarraySelectionOp<double>;
  auto fixture = holoscan::testing::create_operator_test<Op>(extents);
  fixture.set_params(std::move(params));
  fixture.push(&Op::snapshot, *snap);
  constexpr std::size_t kMaximumBytes = 256U * 1024U * 1024U;
  fixture.capture_snapshot<holoscan::Tensor>("snapshot_out", 1U, kMaximumBytes);
  fixture.capture_snapshot<WireArrayGeometry>("array_geometry", 1U, kMaximumBytes);
  fixture.input_source_device("snapshot", holoscan::DeviceId{0});
  fixture.output_device("snapshot_out", holoscan::DeviceId{0});
  fixture.output_device("is_valid_out", holoscan::DeviceId{0});
  holoscan::radar::testing::FixtureCompletion completion;
  auto running = fixture.run_async(completion.run_options());
  (void)completion.wait_for_output(running, "snapshot_out", 1U);
  (void)completion.wait_for_output(running, "array_geometry", 1U);
  running.request_stop();
  auto fixture_result = running.finish();
  completion.check(fixture_result);
  return fixture_result;
}

std::vector<HostTensor> selected_snapshots(const holoscan::testing::TestResult& result) {
  std::vector<HostTensor> frames;
  for (const auto& snapshot : result.template snapshots<holoscan::Tensor>("snapshot_out")) {
    frames.push_back(holoscan::radar::testing::to_host(snapshot));
  }
  return frames;
}

std::vector<WireArrayGeometry> selected_geometries(const holoscan::testing::TestResult& result) {
  std::vector<WireArrayGeometry> geometries;
  for (const auto& snapshot : result.template snapshots<WireArrayGeometry>("array_geometry")) {
    auto wire = snapshot.metadata();
    const auto& packed =
        snapshot.tensor(holoscan::radar::wire::ArrayGeometryTensorFields::packed_geometry);
    std::vector<double> values(packed.bytes().size() / sizeof(double));
    std::memcpy(values.data(), packed.bytes().data(), packed.bytes().size());
    wire.packed_geometry =
        holoscan::radar::testing::to_pageable_host_tensor(values, packed.shape(), packed.dtype());
    geometries.push_back(std::move(wire));
  }
  return geometries;
}

Result run_select(std::shared_ptr<holoscan::Tensor> snap, SubarraySelectionParams params) {
  const auto result = run_select_result(std::move(snap), std::move(params));
  if (!result.ok()) {
    throw std::runtime_error(result.describe_failure());
  }
  const auto snapshots = selected_snapshots(result);
  const auto geometries = selected_geometries(result);
  if (snapshots.size() != 1U || geometries.size() != 1U) {
    throw std::runtime_error("SubarraySelectionOp test: expected one output frame pair");
  }
  auto geometry = from_wire(geometries.front());
  if (!geometry) {
    throw std::runtime_error("SubarraySelectionOp test: invalid WireArrayGeometry");
  }
  return {snapshots.front(), std::move(*geometry)};
}

holoscan::testing::TestResult run_snapshot_only(
    const std::vector<std::shared_ptr<holoscan::Tensor>>& snapshots,
    SubarraySelectionParams params) {
  using Op = SubarraySelectionOp<double>;
  auto fixture = holoscan::testing::create_operator_test<Op>(default_test_extents());
  fixture.set_params(std::move(params));
  for (const auto& snapshot : snapshots) {
    fixture.push(&Op::snapshot, *snapshot);
  }
  constexpr std::size_t kMaximumBytes = 256U * 1024U * 1024U;
  fixture.capture_snapshot<holoscan::Tensor>("snapshot_out", snapshots.size(), kMaximumBytes);
  fixture.input_source_device("snapshot", holoscan::DeviceId{0});
  fixture.output_device("snapshot_out", holoscan::DeviceId{0});
  fixture.output_device("is_valid_out", holoscan::DeviceId{0});
  holoscan::radar::testing::FixtureCompletion completion;
  auto running = fixture.run_async(completion.run_options());
  (void)completion.wait_for_output(running, "snapshot_out", snapshots.size());
  running.request_stop();
  auto fixture_result = running.finish();
  completion.check(fixture_result);
  return fixture_result;
}
std::set<double> y_set(const ArrayGeometry& g) {
  std::set<double> s;
  for (int64_t e = 0; e < g.num_elements(); ++e) {
    s.insert(g.element_positions[static_cast<size_t>(e * 3 + 1)]);
  }
  return s;
}

void expect_channels(const HostTensor& snapshot, const std::vector<int64_t>& channels,
                     int64_t num_snap) {
  const int64_t l = static_cast<int64_t>(channels.size());
  ASSERT_EQ(snapshot.shape, (std::vector<int64_t>{1, l, num_snap}));
  const auto* snap = snapshot.as<std::complex<double>>();
  for (int64_t j = 0; j < l; ++j) {
    for (int64_t s = 0; s < num_snap; ++s) {
      const auto& got = snap[static_cast<size_t>(j * num_snap + s)];
      EXPECT_NEAR(got.real(), static_cast<double>(channels[static_cast<size_t>(j)]), 1e-6)
          << "out channel " << j << " snapshot " << s;
      EXPECT_NEAR(got.imag(), static_cast<double>(s), 1e-6)
          << "out channel " << j << " snapshot " << s;
    }
  }
}

SubarraySelectionParams make_params(ArrayGeometry geom, std::vector<int64_t> subarray = {},
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

// Which override ports the graph wires, and so which control-plane sources
// it builds. The operator declares both entries unconditionally.
struct LiveSubarrayWiring {
  bool geometry{false};
  bool subarray{false};
};

holoscan::testing::TestResult run_live_subarray_graph(
    std::vector<LiveSubarrayTick> ticks, SubarraySelectionParams params,
    [[maybe_unused]] LiveSubarrayWiring wiring,
    holoscan::radar::MaxTensorExtents extents = default_test_extents()) {
  using Op = SubarraySelectionOp<double>;
  auto fixture = holoscan::testing::create_operator_test<Op>(extents);
  fixture.set_params(std::move(params));
  const bool use_independent_sources =
      std::any_of(ticks.begin(), ticks.end(), [](const LiveSubarrayTick& tick) {
        return tick.geometry_skew_ns != 0 || tick.subarray_skew_ns != 0;
      });
  for (std::size_t index = 0; index < ticks.size(); ++index) {
    const auto capture_time = std::chrono::milliseconds{static_cast<int64_t>((index + 1U) * 100U)};
    auto& tick = ticks[index];
    std::optional<WireArrayGeometry> wire;
    if (tick.geometry) {
      wire = to_wire(*tick.geometry);
      if (tick.malformed_geometry) {
        auto packed = pack_array_geometry(*tick.geometry);
        packed[packed.size() - 3U] = -1.0;
        wire->packed_geometry = holoscan::radar::testing::to_pageable_host_tensor(
            packed, {tick.geometry->num_elements() + 1, 3}, holoscan::radar::real_dtype<double>());
      }
    }
    if (use_independent_sources) {
      if (wire) {
        fixture.push_at("array_geometry_in", *wire,
                        capture_time + std::chrono::nanoseconds{tick.geometry_skew_ns});
      }
      if (tick.subarray) {
        fixture.push_at("subarray", *tick.subarray,
                        capture_time + std::chrono::nanoseconds{tick.subarray_skew_ns});
      }
      fixture.push_at("snapshot", *tick.snapshot, capture_time);
    } else if (wire && tick.subarray) {
      fixture.push_frame(holoscan::testing::frame_at(
          capture_time, holoscan::testing::input("array_geometry_in", *wire),
          holoscan::testing::input("subarray", *tick.subarray),
          holoscan::testing::input("snapshot", *tick.snapshot)));
    } else if (wire) {
      fixture.push_frame(holoscan::testing::frame_at(
          capture_time, holoscan::testing::input("array_geometry_in", *wire),
          holoscan::testing::input("snapshot", *tick.snapshot)));
    } else if (tick.subarray) {
      fixture.push_frame(holoscan::testing::frame_at(
          capture_time, holoscan::testing::input("subarray", *tick.subarray),
          holoscan::testing::input("snapshot", *tick.snapshot)));
    } else {
      fixture.push_frame(holoscan::testing::frame_at(
          capture_time, holoscan::testing::input("snapshot", *tick.snapshot)));
    }
  }
  constexpr std::size_t kMaximumBytes = 256U * 1024U * 1024U;
  fixture.capture_snapshot<holoscan::Tensor>("snapshot_out", ticks.size(), kMaximumBytes);
  fixture.capture_snapshot<WireArrayGeometry>("array_geometry", ticks.size(), kMaximumBytes);
  const holoscan::ConnectionOptions queue{.queue_depth = 8U};
  fixture.input_flow("snapshot", queue);
  if (wiring.geometry) {
    fixture.input_flow("array_geometry_in", queue);
  }
  if (wiring.subarray) {
    fixture.input_flow("subarray", queue);
  }
  fixture.input_source_device("snapshot", holoscan::DeviceId{0});
  fixture.output_device("snapshot_out", holoscan::DeviceId{0});
  fixture.output_device("is_valid_out", holoscan::DeviceId{0});
  holoscan::radar::testing::FixtureCompletion completion;
  auto running = fixture.run_async(completion.run_options());
  (void)completion.wait_for_output(running, "snapshot_out", ticks.size());
  (void)completion.wait_for_output(running, "array_geometry", ticks.size());
  running.request_stop();
  auto fixture_result = running.finish();
  completion.check(fixture_result);
  return fixture_result;
}
std::shared_ptr<holoscan::Tensor> host_subarray(const std::vector<int64_t>& values) {
  return to_pageable_host_tensor(values, {static_cast<int64_t>(values.size())},
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
  const auto out = run_select(snap, make_params(geom, {}, /*min_size=*/2));
  EXPECT_EQ(out.snapshot.shape, (std::vector<int64_t>{1, 4, 8}));
  EXPECT_EQ(out.geometry.num_elements(), 4);
  EXPECT_EQ(y_set(out.geometry), (std::set<double>{0.0, kD, 2 * kD, 3 * kD}));
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
  EXPECT_EQ(y_set(out.geometry), (std::set<double>{0.0, kD, 2 * kD, 3 * kD}));
}

TEST(SubarraySelectionOpTest, OptionalGeometryMayBeLeftUnwired) {
  const auto geom = make_linear({0, 1, 2, 3, 5, 6});
  const auto out =
      run_snapshot_only({make_snapshot(1, 6, 2), make_snapshot(1, 6, 2)}, make_params(geom));

  ASSERT_TRUE(out.ok()) << out.describe_failure();
  ASSERT_EQ(selected_snapshots(out).size(), 2U);
  expect_channels(selected_snapshots(out)[0], {0, 1, 2, 3}, 2);
  expect_channels(selected_snapshots(out)[1], {0, 1, 2, 3}, 2);
}

TEST(SubarraySelectionOpTest, ReducedGeometryDirectlyDrivesBeamscan) {
  namespace detail = holoscan::radar::testing::integration;
  detail::FrameBank::reset();
  detail::FrameBankB::reset();
  detail::FrameBank::input_frames = {
      to_device_tensor(std::vector<uint8_t>{0U}, {1, 1}, DLDataType{kDLUInt, 8U, 1U})};
  detail::FrameBankB::input_frames = {to_device_tensor(
      std::vector<std::complex<double>>(6U, {1.0, 0.0}), {1, 6}, complex_dtype<double>())};

  holoscan::Graph graph{"subarray-geometry-to-beamscan"};
  const auto source = graph.op<detail::DualTensorSourceOp>("source");
  auto extents = default_test_extents();
  auto selector = graph.op<SubarraySelectionOp<double>>("selector", extents);
  const ArrayGeometry full = make_linear({0, 1, 2, 3, 5, 6});
  selector.set_params(make_params(full, {0, 1, 2, 3}));

  auto beamscan = graph.op<BeamscanEstimatorOp<double>>("beamscan", extents);
  BeamscanParams beamscan_params;
  // Deliberately incompatible with the 4-channel selected snapshot.
  // Successful steering proves the same-tick wire geometry won over Params.
  beamscan_params.array_geometry = make_linear({0, 1, 2});
  beamscan_params.az_coordinates = {-0.2, 0.0, 0.2};
  beamscan_params.el_coordinates = {0.0};
  beamscan.set_params(std::move(beamscan_params));

  const auto aoa_sink = graph.op<detail::TensorCaptureOp>("aoa_sink");
  const auto spectrum_sink = graph.op<detail::TensorCaptureBOp>("spectrum_sink");
  const holoscan::ConnectionOptions queue{.queue_depth = 4U};
  graph.add_flow(source->out0, selector->is_valid, queue);
  graph.add_flow(source->out1, selector->snapshot, queue);
  graph.add_flow(selector->snapshot_out, beamscan->snapshot, queue);
  graph.add_flow(selector->array_geometry, beamscan->array_geometry, queue);
  graph.add_flow(selector->is_valid_out, beamscan->is_valid, queue);
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
    options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
        .operator_path = op, .output_port = port, .device = holoscan::DeviceId{0}});
  };
  bind("source", "out0");
  bind("source", "out1");
  bind("selector", "snapshot_out");
  bind("selector", "is_valid_out");
  bind("beamscan", "aoa_estimates");
  bind("beamscan", "angle_spectrum");

  const holoscan::ExecutionPlan plan = holoscan::compile(graph, std::move(options));
  ASSERT_TRUE(plan.ok()) << plan.json();
  holoscan::RunSession session = holoscan::run_async(plan);
  const auto deadline = std::chrono::steady_clock::now() + detail::kHarnessDeadline;
  while (session.diagnostics().empty() && std::chrono::steady_clock::now() < deadline) {
    bool aoa_captured = false;
    bool spectrum_captured = false;
    {
      const std::lock_guard lock(detail::FrameBank::mutex);
      aoa_captured = !detail::FrameBank::captured.empty();
    }
    {
      const std::lock_guard lock(detail::FrameBankB::mutex);
      spectrum_captured = !detail::FrameBankB::captured.empty();
    }
    if (aoa_captured && spectrum_captured) {
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
  // The forwarded all-invalid mask must arrive before snapshot_out. If it
  // misses the downstream Exact selector, SafeDefault would produce finite
  // boresight output instead.
  EXPECT_TRUE(std::isnan(aoa.as<double>()[0]));
  EXPECT_TRUE(std::isnan(aoa.as<double>()[1]));
}

TEST(SubarraySelectionOpTest, ExplicitSubarray) {
  const auto geom = make_linear({0, 1, 2, 3, 4, 5, 6, 7});
  auto snap = make_snapshot(1, 8, 3);
  const auto out = run_select(snap, make_params(geom, {2, 3, 4, 5}));
  EXPECT_EQ(out.snapshot.shape, (std::vector<int64_t>{1, 4, 3}));
  EXPECT_EQ(y_set(out.geometry), (std::set<double>{2 * kD, 3 * kD, 4 * kD, 5 * kD}));
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
  const auto exact =
      run_live_subarray_graph({{make_snapshot(1, 8, 2), changed, host_subarray({2, 3, 4})}},
                              make_params(base, {0, 1}), {.geometry = true, .subarray = true});
  const auto miss =
      run_live_subarray_graph({{make_snapshot(1, 8, 2), changed, host_subarray({2, 3, 4}),
                                /*geometry_skew_ns=*/1, /*subarray_skew_ns=*/1}},
                              make_params(base, {0, 1}), {.geometry = true, .subarray = true});

  ASSERT_TRUE(exact.ok()) << exact.describe_failure();
  ASSERT_TRUE(miss.ok()) << miss.describe_failure();
  ASSERT_EQ(selected_snapshots(exact).size(), 1U);
  ASSERT_EQ(selected_geometries(exact).size(), 1U);
  ASSERT_EQ(selected_snapshots(miss).size(), 1U);
  ASSERT_EQ(selected_geometries(miss).size(), 1U);
  expect_channels(selected_snapshots(exact)[0], {2, 3, 4}, 2);
  expect_channels(selected_snapshots(miss)[0], {0, 1}, 2);
  EXPECT_EQ(y_set(unpack_output_geometry(selected_geometries(exact)[0])),
            (std::set<double>{11.0, 11.5, 12.0}));
  EXPECT_EQ(y_set(unpack_output_geometry(selected_geometries(miss)[0])),
            (std::set<double>{0.0, 0.5}));
}

TEST(SubarraySelectionOpTest, LiveOverridesRevertToParamsAfterExactMiss) {
  const auto base = make_linear({0, 1, 2, 3, 4, 5, 6, 7});
  auto changed = base;
  for (int64_t i = 0; i < changed.num_elements(); ++i) {
    changed.element_positions[static_cast<size_t>(i * 3 + 1)] += 30.0;
  }
  SubarraySelectionParams params = make_params(base, {0, 1});
  const auto out =
      run_live_subarray_graph({{make_snapshot(1, 8, 2), changed, host_subarray({4, 5, 6})},
                               {make_snapshot(1, 8, 2), std::nullopt, nullptr}},
                              std::move(params), {.geometry = true, .subarray = true});

  ASSERT_TRUE(out.ok()) << out.describe_failure();
  ASSERT_EQ(selected_snapshots(out).size(), 2U);
  ASSERT_EQ(selected_geometries(out).size(), 2U);
  expect_channels(selected_snapshots(out)[0], {4, 5, 6}, 2);
  expect_channels(selected_snapshots(out)[1], {0, 1}, 2);
  EXPECT_EQ(y_set(unpack_output_geometry(selected_geometries(out)[1])),
            (std::set<double>{0.0, 0.5}));
}

TEST(SubarraySelectionOpTest, RejectsMalformedLivePayloads) {
  const auto geom = make_linear({0, 1, 2, 3});
  SubarraySelectionParams geometry_params = make_params(geom, {0, 1});
  const auto bad_geometry = run_live_subarray_graph({{make_snapshot(1, 4, 2), geom, nullptr, 0, 0,
                                                      /*malformed_geometry=*/true}},
                                                    std::move(geometry_params), {.geometry = true});
  EXPECT_FALSE(bad_geometry.ok());
  EXPECT_TRUE(selected_snapshots(bad_geometry).empty());
  EXPECT_FALSE(bad_geometry.diagnostics().empty());

  SubarraySelectionParams subarray_params = make_params(geom, {0, 1});
  const auto bad_subarray =
      run_live_subarray_graph({{make_snapshot(1, 4, 2), std::nullopt, host_subarray({0, 1, 3})}},
                              std::move(subarray_params), {.subarray = true});
  EXPECT_FALSE(bad_subarray.ok());
  EXPECT_TRUE(selected_snapshots(bad_subarray).empty());
  EXPECT_FALSE(bad_subarray.diagnostics().empty());
}

TEST(SubarraySelectionOpTest, RejectsLiveSubarrayBeyondCapturedBound) {
  const auto geom = make_linear({0, 1, 2, 3});
  SubarraySelectionParams params = make_params(geom, {0, 1});
  auto extents = default_test_extents();
  extents.max_subarray_channels = 2;
  const auto out =
      run_live_subarray_graph({{make_snapshot(1, 4, 2), std::nullopt, host_subarray({0, 1, 2})}},
                              std::move(params), {.subarray = true}, extents);

  EXPECT_FALSE(out.ok());
  EXPECT_TRUE(selected_snapshots(out).empty());
  EXPECT_FALSE(out.diagnostics().empty());
}

TEST(SubarraySelectionOpTest, RejectsSelectionBeyondCapturedOutputBound) {
  const auto geom = make_linear({0, 1, 2, 3});
  auto extents = default_test_extents();
  extents.max_subarray_channels = 2;
  const auto out = run_select_result(make_snapshot(1, 4, 2), make_params(geom), extents);

  EXPECT_FALSE(out.ok());
  EXPECT_TRUE(selected_snapshots(out).empty());
  EXPECT_FALSE(out.diagnostics().empty());
}

TEST(SubarraySelectionOpTest, PackUnpackRoundTrip) {
  auto geom = make_linear({0, 2, 4});
  geom.validate();
  const auto packed = pack_array_geometry(geom);
  const auto back = unpack_array_geometry(packed.data(), geom.num_elements() + 1, 3);
  EXPECT_EQ(back, geom);
}

}  // namespace
