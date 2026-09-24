// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <array>
#include <chrono>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <holoscan/memory/tensor_storage.hpp>
#include <holoscan/memory/unbounded_allocator.hpp>
#include <holoscan/testing/operator_test_harness.hpp>

#include "fmcw/clutter_filter_op/clutter_filter_op.hpp"
#include "radar_common/descriptors/enums.hpp"
#include "radar_common/tensor/tensor_utils.hpp"
#include "utils/tensor_fixtures.hpp"
#include "utils/test_extents.hpp"

namespace {

using Op = holoscan::radar::ClutterFilterOp<float>;
using holoscan::radar::ClutterFilterMethod;
using holoscan::radar::ClutterFilterParams;
using holoscan::radar::complex_dtype;
using holoscan::radar::testing::HostTensor;
using holoscan::radar::testing::default_test_extents;
using holoscan::radar::testing::to_device_tensor;
using holoscan::radar::testing::to_host;

constexpr std::size_t kMaximumSnapshotBytes = 1024U;
constexpr auto kObservationTimeout = std::chrono::seconds{5};

class EmptyTensorProducer final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.output(out, "out")
        .produces_tensor(holoscan::TensorOutputSpec{
            .representation = {.memory_kind = holoscan::MemoryKind::kHost,
                               .dtype = DLDataType{kDLInt, 32, 1},
                               .rank = 1U},
            .bounds = holoscan::tensor_bounds(4U),
            .storage = holoscan::TensorOutputStorage::kRuntimePool});
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = std::chrono::milliseconds{1}})
        .complete_after(holoscan::ComputeCalls{.limit = 1U});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext&) override {
    constexpr std::array<std::int64_t, 1> kShape{0};
    auto loan = out.allocate_tensor(kShape);
    if (!loan) {
      return holoscan::make_unexpected(std::move(loan).error());
    }
    auto committed = loan->write_host_and_commit([](const auto&) {});
    if (!committed) {
      return committed;
    }
    return out.emit(std::move(*loan));
  }

  holoscan::Output<holoscan::Tensor> out;
};

holoscan::Tensor device_frame(float offset = 0.0F) {
  std::vector<std::complex<float>> values(8U);
  for (std::size_t index = 0; index < values.size(); ++index) {
    values[index] = {offset + static_cast<float>(index), -static_cast<float>(index)};
  }
  const std::array<std::int64_t, 2> shape{4, 2};
  auto tensor = holoscan::make_tensor(shape, complex_dtype<float>(), std::nullopt,
                                      holoscan::MemoryStorageType::kDevice,
                                      std::make_shared<holoscan::UnboundedAllocator>());
  if (!tensor) {
    throw holoscan::RuntimeError{std::move(tensor).error()};
  }
  auto destination = tensor->mutable_data();
  if (!destination || cudaMemcpy(*destination, values.data(), values.size() * sizeof(values[0]),
                                 cudaMemcpyHostToDevice) != cudaSuccess) {
    throw std::runtime_error{"device_frame: upload failed"};
  }
  return std::move(*tensor);
}

void place_device_ports(holoscan::testing::OperatorTestHarness<Op>& harness) {
  harness.input_source_device("range_profiles", holoscan::DeviceId{0});
  harness.output_device("range_profiles_out", holoscan::DeviceId{0});
}

bool release_frame(holoscan::testing::RunningFixture& running) {
  const auto deadline = std::chrono::steady_clock::now() + kObservationTimeout;
  while (std::chrono::steady_clock::now() < deadline) {
    const auto released = running.release_frame();
    if (released) {
      return true;
    }
    if (released.error().code != holoscan::ErrorCode::kNotReady &&
        released.error().code != holoscan::ErrorCode::kBackpressured) {
      return false;
    }
    std::this_thread::yield();
  }
  return false;
}

TEST(NativeOperatorTesting, RunsWithConstructorParamsAndOwningSnapshot) {
  auto harness = holoscan::testing::create_operator_test<Op>(default_test_extents());
  harness.set_params(ClutterFilterParams{.method = ClutterFilterMethod::kNone});
  const holoscan::Tensor input = device_frame();
  harness.push(&Op::range_profiles, input);
  harness.capture_snapshot<holoscan::Tensor>("range_profiles_out", 1U, kMaximumSnapshotBytes);
  place_device_ports(harness);

  auto running = harness.run_async({.expect_stop = true});
  const bool observed = running.wait_for_output("range_profiles_out", 1U, kObservationTimeout);
  running.request_stop();
  const auto result = running.finish();

  ASSERT_TRUE(observed) << result.describe_failure();
  EXPECT_EQ(result.termination(), holoscan::RunTermination::kStopped);
  ASSERT_TRUE(result.ok()) << result.describe_failure();
  const HostTensor output = to_host(result.snapshot<holoscan::Tensor>("range_profiles_out"));
  EXPECT_EQ(output.shape, (std::vector<std::int64_t>{4, 2}));
  EXPECT_EQ(output.data.size(), input.nbytes());
}

TEST(NativeOperatorTesting, RepeatedRunsUseFreshOperatorsAndExplicitStaging) {
  auto harness = holoscan::testing::create_operator_test<Op>(default_test_extents());
  harness.set_params(ClutterFilterParams{.method = ClutterFilterMethod::kNone});
  harness.capture_snapshot<holoscan::Tensor>("range_profiles_out", 1U, kMaximumSnapshotBytes);
  place_device_ports(harness);

  harness.push(&Op::range_profiles, device_frame(0.0F));
  auto first_running = harness.run_async({.expect_stop = true});
  const bool first_observed =
      first_running.wait_for_output("range_profiles_out", 1U, kObservationTimeout);
  first_running.request_stop();
  const auto first = first_running.finish();
  ASSERT_TRUE(first_observed) << first.describe_failure();
  ASSERT_TRUE(first.ok()) << first.describe_failure();

  harness.push(&Op::range_profiles, device_frame(100.0F));
  auto second_running = harness.run_async({.expect_stop = true});
  const bool second_observed =
      second_running.wait_for_output("range_profiles_out", 1U, kObservationTimeout);
  second_running.request_stop();
  const auto second = second_running.finish();
  ASSERT_TRUE(second_observed) << second.describe_failure();
  ASSERT_TRUE(second.ok()) << second.describe_failure();

  EXPECT_NE(first.snapshot<holoscan::Tensor>("range_profiles_out").bytes(),
            second.snapshot<holoscan::Tensor>("range_profiles_out").bytes());
}

TEST(NativeOperatorTesting, GroupedExactOverrideFallsBackOnTheNextFrame) {
  using holoscan::testing::frame_at;
  using holoscan::testing::input;

  auto harness = holoscan::testing::create_operator_test<Op>(default_test_extents());
  harness.set_params(ClutterFilterParams{.method = ClutterFilterMethod::kNone});
  harness.push_frame(frame_at(std::chrono::milliseconds{1},
                              input("method", ClutterFilterMethod::kMeanSubtraction),
                              input("range_profiles", device_frame(0.0F))));
  harness.push_frame(
      frame_at(std::chrono::milliseconds{2}, input("range_profiles", device_frame(100.0F))));
  harness.capture_snapshot<holoscan::Tensor>("range_profiles_out", 2U, kMaximumSnapshotBytes);
  place_device_ports(harness);

  auto running = harness.run_async({.timeout = kObservationTimeout,
                                    .expect_stop = true,
                                    .frames = holoscan::testing::FrameRelease::kManual});
  const bool first_released = release_frame(running);
  const bool first_observed =
      running.wait_for_output("range_profiles_out", 1U, kObservationTimeout);
  const bool second_released = release_frame(running);
  const bool second_observed =
      running.wait_for_output("range_profiles_out", 2U, kObservationTimeout);
  running.request_stop();
  const auto result = running.finish();

  ASSERT_TRUE(first_released) << result.describe_failure();
  ASSERT_TRUE(first_observed) << result.describe_failure();
  ASSERT_TRUE(second_released) << result.describe_failure();
  ASSERT_TRUE(second_observed) << result.describe_failure();
  ASSERT_TRUE(result.ok()) << result.describe_failure();
  const auto outputs = result.snapshots<holoscan::Tensor>("range_profiles_out");
  ASSERT_EQ(outputs.size(), 2U);
  EXPECT_NE(outputs[0].bytes(), outputs[1].bytes());
}

TEST(NativeOperatorTesting, ZeroElementDeviceTensorStages) {
  auto harness = holoscan::testing::create_operator_test<Op>(default_test_extents());
  harness.set_params(ClutterFilterParams{.method = ClutterFilterMethod::kNone});
  auto empty = to_device_tensor(std::vector<std::complex<float>>{}, {0, 2}, complex_dtype<float>());

  EXPECT_NO_THROW(harness.push(&Op::range_profiles, *empty));
}

TEST(NativeOperatorTesting, ZeroElementOutputSnapshotsWithMetadata) {
  auto harness = holoscan::testing::create_operator_test<EmptyTensorProducer>();
  harness.capture_snapshot<holoscan::Tensor>("out", 1U, 1U);

  const auto result = harness.run();

  ASSERT_TRUE(result.ok()) << result.describe_failure();
  const auto& snapshot = result.snapshot<holoscan::Tensor>("out");
  EXPECT_EQ(snapshot.shape(), (std::vector<std::int64_t>{0}));
  EXPECT_EQ(snapshot.dtype().code, kDLInt);
  EXPECT_EQ(snapshot.dtype().bits, 32U);
  EXPECT_EQ(snapshot.dtype().lanes, 1U);
  EXPECT_EQ(snapshot.source_device().device_type, kDLCPU);
  EXPECT_TRUE(snapshot.bytes().empty());
}

}  // namespace
