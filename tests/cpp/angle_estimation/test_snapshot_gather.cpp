// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// SnapshotGatherOp contract tests (HSDK 5 Graph harness): RD neighborhood
// gather → snapshot matrix, edge modes, batch selection, Params coverage,
// and error paths.

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/testing/operator_test_harness.hpp>

#include "utils/fixture_completion.hpp"
#include "angle_estimation/snapshot_gather_op/snapshot_gather_op.hpp"
#include "radar_common/descriptors/enums.hpp"
#include "radar_common/descriptors/max_tensor_extents.hpp"
#include "radar_common/tensor/tensor_utils.hpp"
#include "utils/test_extents.hpp"
#include "utils/tensor_fixtures.hpp"

namespace {

using holoscan::radar::SnapshotEdgeMode;
using holoscan::radar::SnapshotGatherOp;
using holoscan::radar::SnapshotGatherParams;
using holoscan::radar::SnapshotWindowOverride;
using holoscan::radar::complex_dtype;
using holoscan::radar::testing::HostTensor;
using holoscan::radar::testing::to_device_tensor;

constexpr int64_t kC = 3;
constexpr int64_t kD = 8;
constexpr int64_t kR = 16;

std::complex<double> cell_value(int64_t c, int64_t d, int64_t r) {
  return {static_cast<double>(c * 100 + d * 10 + r), static_cast<double>(c + d + r)};
}

template <typename T>
std::shared_ptr<holoscan::Tensor> make_rd_maps() {
  std::vector<std::complex<T>> host(static_cast<size_t>(kC * kD * kR));
  for (int64_t c = 0; c < kC; ++c) {
    for (int64_t d = 0; d < kD; ++d) {
      for (int64_t r = 0; r < kR; ++r) {
        const auto v = cell_value(c, d, r);
        host[static_cast<size_t>((c * kD + d) * kR + r)] =
            std::complex<T>(static_cast<T>(v.real()), static_cast<T>(v.imag()));
      }
    }
  }
  return to_device_tensor(host, {kC, kD, kR}, complex_dtype<T>());
}

std::shared_ptr<holoscan::Tensor> make_rd_maps() {
  return make_rd_maps<double>();
}

template <typename T>
std::shared_ptr<holoscan::Tensor> make_batched_rd_maps(int64_t batch) {
  std::vector<std::complex<T>> host(static_cast<size_t>(batch * kC * kD * kR));
  for (int64_t b = 0; b < batch; ++b) {
    for (int64_t c = 0; c < kC; ++c) {
      for (int64_t d = 0; d < kD; ++d) {
        for (int64_t r = 0; r < kR; ++r) {
          const size_t off = static_cast<size_t>(((b * kC + c) * kD + d) * kR + r);
          const auto v = cell_value(c, d, r);
          // Offset batch slice so batch selection is observable.
          host[off] = std::complex<T>(static_cast<T>(v.real() + static_cast<double>(b) * 1000.0),
                                      static_cast<T>(v.imag()));
        }
      }
    }
  }
  return to_device_tensor(host, {batch, kC, kD, kR}, complex_dtype<T>());
}

struct GatherOutputs {
  HostTensor snapshot;
  HostTensor is_valid;
};

template <typename T>
holoscan::testing::TestResult run_gather_result(
    std::shared_ptr<holoscan::Tensor> rd_maps, std::shared_ptr<holoscan::Tensor> detection_idx,
    SnapshotGatherParams params = {}, bool capture_validity = true,
    const holoscan::radar::MaxTensorExtents& extents =
        holoscan::radar::testing::default_test_extents()) {
  using Op = SnapshotGatherOp<T>;
  auto fixture = holoscan::testing::create_operator_test<Op>(extents);
  fixture.set_params(std::move(params));
  fixture.push_frame(holoscan::testing::frame_at(
      std::chrono::nanoseconds{0}, holoscan::testing::input("rd_maps", *rd_maps),
      holoscan::testing::input("detection_idx", *detection_idx)));
  constexpr std::size_t kMaximumBytes = 256U * 1024U * 1024U;
  fixture.template capture_snapshot<holoscan::Tensor>("snapshot", 1U, kMaximumBytes);
  if (capture_validity) {
    fixture.template capture_snapshot<holoscan::Tensor>("is_valid", 1U, kMaximumBytes);
  }
  fixture.input_source_device("rd_maps", holoscan::DeviceId{0});
  fixture.input_source_device("detection_idx", holoscan::DeviceId{0});
  fixture.output_device("snapshot", holoscan::DeviceId{0});
  fixture.output_device("is_valid", holoscan::DeviceId{0});
  holoscan::radar::testing::FixtureCompletion completion;
  auto running = fixture.run_async(completion.run_options());
  const bool observed = completion.wait_for_output(running, "snapshot", 1U);
  (void)observed;
  running.request_stop();
  auto fixture_result = running.finish();
  completion.check(fixture_result);
  return fixture_result;
}

template <typename T>
GatherOutputs run_gather(std::shared_ptr<holoscan::Tensor> rd_maps,
                         std::shared_ptr<holoscan::Tensor> detection_idx,
                         SnapshotGatherParams params = {},
                         const holoscan::radar::MaxTensorExtents& extents =
                             holoscan::radar::testing::default_test_extents()) {
  const auto result = run_gather_result<T>(std::move(rd_maps), std::move(detection_idx),
                                           std::move(params), true, extents);
  if (!result.ok()) {
    throw std::runtime_error(result.describe_failure());
  }
  return {
      holoscan::radar::testing::to_host(result.template snapshot<holoscan::Tensor>("snapshot")),
      holoscan::radar::testing::to_host(result.template snapshot<holoscan::Tensor>("is_valid"))};
}

TEST(SnapshotGatherOpTest, SingleCellGather) {
  std::vector<int64_t> idx = {3, 5};
  auto idx_t = to_device_tensor(idx, {1, 2}, DLDataType{kDLInt, 64, 1});
  const auto out = run_gather<double>(make_rd_maps(), idx_t);

  ASSERT_EQ(out.snapshot.shape, (std::vector<int64_t>{1, kC, 1}));
  const auto* snap = out.snapshot.as<std::complex<double>>();
  for (int64_t c = 0; c < kC; ++c) {
    EXPECT_NEAR(snap[c].real(), cell_value(c, 3, 5).real(), 1e-6);
    EXPECT_NEAR(snap[c].imag(), cell_value(c, 3, 5).imag(), 1e-6);
  }
  ASSERT_EQ(out.is_valid.shape, (std::vector<int64_t>{1, 1}));
  EXPECT_EQ(out.is_valid.as<uint8_t>()[0], 1);
}

TEST(SnapshotGatherOpTest, WindowGatherPreservesChannelAxis) {
  std::vector<int64_t> idx = {3, 5};
  auto idx_t = to_device_tensor(idx, {1, 2}, DLDataType{kDLInt, 64, 1});
  SnapshotGatherParams params{};
  params.snapshot_window = {1, 0};
  const auto out = run_gather<double>(make_rd_maps(), idx_t, params);

  ASSERT_EQ(out.snapshot.shape, (std::vector<int64_t>{1, kC, 3}));
  const auto* snap = out.snapshot.as<std::complex<double>>();
  constexpr int64_t kK = 3;
  for (int64_t slot = 0; slot < kK; ++slot) {
    const int64_t d = 2 + slot;  // centers 3 with half=1 under wrap
    for (int64_t c = 0; c < kC; ++c) {
      // (num_detections, num_channels, num_snapshots) row-major.
      const auto& got = snap[static_cast<size_t>(c * kK + slot)];
      const auto exp = cell_value(c, d, 5);
      EXPECT_NEAR(got.real(), exp.real(), 1e-6) << "slot " << slot << " ch " << c;
      EXPECT_NEAR(got.imag(), exp.imag(), 1e-6);
    }
  }
}

TEST(SnapshotGatherOpTest, SnapshotWindowConfigurableFromParams) {
  // Authoring Params (set_params) must configure snapshot_window.
  std::vector<int64_t> idx = {3, 5};
  auto idx_t = to_device_tensor(idx, {1, 2}, DLDataType{kDLInt, 64, 1});
  SnapshotGatherParams params{};
  params.snapshot_window = {1, 0};
  const auto out = run_gather<double>(make_rd_maps(), idx_t, params);
  ASSERT_EQ(out.snapshot.shape, (std::vector<int64_t>{1, kC, 3}));
}

TEST(SnapshotGatherOpTest, ClampMarksInvalidSlots) {
  std::vector<int64_t> idx = {0, 0};
  auto idx_t = to_device_tensor(idx, {1, 2}, DLDataType{kDLInt, 64, 1});
  SnapshotGatherParams params{};
  params.snapshot_window = {1, 1};
  params.doppler_edge = SnapshotEdgeMode::kClamp;
  params.range_edge = SnapshotEdgeMode::kClamp;
  const auto out = run_gather<double>(make_rd_maps(), idx_t, params);

  ASSERT_EQ(out.snapshot.shape, (std::vector<int64_t>{1, kC, 9}));
  const uint8_t* valid = out.is_valid.as<uint8_t>();
  int valid_count = 0;
  for (int64_t i = 0; i < 9; ++i) {
    valid_count += valid[static_cast<size_t>(i)] ? 1 : 0;
  }
  EXPECT_EQ(valid_count, 4);
}

TEST(SnapshotGatherOpTest, ShiftKeepsSnapshotCount) {
  std::vector<int64_t> idx = {0, 0};
  auto idx_t = to_device_tensor(idx, {1, 2}, DLDataType{kDLInt, 64, 1});
  SnapshotGatherParams params{};
  params.snapshot_window = {1, 1};
  params.doppler_edge = SnapshotEdgeMode::kShift;
  params.range_edge = SnapshotEdgeMode::kShift;
  const auto out = run_gather<double>(make_rd_maps(), idx_t, params);

  ASSERT_EQ(out.snapshot.shape, (std::vector<int64_t>{1, kC, 9}));
  const auto* valid = out.is_valid.as<uint8_t>();
  for (int64_t i = 0; i < 9; ++i) {
    EXPECT_EQ(valid[static_cast<size_t>(i)], 1);
  }
  const auto* snap = out.snapshot.as<std::complex<double>>();
  // (num_detections, num_channels, num_snapshots): channel 0's slot 0 and last
  // slot (8) are at c*K + slot = 0 and 8 for K == 9.
  EXPECT_NEAR(snap[0].real(), cell_value(0, 0, 0).real(), 1e-6);
  EXPECT_NEAR(snap[static_cast<size_t>(8)].real(), cell_value(0, 2, 2).real(), 1e-6);
}

TEST(SnapshotGatherOpTest, EmptyDetections) {
  std::vector<int64_t> idx;
  auto idx_t = to_device_tensor(idx, {0, 2}, DLDataType{kDLInt, 64, 1});
  const auto out = run_gather<double>(make_rd_maps(), idx_t);
  EXPECT_EQ(out.snapshot.shape[0], 0);
  EXPECT_EQ(out.is_valid.shape[0], 0);
}

TEST(SnapshotGatherOpTest, OptionalValidityMayBeLeftUnwired) {
  std::vector<int64_t> idx = {3, 5};
  auto idx_t = to_device_tensor(idx, {1, 2}, DLDataType{kDLInt, 64, 1});
  const auto result =
      run_gather_result<double>(make_rd_maps(), std::move(idx_t), SnapshotGatherParams{}, false);
  ASSERT_TRUE(result.ok()) << result.describe_failure();
  const auto snapshot =
      holoscan::radar::testing::to_host(result.snapshot<holoscan::Tensor>("snapshot"));
  EXPECT_EQ(snapshot.shape, (std::vector<int64_t>{1, kC, 1}));
}

TEST(SnapshotGatherOpTest, LiveSnapshotWindowAppliesForOneTick) {
  using Op = SnapshotGatherOp<double>;
  auto fixture =
      holoscan::testing::create_operator_test<Op>(holoscan::radar::testing::default_test_extents());
  fixture.set_params(SnapshotGatherParams{});
  std::vector<std::shared_ptr<holoscan::Tensor>> maps;
  std::vector<std::shared_ptr<holoscan::Tensor>> indices;
  for (int tick = 0; tick < 3; ++tick) {
    maps.push_back(make_rd_maps());
    indices.push_back(
        to_device_tensor(std::vector<int64_t>{3, 5}, {1, 2}, DLDataType{kDLInt, 64, 1}));
  }
  fixture.push_frame(holoscan::testing::frame_at(
      std::chrono::milliseconds{100}, holoscan::testing::input("rd_maps", *maps[0]),
      holoscan::testing::input("detection_idx", *indices[0])));
  fixture.push_frame(holoscan::testing::frame_at(
      std::chrono::milliseconds{200},
      holoscan::testing::input("snapshot_window", SnapshotWindowOverride{1, 0}),
      holoscan::testing::input("rd_maps", *maps[1]),
      holoscan::testing::input("detection_idx", *indices[1])));
  fixture.push_frame(holoscan::testing::frame_at(
      std::chrono::milliseconds{300}, holoscan::testing::input("rd_maps", *maps[2]),
      holoscan::testing::input("detection_idx", *indices[2])));
  constexpr std::size_t kMaximumBytes = 256U * 1024U * 1024U;
  fixture.capture_snapshot<holoscan::Tensor>("snapshot", 3U, kMaximumBytes);
  fixture.input_source_device("rd_maps", holoscan::DeviceId{0});
  fixture.input_source_device("detection_idx", holoscan::DeviceId{0});
  fixture.output_device("snapshot", holoscan::DeviceId{0});
  fixture.output_device("is_valid", holoscan::DeviceId{0});
  holoscan::radar::testing::FixtureCompletion completion;
  auto running =
      fixture.run_async(completion.run_options(holoscan::testing::FrameRelease::kManual));
  for (std::size_t count = 1; count <= 3; ++count) {
    const auto released = running.release_frame(completion.remaining());
    ASSERT_TRUE(released) << released.error().message;
    ASSERT_TRUE(completion.wait_for_output(running, "snapshot", count));
  }
  running.request_stop();
  const auto result = running.finish();
  completion.check(result);
  ASSERT_TRUE(result.ok()) << result.describe_failure();
  const auto frames = result.snapshots<holoscan::Tensor>("snapshot");
  ASSERT_EQ(frames.size(), 3U);
  EXPECT_EQ(frames[0].shape(), (std::vector<int64_t>{1, kC, 1}));
  EXPECT_EQ(frames[1].shape(), (std::vector<int64_t>{1, kC, 3}));
  EXPECT_EQ(frames[2].shape(), (std::vector<int64_t>{1, kC, 1}));
}

TEST(SnapshotGatherOpTest, FloatInstantiationSmoke) {
  std::vector<int64_t> idx = {3, 5};
  auto idx_t = to_device_tensor(idx, {1, 2}, DLDataType{kDLInt, 64, 1});
  const auto out = run_gather<float>(make_rd_maps<float>(), idx_t);
  ASSERT_EQ(out.snapshot.shape, (std::vector<int64_t>{1, kC, 1}));
  const auto* snap = out.snapshot.as<std::complex<float>>();
  EXPECT_NEAR(snap[0].real(), static_cast<float>(cell_value(0, 3, 5).real()), 1e-5f);
}

TEST(SnapshotGatherOpTest, FloatAndDoubleAgree) {
  std::vector<int64_t> idx = {3, 5};
  auto idx_t = to_device_tensor(idx, {1, 2}, DLDataType{kDLInt, 64, 1});
  SnapshotGatherParams params{};
  params.snapshot_window = {1, 0};
  const auto out_f = run_gather<float>(make_rd_maps<float>(), idx_t, params);
  // Fresh idx tensor — previous run_gather consumed the shared_ptr.
  auto idx_t2 = to_device_tensor(idx, {1, 2}, DLDataType{kDLInt, 64, 1});
  const auto out_d = run_gather<double>(make_rd_maps<double>(), idx_t2, params);
  ASSERT_EQ(out_f.snapshot.shape, out_d.snapshot.shape);
  const auto* f_data = out_f.snapshot.as<std::complex<float>>();
  const auto* d_data = out_d.snapshot.as<std::complex<double>>();
  for (int64_t i = 0; i < out_d.snapshot.size(); ++i) {
    ASSERT_NEAR(std::abs(std::complex<double>(f_data[i]) - d_data[i]), 0.0, 1e-5) << "index " << i;
  }
}

TEST(SnapshotGatherOpTest, DopplerWrapAtBoundary) {
  // Center at doppler bin 0; half=1 under wrap yields bins {7, 0, 1}.
  std::vector<int64_t> idx = {0, 5};
  auto idx_t = to_device_tensor(idx, {1, 2}, DLDataType{kDLInt, 64, 1});
  SnapshotGatherParams params{};
  params.snapshot_window = {1, 0};
  params.doppler_edge = SnapshotEdgeMode::kWrap;
  const auto out = run_gather<double>(make_rd_maps(), idx_t, params);

  ASSERT_EQ(out.snapshot.shape, (std::vector<int64_t>{1, kC, 3}));
  const auto* snap = out.snapshot.as<std::complex<double>>();
  constexpr int64_t kK = 3;
  const int64_t d_bins[] = {7, 0, 1};
  for (int slot = 0; slot < kK; ++slot) {
    for (int64_t c = 0; c < kC; ++c) {
      // (num_detections, num_channels, num_snapshots) row-major.
      const auto& got = snap[static_cast<size_t>(c * kK + slot)];
      const auto exp = cell_value(c, d_bins[slot], 5);
      EXPECT_NEAR(got.real(), exp.real(), 1e-6) << "slot " << slot << " ch " << c;
    }
  }
}

TEST(SnapshotGatherOpTest, PreservedBatchSelectsCorrectSlice) {
  constexpr int64_t kB = 2;
  std::vector<int64_t> idx = {
      0, 3, 5,  // batch 0
      1, 3, 5,  // batch 1
  };
  auto idx_t = to_device_tensor(idx, {2, 3}, DLDataType{kDLInt, 64, 1});
  const auto out = run_gather<double>(make_batched_rd_maps<double>(kB), idx_t);

  ASSERT_EQ(out.snapshot.shape, (std::vector<int64_t>{2, kC, 1}));
  const auto* snap = out.snapshot.as<std::complex<double>>();
  for (int64_t c = 0; c < kC; ++c) {
    EXPECT_NEAR(snap[c].real(), cell_value(c, 3, 5).real() + 0.0, 1e-6);
    EXPECT_NEAR(snap[static_cast<size_t>(kC + c)].real(), cell_value(c, 3, 5).real() + 1000.0,
                1e-6);
  }
}

TEST(SnapshotGatherOpTest, RejectsRdMapsDtypeMismatch) {
  std::vector<int64_t> idx = {3, 5};
  auto idx_t = to_device_tensor(idx, {1, 2}, DLDataType{kDLInt, 64, 1});
  const auto result = run_gather_result<double>(make_rd_maps<float>(), idx_t);
  EXPECT_FALSE(result.ok());
  EXPECT_TRUE(result.snapshots<holoscan::Tensor>("snapshot").empty());
  EXPECT_FALSE(result.diagnostics().empty());
}

TEST(SnapshotGatherOpTest, RejectsNarrowDetectionIdx) {
  std::vector<int64_t> idx = {3};
  auto idx_t = to_device_tensor(idx, {1, 1}, DLDataType{kDLInt, 64, 1});
  const auto result = run_gather_result<double>(make_rd_maps(), idx_t);
  EXPECT_FALSE(result.ok());
  EXPECT_TRUE(result.snapshots<holoscan::Tensor>("snapshot").empty());
  EXPECT_FALSE(result.diagnostics().empty());
}

TEST(SnapshotGatherOpTest, RejectsNegativeSnapshotWindow) {
  std::vector<int64_t> idx = {3, 5};
  auto idx_t = to_device_tensor(idx, {1, 2}, DLDataType{kDLInt, 64, 1});
  SnapshotGatherParams params{};
  params.snapshot_window = {-1, 0};
  const auto result = run_gather_result<double>(make_rd_maps(), idx_t, params);
  EXPECT_FALSE(result.ok());
  EXPECT_TRUE(result.snapshots<holoscan::Tensor>("snapshot").empty());
  EXPECT_FALSE(result.diagnostics().empty());
}

TEST(SnapshotGatherOpTest, RejectsNegativeIsotropicSnapshotWindow) {
  std::vector<int64_t> idx = {3, 5};
  auto idx_t = to_device_tensor(idx, {1, 2}, DLDataType{kDLInt, 64, 1});
  SnapshotGatherParams params{};
  params.snapshot_window = {-1};
  const auto result = run_gather_result<double>(make_rd_maps(), idx_t, params);
  EXPECT_FALSE(result.ok());
  EXPECT_TRUE(result.snapshots<holoscan::Tensor>("snapshot").empty());
  EXPECT_FALSE(result.diagnostics().empty());
}

TEST(SnapshotGatherOpTest, RejectsOversizedShiftWindow) {
  std::vector<int64_t> idx = {3, 5};
  auto idx_t = to_device_tensor(idx, {1, 2}, DLDataType{kDLInt, 64, 1});
  SnapshotGatherParams params{};
  params.snapshot_window = {0, kR};
  params.range_edge = SnapshotEdgeMode::kShift;
  const auto result = run_gather_result<double>(make_rd_maps(), idx_t, params);
  EXPECT_FALSE(result.ok());
  EXPECT_TRUE(result.snapshots<holoscan::Tensor>("snapshot").empty());
  EXPECT_FALSE(result.diagnostics().empty());
}

TEST(SnapshotGatherOpTest, RejectsBatchCoordOutOfRange) {
  std::vector<int64_t> idx = {9, 3, 5};
  auto idx_t = to_device_tensor(idx, {1, 3}, DLDataType{kDLInt, 64, 1});
  const auto result = run_gather_result<double>(make_batched_rd_maps<double>(2), idx_t);
  EXPECT_FALSE(result.ok());
  EXPECT_TRUE(result.snapshots<holoscan::Tensor>("snapshot").empty());
  EXPECT_FALSE(result.diagnostics().empty());
}

TEST(SnapshotGatherOpTest, RejectsCellCoordOutOfRange) {
  std::vector<int64_t> idx = {kD, 5};
  auto idx_t = to_device_tensor(idx, {1, 2}, DLDataType{kDLInt, 64, 1});
  const auto result = run_gather_result<double>(make_rd_maps(), idx_t);
  EXPECT_FALSE(result.ok());
  EXPECT_TRUE(result.snapshots<holoscan::Tensor>("snapshot").empty());
  EXPECT_FALSE(result.diagnostics().empty());
}

// Copy-only changed path: independently enumerate the ordered neighborhood,
// including slots above the 128-thread block and all three edge policies.
template <typename T>
class SnapshotGatherSlotParallel : public ::testing::Test {};
using SnapshotGatherSlotTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(SnapshotGatherSlotParallel, SnapshotGatherSlotTypes);

TYPED_TEST(SnapshotGatherSlotParallel, AsymmetricWideWindowPreservesEveryCellAndMask) {
  using T = TypeParam;
  constexpr int64_t channels = 3;
  constexpr int64_t doppler = 11;
  constexpr int64_t range = 21;
  constexpr int64_t d_half = 4;
  constexpr int64_t r_half = 8;
  constexpr int64_t slots = 153;
  const std::vector<int64_t> centers{0, 0, 5, 10, 10, 20};
  std::vector<std::complex<T>> input(static_cast<size_t>(channels * doppler * range));
  for (int64_t c = 0; c < channels; ++c) {
    for (int64_t d = 0; d < doppler; ++d) {
      for (int64_t r = 0; r < range; ++r) {
        const int64_t marker = 10000 * c + 100 * d + r + 1;
        const int64_t sign = (c + d + r) % 2 == 0 ? 1 : -1;
        input[static_cast<size_t>((c * doppler + d) * range + r)] = {
            static_cast<T>(sign * marker), static_cast<T>(-sign * (marker + 17))};
      }
    }
  }
  auto extents = holoscan::radar::testing::default_test_extents();
  extents.max_channels = channels;
  extents.max_snapshots = slots;
  extents.max_detections = 3;
  extents.max_items = 3;
  for (const auto mode :
       {SnapshotEdgeMode::kWrap, SnapshotEdgeMode::kClamp, SnapshotEdgeMode::kShift}) {
    SnapshotGatherParams params{};
    params.snapshot_window = {d_half, r_half};
    params.doppler_edge = mode;
    params.range_edge = mode;
    const auto result = run_gather<T>(
        to_device_tensor(input, {channels, doppler, range}, complex_dtype<T>()),
        to_device_tensor(centers, {3, 2}, DLDataType{kDLInt, 64, 1}), params, extents);
    ASSERT_EQ(result.snapshot.shape, (std::vector<int64_t>{3, channels, slots}));
    ASSERT_EQ(result.is_valid.shape, (std::vector<int64_t>{3, slots}));
    const auto* data = result.snapshot.template as<std::complex<T>>();
    const auto* valid = result.is_valid.template as<uint8_t>();
    // Host coordinate lists enumerate the contract independently of flattened GPU indexing.
    const auto axis = [mode](int64_t center, int64_t half, int64_t size) {
      std::vector<int64_t> coordinates;
      if (mode == SnapshotEdgeMode::kShift) {
        const int64_t start = std::min(std::max(center - half, int64_t{0}), size - (2 * half + 1));
        for (int64_t value = start; value <= start + 2 * half; ++value) {
          coordinates.push_back(value);
        }
      } else {
        for (int64_t value = center - half; value <= center + half; ++value) {
          coordinates.push_back(mode == SnapshotEdgeMode::kWrap
                                    ? (value % size + size) % size
                                    : (value < 0 || value >= size ? -1 : value));
        }
      }
      return coordinates;
    };
    for (int64_t row = 0; row < 3; ++row) {
      const auto ds = axis(centers[static_cast<size_t>(2 * row)], d_half, doppler);
      const auto rs = axis(centers[static_cast<size_t>(2 * row + 1)], r_half, range);
      int64_t slot = 0;
      for (const int64_t d : ds) {
        for (const int64_t r : rs) {
          const bool present = d >= 0 && r >= 0;
          EXPECT_EQ(valid[row * slots + slot], static_cast<uint8_t>(present));
          for (int64_t c = 0; c < channels; ++c) {
            const std::complex<T> expected =
                present ? input[static_cast<size_t>((c * doppler + d) * range + r)]
                        : std::complex<T>{};
            EXPECT_EQ(data[(row * channels + c) * slots + slot], expected)
                << "row=" << row << " channel=" << c << " slot=" << slot;
          }
          ++slot;
        }
      }
      EXPECT_EQ(slot, slots);
    }
  }
}

TYPED_TEST(SnapshotGatherSlotParallel, LiveWindowPointNeighborhoodPointUsesCurrentTick) {
  using T = TypeParam;
  using Op = SnapshotGatherOp<T>;
  auto fixture =
      holoscan::testing::create_operator_test<Op>(holoscan::radar::testing::default_test_extents());
  fixture.set_params(SnapshotGatherParams{});
  std::vector<std::shared_ptr<holoscan::Tensor>> maps;
  std::vector<std::shared_ptr<holoscan::Tensor>> indices;
  for (int tick = 0; tick < 3; ++tick) {
    maps.push_back(make_rd_maps<T>());
    indices.push_back(
        to_device_tensor(std::vector<int64_t>{3, 5}, {1, 2}, DLDataType{kDLInt, 64, 1}));
  }
  fixture.push_frame(holoscan::testing::frame_at(
      std::chrono::milliseconds{100}, holoscan::testing::input("rd_maps", *maps[0]),
      holoscan::testing::input("detection_idx", *indices[0])));
  fixture.push_frame(holoscan::testing::frame_at(
      std::chrono::milliseconds{200},
      holoscan::testing::input("snapshot_window", SnapshotWindowOverride{1, 0}),
      holoscan::testing::input("rd_maps", *maps[1]),
      holoscan::testing::input("detection_idx", *indices[1])));
  fixture.push_frame(holoscan::testing::frame_at(
      std::chrono::milliseconds{300}, holoscan::testing::input("rd_maps", *maps[2]),
      holoscan::testing::input("detection_idx", *indices[2])));
  constexpr std::size_t kMaximumBytes = 256U * 1024U * 1024U;
  fixture.template capture_snapshot<holoscan::Tensor>("snapshot", 3U, kMaximumBytes);
  fixture.template capture_snapshot<holoscan::Tensor>("is_valid", 3U, kMaximumBytes);
  fixture.input_source_device("rd_maps", holoscan::DeviceId{0});
  fixture.input_source_device("detection_idx", holoscan::DeviceId{0});
  fixture.output_device("snapshot", holoscan::DeviceId{0});
  fixture.output_device("is_valid", holoscan::DeviceId{0});
  auto running =
      fixture.run_async({.expect_stop = true, .frames = holoscan::testing::FrameRelease::kManual});
  for (std::size_t count = 1; count <= 3; ++count) {
    const auto released = running.release_frame(std::chrono::seconds{5});
    ASSERT_TRUE(released) << released.error().message;
    ASSERT_TRUE(running.wait_for_output("snapshot", count, std::chrono::seconds{5}));
    ASSERT_TRUE(running.wait_for_output("is_valid", count, std::chrono::seconds{5}));
  }
  running.request_stop();
  const auto result = running.finish();
  ASSERT_TRUE(result.ok()) << result.describe_failure();
  const auto frames = result.template snapshots<holoscan::Tensor>("snapshot");
  const auto masks = result.template snapshots<holoscan::Tensor>("is_valid");
  ASSERT_EQ(frames.size(), 3U);
  ASSERT_EQ(masks.size(), 3U);
  for (size_t tick = 0; tick < 3; ++tick) {
    const int64_t looks = tick == 1 ? 3 : 1;
    const auto host = holoscan::radar::testing::to_host(frames[tick]);
    const auto mask = holoscan::radar::testing::to_host(masks[tick]);
    ASSERT_EQ(host.shape, (std::vector<int64_t>{1, kC, looks}));
    ASSERT_EQ(mask.shape, (std::vector<int64_t>{1, looks}));
    for (int64_t slot = 0; slot < looks; ++slot) {
      EXPECT_EQ(mask.template as<uint8_t>()[slot], 1);
      const int64_t d = tick == 1 ? 2 + slot : 3;
      for (int64_t c = 0; c < kC; ++c) {
        const auto v = cell_value(c, d, 5);
        EXPECT_EQ(host.template as<std::complex<T>>()[c * looks + slot],
                  (std::complex<T>{static_cast<T>(v.real()), static_cast<T>(v.imag())}));
      }
    }
  }
}

}  // namespace
