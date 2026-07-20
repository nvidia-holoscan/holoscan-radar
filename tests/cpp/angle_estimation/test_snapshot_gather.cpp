// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// SnapshotGatherOp contract tests (RD neighborhood gather → snapshot matrix).

#include <gtest/gtest.h>

#include <complex>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include <holoscan/holoscan.hpp>
#include <holoscan/test/test_harness.hpp>

#include "angle_estimation/snapshot_gather_op/snapshot_gather_op.hpp"
#include "radar_common/descriptors/enums.hpp"
#include "radar_common/tensor/tensor_utils.hpp"
#include "utils/sdk_harness.hpp"
#include "utils/tensor_fixtures.hpp"

namespace {
using holoscan::radar::SnapshotEdgeMode;
using holoscan::radar::SnapshotGatherOp;
using holoscan::radar::complex_dtype;
using holoscan::radar::testing::HostTensor;
using holoscan::radar::testing::single_map;
using holoscan::radar::testing::TensorMapCapture;
using holoscan::radar::testing::to_device_tensor;
using holoscan::radar::testing::to_host;
using holoscan::test::create_operator_test;

constexpr int64_t kC = 3;
constexpr int64_t kD = 8;
constexpr int64_t kR = 16;

std::complex<double> cell_value(int64_t c, int64_t d, int64_t r) {
  return {static_cast<double>(c * 100 + d * 10 + r),
          static_cast<double>(c + d + r)};
}

template <typename T>
std::shared_ptr<holoscan::Tensor> make_rd_maps();

std::shared_ptr<holoscan::Tensor> make_rd_maps() {
  return make_rd_maps<double>();
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

template <typename T>
std::shared_ptr<holoscan::Tensor> make_batched_rd_maps(int64_t batch) {
  std::vector<std::complex<T>> host(
      static_cast<size_t>(batch * kC * kD * kR));
  for (int64_t b = 0; b < batch; ++b) {
    for (int64_t c = 0; c < kC; ++c) {
      for (int64_t d = 0; d < kD; ++d) {
        for (int64_t r = 0; r < kR; ++r) {
          const size_t off = static_cast<size_t>(
              ((b * kC + c) * kD + d) * kR + r);
          const auto v = cell_value(c, d, r);
          // Offset batch slice so batch selection is observable.
          host[off] = std::complex<T>(
              static_cast<T>(v.real() + static_cast<double>(b) * 1000.0),
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

template <typename T, typename... ExtraArgs>
GatherOutputs run_gather(
    std::shared_ptr<holoscan::Tensor> rd_maps,
    std::shared_ptr<holoscan::Tensor> detection_idx,
    ExtraArgs&&... extra_args) {
  auto harness = create_operator_test<SnapshotGatherOp<T>>(
      std::forward<ExtraArgs>(extra_args)...);
  harness->template add_input_port<holoscan::TensorMap>(
      "rd_maps", {single_map(std::move(rd_maps), "rd_maps")});
  harness->template add_input_port<holoscan::TensorMap>(
      "detection_idx",
      {single_map(std::move(detection_idx), "detection_idx")});

  TensorMapCapture snapshot_cap;
  TensorMapCapture valid_cap;
  harness->template add_output_port<holoscan::TensorMap>(
      "snapshot", snapshot_cap.validators());
  harness->template add_output_port<holoscan::TensorMap>(
      "is_valid", valid_cap.validators());
  harness->run_test();

  if (snapshot_cap.frames.size() != 1u || valid_cap.frames.size() != 1u) {
    throw std::runtime_error(
        "SnapshotGatherOp test: expected one output frame");
  }
  return {snapshot_cap.frames.front(), valid_cap.frames.front()};
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
  const auto out = run_gather<double>(
      make_rd_maps(), idx_t,
      holoscan::Arg("snapshot_window") = std::vector<int64_t>{1, 0});

  ASSERT_EQ(out.snapshot.shape, (std::vector<int64_t>{1, kC, 3}));
  const auto* snap = out.snapshot.as<std::complex<double>>();
  for (int64_t slot = 0; slot < 3; ++slot) {
    const int64_t d = 2 + slot;  // centers 3 with half=1 under wrap
    for (int64_t c = 0; c < kC; ++c) {
      const auto& got = snap[static_cast<size_t>(slot * kC + c)];
      const auto exp = cell_value(c, d, 5);
      EXPECT_NEAR(got.real(), exp.real(), 1e-6)
          << "slot " << slot << " ch " << c;
      EXPECT_NEAR(got.imag(), exp.imag(), 1e-6);
    }
  }
}

TEST(SnapshotGatherOpTest, ClampMarksInvalidSlots) {
  std::vector<int64_t> idx = {0, 0};
  auto idx_t = to_device_tensor(idx, {1, 2}, DLDataType{kDLInt, 64, 1});
  const auto out = run_gather<double>(
      make_rd_maps(), idx_t,
      holoscan::Arg("snapshot_window") = std::vector<int64_t>{1, 1},
      holoscan::Arg("doppler_edge") = SnapshotEdgeMode::kClamp,
      holoscan::Arg("range_edge") = SnapshotEdgeMode::kClamp);

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
  const auto out = run_gather<double>(
      make_rd_maps(), idx_t,
      holoscan::Arg("snapshot_window") = std::vector<int64_t>{1, 1},
      holoscan::Arg("doppler_edge") = SnapshotEdgeMode::kShift,
      holoscan::Arg("range_edge") = SnapshotEdgeMode::kShift);

  ASSERT_EQ(out.snapshot.shape, (std::vector<int64_t>{1, kC, 9}));
  const auto* valid = out.is_valid.as<uint8_t>();
  for (int64_t i = 0; i < 9; ++i) {
    EXPECT_EQ(valid[static_cast<size_t>(i)], 1);
  }
  const auto* snap = out.snapshot.as<std::complex<double>>();
  EXPECT_NEAR(snap[0].real(), cell_value(0, 0, 0).real(), 1e-6);
  EXPECT_NEAR(snap[static_cast<size_t>(8 * kC)].real(),
              cell_value(0, 2, 2).real(), 1e-6);
}

TEST(SnapshotGatherOpTest, EmptyDetections) {
  std::vector<int64_t> idx;
  auto idx_t = to_device_tensor(idx, {0, 2}, DLDataType{kDLInt, 64, 1});
  const auto out = run_gather<double>(make_rd_maps(), idx_t);
  EXPECT_EQ(out.snapshot.shape[0], 0);
  EXPECT_EQ(out.is_valid.shape[0], 0);
}

class SnapshotWindowOverrideApp : public holoscan::Application {
 public:
  std::vector<holoscan::TensorMap> rd_frames;
  std::vector<holoscan::TensorMap> idx_frames;
  std::vector<std::vector<int64_t>> window_overrides;
  TensorMapCapture* snapshot = nullptr;
  TensorMapCapture* is_valid = nullptr;

  void compose() override {
    using holoscan::Arg;
    auto op = make_operator<SnapshotGatherOp<double>>(
        "gather", Arg("snapshot_window") = std::vector<int64_t>{0, 0});
    holoscan::radar::testing::wire_data_source(this, op, "rd_maps", rd_frames);
    holoscan::radar::testing::wire_data_source(this, op, "detection_idx",
                                               idx_frames);
    holoscan::radar::testing::wire_leading_override(
        this, op, "snapshot_window", window_overrides);
    holoscan::radar::testing::wire_capture_sink(this, op, "snapshot", snapshot);
    holoscan::radar::testing::wire_capture_sink(this, op, "is_valid", is_valid);
  }
};

TEST(SnapshotGatherOpTest, LiveSnapshotWindowStickyOverride) {
  auto rd = make_rd_maps();
  std::vector<int64_t> idx = {3, 5};
  auto idx_t = to_device_tensor(idx, {1, 2}, DLDataType{kDLInt, 64, 1});

  SnapshotWindowOverrideApp app;
  app.rd_frames = {holoscan::radar::testing::single_map(rd, "rd_maps"),
                   holoscan::radar::testing::single_map(rd, "rd_maps")};
  app.idx_frames = {
      holoscan::radar::testing::single_map(idx_t, "detection_idx"),
      holoscan::radar::testing::single_map(idx_t, "detection_idx")};
  app.window_overrides = {{0, 0}, {1, 0}};

  TensorMapCapture snap_cap;
  TensorMapCapture valid_cap;
  app.snapshot = &snap_cap;
  app.is_valid = &valid_cap;
  app.run();

  ASSERT_EQ(snap_cap.frames.size(), 2u);
  EXPECT_EQ(snap_cap.frames[0].shape[2], 1);
  EXPECT_EQ(snap_cap.frames[1].shape[2], 3);
}

TEST(SnapshotGatherOpTest, FloatInstantiationSmoke) {
  std::vector<int64_t> idx = {3, 5};
  auto idx_t = to_device_tensor(idx, {1, 2}, DLDataType{kDLInt, 64, 1});
  const auto out = run_gather<float>(make_rd_maps<float>(), idx_t);
  ASSERT_EQ(out.snapshot.shape, (std::vector<int64_t>{1, kC, 1}));
  const auto* snap = out.snapshot.as<std::complex<float>>();
  EXPECT_NEAR(snap[0].real(), static_cast<float>(cell_value(0, 3, 5).real()),
              1e-5f);
}

TEST(SnapshotGatherOpTest, DopplerWrapAtBoundary) {
  // Center at doppler bin 0; half=1 under wrap yields bins {7, 0, 1}.
  std::vector<int64_t> idx = {0, 5};
  auto idx_t = to_device_tensor(idx, {1, 2}, DLDataType{kDLInt, 64, 1});
  const auto out = run_gather<double>(
      make_rd_maps(), idx_t,
      holoscan::Arg("snapshot_window") = std::vector<int64_t>{1, 0},
      holoscan::Arg("doppler_edge") = SnapshotEdgeMode::kWrap);

  ASSERT_EQ(out.snapshot.shape, (std::vector<int64_t>{1, kC, 3}));
  const auto* snap = out.snapshot.as<std::complex<double>>();
  const int64_t d_bins[] = {7, 0, 1};
  for (int slot = 0; slot < 3; ++slot) {
    for (int64_t c = 0; c < kC; ++c) {
      const auto& got = snap[static_cast<size_t>(slot * kC + c)];
      const auto exp = cell_value(c, d_bins[slot], 5);
      EXPECT_NEAR(got.real(), exp.real(), 1e-6)
          << "slot " << slot << " ch " << c;
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
    EXPECT_NEAR(snap[static_cast<size_t>(kC + c)].real(),
                cell_value(c, 3, 5).real() + 1000.0, 1e-6);
  }
}

TEST(SnapshotGatherOpTest, RejectsRdMapsDtypeMismatch) {
  std::vector<int64_t> idx = {3, 5};
  auto idx_t = to_device_tensor(idx, {1, 2}, DLDataType{kDLInt, 64, 1});
  EXPECT_THROW(run_gather<double>(make_rd_maps<float>(), idx_t),
               std::invalid_argument);
}

TEST(SnapshotGatherOpTest, RejectsNarrowDetectionIdx) {
  std::vector<int64_t> idx = {3};
  auto idx_t = to_device_tensor(idx, {1, 1}, DLDataType{kDLInt, 64, 1});
  EXPECT_THROW(run_gather<double>(make_rd_maps(), idx_t),
               std::invalid_argument);
}

TEST(SnapshotGatherOpTest, RejectsNegativeSnapshotWindow) {
  std::vector<int64_t> idx = {3, 5};
  auto idx_t = to_device_tensor(idx, {1, 2}, DLDataType{kDLInt, 64, 1});
  EXPECT_THROW(
      run_gather<double>(
          make_rd_maps(), idx_t,
          holoscan::Arg("snapshot_window") = std::vector<int64_t>{-1, 0}),
      std::invalid_argument);
}

TEST(SnapshotGatherOpTest, RejectsBatchCoordOutOfRange) {
  std::vector<int64_t> idx = {9, 3, 5};
  auto idx_t = to_device_tensor(idx, {1, 3}, DLDataType{kDLInt, 64, 1});
  EXPECT_THROW(run_gather<double>(make_batched_rd_maps<double>(2), idx_t),
               std::invalid_argument);
}

}  // namespace
