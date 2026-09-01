// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// SnapshotGatherOp contract tests (HSDK 5 Graph harness): RD neighborhood
// gather → snapshot matrix, edge modes, batch selection, Params coverage,
// and error paths.

#include <gtest/gtest.h>

#include <complex>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/graph.hpp>

#include "angle_estimation/snapshot_gather_op/snapshot_gather_op.hpp"
#include "radar_common/descriptors/enums.hpp"
#include "radar_common/tensor/tensor_utils.hpp"
#include "utils/sdk_harness.hpp"
#include "utils/tensor_fixtures.hpp"

namespace {

using holoscan::radar::SnapshotEdgeMode;
using holoscan::radar::SnapshotGatherOp;
using holoscan::radar::SnapshotGatherParams;
using holoscan::radar::complex_dtype;
using holoscan::radar::testing::DualIOGraphRunResult;
using holoscan::radar::testing::HostTensor;
using holoscan::radar::testing::run_dual_io_operator_graph_result;
using holoscan::radar::testing::to_device_tensor;

constexpr int64_t kC = 3;
constexpr int64_t kD = 8;
constexpr int64_t kR = 16;

std::complex<double> cell_value(int64_t c, int64_t d, int64_t r) {
  return {static_cast<double>(c * 100 + d * 10 + r),
          static_cast<double>(c + d + r)};
}

template <typename T>
std::shared_ptr<holoscan::Tensor> make_rd_maps() {
  std::vector<std::complex<T>> host(static_cast<size_t>(kC * kD * kR));
  for (int64_t c = 0; c < kC; ++c) {
    for (int64_t d = 0; d < kD; ++d) {
      for (int64_t r = 0; r < kR; ++r) {
        const auto v = cell_value(c, d, r);
        host[static_cast<size_t>((c * kD + d) * kR + r)] =
            std::complex<T>(static_cast<T>(v.real()),
                            static_cast<T>(v.imag()));
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

template <typename Source, typename OpRef, typename Sink0, typename Sink1>
void wire_snapshot_gather(holoscan::Graph& graph, const Source& source,
                          OpRef& op, const Sink0& sink0, const Sink1& sink1) {
  const holoscan::ConnectionOptions q{.queue_depth = 4U};
  graph.add_flow(source->out0, op->rd_maps, q);
  graph.add_flow(source->out1, op->detection_idx, q);
  graph.add_flow(op->snapshot, sink0->in, q);
  graph.add_flow(op->is_valid, sink1->in, q);
}

template <typename T>
DualIOGraphRunResult run_gather_result(
    std::shared_ptr<holoscan::Tensor> rd_maps,
    std::shared_ptr<holoscan::Tensor> detection_idx,
    SnapshotGatherParams params = {}) {
  return run_dual_io_operator_graph_result<SnapshotGatherOp<T>>(
      {std::move(rd_maps)}, {std::move(detection_idx)}, std::move(params),
      "snapshot", "is_valid",
      [](holoscan::Graph& graph, const auto& source, auto& op,
         const auto& sink0, const auto& sink1) {
        wire_snapshot_gather(graph, source, op, sink0, sink1);
      });
}

template <typename T>
GatherOutputs run_gather(std::shared_ptr<holoscan::Tensor> rd_maps,
                         std::shared_ptr<holoscan::Tensor> detection_idx,
                         SnapshotGatherParams params = {}) {
  DualIOGraphRunResult result = run_gather_result<T>(
      std::move(rd_maps), std::move(detection_idx), std::move(params));
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
        "SnapshotGatherOp test: expected one output frame pair, got " +
        std::to_string(result.frames0.size()) + "/" +
        std::to_string(result.frames1.size()));
  }
  return {std::move(result.frames0.front()),
          std::move(result.frames1.front())};
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
      EXPECT_NEAR(got.real(), exp.real(), 1e-6)
          << "slot " << slot << " ch " << c;
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
  EXPECT_NEAR(snap[static_cast<size_t>(8)].real(),
              cell_value(0, 2, 2).real(), 1e-6);
}

TEST(SnapshotGatherOpTest, EmptyDetections) {
  std::vector<int64_t> idx;
  auto idx_t = to_device_tensor(idx, {0, 2}, DLDataType{kDLInt, 64, 1});
  const auto out = run_gather<double>(make_rd_maps(), idx_t);
  EXPECT_EQ(out.snapshot.shape[0], 0);
  EXPECT_EQ(out.is_valid.shape[0], 0);
}

TEST(SnapshotGatherOpTest, LiveSnapshotWindowAppliesForOneTick) {
  // [EA HSDK5 ONLY] See docs/planning/HSDK5_PORT.md §1 — Graph mid-stream
  // live override harness fails on EA device/partition binding
  // (RUNTIME_PORT_BINDING_UNAVAILABLE / RUNTIME_EXECUTION_LANE_PLAN_INVALID).
  // Params authoring covers snapshot_window; restore once §1 is closed.
  GTEST_SKIP() << "HSDK 5 EA: live snapshot_window override harness deferred "
                  "(HSDK5_PORT.md §1)";
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

TEST(SnapshotGatherOpTest, FloatAndDoubleAgree) {
  std::vector<int64_t> idx = {3, 5};
  auto idx_t = to_device_tensor(idx, {1, 2}, DLDataType{kDLInt, 64, 1});
  SnapshotGatherParams params{};
  params.snapshot_window = {1, 0};
  const auto out_f =
      run_gather<float>(make_rd_maps<float>(), idx_t, params);
  // Fresh idx tensor — previous run_gather consumed the shared_ptr.
  auto idx_t2 = to_device_tensor(idx, {1, 2}, DLDataType{kDLInt, 64, 1});
  const auto out_d =
      run_gather<double>(make_rd_maps<double>(), idx_t2, params);
  ASSERT_EQ(out_f.snapshot.shape, out_d.snapshot.shape);
  const auto* f_data = out_f.snapshot.as<std::complex<float>>();
  const auto* d_data = out_d.snapshot.as<std::complex<double>>();
  for (int64_t i = 0; i < out_d.snapshot.size(); ++i) {
    ASSERT_NEAR(std::abs(std::complex<double>(f_data[i]) - d_data[i]), 0.0,
                1e-5)
        << "index " << i;
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
  const auto out =
      run_gather<double>(make_batched_rd_maps<double>(kB), idx_t);

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
  const DualIOGraphRunResult result =
      run_gather_result<double>(make_rd_maps<float>(), idx_t);
  EXPECT_TRUE(result.plan_ok);
  EXPECT_TRUE(result.frames0.empty());
  EXPECT_FALSE(result.diagnostics.empty());
}

TEST(SnapshotGatherOpTest, RejectsNarrowDetectionIdx) {
  std::vector<int64_t> idx = {3};
  auto idx_t = to_device_tensor(idx, {1, 1}, DLDataType{kDLInt, 64, 1});
  const DualIOGraphRunResult result =
      run_gather_result<double>(make_rd_maps(), idx_t);
  EXPECT_TRUE(result.plan_ok);
  EXPECT_TRUE(result.frames0.empty());
  EXPECT_FALSE(result.diagnostics.empty());
}

TEST(SnapshotGatherOpTest, RejectsNegativeSnapshotWindow) {
  std::vector<int64_t> idx = {3, 5};
  auto idx_t = to_device_tensor(idx, {1, 2}, DLDataType{kDLInt, 64, 1});
  SnapshotGatherParams params{};
  params.snapshot_window = {-1, 0};
  const DualIOGraphRunResult result =
      run_gather_result<double>(make_rd_maps(), idx_t, params);
  EXPECT_TRUE(result.plan_ok);
  EXPECT_TRUE(result.frames0.empty());
  EXPECT_FALSE(result.diagnostics.empty());
}

TEST(SnapshotGatherOpTest, RejectsBatchCoordOutOfRange) {
  std::vector<int64_t> idx = {9, 3, 5};
  auto idx_t = to_device_tensor(idx, {1, 3}, DLDataType{kDLInt, 64, 1});
  const DualIOGraphRunResult result =
      run_gather_result<double>(make_batched_rd_maps<double>(2), idx_t);
  EXPECT_TRUE(result.plan_ok);
  EXPECT_TRUE(result.frames0.empty());
  EXPECT_FALSE(result.diagnostics.empty());
}

}  // namespace
