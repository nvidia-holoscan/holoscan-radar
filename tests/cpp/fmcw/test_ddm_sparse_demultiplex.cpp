// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cuda_runtime_api.h>

#include <cmath>
#include <complex>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/graph.hpp>

#include "fmcw/ddm_sparse_demultiplex_op/ddm_sparse_demultiplex_op.hpp"
#include "fmcw/ddm_sparse_demultiplex_op/ddm_sparse_demultiplex_stage.hpp"
#include "fmcw/virtual_array_assembly_op/virtual_array_assembly_stage.hpp"
#include "radar_common/dsp/mimo_coherent_compensation_stage.hpp"
#include "radar_common/math/constants.hpp"
#include "radar_common/tensor/dlpack_buffer.hpp"
#include "radar_common/tensor/tensor_utils.hpp"
#include "radar_schemas/array_geometry_bridge.hpp"
#include "utils/sdk_harness.hpp"

namespace {
namespace hr = holoscan::radar;

template <typename T>
std::shared_ptr<holoscan::Tensor> device(
    const std::vector<T>& host, const std::vector<int64_t>& shape,
    DLDataType dtype) {
  auto tensor = hr::make_device_tensor(shape, dtype);
  if (!host.empty()) {
    auto pointer = tensor->mutable_data();
    EXPECT_TRUE(pointer.has_value());
    EXPECT_EQ(cudaMemcpy(*pointer, host.data(), host.size() * sizeof(T),
                         cudaMemcpyHostToDevice),
              cudaSuccess);
  }
  return tensor;
}

template <typename T>
std::vector<T> host(const holoscan::Tensor& tensor, size_t count) {
  std::vector<T> result(count);
  EXPECT_EQ(cudaMemcpy(result.data(), tensor.data(), count * sizeof(T),
                       cudaMemcpyDeviceToHost),
            cudaSuccess);
  return result;
}

hr::DDMCodeDescriptor phase_code() {
  hr::DDMCodeDescriptor code;
  code.num_tx = 2;
  constexpr int64_t kD = 8;
  const std::complex<double> initial[2] = {{0.0, 1.0}, {-1.0, 0.0}};
  for (int64_t tx = 0; tx < 2; ++tx) {
    for (int64_t chirp = 0; chirp < kD; ++chirp) {
      const double phase =
          hr::kTwoPi * static_cast<double>(tx * chirp) / 2.0;
      code.phase_codes.push_back(
          initial[tx] *
          std::complex<double>(std::cos(phase), std::sin(phase)));
    }
  }
  return code;
}

hr::DDMSparseDemultiplexParams sparse_params() {
  return hr::DDMSparseDemultiplexParams{
      .ddm_code = phase_code(),
      .tx_positions = {0.0, 0.0, 0.0, 0.002, 0.0, 0.0},
      .rx_positions = {0.0, 0.0, 0.0},
      .overlap_policy = hr::OverlapPolicy::kKeepAll,
      .wavelength = 0.004,
      .snapshot_window = {0, 0},
  };
}

hr::MaxTensorExtents sparse_extents() {
  hr::MaxTensorExtents extents;
  extents.max_tx = 2;
  extents.max_rx = 1;
  extents.max_doppler_bins = 8;
  extents.max_range_bins = 1;
  extents.max_detections = 2;
  extents.max_items = 2;
  extents.max_virtual_channels = 2;
  extents.max_snapshots = 3;
  return extents;
}

struct IndependentFoldFixture {
  std::shared_ptr<holoscan::Tensor> rd_maps;
  std::shared_ptr<holoscan::Tensor> detection_idx;
  std::shared_ptr<holoscan::Tensor> fold_idx;
  std::vector<std::complex<double>> expected_snapshot;
};

IndependentFoldFixture independent_fold_fixture() {
  constexpr int64_t kD = 8;
  constexpr int64_t kFoldedCenter = 2;
  constexpr double kTargetCyclesPerChirp = 0.5;
  const auto code = phase_code();
  const std::complex<double> amplitudes[2] = {
      {1.25, -0.5}, {-0.75, 0.25}};
  std::vector<std::complex<double>> slow_time(kD);
  for (int64_t chirp = 0; chirp < kD; ++chirp) {
    const double target_phase =
        hr::kTwoPi * kTargetCyclesPerChirp * static_cast<double>(chirp);
    const std::complex<double> target_rotation{
        std::cos(target_phase), std::sin(target_phase)};
    for (int64_t tx = 0; tx < 2; ++tx) {
      slow_time[static_cast<size_t>(chirp)] +=
          amplitudes[tx] *
          code.phase_codes[static_cast<size_t>(tx * kD + chirp)] *
          target_rotation;
    }
  }

  // Independent forward model: form the unnormalized DFT and fftshift it.
  // No production demux table or band-permutation helper is used.
  std::vector<std::complex<double>> rd(static_cast<size_t>(kD));
  for (int64_t fft_bin = 0; fft_bin < kD; ++fft_bin) {
    std::complex<double> sum{};
    for (int64_t chirp = 0; chirp < kD; ++chirp) {
      const double phase =
          -hr::kTwoPi * static_cast<double>(fft_bin * chirp) /
          static_cast<double>(kD);
      sum += slow_time[static_cast<size_t>(chirp)] *
             std::complex<double>{std::cos(phase), std::sin(phase)};
    }
    const int64_t center_zero = (fft_bin - kD / 2 + kD) % kD;
    rd[static_cast<size_t>(center_zero)] = sum;
  }

  std::vector<std::complex<double>> expected;
  for (const auto& amplitude : amplitudes) {
    expected.push_back(static_cast<double>(kD) * amplitude);
  }
  return {
      device(rd, {1, kD, 1}, hr::complex_dtype<double>()),
      device(std::vector<int64_t>{kFoldedCenter, 0}, {1, 2},
             DLDataType{kDLInt, 64, 1}),
      device(std::vector<int64_t>{1}, {1}, DLDataType{kDLInt, 64, 1}),
      std::move(expected),
  };
}

template <typename Source, typename Op, typename Sink>
void wire_sparse_inputs(holoscan::Graph& graph, const Source& source, Op& op,
                        const Sink& sink) {
  const holoscan::ConnectionOptions queue{.queue_depth = 4U};
  graph.add_flow(source->out0, op->rd_maps, queue);
  graph.add_flow(source->out1, op->detection_idx, queue);
  graph.add_flow(op->snapshot, sink->in, queue);
  const auto geometry_sink =
      graph.op<hr::testing::WireCaptureOp<hr::WireArrayGeometry>>(
          "geometry_sink");
  graph.add_flow(op->array_geometry, geometry_sink->in, queue);
}

template <typename Source, typename Op, typename Sink>
void wire_sparse_fold_inputs(holoscan::Graph& graph, const Source& source,
                             Op& op, const Sink& sink) {
  wire_sparse_inputs(graph, source, op, sink);
  const holoscan::ConnectionOptions queue{.queue_depth = 4U};
  graph.add_flow(source->out2, op->fold_idx, queue);
}

std::vector<hr::ArrayGeometry> take_geometries() {
  using Bank = hr::testing::WireCaptureBank<hr::WireArrayGeometry>;
  std::vector<hr::WireArrayGeometry> captured;
  {
    const std::lock_guard lock(Bank::mutex);
    captured = std::move(Bank::payloads);
  }
  std::vector<hr::ArrayGeometry> result;
  result.reserve(captured.size());
  for (const auto& wire : captured) {
    auto geometry = hr::from_wire(wire);
    if (!geometry) {
      throw std::runtime_error("failed to decode emitted array geometry");
    }
    result.push_back(std::move(*geometry));
  }
  return result;
}

template <typename T>
std::vector<std::complex<T>> run_stage(const int64_t* fold_ptr,
                                       int64_t injected_fold) {
  constexpr int64_t kTx = 2;
  constexpr int64_t kRx = 2;
  constexpr int64_t kD = 8;
  constexpr int64_t kR = 2;
  constexpr int64_t kW = 4;
  const auto code = phase_code();
  const auto bands = hr::detail::ddm_exact_band_indices(code, kD);
  const auto table = hr::detail::build_ddm_demux_table(
      hr::detail::ddm_resolve_offsets(code), kTx, kD, kW);
  const auto weights_d = hr::detail::ddm_initial_phase_weights(code);
  std::vector<std::complex<T>> weights;
  std::vector<std::complex<T>> input(
      static_cast<size_t>(kRx * kD * kR));
  std::vector<int64_t> band_inverse(2 * kTx, -1);
  for (int64_t tx = 0; tx < kTx; ++tx) {
    weights.emplace_back(static_cast<T>(weights_d[tx].real()),
                         static_cast<T>(weights_d[tx].imag()));
    band_inverse[tx] = bands[tx];
    band_inverse[kTx + bands[tx]] = tx;
  }
  constexpr int64_t kFoldedD = 2;
  for (int64_t tx = 0; tx < kTx; ++tx) {
    const int64_t source_band = (bands[tx] + injected_fold) % kTx;
    const int64_t source_tx = band_inverse[kTx + source_band];
    const int64_t source_d = table[source_tx * kW + kFoldedD];
    const std::complex<T> initial(
        static_cast<T>(std::conj(weights_d[tx]).real()),
        static_cast<T>(std::conj(weights_d[tx]).imag()));
    for (int64_t rx = 0; rx < kRx; ++rx) {
      input[(rx * kD + source_d) * kR] =
          std::complex<T>(static_cast<T>(10 * tx + rx + 1), 0) * initial;
    }
  }
  const std::vector<int64_t> detection{kFoldedD, 0};
  auto virtual_plan = hr::detail::build_virtual_plan(
      {0, 0, 0, 10, 0, 0}, {0, 0, 0, 1, 0, 0}, kTx, kRx,
      hr::OverlapPolicy::kKeepAll);
  auto d_input = device(input, {kRx, kD, kR}, hr::complex_dtype<T>());
  auto d_detection = device(
      detection, {1, 2}, DLDataType{kDLInt, 64, 1});
  auto d_table = device(
      table, {kTx, kW}, DLDataType{kDLInt, 64, 1});
  auto d_bands = device(
      band_inverse, {2 * kTx}, DLDataType{kDLInt, 64, 1});
  auto d_weights = device(weights, {kTx}, hr::complex_dtype<T>());
  auto d_src = device(
      virtual_plan.src_channel, {virtual_plan.num_virtual},
      DLDataType{kDLInt, 64, 1});
  auto d_offsets = device(
      virtual_plan.member_offsets, {virtual_plan.num_virtual + 1},
      DLDataType{kDLInt, 64, 1});
  auto d_members = device(
      virtual_plan.member_indices,
      {static_cast<int64_t>(virtual_plan.member_indices.size())},
      DLDataType{kDLInt, 64, 1});
  auto d_output = hr::make_device_tensor(
      {1, virtual_plan.num_virtual, 1}, hr::complex_dtype<T>());
  auto d_valid =
      hr::make_device_tensor({1, 1}, DLDataType{kDLUInt, 8, 1});
  auto d_error =
      hr::make_device_tensor({1}, DLDataType{kDLInt, 32, 1});
  EXPECT_EQ(cudaMemset(const_cast<void*>(d_error->data()), 0, sizeof(int)),
            cudaSuccess);
  hr::detail::DdmRdMapsLayout layout{
      .num_rx = kRx, .num_doppler = kD, .num_range = kR};
  hr::detail::run_ddm_sparse_demultiplex_stage<T>(
      d_input->data(), static_cast<const int64_t*>(d_detection->data()),
      fold_ptr, 1, layout, kW, kTx, 1, 0, 0,
      hr::SnapshotEdgeMode::kWrap, hr::SnapshotEdgeMode::kShift, virtual_plan,
      static_cast<const int64_t*>(d_table->data()),
      static_cast<const int64_t*>(d_bands->data()), d_weights->data(),
      static_cast<const int64_t*>(d_src->data()),
      static_cast<const int64_t*>(d_offsets->data()),
      static_cast<const int64_t*>(d_members->data()),
      const_cast<void*>(d_output->data()),
      static_cast<uint8_t*>(const_cast<void*>(d_valid->data())),
      static_cast<int*>(const_cast<void*>(d_error->data())), nullptr);
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  EXPECT_EQ(host<int>(*d_error, 1)[0], 0);
  return host<std::complex<T>>(
      *d_output, static_cast<size_t>(virtual_plan.num_virtual));
}

TEST(DDMSparseDemultiplexStage, DefaultZeroLeavesNonzeroFoldMismatch) {
  const auto output = run_stage<float>(nullptr, 1);
  EXPECT_NEAR(output[0].imag(), 11.0F, 1e-5F);
  EXPECT_NEAR(output[2].imag(), -1.0F, 1e-5F);
  EXPECT_NE(output[0], std::complex<float>(1.0F, 0.0F));
  EXPECT_NE(output[2], std::complex<float>(11.0F, 0.0F));
}

TEST(DDMSparseDemultiplexStage, SuppliedFoldRecoversPhysicalTxAndPhase) {
  const std::vector<int64_t> fold{1};
  auto d_fold = device(fold, {1}, DLDataType{kDLInt, 64, 1});
  const auto output = run_stage<double>(
      static_cast<const int64_t*>(d_fold->data()), 1);
  EXPECT_NEAR(output[0].real(), 1.0, 1e-12);
  EXPECT_NEAR(output[1].real(), 2.0, 1e-12);
  EXPECT_NEAR(output[2].real(), 11.0, 1e-12);
  EXPECT_NEAR(output[3].real(), 12.0, 1e-12);
  for (const auto& value : output) {
    EXPECT_NEAR(value.imag(), 0.0, 1e-12);
  }
}

TEST(DDMSparseDemultiplexMath, GeometryAndInitialPhaseAreCoherent) {
  const auto weights = hr::detail::ddm_initial_phase_weights(phase_code());
  EXPECT_NEAR(weights[0].imag(), -1.0, 1e-12);
  EXPECT_NEAR(weights[1].real(), -1.0, 1e-12);
  const auto plan = hr::detail::build_virtual_plan(
      {0, 0, 0, 2, 0, 0}, {0, 0, 0, 1, 0, 0}, 2, 2,
      hr::OverlapPolicy::kAverage);
  const auto geometry = hr::detail::make_array_geometry(plan, 0.004);
  EXPECT_EQ(geometry.num_elements(), plan.num_virtual);
  EXPECT_DOUBLE_EQ(geometry.wavelength, 0.004);
}

TEST(DDMSparseDemultiplexOp, UnwiredFoldDefaultsToZeroAndMaskMayBeUnwired) {
  auto fixture = independent_fold_fixture();
  using Bank = hr::testing::WireCaptureBank<hr::WireArrayGeometry>;
  Bank::reset();
  const auto result =
      hr::testing::run_dual_in_single_out_operator_graph_result<
          hr::DDMSparseDemultiplexOp<double>>(
          {fixture.rd_maps}, {fixture.detection_idx}, sparse_params(),
          "snapshot",
          [](holoscan::Graph& graph, const auto& source, auto& op,
             const auto& sink) {
            wire_sparse_inputs(graph, source, op, sink);
          },
          sparse_extents(), {"is_valid"});
  ASSERT_TRUE(result.plan_ok) << result.plan_json;
  ASSERT_TRUE(result.diagnostics.empty());
  ASSERT_EQ(result.frames.size(), 1U);
  ASSERT_EQ(result.frames.front().shape,
            (std::vector<int64_t>{1, 2, 1}));
  const auto* actual = result.frames.front().as<std::complex<double>>();
  EXPECT_NE(actual[0], fixture.expected_snapshot[0]);
  EXPECT_NE(actual[1], fixture.expected_snapshot[1]);
  const auto geometries = take_geometries();
  ASSERT_EQ(geometries.size(), 1U);
  EXPECT_DOUBLE_EQ(geometries[0].wavelength, 0.004);
  EXPECT_EQ(geometries[0].element_positions,
            sparse_params().tx_positions);
}

TEST(DDMSparseDemultiplexOp, ConnectedNonzeroFoldRecoversPhysicalChannels) {
  auto fixture = independent_fold_fixture();
  using Bank = hr::testing::WireCaptureBank<hr::WireArrayGeometry>;
  Bank::reset();
  const auto result =
      hr::testing::run_triple_in_single_out_operator_graph_result<
          hr::DDMSparseDemultiplexOp<double>, true>(
          {fixture.rd_maps}, {fixture.detection_idx}, {fixture.fold_idx},
          sparse_params(), "snapshot",
          [](holoscan::Graph& graph, const auto& source, auto& op,
             const auto& sink) {
            wire_sparse_fold_inputs(graph, source, op, sink);
          },
          sparse_extents(), {"is_valid"});
  ASSERT_TRUE(result.plan_ok) << result.plan_json;
  ASSERT_TRUE(result.diagnostics.empty());
  ASSERT_EQ(result.frames.size(), 1U);
  const auto* actual = result.frames.front().as<std::complex<double>>();
  for (size_t channel = 0; channel < fixture.expected_snapshot.size();
       ++channel) {
    EXPECT_NEAR(actual[channel].real(),
                fixture.expected_snapshot[channel].real(), 1e-10);
    EXPECT_NEAR(actual[channel].imag(),
                fixture.expected_snapshot[channel].imag(), 1e-10);
  }
}

TEST(DDMSparseDemultiplexOp, RejectsFoldRowCountMismatch) {
  auto fixture = independent_fold_fixture();
  auto wrong_rows = device(std::vector<int64_t>{0, 1}, {2},
                           DLDataType{kDLInt, 64, 1});
  const auto result =
      hr::testing::run_triple_in_single_out_operator_graph_result<
          hr::DDMSparseDemultiplexOp<double>, true>(
          {fixture.rd_maps}, {fixture.detection_idx}, {wrong_rows},
          sparse_params(), "snapshot",
          [](holoscan::Graph& graph, const auto& source, auto& op,
             const auto& sink) {
            wire_sparse_fold_inputs(graph, source, op, sink);
          },
          sparse_extents(), {"is_valid"});
  EXPECT_TRUE(result.plan_ok) << result.plan_json;
  EXPECT_TRUE(result.frames.empty());
  EXPECT_FALSE(result.diagnostics.empty());
}

TEST(DDMSparseDemultiplexOp, RejectsFoldOutsideTxRange) {
  for (const int64_t invalid_fold : {-1, 2}) {
    auto fixture = independent_fold_fixture();
    auto fold = device(std::vector<int64_t>{invalid_fold}, {1},
                       DLDataType{kDLInt, 64, 1});
    const auto result =
        hr::testing::run_triple_in_single_out_operator_graph_result<
            hr::DDMSparseDemultiplexOp<double>, true>(
            {fixture.rd_maps}, {fixture.detection_idx}, {fold},
            sparse_params(), "snapshot",
            [](holoscan::Graph& graph, const auto& source, auto& op,
               const auto& sink) {
              wire_sparse_fold_inputs(graph, source, op, sink);
            },
            sparse_extents(), {"is_valid"});
    EXPECT_TRUE(result.plan_ok) << result.plan_json;
    EXPECT_TRUE(result.frames.empty());
    EXPECT_FALSE(result.diagnostics.empty());
  }
}

TEST(DDMSparseDemultiplexOp, ZeroDetectionsEmitsExactEmptySnapshot) {
  auto fixture = independent_fold_fixture();
  auto empty = device(std::vector<int64_t>{}, {0, 2},
                      DLDataType{kDLInt, 64, 1});
  using Bank = hr::testing::WireCaptureBank<hr::WireArrayGeometry>;
  Bank::reset();
  const auto result =
      hr::testing::run_dual_in_single_out_operator_graph_result<
          hr::DDMSparseDemultiplexOp<double>>(
          {fixture.rd_maps}, {empty}, sparse_params(), "snapshot",
          [](holoscan::Graph& graph, const auto& source, auto& op,
             const auto& sink) {
            wire_sparse_inputs(graph, source, op, sink);
          },
          sparse_extents(), {"is_valid"});
  ASSERT_TRUE(result.plan_ok) << result.plan_json;
  ASSERT_TRUE(result.diagnostics.empty());
  ASSERT_EQ(result.frames.size(), 1U);
  EXPECT_EQ(result.frames.front().shape,
            (std::vector<int64_t>{0, 2, 1}));
  EXPECT_EQ(take_geometries().size(), 1U);
}

TEST(DDMSparseDemultiplexOp, LiveWavelengthAppliesOneTickThenReverts) {
  auto fixture = independent_fold_fixture();
  using Bank = hr::testing::WireCaptureBank<hr::WireArrayGeometry>;
  Bank::reset();
  const auto result =
      hr::testing::
          run_exact_override_dual_in_single_out_operator_graph_result<
              hr::DDMSparseDemultiplexOp<double>, double>(
              {fixture.rd_maps, fixture.rd_maps, fixture.rd_maps},
              {fixture.detection_idx, fixture.detection_idx,
               fixture.detection_idx},
              {std::nullopt, 0.008, std::nullopt}, sparse_params(),
              "snapshot",
              [](holoscan::Graph& graph, const auto& source, auto& op,
                 const auto& sink) {
                wire_sparse_inputs(graph, source, op, sink);
                const holoscan::ConnectionOptions queue{.queue_depth = 4U};
                graph.add_flow(source->override_out, op->wavelength_in, queue);
              },
              sparse_extents(), {"is_valid"});
  ASSERT_TRUE(result.plan_ok) << result.plan_json;
  ASSERT_TRUE(result.diagnostics.empty());
  ASSERT_EQ(result.frames.size(), 3U);
  const auto geometries = take_geometries();
  ASSERT_EQ(geometries.size(), 3U);
  EXPECT_DOUBLE_EQ(geometries[0].wavelength, 0.004);
  EXPECT_DOUBLE_EQ(geometries[1].wavelength, 0.008);
  EXPECT_DOUBLE_EQ(geometries[2].wavelength, 0.004);
}

}  // namespace
