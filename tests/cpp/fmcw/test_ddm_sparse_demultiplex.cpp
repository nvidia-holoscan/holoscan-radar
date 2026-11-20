// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cuda_runtime_api.h>

#include <cmath>
#include <chrono>
#include <complex>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/connection_options.hpp>
#include <holoscan/testing/operator_test_harness.hpp>

#include "utils/fixture_completion.hpp"
#include "fmcw/ddm_sparse_demultiplex_op/ddm_sparse_demultiplex_op.hpp"
#include "fmcw/ddm_sparse_demultiplex_op/ddm_sparse_demultiplex_stage.hpp"
#include "radar_common/dsp/mimo_coherent_compensation_stage.hpp"
#include "radar_common/dsp/virtual_array.hpp"
#include "radar_common/math/constants.hpp"
#include "radar_common/tensor/dlpack_buffer.hpp"
#include "radar_common/tensor/tensor_utils.hpp"
#include "radar_schemas/array_geometry_bridge.hpp"
#include "utils/tensor_fixtures.hpp"

namespace {
namespace hr = holoscan::radar;

template <typename T>
std::shared_ptr<holoscan::Tensor> device(const std::vector<T>& host,
                                         const std::vector<int64_t>& shape, DLDataType dtype) {
  return hr::testing::to_device_tensor(host, shape, dtype);
}

template <typename T>
std::shared_ptr<holoscan::Tensor> raw_device(const std::vector<T>& host,
                                             const std::vector<int64_t>& shape, DLDataType dtype) {
  auto tensor = hr::make_device_tensor(shape, dtype);
  if (!host.empty()) {
    auto pointer = tensor->mutable_data();
    EXPECT_TRUE(pointer.has_value());
    EXPECT_EQ(cudaMemcpy(*pointer, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice),
              cudaSuccess);
  }
  return tensor;
}

template <typename T>
std::vector<T> host(const holoscan::Tensor& tensor, size_t count) {
  std::vector<T> result(count);
  EXPECT_EQ(cudaMemcpy(result.data(), tensor.data(), count * sizeof(T), cudaMemcpyDeviceToHost),
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
      const double phase = hr::kTwoPi * static_cast<double>(tx * chirp) / 2.0;
      code.phase_codes.push_back(initial[tx] *
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
  const std::complex<double> amplitudes[2] = {{1.25, -0.5}, {-0.75, 0.25}};
  std::vector<std::complex<double>> slow_time(kD);
  for (int64_t chirp = 0; chirp < kD; ++chirp) {
    const double target_phase = hr::kTwoPi * kTargetCyclesPerChirp * static_cast<double>(chirp);
    const std::complex<double> target_rotation{std::cos(target_phase), std::sin(target_phase)};
    for (int64_t tx = 0; tx < 2; ++tx) {
      slow_time[static_cast<size_t>(chirp)] +=
          amplitudes[tx] * code.phase_codes[static_cast<size_t>(tx * kD + chirp)] * target_rotation;
    }
  }

  // Independent forward model: form the unnormalized DFT and fftshift it.
  // No production demux table or band-permutation helper is used.
  std::vector<std::complex<double>> rd(static_cast<size_t>(kD));
  for (int64_t fft_bin = 0; fft_bin < kD; ++fft_bin) {
    std::complex<double> sum{};
    for (int64_t chirp = 0; chirp < kD; ++chirp) {
      const double phase =
          -hr::kTwoPi * static_cast<double>(fft_bin * chirp) / static_cast<double>(kD);
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
      device(std::vector<int64_t>{kFoldedCenter, 0}, {1, 2}, DLDataType{kDLInt, 64, 1}),
      device(std::vector<int64_t>{1}, {1}, DLDataType{kDLInt, 64, 1}),
      std::move(expected),
  };
}

holoscan::testing::TestResult run_sparse_operator(
    const std::vector<std::shared_ptr<holoscan::Tensor>>& rd_maps,
    const std::vector<std::shared_ptr<holoscan::Tensor>>& detections,
    const std::vector<std::shared_ptr<holoscan::Tensor>>& folds = {},
    const std::vector<std::optional<double>>& wavelengths = {}) {
  using Op = hr::DDMSparseDemultiplexOp<double>;
  auto fixture = holoscan::testing::create_operator_test<Op>(sparse_extents());
  fixture.set_params(sparse_params());
  for (std::size_t index = 0; index < rd_maps.size(); ++index) {
    const auto capture_time = std::chrono::milliseconds{static_cast<int64_t>((index + 1U) * 100U)};
    const auto maps = holoscan::testing::input("rd_maps", *rd_maps[index]);
    const auto detection = holoscan::testing::input("detection_idx", *detections[index]);
    const bool has_fold = index < folds.size() && folds[index];
    const bool has_wavelength = index < wavelengths.size() && wavelengths[index].has_value();
    if (has_fold && has_wavelength) {
      fixture.push_frame(holoscan::testing::frame_at(
          capture_time, holoscan::testing::input("wavelength", *wavelengths[index]),
          holoscan::testing::input("fold_idx", *folds[index]), maps, detection));
    } else if (has_fold) {
      fixture.push_frame(holoscan::testing::frame_at(
          capture_time, holoscan::testing::input("fold_idx", *folds[index]), maps, detection));
    } else if (has_wavelength) {
      fixture.push_frame(holoscan::testing::frame_at(
          capture_time, holoscan::testing::input("wavelength", *wavelengths[index]), maps,
          detection));
    } else {
      fixture.push_frame(holoscan::testing::frame_at(capture_time, maps, detection));
    }
  }
  constexpr std::size_t kMaximumBytes = 256U * 1024U * 1024U;
  fixture.capture_snapshot<holoscan::Tensor>("snapshot", rd_maps.size(), kMaximumBytes);
  fixture.capture_snapshot<hr::WireArrayGeometry>("array_geometry", rd_maps.size(), kMaximumBytes);
  for (const char* input : {"rd_maps", "detection_idx"}) {
    fixture.input_source_device(input, holoscan::DeviceId{0});
  }
  if (!folds.empty()) {
    fixture.input_source_device("fold_idx", holoscan::DeviceId{0});
  }
  fixture.output_device("snapshot", holoscan::DeviceId{0});
  fixture.output_device("is_valid", holoscan::DeviceId{0});
  holoscan::radar::testing::FixtureCompletion completion;
  auto running = fixture.run_async(completion.run_options());
  (void)completion.wait_for_output(running, "snapshot", rd_maps.size());
  (void)completion.wait_for_output(running, "array_geometry", rd_maps.size());
  running.request_stop();
  auto fixture_result = running.finish();
  completion.check(fixture_result);
  return fixture_result;
}

std::vector<hr::testing::HostTensor> sparse_snapshots(const holoscan::testing::TestResult& result) {
  std::vector<hr::testing::HostTensor> frames;
  for (const auto& snapshot : result.template snapshots<holoscan::Tensor>("snapshot")) {
    frames.push_back(hr::testing::to_host(snapshot));
  }
  return frames;
}

std::vector<hr::ArrayGeometry> take_geometries(const holoscan::testing::TestResult& result) {
  std::vector<hr::ArrayGeometry> geometries;
  for (const auto& snapshot : result.template snapshots<hr::WireArrayGeometry>("array_geometry")) {
    auto wire = snapshot.metadata();
    const auto& packed = snapshot.tensor(hr::wire::ArrayGeometryTensorFields::packed_geometry);
    std::vector<double> values(packed.bytes().size() / sizeof(double));
    std::memcpy(values.data(), packed.bytes().data(), packed.bytes().size());
    wire.packed_geometry =
        hr::testing::to_pageable_host_tensor(values, packed.shape(), packed.dtype());
    auto geometry = hr::from_wire(wire);
    if (!geometry) {
      throw std::runtime_error("failed to decode emitted array geometry");
    }
    geometries.push_back(std::move(*geometry));
  }
  return geometries;
}

template <typename T>
std::vector<std::complex<T>> run_stage(const int64_t* fold_ptr, int64_t injected_fold) {
  constexpr int64_t kTx = 2;
  constexpr int64_t kRx = 2;
  constexpr int64_t kD = 8;
  constexpr int64_t kR = 2;
  constexpr int64_t kW = 4;
  const auto code = phase_code();
  const auto bands = hr::detail::ddm_exact_band_indices(code, kD);
  const auto table =
      hr::detail::build_ddm_demux_table(hr::detail::ddm_resolve_offsets(code), kTx, kD, kW);
  const auto weights_d = hr::detail::ddm_initial_phase_weights(code);
  std::vector<std::complex<T>> weights;
  std::vector<std::complex<T>> input(static_cast<size_t>(kRx * kD * kR));
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
    const std::complex<T> initial(static_cast<T>(std::conj(weights_d[tx]).real()),
                                  static_cast<T>(std::conj(weights_d[tx]).imag()));
    for (int64_t rx = 0; rx < kRx; ++rx) {
      input[(rx * kD + source_d) * kR] =
          std::complex<T>(static_cast<T>(10 * tx + rx + 1), 0) * initial;
    }
  }
  const std::vector<int64_t> detection{kFoldedD, 0};
  auto virtual_plan =
      hr::detail::build_virtual_plan({0, 0, 0, 10, 0, 0}, {0, 0, 0, 1, 0, 0}, kTx, kRx,
                                     hr::OverlapPolicy::kKeepAll, "DDMSparseDemultiplexOp");
  auto d_input = raw_device(input, {kRx, kD, kR}, hr::complex_dtype<T>());
  auto d_detection = raw_device(detection, {1, 2}, DLDataType{kDLInt, 64, 1});
  auto d_table = raw_device(table, {kTx, kW}, DLDataType{kDLInt, 64, 1});
  auto d_bands = raw_device(band_inverse, {2 * kTx}, DLDataType{kDLInt, 64, 1});
  auto d_weights = raw_device(weights, {kTx}, hr::complex_dtype<T>());
  auto d_src =
      raw_device(virtual_plan.src_channel, {virtual_plan.num_virtual}, DLDataType{kDLInt, 64, 1});
  auto d_offsets = raw_device(virtual_plan.member_offsets, {virtual_plan.num_virtual + 1},
                              DLDataType{kDLInt, 64, 1});
  auto d_members = raw_device(virtual_plan.member_indices,
                              {static_cast<int64_t>(virtual_plan.member_indices.size())},
                              DLDataType{kDLInt, 64, 1});
  auto d_output = hr::make_device_tensor({1, virtual_plan.num_virtual, 1}, hr::complex_dtype<T>());
  auto d_valid = hr::make_device_tensor({1, 1}, DLDataType{kDLUInt, 8, 1});
  auto d_error = hr::make_device_tensor({1}, DLDataType{kDLInt, 32, 1});
  EXPECT_EQ(cudaMemset(const_cast<void*>(d_error->data()), 0, sizeof(int)), cudaSuccess);
  hr::detail::DdmRdMapsLayout layout{.num_rx = kRx, .num_doppler = kD, .num_range = kR};
  hr::detail::run_ddm_sparse_demultiplex_stage<T>(
      d_input->data(), static_cast<const int64_t*>(d_detection->data()), fold_ptr, 1, layout, kW,
      kTx, 1, 0, 0, hr::SnapshotEdgeMode::kWrap, hr::SnapshotEdgeMode::kShift, virtual_plan,
      static_cast<const int64_t*>(d_table->data()), static_cast<const int64_t*>(d_bands->data()),
      d_weights->data(), static_cast<const int64_t*>(d_src->data()),
      static_cast<const int64_t*>(d_offsets->data()),
      static_cast<const int64_t*>(d_members->data()), const_cast<void*>(d_output->data()),
      static_cast<uint8_t*>(const_cast<void*>(d_valid->data())),
      static_cast<int*>(const_cast<void*>(d_error->data())), nullptr);
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  EXPECT_EQ(host<int>(*d_error, 1)[0], 0);
  return host<std::complex<T>>(*d_output, static_cast<size_t>(virtual_plan.num_virtual));
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
  const auto output = run_stage<double>(static_cast<const int64_t*>(d_fold->data()), 1);
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
  const auto plan =
      hr::detail::build_virtual_plan({0, 0, 0, 2, 0, 0}, {0, 0, 0, 1, 0, 0}, 2, 2,
                                     hr::OverlapPolicy::kAverage, "DDMSparseDemultiplexOp");
  const auto geometry = hr::detail::make_array_geometry(plan, 0.004);
  EXPECT_EQ(geometry.num_elements(), plan.num_virtual);
  EXPECT_DOUBLE_EQ(geometry.wavelength, 0.004);
}

TEST(DDMSparseDemultiplexOp, UnwiredFoldDefaultsToZeroAndMaskMayBeUnwired) {
  auto fixture = independent_fold_fixture();
  const auto result = run_sparse_operator({fixture.rd_maps}, {fixture.detection_idx});
  ASSERT_TRUE(result.ok()) << result.describe_failure();
  const auto frames = sparse_snapshots(result);
  ASSERT_EQ(frames.size(), 1U);
  ASSERT_EQ(frames.front().shape, (std::vector<int64_t>{1, 2, 1}));
  const auto* actual = frames.front().as<std::complex<double>>();
  EXPECT_NE(actual[0], fixture.expected_snapshot[0]);
  EXPECT_NE(actual[1], fixture.expected_snapshot[1]);
  const auto geometries = take_geometries(result);
  ASSERT_EQ(geometries.size(), 1U);
  EXPECT_DOUBLE_EQ(geometries[0].wavelength, 0.004);
  EXPECT_EQ(geometries[0].element_positions, sparse_params().tx_positions);
}

TEST(DDMSparseDemultiplexOp, ConnectedNonzeroFoldRecoversPhysicalChannels) {
  auto fixture = independent_fold_fixture();
  const auto result =
      run_sparse_operator({fixture.rd_maps}, {fixture.detection_idx}, {fixture.fold_idx});
  ASSERT_TRUE(result.ok()) << result.describe_failure();
  const auto frames = sparse_snapshots(result);
  ASSERT_EQ(frames.size(), 1U);
  const auto* actual = frames.front().as<std::complex<double>>();
  for (size_t channel = 0; channel < fixture.expected_snapshot.size(); ++channel) {
    EXPECT_NEAR(actual[channel].real(), fixture.expected_snapshot[channel].real(), 1e-10);
    EXPECT_NEAR(actual[channel].imag(), fixture.expected_snapshot[channel].imag(), 1e-10);
  }
}

TEST(DDMSparseDemultiplexOp, RejectsFoldRowCountMismatch) {
  auto fixture = independent_fold_fixture();
  auto wrong_rows = device(std::vector<int64_t>{0, 1}, {2}, DLDataType{kDLInt, 64, 1});
  const auto result = run_sparse_operator({fixture.rd_maps}, {fixture.detection_idx}, {wrong_rows});
  EXPECT_FALSE(result.ok());
  EXPECT_TRUE(sparse_snapshots(result).empty());
  EXPECT_FALSE(result.diagnostics().empty());
}

TEST(DDMSparseDemultiplexOp, RejectsFoldOutsideTxRange) {
  for (const int64_t invalid_fold : {-1, 2}) {
    auto fixture = independent_fold_fixture();
    auto fold = device(std::vector<int64_t>{invalid_fold}, {1}, DLDataType{kDLInt, 64, 1});
    const auto result = run_sparse_operator({fixture.rd_maps}, {fixture.detection_idx}, {fold});
    EXPECT_FALSE(result.ok());
    EXPECT_TRUE(sparse_snapshots(result).empty());
    EXPECT_FALSE(result.diagnostics().empty());
  }
}

TEST(DDMSparseDemultiplexOp, ZeroDetectionsEmitsExactEmptySnapshot) {
  auto fixture = independent_fold_fixture();
  auto empty = device(std::vector<int64_t>{}, {0, 2}, DLDataType{kDLInt, 64, 1});
  const auto result = run_sparse_operator({fixture.rd_maps}, {empty});
  ASSERT_TRUE(result.ok()) << result.describe_failure();
  const auto frames = sparse_snapshots(result);
  ASSERT_EQ(frames.size(), 1U);
  EXPECT_EQ(frames.front().shape, (std::vector<int64_t>{0, 2, 1}));
  EXPECT_EQ(take_geometries(result).size(), 1U);
}

TEST(DDMSparseDemultiplexOp, LiveWavelengthAppliesOneTickThenReverts) {
  auto fixture = independent_fold_fixture();
  const auto result =
      run_sparse_operator({fixture.rd_maps, fixture.rd_maps, fixture.rd_maps},
                          {fixture.detection_idx, fixture.detection_idx, fixture.detection_idx}, {},
                          {std::nullopt, 0.008, std::nullopt});
  ASSERT_TRUE(result.ok()) << result.describe_failure();
  ASSERT_EQ(sparse_snapshots(result).size(), 3U);
  const auto geometries = take_geometries(result);
  ASSERT_EQ(geometries.size(), 3U);
  EXPECT_DOUBLE_EQ(geometries[0].wavelength, 0.004);
  EXPECT_DOUBLE_EQ(geometries[1].wavelength, 0.008);
  EXPECT_DOUBLE_EQ(geometries[2].wavelength, 0.004);
}

}  // namespace
