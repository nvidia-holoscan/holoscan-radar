// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cuda_runtime_api.h>

#include <cmath>
#include <chrono>
#include <complex>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include <holoscan/core/connection_options.hpp>
#include <holoscan/testing/operator_test_harness.hpp>

#include "fmcw/tdm_sparse_compensation_op/tdm_sparse_compensation_op.hpp"
#include "fmcw/tdm_sparse_compensation_op/tdm_sparse_compensation_stage.hpp"
#include "radar_common/descriptors/fmcw_waveform_descriptor.hpp"
#include "radar_common/dsp/mimo_coherent_compensation_stage.hpp"
#include "radar_common/dsp/virtual_array.hpp"
#include "radar_common/math/constants.hpp"
#include "radar_common/tensor/dlpack_buffer.hpp"
#include "radar_common/tensor/tensor_utils.hpp"
#include "radar_schemas/array_geometry_bridge.hpp"
#include "utils/tensor_fixtures.hpp"

namespace {

namespace hr = holoscan::radar;

constexpr int64_t kNumTx = 2;
constexpr int64_t kNumRx = 2;
constexpr int64_t kDoppler = 8;
constexpr int64_t kRange = 3;

hr::FMCWWaveformDescriptor waveform() {
  hr::FMCWWaveformSpec spec;
  spec.carrier_frequency = 77.0e9;
  spec.bandwidth = 1.0e6;
  spec.chirp_duration = 50.0e-6;
  spec.sample_rate = 2.0e6;
  spec.chirp_repetition_interval = 1.0e-3;
  return hr::FMCWWaveformDescriptor(spec);
}

template <typename T>
std::shared_ptr<holoscan::Tensor> device_tensor(const std::vector<T>& host,
                                                const std::vector<int64_t>& shape,
                                                DLDataType dtype) {
  return hr::testing::to_device_tensor(host, shape, dtype);
}

template <typename T>
std::vector<T> copy_to_host(const holoscan::Tensor& tensor, std::size_t count) {
  std::vector<T> result(count);
  EXPECT_EQ(cudaMemcpy(result.data(), tensor.data(), count * sizeof(T), cudaMemcpyDeviceToHost),
            cudaSuccess);
  return result;
}

template <typename T>
struct StageResult {
  std::vector<std::complex<T>> snapshot;
  std::vector<uint8_t> valid;
  hr::detail::VirtualAssemblyPlan virtual_plan;
  int error{0};
};

template <typename T>
StageResult<T> run_stage(hr::OverlapPolicy policy, int64_t doppler_half,
                         const std::vector<double>& tx_positions,
                         const std::vector<std::complex<T>>& corrected_values,
                         std::optional<int64_t> supplied_fold = std::nullopt,
                         const std::vector<int64_t>& injected_fold_by_doppler = {},
                         int64_t detection_doppler = 3) {
  const auto wave = waveform();
  const auto correction_double = hr::detail::build_tdm_correction_table(wave, kNumTx, kDoppler);
  std::vector<std::complex<T>> correction;
  correction.reserve(correction_double.size());
  for (const auto& value : correction_double) {
    correction.emplace_back(static_cast<T>(value.real()), static_cast<T>(value.imag()));
  }
  const auto fold_correction_double = hr::detail::build_tdm_fold_correction_table(kNumTx);
  std::vector<std::complex<T>> fold_correction;
  fold_correction.reserve(fold_correction_double.size());
  for (const auto& value : fold_correction_double) {
    fold_correction.emplace_back(static_cast<T>(value.real()), static_cast<T>(value.imag()));
  }

  std::vector<std::complex<T>> input(static_cast<std::size_t>(kNumTx * kNumRx * kDoppler * kRange));
  for (int64_t tx = 0; tx < kNumTx; ++tx) {
    for (int64_t rx = 0; rx < kNumRx; ++rx) {
      const int64_t channel = tx * kNumRx + rx;
      for (int64_t doppler = 0; doppler < kDoppler; ++doppler) {
        const int64_t injected_fold =
            injected_fold_by_doppler.empty()
                ? 0
                : injected_fold_by_doppler[static_cast<std::size_t>(doppler)];
        const double injected_phase =
            hr::kTwoPi * static_cast<double>(injected_fold * tx) / static_cast<double>(kNumTx);
        const std::complex<T> injected_rotation{static_cast<T>(std::cos(injected_phase)),
                                                static_cast<T>(std::sin(injected_phase))};
        for (int64_t range = 0; range < kRange; ++range) {
          const int64_t index = ((channel * kDoppler + doppler) * kRange) + range;
          input[static_cast<std::size_t>(index)] =
              corrected_values[static_cast<std::size_t>(channel)] *
              std::conj(correction[static_cast<std::size_t>(tx * kDoppler + doppler)]) *
              injected_rotation;
        }
      }
    }
  }

  const std::vector<int64_t> detections{detection_doppler, 1};
  const std::vector<double> rx_positions{0.0, 0.0, 0.0, 1.0, 0.0, 0.0};
  auto virtual_plan = hr::detail::build_virtual_plan(tx_positions, rx_positions, kNumTx, kNumRx,
                                                     policy, "TDMSparseCompensationOp");
  const int64_t snapshots = 2 * doppler_half + 1;

  auto rd_device = device_tensor(input, {kNumTx, kNumRx, kDoppler, kRange}, hr::complex_dtype<T>());
  auto detection_device = device_tensor(detections, {1, 2}, DLDataType{kDLInt, 64, 1});
  auto correction_device = device_tensor(correction, {kNumTx, kDoppler}, hr::complex_dtype<T>());
  auto fold_correction_device =
      device_tensor(fold_correction, {kNumTx, kNumTx}, hr::complex_dtype<T>());
  std::shared_ptr<holoscan::Tensor> fold_device;
  if (supplied_fold.has_value()) {
    fold_device =
        device_tensor(std::vector<int64_t>{*supplied_fold}, {1}, DLDataType{kDLInt, 64, 1});
  }
  const std::vector<int64_t> src_host =
      virtual_plan.src_channel.empty() ? std::vector<int64_t>{0} : virtual_plan.src_channel;
  const std::vector<int64_t> offsets_host =
      virtual_plan.member_offsets.empty() ? std::vector<int64_t>{0} : virtual_plan.member_offsets;
  const std::vector<int64_t> members_host =
      virtual_plan.member_indices.empty() ? std::vector<int64_t>{0} : virtual_plan.member_indices;
  auto src_device =
      device_tensor(src_host, {static_cast<int64_t>(src_host.size())}, DLDataType{kDLInt, 64, 1});
  auto offsets_device = device_tensor(offsets_host, {static_cast<int64_t>(offsets_host.size())},
                                      DLDataType{kDLInt, 64, 1});
  auto members_device = device_tensor(members_host, {static_cast<int64_t>(members_host.size())},
                                      DLDataType{kDLInt, 64, 1});
  auto snapshot_device =
      hr::make_device_tensor({1, virtual_plan.num_virtual, snapshots}, hr::complex_dtype<T>());
  auto valid_device = hr::make_device_tensor({1, snapshots}, DLDataType{kDLUInt, 8, 1});
  auto error_device = hr::make_device_tensor({1}, DLDataType{kDLInt, 32, 1});
  EXPECT_EQ(cudaMemset(const_cast<void*>(error_device->data()), 0, sizeof(int)), cudaSuccess);

  const hr::detail::TdmRdMapsLayout layout{
      .num_tx = kNumTx, .num_rx = kNumRx, .num_doppler = kDoppler, .num_range = kRange};
  hr::detail::run_tdm_sparse_compensation_stage<T>(
      rd_device->data(), static_cast<const int64_t*>(detection_device->data()),
      fold_device == nullptr ? nullptr : static_cast<const int64_t*>(fold_device->data()), 1,
      layout, snapshots, doppler_half, 0, hr::SnapshotEdgeMode::kWrap, hr::SnapshotEdgeMode::kShift,
      virtual_plan, correction_device->data(), fold_correction_device->data(),
      static_cast<const int64_t*>(src_device->data()),
      static_cast<const int64_t*>(offsets_device->data()),
      static_cast<const int64_t*>(members_device->data()),
      const_cast<void*>(snapshot_device->data()),
      static_cast<uint8_t*>(const_cast<void*>(valid_device->data())),
      static_cast<int*>(const_cast<void*>(error_device->data())), nullptr);
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  const auto errors = copy_to_host<int>(*error_device, 1);

  return {copy_to_host<std::complex<T>>(
              *snapshot_device, static_cast<std::size_t>(virtual_plan.num_virtual * snapshots)),
          copy_to_host<uint8_t>(*valid_device, static_cast<std::size_t>(snapshots)),
          std::move(virtual_plan), errors[0]};
}

TEST(TDMSparseCompensationStage, KeepAllCorrectsEveryWindowCell) {
  const std::vector<std::complex<float>> values{
      {1.0F, 0.25F}, {2.0F, 0.5F}, {3.0F, 0.75F}, {4.0F, 1.0F}};
  const std::vector<double> tx_positions{0.0, 0.0, 0.0, 2.0, 0.0, 0.0};
  const auto result = run_stage<float>(hr::OverlapPolicy::kKeepAll, 1, tx_positions, values);
  ASSERT_EQ(result.virtual_plan.num_virtual, 4);
  EXPECT_EQ(result.error, 0);
  ASSERT_EQ(result.snapshot.size(), 12U);
  EXPECT_EQ(result.valid, (std::vector<uint8_t>{1, 1, 1}));
  for (int64_t channel = 0; channel < 4; ++channel) {
    for (int64_t slot = 0; slot < 3; ++slot) {
      const auto actual = result.snapshot[static_cast<std::size_t>(channel * 3 + slot)];
      EXPECT_NEAR(actual.real(), values[channel].real(), 1e-5F);
      EXPECT_NEAR(actual.imag(), values[channel].imag(), 1e-5F);
    }
  }
}

TEST(TDMSparseCompensationStage, DedupAndAverageMatchVirtualPlan) {
  const std::vector<std::complex<double>> values{{1.0, 0.0}, {2.0, 0.0}, {3.0, 0.0}, {4.0, 0.0}};
  const std::vector<double> overlapping_tx_positions{0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

  const auto dedup =
      run_stage<double>(hr::OverlapPolicy::kDedup, 0, overlapping_tx_positions, values);
  ASSERT_EQ(dedup.virtual_plan.num_virtual, 2);
  EXPECT_EQ(dedup.error, 0);
  EXPECT_NEAR(dedup.snapshot[0].real(), 1.0, 1e-12);
  EXPECT_NEAR(dedup.snapshot[1].real(), 2.0, 1e-12);

  const auto average =
      run_stage<double>(hr::OverlapPolicy::kAverage, 0, overlapping_tx_positions, values);
  ASSERT_EQ(average.virtual_plan.num_virtual, 2);
  EXPECT_EQ(average.error, 0);
  EXPECT_NEAR(average.snapshot[0].real(), 2.0, 1e-12);
  EXPECT_NEAR(average.snapshot[1].real(), 3.0, 1e-12);
}

TEST(TDMSparseCompensationStage, GeometryUsesWaveformWavelength) {
  const std::vector<std::complex<float>> values(4, {1.0F, 0.0F});
  const std::vector<double> tx_positions{0.0, 0.0, 0.0, 2.0, 0.0, 0.0};
  const auto result = run_stage<float>(hr::OverlapPolicy::kKeepAll, 0, tx_positions, values);
  const auto geometry =
      hr::detail::make_array_geometry(result.virtual_plan, waveform().wavelength());
  EXPECT_DOUBLE_EQ(geometry.wavelength, waveform().wavelength());
  EXPECT_EQ(geometry.element_positions, result.virtual_plan.virtual_positions);
}

TEST(TDMSparseCompensationStage, DefaultZeroLeavesNonzeroFoldPhase) {
  const std::vector<std::complex<double>> values{{1.0, 0.0}, {2.0, 0.0}, {3.0, 0.0}, {4.0, 0.0}};
  const std::vector<double> tx_positions{0.0, 0.0, 0.0, 2.0, 0.0, 0.0};
  const std::vector<int64_t> injected(kDoppler, 1);
  const auto result = run_stage<double>(hr::OverlapPolicy::kKeepAll, 0, tx_positions, values,
                                        std::nullopt, injected);
  EXPECT_EQ(result.error, 0);
  EXPECT_NEAR(result.snapshot[0].real(), 1.0, 1e-12);
  EXPECT_NEAR(result.snapshot[2].real(), -3.0, 1e-12);
}

TEST(TDMSparseCompensationMath, FoldCorrectionUsesFiringSlot) {
  const auto correction = hr::detail::build_tdm_fold_correction_table(kNumTx, {1, 0});
  ASSERT_EQ(correction.size(), 4U);
  EXPECT_NEAR(correction[2].real(), -1.0, 1e-12);
  EXPECT_NEAR(correction[2].imag(), 0.0, 1e-12);
  EXPECT_NEAR(correction[3].real(), 1.0, 1e-12);
  EXPECT_NEAR(correction[3].imag(), 0.0, 1e-12);
}

TEST(TDMSparseCompensationStage, SuppliedFoldRecoversTxPhase) {
  const std::vector<std::complex<double>> values{{1.0, 0.0}, {2.0, 0.0}, {3.0, 0.0}, {4.0, 0.0}};
  const std::vector<double> tx_positions{0.0, 0.0, 0.0, 2.0, 0.0, 0.0};
  const std::vector<int64_t> injected(kDoppler, 1);
  const auto result =
      run_stage<double>(hr::OverlapPolicy::kKeepAll, 0, tx_positions, values, 1, injected);
  EXPECT_EQ(result.error, 0);
  for (std::size_t channel = 0; channel < values.size(); ++channel) {
    EXPECT_NEAR(result.snapshot[channel].real(), values[channel].real(), 1e-12);
    EXPECT_NEAR(result.snapshot[channel].imag(), values[channel].imag(), 1e-12);
  }
}

TEST(TDMSparseCompensationStage, WrappedWindowAdjustsFoldClass) {
  const std::vector<std::complex<double>> values{{1.0, 0.0}, {2.0, 0.0}, {3.0, 0.0}, {4.0, 0.0}};
  const std::vector<double> tx_positions{0.0, 0.0, 0.0, 2.0, 0.0, 0.0};
  std::vector<int64_t> injected(kDoppler, 1);
  injected.back() = 0;
  const auto result =
      run_stage<double>(hr::OverlapPolicy::kKeepAll, 1, tx_positions, values, 1, injected, 0);
  EXPECT_EQ(result.error, 0);
  for (std::size_t channel = 0; channel < values.size(); ++channel) {
    for (int64_t slot = 0; slot < 3; ++slot) {
      const auto actual = result.snapshot[channel * 3 + slot];
      EXPECT_NEAR(actual.real(), values[channel].real(), 1e-12);
      EXPECT_NEAR(actual.imag(), values[channel].imag(), 1e-12);
    }
  }
}

TEST(TDMSparseCompensationStage, RejectsFoldOutsideTxRange) {
  const std::vector<std::complex<float>> values(4, {1.0F, 0.0F});
  const std::vector<double> tx_positions{0.0, 0.0, 0.0, 2.0, 0.0, 0.0};
  for (const int64_t fold : std::vector<int64_t>{-1, kNumTx}) {
    const auto result =
        run_stage<float>(hr::OverlapPolicy::kKeepAll, 0, tx_positions, values, fold);
    EXPECT_EQ(result.error, hr::detail::kSparseTdmErrFold);
  }
}

hr::TDMSparseCompensationParams sparse_params() {
  return hr::TDMSparseCompensationParams{
      .waveform = waveform(),
      .tx_positions = {0.0, 0.0, 0.0, 2.0, 0.0, 0.0},
      .rx_positions = {0.0, 0.0, 0.0},
      .overlap_policy = hr::OverlapPolicy::kKeepAll,
      .snapshot_window = {0, 0},
  };
}

hr::MaxTensorExtents sparse_extents() {
  hr::MaxTensorExtents extents;
  extents.max_tx = kNumTx;
  extents.max_rx = 1;
  extents.max_doppler_bins = kDoppler;
  extents.max_range_bins = 1;
  extents.max_detections = 2;
  extents.max_virtual_channels = kNumTx;
  extents.max_snapshots = 1;
  return extents;
}

holoscan::testing::TestResult run_sparse_operator(
    std::shared_ptr<holoscan::Tensor> rd_maps, std::shared_ptr<holoscan::Tensor> detection,
    std::shared_ptr<holoscan::Tensor> fold = nullptr) {
  using Op = hr::TDMSparseCompensationOp<double>;
  auto fixture = holoscan::testing::create_operator_test<Op>(sparse_extents());
  fixture.set_params(sparse_params());
  if (fold) {
    fixture.push_frame(holoscan::testing::frame(
        holoscan::testing::input("fold_idx", *fold), holoscan::testing::input("rd_maps", *rd_maps),
        holoscan::testing::input("detection_idx", *detection)));
  } else {
    auto unmatched_fold = device_tensor(std::vector<int64_t>{0}, {1}, DLDataType{kDLInt, 64, 1});
    fixture.push_at("fold_idx", *unmatched_fold,
                    std::chrono::milliseconds(100) + std::chrono::nanoseconds(1));
    fixture.push_frame(holoscan::testing::frame_at(
        std::chrono::milliseconds(100), holoscan::testing::input("rd_maps", *rd_maps),
        holoscan::testing::input("detection_idx", *detection)));
  }
  constexpr std::size_t kMaximumBytes = 256U * 1024U * 1024U;
  fixture.capture_snapshot<holoscan::Tensor>("snapshot", 1U, kMaximumBytes);
  fixture.capture_snapshot<holoscan::Tensor>("is_valid", 1U, kMaximumBytes);
  fixture.capture_snapshot<hr::WireArrayGeometry>("array_geometry", 1U, kMaximumBytes);
  for (const char* input : {"rd_maps", "detection_idx"}) {
    fixture.input_source_device(input, holoscan::DeviceId{0});
  }
  fixture.input_source_device("fold_idx", holoscan::DeviceId{0});
  fixture.output_device("snapshot", holoscan::DeviceId{0});
  fixture.output_device("is_valid", holoscan::DeviceId{0});
  auto running = fixture.run_async({.expect_stop = true});
  (void)running.wait_for_output("snapshot", 1U, std::chrono::seconds{5});
  (void)running.wait_for_output("is_valid", 1U, std::chrono::seconds{5});
  (void)running.wait_for_output("array_geometry", 1U, std::chrono::seconds{5});
  running.request_stop();
  return running.finish();
}

std::vector<hr::testing::HostTensor> sparse_snapshots(const holoscan::testing::TestResult& result) {
  std::vector<hr::testing::HostTensor> frames;
  for (const auto& snapshot : result.template snapshots<holoscan::Tensor>("snapshot")) {
    frames.push_back(hr::testing::to_host(snapshot));
  }
  return frames;
}

TEST(TDMSparseCompensationOp, UnwiredFoldDefaultsToZero) {
  const std::vector<std::complex<double>> host(static_cast<std::size_t>(kNumTx * kDoppler));
  auto rd = device_tensor(host, {kNumTx, 1, kDoppler, 1}, hr::complex_dtype<double>());
  auto detection = device_tensor(std::vector<int64_t>{3, 0}, {1, 2}, DLDataType{kDLInt, 64, 1});
  const auto result = run_sparse_operator(rd, detection);
  ASSERT_TRUE(result.ok()) << result.describe_failure();
  const auto frames = sparse_snapshots(result);
  ASSERT_EQ(frames.size(), 1U);
  EXPECT_EQ(frames.front().shape, (std::vector<int64_t>{1, kNumTx, 1}));
}

TEST(TDMSparseCompensationOp, RejectsFoldRowCountMismatch) {
  const std::vector<std::complex<double>> host(static_cast<std::size_t>(kNumTx * kDoppler));
  auto rd = device_tensor(host, {kNumTx, 1, kDoppler, 1}, hr::complex_dtype<double>());
  auto detection = device_tensor(std::vector<int64_t>{3, 0}, {1, 2}, DLDataType{kDLInt, 64, 1});
  auto fold = device_tensor(std::vector<int64_t>{0, 1}, {2}, DLDataType{kDLInt, 64, 1});
  const auto result = run_sparse_operator(rd, detection, fold);
  EXPECT_FALSE(result.ok());
  EXPECT_TRUE(sparse_snapshots(result).empty());
  EXPECT_FALSE(result.diagnostics().empty());
}

}  // namespace
