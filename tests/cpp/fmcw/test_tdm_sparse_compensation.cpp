// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cuda_runtime_api.h>

#include <complex>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "fmcw/mimo_coherent_compensation_op/mimo_coherent_compensation_stage.hpp"
#include "fmcw/tdm_sparse_compensation_op/tdm_sparse_compensation_op.hpp"
#include "fmcw/tdm_sparse_compensation_op/tdm_sparse_compensation_stage.hpp"
#include "fmcw/virtual_array_assembly_op/virtual_array_assembly_stage.hpp"
#include "radar_common/descriptors/fmcw_waveform_descriptor.hpp"
#include "radar_common/tensor/dlpack_buffer.hpp"
#include "radar_common/tensor/tensor_utils.hpp"

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
std::shared_ptr<holoscan::Tensor> device_tensor(
    const std::vector<T>& host, const std::vector<int64_t>& shape,
    DLDataType dtype) {
  auto result = hr::make_device_tensor(shape, dtype);
  if (!host.empty()) {
    auto pointer = result->mutable_data();
    EXPECT_TRUE(pointer.has_value());
    EXPECT_EQ(
        cudaMemcpy(*pointer, host.data(), host.size() * sizeof(T),
                   cudaMemcpyHostToDevice),
        cudaSuccess);
  }
  return result;
}

template <typename T>
std::vector<T> copy_to_host(
    const holoscan::Tensor& tensor, std::size_t count) {
  std::vector<T> result(count);
  EXPECT_EQ(
      cudaMemcpy(result.data(), tensor.data(), count * sizeof(T),
                 cudaMemcpyDeviceToHost),
      cudaSuccess);
  return result;
}

template <typename T>
struct StageResult {
  std::vector<std::complex<T>> snapshot;
  std::vector<uint8_t> valid;
  hr::detail::VirtualAssemblyPlan virtual_plan;
};

template <typename T>
StageResult<T> run_stage(
    hr::OverlapPolicy policy, int64_t doppler_half,
    const std::vector<double>& tx_positions,
    const std::vector<std::complex<T>>& corrected_values) {
  const auto wave = waveform();
  const auto correction_double =
      hr::detail::build_tdm_correction_table(
          wave, kNumTx, kDoppler);
  std::vector<std::complex<T>> correction;
  correction.reserve(correction_double.size());
  for (const auto& value : correction_double) {
    correction.emplace_back(
        static_cast<T>(value.real()), static_cast<T>(value.imag()));
  }

  std::vector<std::complex<T>> input(
      static_cast<std::size_t>(
          kNumTx * kNumRx * kDoppler * kRange));
  for (int64_t tx = 0; tx < kNumTx; ++tx) {
    for (int64_t rx = 0; rx < kNumRx; ++rx) {
      const int64_t channel = tx * kNumRx + rx;
      for (int64_t doppler = 0; doppler < kDoppler; ++doppler) {
        for (int64_t range = 0; range < kRange; ++range) {
          const int64_t index =
              ((channel * kDoppler + doppler) * kRange) + range;
          input[static_cast<std::size_t>(index)] =
              corrected_values[static_cast<std::size_t>(channel)] *
              std::conj(correction[static_cast<std::size_t>(
                  tx * kDoppler + doppler)]);
        }
      }
    }
  }

  const std::vector<int64_t> detections{3, 1};
  const std::vector<double> rx_positions{
      0.0, 0.0, 0.0,
      1.0, 0.0, 0.0};
  auto virtual_plan = hr::detail::build_virtual_plan(
      tx_positions, rx_positions, kNumTx, kNumRx, policy);
  const int64_t snapshots = 2 * doppler_half + 1;

  auto rd_device = device_tensor(
      input, {kNumTx, kNumRx, kDoppler, kRange},
      hr::complex_dtype<T>());
  auto detection_device = device_tensor(
      detections, {1, 2}, DLDataType{kDLInt, 64, 1});
  auto correction_device = device_tensor(
      correction, {kNumTx, kDoppler}, hr::complex_dtype<T>());
  const std::vector<int64_t> src_host =
      virtual_plan.src_channel.empty()
          ? std::vector<int64_t>{0}
          : virtual_plan.src_channel;
  const std::vector<int64_t> offsets_host =
      virtual_plan.member_offsets.empty()
          ? std::vector<int64_t>{0}
          : virtual_plan.member_offsets;
  const std::vector<int64_t> members_host =
      virtual_plan.member_indices.empty()
          ? std::vector<int64_t>{0}
          : virtual_plan.member_indices;
  auto src_device = device_tensor(
      src_host,
      {static_cast<int64_t>(src_host.size())},
      DLDataType{kDLInt, 64, 1});
  auto offsets_device = device_tensor(
      offsets_host,
      {static_cast<int64_t>(offsets_host.size())},
      DLDataType{kDLInt, 64, 1});
  auto members_device = device_tensor(
      members_host,
      {static_cast<int64_t>(members_host.size())},
      DLDataType{kDLInt, 64, 1});
  auto snapshot_device = hr::make_device_tensor(
      {1, virtual_plan.num_virtual, snapshots},
      hr::complex_dtype<T>());
  auto valid_device = hr::make_device_tensor(
      {1, snapshots}, DLDataType{kDLUInt, 8, 1});
  auto error_device = hr::make_device_tensor(
      {1}, DLDataType{kDLInt, 32, 1});
  EXPECT_EQ(
      cudaMemset(
          const_cast<void*>(error_device->data()), 0, sizeof(int)),
      cudaSuccess);

  const hr::detail::RdMapsLayout layout{
      .num_tx = kNumTx,
      .num_rx = kNumRx,
      .num_doppler = kDoppler,
      .num_range = kRange};
  hr::detail::run_tdm_sparse_compensation_stage<T>(
      rd_device->data(),
      static_cast<const int64_t*>(detection_device->data()), 1, layout,
      snapshots, doppler_half, 0, hr::SnapshotEdgeMode::kWrap,
      hr::SnapshotEdgeMode::kShift, virtual_plan,
      correction_device->data(),
      static_cast<const int64_t*>(src_device->data()),
      static_cast<const int64_t*>(offsets_device->data()),
      static_cast<const int64_t*>(members_device->data()),
      const_cast<void*>(snapshot_device->data()),
      static_cast<uint8_t*>(
          const_cast<void*>(valid_device->data())),
      static_cast<int*>(const_cast<void*>(error_device->data())),
      nullptr);
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  const auto errors = copy_to_host<int>(*error_device, 1);
  EXPECT_EQ(errors[0], 0);

  return {
      copy_to_host<std::complex<T>>(
          *snapshot_device,
          static_cast<std::size_t>(
              virtual_plan.num_virtual * snapshots)),
      copy_to_host<uint8_t>(
          *valid_device, static_cast<std::size_t>(snapshots)),
      std::move(virtual_plan)};
}

TEST(TDMSparseCompensationStage, KeepAllCorrectsEveryWindowCell) {
  const std::vector<std::complex<float>> values{
      {1.0F, 0.25F}, {2.0F, 0.5F},
      {3.0F, 0.75F}, {4.0F, 1.0F}};
  const std::vector<double> tx_positions{
      0.0, 0.0, 0.0,
      2.0, 0.0, 0.0};
  const auto result = run_stage<float>(
      hr::OverlapPolicy::kKeepAll, 1, tx_positions, values);
  ASSERT_EQ(result.virtual_plan.num_virtual, 4);
  ASSERT_EQ(result.snapshot.size(), 12U);
  EXPECT_EQ(result.valid, (std::vector<uint8_t>{1, 1, 1}));
  for (int64_t channel = 0; channel < 4; ++channel) {
    for (int64_t slot = 0; slot < 3; ++slot) {
      const auto actual = result.snapshot[
          static_cast<std::size_t>(channel * 3 + slot)];
      EXPECT_NEAR(actual.real(), values[channel].real(), 1e-5F);
      EXPECT_NEAR(actual.imag(), values[channel].imag(), 1e-5F);
    }
  }
}

TEST(TDMSparseCompensationStage, DedupAndAverageMatchVirtualPlan) {
  const std::vector<std::complex<double>> values{
      {1.0, 0.0}, {2.0, 0.0}, {3.0, 0.0}, {4.0, 0.0}};
  const std::vector<double> overlapping_tx_positions{
      0.0, 0.0, 0.0,
      0.0, 0.0, 0.0};

  const auto dedup = run_stage<double>(
      hr::OverlapPolicy::kDedup, 0,
      overlapping_tx_positions, values);
  ASSERT_EQ(dedup.virtual_plan.num_virtual, 2);
  EXPECT_NEAR(dedup.snapshot[0].real(), 1.0, 1e-12);
  EXPECT_NEAR(dedup.snapshot[1].real(), 2.0, 1e-12);

  const auto average = run_stage<double>(
      hr::OverlapPolicy::kAverage, 0,
      overlapping_tx_positions, values);
  ASSERT_EQ(average.virtual_plan.num_virtual, 2);
  EXPECT_NEAR(average.snapshot[0].real(), 2.0, 1e-12);
  EXPECT_NEAR(average.snapshot[1].real(), 3.0, 1e-12);
}

TEST(TDMSparseCompensationStage, GeometryUsesWaveformWavelength) {
  const std::vector<std::complex<float>> values(4, {1.0F, 0.0F});
  const std::vector<double> tx_positions{
      0.0, 0.0, 0.0,
      2.0, 0.0, 0.0};
  const auto result = run_stage<float>(
      hr::OverlapPolicy::kKeepAll, 0, tx_positions, values);
  const auto geometry = hr::detail::make_array_geometry(
      result.virtual_plan, waveform().wavelength());
  EXPECT_DOUBLE_EQ(geometry.wavelength, waveform().wavelength());
  EXPECT_EQ(
      geometry.element_positions,
      result.virtual_plan.virtual_positions);
}

}  // namespace
