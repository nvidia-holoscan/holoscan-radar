// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Contract tests for MIMOCoherentCompensationOp. TDM: per-Doppler-bin inter-TX
// phase correction, sticky [live] waveform overrides. DDM: standard
// linear-phase Doppler-band demux into an explicit TX axis (offsets +
// linear-phase codes, guard bands, arbitrary-code rejection, inert waveform).
// Plus rank-agnostic batching and cross-cutting input validation.

#include <gtest/gtest.h>

#include <cuda_runtime_api.h>

#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/holoscan.hpp>
#include <holoscan/test/test_harness.hpp>

#include "fmcw/mimo_coherent_compensation_op/mimo_coherent_compensation_op.hpp"
#include "fmcw/mimo_coherent_compensation_op/mimo_coherent_compensation_stage.hpp"
#include "radar_common/descriptors/enums.hpp"
#include "radar_common/descriptors/fmcw_waveform_descriptor.hpp"
#include "radar_common/tensor/tensor_utils.hpp"
#include "utils/sdk_harness.hpp"
#include "utils/tensor_fixtures.hpp"

namespace {

using holoscan::radar::DDMCodeDescriptor;
using holoscan::radar::FMCWWaveformDescriptor;
using holoscan::radar::FMCWWaveformSpec;
using holoscan::radar::MIMOCoherentCompensationOp;
using holoscan::radar::MultiplexingScheme;
using holoscan::radar::complex_dtype;
using holoscan::radar::dldatatype_equals;
using holoscan::radar::testing::HostTensor;
using holoscan::radar::testing::single_map;
using holoscan::radar::testing::TensorMapCapture;
using holoscan::radar::testing::to_device_tensor;
using holoscan::radar::testing::to_host;
using holoscan::radar::testing::to_pageable_host_tensor;
using holoscan::radar::testing::wire_capture_sink;
using holoscan::radar::testing::wire_data_source;
using holoscan::radar::testing::wire_leading_override;
using holoscan::test::create_operator_test;

constexpr int64_t kNumTx = 2;
constexpr int64_t kNumRx = 2;
constexpr int64_t kDoppler = 8;
constexpr int64_t kRange = 4;
constexpr double kPri = 1.0e-3;

FMCWWaveformDescriptor make_test_waveform(double pri = kPri) {
  FMCWWaveformSpec spec;
  spec.carrier_frequency = 77.0e9;
  spec.bandwidth = 1.0e6;
  spec.chirp_duration = 50.0e-6;
  spec.sample_rate = 2.0e6;
  spec.chirp_repetition_interval = pri;
  return FMCWWaveformDescriptor(spec);
}

int64_t tensor_count(const std::vector<int64_t>& shape) {
  int64_t count = 1;
  for (int64_t d : shape) {
    count *= d;
  }
  return count;
}

template <typename T>
std::vector<std::complex<T>> build_misaligned_rd_host(
    const FMCWWaveformDescriptor& waveform, int64_t num_tx, int64_t num_rx,
    int64_t num_doppler, int64_t num_range, int64_t batch = 1) {
  const auto table = holoscan::radar::detail::build_tdm_correction_table(
      waveform, num_tx, num_doppler);
  const int64_t slice = num_tx * num_rx * num_doppler * num_range;
  std::vector<std::complex<T>> host(static_cast<size_t>(batch * slice));
  for (int64_t b = 0; b < batch; ++b) {
    for (int64_t tx = 0; tx < num_tx; ++tx) {
      for (int64_t rx = 0; rx < num_rx; ++rx) {
        for (int64_t d = 0; d < num_doppler; ++d) {
          for (int64_t r = 0; r < num_range; ++r) {
            const size_t idx = static_cast<size_t>(
                b * slice +
                (((tx * num_rx + rx) * num_doppler + d) * num_range) + r);
            const auto corr =
                table[static_cast<size_t>(tx * num_doppler + d)];
            const double mag =
                1.0 + 0.01 * static_cast<double>(rx) +
                0.001 * static_cast<double>(r);
            const std::complex<double> misaligned =
                std::conj(corr) * mag;
            host[idx] = std::complex<T>(
                static_cast<T>(misaligned.real()),
                static_cast<T>(misaligned.imag()));
          }
        }
      }
    }
  }
  return host;
}

template <typename T>
void expect_tx_lanes_aligned(const HostTensor& out, int64_t num_tx,
                             int64_t num_rx, int64_t num_doppler,
                             int64_t num_range, int64_t batch = 1,
                             T tol = static_cast<T>(1e-4)) {
  const auto* data = out.as<std::complex<T>>();
  const int64_t slice = num_tx * num_rx * num_doppler * num_range;
  for (int64_t b = 0; b < batch; ++b) {
    for (int64_t rx = 0; rx < num_rx; ++rx) {
      for (int64_t d = 0; d < num_doppler; ++d) {
        for (int64_t r = 0; r < num_range; ++r) {
          const size_t ref = static_cast<size_t>(
              b * slice + ((0 * num_rx + rx) * num_doppler + d) * num_range +
              r);
          for (int64_t tx = 1; tx < num_tx; ++tx) {
            const size_t other = static_cast<size_t>(
                b * slice + ((tx * num_rx + rx) * num_doppler + d) * num_range +
                r);
            EXPECT_NEAR(data[other].real(), data[ref].real(), tol)
                << "batch=" << b << " tx=" << tx << " rx=" << rx << " d=" << d
                << " r=" << r;
            EXPECT_NEAR(data[other].imag(), data[ref].imag(), tol)
                << "batch=" << b << " tx=" << tx << " rx=" << rx << " d=" << d
                << " r=" << r;
          }
        }
      }
    }
  }
}

namespace cuda_accessible {

struct BufferHolder {
  void* data{nullptr};
  std::vector<int64_t> shape;
  DLDevice device{kDLCUDA, 0};
  DLManagedTensor dl{};

  ~BufferHolder() {
    if (data == nullptr) {
      return;
    }
    if (device.device_type == kDLCUDA) {
      cudaFree(data);
    } else if (device.device_type == kDLCUDAManaged) {
      cudaFree(data);
    } else if (device.device_type == kDLCUDAHost) {
      cudaFreeHost(data);
    }
  }
};

template <typename ElementT>
std::shared_ptr<holoscan::Tensor> make_tensor(
    const std::vector<ElementT>& host, const std::vector<int64_t>& shape,
    DLDataType dtype, DLDeviceType device_type) {
  const size_t nbytes = host.size() * sizeof(ElementT);
  auto holder = std::make_unique<BufferHolder>();
  holder->shape = shape;
  int device_id = 0;
  cudaGetDevice(&device_id);
  holder->device = DLDevice{device_type, device_id};
  if (device_type == kDLCUDA) {
    if (cudaMalloc(&holder->data, nbytes) != cudaSuccess) {
      throw std::runtime_error("cuda_accessible: cudaMalloc failed");
    }
    if (cudaMemcpy(holder->data, host.data(), nbytes,
                   cudaMemcpyHostToDevice) != cudaSuccess) {
      throw std::runtime_error("cuda_accessible: H2D failed");
    }
  } else if (device_type == kDLCUDAManaged) {
    if (cudaMallocManaged(&holder->data, nbytes) != cudaSuccess) {
      throw std::runtime_error("cuda_accessible: cudaMallocManaged failed");
    }
    std::memcpy(holder->data, host.data(), nbytes);
  } else if (device_type == kDLCUDAHost) {
    if (cudaHostAlloc(&holder->data, nbytes, cudaHostAllocMapped) !=
        cudaSuccess) {
      throw std::runtime_error("cuda_accessible: cudaHostAlloc failed");
    }
    std::memcpy(holder->data, host.data(), nbytes);
  } else {
    throw std::invalid_argument("cuda_accessible: unsupported device type");
  }
  holder->dl.dl_tensor.data = holder->data;
  holder->dl.dl_tensor.device = holder->device;
  holder->dl.dl_tensor.ndim = static_cast<int32_t>(shape.size());
  holder->dl.dl_tensor.dtype = dtype;
  holder->dl.dl_tensor.shape = holder->shape.data();
  holder->dl.dl_tensor.strides = nullptr;
  holder->dl.dl_tensor.byte_offset = 0;
  holder->dl.manager_ctx = holder.get();
  holder->dl.deleter = [](DLManagedTensor* self) {
    delete static_cast<BufferHolder*>(self->manager_ctx);
  };
  auto tensor = std::make_shared<holoscan::Tensor>(&holder->dl);
  holder.release();
  return tensor;
}

}  // namespace cuda_accessible

struct StridedRdMapsHolder {
  void* data{nullptr};
  std::vector<int64_t> shape;
  std::vector<int64_t> strides;
  DLManagedTensor dl{};

  ~StridedRdMapsHolder() {
    if (data != nullptr) {
      cudaFree(data);
    }
  }
};

std::shared_ptr<holoscan::Tensor> make_non_contiguous_rd_maps(
    const std::vector<std::complex<float>>& host,
    const std::vector<int64_t>& shape) {
  const int64_t num_tx = shape[shape.size() - 4];
  const int64_t num_rx = shape[shape.size() - 3];
  const int64_t doppler = shape[shape.size() - 2];
  const int64_t range = shape.back();
  const int64_t range_stride = range + 1;
  const size_t alloc_elems = static_cast<size_t>(num_tx * num_rx * doppler *
                                                 range_stride);
  auto holder = std::make_unique<StridedRdMapsHolder>();
  holder->shape = shape;
  holder->strides = {num_rx * doppler * range_stride, doppler * range_stride,
                     range_stride, 1};
  if (cudaMalloc(&holder->data, alloc_elems * sizeof(std::complex<float>)) !=
      cudaSuccess) {
    throw std::runtime_error("make_non_contiguous_rd_maps: cudaMalloc failed");
  }
  std::vector<std::complex<float>> padded(alloc_elems, {0.0f, 0.0f});
  for (int64_t tx = 0; tx < num_tx; ++tx) {
    for (int64_t rx = 0; rx < num_rx; ++rx) {
      for (int64_t d = 0; d < doppler; ++d) {
        for (int64_t r = 0; r < range; ++r) {
          const size_t dst = static_cast<size_t>(
              tx * num_rx * doppler * range_stride +
              rx * doppler * range_stride + d * range_stride + r);
          const size_t src = static_cast<size_t>(
              (((tx * num_rx + rx) * doppler) + d) * range + r);
          padded[dst] = host[src];
        }
      }
    }
  }
  if (cudaMemcpy(holder->data, padded.data(),
                 alloc_elems * sizeof(std::complex<float>),
                 cudaMemcpyHostToDevice) != cudaSuccess) {
    throw std::runtime_error("make_non_contiguous_rd_maps: H2D failed");
  }
  int device_id = 0;
  cudaGetDevice(&device_id);
  holder->dl.dl_tensor.data = holder->data;
  holder->dl.dl_tensor.device = DLDevice{kDLCUDA, device_id};
  holder->dl.dl_tensor.ndim = static_cast<int32_t>(shape.size());
  holder->dl.dl_tensor.dtype = complex_dtype<float>();
  holder->dl.dl_tensor.shape = holder->shape.data();
  holder->dl.dl_tensor.strides = holder->strides.data();
  holder->dl.dl_tensor.byte_offset = 0;
  holder->dl.manager_ctx = holder.get();
  holder->dl.deleter = [](DLManagedTensor* self) {
    delete static_cast<StridedRdMapsHolder*>(self->manager_ctx);
  };
  auto tensor = std::make_shared<holoscan::Tensor>(&holder->dl);
  holder.release();
  return tensor;
}

struct OutputTensorCapture {
  std::vector<DLDevice> devices;
  std::vector<DLDataType> dtypes;

  std::vector<std::function<void(const holoscan::TensorMap&)>> validators() {
    return {[this](const holoscan::TensorMap& map) {
      ASSERT_EQ(map.size(), 1U);
      const auto& tensor = map.begin()->second;
      devices.push_back(tensor->device());
      dtypes.push_back(tensor->dtype());
    }};
  }
};

template <typename T>
HostTensor run_harness_capture_device(
    const holoscan::TensorMap& frame, const FMCWWaveformDescriptor& waveform,
    OutputTensorCapture* capture) {
  TensorMapCapture host_capture;
  auto harness = create_operator_test<MIMOCoherentCompensationOp<T>>(
      holoscan::Arg("multiplexing_scheme") = MultiplexingScheme::kTdm,
      holoscan::Arg("waveform") = waveform);
  harness->template add_input_port<holoscan::TensorMap>("rd_maps", {frame});
  std::vector<std::function<void(const holoscan::TensorMap&)>> validators =
      host_capture.validators();
  if (capture != nullptr) {
    auto device_validators = capture->validators();
    validators.insert(validators.end(), device_validators.begin(),
                      device_validators.end());
  }
  harness->template add_output_port<holoscan::TensorMap>("rd_maps", validators);
  harness->run_test();
  if (host_capture.frames.size() != 1) {
    throw std::runtime_error("expected exactly one captured output frame");
  }
  return std::move(host_capture.frames.front());
}

template <typename T>
HostTensor run_harness(const holoscan::TensorMap& frame,
                       const FMCWWaveformDescriptor& waveform) {
  TensorMapCapture capture;
  auto harness = create_operator_test<MIMOCoherentCompensationOp<T>>(
      holoscan::Arg("multiplexing_scheme") = MultiplexingScheme::kTdm,
      holoscan::Arg("waveform") = waveform);
  harness->template add_input_port<holoscan::TensorMap>("rd_maps", {frame});
  harness->template add_output_port<holoscan::TensorMap>(
      "rd_maps", capture.validators());
  harness->run_test();
  if (capture.frames.size() != 1) {
    throw std::runtime_error("expected exactly one captured output frame");
  }
  return std::move(capture.frames.front());
}

template <typename T>
HostTensor run_compensation(const std::vector<std::complex<T>>& host,
                            const std::vector<int64_t>& shape,
                            const FMCWWaveformDescriptor& waveform) {
  return run_harness<T>(
      single_map(to_device_tensor(host, shape, complex_dtype<T>()), "rd_maps"),
      waveform);
}

// ---- DDM helpers ----------------------------------------------------------

constexpr int64_t kDdmTx = 4;
constexpr int64_t kDdmRx = 2;
constexpr int64_t kDdmDoppler = 16;  // divisible by kDdmTx -> band = 4
constexpr int64_t kDdmRange = 3;

// fftshift1D center (matches the doppler stage / demux table).
int64_t fft_center(int64_t n) { return (n + 1) / 2; }

std::vector<double> uniform_offsets(int64_t num_tx) {
  std::vector<double> offsets(static_cast<size_t>(num_tx));
  for (int64_t m = 0; m < num_tx; ++m) {
    offsets[static_cast<size_t>(m)] =
        static_cast<double>(m) / static_cast<double>(num_tx);
  }
  return offsets;
}

DDMCodeDescriptor make_offset_code(int64_t num_tx, int64_t guard = 0) {
  DDMCodeDescriptor code;
  code.num_tx = num_tx;
  code.doppler_offsets = uniform_offsets(num_tx);
  code.guard_bands = guard;
  return code;
}

DDMCodeDescriptor make_offset_code_from(const std::vector<double>& offsets,
                                        int64_t guard = 0) {
  DDMCodeDescriptor code;
  code.num_tx = static_cast<int64_t>(offsets.size());
  code.doppler_offsets = offsets;
  code.guard_bands = guard;
  return code;
}

// A distinct known complex value per (tx, rx, range) cell.
template <typename T>
std::complex<T> ddm_cell_value(int64_t tx, int64_t rx, int64_t r) {
  return std::complex<T>(
      static_cast<T>(1.0 + 0.1 * static_cast<double>(tx) +
                     0.01 * static_cast<double>(rx)),
      static_cast<T>(0.05 * static_cast<double>(r) -
                     0.02 * static_cast<double>(tx)));
}

// Build a center-zero DDM input (..., N_rx, D, R). Each TX's copy of a single
// target sits at reduced fft-order bin p0 inside that TX's Doppler sub-band
// (start = round(offset * D)); every other bin is zero. Mirrors the forward
// linear-phase DDM model the demux inverts.
template <typename T>
std::vector<std::complex<T>> build_ddm_input_host(
    const std::vector<double>& offsets, int64_t num_tx, int64_t num_rx,
    int64_t num_doppler, int64_t num_range, int64_t p0_reduced,
    int64_t batch = 1) {
  const int64_t D = num_doppler;
  const int64_t c = fft_center(D);
  const int64_t slice = num_rx * D * num_range;
  std::vector<std::complex<T>> host(
      static_cast<size_t>(batch * slice), std::complex<T>(0, 0));
  for (int64_t b = 0; b < batch; ++b) {
    for (int64_t m = 0; m < num_tx; ++m) {
      const int64_t start =
          ((std::llround(offsets[static_cast<size_t>(m)] *
                         static_cast<double>(D)) %
            D) +
           D) %
          D;
      const int64_t j = (start + p0_reduced) % D;  // absolute fft bin
      const int64_t in_cz = ((j - c) % D + D) % D;  // center-zero index
      for (int64_t rx = 0; rx < num_rx; ++rx) {
        for (int64_t r = 0; r < num_range; ++r) {
          const size_t idx = static_cast<size_t>(
              b * slice + (rx * D + in_cz) * num_range + r);
          host[idx] = ddm_cell_value<T>(m, rx, r);
        }
      }
    }
  }
  return host;
}

template <typename T>
HostTensor run_ddm(const holoscan::TensorMap& frame,
                   const DDMCodeDescriptor& code,
                   const FMCWWaveformDescriptor& waveform) {
  TensorMapCapture capture;
  auto harness = create_operator_test<MIMOCoherentCompensationOp<T>>(
      holoscan::Arg("multiplexing_scheme") = MultiplexingScheme::kDdm,
      holoscan::Arg("waveform") = waveform, holoscan::Arg("ddm_code") = code);
  harness->template add_input_port<holoscan::TensorMap>("rd_maps", {frame});
  harness->template add_output_port<holoscan::TensorMap>(
      "rd_maps", capture.validators());
  harness->run_test();
  if (capture.frames.size() != 1) {
    throw std::runtime_error("expected exactly one DDM output frame");
  }
  return std::move(capture.frames.front());
}

template <typename T>
HostTensor run_ddm_from_host(const std::vector<std::complex<T>>& host,
                             const std::vector<int64_t>& shape,
                             const DDMCodeDescriptor& code,
                             const FMCWWaveformDescriptor& waveform) {
  return run_ddm<T>(
      single_map(to_device_tensor(host, shape, complex_dtype<T>()), "rd_maps"),
      code, waveform);
}

template <typename T>
HostTensor run_ddm_capture_device(
    const holoscan::TensorMap& frame, const DDMCodeDescriptor& code,
    const FMCWWaveformDescriptor& waveform, OutputTensorCapture* capture) {
  TensorMapCapture host_capture;
  auto harness = create_operator_test<MIMOCoherentCompensationOp<T>>(
      holoscan::Arg("multiplexing_scheme") = MultiplexingScheme::kDdm,
      holoscan::Arg("waveform") = waveform, holoscan::Arg("ddm_code") = code);
  harness->template add_input_port<holoscan::TensorMap>("rd_maps", {frame});
  std::vector<std::function<void(const holoscan::TensorMap&)>> validators =
      host_capture.validators();
  if (capture != nullptr) {
    auto device_validators = capture->validators();
    validators.insert(validators.end(), device_validators.begin(),
                      device_validators.end());
  }
  harness->template add_output_port<holoscan::TensorMap>("rd_maps", validators);
  harness->run_test();
  if (host_capture.frames.size() != 1) {
    throw std::runtime_error("expected exactly one DDM output frame");
  }
  return std::move(host_capture.frames.front());
}

// Assert the target injected at reduced fft bin p0 lands at the matching
// center-zero output bin in every (tx, rx) lane with the expected value, and
// all other output bins are zero.
template <typename T>
void expect_ddm_recovered(const HostTensor& out, int64_t num_tx,
                          int64_t num_rx, int64_t band_width, int64_t num_range,
                          int64_t p0_reduced, int64_t batch = 1,
                          T tol = static_cast<T>(1e-4)) {
  const int64_t W = band_width;
  const int64_t cW = fft_center(W);
  const int64_t o0 = ((p0_reduced - cW) % W + W) % W;
  const auto* data = out.as<std::complex<T>>();
  const int64_t slice = num_tx * num_rx * W * num_range;
  for (int64_t b = 0; b < batch; ++b) {
    for (int64_t m = 0; m < num_tx; ++m) {
      for (int64_t rx = 0; rx < num_rx; ++rx) {
        for (int64_t o = 0; o < W; ++o) {
          for (int64_t r = 0; r < num_range; ++r) {
            const size_t idx = static_cast<size_t>(
                b * slice + ((m * num_rx + rx) * W + o) * num_range + r);
            const std::complex<T> expected =
                (o == o0) ? ddm_cell_value<T>(m, rx, r)
                          : std::complex<T>(0, 0);
            EXPECT_NEAR(data[idx].real(), expected.real(), tol)
                << "batch=" << b << " tx=" << m << " rx=" << rx << " o=" << o
                << " r=" << r;
            EXPECT_NEAR(data[idx].imag(), expected.imag(), tol)
                << "batch=" << b << " tx=" << m << " rx=" << rx << " o=" << o
                << " r=" << r;
          }
        }
      }
    }
  }
}

TEST(MIMOCoherentCompensationOp, TdmRemovesInterTxPhaseAcrossDopplerBins) {
  const auto waveform = make_test_waveform();
  const std::vector<int64_t> shape{kNumTx, kNumRx, kDoppler, kRange};
  const auto out = run_compensation<float>(
      build_misaligned_rd_host<float>(waveform, kNumTx, kNumRx, kDoppler,
                                      kRange),
      shape, waveform);
  expect_tx_lanes_aligned<float>(out, kNumTx, kNumRx, kDoppler, kRange);
}

TEST(MIMOCoherentCompensationOp, FourTxLanesAlignAfterCompensation) {
  constexpr int64_t kTx = 4;
  const auto waveform = make_test_waveform();
  const std::vector<int64_t> shape{kTx, kNumRx, kDoppler, kRange};
  const auto out = run_compensation<float>(
      build_misaligned_rd_host<float>(waveform, kTx, kNumRx, kDoppler,
                                      kRange),
      shape, waveform);
  expect_tx_lanes_aligned<float>(out, kTx, kNumRx, kDoppler, kRange);
}

TEST(MIMOCoherentCompensationOp, TdmCorrectionTableIsUnitMagnitude) {
  const auto table = holoscan::radar::detail::build_tdm_correction_table(
      make_test_waveform(), kNumTx, kDoppler);
  for (const auto& factor : table) {
    EXPECT_NEAR(std::abs(factor), 1.0, 1e-12);
  }
}

TEST(MIMOCoherentCompensationOp, TdmCorrectionTableTxZeroIsIdentity) {
  const auto table = holoscan::radar::detail::build_tdm_correction_table(
      make_test_waveform(), kNumTx, kDoppler);
  for (int64_t d = 0; d < kDoppler; ++d) {
    const auto& factor = table[static_cast<size_t>(d)];
    EXPECT_NEAR(factor.real(), 1.0, 1e-12) << "doppler bin " << d;
    EXPECT_NEAR(factor.imag(), 0.0, 1e-12) << "doppler bin " << d;
  }
}

TEST(MIMOCoherentCompensationOp, ComplexFloatDtypePreserved) {
  const auto waveform = make_test_waveform();
  const std::vector<int64_t> shape{kNumTx, kNumRx, kDoppler, kRange};
  const auto host = build_misaligned_rd_host<float>(
      waveform, kNumTx, kNumRx, kDoppler, kRange);
  const auto out = run_compensation<float>(host, shape, waveform);
  EXPECT_TRUE(dldatatype_equals(out.dtype, complex_dtype<float>()));
  expect_tx_lanes_aligned<float>(out, kNumTx, kNumRx, kDoppler, kRange);
}

TEST(MIMOCoherentCompensationOp, ComplexDoublePreservesValues) {
  const auto waveform = make_test_waveform();
  const std::vector<int64_t> shape{kNumTx, kNumRx, kDoppler, kRange};
  const auto host = build_misaligned_rd_host<double>(
      waveform, kNumTx, kNumRx, kDoppler, kRange);
  const auto out = run_compensation<double>(host, shape, waveform);
  EXPECT_TRUE(dldatatype_equals(out.dtype, complex_dtype<double>()));
  expect_tx_lanes_aligned<double>(out, kNumTx, kNumRx, kDoppler, kRange);
}

TEST(MIMOCoherentCompensationOp, OutputEmittedOnDevice) {
  const auto waveform = make_test_waveform();
  const std::vector<int64_t> shape{kNumTx, kNumRx, kDoppler, kRange};
  const auto host = build_misaligned_rd_host<float>(
      waveform, kNumTx, kNumRx, kDoppler, kRange);
  OutputTensorCapture capture;
  (void)run_harness_capture_device<float>(
      single_map(to_device_tensor(host, shape, complex_dtype<float>()),
                 "rd_maps"),
      waveform, &capture);
  ASSERT_EQ(capture.devices.size(), 1U);
  EXPECT_EQ(capture.devices[0].device_type, kDLCUDA);
  EXPECT_TRUE(
      dldatatype_equals(capture.dtypes[0], complex_dtype<float>()));
}

TEST(MIMOCoherentCompensationOp, DeterministicAcrossRuns) {
  const auto waveform = make_test_waveform();
  const std::vector<int64_t> shape{kNumTx, kNumRx, kDoppler, kRange};
  const auto host = build_misaligned_rd_host<float>(
      waveform, kNumTx, kNumRx, kDoppler, kRange);
  const auto first = run_compensation<float>(host, shape, waveform);
  const auto second = run_compensation<float>(host, shape, waveform);
  ASSERT_EQ(first.shape, second.shape);
  const auto* a = first.as<std::complex<float>>();
  const auto* b = second.as<std::complex<float>>();
  for (size_t i = 0; i < host.size(); ++i) {
    EXPECT_NEAR(a[i].real(), b[i].real(), 1e-6f);
    EXPECT_NEAR(a[i].imag(), b[i].imag(), 1e-6f);
  }
}

TEST(MIMOCoherentCompensationOp, AcceptsSingleEntryWithArbitraryKey) {
  const auto waveform = make_test_waveform();
  const std::vector<int64_t> shape{kNumTx, kNumRx, kDoppler, kRange};
  const auto host = build_misaligned_rd_host<float>(
      waveform, kNumTx, kNumRx, kDoppler, kRange);
  const auto out = run_harness<float>(
      single_map(to_device_tensor(host, shape, complex_dtype<float>()),
                 "payload"),
      waveform);
  expect_tx_lanes_aligned<float>(out, kNumTx, kNumRx, kDoppler, kRange);
}

TEST(MIMOCoherentCompensationOp, AcceptsPinnedHostInput) {
  const auto waveform = make_test_waveform();
  const std::vector<int64_t> shape{kNumTx, kNumRx, kDoppler, kRange};
  const auto host = build_misaligned_rd_host<float>(
      waveform, kNumTx, kNumRx, kDoppler, kRange);
  const auto tensor = cuda_accessible::make_tensor(
      host, shape, complex_dtype<float>(), kDLCUDAHost);
  const auto out = run_harness<float>(single_map(tensor, "rd_maps"), waveform);
  expect_tx_lanes_aligned<float>(out, kNumTx, kNumRx, kDoppler, kRange);
}

TEST(MIMOCoherentCompensationOp, AcceptsManagedInput) {
  const auto waveform = make_test_waveform();
  const std::vector<int64_t> shape{kNumTx, kNumRx, kDoppler, kRange};
  const auto host = build_misaligned_rd_host<float>(
      waveform, kNumTx, kNumRx, kDoppler, kRange);
  const auto tensor = cuda_accessible::make_tensor(
      host, shape, complex_dtype<float>(), kDLCUDAManaged);
  const auto out = run_harness<float>(single_map(tensor, "rd_maps"), waveform);
  expect_tx_lanes_aligned<float>(out, kNumTx, kNumRx, kDoppler, kRange);
}

TEST(MIMOCoherentCompensationOp, SingleTxIsIdentity) {
  const auto waveform = make_test_waveform();
  const std::vector<int64_t> shape{1, kNumRx, kDoppler, kRange};
  const auto host =
      build_misaligned_rd_host<float>(waveform, 1, kNumRx, kDoppler, kRange);
  const auto out = run_compensation<float>(host, shape, waveform);
  const auto* got = out.as<std::complex<float>>();
  for (size_t i = 0; i < host.size(); ++i) {
    EXPECT_NEAR(got[i].real(), host[i].real(), 1e-5f);
    EXPECT_NEAR(got[i].imag(), host[i].imag(), 1e-5f);
  }
}

TEST(MIMOCoherentCompensationOp, LeadingBatchAxesPreservedAndIndependent) {
  const auto waveform = make_test_waveform();
  const std::vector<int64_t> shape{2, kNumTx, kNumRx, kDoppler, kRange};
  const auto host = build_misaligned_rd_host<float>(
      waveform, kNumTx, kNumRx, kDoppler, kRange, 2);
  const auto out = run_compensation<float>(host, shape, waveform);
  ASSERT_EQ(out.shape.size(), shape.size());
  for (size_t i = 0; i < shape.size(); ++i) {
    EXPECT_EQ(out.shape[i], shape[i]);
  }
  expect_tx_lanes_aligned<float>(out, kNumTx, kNumRx, kDoppler, kRange, 2);
}

TEST(MIMOCoherentCompensationOp, LiveWaveformOverrideSticky) {
  const auto waveform_a = make_test_waveform(kPri);
  const auto waveform_b = make_test_waveform(kPri * 1.1);
  const auto waveform_c = make_test_waveform(kPri * 1.2);
  const std::vector<int64_t> shape{kNumTx, kNumRx, kDoppler, kRange};
  const auto host = build_misaligned_rd_host<float>(
      waveform_a, kNumTx, kNumRx, kDoppler, kRange);
  auto frame = [&]() {
    return single_map(to_device_tensor(host, shape, complex_dtype<float>()),
                      "rd_maps");
  };
  TensorMapCapture capture;
  const auto waveform_a_copy = waveform_a;
  class WaveformApp : public holoscan::Application {
   public:
    FMCWWaveformDescriptor build_waveform;
    std::vector<holoscan::TensorMap> frames;
    std::vector<FMCWWaveformDescriptor> waveform_overrides;
    TensorMapCapture* rd_capture{nullptr};

    void compose() override {
      using holoscan::Arg;
      auto op = make_operator<MIMOCoherentCompensationOp<float>>(
          "compensation", Arg("multiplexing_scheme") = MultiplexingScheme::kTdm,
          Arg("waveform") = build_waveform);
      wire_data_source(this, op, "rd_maps", frames);
      wire_leading_override<FMCWWaveformDescriptor>(
          this, op, "waveform", waveform_overrides);
      wire_capture_sink(this, op, "rd_maps", rd_capture);
    }
  };

  WaveformApp app;
  app.build_waveform = waveform_a_copy;
  app.frames = {frame(), frame(), frame()};
  app.waveform_overrides = {waveform_b, waveform_c};
  app.rd_capture = &capture;
  app.run();

  ASSERT_EQ(capture.frames.size(), 3U);
  // TDM inter-TX phase uses f_d * (PRI/num_tx) with f_d ∝ PRF = 1/PRI, so a
  // PRI-only waveform change does not alter the correction coefficients. This
  // test still exercises the [live] port: leading overrides map to the first
  // ticks and the last override remains sticky on trailing ticks.
  const auto expect_b =
      run_compensation<float>(host, shape, waveform_b);
  const auto expect_c =
      run_compensation<float>(host, shape, waveform_c);
  const auto* exp_b = expect_b.as<std::complex<float>>();
  const auto* exp_c = expect_c.as<std::complex<float>>();
  const auto* tick0 = capture.frames[0].as<std::complex<float>>();
  const auto* tick1 = capture.frames[1].as<std::complex<float>>();
  const auto* tick2 = capture.frames[2].as<std::complex<float>>();
  for (size_t i = 0; i < host.size(); ++i) {
    EXPECT_NEAR(tick0[i].real(), exp_b[i].real(), 1e-4f);
    EXPECT_NEAR(tick0[i].imag(), exp_b[i].imag(), 1e-4f);
    EXPECT_NEAR(tick1[i].real(), exp_c[i].real(), 1e-4f);
    EXPECT_NEAR(tick1[i].imag(), exp_c[i].imag(), 1e-4f);
    EXPECT_NEAR(tick2[i].real(), tick1[i].real(), 1e-5f);
    EXPECT_NEAR(tick2[i].imag(), tick1[i].imag(), 1e-5f);
  }
}

TEST(MIMOCoherentCompensationOp, FloatDoubleAgreement) {
  const auto waveform = make_test_waveform();
  const std::vector<int64_t> shape{kNumTx, kNumRx, kDoppler, kRange};
  const auto host_f = build_misaligned_rd_host<float>(
      waveform, kNumTx, kNumRx, kDoppler, kRange);
  const auto host_d = build_misaligned_rd_host<double>(
      waveform, kNumTx, kNumRx, kDoppler, kRange);
  const auto out_f = run_compensation<float>(host_f, shape, waveform);
  const auto out_d = run_compensation<double>(host_d, shape, waveform);
  const auto* f = out_f.as<std::complex<float>>();
  const auto* d = out_d.as<std::complex<double>>();
  for (size_t i = 0; i < host_f.size(); ++i) {
    EXPECT_NEAR(static_cast<double>(f[i].real()), d[i].real(), 1e-4);
    EXPECT_NEAR(static_cast<double>(f[i].imag()), d[i].imag(), 1e-4);
  }
}

TEST(MIMOCoherentCompensationOp, RejectsEmptyDdmCode) {
  // DDM with a default-constructed (num_tx = 0) ddm_code is invalid.
  const auto waveform = make_test_waveform();
  const std::vector<int64_t> shape{kDdmRx, kDdmDoppler, kDdmRange};
  const std::vector<std::complex<float>> host(
      static_cast<size_t>(kDdmRx * kDdmDoppler * kDdmRange), {1.0f, 0.0f});
  auto harness = create_operator_test<MIMOCoherentCompensationOp<float>>(
      holoscan::Arg("multiplexing_scheme") = MultiplexingScheme::kDdm,
      holoscan::Arg("waveform") = waveform,
      holoscan::Arg("ddm_code") = DDMCodeDescriptor{});
  harness->template add_input_port<holoscan::TensorMap>(
      "rd_maps",
      {single_map(to_device_tensor(host, shape, complex_dtype<float>()),
                  "rd_maps")});
  harness->template add_output_port<holoscan::TensorMap>("rd_maps");
  EXPECT_THROW(harness->run_test(), std::invalid_argument);
}

TEST(MIMOCoherentCompensationOp, RejectsUnspecifiedMultiplexingScheme) {
  const auto waveform = make_test_waveform();
  const std::vector<int64_t> shape{kNumTx, kNumRx, kDoppler, kRange};
  const std::vector<std::complex<float>> host(
      static_cast<size_t>(tensor_count(shape)), {1.0f, 0.0f});
  auto harness = create_operator_test<MIMOCoherentCompensationOp<float>>(
      holoscan::Arg("waveform") = waveform);
  harness->template add_input_port<holoscan::TensorMap>(
      "rd_maps",
      {single_map(to_device_tensor(host, shape, complex_dtype<float>()),
                  "rd_maps")});
  harness->template add_output_port<holoscan::TensorMap>("rd_maps");
  EXPECT_THROW(harness->run_test(), std::invalid_argument);
}

TEST(MIMOCoherentCompensationOp, RejectsInvalidWaveform) {
  const std::vector<int64_t> shape{kNumTx, kNumRx, kDoppler, kRange};
  const std::vector<std::complex<float>> host(
      static_cast<size_t>(tensor_count(shape)), {1.0f, 0.0f});
  auto harness = create_operator_test<MIMOCoherentCompensationOp<float>>(
      holoscan::Arg("multiplexing_scheme") = MultiplexingScheme::kTdm,
      holoscan::Arg("waveform") = FMCWWaveformDescriptor{});
  harness->template add_input_port<holoscan::TensorMap>(
      "rd_maps",
      {single_map(to_device_tensor(host, shape, complex_dtype<float>()),
                  "rd_maps")});
  harness->template add_output_port<holoscan::TensorMap>("rd_maps");
  EXPECT_THROW(harness->run_test(), std::invalid_argument);
}

TEST(MIMOCoherentCompensationOp, RejectsRealInput) {
  const auto waveform = make_test_waveform();
  const std::vector<int64_t> shape{kNumTx, kNumRx, kDoppler, kRange};
  const std::vector<float> host(static_cast<size_t>(tensor_count(shape)), 1.0f);
  auto harness = create_operator_test<MIMOCoherentCompensationOp<float>>(
      holoscan::Arg("multiplexing_scheme") = MultiplexingScheme::kTdm,
      holoscan::Arg("waveform") = waveform);
  harness->template add_input_port<holoscan::TensorMap>(
      "rd_maps", {single_map(to_device_tensor(host, shape, {kDLFloat, 32, 1}),
                             "rd_maps")});
  harness->template add_output_port<holoscan::TensorMap>("rd_maps");
  EXPECT_THROW(harness->run_test(), std::invalid_argument);
}

TEST(MIMOCoherentCompensationOp, RejectsRankTooSmall) {
  const auto waveform = make_test_waveform();
  const std::vector<std::complex<float>> host(8, {1.0f, 0.0f});
  auto harness = create_operator_test<MIMOCoherentCompensationOp<float>>(
      holoscan::Arg("multiplexing_scheme") = MultiplexingScheme::kTdm,
      holoscan::Arg("waveform") = waveform);
  harness->template add_input_port<holoscan::TensorMap>(
      "rd_maps",
      {single_map(to_device_tensor(host, {2, 2, 2}, complex_dtype<float>()),
                  "rd_maps")});
  harness->template add_output_port<holoscan::TensorMap>("rd_maps");
  EXPECT_THROW(harness->run_test(), std::invalid_argument);
}

TEST(MIMOCoherentCompensationOp, RejectsMismatchedPrecision) {
  const auto waveform = make_test_waveform();
  const std::vector<int64_t> shape{kNumTx, kNumRx, kDoppler, kRange};
  const auto host = build_misaligned_rd_host<double>(
      waveform, kNumTx, kNumRx, kDoppler, kRange);
  auto harness = create_operator_test<MIMOCoherentCompensationOp<float>>(
      holoscan::Arg("multiplexing_scheme") = MultiplexingScheme::kTdm,
      holoscan::Arg("waveform") = waveform);
  harness->template add_input_port<holoscan::TensorMap>(
      "rd_maps",
      {single_map(to_device_tensor(host, shape, complex_dtype<double>()),
                  "rd_maps")});
  harness->template add_output_port<holoscan::TensorMap>("rd_maps");
  EXPECT_THROW(harness->run_test(), std::invalid_argument);
}

TEST(MIMOCoherentCompensationOp, RejectsZeroTrailingAxis) {
  const auto waveform = make_test_waveform();
  const std::vector<int64_t> shape{0, kNumRx, kDoppler, kRange};
  const std::vector<std::complex<float>> host;
  auto harness = create_operator_test<MIMOCoherentCompensationOp<float>>(
      holoscan::Arg("multiplexing_scheme") = MultiplexingScheme::kTdm,
      holoscan::Arg("waveform") = waveform);
  harness->template add_input_port<holoscan::TensorMap>(
      "rd_maps",
      {single_map(to_device_tensor(host, shape, complex_dtype<float>()),
                  "rd_maps")});
  harness->template add_output_port<holoscan::TensorMap>("rd_maps");
  EXPECT_THROW(harness->run_test(), std::invalid_argument);
}

TEST(MIMOCoherentCompensationOp, RejectsMultiEntryTensorMap) {
  const auto waveform = make_test_waveform();
  const std::vector<int64_t> shape{kNumTx, kNumRx, kDoppler, kRange};
  const auto host = build_misaligned_rd_host<float>(
      waveform, kNumTx, kNumRx, kDoppler, kRange);
  holoscan::TensorMap map;
  map.insert({"a", to_device_tensor(host, shape, complex_dtype<float>())});
  map.insert({"b", to_device_tensor(host, shape, complex_dtype<float>())});
  EXPECT_THROW((void)run_harness<float>(map, waveform), std::invalid_argument);
}

TEST(MIMOCoherentCompensationOp, RejectsNonContiguousInput) {
  const auto waveform = make_test_waveform();
  const std::vector<int64_t> shape{kNumTx, kNumRx, kDoppler, kRange};
  const auto host = build_misaligned_rd_host<float>(
      waveform, kNumTx, kNumRx, kDoppler, kRange);
  holoscan::TensorMap map;
  map.insert({"rd_maps", make_non_contiguous_rd_maps(host, shape)});
  EXPECT_THROW((void)run_harness<float>(map, waveform), std::invalid_argument);
}

TEST(MIMOCoherentCompensationOp, RejectsPageableHostInput) {
  const auto waveform = make_test_waveform();
  const std::vector<int64_t> shape{kNumTx, kNumRx, kDoppler, kRange};
  const std::vector<std::complex<float>> host(
      static_cast<size_t>(tensor_count(shape)), {1.0f, 0.0f});
  auto harness = create_operator_test<MIMOCoherentCompensationOp<float>>(
      holoscan::Arg("multiplexing_scheme") = MultiplexingScheme::kTdm,
      holoscan::Arg("waveform") = waveform);
  harness->template add_input_port<holoscan::TensorMap>(
      "rd_maps",
      {single_map(
          to_pageable_host_tensor(host, shape, complex_dtype<float>()),
          "rd_maps")});
  harness->template add_output_port<holoscan::TensorMap>("rd_maps");
  EXPECT_THROW(harness->run_test(), std::invalid_argument);
}

// ---- DDM slice ------------------------------------------------------------

TEST(MIMOCoherentCompensationOp, DdmDemuxRecoversEachTxIntoBand) {
  const auto waveform = make_test_waveform();
  const auto code = make_offset_code(kDdmTx);
  const int64_t band = kDdmDoppler / kDdmTx;  // W = 4, guard = 0
  const int64_t p0 = 1;
  const auto offsets = uniform_offsets(kDdmTx);
  const auto host = build_ddm_input_host<float>(
      offsets, kDdmTx, kDdmRx, kDdmDoppler, kDdmRange, p0);
  const std::vector<int64_t> shape{kDdmRx, kDdmDoppler, kDdmRange};
  const auto out = run_ddm_from_host<float>(host, shape, code, waveform);
  ASSERT_EQ(out.shape.size(), 4U);
  EXPECT_EQ(out.shape[0], kDdmTx);
  EXPECT_EQ(out.shape[1], kDdmRx);
  EXPECT_EQ(out.shape[2], band);
  EXPECT_EQ(out.shape[3], kDdmRange);
  expect_ddm_recovered<float>(out, kDdmTx, kDdmRx, band, kDdmRange, p0);
}

TEST(MIMOCoherentCompensationOp, DdmSingleTxIsIdentity) {
  const auto waveform = make_test_waveform();
  const auto code = make_offset_code(1);
  const int64_t p0 = 2;
  const auto offsets = uniform_offsets(1);
  const auto host = build_ddm_input_host<float>(offsets, 1, kDdmRx,
                                                kDdmDoppler, kDdmRange, p0);
  const std::vector<int64_t> shape{kDdmRx, kDdmDoppler, kDdmRange};
  const auto out = run_ddm_from_host<float>(host, shape, code, waveform);
  ASSERT_EQ(out.shape.size(), 4U);
  EXPECT_EQ(out.shape[0], 1);
  EXPECT_EQ(out.shape[2], kDdmDoppler);
  // With M = 1 the demux is a pure fftshift round-trip identity: output equals
  // the input reshaped with a leading unit TX axis.
  const auto* got = out.as<std::complex<float>>();
  for (size_t i = 0; i < host.size(); ++i) {
    EXPECT_NEAR(got[i].real(), host[i].real(), 1e-5f) << "i=" << i;
    EXPECT_NEAR(got[i].imag(), host[i].imag(), 1e-5f) << "i=" << i;
  }
}

TEST(MIMOCoherentCompensationOp, DdmLeadingBatchAxesPreserved) {
  const auto waveform = make_test_waveform();
  const auto code = make_offset_code(kDdmTx);
  const int64_t band = kDdmDoppler / kDdmTx;
  const int64_t p0 = 0;
  const int64_t batch = 2;
  const auto offsets = uniform_offsets(kDdmTx);
  const auto host = build_ddm_input_host<float>(
      offsets, kDdmTx, kDdmRx, kDdmDoppler, kDdmRange, p0, batch);
  const std::vector<int64_t> shape{batch, kDdmRx, kDdmDoppler, kDdmRange};
  const auto out = run_ddm_from_host<float>(host, shape, code, waveform);
  ASSERT_EQ(out.shape.size(), 5U);
  EXPECT_EQ(out.shape[0], batch);
  EXPECT_EQ(out.shape[1], kDdmTx);
  EXPECT_EQ(out.shape[2], kDdmRx);
  EXPECT_EQ(out.shape[3], band);
  EXPECT_EQ(out.shape[4], kDdmRange);
  expect_ddm_recovered<float>(out, kDdmTx, kDdmRx, band, kDdmRange, p0, batch);
}

TEST(MIMOCoherentCompensationOp, DdmLinearPhaseCodesMatchOffsets) {
  const auto waveform = make_test_waveform();
  const int64_t p0 = 1;
  const auto offsets = uniform_offsets(kDdmTx);
  const auto host = build_ddm_input_host<float>(
      offsets, kDdmTx, kDdmRx, kDdmDoppler, kDdmRange, p0);
  const std::vector<int64_t> shape{kDdmRx, kDdmDoppler, kDdmRange};

  // Linear-phase codes phi_m[k] = exp(j*2*pi*(m/M)*k) must reduce to the same
  // per-TX offsets and therefore the same demux.
  DDMCodeDescriptor code;
  code.num_tx = kDdmTx;
  code.guard_bands = 0;
  const int64_t num_chirps = kDdmDoppler;
  code.phase_codes.resize(static_cast<size_t>(kDdmTx * num_chirps));
  for (int64_t m = 0; m < kDdmTx; ++m) {
    for (int64_t k = 0; k < num_chirps; ++k) {
      const double phase = 2.0 * 3.141592653589793 *
                           (static_cast<double>(m) /
                            static_cast<double>(kDdmTx)) *
                           static_cast<double>(k);
      code.phase_codes[static_cast<size_t>(m * num_chirps + k)] =
          std::complex<double>(std::cos(phase), std::sin(phase));
    }
  }

  const int64_t band = kDdmDoppler / kDdmTx;
  const auto out_codes = run_ddm_from_host<float>(host, shape, code, waveform);
  const auto out_offsets =
      run_ddm_from_host<float>(host, shape, make_offset_code(kDdmTx), waveform);
  ASSERT_EQ(out_codes.shape.size(), out_offsets.shape.size());
  const auto* a = out_codes.as<std::complex<float>>();
  const auto* b = out_offsets.as<std::complex<float>>();
  const int64_t total = kDdmTx * kDdmRx * band * kDdmRange;
  for (int64_t i = 0; i < total; ++i) {
    EXPECT_NEAR(a[i].real(), b[i].real(), 1e-5f) << "i=" << i;
    EXPECT_NEAR(a[i].imag(), b[i].imag(), 1e-5f) << "i=" << i;
  }
}

TEST(MIMOCoherentCompensationOp, DdmNonLinearPhaseCodesCorrelateRecovers) {
  // Single-TX forward model in fft-order: Y = val * circshift(C, d0). The
  // operator expects center-zero Doppler (ifftshift on the way in), so the
  // peak appears at the fftshifted bin of d0. Correlate demux yields
  // amplitude val * (||C||^2 / D) = val * D when |c[n]| = 1 and
  // num_chirps = D (Parseval). Multi-TX non-orthogonal codes leave crosstalk,
  // so this test uses M_tx = 1 for amplitude recovery.
  const auto waveform = make_test_waveform();
  constexpr int64_t kNlTx = 1;
  constexpr int64_t kNlRx = 2;
  constexpr int64_t kNlDoppler = 16;
  constexpr int64_t kNlRange = 3;
  const int64_t num_chirps = kNlDoppler;
  constexpr int64_t d0 = 3;
  constexpr double kPi2 = 2.0 * 3.141592653589793;
  const int64_t D = kNlDoppler;
  const int64_t d0_cz = (d0 + D / 2) % D;
  const std::complex<double> val(1.5, -0.3);

  DDMCodeDescriptor code;
  code.num_tx = kNlTx;
  code.guard_bands = 0;
  code.phase_codes.resize(static_cast<size_t>(kNlTx * num_chirps));
  for (int64_t k = 0; k < num_chirps; ++k) {
    const double phase = std::sin(0.7 * static_cast<double>(k)) * 2.0;
    code.phase_codes[static_cast<size_t>(k)] =
        std::complex<double>(std::cos(phase), std::sin(phase));
  }

  std::vector<std::complex<double>> spectrum(static_cast<size_t>(D));
  const auto* c_m = code.phase_codes.data();
  for (int64_t k = 0; k < D; ++k) {
    std::complex<double> sum(0.0, 0.0);
    for (int64_t n = 0; n < num_chirps; ++n) {
      const double angle = -kPi2 * static_cast<double>(n) *
                           static_cast<double>(k) / static_cast<double>(D);
      sum += c_m[n] * std::complex<double>(std::cos(angle), std::sin(angle));
    }
    spectrum[static_cast<size_t>(k)] = sum;
  }

  std::vector<std::complex<float>> host(
      static_cast<size_t>(kNlRx * D * kNlRange), {0.0f, 0.0f});
  for (int64_t rx = 0; rx < kNlRx; ++rx) {
    for (int64_t r = 0; r < kNlRange; ++r) {
      for (int64_t k = 0; k < D; ++k) {
        // fft-order sample, stored at center-zero index (fftshift).
        const int64_t idx = ((k - d0) % D + D) % D;
        const auto y = val * spectrum[static_cast<size_t>(idx)];
        const int64_t d_cz = (k + D / 2) % D;
        host[static_cast<size_t>((rx * D + d_cz) * kNlRange + r)] =
            std::complex<float>(static_cast<float>(y.real()),
                                static_cast<float>(y.imag()));
      }
    }
  }

  const std::vector<int64_t> shape{kNlRx, kNlDoppler, kNlRange};
  const auto out = run_ddm_from_host<float>(host, shape, code, waveform);
  ASSERT_EQ(out.shape.size(), 4U);
  EXPECT_EQ(out.shape[0], kNlTx);
  EXPECT_EQ(out.shape[1], kNlRx);
  EXPECT_EQ(out.shape[2], kNlDoppler);
  EXPECT_EQ(out.shape[3], kNlRange);

  const auto* data = out.as<std::complex<float>>();
  const std::complex<float> expected(
      static_cast<float>(val.real() * static_cast<double>(D)),
      static_cast<float>(val.imag() * static_cast<double>(D)));
  const float tol = 0.05f * std::abs(expected);
  for (int64_t rx = 0; rx < kNlRx; ++rx) {
    for (int64_t r = 0; r < kNlRange; ++r) {
      float peak = 0.0f;
      float other = 0.0f;
      std::complex<float> at_d0{0.0f, 0.0f};
      for (int64_t d = 0; d < D; ++d) {
        const auto z =
            data[static_cast<size_t>(((0 * kNlRx + rx) * D + d) * kNlRange + r)];
        const float mag = std::abs(z);
        if (d == d0_cz) {
          peak = mag;
          at_d0 = z;
        } else {
          other = std::max(other, mag);
        }
      }
      EXPECT_NEAR(at_d0.real(), expected.real(), tol) << "rx=" << rx;
      EXPECT_NEAR(at_d0.imag(), expected.imag(), tol) << "rx=" << rx;
      EXPECT_GT(peak, 5.0f * other) << "rx=" << rx;
    }
  }
}

TEST(MIMOCoherentCompensationOp, DdmNonLinearRejectsNonZeroGuardBands) {
  const auto waveform = make_test_waveform();
  constexpr int64_t kNlTx = 2;
  constexpr int64_t kNlDoppler = 16;
  constexpr int64_t kNlRx = 2;
  constexpr int64_t kNlRange = 3;
  const int64_t num_chirps = kNlDoppler;

  DDMCodeDescriptor code;
  code.num_tx = kNlTx;
  code.guard_bands = 1;  // non-zero → must reject
  code.phase_codes.resize(static_cast<size_t>(kNlTx * num_chirps));
  for (int64_t m = 0; m < kNlTx; ++m) {
    for (int64_t k = 0; k < num_chirps; ++k) {
      const double phase = std::sin(0.7 * static_cast<double>(k) + m) * 2.0;
      code.phase_codes[static_cast<size_t>(m * num_chirps + k)] =
          std::complex<double>(std::cos(phase), std::sin(phase));
    }
  }

  const std::vector<int64_t> shape{kNlRx, kNlDoppler, kNlRange};
  const std::vector<std::complex<float>> host(
      static_cast<size_t>(kNlRx * kNlDoppler * kNlRange), {1.0f, 0.0f});
  auto harness = create_operator_test<MIMOCoherentCompensationOp<float>>(
      holoscan::Arg("multiplexing_scheme") = MultiplexingScheme::kDdm,
      holoscan::Arg("waveform") = waveform, holoscan::Arg("ddm_code") = code);
  harness->template add_input_port<holoscan::TensorMap>(
      "rd_maps",
      {single_map(to_device_tensor(host, shape, complex_dtype<float>()),
                  "rd_maps")});
  harness->template add_output_port<holoscan::TensorMap>("rd_maps");
  EXPECT_THROW(harness->run_test(), std::invalid_argument);
}

TEST(MIMOCoherentCompensationOp, DdmNonLinearRejectsChirpDopplerMismatch) {
  const auto waveform = make_test_waveform();
  constexpr int64_t kNlTx = 2;
  constexpr int64_t kNlDoppler = 16;
  constexpr int64_t kNlRx = 2;
  constexpr int64_t kNlRange = 3;
  // num_chirps != num_doppler (use 8 chirps with 16 Doppler bins)
  const int64_t num_chirps = 8;

  DDMCodeDescriptor code;
  code.num_tx = kNlTx;
  code.guard_bands = 0;
  code.phase_codes.resize(static_cast<size_t>(kNlTx * num_chirps));
  for (int64_t m = 0; m < kNlTx; ++m) {
    for (int64_t k = 0; k < num_chirps; ++k) {
      const double phase = std::sin(0.7 * static_cast<double>(k) + m) * 2.0;
      code.phase_codes[static_cast<size_t>(m * num_chirps + k)] =
          std::complex<double>(std::cos(phase), std::sin(phase));
    }
  }

  const std::vector<int64_t> shape{kNlRx, kNlDoppler, kNlRange};
  const std::vector<std::complex<float>> host(
      static_cast<size_t>(kNlRx * kNlDoppler * kNlRange), {1.0f, 0.0f});
  auto harness = create_operator_test<MIMOCoherentCompensationOp<float>>(
      holoscan::Arg("multiplexing_scheme") = MultiplexingScheme::kDdm,
      holoscan::Arg("waveform") = waveform, holoscan::Arg("ddm_code") = code);
  harness->template add_input_port<holoscan::TensorMap>(
      "rd_maps",
      {single_map(to_device_tensor(host, shape, complex_dtype<float>()),
                  "rd_maps")});
  harness->template add_output_port<holoscan::TensorMap>("rd_maps");
  EXPECT_THROW(harness->run_test(), std::invalid_argument);
}

TEST(MIMOCoherentCompensationOp, DdmGuardBandsReduceWidth) {
  const auto waveform = make_test_waveform();
  const auto code = make_offset_code(kDdmTx, /*guard=*/1);
  const int64_t band = kDdmDoppler / kDdmTx - 1;  // W = 3
  const int64_t p0 = 0;
  const auto offsets = uniform_offsets(kDdmTx);
  const auto host = build_ddm_input_host<float>(
      offsets, kDdmTx, kDdmRx, kDdmDoppler, kDdmRange, p0);
  const std::vector<int64_t> shape{kDdmRx, kDdmDoppler, kDdmRange};
  const auto out = run_ddm_from_host<float>(host, shape, code, waveform);
  ASSERT_EQ(out.shape.size(), 4U);
  EXPECT_EQ(out.shape[2], band);
  expect_ddm_recovered<float>(out, kDdmTx, kDdmRx, band, kDdmRange, p0);
}

TEST(MIMOCoherentCompensationOp, DdmGuardBandDiscardsEdgeBin) {
  // A target parked in the discarded top fft-order bin of each band (band - 1)
  // must not appear in the guarded output.
  const auto waveform = make_test_waveform();
  const auto code = make_offset_code(kDdmTx, /*guard=*/1);
  const int64_t band = kDdmDoppler / kDdmTx - 1;  // W = 3
  const int64_t edge = kDdmDoppler / kDdmTx - 1;  // fft bin 3 (guarded)
  const auto offsets = uniform_offsets(kDdmTx);
  const auto host = build_ddm_input_host<float>(
      offsets, kDdmTx, kDdmRx, kDdmDoppler, kDdmRange, edge);
  const std::vector<int64_t> shape{kDdmRx, kDdmDoppler, kDdmRange};
  const auto out = run_ddm_from_host<float>(host, shape, code, waveform);
  const auto* data = out.as<std::complex<float>>();
  const int64_t total = kDdmTx * kDdmRx * band * kDdmRange;
  for (int64_t i = 0; i < total; ++i) {
    EXPECT_NEAR(data[i].real(), 0.0f, 1e-5f) << "i=" << i;
    EXPECT_NEAR(data[i].imag(), 0.0f, 1e-5f) << "i=" << i;
  }
}

TEST(MIMOCoherentCompensationOp, DdmComplexDoublePreserved) {
  const auto waveform = make_test_waveform();
  const auto code = make_offset_code(kDdmTx);
  const int64_t band = kDdmDoppler / kDdmTx;
  const int64_t p0 = 2;
  const auto offsets = uniform_offsets(kDdmTx);
  const auto host = build_ddm_input_host<double>(
      offsets, kDdmTx, kDdmRx, kDdmDoppler, kDdmRange, p0);
  const std::vector<int64_t> shape{kDdmRx, kDdmDoppler, kDdmRange};
  const auto out = run_ddm_from_host<double>(host, shape, code, waveform);
  EXPECT_TRUE(dldatatype_equals(out.dtype, complex_dtype<double>()));
  expect_ddm_recovered<double>(out, kDdmTx, kDdmRx, band, kDdmRange, p0, 1,
                                 1e-9);
}

TEST(MIMOCoherentCompensationOp, DdmComplexFloatDtypePreserved) {
  const auto waveform = make_test_waveform();
  const auto code = make_offset_code(kDdmTx);
  const int64_t band = kDdmDoppler / kDdmTx;
  const int64_t p0 = 1;
  const auto offsets = uniform_offsets(kDdmTx);
  const auto host = build_ddm_input_host<float>(
      offsets, kDdmTx, kDdmRx, kDdmDoppler, kDdmRange, p0);
  const std::vector<int64_t> shape{kDdmRx, kDdmDoppler, kDdmRange};
  const auto out = run_ddm_from_host<float>(host, shape, code, waveform);
  EXPECT_TRUE(dldatatype_equals(out.dtype, complex_dtype<float>()));
  expect_ddm_recovered<float>(out, kDdmTx, kDdmRx, band, kDdmRange, p0);
}

TEST(MIMOCoherentCompensationOp, DdmFloatDoubleAgreement) {
  const auto waveform = make_test_waveform();
  const auto code = make_offset_code(kDdmTx);
  const int64_t band = kDdmDoppler / kDdmTx;
  const int64_t p0 = 1;
  const auto offsets = uniform_offsets(kDdmTx);
  const std::vector<int64_t> shape{kDdmRx, kDdmDoppler, kDdmRange};
  const auto host_f = build_ddm_input_host<float>(
      offsets, kDdmTx, kDdmRx, kDdmDoppler, kDdmRange, p0);
  const auto host_d = build_ddm_input_host<double>(
      offsets, kDdmTx, kDdmRx, kDdmDoppler, kDdmRange, p0);
  const auto out_f = run_ddm_from_host<float>(host_f, shape, code, waveform);
  const auto out_d = run_ddm_from_host<double>(host_d, shape, code, waveform);
  const auto* f = out_f.as<std::complex<float>>();
  const auto* d = out_d.as<std::complex<double>>();
  const int64_t total = kDdmTx * kDdmRx * band * kDdmRange;
  for (int64_t i = 0; i < total; ++i) {
    EXPECT_NEAR(static_cast<double>(f[i].real()), d[i].real(), 1e-4)
        << "i=" << i;
    EXPECT_NEAR(static_cast<double>(f[i].imag()), d[i].imag(), 1e-4)
        << "i=" << i;
  }
}

TEST(MIMOCoherentCompensationOp, DdmNonUniformOffsetPermutation) {
  // Offsets are valid band placements but permuted across the TX axis (not
  // sorted m/M); demux must follow ddm_code TX order, not sorted offsets.
  const auto waveform = make_test_waveform();
  const std::vector<double> permuted{0.5, 0.0, 0.75, 0.25};
  const auto code = make_offset_code_from(permuted);
  const int64_t band = kDdmDoppler / kDdmTx;
  const int64_t p0 = 0;
  const auto host = build_ddm_input_host<float>(
      permuted, kDdmTx, kDdmRx, kDdmDoppler, kDdmRange, p0);
  const std::vector<int64_t> shape{kDdmRx, kDdmDoppler, kDdmRange};
  const auto out = run_ddm_from_host<float>(host, shape, code, waveform);
  expect_ddm_recovered<float>(out, kDdmTx, kDdmRx, band, kDdmRange, p0);
}

TEST(MIMOCoherentCompensationOp, DdmOutputEmittedOnDevice) {
  const auto waveform = make_test_waveform();
  const auto code = make_offset_code(kDdmTx);
  const int64_t p0 = 1;
  const auto offsets = uniform_offsets(kDdmTx);
  const auto host = build_ddm_input_host<float>(
      offsets, kDdmTx, kDdmRx, kDdmDoppler, kDdmRange, p0);
  const std::vector<int64_t> shape{kDdmRx, kDdmDoppler, kDdmRange};
  OutputTensorCapture capture;
  (void)run_ddm_capture_device<float>(
      single_map(to_device_tensor(host, shape, complex_dtype<float>()),
                 "rd_maps"),
      code, waveform, &capture);
  ASSERT_EQ(capture.devices.size(), 1U);
  EXPECT_EQ(capture.devices[0].device_type, kDLCUDA);
  EXPECT_TRUE(
      dldatatype_equals(capture.dtypes[0], complex_dtype<float>()));
}

TEST(MIMOCoherentCompensationOp, DdmDeterministicAcrossRuns) {
  const auto waveform = make_test_waveform();
  const auto code = make_offset_code(kDdmTx);
  const int64_t p0 = 1;
  const auto offsets = uniform_offsets(kDdmTx);
  const auto host = build_ddm_input_host<float>(
      offsets, kDdmTx, kDdmRx, kDdmDoppler, kDdmRange, p0);
  const std::vector<int64_t> shape{kDdmRx, kDdmDoppler, kDdmRange};
  const auto first = run_ddm_from_host<float>(host, shape, code, waveform);
  const auto second = run_ddm_from_host<float>(host, shape, code, waveform);
  ASSERT_EQ(first.shape, second.shape);
  const auto* a = first.as<std::complex<float>>();
  const auto* b = second.as<std::complex<float>>();
  const int64_t band = kDdmDoppler / kDdmTx;
  const int64_t total = kDdmTx * kDdmRx * band * kDdmRange;
  for (int64_t i = 0; i < total; ++i) {
    EXPECT_NEAR(a[i].real(), b[i].real(), 1e-6f) << "i=" << i;
    EXPECT_NEAR(a[i].imag(), b[i].imag(), 1e-6f) << "i=" << i;
  }
}

TEST(MIMOCoherentCompensationOp, DdmAcceptsSingleEntryWithArbitraryKey) {
  const auto waveform = make_test_waveform();
  const auto code = make_offset_code(kDdmTx);
  const int64_t band = kDdmDoppler / kDdmTx;
  const int64_t p0 = 1;
  const auto offsets = uniform_offsets(kDdmTx);
  const auto host = build_ddm_input_host<float>(
      offsets, kDdmTx, kDdmRx, kDdmDoppler, kDdmRange, p0);
  const std::vector<int64_t> shape{kDdmRx, kDdmDoppler, kDdmRange};
  const auto out = run_ddm<float>(
      single_map(to_device_tensor(host, shape, complex_dtype<float>()),
                 "payload"),
      code, waveform);
  expect_ddm_recovered<float>(out, kDdmTx, kDdmRx, band, kDdmRange, p0);
}

TEST(MIMOCoherentCompensationOp, DdmAcceptsPinnedHostInput) {
  const auto waveform = make_test_waveform();
  const auto code = make_offset_code(kDdmTx);
  const int64_t band = kDdmDoppler / kDdmTx;
  const int64_t p0 = 0;
  const auto offsets = uniform_offsets(kDdmTx);
  const auto host = build_ddm_input_host<float>(
      offsets, kDdmTx, kDdmRx, kDdmDoppler, kDdmRange, p0);
  const std::vector<int64_t> shape{kDdmRx, kDdmDoppler, kDdmRange};
  const auto tensor = cuda_accessible::make_tensor(
      host, shape, complex_dtype<float>(), kDLCUDAHost);
  const auto out =
      run_ddm<float>(single_map(tensor, "rd_maps"), code, waveform);
  expect_ddm_recovered<float>(out, kDdmTx, kDdmRx, band, kDdmRange, p0);
}

TEST(MIMOCoherentCompensationOp, DdmAcceptsManagedInput) {
  const auto waveform = make_test_waveform();
  const auto code = make_offset_code(kDdmTx);
  const int64_t band = kDdmDoppler / kDdmTx;
  const int64_t p0 = 2;
  const auto offsets = uniform_offsets(kDdmTx);
  const auto host = build_ddm_input_host<float>(
      offsets, kDdmTx, kDdmRx, kDdmDoppler, kDdmRange, p0);
  const std::vector<int64_t> shape{kDdmRx, kDdmDoppler, kDdmRange};
  const auto tensor = cuda_accessible::make_tensor(
      host, shape, complex_dtype<float>(), kDLCUDAManaged);
  const auto out =
      run_ddm<float>(single_map(tensor, "rd_maps"), code, waveform);
  expect_ddm_recovered<float>(out, kDdmTx, kDdmRx, band, kDdmRange, p0);
}

TEST(MIMOCoherentCompensationOp, DdmRejectsPageableHostInput) {
  const auto waveform = make_test_waveform();
  const auto code = make_offset_code(kDdmTx);
  const std::vector<int64_t> shape{kDdmRx, kDdmDoppler, kDdmRange};
  const std::vector<std::complex<float>> host(
      static_cast<size_t>(kDdmRx * kDdmDoppler * kDdmRange), {1.0f, 0.0f});
  auto harness = create_operator_test<MIMOCoherentCompensationOp<float>>(
      holoscan::Arg("multiplexing_scheme") = MultiplexingScheme::kDdm,
      holoscan::Arg("waveform") = waveform, holoscan::Arg("ddm_code") = code);
  harness->template add_input_port<holoscan::TensorMap>(
      "rd_maps",
      {single_map(
          to_pageable_host_tensor(host, shape, complex_dtype<float>()),
          "rd_maps")});
  harness->template add_output_port<holoscan::TensorMap>("rd_maps");
  EXPECT_THROW(harness->run_test(), std::invalid_argument);
}

TEST(MIMOCoherentCompensationOp, DdmRejectsBothOffsetForms) {
  const auto waveform = make_test_waveform();
  const std::vector<int64_t> shape{kDdmRx, kDdmDoppler, kDdmRange};
  const std::vector<std::complex<float>> host(
      static_cast<size_t>(kDdmRx * kDdmDoppler * kDdmRange), {1.0f, 0.0f});
  DDMCodeDescriptor code;
  code.num_tx = 2;
  code.doppler_offsets = {0.0, 0.5};
  code.phase_codes = {{1.0, 0.0}, {0.0, 1.0}};
  auto harness = create_operator_test<MIMOCoherentCompensationOp<float>>(
      holoscan::Arg("multiplexing_scheme") = MultiplexingScheme::kDdm,
      holoscan::Arg("waveform") = waveform, holoscan::Arg("ddm_code") = code);
  harness->template add_input_port<holoscan::TensorMap>(
      "rd_maps",
      {single_map(to_device_tensor(host, shape, complex_dtype<float>()),
                  "rd_maps")});
  harness->template add_output_port<holoscan::TensorMap>("rd_maps");
  EXPECT_THROW(harness->run_test(), std::invalid_argument);
}

TEST(MIMOCoherentCompensationOp, DdmRejectsMismatchedPrecision) {
  const auto waveform = make_test_waveform();
  const auto code = make_offset_code(kDdmTx);
  const std::vector<int64_t> shape{kDdmRx, kDdmDoppler, kDdmRange};
  const auto offsets = uniform_offsets(kDdmTx);
  const auto host = build_ddm_input_host<double>(
      offsets, kDdmTx, kDdmRx, kDdmDoppler, kDdmRange, 0);
  auto harness = create_operator_test<MIMOCoherentCompensationOp<float>>(
      holoscan::Arg("multiplexing_scheme") = MultiplexingScheme::kDdm,
      holoscan::Arg("waveform") = waveform, holoscan::Arg("ddm_code") = code);
  harness->template add_input_port<holoscan::TensorMap>(
      "rd_maps",
      {single_map(to_device_tensor(host, shape, complex_dtype<double>()),
                  "rd_maps")});
  harness->template add_output_port<holoscan::TensorMap>("rd_maps");
  EXPECT_THROW(harness->run_test(), std::invalid_argument);
}

TEST(MIMOCoherentCompensationOp, DdmWaveformOverrideAcceptedButInert) {
  // The linear-phase DDM demux is waveform-independent; the [live] waveform
  // port is still accepted and sticky, but must not change the output.
  const auto waveform_a = make_test_waveform(kPri);
  const auto waveform_b = make_test_waveform(kPri * 1.3);
  const auto code = make_offset_code(kDdmTx);
  const int64_t band = kDdmDoppler / kDdmTx;
  const int64_t p0 = 1;
  const auto offsets = uniform_offsets(kDdmTx);
  const auto host = build_ddm_input_host<float>(
      offsets, kDdmTx, kDdmRx, kDdmDoppler, kDdmRange, p0);
  const std::vector<int64_t> shape{kDdmRx, kDdmDoppler, kDdmRange};

  TensorMapCapture capture;
  class DdmWaveformApp : public holoscan::Application {
   public:
    FMCWWaveformDescriptor build_waveform;
    DDMCodeDescriptor ddm_code;
    std::vector<holoscan::TensorMap> frames;
    std::vector<FMCWWaveformDescriptor> waveform_overrides;
    TensorMapCapture* rd_capture{nullptr};

    void compose() override {
      using holoscan::Arg;
      auto op = make_operator<MIMOCoherentCompensationOp<float>>(
          "compensation", Arg("multiplexing_scheme") = MultiplexingScheme::kDdm,
          Arg("waveform") = build_waveform, Arg("ddm_code") = ddm_code);
      wire_data_source(this, op, "rd_maps", frames);
      wire_leading_override<FMCWWaveformDescriptor>(this, op, "waveform",
                                                    waveform_overrides);
      wire_capture_sink(this, op, "rd_maps", rd_capture);
    }
  };

  auto frame = [&]() {
    return single_map(to_device_tensor(host, shape, complex_dtype<float>()),
                      "rd_maps");
  };
  DdmWaveformApp app;
  app.build_waveform = waveform_a;
  app.ddm_code = code;
  app.frames = {frame(), frame()};
  app.waveform_overrides = {waveform_b};
  app.rd_capture = &capture;
  app.run();

  ASSERT_EQ(capture.frames.size(), 2U);
  const auto* tick0 = capture.frames[0].as<std::complex<float>>();
  const auto* tick1 = capture.frames[1].as<std::complex<float>>();
  const int64_t total = kDdmTx * kDdmRx * band * kDdmRange;
  for (int64_t i = 0; i < total; ++i) {
    EXPECT_NEAR(tick0[i].real(), tick1[i].real(), 1e-6f) << "i=" << i;
    EXPECT_NEAR(tick0[i].imag(), tick1[i].imag(), 1e-6f) << "i=" << i;
  }
  expect_ddm_recovered<float>(capture.frames[1], kDdmTx, kDdmRx, band,
                              kDdmRange, p0);
}

TEST(MIMOCoherentCompensationOp, DdmRejectsNonDivisibleDoppler) {
  const auto waveform = make_test_waveform();
  const auto code = make_offset_code(3);  // 16 not divisible by 3
  const std::vector<int64_t> shape{kDdmRx, kDdmDoppler, kDdmRange};
  const std::vector<std::complex<float>> host(
      static_cast<size_t>(kDdmRx * kDdmDoppler * kDdmRange), {1.0f, 0.0f});
  auto harness = create_operator_test<MIMOCoherentCompensationOp<float>>(
      holoscan::Arg("multiplexing_scheme") = MultiplexingScheme::kDdm,
      holoscan::Arg("waveform") = waveform, holoscan::Arg("ddm_code") = code);
  harness->template add_input_port<holoscan::TensorMap>(
      "rd_maps",
      {single_map(to_device_tensor(host, shape, complex_dtype<float>()),
                  "rd_maps")});
  harness->template add_output_port<holoscan::TensorMap>("rd_maps");
  EXPECT_THROW(harness->run_test(), std::invalid_argument);
}

TEST(MIMOCoherentCompensationOp, DdmRejectsGuardTooLarge) {
  const auto waveform = make_test_waveform();
  const auto code = make_offset_code(kDdmTx, /*guard=*/4);  // band = 4
  const std::vector<int64_t> shape{kDdmRx, kDdmDoppler, kDdmRange};
  const std::vector<std::complex<float>> host(
      static_cast<size_t>(kDdmRx * kDdmDoppler * kDdmRange), {1.0f, 0.0f});
  auto harness = create_operator_test<MIMOCoherentCompensationOp<float>>(
      holoscan::Arg("multiplexing_scheme") = MultiplexingScheme::kDdm,
      holoscan::Arg("waveform") = waveform, holoscan::Arg("ddm_code") = code);
  harness->template add_input_port<holoscan::TensorMap>(
      "rd_maps",
      {single_map(to_device_tensor(host, shape, complex_dtype<float>()),
                  "rd_maps")});
  harness->template add_output_port<holoscan::TensorMap>("rd_maps");
  EXPECT_THROW(harness->run_test(), std::invalid_argument);
}

TEST(MIMOCoherentCompensationOp, DdmRejectsRankTooSmall) {
  const auto waveform = make_test_waveform();
  const auto code = make_offset_code(2);
  const std::vector<std::complex<float>> host(
      static_cast<size_t>(kDdmDoppler * kDdmRange), {1.0f, 0.0f});
  auto harness = create_operator_test<MIMOCoherentCompensationOp<float>>(
      holoscan::Arg("multiplexing_scheme") = MultiplexingScheme::kDdm,
      holoscan::Arg("waveform") = waveform, holoscan::Arg("ddm_code") = code);
  harness->template add_input_port<holoscan::TensorMap>(
      "rd_maps",
      {single_map(to_device_tensor(host, {kDdmDoppler, kDdmRange},
                                   complex_dtype<float>()),
                  "rd_maps")});
  harness->template add_output_port<holoscan::TensorMap>("rd_maps");
  EXPECT_THROW(harness->run_test(), std::invalid_argument);
}

}  // namespace
