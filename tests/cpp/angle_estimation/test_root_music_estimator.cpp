// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// RootMUSICEstimatorOp contract tests (gridless ULA sparse-path DOA).

#include <gtest/gtest.h>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <holoscan/holoscan.hpp>
#include <holoscan/test/test_harness.hpp>

#include "angle_estimation/root_music_estimator_op/root_music_estimator_op.hpp"
#include "radar_common/descriptors/angle_convention.hpp"
#include "radar_common/descriptors/array_geometry.hpp"
#include "radar_common/descriptors/enums.hpp"
#include "radar_common/dsp/angle_convention.hpp"
#include "radar_common/math/constants.hpp"
#include "radar_common/sdk/tensor_io.hpp"
#include "radar_common/tensor/tensor_utils.hpp"
#include "utils/sdk_harness.hpp"
#include "utils/tensor_fixtures.hpp"

namespace {
using holoscan::radar::AngleConvention;
using holoscan::radar::AngleConventionPreset;
using holoscan::radar::ArrayGeometry;
using holoscan::radar::kPi;
using holoscan::radar::ArrayGeometryType;
using holoscan::radar::ElevationReference;
using holoscan::radar::RootMUSICEstimatorOp;
using holoscan::radar::SignedAxis;
using holoscan::radar::complex_dtype;
using holoscan::radar::testing::HostTensor;
using holoscan::radar::testing::single_map;
using holoscan::radar::testing::TensorMapCapture;
using holoscan::radar::testing::to_device_bool_tensor;
using holoscan::radar::testing::to_device_tensor;
using holoscan::radar::testing::to_pageable_host_tensor;
using holoscan::test::create_operator_test;


namespace cuda_accessible {

struct BufferHolder {
  void* data{nullptr};
  std::vector<int64_t> shape;
  DLDevice device{kDLCUDA, 0};
  DLManagedTensor dl{};

  ~BufferHolder() {
    if (device.device_type == kDLCUDAHost) {
      cudaFreeHost(data);
    } else {
      cudaFree(data);
    }
  }
};

template <typename ElementT>
std::shared_ptr<holoscan::Tensor> make_tensor(
    const std::vector<ElementT>& host, const std::vector<int64_t>& shape,
    DLDataType dtype, DLDeviceType device_type) {
  auto holder = std::make_unique<BufferHolder>();
  holder->shape = shape;
  const size_t nbytes = host.size() * sizeof(ElementT);
  int device_id = 0;
  cudaGetDevice(&device_id);
  holder->device = DLDevice{device_type, device_id};
  if (device_type == kDLCUDAManaged) {
    if (cudaMallocManaged(&holder->data, nbytes) != cudaSuccess) {
      throw std::runtime_error("cuda_accessible: cudaMallocManaged failed");
    }
  } else if (device_type == kDLCUDAHost) {
    if (cudaHostAlloc(&holder->data, nbytes, cudaHostAllocMapped) !=
        cudaSuccess) {
      throw std::runtime_error("cuda_accessible: cudaHostAlloc failed");
    }
  } else {
    throw std::invalid_argument("cuda_accessible: unsupported device type");
  }
  std::memcpy(holder->data, host.data(), nbytes);
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

ArrayGeometry make_ula_geometry(int64_t num_elements, double spacing,
                                double wavelength) {
  ArrayGeometry geom;
  geom.wavelength = wavelength;
  geom.geometry_type = ArrayGeometryType::kUla;
  geom.element_positions.reserve(static_cast<size_t>(num_elements * 3));
  for (int64_t m = 0; m < num_elements; ++m) {
    geom.element_positions.push_back(0.0);
    geom.element_positions.push_back(static_cast<double>(m) * spacing);
    geom.element_positions.push_back(0.0);
  }
  return geom;
}

ArrayGeometry make_planar_ura_geometry(int64_t ny, int64_t nz, double spacing,
                                       double wavelength) {
  ArrayGeometry geom;
  geom.wavelength = wavelength;
  geom.geometry_type = ArrayGeometryType::kUra;
  geom.element_positions.reserve(static_cast<size_t>(ny * nz * 3));
  for (int64_t iz = 0; iz < nz; ++iz) {
    for (int64_t iy = 0; iy < ny; ++iy) {
      geom.element_positions.push_back(0.0);
      geom.element_positions.push_back(static_cast<double>(iy) * spacing);
      geom.element_positions.push_back(static_cast<double>(iz) * spacing);
    }
  }
  return geom;
}

std::vector<std::complex<double>> steering_vector(const ArrayGeometry& geom,
                                                  const AngleConvention& conv,
                                                  double az, double el) {
  const auto dir = holoscan::radar::boundary_to_canonical(conv, az, el);
  const double k = 2.0 * kPi / geom.wavelength;
  std::vector<std::complex<double>> a(
      static_cast<size_t>(geom.num_elements()));
  for (int64_t m = 0; m < geom.num_elements(); ++m) {
    const size_t base = static_cast<size_t>(m * 3);
    const double phase = k * (geom.element_positions[base] * dir.x +
                            geom.element_positions[base + 1] * dir.y +
                            geom.element_positions[base + 2] * dir.z);
    a[static_cast<size_t>(m)] = {std::cos(phase), std::sin(phase)};
  }
  return a;
}

std::vector<std::complex<double>> make_block(const ArrayGeometry& geom,
                                             const AngleConvention& conv,
                                             double az, double el, int64_t S,
                                             double noise, unsigned seed) {
  const auto a = steering_vector(geom, conv, az, el);
  const int64_t C = geom.num_elements();
  std::vector<std::complex<double>> x(static_cast<size_t>(C * S));
  std::mt19937 rng(seed);
  std::normal_distribution<double> n(0.0, noise);
  for (int64_t s = 0; s < S; ++s) {
    const double gp =
        2.0 * kPi * static_cast<double>(s) / static_cast<double>(S);
    const std::complex<double> g{std::cos(gp), std::sin(gp)};
    for (int64_t c = 0; c < C; ++c) {
      std::complex<double> v = a[static_cast<size_t>(c)] * g;
      v += std::complex<double>(n(rng), n(rng));
      x[static_cast<size_t>(c * S + s)] = v;
    }
  }
  return x;
}

std::vector<std::complex<double>> make_block_two(
    const ArrayGeometry& geom, const AngleConvention& conv, double az0,
    double el0, double az1, double el1, int64_t S, double noise,
    unsigned seed) {
  const auto a0 = steering_vector(geom, conv, az0, el0);
  const auto a1 = steering_vector(geom, conv, az1, el1);
  const int64_t C = geom.num_elements();
  std::vector<std::complex<double>> x(static_cast<size_t>(C * S));
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> phase(0.0, 2.0 * kPi);
  std::normal_distribution<double> n(0.0, noise);
  for (int64_t s = 0; s < S; ++s) {
    const double p0 = phase(rng);
    const double p1 = phase(rng);
    const std::complex<double> g0{std::cos(p0), std::sin(p0)};
    const std::complex<double> g1{std::cos(p1), std::sin(p1)};
    for (int64_t c = 0; c < C; ++c) {
      std::complex<double> v =
          a0[static_cast<size_t>(c)] * g0 + a1[static_cast<size_t>(c)] * g1;
      v += std::complex<double>(n(rng), n(rng));
      x[static_cast<size_t>(c * S + s)] = v;
    }
  }
  return x;
}

template <typename T>
std::shared_ptr<holoscan::Tensor> block_to_device(
    const std::vector<std::complex<double>>& host,
    const std::vector<int64_t>& shape) {
  std::vector<std::complex<T>> cast(host.size());
  for (size_t i = 0; i < host.size(); ++i) {
    cast[i] = {static_cast<T>(host[i].real()), static_cast<T>(host[i].imag())};
  }
  return to_device_tensor(cast, shape, complex_dtype<T>());
}

void wire_noop_output(holoscan::Fragment* app,
                      const std::shared_ptr<holoscan::Operator>& op,
                      const std::string& port) {
  const auto noop = [](const holoscan::TensorMap&) {};
  auto sink = app->make_operator<
      holoscan::test::TestHarnessSinkOp<holoscan::TensorMap>>(port + "_noop");
  sink->set_validators({noop});
  app->add_flow(op, sink, {{port, "input"}});
}

struct RootMusicOutputs {
  HostTensor aoa;
  HostTensor doa_multi;
  HostTensor source_count;
  HostTensor eigenvalues;
};

template <typename T, typename... ExtraArgs>
RootMusicOutputs run_root_music(std::shared_ptr<holoscan::Tensor> snapshot,
                                ExtraArgs&&... extra_args) {
  auto harness = create_operator_test<RootMUSICEstimatorOp<T>>(
      std::forward<ExtraArgs>(extra_args)...);
  harness->template add_input_port<holoscan::TensorMap>(
      "snapshot", {single_map(std::move(snapshot), "snapshot")});

  TensorMapCapture aoa_cap, doa_cap, cnt_cap, eig_cap;
  harness->template add_output_port<holoscan::TensorMap>(
      "aoa_estimates", aoa_cap.validators());
  harness->template add_output_port<holoscan::TensorMap>(
      "doa_multi", doa_cap.validators());
  harness->template add_output_port<holoscan::TensorMap>(
      "source_count", cnt_cap.validators());
  harness->template add_output_port<holoscan::TensorMap>(
      "eigenvalues", eig_cap.validators());
  harness->run_test();

  if (aoa_cap.frames.size() != 1u || doa_cap.frames.size() != 1u ||
      cnt_cap.frames.size() != 1u || eig_cap.frames.size() != 1u) {
    throw std::runtime_error("RootMUSICEstimatorOp test: expected one frame");
  }
  return {aoa_cap.frames.front(), doa_cap.frames.front(),
          cnt_cap.frames.front(), eig_cap.frames.front()};
}

TEST(RootMUSICEstimatorOpTest, RecoversKnownDoaUlaFixedOneSource) {
  constexpr int64_t kN = 8;
  constexpr int64_t kS = 32;
  constexpr double kTruthAz = 0.08;

  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  const auto block = make_block(geom, conv, kTruthAz, 0.0, kS, 0.02, 3u);
  auto snap = block_to_device<double>(block, {1, kN, kS});

  const auto out = run_root_music<double>(
      snap, holoscan::Arg("array_geometry") = geom,
      holoscan::Arg("angle_convention") = conv,
      holoscan::Arg("num_sources") = int64_t{1});

  ASSERT_EQ(out.aoa.shape, (std::vector<int64_t>{1, 2}));
  const double* aoa = out.aoa.as<double>();
  EXPECT_NEAR(aoa[0], kTruthAz, 0.02);
  EXPECT_NEAR(aoa[1], 0.0, 1e-9);
}

TEST(RootMUSICEstimatorOpTest, RecoversTwoSources) {
  constexpr int64_t kN = 12;
  constexpr int64_t kS = 256;
  constexpr double kAz0 = 0.20;
  constexpr double kAz1 = -0.15;

  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  const auto block =
      make_block_two(geom, conv, kAz0, 0.0, kAz1, 0.0, kS, 0.02, 71u);
  auto snap = block_to_device<double>(block, {1, kN, kS});

  const auto out = run_root_music<double>(
      snap, holoscan::Arg("array_geometry") = geom,
      holoscan::Arg("angle_convention") = conv,
      holoscan::Arg("num_sources") = int64_t{2},
      holoscan::Arg("max_sources") = int64_t{2});

  ASSERT_EQ(out.doa_multi.shape, (std::vector<int64_t>{1, 2, 2}));
  EXPECT_EQ(out.source_count.as<int32_t>()[0], 2);

  const double* dm = out.doa_multi.as<double>();
  const double az_a = dm[0];
  const double az_b = dm[2];
  const bool matched =
      (std::abs(az_a - kAz0) < 0.03 && std::abs(az_b - kAz1) < 0.03) ||
      (std::abs(az_a - kAz1) < 0.03 && std::abs(az_b - kAz0) < 0.03);
  EXPECT_TRUE(matched) << "az_a=" << az_a << " az_b=" << az_b;

  const double* aoa = out.aoa.as<double>();
  EXPECT_NEAR(dm[0], aoa[0], 1e-9);
  EXPECT_NEAR(dm[1], aoa[1], 1e-9);
}

TEST(RootMUSICEstimatorOpTest, ResolvesCloselySpacedSources) {
  constexpr int64_t kN = 16;
  constexpr int64_t kS = 512;
  constexpr double kAz0 = -0.035;
  constexpr double kAz1 = 0.035;

  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  const auto block =
      make_block_two(geom, conv, kAz0, 0.0, kAz1, 0.0, kS, 0.005, 73u);
  auto snap = block_to_device<double>(block, {1, kN, kS});

  const auto out = run_root_music<double>(
      snap, holoscan::Arg("array_geometry") = geom,
      holoscan::Arg("angle_convention") = conv,
      holoscan::Arg("num_sources") = int64_t{2},
      holoscan::Arg("max_sources") = int64_t{2});

  ASSERT_EQ(out.source_count.as<int32_t>()[0], 2);
  std::array<double, 2> actual{
      out.doa_multi.as<double>()[0],
      out.doa_multi.as<double>()[2],
  };
  std::sort(actual.begin(), actual.end());
  EXPECT_NEAR(actual[0], kAz0, 0.01);
  EXPECT_NEAR(actual[1], kAz1, 0.01);
}

TEST(RootMUSICEstimatorOpTest, MaxSourcesCapsReportedRoots) {
  constexpr int64_t kN = 12;
  constexpr int64_t kS = 256;
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  const auto block =
      make_block_two(geom, conv, -0.15, 0.0, 0.20, 0.0, kS, 0.02, 78u);
  auto snap = block_to_device<double>(block, {1, kN, kS});

  const auto out = run_root_music<double>(
      snap, holoscan::Arg("array_geometry") = geom,
      holoscan::Arg("angle_convention") = conv,
      holoscan::Arg("num_sources") = int64_t{2},
      holoscan::Arg("max_sources") = int64_t{1});

  EXPECT_EQ(out.doa_multi.shape, (std::vector<int64_t>{1, 1, 2}));
  EXPECT_EQ(out.source_count.as<int32_t>()[0], 1);
  EXPECT_TRUE(std::isfinite(out.doa_multi.as<double>()[0]));
}

TEST(RootMUSICEstimatorOpTest, HighSnrLargeArrayRootRemainsFinite) {
  constexpr int64_t kN = 32;
  constexpr int64_t kS = 128;
  constexpr double kTruthAz = 0.17;

  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  const auto block =
      make_block(geom, conv, kTruthAz, 0.0, kS, 1e-6, 74u);
  auto snap = block_to_device<double>(block, {1, kN, kS});

  const auto out = run_root_music<double>(
      snap, holoscan::Arg("array_geometry") = geom,
      holoscan::Arg("angle_convention") = conv,
      holoscan::Arg("num_sources") = int64_t{1});

  ASSERT_EQ(out.source_count.as<int32_t>()[0], 1);
  const double az = out.aoa.as<double>()[0];
  EXPECT_TRUE(std::isfinite(az));
  EXPECT_NEAR(az, kTruthAz, 1e-4);
}

TEST(RootMUSICEstimatorOpTest, AutoNumSourcesRecoversOneSource) {
  constexpr int64_t kN = 8;
  constexpr int64_t kS = 64;
  constexpr double kTruthAz = 0.1;

  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  const auto block = make_block(geom, conv, kTruthAz, 0.0, kS, 0.02, 9u);
  auto snap = block_to_device<double>(block, {1, kN, kS});

  const auto out = run_root_music<double>(
      snap, holoscan::Arg("array_geometry") = geom,
      holoscan::Arg("angle_convention") = conv,
      holoscan::Arg("max_sources") = int64_t{4});

  EXPECT_EQ(out.source_count.as<int32_t>()[0], 1);
  const double* aoa = out.aoa.as<double>();
  EXPECT_NEAR(aoa[0], kTruthAz, 0.02);
}

TEST(RootMUSICEstimatorOpTest, RecoversUraSingleSourceAzEl) {
  constexpr int64_t kNy = 6;
  constexpr int64_t kNz = 6;
  constexpr int64_t kS = 256;
  constexpr double kAz = 0.15;
  constexpr double kEl = -0.10;

  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto geom = make_planar_ura_geometry(kNy, kNz, 0.5, 1.0);
  const auto block = make_block(geom, conv, kAz, kEl, kS, 0.02, 7u);
  auto snap = block_to_device<double>(block, {1, geom.num_elements(), kS});

  const auto out = run_root_music<double>(
      snap, holoscan::Arg("array_geometry") = geom,
      holoscan::Arg("angle_convention") = conv,
      holoscan::Arg("num_sources") = int64_t{1});

  ASSERT_EQ(out.aoa.shape, (std::vector<int64_t>{1, 2}));
  const double* aoa = out.aoa.as<double>();
  EXPECT_NEAR(aoa[0], kAz, 0.03);
  EXPECT_NEAR(aoa[1], kEl, 0.03);
}

TEST(RootMUSICEstimatorOpTest, RecoversUraTwoSourcesAzElPairing) {
  constexpr int64_t kNy = 8;
  constexpr int64_t kNz = 8;
  constexpr int64_t kS = 512;
  // Distinct in BOTH az and el so a mispaired result is detectable.
  constexpr double kAz0 = 0.20;
  constexpr double kEl0 = 0.12;
  constexpr double kAz1 = -0.15;
  constexpr double kEl1 = -0.10;

  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto geom = make_planar_ura_geometry(kNy, kNz, 0.5, 1.0);
  const auto block =
      make_block_two(geom, conv, kAz0, kEl0, kAz1, kEl1, kS, 0.01, 79u);
  auto snap = block_to_device<double>(block, {1, geom.num_elements(), kS});

  const auto out = run_root_music<double>(
      snap, holoscan::Arg("array_geometry") = geom,
      holoscan::Arg("angle_convention") = conv,
      holoscan::Arg("num_sources") = int64_t{2},
      holoscan::Arg("max_sources") = int64_t{2});

  ASSERT_EQ(out.doa_multi.shape, (std::vector<int64_t>{1, 2, 2}));
  EXPECT_EQ(out.source_count.as<int32_t>()[0], 2);

  const double* dm = out.doa_multi.as<double>();
  const auto near_pair = [](double az, double el, double taz, double tel) {
    return std::abs(az - taz) < 0.04 && std::abs(el - tel) < 0.04;
  };
  const bool matched =
      (near_pair(dm[0], dm[1], kAz0, kEl0) &&
       near_pair(dm[2], dm[3], kAz1, kEl1)) ||
      (near_pair(dm[0], dm[1], kAz1, kEl1) &&
       near_pair(dm[2], dm[3], kAz0, kEl0));
  EXPECT_TRUE(matched) << "pair0=(" << dm[0] << "," << dm[1] << ") pair1=("
                       << dm[2] << "," << dm[3] << ")";
}

TEST(RootMUSICEstimatorOpTest, RecoversUraSingleSourceAzElFloat) {
  constexpr int64_t kNy = 6;
  constexpr int64_t kNz = 6;
  constexpr int64_t kS = 256;
  constexpr double kAz = -0.12;
  constexpr double kEl = 0.08;

  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto geom = make_planar_ura_geometry(kNy, kNz, 0.5, 1.0);
  const auto block = make_block(geom, conv, kAz, kEl, kS, 0.02, 17u);
  auto snap = block_to_device<float>(block, {1, geom.num_elements(), kS});

  const auto out = run_root_music<float>(
      snap, holoscan::Arg("array_geometry") = geom,
      holoscan::Arg("angle_convention") = conv,
      holoscan::Arg("num_sources") = int64_t{1});

  ASSERT_EQ(out.aoa.shape, (std::vector<int64_t>{1, 2}));
  const float* aoa = out.aoa.as<float>();
  EXPECT_NEAR(aoa[0], kAz, 0.03);
  EXPECT_NEAR(aoa[1], kEl, 0.03);
}

TEST(RootMUSICEstimatorOpTest, RejectsNonUniformLinearGeometry) {
  constexpr int64_t kN = 5;
  constexpr int64_t kS = 16;
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  auto geom = make_ula_geometry(kN, 0.5, 1.0);
  geom.element_positions[3 * 3 + 1] = 1.65;
  const auto block = make_block(geom, conv, 0.1, 0.0, kS, 0.02, 75u);
  auto snap = block_to_device<double>(block, {1, kN, kS});

  EXPECT_THROW(
      run_root_music<double>(snap, holoscan::Arg("array_geometry") = geom,
                             holoscan::Arg("angle_convention") = conv),
      std::invalid_argument);
}

TEST(RootMUSICEstimatorOpTest, RejectsUlaParallelToBoresight) {
  constexpr int64_t kN = 4;
  ArrayGeometry geom;
  geom.wavelength = 1.0;
  geom.geometry_type = ArrayGeometryType::kUla;
  for (int64_t channel = 0; channel < kN; ++channel) {
    geom.element_positions.insert(
        geom.element_positions.end(),
        {0.5 * static_cast<double>(channel), 0.0, 0.0});
  }
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  std::vector<std::complex<double>> block(
      static_cast<size_t>(kN * 4), {1.0, 0.0});
  auto snap = block_to_device<double>(block, {1, kN, 4});

  EXPECT_THROW(
      run_root_music<double>(snap, holoscan::Arg("array_geometry") = geom,
                             holoscan::Arg("angle_convention") = conv),
      std::invalid_argument);
}

TEST(RootMUSICEstimatorOpTest, EigenvaluesAscendingAndNonNegative) {
  constexpr int64_t kN = 8;
  constexpr int64_t kS = 128;

  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  const auto block = make_block(geom, conv, 0.1, 0.0, kS, 0.02, 91u);
  auto snap = block_to_device<double>(block, {1, kN, kS});

  const auto out = run_root_music<double>(
      snap, holoscan::Arg("array_geometry") = geom,
      holoscan::Arg("angle_convention") = conv,
      holoscan::Arg("num_sources") = int64_t{1});

  ASSERT_EQ(out.eigenvalues.shape, (std::vector<int64_t>{1, kN}));
  const double* ev = out.eigenvalues.as<double>();
  for (int64_t c = 0; c < kN; ++c) {
    EXPECT_GE(ev[c], -1e-6);
    if (c > 0) {
      EXPECT_GE(ev[c], ev[c - 1] - 1e-9);
    }
  }
  EXPECT_GT(ev[kN - 1], 10.0 * ev[kN - 2]);
}

TEST(RootMUSICEstimatorOpTest, RankDeficientAutoFallsBackToOneSource) {
  constexpr int64_t kN = 8;
  constexpr int64_t kS = 4;

  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  const auto block = make_block(geom, conv, 0.1, 0.0, kS, 0.02, 12u);
  auto snap = block_to_device<double>(block, {1, kN, kS});

  const auto out = run_root_music<double>(
      snap, holoscan::Arg("array_geometry") = geom,
      holoscan::Arg("angle_convention") = conv,
      holoscan::Arg("max_sources") = int64_t{4});

  EXPECT_EQ(out.source_count.as<int32_t>()[0], 1);
}

TEST(RootMUSICEstimatorOpTest, RejectsNumSourcesTooLarge) {
  constexpr int64_t kN = 4;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto a = steering_vector(geom, conv, 0.0, 0.0);
  auto snap = block_to_device<double>(a, {1, kN, 1});

  EXPECT_THROW(
      run_root_music<double>(snap, holoscan::Arg("array_geometry") = geom,
                             holoscan::Arg("angle_convention") = conv,
                             holoscan::Arg("num_sources") = int64_t{kN}),
      std::invalid_argument);
}

TEST(RootMUSICEstimatorOpTest, RejectsInvalidMaxSources) {
  constexpr int64_t kN = 4;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto a = steering_vector(geom, conv, 0.0, 0.0);
  auto snap = block_to_device<double>(a, {1, kN, 1});
  EXPECT_THROW(
      run_root_music<double>(snap, holoscan::Arg("array_geometry") = geom,
                             holoscan::Arg("angle_convention") = conv,
                             holoscan::Arg("max_sources") = int64_t{0}),
      std::invalid_argument);
}

TEST(RootMUSICEstimatorOpTest, RejectsInvalidSourceCountPfa) {
  constexpr int64_t kN = 4;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto a = steering_vector(geom, conv, 0.0, 0.0);
  auto snap = block_to_device<double>(a, {1, kN, 1});
  EXPECT_THROW(
      run_root_music<double>(snap, holoscan::Arg("array_geometry") = geom,
                             holoscan::Arg("angle_convention") = conv,
                             holoscan::Arg("source_count_pfa") = 0.0),
      std::invalid_argument);
}

TEST(RootMUSICEstimatorOpTest, EmptyItemsProducesEmptyOutputs) {
  constexpr int64_t kN = 4;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  std::vector<std::complex<double>> host;
  auto snap = to_device_tensor(host, {0, kN, 4}, complex_dtype<double>());

  const auto out = run_root_music<double>(
      snap, holoscan::Arg("array_geometry") = geom,
      holoscan::Arg("angle_convention") = conv,
      holoscan::Arg("max_sources") = int64_t{3});

  EXPECT_EQ(out.aoa.shape, (std::vector<int64_t>{0, 2}));
  EXPECT_EQ(out.doa_multi.shape, (std::vector<int64_t>{0, 3, 2}));
  EXPECT_EQ(out.source_count.shape, (std::vector<int64_t>{0}));
  EXPECT_EQ(out.eigenvalues.shape, (std::vector<int64_t>{0, kN}));
}

TEST(RootMUSICEstimatorOpTest, RejectsChannelMismatch) {
  constexpr int64_t kN = 4;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  std::vector<std::complex<double>> host(static_cast<size_t>((kN - 1) * 4),
                                         {1.0, 0.0});
  auto snap = to_device_tensor(host, {1, kN - 1, 4}, complex_dtype<double>());
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  EXPECT_THROW(
      run_root_music<double>(snap, holoscan::Arg("array_geometry") = geom,
                             holoscan::Arg("angle_convention") = conv),
      std::invalid_argument);
}

TEST(RootMUSICEstimatorOpTest, RejectsSnapshotDtypeMismatch) {
  constexpr int64_t kN = 4;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto a = steering_vector(geom, conv, 0.0, 0.0);
  auto snap_f = block_to_device<float>(a, {1, kN, 1});
  EXPECT_THROW(
      run_root_music<double>(snap_f, holoscan::Arg("array_geometry") = geom,
                             holoscan::Arg("angle_convention") = conv),
      std::invalid_argument);
}

TEST(RootMUSICEstimatorOpTest, RejectsPageableHostSnapshot) {
  constexpr int64_t kN = 4;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto a = steering_vector(geom, conv, 0.0, 0.0);
  auto snap_cpu =
      to_pageable_host_tensor(a, {1, kN, 1}, complex_dtype<double>());
  EXPECT_THROW(
      run_root_music<double>(snap_cpu, holoscan::Arg("array_geometry") = geom,
                             holoscan::Arg("angle_convention") = conv),
      std::invalid_argument);
}

TEST(RootMUSICEstimatorOpTest, RejectsSnapshotRank2) {
  constexpr int64_t kN = 4;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto a = steering_vector(geom, conv, 0.05, 0.0);
  auto snap = block_to_device<double>(a, {1, kN});
  EXPECT_THROW(
      run_root_music<double>(snap, holoscan::Arg("array_geometry") = geom,
                             holoscan::Arg("angle_convention") = conv),
      std::invalid_argument);
}

TEST(RootMUSICEstimatorOpTest, AcceptsPinnedHostSnapshot) {
  constexpr int64_t kN = 8;
  constexpr int64_t kS = 32;
  constexpr double kTruthAz = 0.08;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto block = make_block(geom, conv, kTruthAz, 0.0, kS, 0.02, 161u);
  const auto tensor = cuda_accessible::make_tensor(
      block, {1, kN, kS}, complex_dtype<double>(), kDLCUDAHost);

  const auto out = run_root_music<double>(
      tensor, holoscan::Arg("array_geometry") = geom,
      holoscan::Arg("angle_convention") = conv,
      holoscan::Arg("num_sources") = int64_t{1});
  EXPECT_NEAR(out.aoa.as<double>()[0], kTruthAz, 0.02);
}

TEST(RootMUSICEstimatorOpTest, AcceptsManagedSnapshot) {
  constexpr int64_t kN = 8;
  constexpr int64_t kS = 32;
  constexpr double kTruthAz = 0.08;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto block = make_block(geom, conv, kTruthAz, 0.0, kS, 0.02, 162u);
  const auto tensor = cuda_accessible::make_tensor(
      block, {1, kN, kS}, complex_dtype<double>(), kDLCUDAManaged);

  const auto out = run_root_music<double>(
      tensor, holoscan::Arg("array_geometry") = geom,
      holoscan::Arg("angle_convention") = conv,
      holoscan::Arg("num_sources") = int64_t{1});
  EXPECT_NEAR(out.aoa.as<double>()[0], kTruthAz, 0.02);
}

TEST(RootMUSICEstimatorOpTest, FloatInstantiationSmoke) {
  constexpr int64_t kN = 8;
  constexpr int64_t kS = 32;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto block = make_block(geom, conv, 0.06, 0.0, kS, 0.02, 11u);
  auto snap = block_to_device<float>(block, {1, kN, kS});

  const auto out = run_root_music<float>(
      snap, holoscan::Arg("array_geometry") = geom,
      holoscan::Arg("angle_convention") = conv,
      holoscan::Arg("num_sources") = int64_t{1});
  ASSERT_EQ(out.aoa.shape, (std::vector<int64_t>{1, 2}));
  const float* aoa = out.aoa.as<float>();
  EXPECT_NEAR(aoa[0], 0.06f, 0.03f);
}

TEST(RootMUSICEstimatorOpTest, FloatDoubleAgreeOnDoa) {
  constexpr int64_t kN = 8;
  constexpr int64_t kS = 32;
  constexpr double kTruthAz = 0.09;

  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  const auto block = make_block(geom, conv, kTruthAz, 0.0, kS, 0.05, 5u);
  auto snap_f = block_to_device<float>(block, {1, kN, kS});
  auto snap_d = block_to_device<double>(block, {1, kN, kS});

  const auto args = std::make_tuple(
      holoscan::Arg("array_geometry") = geom,
      holoscan::Arg("angle_convention") = conv,
      holoscan::Arg("num_sources") = int64_t{1});

  const auto out_f = std::apply(
      [&](auto&&... a) { return run_root_music<float>(snap_f, a...); }, args);
  const auto out_d = std::apply(
      [&](auto&&... a) { return run_root_music<double>(snap_d, a...); },
      args);

  const float* aoa_f = out_f.aoa.as<float>();
  const double* aoa_d = out_d.aoa.as<double>();
  EXPECT_NEAR(aoa_f[0], static_cast<float>(aoa_d[0]), 1e-2f);
}

TEST(RootMUSICEstimatorOpTest, DeterministicAcrossRuns) {
  constexpr int64_t kN = 8;
  constexpr int64_t kS = 64;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto block = make_block(geom, conv, 0.09, 0.0, kS, 0.02, 171u);

  const auto run = [&]() {
    return run_root_music<double>(
        block_to_device<double>(block, {1, kN, kS}),
        holoscan::Arg("array_geometry") = geom,
        holoscan::Arg("angle_convention") = conv,
        holoscan::Arg("num_sources") = int64_t{1},
        holoscan::Arg("max_sources") = int64_t{2});
  };
  const auto first = run();
  const auto second = run();

  EXPECT_EQ(first.aoa.data, second.aoa.data);
  EXPECT_EQ(first.doa_multi.data, second.doa_multi.data);
  EXPECT_EQ(first.source_count.data, second.source_count.data);
  EXPECT_EQ(first.eigenvalues.data, second.eigenvalues.data);
}

TEST(RootMUSICEstimatorOpTest, DirectionCosineConventionMapsUlaProjection) {
  constexpr int64_t kN = 8;
  constexpr int64_t kS = 64;
  constexpr double kTruthU = 0.23;
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kDirectionCosine;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  const auto block =
      make_block(geom, conv, kTruthU, 0.0, kS, 0.01, 76u);
  auto snap = block_to_device<double>(block, {1, kN, kS});

  const auto out = run_root_music<double>(
      snap, holoscan::Arg("array_geometry") = geom,
      holoscan::Arg("angle_convention") = conv,
      holoscan::Arg("num_sources") = int64_t{1});

  EXPECT_NEAR(out.aoa.as<double>()[0], kTruthU, 0.01);
  EXPECT_NEAR(out.aoa.as<double>()[1], 0.0, 1e-9);
}

TEST(RootMUSICEstimatorOpTest, CustomConventionMapsSignedAxes) {
  constexpr int64_t kN = 8;
  constexpr int64_t kS = 64;
  constexpr double kCanonicalAz = 0.18;
  AngleConvention radar;
  radar.preset = AngleConventionPreset::kRadarBoresight;
  AngleConvention custom;
  custom.preset = AngleConventionPreset::kCustom;
  custom.boresight_axis = SignedAxis::kPlusX;
  custom.azimuth_positive_axis = SignedAxis::kMinusY;
  custom.elevation_positive_axis = SignedAxis::kMinusZ;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  const auto block =
      make_block(geom, radar, kCanonicalAz, 0.0, kS, 0.01, 176u);

  const auto out = run_root_music<double>(
      block_to_device<double>(block, {1, kN, kS}),
      holoscan::Arg("array_geometry") = geom,
      holoscan::Arg("angle_convention") = custom,
      holoscan::Arg("num_sources") = int64_t{1});

  EXPECT_NEAR(out.aoa.as<double>()[0], -kCanonicalAz, 0.01);
  EXPECT_NEAR(out.aoa.as<double>()[1], 0.0, 1e-9);
}

TEST(RootMUSICEstimatorOpTest, CustomConventionSupportsZenithReference) {
  constexpr int64_t kN = 8;
  constexpr int64_t kS = 64;
  AngleConvention radar;
  radar.preset = AngleConventionPreset::kRadarBoresight;
  AngleConvention custom;
  custom.preset = AngleConventionPreset::kCustom;
  custom.elevation_reference = ElevationReference::kZenith;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  const auto block = make_block(geom, radar, 0.1, 0.0, kS, 0.01, 177u);

  const auto out = run_root_music<double>(
      block_to_device<double>(block, {1, kN, kS}),
      holoscan::Arg("array_geometry") = geom,
      holoscan::Arg("angle_convention") = custom,
      holoscan::Arg("num_sources") = int64_t{1});

  EXPECT_NEAR(out.aoa.as<double>()[0], 0.1, 0.01);
  EXPECT_NEAR(out.aoa.as<double>()[1], kPi / 2.0, 1e-9);
}

TEST(RootMUSICEstimatorOpTest, RejectsDegenerateCustomConvention) {
  constexpr int64_t kN = 4;
  AngleConvention radar;
  radar.preset = AngleConventionPreset::kRadarBoresight;
  AngleConvention custom;
  custom.preset = AngleConventionPreset::kCustom;
  custom.azimuth_positive_axis = custom.boresight_axis;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  const auto block = make_block(geom, radar, 0.1, 0.0, 8, 0.01, 178u);

  EXPECT_THROW(
      run_root_music<double>(
          block_to_device<double>(block, {1, kN, 8}),
          holoscan::Arg("array_geometry") = geom,
          holoscan::Arg("angle_convention") = custom,
          holoscan::Arg("num_sources") = int64_t{1}),
      std::invalid_argument);
}

TEST(RootMUSICEstimatorOpTest, DirectionCosineNearEndfirePreservesSign) {
  constexpr int64_t kN = 16;
  constexpr int64_t kS = 128;
  constexpr double kU0 = -0.92;
  constexpr double kU1 = 0.92;
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kDirectionCosine;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  const auto b0 = make_block(geom, conv, kU0, 0.0, kS, 0.005, 79u);
  const auto b1 = make_block(geom, conv, kU1, 0.0, kS, 0.005, 80u);
  std::vector<std::complex<double>> batch;
  batch.insert(batch.end(), b0.begin(), b0.end());
  batch.insert(batch.end(), b1.begin(), b1.end());
  auto snap = block_to_device<double>(batch, {2, kN, kS});

  const auto out = run_root_music<double>(
      snap, holoscan::Arg("array_geometry") = geom,
      holoscan::Arg("angle_convention") = conv,
      holoscan::Arg("num_sources") = int64_t{1});

  EXPECT_NEAR(out.aoa.as<double>()[0], kU0, 0.01);
  EXPECT_NEAR(out.aoa.as<double>()[2], kU1, 0.01);
}

TEST(RootMUSICEstimatorOpTest, UnusedDoaRowsAreNan) {
  constexpr int64_t kN = 8;
  constexpr int64_t kS = 64;
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  const auto block = make_block(geom, conv, 0.1, 0.0, kS, 0.01, 77u);
  auto snap = block_to_device<double>(block, {1, kN, kS});

  const auto out = run_root_music<double>(
      snap, holoscan::Arg("array_geometry") = geom,
      holoscan::Arg("angle_convention") = conv,
      holoscan::Arg("num_sources") = int64_t{1},
      holoscan::Arg("max_sources") = int64_t{3});

  ASSERT_EQ(out.source_count.as<int32_t>()[0], 1);
  const double* doa = out.doa_multi.as<double>();
  EXPECT_TRUE(std::isfinite(doa[0]));
  EXPECT_TRUE(std::isfinite(doa[1]));
  for (int64_t index = 2; index < 6; ++index) {
    EXPECT_TRUE(std::isnan(doa[index]));
  }
}

class SnapshotMaskRelayOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(SnapshotMaskRelayOp)

  std::shared_ptr<holoscan::Tensor> snapshot;
  std::shared_ptr<holoscan::Tensor> is_valid;

  void setup(holoscan::OperatorSpec& spec) override {
    spec.output<holoscan::TensorMap>("snapshot");
    spec.output<holoscan::TensorMap>("is_valid");
  }

  void compute(holoscan::InputContext&, holoscan::OutputContext& op_output,
               holoscan::ExecutionContext&) override {
    holoscan::radar::emit_single(op_output, "snapshot", snapshot);
    holoscan::radar::emit_single(op_output, "is_valid", is_valid);
  }
};

class IsValidMaskApp : public holoscan::Application {
 public:
  std::shared_ptr<holoscan::Tensor> snapshot;
  std::shared_ptr<holoscan::Tensor> is_valid;
  ArrayGeometry geometry;
  AngleConvention convention;
  TensorMapCapture* aoa = nullptr;
  TensorMapCapture* doa_multi = nullptr;
  TensorMapCapture* source_count = nullptr;

  void compose() override {
    using holoscan::Arg;
    auto relay = make_operator<SnapshotMaskRelayOp>(
        "relay", make_condition<holoscan::CountCondition>(1));
    relay->snapshot = snapshot;
    relay->is_valid = is_valid;

    auto root_music = make_operator<RootMUSICEstimatorOp<double>>(
        "root_music", Arg("array_geometry") = geometry,
        Arg("angle_convention") = convention, Arg("num_sources") = int64_t{1},
        Arg("max_sources") = int64_t{2});

    add_flow(relay, root_music,
             {{"snapshot", "snapshot"}, {"is_valid", "is_valid"}});
    holoscan::radar::testing::wire_capture_sink(this, root_music,
                                                "aoa_estimates", aoa);
    if (doa_multi != nullptr) {
      holoscan::radar::testing::wire_capture_sink(this, root_music,
                                                  "doa_multi", doa_multi);
    } else {
      wire_noop_output(this, root_music, "doa_multi");
    }
    if (source_count != nullptr) {
      holoscan::radar::testing::wire_capture_sink(this, root_music,
                                                  "source_count", source_count);
    } else {
      wire_noop_output(this, root_music, "source_count");
    }
    wire_noop_output(this, root_music, "eigenvalues");
  }
};

TEST(RootMUSICEstimatorOpTest, IsValidMaskExcludesWrongSnapshots) {
  constexpr int64_t kN = 8;
  constexpr int64_t kS = 32;
  constexpr double kTruthAz = 0.1;
  constexpr double kWrongAz = -0.2;

  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);

  const auto good = make_block(geom, conv, kTruthAz, 0.0, kS / 2, 0.02, 31u);
  const auto bad = make_block(geom, conv, kWrongAz, 0.0, kS / 2, 0.02, 32u);
  std::vector<std::complex<double>> block(static_cast<size_t>(kN * kS));
  std::vector<uint8_t> mask(static_cast<size_t>(kS));
  for (int64_t c = 0; c < kN; ++c) {
    for (int64_t s = 0; s < kS / 2; ++s) {
      block[static_cast<size_t>(c * kS + s)] =
          good[static_cast<size_t>(c * (kS / 2) + s)];
      block[static_cast<size_t>(c * kS + (kS / 2) + s)] =
          bad[static_cast<size_t>(c * (kS / 2) + s)];
    }
  }
  for (int64_t s = 0; s < kS; ++s) {
    mask[static_cast<size_t>(s)] = s < kS / 2 ? 1 : 0;
  }

  IsValidMaskApp app;
  app.snapshot = block_to_device<double>(block, {1, kN, kS});
  app.is_valid = to_device_bool_tensor(mask, {1, kS});
  app.geometry = geom;
  app.convention = conv;

  TensorMapCapture aoa_cap;
  app.aoa = &aoa_cap;
  app.run();

  ASSERT_EQ(aoa_cap.frames.size(), 1u);
  const double* aoa = aoa_cap.frames[0].as<double>();
  EXPECT_NEAR(aoa[0], kTruthAz, 0.03);
}

TEST(RootMUSICEstimatorOpTest, AllInvalidMaskBlanksDoa) {
  constexpr int64_t kN = 8;
  constexpr int64_t kS = 16;

  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  const auto block = make_block(geom, conv, 0.1, 0.0, kS, 0.02, 33u);
  std::vector<uint8_t> mask(static_cast<size_t>(kS), 0);

  IsValidMaskApp app;
  app.snapshot = block_to_device<double>(block, {1, kN, kS});
  app.is_valid = to_device_bool_tensor(mask, {1, kS});
  app.geometry = geom;
  app.convention = conv;

  TensorMapCapture aoa_cap;
  TensorMapCapture doa_cap;
  TensorMapCapture cnt_cap;
  app.aoa = &aoa_cap;
  app.doa_multi = &doa_cap;
  app.source_count = &cnt_cap;

  app.run();

  ASSERT_EQ(aoa_cap.frames.size(), 1u);
  ASSERT_EQ(doa_cap.frames.size(), 1u);
  ASSERT_EQ(cnt_cap.frames.size(), 1u);

  const double* aoa = aoa_cap.frames[0].as<double>();
  EXPECT_TRUE(std::isnan(aoa[0]));
  EXPECT_TRUE(std::isnan(aoa[1]));

  EXPECT_EQ(cnt_cap.frames[0].as<int32_t>()[0], 0);
  const double* dm = doa_cap.frames[0].as<double>();
  for (int64_t k = 0; k < 2; ++k) {
    EXPECT_TRUE(std::isnan(dm[k * 2]));
    EXPECT_TRUE(std::isnan(dm[k * 2 + 1]));
  }
}

class NumSourcesOverrideApp : public holoscan::Application {
 public:
  std::vector<holoscan::TensorMap> snapshot_frames;
  std::vector<int64_t> num_sources_overrides;
  TensorMapCapture* source_count = nullptr;
  ArrayGeometry geometry;
  AngleConvention convention;

  void compose() override {
    using holoscan::Arg;
    auto op = make_operator<RootMUSICEstimatorOp<double>>(
        "root_music", Arg("array_geometry") = geometry,
        Arg("angle_convention") = convention,
        Arg("num_sources") = num_sources_overrides.front(),
        Arg("max_sources") = int64_t{4});
    holoscan::radar::testing::wire_data_source(this, op, "snapshot",
                                               snapshot_frames);
    holoscan::radar::testing::wire_leading_override(this, op, "num_sources",
                                                    num_sources_overrides);
    holoscan::radar::testing::wire_capture_sink(this, op, "source_count",
                                                source_count);
    wire_noop_output(this, op, "aoa_estimates");
    wire_noop_output(this, op, "doa_multi");
    wire_noop_output(this, op, "eigenvalues");
  }
};

TEST(RootMUSICEstimatorOpTest, LiveNumSourcesStickyOverride) {
  constexpr int64_t kN = 12;
  constexpr int64_t kS = 128;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto block =
      make_block_two(geom, conv, 0.2, 0.0, -0.15, 0.0, kS, 0.02, 111u);
  auto snap = block_to_device<double>(block, {1, kN, kS});

  NumSourcesOverrideApp app;
  app.geometry = geom;
  app.convention = conv;
  app.snapshot_frames = {single_map(snap, "snapshot"),
                         single_map(snap, "snapshot")};
  app.num_sources_overrides = {int64_t{1}, int64_t{2}};

  TensorMapCapture cnt_cap;
  app.source_count = &cnt_cap;
  app.run();

  ASSERT_EQ(cnt_cap.frames.size(), 2u);
  EXPECT_EQ(cnt_cap.frames[0].as<int32_t>()[0], 1);
  EXPECT_EQ(cnt_cap.frames[1].as<int32_t>()[0], 2);
}

class RootMusicArrayGeometryOverrideApp : public holoscan::Application {
 public:
  std::vector<holoscan::TensorMap> snapshot_frames;
  std::vector<ArrayGeometry> geometry_overrides;
  TensorMapCapture* aoa = nullptr;
  AngleConvention convention;

  void compose() override {
    using holoscan::Arg;
    auto op = make_operator<RootMUSICEstimatorOp<double>>(
        "root_music", Arg("array_geometry") = geometry_overrides.front(),
        Arg("angle_convention") = convention, Arg("num_sources") = int64_t{1});
    holoscan::radar::testing::wire_data_source(this, op, "snapshot",
                                               snapshot_frames);
    holoscan::radar::testing::wire_leading_override(
        this, op, "array_geometry", geometry_overrides);
    holoscan::radar::testing::wire_capture_sink(this, op, "aoa_estimates", aoa);
    wire_noop_output(this, op, "doa_multi");
    wire_noop_output(this, op, "source_count");
    wire_noop_output(this, op, "eigenvalues");
  }
};

TEST(RootMUSICEstimatorOpTest, LiveArrayGeometryStickyOverride) {
  constexpr int64_t kN = 4;
  constexpr int64_t kS = 16;
  const auto geom_a = make_ula_geometry(kN, 0.5, 1.0);
  auto geom_b = make_ula_geometry(kN, 0.5, 1.0);
  geom_b.wavelength = 0.8;
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto block = make_block(geom_a, conv, 0.05, 0.0, kS, 0.05, 61u);
  auto snap = block_to_device<double>(block, {1, kN, kS});

  RootMusicArrayGeometryOverrideApp app;
  app.convention = conv;
  app.geometry_overrides = {geom_a, geom_b, geom_a};
  app.snapshot_frames = {single_map(snap, "snapshot"),
                         single_map(snap, "snapshot"),
                         single_map(snap, "snapshot")};

  TensorMapCapture aoa_cap;
  app.aoa = &aoa_cap;
  testing::internal::CaptureStderr();
  app.run();
  const std::string logs = testing::internal::GetCapturedStderr();

  ASSERT_EQ(aoa_cap.frames.size(), 3u);
  EXPECT_NE(aoa_cap.frames[0].as<double>()[0],
            aoa_cap.frames[1].as<double>()[0]);
  EXPECT_EQ(aoa_cap.frames[0].data, aoa_cap.frames[2].data);
  const std::string warning =
      "unseen array_geometry override; revalidating ULA and solver dimensions";
  const size_t first_warning = logs.find(warning);
  ASSERT_NE(first_warning, std::string::npos);
  EXPECT_EQ(logs.find(warning, first_warning + warning.size()),
            std::string::npos);
}

class SpatialSmoothingOverrideApp : public holoscan::Application {
 public:
  std::vector<holoscan::TensorMap> snapshot_frames;
  std::vector<std::vector<int64_t>> smoothing_overrides;
  TensorMapCapture* eigenvalues = nullptr;
  ArrayGeometry geometry;
  AngleConvention convention;

  void compose() override {
    using holoscan::Arg;
    auto op = make_operator<RootMUSICEstimatorOp<double>>(
        "root_music", Arg("array_geometry") = geometry,
        Arg("angle_convention") = convention,
        Arg("spatial_smoothing") = smoothing_overrides.front(),
        Arg("num_sources") = int64_t{1});
    holoscan::radar::testing::wire_data_source(this, op, "snapshot",
                                               snapshot_frames);
    holoscan::radar::testing::wire_leading_override(
        this, op, "spatial_smoothing", smoothing_overrides);
    holoscan::radar::testing::wire_capture_sink(this, op, "eigenvalues",
                                                eigenvalues);
    wire_noop_output(this, op, "aoa_estimates");
    wire_noop_output(this, op, "doa_multi");
    wire_noop_output(this, op, "source_count");
  }
};

TEST(RootMUSICEstimatorOpTest, LiveSpatialSmoothingStickyOverride) {
  constexpr int64_t kN = 8;
  constexpr int64_t kS = 32;
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  const auto block = make_block(geom, conv, 0.1, 0.0, kS, 0.02, 141u);
  auto snap = block_to_device<double>(block, {1, kN, kS});

  SpatialSmoothingOverrideApp app;
  app.geometry = geom;
  app.convention = conv;
  app.snapshot_frames = {single_map(snap, "snapshot"),
                         single_map(snap, "snapshot"),
                         single_map(snap, "snapshot")};
  app.smoothing_overrides = {{}, {2}};

  TensorMapCapture eig_cap;
  app.eigenvalues = &eig_cap;
  app.run();

  ASSERT_EQ(eig_cap.frames.size(), 3u);
  EXPECT_EQ(eig_cap.frames[0].shape, (std::vector<int64_t>{1, kN}));
  EXPECT_EQ(eig_cap.frames[1].shape, (std::vector<int64_t>{1, kN - 1}));
  EXPECT_EQ(eig_cap.frames[2].shape, eig_cap.frames[1].shape);
}

class SourceCountPfaOverrideApp : public holoscan::Application {
 public:
  std::vector<holoscan::TensorMap> snapshot_frames;
  std::vector<double> pfa_overrides;
  TensorMapCapture* source_count = nullptr;
  ArrayGeometry geometry;
  AngleConvention convention;

  void compose() override {
    using holoscan::Arg;
    auto op = make_operator<RootMUSICEstimatorOp<double>>(
        "root_music", Arg("array_geometry") = geometry,
        Arg("angle_convention") = convention, Arg("num_sources") = int64_t{-1},
        Arg("source_count_pfa") = pfa_overrides.front(),
        Arg("max_sources") = int64_t{4});
    holoscan::radar::testing::wire_data_source(this, op, "snapshot",
                                               snapshot_frames);
    holoscan::radar::testing::wire_leading_override(
        this, op, "source_count_pfa", pfa_overrides);
    holoscan::radar::testing::wire_capture_sink(this, op, "source_count",
                                                source_count);
    wire_noop_output(this, op, "aoa_estimates");
    wire_noop_output(this, op, "doa_multi");
    wire_noop_output(this, op, "eigenvalues");
  }
};

TEST(RootMUSICEstimatorOpTest, LiveSourceCountPfaStickyOverride) {
  constexpr int64_t kN = 8;
  constexpr int64_t kS = 40;
  constexpr double kTruthAz = 0.08;

  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  const auto block =
      make_block(geom, conv, kTruthAz, 0.0, kS, 0.45, 191u);
  auto snap = block_to_device<double>(block, {1, kN, kS});

  SourceCountPfaOverrideApp app;
  app.geometry = geom;
  app.convention = conv;
  app.snapshot_frames = {single_map(snap, "snapshot"),
                         single_map(snap, "snapshot"),
                         single_map(snap, "snapshot")};
  app.pfa_overrides = {1e-4, 0.5};

  TensorMapCapture cnt_cap;
  app.source_count = &cnt_cap;
  app.run();

  ASSERT_EQ(cnt_cap.frames.size(), 3u);
  const int32_t tight = cnt_cap.frames[0].as<int32_t>()[0];
  const int32_t loose = cnt_cap.frames[1].as<int32_t>()[0];
  EXPECT_LE(tight, loose);
  EXPECT_EQ(loose, cnt_cap.frames[2].as<int32_t>()[0]);
}

class MaxSourcesOverrideApp : public holoscan::Application {
 public:
  std::vector<holoscan::TensorMap> snapshot_frames;
  std::vector<int64_t> max_sources_overrides;
  TensorMapCapture* doa_multi = nullptr;
  ArrayGeometry geometry;
  AngleConvention convention;

  void compose() override {
    using holoscan::Arg;
    auto op = make_operator<RootMUSICEstimatorOp<double>>(
        "root_music", Arg("array_geometry") = geometry,
        Arg("angle_convention") = convention, Arg("num_sources") = int64_t{2},
        Arg("max_sources") = max_sources_overrides.front());
    holoscan::radar::testing::wire_data_source(this, op, "snapshot",
                                               snapshot_frames);
    holoscan::radar::testing::wire_leading_override(this, op, "max_sources",
                                                    max_sources_overrides);
    holoscan::radar::testing::wire_capture_sink(this, op, "doa_multi",
                                                doa_multi);
    wire_noop_output(this, op, "aoa_estimates");
    wire_noop_output(this, op, "source_count");
    wire_noop_output(this, op, "eigenvalues");
  }
};

TEST(RootMUSICEstimatorOpTest, LiveMaxSourcesStickyOverride) {
  constexpr int64_t kN = 12;
  constexpr int64_t kS = 128;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto block =
      make_block_two(geom, conv, 0.2, 0.0, -0.15, 0.0, kS, 0.02, 101u);
  auto snap = block_to_device<double>(block, {1, kN, kS});

  MaxSourcesOverrideApp app;
  app.geometry = geom;
  app.convention = conv;
  app.snapshot_frames = {single_map(snap, "snapshot"),
                         single_map(snap, "snapshot")};
  app.max_sources_overrides = {int64_t{1}, int64_t{3}};

  TensorMapCapture doa_cap;
  app.doa_multi = &doa_cap;
  app.run();

  ASSERT_EQ(doa_cap.frames.size(), 2u);
  EXPECT_EQ(doa_cap.frames[0].shape, (std::vector<int64_t>{1, 1, 2}));
  EXPECT_EQ(doa_cap.frames[1].shape, (std::vector<int64_t>{1, 3, 2}));
}

TEST(RootMUSICEstimatorOpTest, MultipleItemsRecoverIndependentDoa) {
  constexpr int64_t kN = 8;
  constexpr int64_t kS = 48;
  constexpr double kAz0 = 0.11;
  constexpr double kAz1 = -0.07;

  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  const auto b0 = make_block(geom, conv, kAz0, 0.0, kS, 0.02, 21u);
  const auto b1 = make_block(geom, conv, kAz1, 0.0, kS, 0.02, 22u);
  std::vector<std::complex<double>> batch;
  batch.insert(batch.end(), b0.begin(), b0.end());
  batch.insert(batch.end(), b1.begin(), b1.end());
  auto snap = block_to_device<double>(batch, {2, kN, kS});

  const auto out = run_root_music<double>(
      snap, holoscan::Arg("array_geometry") = geom,
      holoscan::Arg("angle_convention") = conv,
      holoscan::Arg("num_sources") = int64_t{1});

  ASSERT_EQ(out.aoa.shape, (std::vector<int64_t>{2, 2}));
  const double* aoa = out.aoa.as<double>();
  EXPECT_NEAR(aoa[0], kAz0, 0.02);
  EXPECT_NEAR(aoa[1], 0.0, 1e-9);
  EXPECT_NEAR(aoa[2], kAz1, 0.02);
  EXPECT_NEAR(aoa[3], 0.0, 1e-9);
}

TEST(RootMUSICEstimatorOpTest, SpatialSmoothing1DRecoversDoa) {
  constexpr int64_t kN = 10;
  constexpr int64_t kS = 64;
  constexpr double kAz0 = 0.30;
  constexpr double kAz1 = -0.30;

  AngleConvention conv;
  conv.preset = AngleConventionPreset::kRadarBoresight;
  const auto geom = make_ula_geometry(kN, 0.5, 1.0);
  const auto a0 = steering_vector(geom, conv, kAz0, 0.0);
  const auto a1 = steering_vector(geom, conv, kAz1, 0.0);
  std::mt19937 rng(70u);
  std::uniform_real_distribution<double> phase(0.0, 2.0 * kPi);
  std::normal_distribution<double> n(0.0, 0.02);
  std::vector<std::complex<double>> block(static_cast<size_t>(kN * kS));
  for (int64_t s = 0; s < kS; ++s) {
    const double ps = phase(rng);
    const std::complex<double> g{std::cos(ps), std::sin(ps)};
    for (int64_t c = 0; c < kN; ++c) {
      std::complex<double> v =
          (a0[static_cast<size_t>(c)] + a1[static_cast<size_t>(c)]) * g;
      v += std::complex<double>(n(rng), n(rng));
      block[static_cast<size_t>(c * kS + s)] = v;
    }
  }
  auto snap = block_to_device<double>(block, {1, kN, kS});

  const auto out = run_root_music<double>(
      snap, holoscan::Arg("array_geometry") = geom,
      holoscan::Arg("angle_convention") = conv,
      holoscan::Arg("spatial_smoothing") = std::vector<int64_t>{4},
      holoscan::Arg("forward_backward") = true,
      holoscan::Arg("num_sources") = int64_t{2},
      holoscan::Arg("max_sources") = int64_t{2});

  ASSERT_EQ(out.eigenvalues.shape, (std::vector<int64_t>{1, 7}));
  EXPECT_EQ(out.source_count.as<int32_t>()[0], 2);

  const double* dm = out.doa_multi.as<double>();
  const double a = dm[0];
  const double b = dm[2];
  const bool matched =
      (std::abs(a - kAz0) < 0.04 && std::abs(b - kAz1) < 0.04) ||
      (std::abs(a - kAz1) < 0.04 && std::abs(b - kAz0) < 0.04);
  EXPECT_TRUE(matched) << "a=" << a << " b=" << b;
}

}  // namespace
