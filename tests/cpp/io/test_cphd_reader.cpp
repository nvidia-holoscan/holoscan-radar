// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// CPHDReaderOp and the host-side CPHD parser behind it, against the
// simulator's mode01_smoke fixture. The fixture is generated into the build
// tree by the `sar_fixtures` CTest setup fixture, never committed
// (docs/planning/SAR/INPUT_DATA.md -> Policy), so these tests run against a
// file produced by the same generator a user would run.
//
// Two layers are covered. The parser tests assert properties the recipe
// guarantees -- a fixed reference point, a uniform pulse cadence, a constant
// frequency axis -- rather than transcribed constants, so regenerating the
// fixture with a different recipe does not silently invalidate them. The
// operator tests cover the batching and boundary-announcement behaviour that
// only appears once the reader is running in a graph.

#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/holoscan.hpp>

#include "io/cphd_reader_op/cphd_cf8_decode.hpp"
#include "io/cphd_reader_op/collection_paths.hpp"
#include "io/cphd_reader_op/cphd_file.hpp"
#include "io/cphd_reader_op/cphd_reader_op.hpp"
#include "radar_common/descriptors/coordinate_frame.hpp"
#include "radar_common/descriptors/enums.hpp"
#include "radar_common/math/constants.hpp"
#include "radar_common/sdk/stream_boundary.hpp"

namespace holoscan::radar {
namespace {

using cphd::CphdFile;

std::string fixture_path() {
  return std::string(SAR_FIXTURE_DIR) + "/mode01_smoke.cphd";
}

// The same collection stored as integer pairs with per-vector scale factors.
// Generated from a recipe identical to mode01_smoke's but for the storage
// section, which is what lets the two be differenced.
std::string fixture_path(const char* recipe) {
  return std::string(SAR_FIXTURE_DIR) + "/" + recipe + ".cphd";
}

// One least-significant bit of a signed integer sample type saturating at
// `saturation`, as a fraction of full scale. Half of it is the ideal rounding
// bound; the whole step is the margin the per-pulse scaling needs, since a
// quiet pulse is quantized against its own peak rather than the collection's.
constexpr double quantization_step(double saturation) {
  return 1.0 / saturation;
}

uint32_t host_bswap32(uint32_t value) {
  return ((value & 0x000000ffU) << 24) | ((value & 0x0000ff00U) << 8) |
         ((value & 0x00ff0000U) >> 8) | ((value & 0xff000000U) >> 24);
}

void store_be_float(char* dest, float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  bits = host_bswap32(bits);
  std::memcpy(dest, &bits, sizeof(bits));
}

struct DeviceBuf {
  void* p{nullptr};
  explicit DeviceBuf(size_t bytes) {
    if (cudaMalloc(&p, bytes) != cudaSuccess) {
      throw std::runtime_error("DeviceBuf: cudaMalloc failed");
    }
  }
  ~DeviceBuf() { cudaFree(p); }
  DeviceBuf(const DeviceBuf&) = delete;
  DeviceBuf& operator=(const DeviceBuf&) = delete;
};

CoordinateFrame ecef_frame() {
  CoordinateFrameSpec spec;
  spec.frame = CoordinateFrameType::kEcef;
  return CoordinateFrame(spec);
}

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------

TEST(CphdFile, ReadsHeaderAndXmlMetadataOfTheSmokeFixture) {
  const CphdFile file(fixture_path());

  EXPECT_EQ(file.version().rfind("1.", 0), 0u) << "expected CPHD 1.x";
  EXPECT_EQ(file.sample_format(), cphd::SampleFormat::kCF8);
  EXPECT_EQ(file.sample_domain(), cphd::SampleDomain::kFX);
  // Paired with the simulator's phase model; flipping one without the other
  // mirrors the image, so the reader surfacing the file's value matters.
  EXPECT_EQ(file.fourier_sign(), -1);
  EXPECT_EQ(file.num_channels(), 1);

  const cphd::ChannelInfo& info = file.channel(0);
  EXPECT_GT(info.num_vectors, 0);
  EXPECT_GT(info.num_samples, 0);
  EXPECT_FALSE(info.identifier.empty());

  EXPECT_TRUE(file.has_pvp_field("TxPos"));
  EXPECT_TRUE(file.has_pvp_field("SRPPos"));
  EXPECT_TRUE(file.has_pvp_field("SC0"));
  // A CF8 collection declares no amplitude scale factors. The recipe leaves
  // the field out rather than writing a constant 1, so that a reader which
  // ignored AmpSF entirely could not pass by reading this file.
  EXPECT_FALSE(file.has_pvp_field("AmpSF"));
  EXPECT_FALSE(file.has_amp_sf());
}

TEST(CphdFile, PvpFieldsMatchTheGeometryTheRecipeDescribes) {
  const CphdFile file(fixture_path());
  const int64_t pulses = file.channel(0).num_vectors;

  // The platform is airborne, so every phase centre sits a few kilometres
  // above the ellipsoid. A frame or byte-order error puts this wildly off.
  const std::vector<double> tx = file.read_pvp(0, "TxPos", 0, pulses, 3);
  for (int64_t pulse = 0; pulse < pulses; ++pulse) {
    const double radius = std::sqrt(tx[pulse * 3 + 0] * tx[pulse * 3 + 0] +
                                    tx[pulse * 3 + 1] * tx[pulse * 3 + 1] +
                                    tx[pulse * 3 + 2] * tx[pulse * 3 + 2]);
    EXPECT_GT(radius, 6.3e6);
    EXPECT_LT(radius, 6.5e6);
  }

  // SRPFixed is true for a spotlight collection staring at the IARP, so the
  // reference point must not move across the aperture.
  const std::vector<double> srp = file.read_pvp(0, "SRPPos", 0, pulses, 3);
  for (int64_t pulse = 1; pulse < pulses; ++pulse) {
    for (int axis = 0; axis < 3; ++axis) {
      EXPECT_DOUBLE_EQ(srp[pulse * 3 + axis], srp[axis]);
    }
  }

  // A uniform pulse cadence, increasing and non-negative: the CPHD schema
  // requires TxTime1 >= 0, which the simulator had to be corrected for.
  const std::vector<double> times = file.read_pvp(0, "TxTime", 0, pulses, 1);
  EXPECT_GE(times[0], 0.0);
  const double interval = times[1] - times[0];
  EXPECT_GT(interval, 0.0);
  for (int64_t pulse = 1; pulse < pulses; ++pulse) {
    EXPECT_NEAR(times[pulse] - times[pulse - 1], interval, 1e-12);
  }

  // FXFixed: one frequency axis serves the whole collection, which is the
  // precondition the operator validates before emitting sample_axis.
  const std::vector<double> sc0 = file.read_pvp(0, "SC0", 0, pulses, 1);
  const std::vector<double> scss = file.read_pvp(0, "SCSS", 0, pulses, 1);
  for (int64_t pulse = 1; pulse < pulses; ++pulse) {
    EXPECT_DOUBLE_EQ(sc0[pulse], sc0[0]);
    EXPECT_DOUBLE_EQ(scss[pulse], scss[0]);
  }
  EXPECT_GT(sc0[0], 1e9);   // X-band start frequency
  EXPECT_GT(scss[0], 0.0);  // positive sample spacing
}

// The regression that catches a wrong stride or a base offset applied twice:
// reading in pieces must give exactly what reading in one go gives.
TEST(CphdFile, ChunkedReadsConcatenateToTheWholeRead) {
  const CphdFile file(fixture_path());
  const int64_t pulses = file.channel(0).num_vectors;
  const int64_t samples = file.channel(0).num_samples;

  const std::vector<std::complex<float>> whole =
      file.read_signal(0, 0, pulses);
  ASSERT_EQ(whole.size(), static_cast<size_t>(pulses * samples));

  std::vector<std::complex<float>> pieced;
  constexpr int64_t kChunk = 37;  // deliberately not a divisor of the count
  for (int64_t first = 0; first < pulses; first += kChunk) {
    const int64_t count = std::min(kChunk, pulses - first);
    const std::vector<std::complex<float>> chunk =
        file.read_signal(0, first, count);
    pieced.insert(pieced.end(), chunk.begin(), chunk.end());
  }
  ASSERT_EQ(pieced.size(), whole.size());
  for (size_t i = 0; i < whole.size(); ++i) {
    ASSERT_EQ(pieced[i], whole[i]) << "sample " << i;
  }

  const std::vector<double> whole_pvp = file.read_pvp(0, "TxPos", 0, pulses, 3);
  const std::vector<double> tail_pvp =
      file.read_pvp(0, "TxPos", pulses - 5, 5, 3);
  for (size_t i = 0; i < tail_pvp.size(); ++i) {
    EXPECT_DOUBLE_EQ(tail_pvp[i], whole_pvp[whole_pvp.size() - 15 + i]);
  }
}

// The block form is what the operator uses, so it has to agree field for
// field with the single-field form the other tests assert against.
TEST(CphdFile, PvpBlockAgreesWithSingleFieldReads) {
  const CphdFile file(fixture_path());
  const int64_t first = 11;
  const int64_t count = 29;
  const cphd::PvpBlock block = file.read_pvp_block(0, first, count);

  EXPECT_EQ(block.count(), count);
  EXPECT_TRUE(block.has("TxPos"));
  EXPECT_FALSE(block.has("NotAField"));

  for (const auto& [name, components] :
       std::vector<std::pair<std::string, int64_t>>{
           {"TxPos", 3}, {"RcvPos", 3}, {"SRPPos", 3},
           {"TxVel", 3}, {"TxTime", 1}, {"SC0", 1},
           {"SCSS", 1}}) {
    const std::vector<double> one_at_a_time =
        file.read_pvp(0, name, first, count, components);
    const std::vector<double> from_block = block.field(name, components);
    ASSERT_EQ(from_block.size(), one_at_a_time.size()) << name;
    for (size_t i = 0; i < from_block.size(); ++i) {
      EXPECT_DOUBLE_EQ(from_block[i], one_at_a_time[i]) << name << "[" << i
                                                        << "]";
    }
  }

  EXPECT_THROW(block.field("NotAField", 1), std::runtime_error);
  EXPECT_THROW(block.field("SC0", 3), std::runtime_error);
  EXPECT_THROW(file.read_pvp_block(0, 0, file.channel(0).num_vectors + 1),
               std::runtime_error);
}

// PVP fields are not all doubles: SIGNAL is declared I8. Decoding it as a
// double reinterprets the bit pattern instead of converting it, turning the
// integer 1 into 5e-324 -- a wrong value that reads as a very small number
// rather than as an error, so nothing downstream would flag it. Caught by
// comparing against sarkit over 218 public collections; pinned here because
// the fixture carries the field too.
TEST(CphdFile, IntegerPvpFieldsDecodeAsIntegersNotBitPatterns) {
  const CphdFile file(fixture_path());
  const cphd::PvpBlock block = file.read_pvp_block(0, 0, 8);
  ASSERT_TRUE(block.has("SIGNAL"));

  const std::vector<double> signal = block.field("SIGNAL", 1);
  ASSERT_EQ(signal.size(), 8U);
  for (size_t i = 0; i < signal.size(); ++i) {
    EXPECT_DOUBLE_EQ(signal[i], 1.0) << "SIGNAL[" << i << "]";
    // The specific failure this guards: the denormal you get from reading
    // the integer 1's bits as a double.
    EXPECT_GT(std::abs(signal[i]), 1e-300) << "SIGNAL[" << i << "]";
  }

  // The float fields must be untouched by the format dispatch.
  const std::vector<double> times = block.field("TxTime", 1);
  const std::vector<double> direct = file.read_pvp(0, "TxTime", 0, 8, 1);
  for (size_t i = 0; i < times.size(); ++i) {
    EXPECT_DOUBLE_EQ(times[i], direct[i]) << "TxTime[" << i << "]";
  }
}

TEST(CphdFile, IarpIsDeclaredInEcefMetres) {
  const CphdFile file(fixture_path());
  ASSERT_TRUE(file.has_iarp());
  const std::array<double, 3>& iarp = file.iarp_ecef();
  const double radius = std::hypot(iarp[0], iarp[1], iarp[2]);
  // WGS-84 surface ~6371 km; the smoke IARP is Atlanta HAE 313 m.
  EXPECT_GT(radius, 6.3e6);
  EXPECT_LT(radius, 6.5e6);
  EXPECT_GT(iarp[2], 3.0e6);
}

TEST(CphdFile, ImageAreaIsTheDeclaredSceneRectangle) {
  const CphdFile file(fixture_path());
  ASSERT_TRUE(file.has_image_area());
  const auto& area = file.image_area();
  // tools/sar_sim/recipes/mode01_smoke.yaml image_half_extent_m: 20
  EXPECT_DOUBLE_EQ(area.x1, -20.0);
  EXPECT_DOUBLE_EQ(area.y1, -20.0);
  EXPECT_DOUBLE_EQ(area.x2, 20.0);
  EXPECT_DOUBLE_EQ(area.y2, 20.0);
  const double nx =
      std::hypot(area.u_iax[0], area.u_iax[1], area.u_iax[2]);
  const double ny =
      std::hypot(area.u_iay[0], area.u_iay[1], area.u_iay[2]);
  EXPECT_NEAR(nx, 1.0, 1e-12);
  EXPECT_NEAR(ny, 1.0, 1e-12);
  const double dot = area.u_iax[0] * area.u_iay[0] +
                     area.u_iax[1] * area.u_iay[1] +
                     area.u_iax[2] * area.u_iay[2];
  EXPECT_NEAR(dot, 0.0, 1e-12);
}

TEST(CphdFile, XySquareFitsTheRangeAzimuthRectangle) {
  EXPECT_DOUBLE_EQ(
      CphdFile::xy_square_in_range_azimuth(1000.0, 2000.0, 0.0),
      1000.0);
  EXPECT_DOUBLE_EQ(
      CphdFile::xy_square_in_range_azimuth(1000.0, 2000.0,
                                          kHalfPi),
      1000.0);
  EXPECT_NEAR(
      CphdFile::xy_square_in_range_azimuth(1000.0, 2000.0,
                                          kPi / 4.0),
      1000.0 / std::sqrt(2.0), 1e-12);
  EXPECT_THROW(CphdFile::xy_square_in_range_azimuth(0.0, 2000.0,
                                                   0.0),
               std::runtime_error);
}

TEST(CphdFile, SmokeFixtureHasNoCollectionPhysics) {
  const CphdFile file(fixture_path());
  EXPECT_FALSE(file.has_collection_physics());
  EXPECT_THROW(file.collection_physics(), std::runtime_error);
}

TEST(CphdFile, PublicSnapshotsMatchTheMode5PhysicsTable) {
  struct Row {
    const char* file;
    double heading_deg;
    double square_km;
    double gsd_m;
    int64_t native_count;
  };
  const Row rows[] = {
      {"2023-09-12-02-36-33_UMBRA-04_CPHD.cphd", 122.3, 4.75,
       0.831, 5715},
      {"2023-07-14-23-20-06_UMBRA-05_CPHD.cphd", 59.6, 5.14, 0.689,
       7463},
      {"2023-10-20-04-22-56_UMBRA-04_CPHD.cphd", 103.6, 5.67, 0.656,
       8643},
      {"2023-02-22-12-54-56_UMBRA-04_CPHD.cphd", 140.1, 4.43, 0.620,
       7150},
      {"2023-02-08-02-11-21_UMBRA-05_CPHD.cphd", 255.4, 4.77, 0.521,
       9162},
      {"2023-10-11-02-20-47_UMBRA-05_CPHD.cphd", 118.2, 4.44, 0.375,
       11848},
  };
  const std::string dest = "/tmp/sar/mode5/";
  try {
    CphdFile probe(dest + rows[0].file);
    (void)probe;
  } catch (const std::runtime_error&) {
    GTEST_SKIP() << "mode-5 CPHDs not at /tmp/sar/mode5";
  }
  for (const Row& row : rows) {
    const CphdFile file(dest + row.file);
    ASSERT_TRUE(file.has_collection_physics()) << row.file;
    const auto phys = file.collection_physics();
    EXPECT_NEAR(phys.heading_cw_rad * 180.0 / kPi, row.heading_deg,
                0.15)
        << row.file;
    EXPECT_NEAR(phys.xy_square_m / 1000.0, row.square_km, 0.05)
        << row.file;
    EXPECT_NEAR(phys.gsd_m, row.gsd_m, 0.005) << row.file;
    EXPECT_NEAR(static_cast<double>(phys.native_count),
                static_cast<double>(row.native_count), 2.0)
        << row.file;
  }
}

TEST(CphdFile, SignalSamplesAreFiniteAndCarryEnergy) {
  const CphdFile file(fixture_path());
  const std::vector<std::complex<float>> signal = file.read_signal(0, 0, 4);

  double power = 0.0;
  for (const std::complex<float>& sample : signal) {
    ASSERT_TRUE(std::isfinite(sample.real()));
    ASSERT_TRUE(std::isfinite(sample.imag()));
    power += std::norm(sample);
  }
  // Byte-swapping is the failure this guards: big-endian floats read as
  // little-endian come out as denormals or enormous values, not as a
  // sensible per-pulse power.
  EXPECT_GT(power, 0.0);
  EXPECT_LT(power / static_cast<double>(signal.size()), 1e6);
}

// The two precisions take genuinely different code paths — a float instance
// byte-swaps in place in the caller's buffer, a double instance widens
// through a scratch buffer — so they have to be pinned against each other.
// Widening float32 to double is exact, hence EXPECT_EQ rather than a
// tolerance: any difference here is a bug in one of the two paths, not
// rounding.
TEST(CphdFile, SinglePrecisionAndDoublePrecisionReadsAgreeExactly) {
  const CphdFile file(fixture_path());
  const int64_t count = 9;
  const int64_t samples = file.channel(0).num_samples;
  const size_t total = static_cast<size_t>(count * samples);

  std::vector<std::complex<float>> single(total);
  std::vector<std::complex<double>> twin(total);
  file.read_signal_into(0, 3, count, single.data());
  file.read_signal_into(0, 3, count, twin.data());

  for (size_t i = 0; i < total; ++i) {
    ASSERT_EQ(static_cast<double>(single[i].real()), twin[i].real()) << i;
    ASSERT_EQ(static_cast<double>(single[i].imag()), twin[i].imag()) << i;
  }

  // And the allocating wrapper agrees with the in-place form it wraps.
  const std::vector<std::complex<float>> wrapped =
      file.read_signal(0, 3, count);
  ASSERT_EQ(wrapped.size(), single.size());
  for (size_t i = 0; i < total; ++i) {
    ASSERT_EQ(wrapped[i], single[i]) << i;
  }
}

// Stored bytes plus the GPU decode must match the host widen: that is
// the operator's CF8 path, and float32→double is exact, so EXPECT_EQ.
TEST(CphdFile, ReadSignalBytesGpuDecodeMatchesHostWiden) {
  const CphdFile file(fixture_path());
  const int64_t count = 5;
  const int64_t samples = file.channel(0).num_samples;
  const size_t nsamp = static_cast<size_t>(count * samples);
  const size_t bytes = nsamp * 8;

  std::vector<char> raw(bytes);
  file.read_signal_bytes(0, 2, count, raw.data(), raw.size());

  std::vector<std::complex<double>> host(nsamp);
  file.read_signal_into(0, 2, count, host.data());

  DeviceBuf in(bytes);
  DeviceBuf out(nsamp * sizeof(std::complex<double>));
  ASSERT_EQ(cudaMemcpy(in.p, raw.data(), bytes, cudaMemcpyHostToDevice),
            cudaSuccess);
  ASSERT_EQ(cphd_decode_cf8_f64(in.p, out.p, nsamp, samples, nullptr,
                                cudaStreamDefault),
            cudaSuccess);
  std::vector<std::complex<double>> gpu(nsamp);
  ASSERT_EQ(cudaMemcpy(gpu.data(), out.p, nsamp * sizeof(gpu[0]),
                       cudaMemcpyDeviceToHost),
            cudaSuccess);
  ASSERT_EQ(gpu.size(), host.size());
  for (size_t i = 0; i < nsamp; ++i) {
    ASSERT_EQ(gpu[i], host[i]) << i;
  }

  std::vector<char> too_small(bytes - 1);
  EXPECT_THROW(file.read_signal_bytes(0, 2, count, too_small.data(),
                                      too_small.size()),
               std::runtime_error);
}

// AmpSF is applied as T(amp) on the host; the GPU kernels must do the
// same, not a double multiply rounded back to float.
TEST(CphdCf8Decode, AmpSfMatchesHostArithmetic) {
  constexpr int64_t kSamples = 2;
  constexpr int64_t kPulses = 2;
  constexpr size_t kNsamp = 4;
  const float pair[8] = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F};
  std::vector<char> stored(kNsamp * 8);
  for (size_t i = 0; i < 8; ++i) {
    store_be_float(stored.data() + i * sizeof(float), pair[i]);
  }
  const std::vector<double> amp{2.0, 0.25};

  std::vector<std::complex<float>> host_f(kNsamp);
  std::vector<std::complex<double>> host_d(kNsamp);
  for (size_t i = 0; i < kNsamp; ++i) {
    const std::complex<float> z(pair[2 * i], pair[2 * i + 1]);
    const size_t pulse = i / static_cast<size_t>(kSamples);
    host_f[i] = z * static_cast<float>(amp[pulse]);
    host_d[i] = std::complex<double>(pair[2 * i], pair[2 * i + 1]) *
                amp[pulse];
  }

  DeviceBuf in(stored.size());
  DeviceBuf amp_dev(amp.size() * sizeof(double));
  DeviceBuf out_f(kNsamp * sizeof(std::complex<float>));
  DeviceBuf out_d(kNsamp * sizeof(std::complex<double>));
  ASSERT_EQ(cudaMemcpy(in.p, stored.data(), stored.size(),
                       cudaMemcpyHostToDevice),
            cudaSuccess);
  ASSERT_EQ(cudaMemcpy(amp_dev.p, amp.data(), amp.size() * sizeof(double),
                       cudaMemcpyHostToDevice),
            cudaSuccess);
  ASSERT_EQ(cphd_decode_cf8_f32(in.p, out_f.p, kNsamp, kSamples,
                                static_cast<const double*>(amp_dev.p),
                                cudaStreamDefault),
            cudaSuccess);
  ASSERT_EQ(cphd_decode_cf8_f64(in.p, out_d.p, kNsamp, kSamples,
                                static_cast<const double*>(amp_dev.p),
                                cudaStreamDefault),
            cudaSuccess);
  std::vector<std::complex<float>> gpu_f(kNsamp);
  std::vector<std::complex<double>> gpu_d(kNsamp);
  ASSERT_EQ(cudaMemcpy(gpu_f.data(), out_f.p, kNsamp * sizeof(gpu_f[0]),
                       cudaMemcpyDeviceToHost),
            cudaSuccess);
  ASSERT_EQ(cudaMemcpy(gpu_d.data(), out_d.p, kNsamp * sizeof(gpu_d[0]),
                       cudaMemcpyDeviceToHost),
            cudaSuccess);
  for (size_t i = 0; i < kNsamp; ++i) {
    ASSERT_EQ(gpu_f[i], host_f[i]) << i;
    ASSERT_EQ(gpu_d[i], host_d[i]) << i;
  }
}

// The integer formats carry the same collection as the CF8 fixture, so
// decoding them correctly means reproducing it. That is a far stronger
// assertion than "the samples look plausible": a wrong integer width, a
// missed byte swap, an unsigned read, or an unapplied AmpSF each move the
// result by orders of magnitude, while a correct decode lands inside one
// quantization step. Capella's real CI4 collections check the same paths
// against sarpy, but they are gigabytes and cannot be fetched by a test.
TEST(CphdFile, IntegerFormatsRoundTripToTheFloatCollection) {
  const CphdFile reference(fixture_path());
  const int64_t count = 12;
  const int64_t samples = reference.channel(0).num_samples;
  const std::vector<std::complex<float>> expected =
      reference.read_signal(0, 0, count);

  double peak = 0.0;
  for (const std::complex<float>& sample : expected) {
    peak = std::max(peak, static_cast<double>(std::abs(sample)));
  }
  ASSERT_GT(peak, 0.0);

  // Saturation points of the two stored types, which are the simulator's
  // quantization peaks: int16 for CI4, int8 for CI2.
  for (const auto& [recipe, saturation] :
       std::vector<std::pair<const char*, double>>{{"mode01_ci4", 32767.0},
                                                   {"mode01_ci2", 127.0}}) {
    const double lsb = quantization_step(saturation);
    const CphdFile file(fixture_path(recipe));
    EXPECT_TRUE(file.has_amp_sf()) << recipe;
    EXPECT_EQ(file.channel(0).num_samples, samples) << recipe;

    const std::vector<std::complex<float>> actual = file.read_signal(0, 0,
                                                                     count);
    ASSERT_EQ(actual.size(), expected.size()) << recipe;
    double worst = 0.0;
    for (size_t i = 0; i < actual.size(); ++i) {
      worst = std::max(worst,
                       static_cast<double>(std::abs(actual[i] - expected[i])));
    }
    EXPECT_LT(worst / peak, lsb) << recipe << ": decoded signal is "
                                 << worst / peak
                                 << " of full scale away from the CF8 "
                                    "collection, more than one quantization "
                                    "step";
  }
}

// Reading a scaled collection without the parameter block it is scaled by
// would return samples smaller by a factor of AmpSF -- smooth, plausible, and
// wrong. The signature makes that unrepresentable rather than unlikely.
TEST(CphdFile, ReadingAScaledCollectionWithoutItsScaleFactorsIsRefused) {
  const CphdFile file(fixture_path("mode01_ci4"));
  ASSERT_TRUE(file.has_amp_sf());

  std::vector<std::complex<float>> samples(
      static_cast<size_t>(file.channel(0).num_samples * 4));
  EXPECT_THROW(file.read_signal_into(0, 0, 4, samples.data()),
               std::runtime_error);

  // A block covering the wrong vectors is the subtler version of the same
  // mistake, and is refused for the same reason.
  const cphd::PvpBlock mismatched = file.read_pvp_block(0, 0, 3);
  EXPECT_THROW(file.read_signal_into(0, 0, 4, samples.data(), &mismatched),
               std::runtime_error);

  const cphd::PvpBlock matching = file.read_pvp_block(0, 0, 4);
  EXPECT_NO_THROW(file.read_signal_into(0, 0, 4, samples.data(), &matching));
}

TEST(CphdFile, MalformedRequestsFailWithAClearError) {
  EXPECT_THROW(CphdFile("/nonexistent/path/to.cphd"), std::runtime_error);
  // A file that is not CPHD at all must be rejected on its first line rather
  // than read as garbage.
  EXPECT_THROW(
      CphdFile(std::string(SAR_FIXTURE_DIR) + "/mode01_smoke_truth.json"),
      std::runtime_error);

  const CphdFile file(fixture_path());
  const int64_t pulses = file.channel(0).num_vectors;
  EXPECT_THROW(file.read_signal(0, 0, pulses + 1), std::runtime_error);
  EXPECT_THROW(file.read_signal(0, -1, 4), std::runtime_error);
  EXPECT_THROW(file.read_pvp(0, "NotAField", 0, 1, 1), std::runtime_error);
  EXPECT_THROW(file.channel(1), std::runtime_error);
  // Asking for more words than the field holds is a caller bug, not a read
  // that should quietly spill into the neighbouring field.
  EXPECT_THROW(file.read_pvp(0, "SC0", 0, 1, 3), std::runtime_error);
}

// ---------------------------------------------------------------------------
// Operator
// ---------------------------------------------------------------------------

// Records one entry per tick: the pulse count, the samples, and whichever
// boundary value the tick's metadata carried.
struct ReaderCapture {
  std::vector<int64_t> pulse_counts;
  std::vector<std::string> boundaries;
  std::vector<std::complex<float>> samples;
  std::vector<float> positions;
  std::vector<float> reference_positions;
  std::vector<float> velocities;
  std::vector<double> pulse_times;
  std::vector<float> sample_axis;  // last tick's copy
  std::vector<CoordinateFrame> frames;
  DLDataType pulse_time_dtype{};
  int64_t num_samples{0};
};

// Copy a device tensor's contents back to the host.
template <typename Element>
std::vector<Element> download(const std::shared_ptr<holoscan::Tensor>& tensor) {
  int64_t count = 1;
  for (const int64_t extent : tensor->shape()) {
    count *= extent;
  }
  std::vector<Element> host(static_cast<size_t>(count));
  const cudaError_t status =
      cudaMemcpy(host.data(), tensor->data(), sizeof(Element) * host.size(),
                 cudaMemcpyDeviceToHost);
  EXPECT_EQ(status, cudaSuccess);
  return host;
}

class CaptureSinkOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(CaptureSinkOp)
  CaptureSinkOp() = default;

  ReaderCapture* capture{nullptr};

  // All three required outputs are consumed. A source's required outputs
  // gate its scheduling, so leaving one unconnected stops the graph before
  // the first tick rather than merely discarding a port.
  //
  // The optional ones are kNone on this side too: coordinate_frame arrives
  // only on the first tick, so a default condition would block every tick
  // after it.
  void setup(holoscan::OperatorSpec& spec) override {
    spec.input<holoscan::TensorMap>("phase_history");
    spec.input<holoscan::TensorMap>("positions");
    spec.input<holoscan::TensorMap>("reference_position");
    spec.input<holoscan::TensorMap>("sample_axis")
        .condition(holoscan::ConditionType::kNone);
    spec.input<holoscan::TensorMap>("pulse_times")
        .condition(holoscan::ConditionType::kNone);
    spec.input<holoscan::TensorMap>("velocities")
        .condition(holoscan::ConditionType::kNone);
    spec.input<CoordinateFrame>("coordinate_frame")
        .condition(holoscan::ConditionType::kNone);
  }

  void compute(holoscan::InputContext& op_input,
               [[maybe_unused]] holoscan::OutputContext& op_output,
               [[maybe_unused]] holoscan::ExecutionContext& context) override {
    auto history =
        op_input.receive<holoscan::TensorMap>("phase_history").value();
    auto positions = op_input.receive<holoscan::TensorMap>("positions").value();
    auto reference =
        op_input.receive<holoscan::TensorMap>("reference_position").value();

    const std::shared_ptr<holoscan::Tensor> tensor =
        history.begin()->second;
    capture->pulse_counts.push_back(tensor->shape()[0]);
    capture->num_samples = tensor->shape()[1];

    const std::vector<std::complex<float>> samples =
        download<std::complex<float>>(tensor);
    capture->samples.insert(capture->samples.end(), samples.begin(),
                            samples.end());

    const std::vector<float> apc = download<float>(positions.begin()->second);
    capture->positions.insert(capture->positions.end(), apc.begin(),
                              apc.end());
    const std::vector<float> srp = download<float>(reference.begin()->second);
    capture->reference_positions.insert(capture->reference_positions.end(),
                                        srp.begin(), srp.end());

    auto axis = op_input.receive<holoscan::TensorMap>("sample_axis");
    if (axis && !axis.value().empty()) {
      capture->sample_axis = download<float>(axis.value().begin()->second);
    }

    auto times = op_input.receive<holoscan::TensorMap>("pulse_times");
    if (times && !times.value().empty()) {
      const std::shared_ptr<holoscan::Tensor> t = times.value().begin()->second;
      capture->pulse_time_dtype = t->dtype();
      const std::vector<double> host = download<double>(t);
      capture->pulse_times.insert(capture->pulse_times.end(), host.begin(),
                                  host.end());
    }

    auto velocity = op_input.receive<holoscan::TensorMap>("velocities");
    if (velocity && !velocity.value().empty()) {
      const std::vector<float> host =
          download<float>(velocity.value().begin()->second);
      capture->velocities.insert(capture->velocities.end(), host.begin(),
                                 host.end());
    }

    // The descriptor edge, not a tensor: emitted once, on the first tick.
    auto frame = op_input.receive<CoordinateFrame>("coordinate_frame");
    if (frame) {
      capture->frames.push_back(frame.value());
    }

    capture->boundaries.push_back(metadata()->get<std::string>(
        kStreamBoundaryKey, std::string()));
  }
};

struct ReaderOptions {
  bool loop{false};
  double playback_rate{0.0};
  double duty_cycle{1.0};
  int64_t max_ticks{0};
};

class ReaderApp : public holoscan::Application {
 public:
  ReaderApp(ReaderCapture* capture, int64_t pulses_per_tick,
            std::vector<int64_t> pulse_range,
            CollectionPaths paths = CollectionPaths(fixture_path()),
            ReaderOptions options = {})
      : capture_(capture),
        pulses_per_tick_(pulses_per_tick),
        pulse_range_(std::move(pulse_range)),
        paths_(std::move(paths)),
        options_(options) {}

  void compose() override {
    holoscan::ArgList args{holoscan::Arg("path", paths_),
                           holoscan::Arg("coordinate_frame", ecef_frame()),
                           holoscan::Arg("pulses_per_tick", pulses_per_tick_),
                           holoscan::Arg("loop", options_.loop),
                           holoscan::Arg("playback_rate",
                                        options_.playback_rate),
                           holoscan::Arg("duty_cycle", options_.duty_cycle)};
    if (!pulse_range_.empty()) {
      args.add(holoscan::Arg("pulse_range", pulse_range_));
    }
    if (options_.max_ticks > 0) {
      auto reader = make_operator<CPHDReaderOp<float>>(
          "reader", args,
          make_condition<holoscan::CountCondition>("tick_limit",
                                                  options_.max_ticks));
      reader_op = reader;
      wire_sink(reader);
      return;
    }
    auto reader = make_operator<CPHDReaderOp<float>>("reader", args);
    reader_op = reader;
    wire_sink(reader);
  }

  std::shared_ptr<CPHDReaderOp<float>> reader_op;

 private:
  template <typename Reader>
  void wire_sink(Reader reader) {
    auto sink = make_operator<CaptureSinkOp>("sink");
    sink->capture = capture_;
    sink->metadata_policy(holoscan::MetadataPolicy::kUpdate);
    add_flow(reader, sink,
             {{"phase_history", "phase_history"},
              {"positions", "positions"},
              {"reference_position", "reference_position"},
              {"sample_axis", "sample_axis"},
              {"pulse_times", "pulse_times"},
              {"velocities", "velocities"},
              {"coordinate_frame", "coordinate_frame"}});
  }

  ReaderCapture* capture_;
  int64_t pulses_per_tick_;
  std::vector<int64_t> pulse_range_;
  CollectionPaths paths_;
  ReaderOptions options_;
};

TEST(CPHDReaderOp, BatchingCoversTheCollectionWithAShortFinalTick) {
  const CphdFile file(fixture_path());
  const int64_t pulses = file.channel(0).num_vectors;
  const int64_t samples = file.channel(0).num_samples;
  // Deliberately not a divisor: the final batch must be short, which is the
  // regression that catches a stage reading pulses_per_tick instead of the
  // tensor.
  const int64_t per_tick = 48;
  ASSERT_NE(pulses % per_tick, 0);

  ReaderCapture capture;
  ReaderApp(&capture, per_tick, {}).run();

  const int64_t expected_ticks = (pulses + per_tick - 1) / per_tick;
  ASSERT_EQ(static_cast<int64_t>(capture.pulse_counts.size()), expected_ticks);
  EXPECT_EQ(capture.num_samples, samples);

  int64_t total = 0;
  for (size_t tick = 0; tick + 1 < capture.pulse_counts.size(); ++tick) {
    EXPECT_EQ(capture.pulse_counts[tick], per_tick);
    total += capture.pulse_counts[tick];
  }
  EXPECT_EQ(capture.pulse_counts.back(), pulses - total);
  EXPECT_LT(capture.pulse_counts.back(), per_tick);

  // Nothing dropped or duplicated at a batch boundary: the concatenation of
  // every tick is exactly the file, in order.
  const std::vector<std::complex<float>> expected =
      file.read_signal(0, 0, pulses);
  ASSERT_EQ(capture.samples.size(), expected.size());
  for (size_t i = 0; i < expected.size(); ++i) {
    ASSERT_EQ(capture.samples[i], expected[i]) << "sample " << i;
  }

  // The emitted geometry is the file's own record, not a re-derivation. This
  // collection is monostatic, so the transmit/receive midpoint the reader
  // takes as the phase centre coincides with TxPos.
  const std::vector<double> tx = file.read_pvp(0, "TxPos", 0, pulses, 3);
  ASSERT_EQ(capture.positions.size(), tx.size());
  for (size_t i = 0; i < tx.size(); ++i) {
    EXPECT_NEAR(capture.positions[i], static_cast<float>(tx[i]),
                std::abs(tx[i]) * 1e-6);
  }
  const std::vector<double> srp = file.read_pvp(0, "SRPPos", 0, pulses, 3);
  ASSERT_EQ(capture.reference_positions.size(), srp.size());
  for (size_t i = 0; i < srp.size(); ++i) {
    EXPECT_NEAR(capture.reference_positions[i], static_cast<float>(srp[i]),
                std::abs(srp[i]) * 1e-6);
  }
}

TEST(CPHDReaderOp, LastMeanApcIsTheLastBatchMidpoint) {
  const CphdFile file(fixture_path());
  ReaderCapture capture;
  ReaderOptions options;
  options.max_ticks = 1;
  const int64_t per_tick = 48;
  ReaderApp app(&capture, per_tick, {}, CollectionPaths(fixture_path()),
                options);
  app.run();
  ASSERT_NE(app.reader_op, nullptr);
  const auto apc = app.reader_op->last_mean_apc();
  ASSERT_TRUE(apc.valid);
  EXPECT_EQ(apc.file_index, 0);
  const std::vector<double> tx =
      file.read_pvp(0, "TxPos", 0, per_tick, 3);
  const std::vector<double> rcv =
      file.read_pvp(0, "RcvPos", 0, per_tick, 3);
  double sx = 0.0;
  double sy = 0.0;
  double sz = 0.0;
  for (int64_t i = 0; i < per_tick; ++i) {
    const size_t b = static_cast<size_t>(i) * 3;
    sx += 0.5 * (tx[b] + rcv[b]);
    sy += 0.5 * (tx[b + 1] + rcv[b + 1]);
    sz += 0.5 * (tx[b + 2] + rcv[b + 2]);
  }
  const double inv = 1.0 / static_cast<double>(per_tick);
  EXPECT_DOUBLE_EQ(apc.ecef[0], sx * inv);
  EXPECT_DOUBLE_EQ(apc.ecef[1], sy * inv);
  EXPECT_DOUBLE_EQ(apc.ecef[2], sz * inv);
}

// CPHDReaderOp<double> on CF8 is the mode-5 path: GPU widen, not the host
// scalar loop. Pin it against read_signal_into; float32→double is exact.
class DoubleHistorySinkOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(DoubleHistorySinkOp)
  DoubleHistorySinkOp() = default;

  std::vector<std::complex<double>>* samples{nullptr};

  void setup(holoscan::OperatorSpec& spec) override {
    spec.input<holoscan::TensorMap>("phase_history");
    spec.input<holoscan::TensorMap>("positions");
    spec.input<holoscan::TensorMap>("reference_position");
  }

  void compute(holoscan::InputContext& op_input,
               [[maybe_unused]] holoscan::OutputContext& op_output,
               [[maybe_unused]] holoscan::ExecutionContext& context) override {
    auto history =
        op_input.receive<holoscan::TensorMap>("phase_history").value();
    (void)op_input.receive<holoscan::TensorMap>("positions");
    (void)op_input.receive<holoscan::TensorMap>("reference_position");
    const auto tick =
        download<std::complex<double>>(history.begin()->second);
    samples->insert(samples->end(), tick.begin(), tick.end());
  }
};

class DoubleReaderApp : public holoscan::Application {
 public:
  DoubleReaderApp(std::vector<std::complex<double>>* samples,
                  int64_t pulses_per_tick)
      : samples_(samples), pulses_per_tick_(pulses_per_tick) {}

  void compose() override {
    auto reader = make_operator<CPHDReaderOp<double>>(
        "reader", holoscan::Arg("path", CollectionPaths(fixture_path())),
        holoscan::Arg("coordinate_frame", ecef_frame()),
        holoscan::Arg("pulses_per_tick", pulses_per_tick_));
    auto sink = make_operator<DoubleHistorySinkOp>("sink");
    sink->samples = samples_;
    sink->metadata_policy(holoscan::MetadataPolicy::kUpdate);
    add_flow(reader, sink,
             {{"phase_history", "phase_history"},
              {"positions", "positions"},
              {"reference_position", "reference_position"}});
  }

 private:
  std::vector<std::complex<double>>* samples_;
  int64_t pulses_per_tick_;
};

TEST(CPHDReaderOp, DoubleCf8DecodeMatchesHostWiden) {
  const CphdFile file(fixture_path());
  const int64_t pulses = file.channel(0).num_vectors;
  const size_t nsamp =
      static_cast<size_t>(pulses * file.channel(0).num_samples);
  std::vector<std::complex<double>> expected(nsamp);
  file.read_signal_into(0, 0, pulses, expected.data());

  std::vector<std::complex<double>> got;
  DoubleReaderApp(&got, 48).run();
  ASSERT_EQ(got.size(), expected.size());
  for (size_t i = 0; i < expected.size(); ++i) {
    ASSERT_EQ(got[i], expected[i]) << "sample " << i;
  }
}

// The fast-time axis the file declares, rebuilt from SC0 and SCSS. It is
// deliberately *not* the compressed range axis: two fast-time axes exist and
// only one of them is range_bins, so a test that treated this as a range axis
// would be asserting the confusion the contract exists to prevent.
TEST(CPHDReaderOp, SampleAxisIsTheFilesDeclaredFrequencyAxis) {
  const CphdFile file(fixture_path());
  const int64_t samples = file.channel(0).num_samples;
  const double sc0 = file.read_pvp(0, "SC0", 0, 1, 1)[0];
  const double scss = file.read_pvp(0, "SCSS", 0, 1, 1)[0];

  ReaderCapture capture;
  ReaderApp(&capture, 48, {}).run();

  ASSERT_EQ(static_cast<int64_t>(capture.sample_axis.size()), samples);
  EXPECT_NEAR(capture.sample_axis[0], static_cast<float>(sc0), sc0 * 1e-6);
  for (int64_t sample = 0; sample < samples; ++sample) {
    const double expected = sc0 + static_cast<double>(sample) * scss;
    EXPECT_NEAR(capture.sample_axis[static_cast<size_t>(sample)],
                static_cast<float>(expected), expected * 1e-6)
        << "sample " << sample;
  }
  // An FX-domain collection: the axis is a frequency in Hz, not a delay.
  EXPECT_GT(capture.sample_axis.back(), capture.sample_axis.front());
}

TEST(CPHDReaderOp, PulseTimesStayDoubleInASinglePrecisionInstance) {
  const CphdFile file(fixture_path());
  const int64_t pulses = file.channel(0).num_vectors;
  const std::vector<double> expected =
      file.read_pvp(0, "TxTime", 0, pulses, 1);

  ReaderCapture capture;
  ReaderApp(&capture, 48, {}).run();

  // The reader is instantiated as CPHDReaderOp<float>, and pulse_times is
  // still float64: a collection-relative timestamp does not survive single
  // precision to useful accuracy, which is why the contract pins the dtype
  // rather than letting it follow the instantiation.
  EXPECT_EQ(capture.pulse_time_dtype.code, kDLFloat);
  EXPECT_EQ(capture.pulse_time_dtype.bits, 64);

  ASSERT_EQ(capture.pulse_times.size(), expected.size());
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_DOUBLE_EQ(capture.pulse_times[i], expected[i]) << "pulse " << i;
  }
}

TEST(CPHDReaderOp, VelocitiesFollowTheFilesPhaseCentreRecord) {
  const CphdFile file(fixture_path());
  const int64_t pulses = file.channel(0).num_vectors;
  const std::vector<double> expected = file.read_pvp(0, "TxVel", 0, pulses, 3);

  ReaderCapture capture;
  ReaderApp(&capture, 48, {}).run();

  ASSERT_EQ(capture.velocities.size(), expected.size());
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_NEAR(capture.velocities[i], static_cast<float>(expected[i]),
                std::max(std::abs(expected[i]) * 1e-6, 1e-3));
  }
}

// The descriptor output D21 provides, so a consumer can wire the frame into
// its own [live] override instead of mirroring it by hand. Emitted once,
// overrides being sticky, and it must equal the frame the reader validated
// rather than a re-derivation of it.
TEST(CPHDReaderOp, EmitsTheValidatedCoordinateFrameExactlyOnce) {
  ReaderCapture capture;
  ReaderApp(&capture, 48, {}).run();

  ASSERT_GT(capture.pulse_counts.size(), 1u)
      << "need several ticks for 'once' to mean anything";
  ASSERT_EQ(capture.frames.size(), 1u);
  EXPECT_TRUE(capture.frames[0].is_valid());
  EXPECT_EQ(capture.frames[0].frame(), CoordinateFrameType::kEcef);
  EXPECT_TRUE(capture.frames[0].compatible_with(ecef_frame()));
}

TEST(CPHDReaderOp, AnnouncesEndOfStreamOnTheFinalBatchAndNoOther) {
  ReaderCapture capture;
  ReaderApp(&capture, 48, {}).run();

  ASSERT_FALSE(capture.boundaries.empty());
  for (size_t tick = 0; tick + 1 < capture.boundaries.size(); ++tick) {
    EXPECT_EQ(capture.boundaries[tick], "") << "tick " << tick;
  }
  // Announcing on the batch *after* the last is the plausible off-by-one, and
  // it would contaminate the closing aperture, so the assertion is about
  // which batch carries the key rather than that some batch does.
  EXPECT_EQ(capture.boundaries.back(), kStreamBoundaryEndOfStream);
}

TEST(CPHDReaderOp, PulseRangeSelectsTheRequestedSubset) {
  const CphdFile file(fixture_path());
  const int64_t first = 20;
  const int64_t last = 100;

  ReaderCapture capture;
  ReaderApp(&capture, 32, {first, last}).run();

  int64_t total = 0;
  for (const int64_t count : capture.pulse_counts) {
    total += count;
  }
  EXPECT_EQ(total, last - first);

  const std::vector<std::complex<float>> expected =
      file.read_signal(0, first, last - first);
  ASSERT_EQ(capture.samples.size(), expected.size());
  for (size_t i = 0; i < expected.size(); ++i) {
    ASSERT_EQ(capture.samples[i], expected[i]) << "sample " << i;
  }
}

TEST(CPHDReaderOp, RejectsAnOutOfBoundsPulseRange) {
  const CphdFile file(fixture_path());
  const int64_t pulses = file.channel(0).num_vectors;
  ReaderCapture capture;
  EXPECT_THROW(ReaderApp(&capture, 32, {0, pulses + 1}).run(),
               std::runtime_error);
}

// A path naming several files reads them as separate collections, back to
// back. The seam is what makes this more than concatenation: the accumulating
// formation stage downstream must flush and reset there rather than
// integrating two scenes into one image, and the only way it learns of the
// seam is this key.
TEST(CPHDReaderOp, MultipleFilesReadInOrderAndAnnounceEachSeam) {
  const CphdFile first_file(fixture_path());
  const CphdFile second_file(fixture_path("mode01_ci4"));
  const int64_t pulses = first_file.channel(0).num_vectors;
  ASSERT_EQ(second_file.channel(0).num_vectors, pulses);

  // A short window per file keeps the test quick and, more usefully, pins
  // that pulse_range applies per file rather than across the concatenation.
  const int64_t first = 0;
  const int64_t last = 64;
  const int64_t per_tick = 32;

  ReaderCapture capture;
  ReaderApp(&capture, per_tick, {first, last},
            CollectionPaths(std::vector<std::string>{
                fixture_path(), fixture_path("mode01_ci4")}))
      .run();

  int64_t total = 0;
  for (const int64_t count : capture.pulse_counts) {
    total += count;
  }
  EXPECT_EQ(total, 2 * (last - first)) << "pulse_range applies to each file";

  // Two ticks per file, and the boundary lands on the last tick of each: the
  // first seam is a collection boundary, the last is the end of the stream.
  ASSERT_EQ(capture.boundaries.size(), 4u);
  EXPECT_EQ(capture.boundaries[0], "");
  EXPECT_EQ(capture.boundaries[1], kStreamBoundaryCollection);
  EXPECT_EQ(capture.boundaries[2], "");
  EXPECT_EQ(capture.boundaries[3], kStreamBoundaryEndOfStream);

  // The frame descriptor is emitted once per collection, not once per
  // instance: a new file may declare a new frame, and the consumer's
  // override is sticky until replaced.
  EXPECT_EQ(capture.frames.size(), 2u);

  // The data is each file's own, in order, and the second file's samples are
  // its dequantized ones rather than the first file's repeated.
  const std::vector<std::complex<float>> expected_first =
      first_file.read_signal(0, first, last - first);
  const std::vector<std::complex<float>> expected_second =
      second_file.read_signal(0, first, last - first);
  ASSERT_EQ(capture.samples.size(),
            expected_first.size() + expected_second.size());
  for (size_t i = 0; i < expected_first.size(); ++i) {
    ASSERT_EQ(capture.samples[i], expected_first[i]) << "first file, " << i;
  }
  for (size_t i = 0; i < expected_second.size(); ++i) {
    ASSERT_EQ(capture.samples[expected_first.size() + i], expected_second[i])
        << "second file, " << i;
  }
}

// A directory stands in for the list, enumerated lexicographically. The
// fixture directory holds all three generated collections, so this also
// covers the reader opening CI2, CI4 and CF8 in one run.
TEST(CPHDReaderOp, ADirectoryIsReadInLexicographicOrder) {
  ReaderCapture capture;
  ReaderApp(&capture, 64, {0, 64}, CollectionPaths(std::string(
                                       SAR_FIXTURE_DIR)))
      .run();

  // mode01_ci2, mode01_ci4, mode01_smoke -- one tick each at this window.
  ASSERT_EQ(capture.boundaries.size(), 3u);
  EXPECT_EQ(capture.boundaries[0], kStreamBoundaryCollection);
  EXPECT_EQ(capture.boundaries[1], kStreamBoundaryCollection);
  EXPECT_EQ(capture.boundaries[2], kStreamBoundaryEndOfStream);

  const CphdFile ci2(fixture_path("mode01_ci2"));
  const std::vector<std::complex<float>> expected = ci2.read_signal(0, 0, 64);
  ASSERT_GE(capture.samples.size(), expected.size());
  for (size_t i = 0; i < expected.size(); ++i) {
    ASSERT_EQ(capture.samples[i], expected[i])
        << "lexicographically first file, sample " << i;
  }
}

TEST(CPHDReaderOp, RejectsAPathThatSelectsNothing) {
  ReaderCapture capture;
  EXPECT_THROW(ReaderApp(&capture, 32, {},
                         CollectionPaths(std::string("/nonexistent/dir")))
                   .run(),
               std::runtime_error);
}

// A frame disagreeing with the file is an error, not a silent
// reinterpretation of the same numbers under a different meaning.
TEST(CPHDReaderOp, RejectsACoordinateFrameTheFileDoesNotDeclare) {
  class MismatchedApp : public holoscan::Application {
   public:
    void compose() override {
      CoordinateFrameSpec spec;
      spec.frame = CoordinateFrameType::kEnu;
      spec.origin_llh = std::array<double, 3>{33.64, -84.43, 300.0};
      auto reader = make_operator<CPHDReaderOp<float>>(
          "reader", holoscan::Arg("path", CollectionPaths(fixture_path())),
          holoscan::Arg("coordinate_frame", CoordinateFrame(spec)),
          holoscan::Arg("pulses_per_tick", static_cast<int64_t>(32)));
      add_operator(reader);
    }
  };
  EXPECT_THROW(MismatchedApp().run(), std::runtime_error);
}

// loop wraps last→first. Every seam is collection_boundary; the graph
// never emits end_of_stream. CountCondition caps the otherwise infinite
// run at four ticks (two files, one tick each, two full cycles).
TEST(CPHDReaderOp, LoopWrapsWithoutEndOfStream) {
  ReaderCapture capture;
  ReaderOptions options;
  options.loop = true;
  options.max_ticks = 4;
  ReaderApp(&capture, 256, {},
            CollectionPaths(std::vector<std::string>{
                fixture_path(), fixture_path("mode01_ci4")}),
            options)
      .run();

  ASSERT_EQ(capture.boundaries.size(), 4u);
  for (const auto& boundary : capture.boundaries) {
    EXPECT_EQ(boundary, kStreamBoundaryCollection);
  }
}

// playback_rate 1 sleeps collect − wall (catch-up). Smoke is 256
// pulses at 2 ms CRI ≈ 0.51 s; one tick of 256 pulses.
TEST(CPHDReaderOp, PlaybackRateOneMatchesCollectTime) {
  ReaderCapture capture;
  ReaderOptions options;
  options.playback_rate = 1.0;
  const auto t0 = std::chrono::steady_clock::now();
  ReaderApp(&capture, 256, {}, CollectionPaths(fixture_path()), options)
      .run();
  const double wall_s =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
          .count();
  ASSERT_EQ(capture.pulse_times.size(), 256u);
  const double span =
      capture.pulse_times.back() - capture.pulse_times.front();
  EXPECT_GT(span, 0.4);
  EXPECT_GE(wall_s, span * 0.9);
}

// duty_cycle < 1 inserts a cooldown after a collection when another
// follows. Two one-tick files with D=0.5 add roughly one collect-span of
// sleep between them.
TEST(CPHDReaderOp, DutyCycleInsertsCooldownBetweenFiles) {
  ReaderCapture capture;
  ReaderOptions options;
  options.duty_cycle = 0.5;
  const auto t0 = std::chrono::steady_clock::now();
  ReaderApp(&capture, 256, {},
            CollectionPaths(std::vector<std::string>{
                fixture_path(), fixture_path("mode01_ci4")}),
            options)
      .run();
  const double wall_s =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
          .count();
  ASSERT_EQ(capture.boundaries.size(), 2u);
  EXPECT_EQ(capture.boundaries[0], kStreamBoundaryCollection);
  EXPECT_EQ(capture.boundaries[1], kStreamBoundaryEndOfStream);
  ASSERT_GE(capture.pulse_times.size(), 2u);
  const double span =
      capture.pulse_times.back() - capture.pulse_times.front();
  // CaptureSink appends both files' TxTime (both recipes start near 0,
  // so the span is one collect). Cooldown is ~that span at D=0.5.
  EXPECT_GE(wall_s, span * 0.9);
}

// A single file with loop false never cooldowns, even at D<1: there is
// no following collection.
TEST(CPHDReaderOp, DutyCycleSkippedOnLastFileWhenNotLooping) {
  ReaderCapture capture;
  ReaderOptions options;
  options.duty_cycle = 0.25;
  const auto t0 = std::chrono::steady_clock::now();
  ReaderApp(&capture, 256, {}, CollectionPaths(fixture_path()), options)
      .run();
  const double wall_s =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
          .count();
  ASSERT_EQ(capture.boundaries.size(), 1u);
  EXPECT_EQ(capture.boundaries[0], kStreamBoundaryEndOfStream);
  // Formation of one 256-pulse smoke tick is well under a quarter-second
  // without a cooldown. 0.25 duty on a 0.51 s collect would sleep ~1.5 s
  // if the last-file skip were broken.
  EXPECT_LT(wall_s, 1.0);
}

TEST(CPHDReaderOp, RejectsNonPositiveDutyCycle) {
  class BadDutyApp : public holoscan::Application {
   public:
    void compose() override {
      auto reader = make_operator<CPHDReaderOp<float>>(
          "reader", holoscan::Arg("path", CollectionPaths(fixture_path())),
          holoscan::Arg("coordinate_frame", ecef_frame()),
          holoscan::Arg("pulses_per_tick", static_cast<int64_t>(32)),
          holoscan::Arg("duty_cycle", 0.0));
      add_operator(reader);
    }
  };
  EXPECT_THROW(BadDutyApp().run(), std::runtime_error);
}

TEST(CPHDReaderOp, RejectsDutyCycleAboveOne) {
  class BadDutyApp : public holoscan::Application {
   public:
    void compose() override {
      auto reader = make_operator<CPHDReaderOp<float>>(
          "reader", holoscan::Arg("path", CollectionPaths(fixture_path())),
          holoscan::Arg("coordinate_frame", ecef_frame()),
          holoscan::Arg("pulses_per_tick", static_cast<int64_t>(32)),
          holoscan::Arg("duty_cycle", 1.5));
      add_operator(reader);
    }
  };
  EXPECT_THROW(BadDutyApp().run(), std::runtime_error);
}

TEST(CPHDReaderOp, RejectsNegativePlaybackRate) {
  class BadRateApp : public holoscan::Application {
   public:
    void compose() override {
      auto reader = make_operator<CPHDReaderOp<float>>(
          "reader", holoscan::Arg("path", CollectionPaths(fixture_path())),
          holoscan::Arg("coordinate_frame", ecef_frame()),
          holoscan::Arg("pulses_per_tick", static_cast<int64_t>(32)),
          holoscan::Arg("playback_rate", -1.0));
      add_operator(reader);
    }
  };
  EXPECT_THROW(BadRateApp().run(), std::runtime_error);
}

}  // namespace
}  // namespace holoscan::radar
