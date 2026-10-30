// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// App-owned look-axis APC liar, oracle and PGA scorer. Not contract
// operators: they exist to give autofocus a known answer to find, so that
// "the image got sharper" can be checked against "the estimate matched the
// error we injected".
//
// The liar runs on ingest ticks and AutofocusOp emits phase_correction on the
// single reform tick, so the recorded lie travels between them as a shared
// host buffer rather than a Holoscan edge -- there is no tick on which both
// could be joined.

#pragma once

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/expected_macros.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/logger/logger.hpp>

#include "error_profile.hpp"  // NOLINT(build/include_subdir)
#include "io/tensor_dump_op/npy_writer.hpp"
#include "radar_common/descriptors/max_tensor_extents.hpp"
#include "radar_common/math/constants.hpp"
#include "radar_common/sdk/tensor_io.hpp"
#include "radar_common/tensor/max_byte_span.hpp"
#include "radar_common/tensor/tensor_utils.hpp"

namespace holoscan::radar::app::sar {

inline double phase_rms(const std::vector<double>& a, const std::vector<double>& b) {
  if (a.size() != b.size() || a.empty()) {
    return 0.0;
  }
  double acc = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const double d = a[i] - b[i];
    acc += d * d;
  }
  return std::sqrt(acc / static_cast<double>(a.size()));
}

inline double peak_to_peak(const std::vector<double>& a, const std::vector<double>& b) {
  double lo = 0.0;
  double hi = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const double d = a[i] - b[i];
    lo = std::min(lo, d);
    hi = std::max(hi, d);
  }
  return hi - lo;
}

struct LieRecord {
  std::vector<double> los_error_m;
  std::vector<double> expected_phase;
  int64_t num_pulses{0};

  void resize(int64_t n) {
    num_pulses = n;
    los_error_m.assign(static_cast<std::size_t>(n), 0.0);
    expected_phase.assign(static_cast<std::size_t>(n), 0.0);
  }
};

// Look-direction APC displacement plus the equivalent per-pulse phase on range
// profiles. SRP-relative backprojection cancels a pure line-of-sight APC lie
// in dR, so the leftover that PGA has to estimate is applied to the profiles
// (4 pi f delta / c, sense add) rather than left in the geometry alone.
// `T` is the sample precision and `TGeom` the geometry precision, matching the
// formation operators this sits in front of. The three publications already
// carry the dtype they received, so only the declared pools, the profile
// check, and the host staging buffer follow the parameters.
template <typename T, typename TGeom = T>
class PerturbPositionsOp : public holoscan::Operator<> {
 public:
  PerturbPositionsOp(std::shared_ptr<LieRecord> record, std::array<double, 3> look,
                     ErrorProfile profile, int64_t num_pulses, double reference_frequency,
                     const MaxTensorExtents& extents)
  : record_(std::move(record)),
    look_(look),
    profile_(std::move(profile)),
    num_pulses_(num_pulses),
    reference_frequency_(reference_frequency) {
    constexpr const char* kWho = "PerturbPositionsOp";
    const auto pulses = require_extent(extents.max_pulses, "max_pulses", kWho);
    const auto bins = require_extent(extents.max_range_bins, "max_range_bins", kWho);
    profiles_span_ = byte_span_complex_cube<T>({pulses, bins}, kWho);
    positions_span_ = byte_span_real_cube<TGeom>({pulses, 3}, kWho);
  }

  void setup(OperatorSpec& spec) override {
    constexpr TensorRepresentation device{
        .memory_kind = MemoryKind::kCudaDevice, .dtype = std::nullopt, .rank = std::nullopt};
    spec.input(range_profiles).queue_depth(2U).expects_tensor(tensor_input(device));
    spec.input(positions).queue_depth(2U).expects_tensor(tensor_input(device));
    spec.input(reference_position).queue_depth(2U).expects_tensor(tensor_input(device));
    spec.output(range_profiles_out)
        .produces_tensor(pooled_tensor_output(device, 2U, profiles_span_));
    spec.output(positions_out).produces_tensor(pooled_tensor_output(device, 2U, positions_span_));
    spec.output(reference_position_out)
        .produces_tensor(pooled_tensor_output(device, 2U, positions_span_));
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result
        .trigger(OnAligned{.ports = ports(range_profiles, positions, reference_position),
                           .window = std::chrono::nanoseconds{0}})
        .input(range_profiles, Exact{})
        .input(positions, Exact{})
        .input(reference_position, Exact{});
    return result;
  }

  void start() override {
    if (!record_) {
      throw std::invalid_argument("PerturbPositionsOp: LieRecord is null");
    }
    if (num_pulses_ < 1) {
      throw std::invalid_argument("PerturbPositionsOp: num_pulses must be >= 1");
    }
    if (profile_.empty()) {
      throw std::invalid_argument("PerturbPositionsOp: error profile is empty");
    }
    if (!(reference_frequency_ > 0.0) || !std::isfinite(reference_frequency_)) {
      throw std::invalid_argument("PerturbPositionsOp: reference_frequency must be finite and > 0");
    }
    const double ln = std::sqrt(look_[0] * look_[0] + look_[1] * look_[1] + look_[2] * look_[2]);
    if (!(std::abs(ln - 1.0) < 1.0e-6)) {
      throw std::invalid_argument("PerturbPositionsOp: look must be unit length");
    }
    record_->resize(num_pulses_);
    rad_per_metre_ = 4.0 * kPi * reference_frequency_ / kSpeedOfLight;
    next_pulse_ = 0;
    // The whole sequence is generated once rather than per pulse: the random
    // walk is cumulative, so it cannot be evaluated from a pulse index alone,
    // and compute() sees the aperture in batches across ticks.
    metres_ = profile_metres(profile_, num_pulses_);
    std::vector<double> observable(metres_);
    remove_affine(&observable);
    double sq = 0.0;
    double peak = 0.0;
    for (const double v : observable) {
      sq += v * v;
      peak = std::max(peak, std::abs(v));
    }
    const double rms_m = std::sqrt(sq / static_cast<double>(observable.size()));
    HOLOSCAN_LOG_INFO(
        "PerturbPositionsOp: look [{:.6f}, {:.6f}, {:.6f}]  {} pulses  {:.3f} rad/m  "
        "{}  affine-removed {:.4f} rad RMS / {:.4f} rad peak  (profile phase; dR cancels LOS APC)",
        look_[0], look_[1], look_[2], num_pulses_, rad_per_metre_, describe(profile_),
        rms_m * rad_per_metre_, peak * rad_per_metre_);
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    constexpr const char* kWho = "PerturbPositionsOp";
    HOLOSCAN_TRY(auto profiles, receive_tensor(range_profiles, kWho));
    HOLOSCAN_TRY(auto pos, receive_tensor(positions, kWho));
    HOLOSCAN_TRY(auto reference, receive_tensor(reference_position, kWho));
    try {
      require_contiguous(profiles, "range_profiles", kWho);
      require_contiguous(pos, "positions", kWho);
      require_contiguous(reference, "reference_position", kWho);
      require_cuda_accessible(profiles, "range_profiles", kWho);
      require_cuda_accessible(pos, "positions", kWho);
    } catch (const std::invalid_argument& e) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument, e.what()});
    }
    cudaStream_t stream = context.cuda_stream().get();
    if (stream == nullptr) {
      return make_unexpected(Error{ErrorCode::kNotSupported, "PerturbPositionsOp: no CUDA stream"});
    }

    const auto pos_shape = pos.shape();
    if (pos_shape.size() != 2U || pos_shape[1] != 3) {
      return make_unexpected(
          Error{ErrorCode::kInvalidArgument, "PerturbPositionsOp: expected positions (n,3)"});
    }
    if (pos.dtype().code != kDLFloat || pos.dtype().bits != 64) {
      return make_unexpected(
          Error{ErrorCode::kInvalidArgument, "PerturbPositionsOp: expected float64 positions"});
    }
    const int64_t count = pos_shape[0];
    const int64_t p0 = next_pulse_;
    if (p0 + count > num_pulses_) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument,
                                   "PerturbPositionsOp: more pulses than the configured aperture"});
    }

    const auto prof_shape = profiles.shape();
    if (prof_shape.size() != 2U || prof_shape[0] != count) {
      return make_unexpected(
          Error{ErrorCode::kInvalidArgument,
                "PerturbPositionsOp: range_profiles (num_pulses,num_bins) must match positions"});
    }
    require_dtype(profiles, complex_dtype<T>(), "range_profiles", "PerturbPositionsOp");

    // Record the lie for this batch first: both outputs read it, and the
    // scorer reads it again on the reform tick.
    HOLOSCAN_TRY(auto host_pos, read_positions(pos, count, stream));
    for (int64_t i = 0; i < count; ++i) {
      const int64_t p = p0 + i;
      const double delta = metres_[static_cast<std::size_t>(p)];
      host_pos[static_cast<std::size_t>(i) * 3 + 0] += look_[0] * delta;
      host_pos[static_cast<std::size_t>(i) * 3 + 1] += look_[1] * delta;
      host_pos[static_cast<std::size_t>(i) * 3 + 2] += look_[2] * delta;
      record_->los_error_m[static_cast<std::size_t>(p)] = delta;
      record_->expected_phase[static_cast<std::size_t>(p)] = rad_per_metre_ * delta;
    }
    next_pulse_ = p0 + count;

    HOLOSCAN_TRY(emit_required_tensor(
        positions_out, context, "PerturbPositionsOp.positions_out", false, [&] {
          return prepare_device_tensor(
              positions_out, context, pos_shape, pos.dtype(),
              [&](void* destination, std::size_t bytes) -> expected<void, Error> {
                return copy_to_device(destination, host_pos.data(), bytes, stream);
              });
        }));

    HOLOSCAN_TRY(auto host_prof, read_profiles(profiles, count, prof_shape[1], stream));
    for (int64_t i = 0; i < count; ++i) {
      const double phi = record_->expected_phase[static_cast<std::size_t>(p0 + i)];
      const std::complex<T> phasor{static_cast<T>(std::cos(phi)), static_cast<T>(std::sin(phi))};
      const std::size_t row = static_cast<std::size_t>(i) * static_cast<std::size_t>(prof_shape[1]);
      for (int64_t b = 0; b < prof_shape[1]; ++b) {
        host_prof[row + static_cast<std::size_t>(b)] *= phasor;
      }
    }
    HOLOSCAN_TRY(emit_required_tensor(
        range_profiles_out, context, "PerturbPositionsOp.range_profiles_out", false, [&] {
          return prepare_device_tensor(
              range_profiles_out, context, prof_shape, profiles.dtype(),
              [&](void* destination, std::size_t bytes) -> expected<void, Error> {
                return copy_to_device(destination, host_prof.data(), bytes, stream);
              });
        }));

    // The reference point is untouched; it rides through so the downstream
    // join still sees one producer for the whole geometry triple.
    HOLOSCAN_TRY(emit_required_tensor(
        reference_position_out, context, "PerturbPositionsOp.reference_position_out", false, [&] {
          return prepare_device_tensor(
              reference_position_out, context, reference.shape(), reference.dtype(),
              [&](void* destination, std::size_t bytes) -> expected<void, Error> {
                if (cudaMemcpyAsync(destination, reference.data(), bytes, cudaMemcpyDeviceToDevice,
                                    stream) != cudaSuccess) {
                  return make_unexpected(
                      Error{ErrorCode::kFailure, "PerturbPositionsOp: reference copy failed"});
                }
                return {};
              });
        }));
    retained_ = std::move(profiles);
    return {};
  }

  Input<Tensor> range_profiles{"range_profiles"};
  Input<Tensor> positions{"positions"};
  Input<Tensor> reference_position{"reference_position"};
  Output<Tensor> range_profiles_out{"range_profiles_out"};
  Output<Tensor> positions_out{"positions_out"};
  Output<Tensor> reference_position_out{"reference_position_out"};

 private:
  static expected<void, Error> copy_to_device(void* destination, const void* source,
                                              std::size_t bytes, cudaStream_t stream) {
    if (cudaMemcpyAsync(destination, source, bytes, cudaMemcpyHostToDevice, stream) !=
            cudaSuccess ||
        cudaStreamSynchronize(stream) != cudaSuccess) {
      return make_unexpected(Error{ErrorCode::kFailure, "PerturbPositionsOp: H2D failed"});
    }
    return {};
  }

  static expected<std::vector<double>, Error> read_positions(const Tensor& in, int64_t count,
                                                             cudaStream_t stream) {
    std::vector<double> host(static_cast<std::size_t>(count) * 3U);
    if (cudaMemcpyAsync(host.data(), in.data(), host.size() * sizeof(double),
                        cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
        cudaStreamSynchronize(stream) != cudaSuccess) {
      return make_unexpected(
          Error{ErrorCode::kFailure, "PerturbPositionsOp: positions D2H failed"});
    }
    return host;
  }

  static expected<std::vector<std::complex<T>>, Error> read_profiles(const Tensor& in,
                                                                     int64_t pulses, int64_t bins,
                                                                     cudaStream_t stream) {
    std::vector<std::complex<T>> host(static_cast<std::size_t>(pulses * bins));
    if (cudaMemcpyAsync(host.data(), in.data(), host.size() * sizeof(std::complex<T>),
                        cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
        cudaStreamSynchronize(stream) != cudaSuccess) {
      return make_unexpected(Error{ErrorCode::kFailure, "PerturbPositionsOp: profile D2H failed"});
    }
    return host;
  }

  std::shared_ptr<LieRecord> record_;
  std::array<double, 3> look_{};
  ErrorProfile profile_;
  std::vector<double> metres_;
  int64_t num_pulses_{0};
  double reference_frequency_{0.0};
  std::size_t profiles_span_{0U};
  std::size_t positions_span_{0U};
  double rad_per_metre_{0.0};
  int64_t next_pulse_{0};
  Tensor retained_;
};

// Figure 2 apply-path diagnostic: emit the liar's recorded leftover as
// phase_correction on the reform tick, so PhaseCorrectionOp undoes the
// profile multiply without AutofocusOp in the loop. Positions stay lied, since
// dR is supposed to cancel a pure look-axis APC shift. If the product comes
// back focused, the correction path is exact and any shortfall in the full
// run belongs to the estimator.
// `T` matches the PhaseCorrectionOp it feeds, which requires the phase at its
// own precision.
template <typename T>
class OraclePhaseOp : public holoscan::Operator<> {
 public:
  OraclePhaseOp(std::shared_ptr<LieRecord> record, const MaxTensorExtents& extents)
  : record_(std::move(record)) {
    phase_span_ = byte_span_real_cube<T>(
        {require_extent(extents.max_pulses, "max_pulses", "OraclePhaseOp")}, "OraclePhaseOp");
  }

  void setup(OperatorSpec& spec) override {
    constexpr TensorRepresentation device{
        .memory_kind = MemoryKind::kCudaDevice, .dtype = std::nullopt, .rank = std::nullopt};
    spec.input(range_profiles).queue_depth(2U).expects_tensor(tensor_input(device));
    spec.output(phase_correction).produces_tensor(pooled_tensor_output(device, 1U, phase_span_));
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnEach{range_profiles});
    return result;
  }

  void start() override {
    if (!record_) {
      throw std::invalid_argument("OraclePhaseOp: LieRecord is null");
    }
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    constexpr const char* kWho = "OraclePhaseOp";
    HOLOSCAN_TRY(auto profiles, receive_tensor(range_profiles, kWho));
    const auto shape = profiles.shape();
    if (shape.size() != 2U) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument,
                                   "OraclePhaseOp: expected range_profiles (num_pulses,num_bins)"});
    }
    const int64_t n_pulses = shape[0];
    if (record_->num_pulses < 1 ||
        static_cast<int64_t>(record_->expected_phase.size()) != record_->num_pulses) {
      return make_unexpected(
          Error{ErrorCode::kFailure, "OraclePhaseOp: LieRecord is empty or inconsistent"});
    }
    if (n_pulses != record_->num_pulses) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument,
                                   "OraclePhaseOp: replay pulse count " + std::to_string(n_pulses) +
                                       " != liar aperture " + std::to_string(record_->num_pulses)});
    }
    cudaStream_t stream = context.cuda_stream().get();
    if (stream == nullptr) {
      return make_unexpected(Error{ErrorCode::kNotSupported, "OraclePhaseOp: no CUDA stream"});
    }
    // LieRecord keeps the truth in double because the geometry that produced
    // it is double; the port is real_dtype<T>. Narrow through a staging buffer
    // rather than copying the record's bytes straight out: at T=float the two
    // byte counts differ by two, so the device would receive the first half of
    // a double array reinterpreted as floats, which lands on bit patterns that
    // decode to inf and NaN and takes the whole image with them.
    const std::vector<T> staging(record_->expected_phase.begin(), record_->expected_phase.end());
    // The oracle is the reference every other path is scored against, so it
    // must not be the thing that is quietly wrong. A non-finite phase makes the
    // whole corrected image NaN, which the smoke tests cannot see because they
    // read the log rather than the pixels; failing the tick instead turns that
    // into a non-zero exit they do check.
    if (!std::all_of(staging.begin(), staging.end(), [](T value) {
          return std::isfinite(value);
        })) {
      return make_unexpected(
          Error{ErrorCode::kFailure, "OraclePhaseOp: staged phase is not finite"});
    }
    return emit_required_tensor(
        phase_correction, context, "OraclePhaseOp.phase_correction", false, [&] {
          return prepare_device_tensor(
              phase_correction, context, std::vector<int64_t>{n_pulses}, real_dtype<T>(),
              [&](void* destination, std::size_t bytes) -> expected<void, Error> {
                if (bytes != staging.size() * sizeof(T)) {
                  return make_unexpected(
                      Error{ErrorCode::kFailure, "OraclePhaseOp: phase_correction is " +
                                                     std::to_string(bytes) + " bytes, staged " +
                                                     std::to_string(staging.size() * sizeof(T))});
                }
                if (cudaMemcpyAsync(destination, staging.data(), bytes, cudaMemcpyHostToDevice,
                                    stream) != cudaSuccess ||
                    cudaStreamSynchronize(stream) != cudaSuccess) {
                  return make_unexpected(Error{ErrorCode::kFailure, "OraclePhaseOp: H2D failed"});
                }
                return {};
              });
        });
  }

  Input<Tensor> range_profiles{"range_profiles"};
  Output<Tensor> phase_correction{"phase_correction"};

 private:
  std::shared_ptr<LieRecord> record_;
  std::size_t phase_span_{0U};
};

// Compares the estimate to the liar's recorded leftover phase, after dropping
// the affine PGA cannot observe. The sign is reported rather than assumed:
// which way round the estimate comes out depends on the image's conjugate
// convention, and picking the better of the two is what makes the residual
// meaningful instead of a sign error dominating it.
class AutofocusScoreOp : public holoscan::Operator<> {
 public:
  AutofocusScoreOp(std::shared_ptr<LieRecord> record, double warn_rms_rad,
                   double reference_frequency, std::string dump_dir)
  : record_(std::move(record)),
    warn_rms_rad_(warn_rms_rad),
    reference_frequency_(reference_frequency),
    dump_dir_(std::move(dump_dir)) {}

  void setup(OperatorSpec& spec) override { spec.input(phase_correction).queue_depth(2U); }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnEach{phase_correction});
    return result;
  }

  void start() override {
    if (!record_) {
      throw std::invalid_argument("AutofocusScoreOp: LieRecord is null");
    }
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    constexpr const char* kWho = "AutofocusScoreOp";
    HOLOSCAN_TRY(auto phase, receive_tensor(phase_correction, kWho));
    try {
      require_contiguous(phase, "phase_correction", kWho);
      require_cuda_accessible(phase, "phase_correction", kWho);
    } catch (const std::invalid_argument& e) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument, e.what()});
    }
    const auto shape = phase.shape();
    if (shape.size() != 1U || shape[0] != record_->num_pulses) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument,
                                   "AutofocusScoreOp: phase_correction length does not match the "
                                   "recorded aperture"});
    }
    const int64_t n = shape[0];
    const bool f64 = phase.dtype().code == kDLFloat && phase.dtype().bits == 64;
    const bool f32 = phase.dtype().code == kDLFloat && phase.dtype().bits == 32;
    if (!f64 && !f32) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument,
                                   "AutofocusScoreOp: expected float32/float64 phase_correction"});
    }
    cudaStream_t stream = context.cuda_stream().get();
    if (stream == nullptr) {
      return make_unexpected(Error{ErrorCode::kNotSupported, "AutofocusScoreOp: no CUDA stream"});
    }
    std::vector<double> est(static_cast<std::size_t>(n));
    if (f64) {
      if (cudaMemcpyAsync(est.data(), phase.data(), est.size() * sizeof(double),
                          cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
          cudaStreamSynchronize(stream) != cudaSuccess) {
        return make_unexpected(Error{ErrorCode::kFailure, "AutofocusScoreOp: D2H failed"});
      }
    } else {
      std::vector<float> tmp(static_cast<std::size_t>(n));
      if (cudaMemcpyAsync(tmp.data(), phase.data(), tmp.size() * sizeof(float),
                          cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
          cudaStreamSynchronize(stream) != cudaSuccess) {
        return make_unexpected(Error{ErrorCode::kFailure, "AutofocusScoreOp: D2H failed"});
      }
      for (int64_t i = 0; i < n; ++i) {
        est[static_cast<std::size_t>(i)] = static_cast<double>(tmp[static_cast<std::size_t>(i)]);
      }
    }

    auto want_pos = record_->expected_phase;
    auto want_neg = record_->expected_phase;
    for (double& v : want_neg) {
      v = -v;
    }
    auto est_pos = est;
    auto est_neg = est;
    remove_affine(&est_pos);
    remove_affine(&est_neg);
    remove_affine(&want_pos);
    remove_affine(&want_neg);
    const double rms_pos = phase_rms(est_pos, want_pos);
    const double rms_neg = phase_rms(est_neg, want_neg);
    const bool use_neg = rms_neg < rms_pos;
    const double rms = use_neg ? rms_neg : rms_pos;
    const double p2p = use_neg ? peak_to_peak(est_neg, want_neg) : peak_to_peak(est_pos, want_pos);
    const double rad_per_m = 4.0 * kPi * reference_frequency_ / kSpeedOfLight;
    const double rms_mm = (rad_per_m > 0.0) ? (rms / rad_per_m) * 1.0e3 : 0.0;
    // The baseline is the affine-removed RMS of the truth alone -- what PGA
    // had to remove. Reporting it beside the leftover makes a sweep row
    // self-describing and comparable across error shapes, where a peak
    // amplitude is only meaningful for a polynomial.
    double peak = 0.0;
    double base_sq = 0.0;
    for (const double v : want_pos) {
      peak = std::max(peak, std::abs(v));
      base_sq += v * v;
    }
    const double baseline = std::sqrt(base_sq / static_cast<double>(want_pos.size()));
    HOLOSCAN_LOG_INFO(
        "AutofocusScoreOp: affine-removed RMS {:.4f} rad ({:.3f} mm LOS)  p2p {:.4f} rad  "
        "sign {}  baseline {:.4f} rad RMS / {:.4f} rad peak  removed {:.1f}x",
        rms, rms_mm, p2p, use_neg ? "-4pi f d / c" : "+4pi f d / c", baseline, peak,
        (rms > 0.0) ? (baseline / rms) : 0.0);
    if (rms > warn_rms_rad_) {
      HOLOSCAN_LOG_WARN(
          "AutofocusScoreOp: RMS {:.4f} rad exceeds warn threshold {:.4f} rad "
          "(PGA did not recover the lie)",
          rms, warn_rms_rad_);
    }
    if (!dump_dir_.empty()) {
      const std::string exp_path = dump_dir_ + "/af_expected_phase.npy";
      const std::string est_path = dump_dir_ + "/af_estimated_phase.npy";
      const std::vector<int64_t> shp{n};
      npy::write(exp_path, real_dtype<double>(), shp, record_->expected_phase.data(),
                 static_cast<std::size_t>(n) * sizeof(double));
      npy::write(est_path, real_dtype<double>(), shp, est.data(),
                 static_cast<std::size_t>(n) * sizeof(double));
      HOLOSCAN_LOG_INFO("AutofocusScoreOp: wrote {} and {}", exp_path, est_path);
    }
    return {};
  }

  Input<Tensor> phase_correction{"phase_correction"};

 private:
  std::shared_ptr<LieRecord> record_;
  double warn_rms_rad_{1.0};
  double reference_frequency_{0.0};
  std::string dump_dir_;
};

}  // namespace holoscan::radar::app::sar
