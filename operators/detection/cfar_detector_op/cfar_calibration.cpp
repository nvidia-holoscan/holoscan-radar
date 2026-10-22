// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "detection/cfar_detector_op/cfar_calibration.hpp"
#include "detection/cfar_detector_op/cfar_detector_op.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <ctime>
#include <iomanip>
#include <limits>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/logger/logger.hpp>

#include "detection/cfar_detector_op/cfar_alpha.hpp"
#include "detection/cfar_detector_op/nci_plan.hpp"

namespace holoscan::radar {
namespace {
using Clock = std::chrono::steady_clock;

// Sum normalized binomial masses outward from the mode. Starting with a
// unit modal mass avoids exp(-N*p) underflow, even for enormous N. Only the
// numerically non-negligible support is visited; no O(N) loop or factorials.
long double binomial_cdf(std::uint64_t n, long double p, std::uint64_t upper) {
  if (upper >= n || p <= 0)
    return 1;
  if (p >= 1)
    return 0;
  const auto mode = static_cast<std::uint64_t>(std::floor((static_cast<long double>(n) + 1) * p));
  long double mass = 1, total = 1, partial = mode <= upper ? 1 : 0;
  auto j = mode;
  while (j > 0) {
    mass *= static_cast<long double>(j) / (n - j + 1) * (1 - p) / p;
    --j;
    total += mass;
    if (j <= upper)
      partial += mass;
    if (mass < total * 1e-22L)
      break;
  }
  mass = 1;
  j = mode;
  while (j < n) {
    mass *= static_cast<long double>(n - j) / (j + 1) * p / (1 - p);
    ++j;
    total += mass;
    if (j <= upper)
      partial += mass;
    if (mass < total * 1e-22L)
      break;
  }
  return partial / total;
}

std::uint64_t count(long double value) {
  if (!std::isfinite(value) || value < 1 ||
      value >= static_cast<long double>(std::numeric_limits<std::uint64_t>::max()))
    throw std::invalid_argument("CFAR calibration trial budget exceeds uint64 capacity");
  return static_cast<std::uint64_t>(std::ceil(value));
}
std::uint64_t multiply(std::uint64_t a, std::uint64_t b) {
  if (b && a > std::numeric_limits<std::uint64_t>::max() / b)
    throw std::invalid_argument("CFAR calibration total trial count overflows uint64");
  return a * b;
}
std::string timestamp(double offset = 0) {
  if (!std::isfinite(offset) || offset > 2e9)
    return "beyond-clock-range";
  const auto now = std::chrono::system_clock::to_time_t(
      std::chrono::system_clock::now() + std::chrono::seconds{static_cast<std::int64_t>(offset)});
  std::tm value{};
  gmtime_r(&now, &value);
  std::ostringstream out;
  out << std::put_time(&value, "%Y-%m-%dT%H:%M:%SZ");
  return out.str();
}
void validate_accuracy(double p, double error, double confidence) {
  for (double value : {p, error, confidence}) {
    if (!std::isfinite(value) || value <= 0 || value >= 1)
      throw std::invalid_argument(
          "CFAR pfa, calibration_relative_error and calibration_confidence must be finite in "
          "(0,1)");
  }
}
std::vector<int> coordinates(int size, int half) {
  std::set<int> positions;
  for (int i = 0; i <= std::min(half, size - 1); ++i) {
    positions.insert(i);
    positions.insert(size - 1 - i);
  }
  return {positions.begin(), positions.end()};
}
double estimate_seconds(const CfarCalibrationPlan& plan, std::stop_token stop) {
  const auto& geometry =
      *std::max_element(plan.geometries.begin(), plan.geometries.end(), [](auto a, auto b) {
        return a.lut_index < b.lut_index;
      });
  // Warm CUDA/module loading before measuring trial throughput. Otherwise a
  // one-time context startup gets multiplied by billions of batches.
  detail::calibrate_cfar_geometry_gpu(plan, geometry, {}, stop, true);
  std::array<double, 3> samples{};
  for (auto& sample : samples) {
    const auto start = Clock::now();
    detail::calibrate_cfar_geometry_gpu(plan, geometry, {}, stop, true);
    sample = std::chrono::duration<double>(Clock::now() - start).count();
  }
  std::sort(samples.begin(), samples.end());
  return samples[1] * plan.total_trials / std::min<std::uint64_t>(65536, plan.budget.trials);
}
}  // namespace

CfarCalibrationBudget cfar_calibration_budget(double pfa, double relative_error,
                                              double confidence) {
  validate_accuracy(pfa, relative_error, confidence);
  long double lo = 0, hi = 16;
  for (int i = 0; i < 80; ++i) {
    const auto mid = (lo + hi) / 2;
    if (std::erfc(mid / std::sqrt(2.0L)) > 1 - confidence)
      lo = mid;
    else
      hi = mid;
  }
  auto n = count(std::max(4096.0L, hi * hi * (1 - pfa) / (relative_error * relative_error * pfa)));
  for (;;) {
    const auto rank = std::clamp<std::uint64_t>(
        count(std::max(1.0L, std::round(static_cast<long double>(n) * pfa))), 1, n);
    const auto coverage = binomial_cdf(n, (1 - relative_error) * pfa, rank - 1) -
                          binomial_cdf(n, std::min(1.0, (1 + relative_error) * pfa), rank - 1);
    if (coverage >= confidence)
      return {n, rank, static_cast<double>(coverage)};
    n = count(static_cast<long double>(n) * 1.02L + 1);
  }
}

namespace detail {
CfarCalibrationPlan prepare_cfar_calibration(const CFARParams& params,
                                             const std::vector<std::int64_t>& shape) {
  if (params.cfar_variant != CfarVariant::kCA && params.cfar_variant != CfarVariant::kGO &&
      params.cfar_variant != CfarVariant::kSO && params.cfar_variant != CfarVariant::kOS)
    throw std::invalid_argument("CFAR calibration invalid variant");
  validate_accuracy(params.pfa, params.calibration_relative_error, params.calibration_confidence);
  if (shape.size() < 2 || params.guard_cells.size() != 2 || params.training_cells.size() != 2 ||
      params.preintegrated_looks <= 0)
    throw std::invalid_argument(
        "CFAR calibration requires input_shape, 2D guard/training cells and positive looks");
  for (auto v : shape)
    if (v <= 0 || v > std::numeric_limits<int>::max())
      throw std::invalid_argument("CFAR calibration invalid input extent");
  for (auto v : params.guard_cells)
    if (v < 0 || v > std::numeric_limits<int>::max() / 4)
      throw std::invalid_argument("CFAR calibration invalid guard extent");
  for (auto v : params.training_cells)
    if (v < 0 || v > std::numeric_limits<int>::max() / 4)
      throw std::invalid_argument("CFAR calibration invalid training extent");
  if (params.cfar_variant == CfarVariant::kOS && params.os_rank < 0)
    throw std::invalid_argument("CFAR OS calibration requires os_rank");
  CfarCalibrationPlan plan;
  plan.params = std::make_shared<const CFARParams>(params);
  plan.input_shape = shape;
  std::uint64_t cells = 1;
  for (auto v : shape)
    cells = multiply(cells, static_cast<std::uint64_t>(v));
  if (cells > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))
    throw std::invalid_argument("CFAR calibration shape product overflows int64");
  const auto nci = NciPlan::from_shape(shape, params.nci_axes, "CFAR calibration", false);
  const auto looks = multiply(nci.fold, params.preintegrated_looks);
  if (looks > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))
    throw std::invalid_argument("CFAR calibration look count overflows int64");
  plan.effective_looks = static_cast<std::int64_t>(looks);
  const int nd = shape[shape.size() - 2], nr = shape.back();
  const int gd = params.guard_cells[0], gr = params.guard_cells[1];
  const int td = params.training_cells[0], tr = params.training_cells[1];
  const bool go =
      params.cfar_variant == CfarVariant::kGO || params.cfar_variant == CfarVariant::kSO;
  const auto max_index = go ? max_go_lut_index(nd, nr, gd, gr, td, tr)
                            : max_training_cell_count(nd, nr, gd, gr, td, tr);
  if (max_index < 0)
    throw std::invalid_argument("CFAR calibration training count overflow");
  if (params.cfar_variant == CfarVariant::kOS && max_index > kMaxOsTrainingCells)
    throw std::invalid_argument("CFAR OS training footprint exceeds 512 cells");
  if (params.cfar_variant == CfarVariant::kOS && params.os_rank >= max_index)
    throw std::invalid_argument("CFAR os_rank exceeds training count");
  plan.lut_size = static_cast<std::size_t>(max_index) + 1;
  std::set<std::array<std::int64_t, 5>> seen;
  for (int d : coordinates(nd, gd + td))
    for (int r : coordinates(nr, gr + tr)) {
      if (go) {
        std::int64_t rn, rp, dn, dp;
        go_training_side_counts(d, r, nd, nr, gd, gr, td, tr, &rn, &rp, &dn, &dp);
        const auto index = go_lut_index(rn, rp, dn, dp);
        if (index && seen.insert({index, rn, rp, dn, dp}).second)
          plan.geometries.push_back({index, d, r});
      } else {
        const auto index = training_cell_count(d, r, nd, nr, gd, gr, td, tr);
        if (index && seen.insert({index, 0, 0, 0, 0}).second)
          plan.geometries.push_back({index, d, r});
      }
    }
  if (!params.alpha_lut.empty()) {
    if (params.alpha_lut.size() < plan.lut_size)
      throw std::invalid_argument("CFAR alpha_lut too short");
    for (double a : params.alpha_lut)
      if (!std::isfinite(a) || a < 0)
        throw std::invalid_argument("CFAR alpha_lut must be finite and nonnegative");
    for (auto g : plan.geometries)
      if (params.alpha_lut[g.lut_index] <= 0)
        throw std::invalid_argument("CFAR alpha_lut reachable entry must be positive");
    if (!go && (params.cfar_variant == CfarVariant::kCA || looks == 1)) {
      for (std::size_t i = 1; i < plan.lut_size; ++i)
        if (params.alpha_lut[i] <= 0)
          throw std::invalid_argument("CFAR alpha_lut used entry must be positive");
    }
    plan.method = "external";
  } else if (max_index == 0) {
    plan.method = "empty";
  } else if (params.cfar_variant == CfarVariant::kCA ||
             (params.cfar_variant == CfarVariant::kOS && looks == 1)) {
    plan.method = "analytic";
  } else if (go || params.cfar_variant == CfarVariant::kOS) {
    plan.method = "direct_monte_carlo";
    plan.budget = cfar_calibration_budget(params.pfa, params.calibration_relative_error,
                                          params.calibration_confidence);
    plan.total_trials = multiply(plan.budget.trials, plan.geometries.size());
  } else {
    throw std::invalid_argument("CFAR calibration invalid variant");
  }
  return plan;
}

CfarCalibrationResult execute_cfar_calibration(CfarCalibrationPlan plan,
                                               CfarCalibrationCallback progress,
                                               std::stop_token stop) {
  CfarCalibrationResult result;
  result.started_at = timestamp();
  const auto start = Clock::now();
  CfarCalibrationProgress state;
  state.started_at = result.started_at;
  state.total_trials = plan.total_trials;
  state.total_geometries = plan.geometries.size();
  auto notify = [&](const char* status) {
    state.status = status;
    state.elapsed_seconds = std::chrono::duration<double>(Clock::now() - start).count();
    state.remaining_seconds = state.completed_trials
                                  ? state.elapsed_seconds / state.completed_trials *
                                        (state.total_trials - state.completed_trials)
                                  : plan.estimated_seconds;
    if (stop.stop_requested())
      throw CfarCalibrationCancelled{};
    if (progress)
      progress(state);
  };
  if (plan.method == "direct_monte_carlo")
    HOLOSCAN_LOG_WARN(
        "CFAR calibration {} start={} pfa={} relative_error={} confidence={} geometries={} "
        "trials_per_geometry={} total_trials={}; supply alpha_lut from calibrate_cfar to bypass "
        "runtime Monte Carlo",
        to_string(plan.params->cfar_variant), result.started_at, plan.params->pfa,
        plan.params->calibration_relative_error, plan.params->calibration_confidence,
        plan.geometries.size(), plan.budget.trials, plan.total_trials);
  try {
    notify("started");
    result.alpha_lut.assign(plan.lut_size, 0.0);
    const int nd = plan.input_shape[plan.input_shape.size() - 2], nr = plan.input_shape.back();
    const auto& p = *plan.params;
    if (plan.method == "external") {
      result.alpha_lut = p.alpha_lut;
    } else if (plan.method == "analytic") {
      if (p.cfar_variant == CfarVariant::kCA)
        result.alpha_lut =
            build_ca_alpha_lut(p.pfa, plan.effective_looks, nd, nr, p.guard_cells[0],
                               p.guard_cells[1], p.training_cells[0], p.training_cells[1]);
      else
        result.alpha_lut =
            build_os_alpha_lut(p.pfa, plan.effective_looks, p.os_rank, nd, nr, p.guard_cells[0],
                               p.guard_cells[1], p.training_cells[0], p.training_cells[1]);
    } else if (plan.method == "direct_monte_carlo") {
      if (!plan.estimated_seconds) {
        plan.estimated_seconds = estimate_seconds(plan, stop);
      }
      HOLOSCAN_LOG_WARN(
          "CFAR calibration estimated duration={} s start={} estimated_completion={} (conservative "
          "measured GPU "
          "probe at the largest LUT index; includes generation and tail selection)",
          plan.estimated_seconds, result.started_at, timestamp(plan.estimated_seconds));
      notify("estimated");
      auto last = Clock::now();
      for (const auto& g : plan.geometries) {
        const double a = calibrate_cfar_geometry_gpu(
            plan, g,
            [&](std::uint64_t n) {
              state.completed_trials += n;
              // Callbacks also service cancellation/signals once per batch; logs are throttled.
              notify("running");
              if (Clock::now() - last >= std::chrono::seconds{5}) {
                HOLOSCAN_LOG_INFO(
                    "CFAR calibration progress trials={}/{} geometries={}/{} elapsed={} s "
                    "remaining~{} s start={}",
                    state.completed_trials, state.total_trials, state.completed_geometries,
                    state.total_geometries, state.elapsed_seconds, state.remaining_seconds,
                    state.started_at);
                last = Clock::now();
              }
            },
            stop);
        if (!std::isfinite(a) || a <= 0)
          throw std::runtime_error("CFAR calibration produced invalid alpha");
        result.geometry_alphas.push_back(a);
        result.alpha_lut[g.lut_index] = std::max(result.alpha_lut[g.lut_index], a);
        ++state.completed_geometries;
      }
    }
    notify("completed");
    result.finished_at = timestamp();
    result.elapsed_seconds = state.elapsed_seconds;
    if (plan.method == "direct_monte_carlo")
      HOLOSCAN_LOG_INFO("CFAR calibration completed start={} end={} elapsed={} s trials={}",
                        result.started_at, result.finished_at, result.elapsed_seconds,
                        state.completed_trials);
    result.plan = std::move(plan);
    return result;
  } catch (const std::exception& e) {
    HOLOSCAN_LOG_WARN("CFAR calibration stopped start={} elapsed={} s trials={}/{}: {}",
                      state.started_at, std::chrono::duration<double>(Clock::now() - start).count(),
                      state.completed_trials, state.total_trials, e.what());
    throw;
  }
}
}  // namespace detail

CfarCalibrationPlan plan_cfar_calibration(const CFARParams& params,
                                          const std::vector<std::int64_t>& shape) {
  auto plan = detail::prepare_cfar_calibration(params, shape);
  if (plan.method == "direct_monte_carlo") {
    plan.estimated_seconds = estimate_seconds(plan, {});
  }
  return plan;
}
CfarCalibrationResult calibrate_cfar(const CFARParams& params,
                                     const std::vector<std::int64_t>& shape,
                                     CfarCalibrationCallback progress, std::stop_token stop) {
  return detail::execute_cfar_calibration(detail::prepare_cfar_calibration(params, shape),
                                          std::move(progress), stop);
}
}  // namespace holoscan::radar
