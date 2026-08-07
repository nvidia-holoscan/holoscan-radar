// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace holoscan::radar {

// Canonical high-precision math/physics constants for analytic DSP, simulation,
// and their tests. Prefer these over per-translation-unit literals so every
// translation unit (host and CUDA device code) agrees to the last ulp. Value
// use in device kernels is fine — the compiler constant-folds these.
inline constexpr double kPi = 3.14159265358979323846;
inline constexpr double kTwoPi = 6.28318530717958647692;   // 2*pi
inline constexpr double kHalfPi = 1.57079632679489661923;  // pi/2

// Speed of light in vacuum (m/s), exact by SI definition. The default RF
// propagation speed for FMCW range/velocity math and the simulator.
inline constexpr double kSpeedOfLight = 299792458.0;

}  // namespace holoscan::radar
