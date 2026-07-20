// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace holoscan::radar {

// Canonical high-precision pi for analytic DSP / simulation math. Prefer this
// over a per-translation-unit literal so window generation, the point-target
// simulator, and their tests all agree to the last ulp.
inline constexpr double kPi = 3.14159265358979323846;

}  // namespace holoscan::radar
