// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <vector>

#include "radar_common/descriptors/enums.hpp"

namespace holoscan::radar {

// Window taper specification, one of the module's shared config-plane
// descriptor types (see operators/common/README.md). Precision-agnostic:
// coefficients are generated in double
// and cast to the operator instance's precision when uploaded to the device.
struct WindowSpec {
  WindowType type{WindowType::kNone};

  // Target peak-sidelobe attenuation in (positive) dB. Consumed by the
  // Chebyshev and Taylor windows, and by Kaiser when kaiser_beta is unset.
  double sidelobe_attenuation_db{30.0};

  // Kaiser shape parameter. When unset, derived from
  // sidelobe_attenuation_db via Kaiser's empirical formula.
  std::optional<double> kaiser_beta{};

  // Number of nearly-constant-level sidelobes for the Taylor window.
  int taylor_nbar{4};

  // Tukey taper fraction in [0, 1] (0 → rectangular, 1 → Hann).
  double tukey_alpha{0.5};

  // Explicit coefficients for WindowType::kCustom. Must match the window
  // length at generation time.
  std::vector<double> coefficients{};

  // Divide the window by its mean so a coherent (in-bin) tone keeps its
  // amplitude through the taper.
  bool coherent_gain_normalization{true};

  friend bool operator==(const WindowSpec& a, const WindowSpec& b) {
    return a.type == b.type &&
           a.sidelobe_attenuation_db == b.sidelobe_attenuation_db &&
           a.kaiser_beta == b.kaiser_beta &&
           a.taylor_nbar == b.taylor_nbar &&
           a.tukey_alpha == b.tukey_alpha &&
           a.coefficients == b.coefficients &&
           a.coherent_gain_normalization == b.coherent_gain_normalization;
  }
  friend bool operator!=(const WindowSpec& a, const WindowSpec& b) {
    return !(a == b);
  }
};

}  // namespace holoscan::radar
