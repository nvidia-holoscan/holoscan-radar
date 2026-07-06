// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string_view>

namespace holoscan::radar {

// Window taper families supported by WindowSpec.
enum class WindowType {
  kNone,
  kHann,
  kHamming,
  kBlackman,
  kBlackmanHarris,
  kKaiser,
  kChebyshev,
  kTaylor,
  kFlatTop,
  kTukey,
  kCustom,
};

// FFT normalization convention. kBackward (unnormalized forward transform)
// is the radar standard: it keeps coherent-integration gain visible.
enum class FftNorm {
  kBackward,
  kOrtho,
  kForward,
};

// Slow-time clutter suppression method. kNone is only meaningful for the
// fused operator's optional inline clutter stage; ClutterFilterOp itself
// requires one of the active methods.
enum class ClutterFilterMethod {
  kNone,
  kMeanSubtraction,
  kSingleDelay,
  kDoubleDelay,
};

// Canonical spellings (lowercase snake_case, for logs and future YAML
// configs).
constexpr std::string_view to_string(WindowType type) {
  switch (type) {
    case WindowType::kNone: return "none";
    case WindowType::kHann: return "hann";
    case WindowType::kHamming: return "hamming";
    case WindowType::kBlackman: return "blackman";
    case WindowType::kBlackmanHarris: return "blackman_harris";
    case WindowType::kKaiser: return "kaiser";
    case WindowType::kChebyshev: return "chebyshev";
    case WindowType::kTaylor: return "taylor";
    case WindowType::kFlatTop: return "flat_top";
    case WindowType::kTukey: return "tukey";
    case WindowType::kCustom: return "custom";
  }
  return "unknown";
}

constexpr std::string_view to_string(FftNorm norm) {
  switch (norm) {
    case FftNorm::kBackward: return "backward";
    case FftNorm::kOrtho: return "ortho";
    case FftNorm::kForward: return "forward";
  }
  return "unknown";
}

constexpr std::string_view to_string(ClutterFilterMethod method) {
  switch (method) {
    case ClutterFilterMethod::kNone: return "none";
    case ClutterFilterMethod::kMeanSubtraction: return "mean_subtraction";
    case ClutterFilterMethod::kSingleDelay: return "single_delay";
    case ClutterFilterMethod::kDoubleDelay: return "double_delay";
  }
  return "unknown";
}

}  // namespace holoscan::radar
