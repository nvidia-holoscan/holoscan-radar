// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string_view>

namespace holoscan::radar {

enum class RepresentativePolicy {
  kPeak,
  kMedoid,
  kFirst,
};

enum class ScoreMethod {
  kWeightedSum,
  kWeightedL2,
  kWeightedMax,
  kCustom,
};

// Default second template argument for DetectionReductionOp: score_method=
// custom is rejected unless the caller supplies a real device ScoreFn.
struct NoCustomScore {};

constexpr std::string_view to_string(RepresentativePolicy policy) {
  switch (policy) {
    case RepresentativePolicy::kPeak: return "peak";
    case RepresentativePolicy::kMedoid: return "medoid";
    case RepresentativePolicy::kFirst: return "first";
  }
  return "unknown";
}

constexpr std::string_view to_string(ScoreMethod method) {
  switch (method) {
    case ScoreMethod::kWeightedSum: return "weighted_sum";
    case ScoreMethod::kWeightedL2: return "weighted_l2";
    case ScoreMethod::kWeightedMax: return "weighted_max";
    case ScoreMethod::kCustom: return "custom";
  }
  return "unknown";
}

}  // namespace holoscan::radar
