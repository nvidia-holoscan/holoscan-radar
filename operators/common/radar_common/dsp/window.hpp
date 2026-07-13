// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "radar_common/descriptors/window_spec.hpp"

namespace holoscan::radar {

// Generate the window coefficients described by `spec` for `length` samples,
// in double precision (callers cast to their instantiation precision when
// uploading to the device).
//
// All windows are generated in their symmetric form. When
// spec.coherent_gain_normalization is true, the result is divided by its
// mean so an in-bin tone keeps its amplitude through the taper.
//
// WindowType::kNone returns all-ones (rectangular). Throws
// std::invalid_argument on bad specs (length < 1, custom coefficients of
// the wrong length, tukey_alpha outside [0, 1], taylor_nbar < 1, ...).
std::vector<double> generate_window(const WindowSpec& spec, int64_t length);

// Canonical cache key for (spec, length): includes only the fields the
// window type actually consumes, so equivalent specs share a cache entry.
// Used with ConfigCache to implement the warn-once-and-cache rule for
// window overrides.
std::string window_cache_key(const WindowSpec& spec, int64_t length);

}  // namespace holoscan::radar
