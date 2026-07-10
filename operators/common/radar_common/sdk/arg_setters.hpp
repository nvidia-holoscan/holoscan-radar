// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <holoscan/holoscan.hpp>

#include "radar_common/descriptors/enums.hpp"
#include "radar_common/descriptors/fmcw_waveform_descriptor.hpp"
#include "radar_common/descriptors/window_spec.hpp"
#include "radar_common/sdk/yaml_converters.hpp"

namespace holoscan::radar {

// Holoscan's ArgumentSetter only knows built-in parameter types; every
// custom Parameter<T> type needs a registered setter or operator
// initialization fails with "No argument setter for type ...". The SDK's
// default setter (ArgumentSetter::ensure_type<T>) handles both native C++
// Args (direct any_cast) and YAML-sourced Args from from_config(), the
// latter through the YAML::convert<T> specializations in
// yaml_converters.hpp — included here so every registering translation
// unit sees them.
//
// Registration is global and idempotent (the setter map keeps the first
// registration for a type; all radar registrations are identical). Each
// operator calls ensure_radar_arg_setters() at the top of its initialize()
// override, before Operator::initialize() processes the arguments
// (default-value assignment also runs through the setters).
inline void ensure_radar_arg_setters() {
  static const bool registered = [] {
    ArgumentSetter::ensure_type<WindowSpec>();
    ArgumentSetter::ensure_type<FMCWWaveformDescriptor>();
    ArgumentSetter::ensure_type<FftNorm>();
    ArgumentSetter::ensure_type<ClutterFilterMethod>();
    return true;
  }();
  (void)registered;
}

}  // namespace holoscan::radar
