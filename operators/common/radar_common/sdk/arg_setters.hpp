// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <any>
#include <utility>

#include <holoscan/holoscan.hpp>

#include "radar_common/descriptors/enums.hpp"
#include "radar_common/descriptors/fmcw_waveform_descriptor.hpp"
#include "radar_common/descriptors/window_spec.hpp"

namespace holoscan::radar {

// Holoscan's ArgumentSetter only knows built-in parameter types; every
// custom Parameter<T> type needs a registered setter or operator
// initialization fails with "No argument setter for type ...". The
// YAML-backed registration path (Component::register_converter<T> /
// ArgumentSetter::ensure_type<T>) requires a YAML::convert<T>
// specialization, which the radar descriptor types defer until
// from_config() support lands — so register a plain any_cast setter
// instead: native C++ Args round-trip through std::any, YAML configs are
// (for now) rejected with a clear log message.
//
// Registration is global and idempotent (the setter map keeps the last
// registration; all radar registrations are identical).
template <typename T>
void register_any_argument_setter(const char* type_name) {
  ArgumentSetter::get_instance().add_argument_setter<T>(
      [type_name](ParameterWrapper& param_wrap, Arg& arg) -> bool {
        auto& param =
            *std::any_cast<Parameter<T>*>(param_wrap.value());
        if (arg.name().empty() && !arg.has_value()) {
          param.set_default_value();
          return true;
        }
        try {
          param = std::any_cast<T>(arg.value());
          return true;
        } catch (const std::bad_any_cast&) {
          HOLOSCAN_LOG_ERROR(
              "argument '{}' is not a {} (native C++ Arg required; YAML "
              "configuration of this type is not supported yet)",
              arg.name(), type_name);
          return false;
        }
      });
}

// Register the setters for every custom parameter type the radar
// operators declare. Idempotent; each operator calls this at the top of
// its initialize() override, before Operator::initialize() processes the
// arguments (default-value assignment also runs through the setters).
inline void ensure_radar_arg_setters() {
  static const bool registered = [] {
    register_any_argument_setter<WindowSpec>("WindowSpec");
    register_any_argument_setter<FMCWWaveformDescriptor>(
        "FMCWWaveformDescriptor");
    register_any_argument_setter<FftNorm>("FftNorm");
    register_any_argument_setter<ClutterFilterMethod>("ClutterFilterMethod");
    return true;
  }();
  (void)registered;
}

}  // namespace holoscan::radar
