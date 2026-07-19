// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <string>

#include <pybind11/pybind11.h>

#include <holoscan/core/arg.hpp>
#include <holoscan/core/condition.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/resource.hpp>
#include <holoscan/logger/logger.hpp>

namespace py = pybind11;

namespace holoscan::radar::python {

// Forward positional *args onto the operator: Condition, Resource, Arg, and
// ArgList (Fragment.from_config). Call this AFTER any explicit typed kwargs
// so YAML-sourced Args override pybind constructor defaults — matching C++
// `make_operator(..., from_config("block"))`.
//
// Takes a raw Operator* — calling shared_from_this() inside the constructor
// doesn't work because the object isn't yet owned by a shared_ptr.
inline void add_conditions_and_resources(holoscan::Operator* op,
                                         const py::args& args) {
  for (const auto& item : args) {
    auto obj = py::cast<py::object>(item);
    if (py::isinstance<holoscan::Condition>(obj)) {
      op->add_arg(obj.cast<std::shared_ptr<holoscan::Condition>>());
      continue;
    }
    if (py::isinstance<holoscan::Resource>(obj)) {
      op->add_arg(obj.cast<std::shared_ptr<holoscan::Resource>>());
      continue;
    }
    if (py::isinstance<holoscan::ArgList>(obj)) {
      op->add_arg(obj.cast<holoscan::ArgList>());
      continue;
    }
    if (py::isinstance<holoscan::Arg>(obj)) {
      op->add_arg(obj.cast<holoscan::Arg>());
      continue;
    }
    HOLOSCAN_LOG_WARN(
        "Operator '{}': positional arg of type '{}' is not a Condition, "
        "Resource, Arg, or ArgList — add_arg skipped.",
        op->name(),
        py::str(obj.get_type()).cast<std::string>());
  }
}

}  // namespace holoscan::radar::python
