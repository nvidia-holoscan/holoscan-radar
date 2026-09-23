// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <pybind11/pybind11.h>

#include <cstdint>
#include <memory>
#include <string>

#include <holoscan/core/fragment.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>

#include "detection/multi_source_expansion_op/multi_source_expansion_op.hpp"
#include "radar_common/descriptors/max_tensor_extents.hpp"
#include "radar_common/python/py_operator_utils.hpp"
#include "radar_common/sdk/arg_setters.hpp"

namespace py = pybind11;

namespace holoscan::radar {
namespace {

template <typename T>
class PyMultiSourceExpansionOp : public MultiSourceExpansionOp<T> {
 public:
  PyMultiSourceExpansionOp(Fragment* fragment, const py::args& args, int64_t max_items = 8192,
                           int64_t max_points = 16384, int64_t max_sources = 2,
                           const std::string& name = "multi_source_expansion")
  : MultiSourceExpansionOp<T>([&] {
      MaxTensorExtents extents;
      extents.max_items = max_items;
      extents.max_points = max_points;
      extents.max_sources = max_sources;
      return extents;
    }()) {
    this->name_ = name;
    this->fragment_ = fragment;
    this->spec_ = std::make_shared<OperatorSpec>(fragment);
    this->setup(*this->spec_);
    python::add_conditions_and_resources(this, args);
  }
};

template <typename T>
void bind_expansion(py::module_& module, const char* class_name) {
  using Op = MultiSourceExpansionOp<T>;
  using PyOp = PyMultiSourceExpansionOp<T>;
  py::class_<Op, PyOp, Operator, std::shared_ptr<Op>>(
      module, class_name, "Expands selected sparse detections into resolved angular sources.")
      .def(py::init<Fragment*, const py::args&, int64_t, int64_t, int64_t, const std::string&>(),
           py::arg("fragment"), py::arg("max_items") = int64_t{8192},
           py::arg("max_points") = int64_t{16384}, py::arg("max_sources") = int64_t{2},
           py::arg("name") = "multi_source_expansion")
      .def("setup", &Op::setup, py::arg("spec"));
}

}  // namespace
}  // namespace holoscan::radar

PYBIND11_MODULE(_multi_source_expansion_op, module) {
  holoscan::radar::ensure_radar_arg_setters();
  module.doc() = "MultiSourceExpansionOp Python bindings";
  holoscan::radar::bind_expansion<float>(module, "MultiSourceExpansionOpF32");
  holoscan::radar::bind_expansion<double>(module, "MultiSourceExpansionOpF64");
}
