// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <pybind11/pybind11.h>

#include <cstdint>
#include <memory>
#include <string>

#include <holoscan/core/fragment.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>

#include "fmcw/ddm_power_fold_op/ddm_power_fold_op.hpp"
#include "radar_common/python/py_operator_utils.hpp"
#include "radar_common/sdk/arg_setters.hpp"

namespace py = pybind11;
namespace holoscan::radar {
namespace {

template <typename T>
class PyDDMPowerFoldOp : public DDMPowerFoldOp<T> {
 public:
  PyDDMPowerFoldOp(Fragment* fragment, const py::args& args, int64_t max_rx = 16,
                   int64_t max_doppler_bins = 512, int64_t max_range_bins = 4096,
                   const DDMCodeDescriptor& ddm_code = {},
                   const std::string& name = "ddm_power_fold")
  : DDMPowerFoldOp<T>([&] {
      MaxTensorExtents extents;
      extents.max_rx = max_rx;
      extents.max_doppler_bins = max_doppler_bins;
      extents.max_range_bins = max_range_bins;
      return extents;
    }()) {
    this->name_ = name;
    this->fragment_ = fragment;
    this->spec_ = std::make_shared<OperatorSpec>(fragment);
    this->add_arg(Arg("ddm_code") = ddm_code);
    this->setup(*this->spec_);
    python::add_conditions_and_resources(this, args);
  }
};

template <typename T>
void bind(py::module_& module, const char* name) {
  using Op = DDMPowerFoldOp<T>;
  using PyOp = PyDDMPowerFoldOp<T>;
  py::class_<Op, PyOp, Operator, std::shared_ptr<Op>>(module, name)
      .def(py::init<Fragment*, const py::args&, int64_t, int64_t, int64_t, const DDMCodeDescriptor&,
                    const std::string&>(),
           py::arg("fragment"), py::arg("max_rx") = 16, py::arg("max_doppler_bins") = 512,
           py::arg("max_range_bins") = 4096, py::arg("ddm_code") = DDMCodeDescriptor{},
           py::arg("name") = "ddm_power_fold")
      .def("setup", &Op::setup, py::arg("spec"));
}

}  // namespace
}  // namespace holoscan::radar

PYBIND11_MODULE(_ddm_power_fold_op, module) {
  holoscan::radar::ensure_radar_arg_setters();
  holoscan::radar::bind<float>(module, "DDMPowerFoldOpF32");
  holoscan::radar::bind<double>(module, "DDMPowerFoldOpF64");
}
