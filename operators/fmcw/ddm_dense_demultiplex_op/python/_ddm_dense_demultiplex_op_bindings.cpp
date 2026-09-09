// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <pybind11/pybind11.h>

#include <cstdint>
#include <memory>
#include <string>

#include <holoscan/core/fragment.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>

#include "fmcw/ddm_dense_demultiplex_op/ddm_dense_demultiplex_op.hpp"
#include "radar_common/descriptors/ddm_code_descriptor.hpp"
#include "radar_common/descriptors/max_tensor_extents.hpp"
#include "radar_common/python/py_operator_utils.hpp"
#include "radar_common/sdk/arg_setters.hpp"

namespace py = pybind11;
namespace holoscan::radar {
namespace {
template <typename T>
class PyDDMDenseDemultiplexOp : public DDMDenseDemultiplexOp<T> {
 public:
  using DDMDenseDemultiplexOp<T>::DDMDenseDemultiplexOp;
  PyDDMDenseDemultiplexOp(
      Fragment* fragment, const py::args& args, int64_t max_tx = 8,
      int64_t max_rx = 16, int64_t max_doppler_bins = 512,
      int64_t max_range_bins = 4096,
      const DDMCodeDescriptor& ddm_code = DDMCodeDescriptor{},
      const std::string& name = "ddm_dense_demultiplex")
      : DDMDenseDemultiplexOp<T>([&] {
          MaxTensorExtents e;
          e.max_tx = max_tx;
          e.max_rx = max_rx;
          e.max_doppler_bins = max_doppler_bins;
          e.max_range_bins = max_range_bins;
          return e;
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
void bind(py::module_& m, const char* name) {
  using Op = DDMDenseDemultiplexOp<T>;
  using PyOp = PyDDMDenseDemultiplexOp<T>;
  py::class_<Op, PyOp, Operator, std::shared_ptr<Op>>(m, name)
      .def(py::init<Fragment*, const py::args&, int64_t, int64_t, int64_t,
                    int64_t, const DDMCodeDescriptor&, const std::string&>(),
           py::arg("fragment"), py::arg("max_tx") = 8, py::arg("max_rx") = 16,
           py::arg("max_doppler_bins") = 512, py::arg("max_range_bins") = 4096,
           py::arg("ddm_code") = DDMCodeDescriptor{},
           py::arg("name") = "ddm_dense_demultiplex")
      .def("setup", &Op::setup, py::arg("spec"));
}
}  // namespace
}  // namespace holoscan::radar
PYBIND11_MODULE(_ddm_dense_demultiplex_op, m) {
  holoscan::radar::ensure_radar_arg_setters();
  holoscan::radar::bind<float>(m, "DDMDenseDemultiplexOpF32");
  holoscan::radar::bind<double>(m, "DDMDenseDemultiplexOpF64");
}
