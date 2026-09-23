// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <holoscan/core/fragment.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>

#include "fmcw/virtual_array_assembly_op/virtual_array_assembly_op.hpp"
#include "radar_common/descriptors/enums.hpp"
#include "radar_common/descriptors/max_tensor_extents.hpp"
#include "radar_common/python/py_operator_utils.hpp"
#include "radar_common/sdk/arg_setters.hpp"

namespace py = pybind11;

namespace holoscan::radar {

namespace {

template <typename T>
class PyVirtualArrayAssemblyOp : public VirtualArrayAssemblyOp<T> {
 public:
  using VirtualArrayAssemblyOp<T>::VirtualArrayAssemblyOp;

  PyVirtualArrayAssemblyOp(
      Fragment* fragment, const py::args& args,
      std::int64_t max_tx, std::int64_t max_rx,
      std::int64_t max_virtual_channels,
      std::int64_t max_doppler_bins,
      std::int64_t max_range_bins,
      const std::vector<double>& tx_positions = {},
      const std::vector<double>& rx_positions = {},
      double wavelength = 0.0,
      OverlapPolicy overlap_policy = OverlapPolicy::kDedup,
      const std::string& name = "virtual_array_assembly")
      : VirtualArrayAssemblyOp<T>([&] {
          MaxTensorExtents extents;
          extents.max_tx = max_tx;
          extents.max_rx = max_rx;
          if (max_virtual_channels > 0) {
            extents.max_virtual_channels =
                max_virtual_channels;
          }
          extents.max_doppler_bins = max_doppler_bins;
          extents.max_range_bins = max_range_bins;
          return extents;
        }()) {
    this->name_ = name;
    this->fragment_ = fragment;
    this->spec_ = std::make_shared<OperatorSpec>(fragment);
    this->add_arg(Arg("tx_positions") = tx_positions);
    this->add_arg(Arg("rx_positions") = rx_positions);
    this->add_arg(Arg("wavelength") = wavelength);
    this->add_arg(Arg("overlap_policy") = overlap_policy);
    this->setup(*this->spec_);
    python::add_conditions_and_resources(this, args);
  }
};

template <typename T>
void bind_virtual_array_assembly(py::module_& m, const char* class_name) {
  using Op = VirtualArrayAssemblyOp<T>;
  using PyOp = PyVirtualArrayAssemblyOp<T>;

  py::class_<Op, PyOp, Operator, std::shared_ptr<Op>>(
      m, class_name,
      "Flatten (tx, rx) RD maps into a virtual channel axis.\n\n"
      "Parameters\n----------\n"
      "fragment : holoscan.core.Fragment\n"
      "max_tx : int, optional\n"
      "max_rx : int, optional\n"
      "max_virtual_channels : int, optional\n"
      "max_doppler_bins : int, optional\n"
      "max_range_bins : int, optional\n"
      "tx_positions : list[float], optional\n"
      "rx_positions : list[float], optional\n"
      "wavelength : float, optional\n"
      "overlap_policy : OverlapPolicy, optional\n"
      "name : str, optional\n")
      .def(py::init<Fragment*, const py::args&,
                    std::int64_t, std::int64_t, std::int64_t,
                    std::int64_t, std::int64_t,
                    const std::vector<double>&,
                    const std::vector<double>&, double,
                    OverlapPolicy, const std::string&>(),
           py::arg("fragment"),
           py::arg("max_tx") = std::int64_t{8},
           py::arg("max_rx") = std::int64_t{16},
           py::arg("max_virtual_channels") =
               std::int64_t{0},
           py::arg("max_doppler_bins") =
               std::int64_t{512},
           py::arg("max_range_bins") =
               std::int64_t{4096},
           py::arg("tx_positions") =
               std::vector<double>{},
           py::arg("rx_positions") =
               std::vector<double>{},
           py::arg("wavelength") = 0.0,
           py::arg("overlap_policy") =
               OverlapPolicy::kDedup,
           py::arg("name") =
               "virtual_array_assembly")
      .def("setup", &Op::setup, py::arg("spec"));
}

}  // namespace

}  // namespace holoscan::radar

PYBIND11_MODULE(_virtual_array_assembly_op, m) {
  holoscan::radar::ensure_radar_arg_setters();
  m.doc() = "VirtualArrayAssemblyOp Python bindings";
  holoscan::radar::bind_virtual_array_assembly<float>(
      m, "VirtualArrayAssemblyOpF32");
  holoscan::radar::bind_virtual_array_assembly<double>(
      m, "VirtualArrayAssemblyOpF64");
}
