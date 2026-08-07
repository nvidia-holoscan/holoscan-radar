// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <string>
#include <vector>

#include <holoscan/core/fragment.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>

#include "detection/feature_assembly_op/feature_assembly_op.hpp"
#include "radar_common/python/py_operator_utils.hpp"
#include "radar_common/sdk/arg_setters.hpp"

namespace py = pybind11;

namespace holoscan::radar {

namespace {

template <typename T>
class PyFeatureAssemblyOp : public FeatureAssemblyOp<T> {
 public:
  using FeatureAssemblyOp<T>::FeatureAssemblyOp;

  PyFeatureAssemblyOp(
      Fragment* fragment, const py::args& args,
      const std::vector<AssemblyFeature>& features,
      const std::string& name = "feature_assembly")
      : FeatureAssemblyOp<T>() {
    this->name_ = name;
    this->fragment_ = fragment;
    this->spec_ = std::make_shared<OperatorSpec>(fragment);
    this->add_arg(Arg("features") = features);
    this->setup(*this->spec_);
    python::add_conditions_and_resources(this, args);
  }
};

template <typename T>
void bind_feature_assembly(py::module_& m, const char* class_name) {
  using Op = FeatureAssemblyOp<T>;
  using PyOp = PyFeatureAssemblyOp<T>;

  py::class_<Op, PyOp, Operator, std::shared_ptr<Op>>(
      m, class_name,
      "Assembles the points feature matrix for clustering.\n\n"
      "Parameters\n----------\n"
      "fragment : holoscan.core.Fragment\n"
      "features : list[AssemblyFeature]\n"
      "name : str, optional\n")
      .def(py::init<Fragment*, const py::args&,
                    const std::vector<AssemblyFeature>&, const std::string&>(),
           py::arg("fragment"),
           py::arg("features"),
           py::arg("name") = "feature_assembly")
      .def("setup", &Op::setup, py::arg("spec"));
}

void bind_assembly_feature(py::module_& m) {
  py::enum_<AssemblyFeature>(m, "AssemblyFeature")
      .value("RANGE", AssemblyFeature::kRange)
      .value("VELOCITY", AssemblyFeature::kVelocity)
      .value("SCORE", AssemblyFeature::kScore)
      .value("AZ", AssemblyFeature::kAz)
      .value("EL", AssemblyFeature::kEl)
      .value("DOPPLER_BIN", AssemblyFeature::kDopplerBin)
      .value("RANGE_BIN", AssemblyFeature::kRangeBin);
}

}  // namespace

}  // namespace holoscan::radar

PYBIND11_MODULE(_feature_assembly_op, m) {
  holoscan::radar::ensure_radar_arg_setters();
  m.doc() = "FeatureAssemblyOp Python bindings";
  holoscan::radar::bind_assembly_feature(m);
  holoscan::radar::bind_feature_assembly<float>(m, "FeatureAssemblyOpF32");
  holoscan::radar::bind_feature_assembly<double>(m, "FeatureAssemblyOpF64");
}
