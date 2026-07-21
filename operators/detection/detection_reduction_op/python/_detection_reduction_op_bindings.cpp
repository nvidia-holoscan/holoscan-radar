// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <string>
#include <vector>

#include <holoscan/core/fragment.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>

#include "detection/detection_reduction_op/detection_reduction_op.hpp"
#include "radar_common/python/py_operator_utils.hpp"
#include "radar_common/reduce/reduction_enums.hpp"
#include "radar_common/sdk/arg_setters.hpp"

namespace py = pybind11;

namespace holoscan::radar {

namespace {

template <typename T>
class PyDetectionReductionOp : public DetectionReductionOp<T, NoCustomScore> {
 public:
  using DetectionReductionOp<T, NoCustomScore>::DetectionReductionOp;

  PyDetectionReductionOp(
      Fragment* fragment, const py::args& args,
      RepresentativePolicy representative = RepresentativePolicy::kPeak,
      ScoreMethod score_method = ScoreMethod::kWeightedSum,
      const std::vector<double>& feature_weights = {},
      const std::string& name = "detection_reduction")
      : DetectionReductionOp<T, NoCustomScore>() {
    this->name_ = name;
    this->fragment_ = fragment;
    this->spec_ = std::make_shared<OperatorSpec>(fragment);
    this->add_arg(Arg("representative") = representative);
    this->add_arg(Arg("score_method") = score_method);
    this->add_arg(Arg("feature_weights") = feature_weights);
    this->setup(*this->spec_);
    python::add_conditions_and_resources(this, args);
  }
};

template <typename T>
void bind_detection_reduction(py::module_& m, const char* class_name) {
  using Op = DetectionReductionOp<T, NoCustomScore>;
  using PyOp = PyDetectionReductionOp<T>;

  py::class_<Op, PyOp, Operator, std::shared_ptr<Op>>(
      m, class_name,
      "Reduces each cluster to one representative detection cell.\n\n"
      "Parameters\n----------\n"
      "fragment : holoscan.core.Fragment\n"
      "representative : RepresentativePolicy, optional\n"
      "score_method : ScoreMethod, optional\n"
      "feature_weights : list[float], optional\n"
      "name : str, optional\n")
      .def(py::init<Fragment*, const py::args&, RepresentativePolicy,
                    ScoreMethod, const std::vector<double>&,
                    const std::string&>(),
           py::arg("fragment"),
           py::arg("representative") = RepresentativePolicy::kPeak,
           py::arg("score_method") = ScoreMethod::kWeightedSum,
           py::arg("feature_weights") = std::vector<double>{},
           py::arg("name") = "detection_reduction")
      .def("setup", &Op::setup, py::arg("spec"));
}

}  // namespace

}  // namespace holoscan::radar

PYBIND11_MODULE(_detection_reduction_op, m) {
  holoscan::radar::ensure_radar_arg_setters();
  m.doc() = "DetectionReductionOp Python bindings (NoCustomScore)";
  holoscan::radar::bind_detection_reduction<float>(
      m, "DetectionReductionOpF32");
  holoscan::radar::bind_detection_reduction<double>(
      m, "DetectionReductionOpF64");
}
