// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/fragment.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>

#include "radar_app/report/cadence_meter.hpp"
#include "radar_app/report/report_ops.hpp"
#include "radar_common/python/py_operator_utils.hpp"
#include "radar_common/sdk/arg_setters.hpp"

namespace py = pybind11;

namespace holoscan::radar::app {

namespace {

class PyDetectionReportOp : public DetectionReportOp {
 public:
  using DetectionReportOp::DetectionReportOp;

  PyDetectionReportOp(Fragment* fragment, const py::args& args, DetectionReportMode mode,
                      std::vector<int64_t> point_index_columns,
                      const std::string& name = "detection_report")
  : DetectionReportOp(DetectionReportOp::Config{
        .mode = mode, .point_index_columns = std::move(point_index_columns)}) {
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<OperatorSpec>(fragment);
    setup(*spec_);
    python::add_conditions_and_resources(this, args);
  }
};

class PyNullSinkOp : public NullSinkOp {
 public:
  using NullSinkOp::NullSinkOp;

  PyNullSinkOp(Fragment* fragment, const py::args& args, const std::string& name = "null_sink")
  : NullSinkOp() {
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<OperatorSpec>(fragment);
    setup(*spec_);
    python::add_conditions_and_resources(this, args);
  }
};

void bind_radar_app(py::module_& m) {
  py::class_<CadenceMeter, std::shared_ptr<CadenceMeter>>(
      m, "CadenceMeter", "Wall-clock source / end-to-end cadence probe for the FMCW demos.")
      .def(py::init<>())
      .def_readwrite("frame_period_s", &CadenceMeter::frame_period_s)
      .def_readwrite("slack_s", &CadenceMeter::slack_s)
      .def_readwrite("log_every", &CadenceMeter::log_every)
      .def("effective_slack_s", &CadenceMeter::effective_slack_s)
      .def("budget_s", &CadenceMeter::budget_s)
      .def("on_source_emit", &CadenceMeter::on_source_emit)
      .def("on_sink_done", &CadenceMeter::on_sink_done)
      .def("log_summary", &CadenceMeter::log_summary)
      .def_property_readonly("source_ticks", &CadenceMeter::source_ticks)
      .def_property_readonly("sink_ticks", &CadenceMeter::sink_ticks)
      .def_property_readonly("overdue_emits", &CadenceMeter::overdue_emits)
      .def_property_readonly("late_frames", &CadenceMeter::late_frames);

  py::enum_<DetectionReportMode>(m, "DetectionReportMode")
      .value("RAW_DETECTIONS", DetectionReportMode::kRawDetections)
      .value("CLUSTERS", DetectionReportMode::kClusters);

  py::class_<DetectionReportOp, PyDetectionReportOp, Operator, std::shared_ptr<DetectionReportOp>>(
      m, "DetectionReportOp", "Log raw or clustered detections in physical units.")
      .def(py::init<Fragment*, const py::args&, DetectionReportMode, std::vector<int64_t>,
                    const std::string&>(),
           py::arg("fragment"), py::arg("mode") = DetectionReportMode::kRawDetections,
           py::arg("point_index_columns") = std::vector<int64_t>{},
           py::arg("name") = "detection_report")
      .def("setup", &DetectionReportOp::setup, py::arg("spec"))
      .def_readwrite("cadence", &DetectionReportOp::cadence);

  py::class_<NullSinkOp, PyNullSinkOp, Operator, std::shared_ptr<NullSinkOp>>(
      m, "NullSinkOp", "Drain an input Entity without deserializing as a TensorMap.")
      .def(py::init<Fragment*, const py::args&, const std::string&>(), py::arg("fragment"),
           py::arg("name") = "null_sink")
      .def("setup", &NullSinkOp::setup, py::arg("spec"));
}

}  // namespace

}  // namespace holoscan::radar::app

PYBIND11_MODULE(_radar_app, m) {
  holoscan::radar::ensure_radar_arg_setters();
  m.doc() = "radar_app report operator Python bindings";
  holoscan::radar::app::bind_radar_app(m);
}
