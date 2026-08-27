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

#include "radar_app/report/cadence_meter.hpp"
#include "radar_app/report/report_ops.hpp"
#if defined(HOLOSCAN_RADAR_HAS_HOLOVIZ)
#include "radar_app/viz/display.hpp"
#include "radar_app/viz/viz.hpp"
#endif
#include "radar_common/python/py_operator_utils.hpp"
#include "radar_common/sdk/arg_setters.hpp"

namespace py = pybind11;

namespace holoscan::radar::app {

namespace {

class PyDetectionReportOp : public DetectionReportOp {
 public:
  using DetectionReportOp::DetectionReportOp;

  PyDetectionReportOp(Fragment* fragment, const py::args& args,
                      const std::string& name = "detection_report")
      : DetectionReportOp() {
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<OperatorSpec>(fragment);
    setup(*spec_);
    python::add_conditions_and_resources(this, args);
  }
};

class PyClusterReportOp : public ClusterReportOp {
 public:
  using ClusterReportOp::ClusterReportOp;

  PyClusterReportOp(Fragment* fragment, const py::args& args,
                    const std::string& name = "cluster_report")
      : ClusterReportOp() {
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

  PyNullSinkOp(Fragment* fragment, const py::args& args,
               const std::string& name = "null_sink")
      : NullSinkOp() {
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<OperatorSpec>(fragment);
    setup(*spec_);
    python::add_conditions_and_resources(this, args);
  }
};

#if defined(HOLOSCAN_RADAR_HAS_HOLOVIZ)

class PyRdChannelSliceOp : public RdChannelSliceOp {
 public:
  using RdChannelSliceOp::RdChannelSliceOp;

  PyRdChannelSliceOp(Fragment* fragment, const py::args& args,
                     int64_t channel = 0,
                     const std::string& name = "rd_channel_slice")
      : RdChannelSliceOp() {
    name_ = name;
    fragment_ = fragment;
    this->channel = channel;
    spec_ = std::make_shared<OperatorSpec>(fragment);
    setup(*spec_);
    python::add_conditions_and_resources(this, args);
  }
};

class PyRdAngleMaxOp : public RdAngleMaxOp {
 public:
  using RdAngleMaxOp::RdAngleMaxOp;

  PyRdAngleMaxOp(Fragment* fragment, const py::args& args,
                 const std::string& name = "rd_angle_max")
      : RdAngleMaxOp() {
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<OperatorSpec>(fragment);
    setup(*spec_);
    python::add_conditions_and_resources(this, args);
  }
};

class PyRdMapToImageOp : public RdMapToImageOp {
 public:
  using RdMapToImageOp::RdMapToImageOp;

  PyRdMapToImageOp(Fragment* fragment, const py::args& args,
                   const std::string& name = "rd_map_to_image")
      : RdMapToImageOp() {
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<OperatorSpec>(fragment);
    setup(*spec_);
    python::add_conditions_and_resources(this, args);
  }
};

class PyClusterOverlayOp : public ClusterOverlayOp {
 public:
  using ClusterOverlayOp::ClusterOverlayOp;

  PyClusterOverlayOp(Fragment* fragment, const py::args& args,
                     const std::string& name = "cluster_overlay")
      : ClusterOverlayOp() {
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<OperatorSpec>(fragment);
    setup(*spec_);
    python::add_conditions_and_resources(this, args);
  }
};

class PyRollingPlotOp : public RollingPlotOp {
 public:
  using RollingPlotOp::RollingPlotOp;

  PyRollingPlotOp(Fragment* fragment, const py::args& args,
                  int64_t window = 120, double frame_period = 0.1,
                  const std::string& name = "rolling_plot")
      : RollingPlotOp() {
    name_ = name;
    fragment_ = fragment;
    this->window = window;
    this->frame_period = frame_period;
    spec_ = std::make_shared<OperatorSpec>(fragment);
    setup(*spec_);
    python::add_conditions_and_resources(this, args);
  }
};

class PyBoresightViewOp : public BoresightViewOp {
 public:
  using BoresightViewOp::BoresightViewOp;

  PyBoresightViewOp(Fragment* fragment, const py::args& args,
                    double az_lo = -0.35, double az_hi = 0.35,
                    double el_lo = -0.1, double el_hi = 0.1,
                    const std::string& name = "boresight_view")
      : BoresightViewOp() {
    name_ = name;
    fragment_ = fragment;
    this->az_lo = az_lo;
    this->az_hi = az_hi;
    this->el_lo = el_lo;
    this->el_hi = el_hi;
    spec_ = std::make_shared<OperatorSpec>(fragment);
    setup(*spec_);
    python::add_conditions_and_resources(this, args);
  }
};

#endif  // HOLOSCAN_RADAR_HAS_HOLOVIZ

void bind_radar_app(py::module_& m) {
  py::class_<CadenceMeter, std::shared_ptr<CadenceMeter>>(
      m, "CadenceMeter",
      "Wall-clock source / end-to-end cadence probe for the FMCW demos.")
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

  py::class_<DetectionReportOp, PyDetectionReportOp, Operator,
             std::shared_ptr<DetectionReportOp>>(
      m, "DetectionReportOp",
      "Log CFAR detections in physical units via axis bins.")
      .def(py::init<Fragment*, const py::args&, const std::string&>(),
           py::arg("fragment"), py::arg("name") = "detection_report")
      .def("setup", &DetectionReportOp::setup, py::arg("spec"))
      .def_readwrite("cadence", &DetectionReportOp::cadence);

  py::class_<ClusterReportOp, PyClusterReportOp, Operator,
             std::shared_ptr<ClusterReportOp>>(
      m, "ClusterReportOp",
      "Log DBSCAN clusters (centroids + representative cells).")
      .def(py::init<Fragment*, const py::args&, const std::string&>(),
           py::arg("fragment"), py::arg("name") = "cluster_report")
      .def("setup", &ClusterReportOp::setup, py::arg("spec"));

  py::class_<NullSinkOp, PyNullSinkOp, Operator, std::shared_ptr<NullSinkOp>>(
      m, "NullSinkOp",
      "Drain an input Entity without deserializing as a TensorMap.")
      .def(py::init<Fragment*, const py::args&, const std::string&>(),
           py::arg("fragment"), py::arg("name") = "null_sink")
      .def("setup", &NullSinkOp::setup, py::arg("spec"));

#if defined(HOLOSCAN_RADAR_HAS_HOLOVIZ)

  py::class_<RdChannelSliceOp, PyRdChannelSliceOp, Operator,
             std::shared_ptr<RdChannelSliceOp>>(
      m, "RdChannelSliceOp",
      "Slice one rank-2 (doppler, range) plane out of a batched map. Use "
      "`channel` for a rank-3 (N_ch, D, R) input, or `leading_index` (one "
      "entry per leading axis) for higher ranks such as the dense path's "
      "(az, el, D, R) cube.")
      .def(py::init<Fragment*, const py::args&, int64_t, const std::string&>(),
           py::arg("fragment"), py::arg("channel") = 0,
           py::arg("name") = "rd_channel_slice")
      .def_readwrite("channel", &RdChannelSliceOp::channel)
      .def_readwrite("leading_index", &RdChannelSliceOp::leading_index)
      .def("setup", &RdChannelSliceOp::setup, py::arg("spec"));

  py::class_<RdAngleMaxOp, PyRdAngleMaxOp, Operator,
             std::shared_ptr<RdAngleMaxOp>>(
      m, "RdAngleMaxOp",
      "Max-project a (..., D, R) cube down all its leading axes to one "
      "(D, R) float32 power plane. Drop-in alternative to RdChannelSliceOp "
      "for the dense path, where the leading axes are a beam grid: it matches "
      "the max-over-beams projection AnglePeakPickOp already applies to the "
      "detections, at the cost of a higher noise floor.")
      .def(py::init<Fragment*, const py::args&, const std::string&>(),
           py::arg("fragment"), py::arg("name") = "rd_angle_max")
      .def("setup", &RdAngleMaxOp::setup, py::arg("spec"));

  py::class_<RdMapToImageOp, PyRdMapToImageOp, Operator,
             std::shared_ptr<RdMapToImageOp>>(
      m, "RdMapToImageOp",
      "Convert a complex RD map or a real power surface (CFAR cfar_power) "
      "to a log-magnitude float image for Holoviz.")
      .def(py::init<Fragment*, const py::args&, const std::string&>(),
           py::arg("fragment"), py::arg("name") = "rd_map_to_image")
      .def("setup", &RdMapToImageOp::setup, py::arg("spec"));

  py::class_<ClusterOverlayOp, PyClusterOverlayOp, Operator,
             std::shared_ptr<ClusterOverlayOp>>(
      m, "ClusterOverlayOp",
      "Emit Holoviz filled-disc layers for CFAR detections by cluster.")
      .def(py::init<Fragment*, const py::args&, const std::string&>(),
           py::arg("fragment"), py::arg("name") = "cluster_overlay")
      .def_readwrite("view_aspect", &ClusterOverlayOp::view_aspect)
      .def_readwrite("frame_period", &ClusterOverlayOp::frame_period)
      .def_readwrite("rd_region", &ClusterOverlayOp::rd_region)
      .def_readwrite("rd_title", &ClusterOverlayOp::rd_title)
      .def_readwrite("rd_view_px_height", &ClusterOverlayOp::rd_view_px_height)
      .def_readwrite("marker_radius", &ClusterOverlayOp::marker_radius)
      .def_readwrite("marker_outline", &ClusterOverlayOp::marker_outline)
      .def_readwrite("marker_segments", &ClusterOverlayOp::marker_segments)
      .def("setup", &ClusterOverlayOp::setup, py::arg("spec"));

  // Panel placement (window-normalized view
  // rect) for the primitive side panels.
  py::class_<PanelRect>(m, "PanelRect", "Window-normalized panel view rect.")
      .def(py::init([](float x, float y, float w, float h) {
             return PanelRect{x, y, w, h};
           }),
           py::arg("x") = 0.0F, py::arg("y") = 0.0F, py::arg("w") = 1.0F,
           py::arg("h") = 1.0F)
      .def_readwrite("x", &PanelRect::x)
      .def_readwrite("y", &PanelRect::y)
      .def_readwrite("w", &PanelRect::w)
      .def_readwrite("h", &PanelRect::h);

  py::class_<RollingPlotOp, PyRollingPlotOp, Operator,
             std::shared_ptr<RollingPlotOp>>(
      m, "RollingPlotOp",
      "Rolling range/velocity centroid plots for Holoviz side panels.")
      .def(py::init<Fragment*, const py::args&, int64_t, double,
                    const std::string&>(),
           py::arg("fragment"), py::arg("window") = 120,
           py::arg("frame_period") = 0.1, py::arg("name") = "rolling_plot")
      .def_readwrite("window", &RollingPlotOp::window)
      .def_readwrite("frame_period", &RollingPlotOp::frame_period)
      .def_readwrite("velocity_ylim_lo", &RollingPlotOp::velocity_ylim_lo)
      .def_readwrite("velocity_ylim_hi", &RollingPlotOp::velocity_ylim_hi)
      .def_readwrite("primitive", &RollingPlotOp::primitive)
      .def_readwrite("range_view", &RollingPlotOp::range_view)
      .def_readwrite("velocity_view", &RollingPlotOp::velocity_view)
      .def_readwrite("range_aspect", &RollingPlotOp::range_aspect)
      .def_readwrite("velocity_aspect", &RollingPlotOp::velocity_aspect)
      .def_readwrite("marker_radius", &RollingPlotOp::marker_radius)
      .def_readwrite("marker_outline", &RollingPlotOp::marker_outline)
      .def_readwrite("marker_segments", &RollingPlotOp::marker_segments)
      .def("setup", &RollingPlotOp::setup, py::arg("spec"));

  py::class_<BoresightViewOp, PyBoresightViewOp, Operator,
             std::shared_ptr<BoresightViewOp>>(
      m, "BoresightViewOp",
      "Azimuth/elevation boresight scatter panel for Holoviz.")
      .def(py::init<Fragment*, const py::args&, double, double, double, double,
                    const std::string&>(),
           py::arg("fragment"), py::arg("az_lo") = -0.35,
           py::arg("az_hi") = 0.35, py::arg("el_lo") = -0.1,
           py::arg("el_hi") = 0.1, py::arg("name") = "boresight_view")
      .def_readwrite("az_lo", &BoresightViewOp::az_lo)
      .def_readwrite("az_hi", &BoresightViewOp::az_hi)
      .def_readwrite("el_lo", &BoresightViewOp::el_lo)
      .def_readwrite("el_hi", &BoresightViewOp::el_hi)
      .def_readwrite("primitive", &BoresightViewOp::primitive)
      .def_readwrite("view", &BoresightViewOp::view)
      .def_readwrite("aspect", &BoresightViewOp::aspect)
      .def_readwrite("marker_radius", &BoresightViewOp::marker_radius)
      .def_readwrite("marker_outline", &BoresightViewOp::marker_outline)
      .def_readwrite("marker_segments", &BoresightViewOp::marker_segments)
      .def("setup", &BoresightViewOp::setup, py::arg("spec"));

  m.def("gui_available", &gui_available,
        "Probe whether Holoviz can open a display (safe to call before run).");
  m.def("make_turbo_lut", &make_turbo_lut,
        "kLutSize-entry RGBA turbo colormap for HolovizOp color_lut.");
  m.def("cluster_palette", &cluster_palette,
        "Categorical RGBA palette for cluster-overlay disc layers.");

#endif  // HOLOSCAN_RADAR_HAS_HOLOVIZ
}

}  // namespace

}  // namespace holoscan::radar::app

PYBIND11_MODULE(_radar_app, m) {
  holoscan::radar::ensure_radar_arg_setters();
  m.doc() = "radar_app report / viz operator Python bindings";
  holoscan::radar::app::bind_radar_app(m);
}
