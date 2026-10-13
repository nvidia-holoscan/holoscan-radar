// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <holoscan/core/fragment.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>

#include "detection/cfar_detector_op/cfar_alpha.hpp"
#include "detection/cfar_detector_op/cfar_detector_op.hpp"
#include "radar_common/descriptors/max_tensor_extents.hpp"
#include "radar_common/python/py_operator_utils.hpp"
#include "radar_common/sdk/arg_setters.hpp"

namespace py = pybind11;

namespace holoscan::radar {

namespace {

template <typename T>
class PyCFARDetectorOp : public CFARDetectorOp<T> {
 public:
  using CFARDetectorOp<T>::CFARDetectorOp;

  PyCFARDetectorOp(Fragment* fragment, const py::args& args,
                   CfarVariant cfar_variant = CfarVariant::kCA, int64_t os_rank = -1,
                   const std::vector<int64_t>& guard_cells = {},
                   const std::vector<int64_t>& training_cells = {}, double pfa = 1e-6,
                   const std::vector<double>& alpha_lut = {}, double threshold_scale = 1.0,
                   const std::vector<int64_t>& nci_axes = {}, int64_t preintegrated_looks = 1,
                   int64_t nci_fold = 0, int64_t max_detections = 0,
                   bool blocking_alpha_lut_rebuild = false,
                   CfarPeakMode peak_mode = CfarPeakMode::kNone,
                   const std::vector<int64_t>& peak_radius = {1}, int64_t max_doppler_bins = 512,
                   int64_t max_range_bins = 4096, int64_t max_det = 8192, int64_t max_rx = 1,
                   int64_t max_tx = 0, const std::string& name = "cfar_detector")
  : CFARDetectorOp<T>([&] {
      MaxTensorExtents extents;
      extents.max_doppler_bins = max_doppler_bins;
      extents.max_range_bins = max_range_bins;
      extents.max_detections = max_det;
      if (max_rx > 1) {
        extents.max_rx = max_rx;
      }
      if (max_tx > 0) {
        extents.max_tx = max_tx;
        extents.max_rx = std::max(max_rx, int64_t{1});
      }
      return extents;
    }()) {
    this->name_ = name;
    this->fragment_ = fragment;
    this->spec_ = std::make_shared<OperatorSpec>(fragment);
    this->add_arg(Arg("cfar_variant") = cfar_variant);
    this->add_arg(Arg("os_rank") = os_rank);
    this->add_arg(Arg("guard_cells") = guard_cells);
    this->add_arg(Arg("training_cells") = training_cells);
    this->add_arg(Arg("pfa") = pfa);
    this->add_arg(Arg("alpha_lut") = alpha_lut);
    this->add_arg(Arg("threshold_scale") = threshold_scale);
    this->add_arg(Arg("nci_axes") = nci_axes);
    this->add_arg(Arg("preintegrated_looks") = preintegrated_looks);
    this->add_arg(Arg("nci_fold") = nci_fold);
    this->add_arg(Arg("max_detections") = max_detections);
    this->add_arg(Arg("blocking_alpha_lut_rebuild") = blocking_alpha_lut_rebuild);
    this->add_arg(Arg("peak_mode") = peak_mode);
    this->add_arg(Arg("peak_radius") = peak_radius);
    this->setup(*this->spec_);
    python::add_conditions_and_resources(this, args);
  }
};

template <typename T>
void bind_cfar_detector(py::module_& m, const char* class_name) {
  using Op = CFARDetectorOp<T>;
  using PyOp = PyCFARDetectorOp<T>;

  py::class_<Op, PyOp, Operator, std::shared_ptr<Op>>(
      m, class_name,
      "2-D CFAR detector on the (Doppler, range) plane.\n\n"
      "Parameters\n----------\n"
      "fragment : holoscan.core.Fragment\n"
      "cfar_variant : CfarVariant, optional\n"
      "os_rank : int, optional\n"
      "guard_cells : list[int], optional\n"
      "training_cells : list[int], optional\n"
      "pfa : float, optional\n"
      "alpha_lut : list[float], optional\n"
      "threshold_scale : float, optional\n"
      "nci_axes : list[int], optional\n"
      "preintegrated_looks : int, optional\n"
      "nci_fold : int, optional\n"
      "max_detections : int, optional\n"
      "blocking_alpha_lut_rebuild : bool, optional\n"
      "peak_mode : CfarPeakMode, optional\n"
      "peak_radius : list[int], optional\n"
      "name : str, optional\n")
      .def(py::init<Fragment*, const py::args&, CfarVariant, int64_t, const std::vector<int64_t>&,
                    const std::vector<int64_t>&, double, const std::vector<double>&, double,
                    const std::vector<int64_t>&, int64_t, int64_t, int64_t, bool, CfarPeakMode,
                    const std::vector<int64_t>&, int64_t, int64_t, int64_t, int64_t, int64_t,
                    const std::string&>(),
           py::arg("fragment"), py::arg("cfar_variant") = CfarVariant::kCA,
           py::arg("os_rank") = int64_t{-1}, py::arg("guard_cells") = std::vector<int64_t>{},
           py::arg("training_cells") = std::vector<int64_t>{}, py::arg("pfa") = 1e-6,
           py::arg("alpha_lut") = std::vector<double>{}, py::arg("threshold_scale") = 1.0,
           py::arg("nci_axes") = std::vector<int64_t>{},
           py::arg("preintegrated_looks") = int64_t{1}, py::arg("nci_fold") = int64_t{0},
           py::arg("max_detections") = int64_t{0}, py::arg("blocking_alpha_lut_rebuild") = false,
           py::arg("peak_mode") = CfarPeakMode::kNone,
           py::arg("peak_radius") = std::vector<int64_t>{1},
           py::arg("max_doppler_bins") = int64_t{512}, py::arg("max_range_bins") = int64_t{4096},
           py::arg("max_det") = int64_t{8192}, py::arg("max_rx") = int64_t{1},
           py::arg("max_tx") = int64_t{0}, py::arg("name") = "cfar_detector")
      .def("setup", &Op::setup, py::arg("spec"));
}

void bind_cfar_variant(py::module_& m) {
  py::enum_<CfarVariant>(m, "CfarVariant")
      .value("CA", CfarVariant::kCA)
      .value("GO", CfarVariant::kGO)
      .value("SO", CfarVariant::kSO)
      .value("OS", CfarVariant::kOS);
}

void bind_cfar_peak_mode(py::module_& m) {
  py::enum_<CfarPeakMode>(m, "CfarPeakMode")
      .value("NONE", CfarPeakMode::kNone)
      .value("RANGE_DOPPLER", CfarPeakMode::kRangeDoppler)
      .value("ALL", CfarPeakMode::kAll);
}

}  // namespace

}  // namespace holoscan::radar

PYBIND11_MODULE(_cfar_detector_op, m) {
  holoscan::radar::ensure_radar_arg_setters();
  m.doc() = "CFARDetectorOp Python bindings";
  holoscan::radar::bind_cfar_variant(m);
  holoscan::radar::bind_cfar_peak_mode(m);
  holoscan::radar::bind_cfar_detector<float>(m, "CFARDetectorOpF32");
  holoscan::radar::bind_cfar_detector<double>(m, "CFARDetectorOpF64");

  // Host alpha helpers (TESTING.md / CaAlphaK1MatchesContractClosedForm).
  m.def("ca_cfar_alpha", &holoscan::radar::detail::ca_cfar_alpha, py::arg("pfa"), py::arg("n"),
        py::arg("k") = int64_t{1}, "Invert CA-CFAR PFA→α for N training cells and fold K.");
  m.def("ca_cfar_pfa_from_alpha", &holoscan::radar::detail::ca_cfar_pfa_from_alpha,
        py::arg("alpha"), py::arg("n"), py::arg("k") = int64_t{1},
        "Forward CA-CFAR α→PFA for N training cells and fold K.");

  // Offline / harness LUT builders (same entry points the C++ perf harness
  // and operator cold-path use). Needed so Python can exercise the alpha_lut
  // fast path without blocking Monte Carlo on GO/SO/OS.
  m.def("build_ca_alpha_lut", &holoscan::radar::detail::build_ca_alpha_lut, py::arg("pfa"),
        py::arg("fold"), py::arg("num_doppler"), py::arg("num_range"), py::arg("guard_d"),
        py::arg("guard_r"), py::arg("train_d"), py::arg("train_r"),
        py::call_guard<py::gil_scoped_release>(),
        "Build CA alpha_lut[n] for the given map footprint.");
  m.def("build_go_alpha_lut", &holoscan::radar::detail::build_go_alpha_lut, py::arg("pfa"),
        py::arg("fold"), py::arg("num_doppler"), py::arg("num_range"), py::arg("guard_d"),
        py::arg("guard_r"), py::arg("train_d"), py::arg("train_r"),
        py::call_guard<py::gil_scoped_release>(),
        "Build GO alpha_lut via Monte Carlo (offline / harness use).");
  m.def("build_so_alpha_lut", &holoscan::radar::detail::build_so_alpha_lut, py::arg("pfa"),
        py::arg("fold"), py::arg("num_doppler"), py::arg("num_range"), py::arg("guard_d"),
        py::arg("guard_r"), py::arg("train_d"), py::arg("train_r"),
        py::call_guard<py::gil_scoped_release>(),
        "Build SO alpha_lut via Monte Carlo (offline / harness use).");
  m.def("build_os_alpha_lut", &holoscan::radar::detail::build_os_alpha_lut, py::arg("pfa"),
        py::arg("fold"), py::arg("os_rank"), py::arg("num_doppler"), py::arg("num_range"),
        py::arg("guard_d"), py::arg("guard_r"), py::arg("train_d"), py::arg("train_r"),
        py::call_guard<py::gil_scoped_release>(),
        "Build OS alpha_lut via Monte Carlo (offline / harness use).");
}
