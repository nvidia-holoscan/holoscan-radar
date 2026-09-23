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

#include "fmcw/tdm_sparse_compensation_op/tdm_sparse_compensation_op.hpp"
#include "radar_common/descriptors/enums.hpp"
#include "radar_common/descriptors/fmcw_waveform_descriptor.hpp"
#include "radar_common/descriptors/max_tensor_extents.hpp"
#include "radar_common/descriptors/tx_schedule.hpp"
#include "radar_common/python/py_operator_utils.hpp"
#include "radar_common/sdk/arg_setters.hpp"

namespace py = pybind11;

namespace holoscan::radar {

namespace {

template <typename T>
class PyTDMSparseCompensationOp : public TDMSparseCompensationOp<T> {
 public:
  PyTDMSparseCompensationOp(Fragment* fragment, const py::args& args, int64_t max_tx,
                            int64_t max_rx, int64_t max_virtual_channels, int64_t max_doppler_bins,
                            int64_t max_range_bins, int64_t max_detections, int64_t max_snapshots,
                            const FMCWWaveformDescriptor& waveform = {},
                            const TXSchedule& tx_schedule = {},
                            const std::vector<double>& tx_positions = {},
                            const std::vector<double>& rx_positions = {},
                            OverlapPolicy overlap_policy = OverlapPolicy::kDedup,
                            const std::vector<int64_t>& snapshot_window = {0, 0},
                            SnapshotEdgeMode doppler_edge = SnapshotEdgeMode::kWrap,
                            SnapshotEdgeMode range_edge = SnapshotEdgeMode::kShift,
                            const std::string& name = "tdm_sparse_compensation")
  : TDMSparseCompensationOp<T>([&] {
      MaxTensorExtents extents;
      extents.max_tx = max_tx;
      extents.max_rx = max_rx;
      if (max_virtual_channels > 0) {
        extents.max_virtual_channels = max_virtual_channels;
      }
      extents.max_doppler_bins = max_doppler_bins;
      extents.max_range_bins = max_range_bins;
      extents.max_detections = max_detections;
      extents.max_snapshots = max_snapshots;
      return extents;
    }()) {
    this->name_ = name;
    this->fragment_ = fragment;
    this->spec_ = std::make_shared<OperatorSpec>(fragment);
    this->add_arg(Arg("waveform") = waveform);
    this->add_arg(Arg("tx_schedule") = tx_schedule);
    this->add_arg(Arg("tx_positions") = tx_positions);
    this->add_arg(Arg("rx_positions") = rx_positions);
    this->add_arg(Arg("overlap_policy") = overlap_policy);
    this->add_arg(Arg("snapshot_window") = snapshot_window);
    this->add_arg(Arg("doppler_edge") = doppler_edge);
    this->add_arg(Arg("range_edge") = range_edge);
    this->setup(*this->spec_);
    python::add_conditions_and_resources(this, args);
  }
};

template <typename T>
void bind_tdm_sparse_compensation(py::module_& module, const char* class_name) {
  using Op = TDMSparseCompensationOp<T>;
  using PyOp = PyTDMSparseCompensationOp<T>;

  py::class_<Op, PyOp, Operator, std::shared_ptr<Op>>(
      module, class_name, "Gather and coherently compensate sparse TDM detections.")
      .def(py::init<Fragment*, const py::args&, int64_t, int64_t, int64_t, int64_t, int64_t,
                    int64_t, int64_t, const FMCWWaveformDescriptor&, const TXSchedule&,
                    const std::vector<double>&, const std::vector<double>&, OverlapPolicy,
                    const std::vector<int64_t>&, SnapshotEdgeMode, SnapshotEdgeMode,
                    const std::string&>(),
           py::arg("fragment"), py::arg("max_tx") = int64_t{8}, py::arg("max_rx") = int64_t{16},
           py::arg("max_virtual_channels") = int64_t{0}, py::arg("max_doppler_bins") = int64_t{512},
           py::arg("max_range_bins") = int64_t{4096}, py::arg("max_detections") = int64_t{4096},
           py::arg("max_snapshots") = int64_t{81}, py::arg("waveform") = FMCWWaveformDescriptor{},
           py::arg("tx_schedule") = TXSchedule{}, py::arg("tx_positions") = std::vector<double>{},
           py::arg("rx_positions") = std::vector<double>{},
           py::arg("overlap_policy") = OverlapPolicy::kDedup,
           py::arg("snapshot_window") = std::vector<int64_t>{0, 0},
           py::arg("doppler_edge") = SnapshotEdgeMode::kWrap,
           py::arg("range_edge") = SnapshotEdgeMode::kShift,
           py::arg("name") = "tdm_sparse_compensation")
      .def("setup", &Op::setup, py::arg("spec"));
}

}  // namespace

}  // namespace holoscan::radar

PYBIND11_MODULE(_tdm_sparse_compensation_op, module) {
  holoscan::radar::ensure_radar_arg_setters();
  module.doc() = "TDMSparseCompensationOp Python bindings";
  holoscan::radar::bind_tdm_sparse_compensation<float>(module, "TDMSparseCompensationOpF32");
  holoscan::radar::bind_tdm_sparse_compensation<double>(module, "TDMSparseCompensationOpF64");
}
