// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <pybind11/pybind11.h>

#include <cstdint>
#include <memory>
#include <string>

#include <holoscan/core/fragment.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>

#include "fmcw/tdm_dense_compensation_op/tdm_dense_compensation_op.hpp"
#include "radar_common/descriptors/fmcw_waveform_descriptor.hpp"
#include "radar_common/descriptors/max_tensor_extents.hpp"
#include "radar_common/descriptors/tx_schedule.hpp"
#include "radar_common/python/py_operator_utils.hpp"
#include "radar_common/sdk/arg_setters.hpp"

namespace py = pybind11;
namespace holoscan::radar {
namespace {
template <typename T>
class PyTDMDenseCompensationOp : public TDMDenseCompensationOp<T> {
 public:
  using TDMDenseCompensationOp<T>::TDMDenseCompensationOp;
  PyTDMDenseCompensationOp(Fragment* fragment, const py::args& args, int64_t max_tx = 8,
                           int64_t max_rx = 16, int64_t max_doppler_bins = 512,
                           int64_t max_range_bins = 4096,
                           const FMCWWaveformDescriptor& waveform = FMCWWaveformDescriptor{},
                           const TXSchedule& tx_schedule = TXSchedule{},
                           const std::string& name = "tdm_dense_compensation")
  : TDMDenseCompensationOp<T>([&] {
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
    this->add_arg(Arg("waveform") = waveform);
    this->add_arg(Arg("tx_schedule") = tx_schedule);
    this->setup(*this->spec_);
    python::add_conditions_and_resources(this, args);
  }
};
template <typename T>
void bind(py::module_& m, const char* name) {
  using Op = TDMDenseCompensationOp<T>;
  using PyOp = PyTDMDenseCompensationOp<T>;
  py::class_<Op, PyOp, Operator, std::shared_ptr<Op>>(m, name)
      .def(py::init<Fragment*, const py::args&, int64_t, int64_t, int64_t, int64_t,
                    const FMCWWaveformDescriptor&, const TXSchedule&, const std::string&>(),
           py::arg("fragment"), py::arg("max_tx") = 8, py::arg("max_rx") = 16,
           py::arg("max_doppler_bins") = 512, py::arg("max_range_bins") = 4096,
           py::arg("waveform") = FMCWWaveformDescriptor{}, py::arg("tx_schedule") = TXSchedule{},
           py::arg("name") = "tdm_dense_compensation")
      .def("setup", &Op::setup, py::arg("spec"));
}
}  // namespace
}  // namespace holoscan::radar
PYBIND11_MODULE(_tdm_dense_compensation_op, m) {
  holoscan::radar::ensure_radar_arg_setters();
  holoscan::radar::bind<float>(m, "TDMDenseCompensationOpF32");
  holoscan::radar::bind<double>(m, "TDMDenseCompensationOpF64");
}
