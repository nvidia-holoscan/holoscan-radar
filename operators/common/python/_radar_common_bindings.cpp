// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <pybind11/complex.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <array>
#include <complex>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <holoscan/python/core/emitter_receiver_registry.hpp>

#include "radar_common/descriptors/angle_convention.hpp"
#include "radar_common/descriptors/array_geometry.hpp"
#include "radar_common/descriptors/coordinate_frame.hpp"
#include "radar_common/descriptors/ddm_code_descriptor.hpp"
#include "radar_common/descriptors/enums.hpp"
#include "radar_common/descriptors/fmcw_waveform_descriptor.hpp"
#include "radar_common/descriptors/lfm_waveform_descriptor.hpp"
#include "radar_common/descriptors/sample_geometry.hpp"
#include "radar_common/descriptors/tx_schedule.hpp"
#include "radar_common/descriptors/window_spec.hpp"
#include "radar_common/reduce/reduction_enums.hpp"
#include "radar_common/sdk/arg_setters.hpp"
#include "radar_common/tensor/dlpack_buffer.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_radar_common, m) {
  using namespace holoscan::radar;  // NOLINT(build/namespaces)

  m.doc() = "holoscan.radar shared descriptors and enums";

  ensure_radar_arg_setters();

  py::enum_<WindowType>(m, "WindowType")
      .value("NONE", WindowType::kNone)
      .value("HANN", WindowType::kHann)
      .value("HAMMING", WindowType::kHamming)
      .value("BLACKMAN", WindowType::kBlackman)
      .value("BLACKMAN_HARRIS", WindowType::kBlackmanHarris)
      .value("KAISER", WindowType::kKaiser)
      .value("CHEBYSHEV", WindowType::kChebyshev)
      .value("TAYLOR", WindowType::kTaylor)
      .value("FLAT_TOP", WindowType::kFlatTop)
      .value("TUKEY", WindowType::kTukey)
      .value("CUSTOM", WindowType::kCustom);

  py::enum_<FftNorm>(m, "FftNorm")
      .value("BACKWARD", FftNorm::kBackward)
      .value("ORTHO", FftNorm::kOrtho)
      .value("FORWARD", FftNorm::kForward);

  py::enum_<ClutterFilterMethod>(m, "ClutterFilterMethod")
      .value("NONE", ClutterFilterMethod::kNone)
      .value("MEAN_SUBTRACTION", ClutterFilterMethod::kMeanSubtraction)
      .value("SINGLE_DELAY", ClutterFilterMethod::kSingleDelay)
      .value("DOUBLE_DELAY", ClutterFilterMethod::kDoubleDelay);

  py::enum_<SnapshotEdgeMode>(m, "SnapshotEdgeMode")
      .value("WRAP", SnapshotEdgeMode::kWrap)
      .value("CLAMP", SnapshotEdgeMode::kClamp)
      .value("SHIFT", SnapshotEdgeMode::kShift);

  py::enum_<ArrayGeometryType>(m, "ArrayGeometryType")
      .value("ULA", ArrayGeometryType::kUla)
      .value("URA", ArrayGeometryType::kUra)
      .value("UCA", ArrayGeometryType::kUca)
      .value("CONFORMAL", ArrayGeometryType::kConformal);

  py::enum_<AngleConventionPreset>(m, "AngleConventionPreset")
      .value("RADAR_BORESIGHT", AngleConventionPreset::kRadarBoresight)
      .value("ROS_REP103", AngleConventionPreset::kRosRep103)
      .value("ISO8855_FLU", AngleConventionPreset::kIso8855Flu)
      .value("AEROSPACE_NED", AngleConventionPreset::kAerospaceNed)
      .value("PHYSICS_SPHERICAL", AngleConventionPreset::kPhysicsSpherical)
      .value("GEODETIC_ENU", AngleConventionPreset::kGeodeticEnu)
      .value("GEODETIC_NED", AngleConventionPreset::kGeodeticNed)
      .value("DIRECTION_COSINE", AngleConventionPreset::kDirectionCosine)
      .value("CUSTOM", AngleConventionPreset::kCustom);

  py::enum_<SignedAxis>(m, "SignedAxis")
      .value("PLUS_X", SignedAxis::kPlusX)
      .value("MINUS_X", SignedAxis::kMinusX)
      .value("PLUS_Y", SignedAxis::kPlusY)
      .value("MINUS_Y", SignedAxis::kMinusY)
      .value("PLUS_Z", SignedAxis::kPlusZ)
      .value("MINUS_Z", SignedAxis::kMinusZ);

  py::enum_<ElevationReference>(m, "ElevationReference")
      .value("HORIZON", ElevationReference::kHorizon)
      .value("ZENITH", ElevationReference::kZenith);

  py::enum_<PeakInterpolation>(m, "PeakInterpolation")
      .value("NONE", PeakInterpolation::kNone)
      .value("PARABOLIC", PeakInterpolation::kParabolic);

  py::enum_<OverlapPolicy>(m, "OverlapPolicy")
      .value("DEDUP", OverlapPolicy::kDedup)
      .value("AVERAGE", OverlapPolicy::kAverage)
      .value("KEEP_ALL", OverlapPolicy::kKeepAll);

  py::enum_<MultiplexingScheme>(m, "MultiplexingScheme")
      .value("UNSPECIFIED", MultiplexingScheme::kUnspecified)
      .value("TDM", MultiplexingScheme::kTdm)
      .value("DDM", MultiplexingScheme::kDdm);

  // UNSPECIFIED is exposed because it is the CoordinateFrameSpec default a
  // Python caller starts from; constructing a CoordinateFrame from it raises.
  py::enum_<CoordinateFrameType>(m, "CoordinateFrameType")
      .value("UNSPECIFIED", CoordinateFrameType::kUnspecified)
      .value("ECEF", CoordinateFrameType::kEcef)
      .value("ENU", CoordinateFrameType::kEnu)
      .value("NED", CoordinateFrameType::kNed)
      .value("SCENE_LOCAL", CoordinateFrameType::kSceneLocal);

  py::enum_<Ellipsoid>(m, "Ellipsoid").value("WGS84", Ellipsoid::kWgs84);

  py::enum_<FormationControl>(m, "FormationControl")
      .value("EMIT_AND_CONTINUE", FormationControl::kEmitAndContinue)
      .value("EMIT_AND_RESET", FormationControl::kEmitAndReset)
      .value("RESET_WITHOUT_EMIT", FormationControl::kResetWithoutEmit)
      .value("RESET_THEN_EMIT", FormationControl::kResetThenEmit);

  py::enum_<RangeInterpolation>(m, "RangeInterpolation")
      .value("NEAREST", RangeInterpolation::kNearest)
      .value("LINEAR", RangeInterpolation::kLinear)
      .value("SINC", RangeInterpolation::kSinc);

  py::enum_<BackprojectionBackend>(m, "BackprojectionBackend")
      .value("MATX_DOUBLE", BackprojectionBackend::kMatxDouble)
      .value("MATX_DOUBLE_PHASE_LUT",
             BackprojectionBackend::kMatxDoublePhaseLut)
      .value("MATX_MIXED", BackprojectionBackend::kMatxMixed)
      .value("MATX_MIXED_PHASE_LUT",
             BackprojectionBackend::kMatxMixedPhaseLut)
      .value("MATX_FLOAT_FLOAT", BackprojectionBackend::kMatxFloatFloat)
      .value("MATX_FLOAT", BackprojectionBackend::kMatxFloat)
      .value("MATX_FLOAT_PHASE_LUT",
             BackprojectionBackend::kMatxFloatPhaseLut)
      .value("MATX_TAYLOR_FAST", BackprojectionBackend::kMatxTaylorFast)
      .value("NVRADAR_CUDA_FP32", BackprojectionBackend::kNvradarCudaFp32)
      .value("NVRADAR_CUDA_FP32_HQ",
             BackprojectionBackend::kNvradarCudaFp32Hq)
      .value("NVRADAR_CUDA_FP32_HQ_FAST",
             BackprojectionBackend::kNvradarCudaFp32HqFast)
      .value("NVRADAR_FP64_REFERENCE",
             BackprojectionBackend::kNvradarFp64Reference)
      .value("NVRADAR_BLOCK_ANCHORED",
             BackprojectionBackend::kNvradarBlockAnchored)
      .value("NVRADAR_BLOCK_ANCHORED_FP64_TRAJECTORY",
             BackprojectionBackend::kNvradarBlockAnchoredFp64Trajectory)
      .value("NVRADAR_BLOCK_ANCHORED_REFINED_THROUGHPUT",
             BackprojectionBackend::kNvradarBlockAnchoredRefinedThroughput)
      .value("NVRADAR_BLOCK_ANCHORED_REFINED_QUALITY",
             BackprojectionBackend::kNvradarBlockAnchoredRefinedQuality)
      .value("NVRADAR_BLOCK_ANCHORED_EXPANDED_THROUGHPUT",
             BackprojectionBackend::kNvradarBlockAnchoredExpandedThroughput)
      .value("NVRADAR_BLOCK_ANCHORED_EXPANDED_QUALITY",
             BackprojectionBackend::kNvradarBlockAnchoredExpandedQuality)
      .value("NVRADAR_BLOCK_ANCHORED_PHASE_LUT",
             BackprojectionBackend::kNvradarBlockAnchoredPhaseLut)
      .value("NVRADAR_BLOCK_ANCHORED_REFINED_QUALITY_PHASE_LUT",
             BackprojectionBackend::
                 kNvradarBlockAnchoredRefinedQualityPhaseLut)
      .value("NVRADAR_BLOCK_ANCHORED_EXPANDED_THROUGHPUT_PHASE_LUT",
             BackprojectionBackend::
                 kNvradarBlockAnchoredExpandedThroughputPhaseLut)
      .value("NVRADAR_BLOCK_ANCHORED_EXPANDED_THROUGHPUT_PHASE_LUT_VEC",
             BackprojectionBackend::
                 kNvradarBlockAnchoredExpandedThroughputPhaseLutVec)
      .value("NVRADAR_BLOCK_ANCHORED_EXPANDED_THROUGHPUT_PHASE_LUT_VEC_Z0",
             BackprojectionBackend::
                 kNvradarBlockAnchoredExpandedThroughputPhaseLutVecZ0)
      .value("NVRADAR_BLOCK_ANCHORED_EXPANDED_QUALITY_PHASE_LUT",
             BackprojectionBackend::
                 kNvradarBlockAnchoredExpandedQualityPhaseLut)
      .value("NVRADAR_BLOCK_ANCHORED_PHASE_LUT_LEGACY",
             BackprojectionBackend::kNvradarBlockAnchoredPhaseLutLegacy)
      .value("NVRADAR_BLOCK_ANCHORED_FP64_TRAJECTORY_PHASE_LUT",
             BackprojectionBackend::
                 kNvradarBlockAnchoredFp64TrajectoryPhaseLut);

  py::enum_<GeometryChange>(m, "GeometryChange")
      .value("RESET", GeometryChange::kReset)
      .value("SCROLL", GeometryChange::kScroll);

  py::enum_<ApertureEmitMode>(m, "ApertureEmitMode")
      .value("CONTINUE", ApertureEmitMode::kContinue)
      .value("RESET", ApertureEmitMode::kReset)
      .value("REFORM", ApertureEmitMode::kReform);

  py::enum_<PhaseCorrectionSense>(m, "PhaseCorrectionSense")
      .value("UNSPECIFIED", PhaseCorrectionSense::kUnspecified)
      .value("SUBTRACT", PhaseCorrectionSense::kSubtract)
      .value("ADD", PhaseCorrectionSense::kAdd);

  py::enum_<RepresentativePolicy>(m, "RepresentativePolicy")
      .value("PEAK", RepresentativePolicy::kPeak)
      .value("MEDOID", RepresentativePolicy::kMedoid)
      .value("FIRST", RepresentativePolicy::kFirst);

  py::enum_<ScoreMethod>(m, "ScoreMethod")
      .value("WEIGHTED_SUM", ScoreMethod::kWeightedSum)
      .value("WEIGHTED_L2", ScoreMethod::kWeightedL2)
      .value("WEIGHTED_MAX", ScoreMethod::kWeightedMax)
      .value("CUSTOM", ScoreMethod::kCustom);

  py::class_<WindowSpec>(m, "WindowSpec")
      .def(py::init<>())
      .def_readwrite("type", &WindowSpec::type)
      .def_readwrite("sidelobe_attenuation_db",
                     &WindowSpec::sidelobe_attenuation_db)
      .def_readwrite("kaiser_beta", &WindowSpec::kaiser_beta)
      .def_readwrite("taylor_nbar", &WindowSpec::taylor_nbar)
      .def_readwrite("tukey_alpha", &WindowSpec::tukey_alpha)
      .def_readwrite("coefficients", &WindowSpec::coefficients)
      .def_readwrite("coherent_gain_normalization",
                     &WindowSpec::coherent_gain_normalization);

  py::class_<ArrayGeometry>(m, "ArrayGeometry")
      .def(py::init<>())
      .def_readwrite("element_positions", &ArrayGeometry::element_positions)
      .def_readwrite("wavelength", &ArrayGeometry::wavelength)
      .def_readwrite("geometry_type", &ArrayGeometry::geometry_type)
      .def("num_elements", &ArrayGeometry::num_elements)
      .def("validate", &ArrayGeometry::validate);

  py::class_<AngleConvention>(m, "AngleConvention")
      .def(py::init<>())
      .def_readwrite("preset", &AngleConvention::preset)
      .def_readwrite("boresight_axis", &AngleConvention::boresight_axis)
      .def_readwrite("azimuth_positive_axis",
                     &AngleConvention::azimuth_positive_axis)
      .def_readwrite("elevation_positive_axis",
                     &AngleConvention::elevation_positive_axis)
      .def_readwrite("elevation_reference",
                     &AngleConvention::elevation_reference)
      .def("uses_direction_cosine", &AngleConvention::uses_direction_cosine);

  py::class_<TXSchedule>(m, "TXSchedule")
      .def(py::init<>())
      .def_readwrite("num_tx", &TXSchedule::num_tx)
      .def_readwrite("tx_order", &TXSchedule::tx_order)
      .def_readwrite("chirp_tx_index", &TXSchedule::chirp_tx_index)
      .def("validate", &TXSchedule::validate);

  py::class_<DDMCodeDescriptor>(m, "DDMCodeDescriptor")
      .def(py::init<>())
      .def_readwrite("num_tx", &DDMCodeDescriptor::num_tx)
      .def_readwrite("phase_codes", &DDMCodeDescriptor::phase_codes)
      .def_readwrite("doppler_offsets", &DDMCodeDescriptor::doppler_offsets)
      .def_readwrite("guard_bands", &DDMCodeDescriptor::guard_bands)
      .def("validate", &DDMCodeDescriptor::validate);

  py::class_<FMCWWaveformSpec>(m, "FMCWWaveformSpec")
      .def(py::init<>())
      .def_readwrite("carrier_frequency", &FMCWWaveformSpec::carrier_frequency)
      .def_readwrite("bandwidth", &FMCWWaveformSpec::bandwidth)
      .def_readwrite("chirp_slope", &FMCWWaveformSpec::chirp_slope)
      .def_readwrite("chirp_duration", &FMCWWaveformSpec::chirp_duration)
      .def_readwrite("sample_rate", &FMCWWaveformSpec::sample_rate)
      .def_readwrite("chirp_repetition_interval",
                     &FMCWWaveformSpec::chirp_repetition_interval)
      .def_readwrite("reference_samples",
                     &FMCWWaveformSpec::reference_samples);

  py::class_<FMCWWaveformDescriptor>(m, "FMCWWaveformDescriptor")
      .def(py::init<>())
      .def(py::init<FMCWWaveformSpec>(), py::arg("spec"))
      .def("is_valid", &FMCWWaveformDescriptor::is_valid)
      .def("carrier_frequency", &FMCWWaveformDescriptor::carrier_frequency)
      .def("bandwidth", &FMCWWaveformDescriptor::bandwidth)
      .def("slope", &FMCWWaveformDescriptor::slope)
      .def("chirp_duration", &FMCWWaveformDescriptor::chirp_duration)
      .def("sample_rate", &FMCWWaveformDescriptor::sample_rate)
      .def("chirp_repetition_interval",
           &FMCWWaveformDescriptor::chirp_repetition_interval)
      .def("wavelength",
           py::overload_cast<>(&FMCWWaveformDescriptor::wavelength, py::const_))
      .def("wavelength",
           py::overload_cast<double>(&FMCWWaveformDescriptor::wavelength,
                                     py::const_),
           py::arg("propagation_speed"))
      .def("reference_samples", &FMCWWaveformDescriptor::reference_samples,
           py::return_value_policy::reference_internal);

  py::class_<CoordinateFrameSpec>(m, "CoordinateFrameSpec")
      .def(py::init<>())
      .def_readwrite("frame", &CoordinateFrameSpec::frame)
      .def_readwrite("origin_llh", &CoordinateFrameSpec::origin_llh)
      .def_readwrite("epoch", &CoordinateFrameSpec::epoch)
      .def_readwrite("ellipsoid", &CoordinateFrameSpec::ellipsoid);

  py::class_<CoordinateFrame>(m, "CoordinateFrame")
      .def(py::init<>())
      .def(py::init<CoordinateFrameSpec>(), py::arg("spec"))
      .def("is_valid", &CoordinateFrame::is_valid)
      .def("frame", &CoordinateFrame::frame)
      .def("origin_llh", &CoordinateFrame::origin_llh)
      .def("epoch", &CoordinateFrame::epoch)
      .def("ellipsoid", &CoordinateFrame::ellipsoid)
      .def("compatible_with", &CoordinateFrame::compatible_with,
           py::arg("other"));

  py::class_<SampleGeometrySpec>(m, "SampleGeometrySpec")
      .def(py::init<>())
      .def_readwrite("origin", &SampleGeometrySpec::origin)
      .def_readwrite("axis_u", &SampleGeometrySpec::axis_u)
      .def_readwrite("axis_v", &SampleGeometrySpec::axis_v)
      .def_readwrite("spacing", &SampleGeometrySpec::spacing)
      .def_readwrite("counts", &SampleGeometrySpec::counts);

  py::class_<SampleGeometry>(m, "SampleGeometry")
      .def(py::init<>())
      .def(py::init<SampleGeometrySpec>(), py::arg("spec"))
      .def("is_valid", &SampleGeometry::is_valid)
      .def("origin", &SampleGeometry::origin)
      .def("axis_u", &SampleGeometry::axis_u)
      .def("axis_v", &SampleGeometry::axis_v)
      .def("spacing", &SampleGeometry::spacing)
      .def("counts", &SampleGeometry::counts)
      .def("num_v", &SampleGeometry::num_v)
      .def("num_u", &SampleGeometry::num_u)
      .def("num_samples", &SampleGeometry::num_samples)
      .def("sample_position", &SampleGeometry::sample_position, py::arg("iv"),
           py::arg("iu"));

  py::class_<LFMWaveformSpec>(m, "LFMWaveformSpec")
      .def(py::init<>())
      .def_readwrite("carrier_frequency", &LFMWaveformSpec::carrier_frequency)
      .def_readwrite("bandwidth", &LFMWaveformSpec::bandwidth)
      .def_readwrite("chirp_slope", &LFMWaveformSpec::chirp_slope)
      .def_readwrite("chirp_duration", &LFMWaveformSpec::chirp_duration)
      .def_readwrite("sample_rate", &LFMWaveformSpec::sample_rate)
      .def_readwrite("chirp_repetition_interval",
                     &LFMWaveformSpec::chirp_repetition_interval)
      .def_readwrite("reference_samples", &LFMWaveformSpec::reference_samples);

  py::class_<LFMWaveformDescriptor>(m, "LFMWaveformDescriptor")
      .def(py::init<>())
      .def(py::init<LFMWaveformSpec>(), py::arg("spec"))
      .def("is_valid", &LFMWaveformDescriptor::is_valid)
      .def("carrier_frequency", &LFMWaveformDescriptor::carrier_frequency)
      .def("bandwidth", &LFMWaveformDescriptor::bandwidth)
      .def("slope", &LFMWaveformDescriptor::slope)
      .def("chirp_duration", &LFMWaveformDescriptor::chirp_duration)
      .def("sample_rate", &LFMWaveformDescriptor::sample_rate)
      .def("chirp_repetition_interval",
           &LFMWaveformDescriptor::chirp_repetition_interval)
      .def("wavelength",
           py::overload_cast<>(&LFMWaveformDescriptor::wavelength, py::const_))
      .def("wavelength",
           py::overload_cast<double>(&LFMWaveformDescriptor::wavelength,
                                     py::const_),
           py::arg("propagation_speed"))
      .def("reference_samples", &LFMWaveformDescriptor::reference_samples,
           py::return_value_policy::reference_internal);

  m.attr("DEFAULT_PROPAGATION_SPEED") = kDefaultPropagationSpeed;

  // Allocate a CUDA holoscan::Tensor (including zero-size shapes). Python's
  // CuPy→TensorMap path labels empty arrays as kDLCPU; operators reject that.
  m.def(
      "make_device_tensor",
      [](const std::vector<int64_t>& shape, const py::object& dtype_obj) {
        std::string name;
        if (py::hasattr(dtype_obj, "name")) {
          name = py::str(dtype_obj.attr("name"));
        } else {
          name = py::str(dtype_obj);
        }
        DLDataType dtype{};
        if (name == "float32" || name == "f4") {
          dtype = DLDataType{kDLFloat, 32, 1};
        } else if (name == "float64" || name == "f8" || name == "float") {
          dtype = DLDataType{kDLFloat, 64, 1};
        } else if (name == "complex64" || name == "c8") {
          dtype = DLDataType{kDLComplex, 64, 1};
        } else if (name == "complex128" || name == "c16" ||
                   name == "complex") {
          dtype = DLDataType{kDLComplex, 128, 1};
        } else if (name == "int64" || name == "i8" || name == "int") {
          dtype = DLDataType{kDLInt, 64, 1};
        } else if (name == "int32" || name == "i4") {
          dtype = DLDataType{kDLInt, 32, 1};
        } else if (name == "bool" || name == "bool_") {
          dtype = DLDataType{kDLBool, 8, 1};
        } else if (name == "uint8" || name == "u1") {
          dtype = DLDataType{kDLUInt, 8, 1};
        } else {
          throw py::value_error(
              "make_device_tensor: unsupported dtype '" + name + "'");
        }
        return make_device_tensor(shape, dtype);
      },
      py::arg("shape"), py::arg("dtype"),
      "Allocate a CUDA holoscan.Tensor for ``shape`` (zero dims allowed).");

  // Emitter/receivers so Python can emit typed [live] overrides into C++ ops
  // (must pass the registered name as emit's 3rd arg) and consume custom
  // descriptor output ports (e.g. VirtualArrayAssemblyOp.array_geometry).
  m.def("register_types", [](holoscan::EmitterReceiverRegistry& registry) {
    registry.add_emitter_receiver<ArrayGeometry>("ArrayGeometry", true);
    registry.add_emitter_receiver<AngleConvention>("AngleConvention", true);
    registry.add_emitter_receiver<WindowSpec>("WindowSpec", true);
    registry.add_emitter_receiver<FMCWWaveformDescriptor>(
        "FMCWWaveformDescriptor", true);
    registry.add_emitter_receiver<DDMCodeDescriptor>("DDMCodeDescriptor", true);
    registry.add_emitter_receiver<TXSchedule>("TXSchedule", true);
    registry.add_emitter_receiver<ClutterFilterMethod>("ClutterFilterMethod",
                                                      true);
    registry.add_emitter_receiver<WindowType>("WindowType", true);
    registry.add_emitter_receiver<FftNorm>("FftNorm", true);
    registry.add_emitter_receiver<PeakInterpolation>("PeakInterpolation", true);
    registry.add_emitter_receiver<MultiplexingScheme>("MultiplexingScheme",
                                                     true);
    registry.add_emitter_receiver<OverlapPolicy>("OverlapPolicy", true);
    registry.add_emitter_receiver<SnapshotEdgeMode>("SnapshotEdgeMode", true);
    registry.add_emitter_receiver<RepresentativePolicy>(
        "RepresentativePolicy", true);
    registry.add_emitter_receiver<ScoreMethod>("ScoreMethod", true);
    registry.add_emitter_receiver<CoordinateFrame>("CoordinateFrame", true);
    registry.add_emitter_receiver<SampleGeometry>("SampleGeometry", true);
    registry.add_emitter_receiver<LFMWaveformDescriptor>(
        "LFMWaveformDescriptor", true);
    registry.add_emitter_receiver<RangeInterpolation>("RangeInterpolation",
                                                      true);
    registry.add_emitter_receiver<BackprojectionBackend>(
        "BackprojectionBackend", true);
    registry.add_emitter_receiver<GeometryChange>("GeometryChange", true);
    registry.add_emitter_receiver<FormationControl>("FormationControl", true);
    registry.add_emitter_receiver<ApertureEmitMode>("ApertureEmitMode", true);
    registry.add_emitter_receiver<PhaseCorrectionSense>(
        "PhaseCorrectionSense", true);
  });
}
