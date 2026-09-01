// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <holoscan/holoscan.hpp>

#include "radar_common/descriptors/array_geometry.hpp"
#include "radar_common/descriptors/angle_convention.hpp"
#include "radar_common/descriptors/coordinate_frame.hpp"
#include "radar_common/descriptors/enums.hpp"
#include "radar_common/descriptors/fmcw_waveform_descriptor.hpp"
#include "radar_common/descriptors/lfm_waveform_descriptor.hpp"
#include "radar_common/descriptors/sample_geometry.hpp"
#include "radar_common/descriptors/tx_schedule.hpp"
#include "radar_common/descriptors/window_spec.hpp"
#include "radar_common/sdk/yaml_converters.hpp"

namespace holoscan::radar {

// Holoscan's ArgumentSetter only knows built-in parameter types; every
// custom Parameter<T> type needs a registered setter or operator
// initialization fails with "No argument setter for type ...". The SDK's
// default setter (ArgumentSetter::ensure_type<T>) handles both native C++
// Args (direct any_cast) and YAML-sourced Args from from_config(), the
// latter through the YAML::convert<T> specializations in
// yaml_converters.hpp — included here so every registering translation
// unit sees them.
//
// Registration is global and idempotent (the setter map keeps the first
// registration for a type; all radar registrations are identical). Each
// operator calls ensure_radar_arg_setters() at the top of its initialize()
// override, before Operator::initialize() processes the arguments
// (default-value assignment also runs through the setters).
inline void ensure_radar_arg_setters() {
  static const bool registered = [] {
    ArgumentSetter::ensure_type<WindowSpec>();
    ArgumentSetter::ensure_type<FMCWWaveformDescriptor>();
    ArgumentSetter::ensure_type<ArrayGeometry>();
    ArgumentSetter::ensure_type<TXSchedule>();
    ArgumentSetter::ensure_type<AngleConvention>();
    ArgumentSetter::ensure_type<FftNorm>();
    ArgumentSetter::ensure_type<ClutterFilterMethod>();
    ArgumentSetter::ensure_type<SnapshotEdgeMode>();
    ArgumentSetter::ensure_type<PeakInterpolation>();
    ArgumentSetter::ensure_type<OverlapPolicy>();
    ArgumentSetter::ensure_type<MultiplexingScheme>();
    ArgumentSetter::ensure_type<DDMCodeDescriptor>();
    ArgumentSetter::ensure_type<RepresentativePolicy>();
    ArgumentSetter::ensure_type<ScoreMethod>();
    ArgumentSetter::ensure_type<CoordinateFrame>();
    ArgumentSetter::ensure_type<CoordinateFrameType>();
    ArgumentSetter::ensure_type<Ellipsoid>();
    ArgumentSetter::ensure_type<SampleGeometry>();
    ArgumentSetter::ensure_type<LFMWaveformDescriptor>();
    ArgumentSetter::ensure_type<FormationControl>();
    ArgumentSetter::ensure_type<RangeInterpolation>();
    ArgumentSetter::ensure_type<BackprojectionBackend>();
    ArgumentSetter::ensure_type<GeometryChange>();
    ArgumentSetter::ensure_type<ApertureEmitMode>();
    ArgumentSetter::ensure_type<PhaseCorrectionSense>();
    return true;
  }();
  (void)registered;
}

}  // namespace holoscan::radar
