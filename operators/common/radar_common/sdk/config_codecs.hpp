// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <complex>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <holoscan/config/codec.hpp>
#include <holoscan/core/parameter.hpp>

#include "radar_common/descriptors/angle_convention.hpp"
#include "radar_common/descriptors/array_geometry.hpp"
#include "radar_common/descriptors/coordinate_frame.hpp"
#include "radar_common/descriptors/fmcw_waveform_descriptor.hpp"
#include "radar_common/descriptors/lfm_waveform_descriptor.hpp"
#include "radar_common/descriptors/sample_geometry.hpp"
#include "radar_common/descriptors/tx_schedule.hpp"
#include "radar_common/descriptors/shift_invariant_subarray_pair.hpp"
#include "radar_common/descriptors/ddm_code_descriptor.hpp"
#include "radar_common/descriptors/window_spec.hpp"
#include "radar_common/reduce/reduction_enums.hpp"

namespace holoscan::radar::config_detail {

inline constexpr WindowType kWindowTypes[] = {
    WindowType::kNone,           WindowType::kHann,   WindowType::kHamming,   WindowType::kBlackman,
    WindowType::kBlackmanHarris, WindowType::kKaiser, WindowType::kChebyshev, WindowType::kTaylor,
    WindowType::kFlatTop,        WindowType::kTukey,  WindowType::kCustom,
};
inline constexpr FftNorm kFftNorms[] = {FftNorm::kBackward, FftNorm::kOrtho,
                                        FftNorm::kForward};  // NOLINT
inline constexpr ClutterFilterMethod kClutterFilterMethods[] = {
    ClutterFilterMethod::kNone,
    ClutterFilterMethod::kMeanSubtraction,
    ClutterFilterMethod::kSingleDelay,
    ClutterFilterMethod::kDoubleDelay,
};
inline constexpr SnapshotEdgeMode kSnapshotEdgeModes[] = {
    SnapshotEdgeMode::kWrap,
    SnapshotEdgeMode::kClamp,
    SnapshotEdgeMode::kShift,
};
inline constexpr RepresentativePolicy kRepresentativePolicies[] = {
    RepresentativePolicy::kPeak,
    RepresentativePolicy::kMedoid,
    RepresentativePolicy::kFirst,
    RepresentativePolicy::kWeightedMean,
};
inline constexpr ScoreMethod kScoreMethods[] = {
    ScoreMethod::kWeightedSum,
    ScoreMethod::kWeightedL2,
    ScoreMethod::kWeightedMax,
    ScoreMethod::kCustom,
};
inline constexpr ArrayGeometryType kArrayGeometryTypes[] = {
    ArrayGeometryType::kUla,
    ArrayGeometryType::kUra,
    ArrayGeometryType::kUca,
    ArrayGeometryType::kConformal,
};
inline constexpr AngleConventionPreset kAngleConventionPresets[] = {
    AngleConventionPreset::kRadarBoresight,   AngleConventionPreset::kRosRep103,
    AngleConventionPreset::kIso8855Flu,       AngleConventionPreset::kAerospaceNed,
    AngleConventionPreset::kPhysicsSpherical, AngleConventionPreset::kGeodeticEnu,
    AngleConventionPreset::kGeodeticNed,      AngleConventionPreset::kDirectionCosine,
    AngleConventionPreset::kCustom,
};
inline constexpr SignedAxis kSignedAxes[] = {
    SignedAxis::kPlusX,  SignedAxis::kMinusX, SignedAxis::kPlusY,
    SignedAxis::kMinusY, SignedAxis::kPlusZ,  SignedAxis::kMinusZ,
};
inline constexpr ElevationReference kElevationReferences[] = {
    ElevationReference::kHorizon,
    ElevationReference::kZenith,
};
inline constexpr PhaseCorrectionSense kPhaseCorrectionSenses[] = {
    PhaseCorrectionSense::kUnspecified,
    PhaseCorrectionSense::kSubtract,
    PhaseCorrectionSense::kAdd,
};
inline constexpr CoordinateFrameType kCoordinateFrameTypes[] = {
    CoordinateFrameType::kUnspecified, CoordinateFrameType::kEcef,       CoordinateFrameType::kEnu,
    CoordinateFrameType::kNed,         CoordinateFrameType::kSceneLocal,
};
inline constexpr Ellipsoid kEllipsoids[] = {Ellipsoid::kWgs84};
inline constexpr ApertureEmitMode kApertureEmitModes[] = {
    ApertureEmitMode::kContinue,
    ApertureEmitMode::kReset,
    ApertureEmitMode::kReform,
};
inline constexpr GeometryChange kGeometryChanges[] = {
    GeometryChange::kReset,
    GeometryChange::kScroll,
};
inline constexpr RangeInterpolation kRangeInterpolations[] = {
    RangeInterpolation::kNearest,
    RangeInterpolation::kLinear,
    RangeInterpolation::kSinc,
};
inline constexpr BackprojectionBackend kBackprojectionBackends[] = {
    BackprojectionBackend::kMatxDouble,
    BackprojectionBackend::kMatxDoublePhaseLut,
    BackprojectionBackend::kMatxMixed,
    BackprojectionBackend::kMatxMixedPhaseLut,
    BackprojectionBackend::kMatxFloatFloat,
    BackprojectionBackend::kMatxFloat,
    BackprojectionBackend::kMatxFloatPhaseLut,
    BackprojectionBackend::kMatxTaylorFast,
    BackprojectionBackend::kNvradarCudaFp32,
    BackprojectionBackend::kNvradarCudaFp32Hq,
    BackprojectionBackend::kNvradarCudaFp32HqFast,
    BackprojectionBackend::kNvradarFp64Reference,
    BackprojectionBackend::kNvradarBlockAnchored,
    BackprojectionBackend::kNvradarBlockAnchoredFp64Trajectory,
    BackprojectionBackend::kNvradarBlockAnchoredRefinedThroughput,
    BackprojectionBackend::kNvradarBlockAnchoredRefinedQuality,
    BackprojectionBackend::kNvradarBlockAnchoredExpandedThroughput,
    BackprojectionBackend::kNvradarBlockAnchoredExpandedQuality,
    BackprojectionBackend::kNvradarBlockAnchoredPhaseLut,
    BackprojectionBackend::kNvradarBlockAnchoredRefinedQualityPhaseLut,
    BackprojectionBackend::kNvradarBlockAnchoredExpandedThroughputPhaseLut,
    BackprojectionBackend::kNvradarBlockAnchoredExpandedThroughputPhaseLutVec,
    BackprojectionBackend::kNvradarBlockAnchoredExpandedThroughputPhaseLutVecZ0,
    BackprojectionBackend::kNvradarBlockAnchoredExpandedQualityPhaseLut,
    BackprojectionBackend::kNvradarBlockAnchoredPhaseLutLegacy,
    BackprojectionBackend::kNvradarBlockAnchoredFp64TrajectoryPhaseLut,
};
inline constexpr PeakInterpolation kPeakInterpolations[] = {
    PeakInterpolation::kNone,
    PeakInterpolation::kParabolic,
};

template <typename Enum, const auto& Candidates>
struct CanonicalEnumCodec {
  static config::ConfigResult<Enum> decode(config::ConfigView view,
                                           const config::DecodeOptions& options,
                                           config::DecodeContext& context) {
    auto text = config::ConfigCodec<std::string>::decode(std::move(view), options, context);
    if (!text) {
      return tl::make_unexpected(text.error());
    }
    for (const Enum candidate : Candidates) {
      if (*text == to_string(candidate)) {
        return candidate;
      }
    }
    return tl::make_unexpected(
        config::ConfigError::make(config::ConfigErrorCode::kUnsupportedValue,
                                  "unknown canonical enum spelling: " + *text, context.path()));
  }

  static config::ConfigResult<config::ConfigValue> encode(const Enum& value,
                                                          const config::EncodeOptions& options,
                                                          config::EncodeContext& context) {
    return config::ConfigCodec<std::string>::encode(std::string(to_string(value)), options,
                                                    context);
  }
};

template <typename T>
struct PortableParamStorageTraits {
  static consteval holoscan::ParamStorageTraitsSummary summary() noexcept {
    constexpr bool intrinsically_bounded = std::is_arithmetic_v<T> || std::is_enum_v<T>;
    return holoscan::ParamStorageTraitsSummary{
        .owns_dynamic_storage = !intrinsically_bounded,
        .fixed_object_extent = intrinsically_bounded,
        .has_portable_codec = true,
        .supports_bounded_materialization = intrinsically_bounded,
        .inline_bytes = intrinsically_bounded ? sizeof(T) : 0,
    };
  }
};

using WindowCodec = CanonicalEnumCodec<WindowType, kWindowTypes>;
using FftNormCodec = CanonicalEnumCodec<FftNorm, kFftNorms>;
using ClutterCodec = CanonicalEnumCodec<ClutterFilterMethod, kClutterFilterMethods>;  // NOLINT
using SnapshotEdgeCodec = CanonicalEnumCodec<SnapshotEdgeMode, kSnapshotEdgeModes>;
using RepresentativeCodec =
    CanonicalEnumCodec<RepresentativePolicy, kRepresentativePolicies>;  // NOLINT
using ScoreCodec = CanonicalEnumCodec<ScoreMethod, kScoreMethods>;
using GeometryTypeCodec = CanonicalEnumCodec<ArrayGeometryType, kArrayGeometryTypes>;
using ConventionPresetCodec =  // NOLINT(whitespace/indent_namespace)
    CanonicalEnumCodec<AngleConventionPreset, kAngleConventionPresets>;  // NOLINT
using SignedAxisCodec = CanonicalEnumCodec<SignedAxis, kSignedAxes>;
using ElevationCodec = CanonicalEnumCodec<ElevationReference, kElevationReferences>;
using PeakCodec = CanonicalEnumCodec<PeakInterpolation, kPeakInterpolations>;
using PhaseSenseCodec = CanonicalEnumCodec<PhaseCorrectionSense, kPhaseCorrectionSenses>;
using CoordinateFrameTypeCodec = CanonicalEnumCodec<CoordinateFrameType, kCoordinateFrameTypes>;
using EllipsoidCodec = CanonicalEnumCodec<Ellipsoid, kEllipsoids>;
using ApertureEmitCodec = CanonicalEnumCodec<ApertureEmitMode, kApertureEmitModes>;
using GeometryChangeCodec = CanonicalEnumCodec<GeometryChange, kGeometryChanges>;
using RangeInterpolationCodec = CanonicalEnumCodec<RangeInterpolation, kRangeInterpolations>;
using BackprojectionBackendCodec =
    CanonicalEnumCodec<BackprojectionBackend,     // NOLINT(whitespace/indent_namespace)
                       kBackprojectionBackends>;  // NOLINT(whitespace/indent_namespace)

using WindowStorage = PortableParamStorageTraits<WindowType>;
using FftNormStorage = PortableParamStorageTraits<FftNorm>;
using ClutterStorage = PortableParamStorageTraits<ClutterFilterMethod>;
using SnapshotEdgeStorage = PortableParamStorageTraits<SnapshotEdgeMode>;
using RepresentativeStorage = PortableParamStorageTraits<RepresentativePolicy>;
using ScoreMethodStorage = PortableParamStorageTraits<ScoreMethod>;
using GeometryTypeStorage = PortableParamStorageTraits<ArrayGeometryType>;
using ConventionPresetStorage = PortableParamStorageTraits<AngleConventionPreset>;
using SignedAxisStorage = PortableParamStorageTraits<SignedAxis>;
using ElevationStorage = PortableParamStorageTraits<ElevationReference>;
using PeakStorage = PortableParamStorageTraits<PeakInterpolation>;
using PhaseSenseStorage = PortableParamStorageTraits<PhaseCorrectionSense>;
using WindowSpecStorage = PortableParamStorageTraits<WindowSpec>;
using WaveformStorage = PortableParamStorageTraits<FMCWWaveformDescriptor>;
using LFMWaveformStorage = PortableParamStorageTraits<LFMWaveformDescriptor>;
using CoordinateFrameStorage = PortableParamStorageTraits<CoordinateFrame>;
using SampleGeometryStorage = PortableParamStorageTraits<SampleGeometry>;

struct SampleGeometryFields {
  std::array<double, 3> origin{};
  std::array<double, 3> axis_u{};
  std::array<double, 3> axis_v{};
  std::array<double, 2> spacing{};
  std::array<int64_t, 2> counts{};
};

struct CoordinateFrameFields {
  CoordinateFrameType frame{CoordinateFrameType::kUnspecified};
  std::optional<std::array<double, 3>> origin_llh{};
  std::optional<double> epoch{};
  Ellipsoid ellipsoid{Ellipsoid::kWgs84};
};
using ArrayGeometryStorage = PortableParamStorageTraits<ArrayGeometry>;
using AngleConventionStorage = PortableParamStorageTraits<AngleConvention>;

struct WaveformFields {
  double carrier_frequency{0.0};
  std::optional<double> bandwidth{};
  std::optional<double> chirp_slope{};
  double chirp_duration{0.0};
  double sample_rate{0.0};
  double chirp_repetition_interval{0.0};
  std::vector<std::array<double, 2>> reference_samples{};
};

}  // namespace holoscan::radar::config_detail

namespace holoscan::config {

template <>
struct ConfigCodec<radar::WindowType> : radar::config_detail::WindowCodec {};

template <>
struct ConfigCodec<radar::FftNorm> : radar::config_detail::FftNormCodec {};

template <>
struct ConfigCodec<radar::ClutterFilterMethod> : radar::config_detail::ClutterCodec {
};  // NOLINT(whitespace/indent_namespace)

template <>
struct ConfigCodec<radar::SnapshotEdgeMode> : radar::config_detail::SnapshotEdgeCodec {
};  // NOLINT(whitespace/indent_namespace)

template <>
struct ConfigCodec<radar::RepresentativePolicy> : radar::config_detail::RepresentativeCodec {
};  // NOLINT(whitespace/indent_namespace)

template <>
struct ConfigCodec<radar::ScoreMethod> : radar::config_detail::ScoreCodec {};

template <>
struct ConfigCodec<radar::ArrayGeometryType> : radar::config_detail::GeometryTypeCodec {
};  // NOLINT(whitespace/indent_namespace)

template <>
struct ConfigCodec<radar::AngleConventionPreset> : radar::config_detail::ConventionPresetCodec {
};  // NOLINT(whitespace/indent_namespace)

template <>
struct ConfigCodec<radar::SignedAxis> : radar::config_detail::SignedAxisCodec {};

template <>
struct ConfigCodec<radar::ElevationReference> : radar::config_detail::ElevationCodec {
};  // NOLINT(whitespace/indent_namespace)

template <>
struct ConfigCodec<radar::PhaseCorrectionSense> : radar::config_detail::PhaseSenseCodec {};

template <>
struct ConfigCodec<radar::CoordinateFrameType> : radar::config_detail::CoordinateFrameTypeCodec {};

template <>
struct ConfigCodec<radar::Ellipsoid> : radar::config_detail::EllipsoidCodec {};

template <>
struct ConfigCodec<radar::config_detail::CoordinateFrameFields>
: ObjectCodec<radar::config_detail::CoordinateFrameFields> {
  using Fields = radar::config_detail::CoordinateFrameFields;
  static constexpr auto fields =
      std::make_tuple(object_field("frame", &Fields::frame),
                      defaulted_object_field("origin_llh", &Fields::origin_llh),
                      defaulted_object_field("epoch", &Fields::epoch),
                      defaulted_object_field("ellipsoid", &Fields::ellipsoid));
};

template <>
struct ConfigCodec<radar::CoordinateFrame> {
  static ConfigResult<radar::CoordinateFrame> decode(ConfigView view, const DecodeOptions& options,
                                                     DecodeContext& context) {
    if (view.kind() == ConfigValueKind::kString) {
      auto frame =
          ConfigCodec<radar::CoordinateFrameType>::decode(std::move(view), options, context);
      if (!frame)
        return tl::make_unexpected(frame.error());
      try {
        return radar::CoordinateFrame(radar::CoordinateFrameSpec{.frame = *frame});
      } catch (const std::invalid_argument& error) {
        return tl::make_unexpected(
            ConfigError::make(ConfigErrorCode::kUnsupportedValue, error.what(), context.path()));
      }
    }
    auto fields = ConfigCodec<radar::config_detail::CoordinateFrameFields>::decode(
        std::move(view), options, context);
    if (!fields)
      return tl::make_unexpected(fields.error());
    try {
      return radar::CoordinateFrame(radar::CoordinateFrameSpec{.frame = fields->frame,
                                                               .origin_llh = fields->origin_llh,
                                                               .epoch = fields->epoch,
                                                               .ellipsoid = fields->ellipsoid});
    } catch (const std::invalid_argument& error) {
      return tl::make_unexpected(
          ConfigError::make(ConfigErrorCode::kUnsupportedValue, error.what(), context.path()));
    }
  }
  static ConfigResult<ConfigValue> encode(const radar::CoordinateFrame& value,
                                          const EncodeOptions& options, EncodeContext& context) {
    radar::config_detail::CoordinateFrameFields fields;
    fields.frame = value.frame();
    fields.origin_llh = value.origin_llh();
    fields.epoch = value.epoch();
    fields.ellipsoid = value.ellipsoid();
    return ConfigCodec<radar::config_detail::CoordinateFrameFields>::encode(fields, options,
                                                                            context);
  }
};

template <>
struct ConfigCodec<radar::ApertureEmitMode> : radar::config_detail::ApertureEmitCodec {};
template <>
struct ConfigCodec<radar::GeometryChange> : radar::config_detail::GeometryChangeCodec {};
template <>
struct ConfigCodec<radar::RangeInterpolation> : radar::config_detail::RangeInterpolationCodec {};
template <>
struct ConfigCodec<radar::BackprojectionBackend>
: radar::config_detail::BackprojectionBackendCodec {};

template <>
struct ConfigCodec<radar::config_detail::SampleGeometryFields>
: ObjectCodec<radar::config_detail::SampleGeometryFields> {
  using Fields = radar::config_detail::SampleGeometryFields;
  static constexpr auto fields = std::make_tuple(
      object_field("origin", &Fields::origin), object_field("axis_u", &Fields::axis_u),
      object_field("axis_v", &Fields::axis_v), object_field("spacing", &Fields::spacing),
      object_field("counts", &Fields::counts));
};

template <>
struct ConfigCodec<radar::SampleGeometry> {
  static ConfigResult<radar::SampleGeometry> decode(ConfigView view, const DecodeOptions& options,
                                                    DecodeContext& context) {
    auto fields = ConfigCodec<radar::config_detail::SampleGeometryFields>::decode(std::move(view),
                                                                                  options, context);
    if (!fields)
      return tl::make_unexpected(fields.error());
    try {
      return radar::SampleGeometry(radar::SampleGeometrySpec{.origin = fields->origin,
                                                             .axis_u = fields->axis_u,
                                                             .axis_v = fields->axis_v,
                                                             .spacing = fields->spacing,
                                                             .counts = fields->counts});
    } catch (const std::invalid_argument& error) {
      return tl::make_unexpected(
          ConfigError::make(ConfigErrorCode::kUnsupportedValue, error.what(), context.path()));
    }
  }
  static ConfigResult<ConfigValue> encode(const radar::SampleGeometry& value,
                                          const EncodeOptions& options, EncodeContext& context) {
    radar::config_detail::SampleGeometryFields fields;
    fields.origin = value.origin();
    fields.axis_u = value.axis_u();
    fields.axis_v = value.axis_v();
    fields.spacing = value.spacing();
    fields.counts = value.counts();
    return ConfigCodec<radar::config_detail::SampleGeometryFields>::encode(fields, options,
                                                                           context);
  }
};

template <>
struct ConfigCodec<radar::PeakInterpolation> : radar::config_detail::PeakCodec {
};  // NOLINT(whitespace/indent_namespace)

template <>
struct ConfigCodec<radar::config_detail::WaveformFields>
: ObjectCodec<radar::config_detail::WaveformFields> {
  using Fields = radar::config_detail::WaveformFields;
  static constexpr auto fields =
      std::make_tuple(object_field("carrier_frequency", &Fields::carrier_frequency),
                      defaulted_object_field("bandwidth", &Fields::bandwidth),
                      defaulted_object_field("chirp_slope", &Fields::chirp_slope),
                      object_field("chirp_duration", &Fields::chirp_duration),
                      object_field("sample_rate", &Fields::sample_rate),
                      object_field("chirp_repetition_interval", &Fields::chirp_repetition_interval),
                      defaulted_object_field("reference_samples", &Fields::reference_samples));
};

template <>
struct ConfigCodec<radar::WindowSpec> {
  static constexpr auto fields =
      std::make_tuple(defaulted_object_field("type", &radar::WindowSpec::type),
                      defaulted_object_field("sidelobe_attenuation_db",
                                             &radar::WindowSpec::sidelobe_attenuation_db),
                      defaulted_object_field("kaiser_beta", &radar::WindowSpec::kaiser_beta),
                      defaulted_object_field("taylor_nbar", &radar::WindowSpec::taylor_nbar),
                      defaulted_object_field("tukey_alpha", &radar::WindowSpec::tukey_alpha),
                      defaulted_object_field("coefficients", &radar::WindowSpec::coefficients),
                      defaulted_object_field("coherent_gain_normalization",
                                             &radar::WindowSpec::coherent_gain_normalization));

  static ConfigResult<radar::WindowSpec> decode(ConfigView view, const DecodeOptions& options,
                                                DecodeContext& context) {
    if (view.kind() == ConfigValueKind::kString) {
      auto type = ConfigCodec<radar::WindowType>::decode(std::move(view), options, context);
      if (!type) {
        return tl::make_unexpected(type.error());
      }
      radar::WindowSpec result;
      result.type = *type;
      return result;
    }
    return ObjectCodec<radar::WindowSpec>::decode(std::move(view), options, context);
  }

  static ConfigResult<ConfigValue> encode(const radar::WindowSpec& value,
                                          const EncodeOptions& options, EncodeContext& context) {
    return ObjectCodec<radar::WindowSpec>::encode(value, options, context);
  }
};

template <>
struct ConfigCodec<radar::FMCWWaveformDescriptor> {
  static ConfigResult<radar::FMCWWaveformDescriptor> decode(ConfigView view,
                                                            const DecodeOptions& options,
                                                            DecodeContext& context) {
    auto fields = ConfigCodec<radar::config_detail::WaveformFields>::decode(std::move(view),
                                                                            options, context);
    if (!fields) {
      return tl::make_unexpected(fields.error());
    }
    try {
      radar::FMCWWaveformSpec spec;
      spec.carrier_frequency = fields->carrier_frequency;
      spec.bandwidth = fields->bandwidth;
      spec.chirp_slope = fields->chirp_slope;
      spec.chirp_duration = fields->chirp_duration;
      spec.sample_rate = fields->sample_rate;
      spec.chirp_repetition_interval = fields->chirp_repetition_interval;
      spec.reference_samples.reserve(fields->reference_samples.size());
      for (const auto& sample : fields->reference_samples) {
        spec.reference_samples.emplace_back(sample[0], sample[1]);
      }
      return radar::FMCWWaveformDescriptor(std::move(spec));
    } catch (const std::invalid_argument& error) {
      return tl::make_unexpected(
          ConfigError::make(ConfigErrorCode::kUnsupportedValue, error.what(), context.path()));
    }
  }

  static ConfigResult<ConfigValue> encode(const radar::FMCWWaveformDescriptor& value,
                                          const EncodeOptions& options, EncodeContext& context) {
    radar::config_detail::WaveformFields fields;
    fields.carrier_frequency = value.carrier_frequency();
    fields.bandwidth = value.bandwidth();
    fields.chirp_duration = value.chirp_duration();
    fields.sample_rate = value.sample_rate();
    fields.chirp_repetition_interval = value.chirp_repetition_interval();
    fields.reference_samples.reserve(value.reference_samples().size());
    for (const auto& sample : value.reference_samples()) {
      fields.reference_samples.push_back({sample.real(), sample.imag()});
    }
    return ConfigCodec<radar::config_detail::WaveformFields>::encode(fields, options, context);
  }
};

template <>
struct ConfigCodec<radar::LFMWaveformDescriptor> {
  static ConfigResult<radar::LFMWaveformDescriptor> decode(ConfigView view,
                                                           const DecodeOptions& options,
                                                           DecodeContext& context) {
    auto fields = ConfigCodec<radar::config_detail::WaveformFields>::decode(std::move(view),
                                                                            options, context);
    if (!fields) {
      return tl::make_unexpected(fields.error());
    }
    try {
      radar::LFMWaveformSpec spec;
      spec.carrier_frequency = fields->carrier_frequency;
      spec.bandwidth = fields->bandwidth;
      spec.chirp_slope = fields->chirp_slope;
      spec.chirp_duration = fields->chirp_duration;
      spec.sample_rate = fields->sample_rate;
      spec.chirp_repetition_interval = fields->chirp_repetition_interval;
      spec.reference_samples.reserve(fields->reference_samples.size());
      for (const auto& sample : fields->reference_samples) {
        spec.reference_samples.emplace_back(sample[0], sample[1]);
      }
      return radar::LFMWaveformDescriptor(std::move(spec));
    } catch (const std::invalid_argument& error) {
      return tl::make_unexpected(
          ConfigError::make(ConfigErrorCode::kUnsupportedValue, error.what(), context.path()));
    }
  }

  static ConfigResult<ConfigValue> encode(const radar::LFMWaveformDescriptor& value,
                                          const EncodeOptions& options, EncodeContext& context) {
    radar::config_detail::WaveformFields fields;
    fields.carrier_frequency = value.carrier_frequency();
    fields.bandwidth = value.bandwidth();
    fields.chirp_duration = value.chirp_duration();
    fields.sample_rate = value.sample_rate();
    fields.chirp_repetition_interval = value.chirp_repetition_interval();
    fields.reference_samples.reserve(value.reference_samples().size());
    for (const auto& sample : value.reference_samples()) {
      fields.reference_samples.push_back({sample.real(), sample.imag()});
    }
    return ConfigCodec<radar::config_detail::WaveformFields>::encode(fields, options, context);
  }
};

template <>
struct ConfigCodec<radar::ArrayGeometry> {
  static constexpr auto fields = std::make_tuple(
      defaulted_object_field("element_positions", &radar::ArrayGeometry::element_positions),
      defaulted_object_field("wavelength", &radar::ArrayGeometry::wavelength),
      defaulted_object_field("geometry_type", &radar::ArrayGeometry::geometry_type));

  static ConfigResult<radar::ArrayGeometry> decode(ConfigView view, const DecodeOptions& options,
                                                   DecodeContext& context) {
    auto result = ObjectCodec<radar::ArrayGeometry>::decode(std::move(view), options, context);
    if (!result) {
      return tl::make_unexpected(result.error());
    }
    try {
      result->validate();
      return result;
    } catch (const std::invalid_argument& error) {
      return tl::make_unexpected(
          ConfigError::make(ConfigErrorCode::kUnsupportedValue, error.what(), context.path()));
    }
  }

  static ConfigResult<ConfigValue> encode(const radar::ArrayGeometry& value,
                                          const EncodeOptions& options, EncodeContext& context) {
    return ObjectCodec<radar::ArrayGeometry>::encode(value, options, context);
  }
};

template <>
struct ConfigCodec<radar::TXSchedule> : ObjectCodec<radar::TXSchedule> {
  static constexpr auto fields =
      std::make_tuple(defaulted_object_field("num_tx", &radar::TXSchedule::num_tx),
                      defaulted_object_field("tx_order", &radar::TXSchedule::tx_order),
                      defaulted_object_field("chirp_tx_index", &radar::TXSchedule::chirp_tx_index));
};

struct DdmCodeFields {
  int64_t num_tx{0};
  std::vector<double> phase_codes{};
  std::vector<double> doppler_offsets{};
  int64_t guard_bands{0};
};

template <>
struct ConfigCodec<DdmCodeFields> : ObjectCodec<DdmCodeFields> {
  static constexpr auto fields =
      std::make_tuple(defaulted_object_field("num_tx", &DdmCodeFields::num_tx),
                      defaulted_object_field("phase_codes", &DdmCodeFields::phase_codes),
                      defaulted_object_field("doppler_offsets", &DdmCodeFields::doppler_offsets),
                      defaulted_object_field("guard_bands", &DdmCodeFields::guard_bands));
};

template <>
struct ConfigCodec<radar::DDMCodeDescriptor> {
  static ConfigResult<radar::DDMCodeDescriptor> decode(ConfigView view,
                                                       const DecodeOptions& options,
                                                       DecodeContext& context) {
    auto fields = ConfigCodec<DdmCodeFields>::decode(std::move(view), options, context);
    if (!fields) {
      return tl::make_unexpected(fields.error());
    }
    if (fields->phase_codes.size() % 2U != 0U) {
      return tl::make_unexpected(ConfigError::make(
          ConfigErrorCode::kUnsupportedValue,
          "phase_codes must contain interleaved real/imaginary pairs", context.path()));
    }
    radar::DDMCodeDescriptor result;
    result.num_tx = fields->num_tx;
    result.doppler_offsets = std::move(fields->doppler_offsets);
    result.guard_bands = fields->guard_bands;
    result.phase_codes.reserve(fields->phase_codes.size() / 2U);
    for (std::size_t i = 0; i < fields->phase_codes.size(); i += 2U) {
      result.phase_codes.emplace_back(fields->phase_codes[i], fields->phase_codes[i + 1U]);
    }
    try {
      result.validate();
    } catch (const std::invalid_argument& error) {
      return tl::make_unexpected(
          ConfigError::make(ConfigErrorCode::kUnsupportedValue, error.what(), context.path()));
    }
    return result;
  }

  static ConfigResult<ConfigValue> encode(const radar::DDMCodeDescriptor& value,
                                          const EncodeOptions& options, EncodeContext& context) {
    DdmCodeFields fields;
    fields.num_tx = value.num_tx;
    fields.doppler_offsets = value.doppler_offsets;
    fields.guard_bands = value.guard_bands;
    fields.phase_codes.reserve(value.phase_codes.size() * 2U);
    for (const std::complex<double>& sample : value.phase_codes) {
      fields.phase_codes.push_back(sample.real());
      fields.phase_codes.push_back(sample.imag());
    }
    return ConfigCodec<DdmCodeFields>::encode(fields, options, context);
  }
};

inline constexpr radar::OverlapPolicy kOverlapPolicies[] = {
    radar::OverlapPolicy::kDedup,
    radar::OverlapPolicy::kAverage,
    radar::OverlapPolicy::kKeepAll,
};

template <>
struct ConfigCodec<radar::OverlapPolicy>
: radar::config_detail::CanonicalEnumCodec<radar::OverlapPolicy, kOverlapPolicies> {};  // NOLINT

template <>
struct ConfigCodec<radar::ShiftInvariantSubarrayPair> {
  static constexpr auto fields =
      std::make_tuple(defaulted_object_field("reference_indices",
                                             &radar::ShiftInvariantSubarrayPair::reference_indices),
                      defaulted_object_field("shifted_indices",
                                             &radar::ShiftInvariantSubarrayPair::shifted_indices));
  static ConfigResult<radar::ShiftInvariantSubarrayPair> decode(ConfigView view,
                                                                const DecodeOptions& options,
                                                                DecodeContext& context) {
    auto result =
        ObjectCodec<radar::ShiftInvariantSubarrayPair>::decode(std::move(view), options, context);
    if (!result) {
      return tl::make_unexpected(result.error());
    }
    try {
      result->validate();
      return result;
    } catch (const std::invalid_argument& error) {
      return tl::make_unexpected(
          ConfigError::make(ConfigErrorCode::kUnsupportedValue, error.what(), context.path()));
    }
  }
  static ConfigResult<ConfigValue> encode(const radar::ShiftInvariantSubarrayPair& value,
                                          const EncodeOptions& options, EncodeContext& context) {
    return ObjectCodec<radar::ShiftInvariantSubarrayPair>::encode(value, options, context);
  }
};

template <>
struct ConfigCodec<radar::AngleConvention> : ObjectCodec<radar::AngleConvention> {
  static constexpr auto fields = std::make_tuple(
      defaulted_object_field("preset", &radar::AngleConvention::preset),
      defaulted_object_field("boresight_axis", &radar::AngleConvention::boresight_axis),
      defaulted_object_field("azimuth_positive_axis",
                             &radar::AngleConvention::azimuth_positive_axis),
      defaulted_object_field("elevation_positive_axis",
                             &radar::AngleConvention::elevation_positive_axis),
      defaulted_object_field("elevation_reference", &radar::AngleConvention::elevation_reference));
};

}  // namespace holoscan::config

namespace holoscan {

template <>
struct ParamStorageTraits<radar::WindowType> : radar::config_detail::WindowStorage {
};  // NOLINT(whitespace/indent_namespace)
template <>
struct ParamStorageTraits<radar::FftNorm> : radar::config_detail::FftNormStorage {};  // NOLINT
template <>
struct ParamStorageTraits<radar::ClutterFilterMethod> : radar::config_detail::ClutterStorage {
};  // NOLINT(whitespace/indent_namespace)
template <>
struct ParamStorageTraits<radar::SnapshotEdgeMode> : radar::config_detail::SnapshotEdgeStorage {
};  // NOLINT(whitespace/indent_namespace)
template <>
struct ParamStorageTraits<radar::RepresentativePolicy>
: radar::config_detail::RepresentativeStorage {};  // NOLINT(whitespace/indent_namespace)
template <>
struct ParamStorageTraits<radar::ScoreMethod> : radar::config_detail::ScoreMethodStorage {
};  // NOLINT
template <>
struct ParamStorageTraits<radar::ArrayGeometryType> : radar::config_detail::GeometryTypeStorage {
};  // NOLINT(whitespace/indent_namespace)
template <>
struct ParamStorageTraits<radar::AngleConventionPreset>
: radar::config_detail::ConventionPresetStorage {};  // NOLINT(whitespace/indent_namespace)
template <>
struct ParamStorageTraits<radar::SignedAxis> : radar::config_detail::SignedAxisStorage {
};  // NOLINT
template <>
struct ParamStorageTraits<radar::ElevationReference> : radar::config_detail::ElevationStorage {
};  // NOLINT(whitespace/indent_namespace)
template <>
struct ParamStorageTraits<radar::TXSchedule>
: radar::config_detail::PortableParamStorageTraits<radar::TXSchedule> {};  // NOLINT
template <>
struct ParamStorageTraits<radar::DDMCodeDescriptor>
: radar::config_detail::PortableParamStorageTraits<radar::DDMCodeDescriptor> {};  // NOLINT
template <>
struct ParamStorageTraits<radar::OverlapPolicy>
: radar::config_detail::PortableParamStorageTraits<radar::OverlapPolicy> {};  // NOLINT

template <>
struct ParamStorageTraits<radar::ShiftInvariantSubarrayPair>
: radar::config_detail::PortableParamStorageTraits<radar::ShiftInvariantSubarrayPair> {};  // NOLINT
template <>
struct ParamStorageTraits<radar::PhaseCorrectionSense> : radar::config_detail::PhaseSenseStorage {
};  // NOLINT
template <>
struct ParamStorageTraits<radar::PeakInterpolation> : radar::config_detail::PeakStorage {
};  // NOLINT(whitespace/indent_namespace)
template <>
struct ParamStorageTraits<radar::WindowSpec> : radar::config_detail::WindowSpecStorage {
};  // NOLINT
template <>
struct ParamStorageTraits<radar::FMCWWaveformDescriptor> : radar::config_detail::WaveformStorage {
};  // NOLINT
template <>
struct ParamStorageTraits<radar::LFMWaveformDescriptor> : radar::config_detail::LFMWaveformStorage {
};  // NOLINT
template <>
struct ParamStorageTraits<radar::CoordinateFrame> : radar::config_detail::CoordinateFrameStorage {
};  // NOLINT
template <>
struct ParamStorageTraits<radar::SampleGeometry> : radar::config_detail::SampleGeometryStorage {
};  // NOLINT
template <>
struct ParamStorageTraits<radar::ApertureEmitMode>
: radar::config_detail::PortableParamStorageTraits<radar::ApertureEmitMode> {};  // NOLINT
template <>
struct ParamStorageTraits<radar::GeometryChange>
: radar::config_detail::PortableParamStorageTraits<radar::GeometryChange> {};  // NOLINT
template <>
struct ParamStorageTraits<radar::RangeInterpolation>
: radar::config_detail::PortableParamStorageTraits<radar::RangeInterpolation> {};  // NOLINT
template <>
struct ParamStorageTraits<radar::BackprojectionBackend>
: radar::config_detail::PortableParamStorageTraits<radar::BackprojectionBackend> {};  // NOLINT
template <>
struct ParamStorageTraits<radar::ArrayGeometry> : radar::config_detail::ArrayGeometryStorage {
};  // NOLINT
template <>
struct ParamStorageTraits<radar::AngleConvention> : radar::config_detail::AngleConventionStorage {
};  // NOLINT

}  // namespace holoscan
