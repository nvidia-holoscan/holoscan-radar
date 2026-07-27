// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "detection/cfar_detector_op/cfar_detector_op.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "detection/cfar_detector_op/cfar_alpha.hpp"
#include "detection/cfar_detector_op/cfar_compact.cuh"
#include "detection/cfar_detector_op/cfar_stage.hpp"
#include "detection/cfar_detector_op/nci_plan.hpp"
#include "radar_common/sdk/arg_setters.hpp"
#include "radar_common/sdk/override_param.hpp"
#include "radar_common/sdk/tensor_io.hpp"
#include "radar_common/tensor/dlpack_buffer.hpp"
#include "radar_common/tensor/tensor_utils.hpp"

// YAML::convert for the operator-local CfarVariant enum, mirroring the shared
// converters in radar_common/sdk/yaml_converters.hpp (reusing
// detail::parse_enum) so ArgumentSetter::ensure_type<CfarVariant>() supports
// from_config(). Kept in this TU because CfarVariant is cfar-local; common
// must not depend on it.
namespace YAML {
template <>
struct convert<holoscan::radar::CfarVariant> {
  static Node encode(const holoscan::radar::CfarVariant& value) {
    return Node(std::string(holoscan::radar::to_string(value)));
  }
  static bool decode(const Node& node, holoscan::radar::CfarVariant& value) {
    if (!node.IsScalar()) {
      return false;
    }
    static constexpr holoscan::radar::CfarVariant kAll[] = {
        holoscan::radar::CfarVariant::kCA, holoscan::radar::CfarVariant::kGO,
        holoscan::radar::CfarVariant::kSO, holoscan::radar::CfarVariant::kOS};
    value = holoscan::radar::detail::parse_enum(node.as<std::string>(),
                                                "CfarVariant", kAll);
    return true;
  }
};

template <>
struct convert<holoscan::radar::CfarPeakNeighborhood> {
  static Node encode(const holoscan::radar::CfarPeakNeighborhood& value) {
    return Node(std::string(holoscan::radar::to_string(value)));
  }
  static bool decode(const Node& node,
                     holoscan::radar::CfarPeakNeighborhood& value) {
    if (!node.IsScalar()) {
      return false;
    }
    static constexpr holoscan::radar::CfarPeakNeighborhood kAll[] = {
        holoscan::radar::CfarPeakNeighborhood::kCross4,
        holoscan::radar::CfarPeakNeighborhood::kSquare8};
    value = holoscan::radar::detail::parse_enum(
        node.as<std::string>(), "CfarPeakNeighborhood", kAll);
    return true;
  }
};
}  // namespace YAML

namespace holoscan::radar {

namespace {

constexpr const char* kOpName = "CFARDetectorOp";
// detection_mask is emitted as uint8 (0/1), not kDLBool: HSDK 4.3's DLPack
// message path cannot map kDLBool to a NumPy typestr, so a bool port throws
// when a Python consumer wraps it. The buffer holds 0/1 bytes (written as bool
// internally); uint8 is byte-identical, NumPy-visible, and every mask consumer
// treats any nonzero byte as set.
constexpr DLDataType kMaskDtype{kDLUInt, 8, 1};
constexpr DLDataType kInt64Dtype{kDLInt, 64, 1};

// GO/SO always calibrate alpha from Pfa via Monte Carlo; OS uses MC when K>1
// (closed-form when K=1). Warn once so callers know to supply alpha_lut for
// low-latency live updates.
bool uses_slow_pfa_lut_calibration(CfarVariant variant, int64_t fold) {
  return variant == CfarVariant::kGO || variant == CfarVariant::kSO ||
         (variant == CfarVariant::kOS && fold > 1);
}

int64_t max_alpha_lut_index(CfarVariant variant, int num_doppler,
                            int num_range, int guard_d, int guard_r,
                            int train_d, int train_r) {
  if (variant == CfarVariant::kGO || variant == CfarVariant::kSO) {
    return detail::max_go_lut_index(num_doppler, num_range, guard_d, guard_r,
                                    train_d, train_r);
  }
  return detail::max_training_cell_count(num_doppler, num_range, guard_d,
                                         guard_r, train_d, train_r);
}

void validate_external_alpha_lut(CfarVariant variant, int64_t fold,
                                 const std::vector<double>& lut,
                                 int64_t max_lut_index) {
  if (max_lut_index <= 0) {
    return;
  }
  if (lut.size() <= static_cast<size_t>(max_lut_index)) {
    throw std::invalid_argument(
        "CFARDetectorOp: alpha_lut length must exceed the maximum LUT index "
        "for the configured variant, footprint, and map size");
  }
  // O(max_index) only — do not walk the map for reachability. Sparse offline
  // tables (GO/SO always; OS when K>1) may leave unreachable slots at zero;
  // the kernel treats alpha==0 as no detect. CA and OS K=1 are dense and
  // require every slot in 1..max to be positive.
  //
  // Intentional footgun: for sparse variants we do not verify that reachable
  // indices are positive. A caller-supplied table of all zeros (or zeros on
  // used indices) validates and yields no detections; supply a correctly
  // filled LUT (e.g. from build_*_alpha_lut).
  for (int64_t idx = 1; idx <= max_lut_index; ++idx) {
    const double alpha = lut[static_cast<size_t>(idx)];
    if (!std::isfinite(alpha) || alpha < 0.0) {
      throw std::invalid_argument(
          "CFARDetectorOp: alpha_lut entries for used indices must be finite "
          "and non-negative");
    }
  }
  const bool sparse_ok = variant == CfarVariant::kGO ||
                         variant == CfarVariant::kSO ||
                         (variant == CfarVariant::kOS && fold > 1);
  if (sparse_ok) {
    return;
  }
  for (int64_t idx = 1; idx <= max_lut_index; ++idx) {
    if (lut[static_cast<size_t>(idx)] <= 0.0) {
      throw std::invalid_argument(
          "CFARDetectorOp: alpha_lut entries for used indices must be finite "
          "and positive");
    }
  }
}

void require_pair(const std::vector<int64_t>& v, const char* name) {
  if (v.size() != 2) {
    throw std::invalid_argument(
        std::string("CFARDetectorOp: '") + name +
        "' must have exactly 2 entries (doppler, range), got " +
        std::to_string(v.size()));
  }
  if (v[0] < 0 || v[1] < 0) {
    throw std::invalid_argument(std::string("CFARDetectorOp: '") + name +
                                "' entries must be >= 0");
  }
}

}  // namespace

template <typename T>
CFARDetectorOp<T>::~CFARDetectorOp() = default;

template <typename T>
void CFARDetectorOp<T>::initialize() {
  ensure_radar_arg_setters();
  ArgumentSetter::ensure_type<CfarVariant>();
  ArgumentSetter::ensure_type<CfarPeakNeighborhood>();
  Operator::initialize();
}

template <typename T>
void CFARDetectorOp<T>::start() {
  // Pre-calibrate the alpha LUT at startup so the first tick doesn't stall on a
  // Monte Carlo build. Only the pfa-derived path needs calibration (an external
  // alpha_lut is used as-is). The map dimensions are unknown here, so this is
  // limited to CA and OS, whose LUTs are indexed by training-cell count (N) and
  // are map-size-independent while the map is at least as large as the CFAR
  // window: a synthetic window-sized map reproduces the identical table. GO and
  // SO are indexed by the aliased go_lut_index and are genuinely map-dependent,
  // so they cannot be pre-warmed without the RD-map shape; they calibrate on
  // the first tick (compute() warns).
  if (!alpha_lut_.get().empty()) {
    return;  // external table: no calibration to pre-warm.
  }
  const CfarVariant variant = cfar_variant_.get();
  if (variant != CfarVariant::kCA && variant != CfarVariant::kOS) {
    HOLOSCAN_LOG_INFO(
        "CFARDetectorOp: {} calibrates its alpha LUT on the first tick "
        "(go_lut_index tables are map-dependent and cannot be pre-warmed "
        "without the RD-map shape); supply alpha_lut to avoid a runtime build.",
        to_string(variant));
    return;
  }
  const std::vector<int64_t> guard = guard_cells_.get();
  const std::vector<int64_t> train = training_cells_.get();
  if (guard.size() != 2 || train.size() != 2) {
    return;  // invalid/unset footprint: compute() will surface the error.
  }

  // Fold K: no nci_axes → K = 1; otherwise the fold is only known from the
  // input leading-axis sizes at tick time, so require the nci_fold hint.
  int64_t fold = 1;
  if (!nci_axes_.get().empty()) {
    const int64_t hint = nci_fold_hint_.get();
    if (hint <= 0) {
      HOLOSCAN_LOG_INFO(
          "CFARDetectorOp: nci_axes is set but nci_fold hint is unset; "
          "deferring alpha-LUT calibration to the first tick (set nci_fold to "
          "the integration fold K to pre-warm at start).");
      return;
    }
    fold = hint;
  }

  const int gd = static_cast<int>(guard[0]);
  const int gr = static_cast<int>(guard[1]);
  const int td = static_cast<int>(train[0]);
  const int tr = static_cast<int>(train[1]);
  const int win_d = 2 * (gd + td) + 1;
  const int win_r = 2 * (gr + tr) + 1;

  const AlphaLutKey warm_key{variant, pfa_.get(), fold,  os_rank_.get(), gd, gr,
                             td,      tr,         win_d, win_r};
  try {
    // Blocking build into the process-wide cache; the first tick then finds it
    // ready (poll) and only uploads — no runtime calibration, no warning.
    (void)detail::get_cached_alpha_lut(warm_key);
    HOLOSCAN_LOG_INFO(
        "CFARDetectorOp: pre-calibrated alpha LUT at start (variant={}, "
        "pfa={}, K={}).",
        to_string(variant), pfa_.get(), fold);
  } catch (const std::exception& e) {
    HOLOSCAN_LOG_WARN(
        "CFARDetectorOp: start-time alpha-LUT pre-calibration skipped ({}); "
        "the first tick will calibrate.",
        e.what());
  }
}

template <typename T>
void CFARDetectorOp<T>::stop() {
  // Wait for any background alpha-LUT build (non-blocking serve-stale path) to
  // finish before Holoscan tears down the CUDA context. A GO/SO/OS build runs
  // GPU Monte-Carlo kernels on a detached thread; without this drain those
  // launches can hit a destroyed context and SEGFAULT in the driver. stop()
  // runs after the last compute() and before teardown, so the context is still
  // valid here. Blocks only if a build is still running at shutdown.
  detail::join_inflight_alpha_lut_builds();
}

template <typename T>
void CFARDetectorOp<T>::setup(OperatorSpec& spec) {
  spec.input<holoscan::TensorMap>("rd_maps");
  spec.output<holoscan::TensorMap>("detection_mask");
  spec.output<holoscan::TensorMap>("detection_score");
  spec.output<holoscan::TensorMap>("noise_estimate");
  spec.output<holoscan::TensorMap>("detection_idx");
  // The exact surface the threshold is applied to: the post-NCI integrated
  // power when nci_axes is non-empty, else the plain square-law power map.
  // Emitted every tick (the workspace buffer, no copy) so consumers such as
  // the demo apps' RD display can show what the detector actually saw rather
  // than re-deriving it.
  //
  // kNone drops the default DownstreamReceptiveSchedulingTerm, which is what
  // makes this port genuinely optional: with the default term an unconnected
  // output stops the operator from ever ticking ("No receiver connected to
  // transmitter ... The entity will never tick"), so adding a mandatory output
  // would break every existing consumer that does not drain it.
  spec.output<holoscan::TensorMap>("cfar_power")
      .condition(holoscan::ConditionType::kNone);

  spec.param(cfar_variant_, "cfar_variant", "CFAR variant",
             "Noise-estimation rule {CA, GO, SO, OS}. Static: selects the "
             "kernel; instantiate separate detectors to vary it.",
             CfarVariant::kCA);
  spec.param(os_rank_, "os_rank", "Order-statistic rank",
             "Rank into the sorted training cells; used only for OS.",
             int64_t{0});
  spec.param(guard_cells_, "guard_cells", "Guard cells (doppler, range)",
             "Guard extent each side of the CUT, excluded from the noise "
             "estimate.",
             std::vector<int64_t>{});
  spec.param(training_cells_, "training_cells",
             "Training cells (doppler, range)",
             "Training/reference extent each side, for the noise estimate.",
             std::vector<int64_t>{});
  spec.param(pfa_, "pfa", "Probability of false alarm",
             "Sets the per-variant threshold multiplier alpha when alpha_lut "
             "is unset.",
             1e-6);
  spec.param(alpha_lut_, "alpha_lut", "Precomputed alpha LUT",
             "Optional threshold multiplier table; when non-empty, used "
             "directly and pfa is ignored for alpha calibration.",
             std::vector<double>{});
  spec.param(threshold_scale_, "threshold_scale", "Threshold scale",
             "Empirical multiplier on top of the Pfa-derived threshold; "
             "1.0 = pure CFAR.",
             1.0);
  spec.param(nci_axes_, "nci_axes", "Non-coherent integration axes",
             "Leading-axis indices to non-coherently integrate before "
             "thresholding.",
             std::vector<int64_t>{});
  spec.param(nci_fold_hint_, "nci_fold", "NCI fold hint (K)",
             "Optional integration fold K = product of nci_axes sizes. Static. "
             "Used only to pre-calibrate the alpha LUT at start() when "
             "nci_axes is set (K is unknown until the first tick). <= 0 → "
             "unset; ignored when nci_axes is empty.",
             int64_t{0});
  spec.param(max_detections_, "max_detections", "Max detections",
             "Cap on emitted detections (<= 0 → uncapped).", int64_t{0});
  spec.param(blocking_alpha_lut_rebuild_, "blocking_alpha_lut_rebuild",
             "Blocking alpha LUT rebuild",
             "When false, serve the previous LUT while a new one calibrates "
             "after a [live] config change; when true, block compute() until "
             "the new LUT is ready.",
             false);
  spec.param(peak_only_, "peak_only", "Peak grouping (one point per peak)",
             "When true, collapse each contiguous above-threshold blob to its "
             "local-maximum cell (non-maximum suppression) so detection_idx / "
             "detection_mask carry one point per CFAR peak. Static. Off "
             "(default) emits the raw candidate mask.",
             false);
  spec.param(peak_neighborhood_, "peak_neighborhood",
             "Peak-grouping neighborhood",
             "Neighborhood for the peak_only NMS test: {cross4, square8}. "
             "cross4 = 2 range + 2 Doppler neighbors; square8 = full 8-cell. "
             "Static; used only when peak_only is set.",
             CfarPeakNeighborhood::kSquare8);

  add_override_port<int64_t>(spec, "os_rank");
  add_override_port<std::vector<int64_t>>(spec, "guard_cells");
  add_override_port<std::vector<int64_t>>(spec, "training_cells");
  add_override_port<double>(spec, "pfa");
  add_override_port<std::vector<double>>(spec, "alpha_lut");
  add_override_port<double>(spec, "threshold_scale");
  add_override_port<std::vector<int64_t>>(spec, "nci_axes");
  add_override_port<int64_t>(spec, "max_detections");
}

template <typename T>
void CFARDetectorOp<T>::compute(InputContext& op_input,
                                OutputContext& op_output,
                                [[maybe_unused]] ExecutionContext& context) {
  std::shared_ptr<holoscan::Tensor> input =
      receive_single(op_input, "rd_maps", kOpName);

  if (!seeded_) {
    effective_guard_cells_ = guard_cells_.get();
    effective_training_cells_ = training_cells_.get();
    effective_pfa_ = pfa_.get();
    effective_alpha_lut_ = alpha_lut_.get();
    effective_threshold_scale_ = threshold_scale_.get();
    effective_os_rank_ = os_rank_.get();
    effective_nci_axes_ = nci_axes_.get();
    effective_max_detections_ = max_detections_.get();
    seeded_ = true;
  }
  effective_guard_cells_ = resolve_override_param<std::vector<int64_t>>(
      op_input, "guard_cells", effective_guard_cells_, nullptr);
  effective_training_cells_ = resolve_override_param<std::vector<int64_t>>(
      op_input, "training_cells", effective_training_cells_, nullptr);
  bool pfa_override_changed = false;
  effective_pfa_ = resolve_override_param<double>(op_input, "pfa",
                                                  effective_pfa_,
                                                  &pfa_override_changed);
  effective_alpha_lut_ = resolve_override_param<std::vector<double>>(
      op_input, "alpha_lut", effective_alpha_lut_, nullptr);
  effective_threshold_scale_ = resolve_override_param<double>(
      op_input, "threshold_scale", effective_threshold_scale_, nullptr);
  effective_os_rank_ = resolve_override_param<int64_t>(
      op_input, "os_rank", effective_os_rank_, nullptr);
  effective_nci_axes_ = resolve_override_param<std::vector<int64_t>>(
      op_input, "nci_axes", effective_nci_axes_, nullptr);
  effective_max_detections_ = resolve_override_param<int64_t>(
      op_input, "max_detections", effective_max_detections_, nullptr);

  require_pair(effective_guard_cells_, "guard_cells");
  require_pair(effective_training_cells_, "training_cells");

  const InputPrecision precision =
      classify_input_dtype<T>(*input, "rd_maps", /*allow_real=*/true);
  const bool input_is_complex = (precision == InputPrecision::kComplex);
  require_cuda_accessible(*input, "rd_maps", kOpName);
  require_contiguous(*input, "rd_maps", kOpName);

  const detail::NciPlan plan =
      detail::NciPlan::from_shape(input->shape(), effective_nci_axes_,
                                  "rd_maps");
  const int64_t num_doppler = input->shape()[input->shape().size() - 2];
  const int64_t num_range = input->shape()[input->shape().size() - 1];
  const std::vector<int64_t> output_shape =
      plan.output_shape(num_doppler, num_range);

  cudaStream_t stream = op_input.receive_cuda_stream("rd_maps");

  // Reuse a ping-ponged dense workspace when the CFAR geometry is unchanged.
  dense_workspace_idx_ ^= 1;
  DenseWorkspace& ws = dense_workspace_[dense_workspace_idx_];
  const bool ws_hit = ws.score && ws.batch_in == plan.batch_in &&
                      ws.out_batch == plan.out_batch &&
                      ws.num_doppler == num_doppler &&
                      ws.num_range == num_range;
  if (!ws_hit) {
    ws.score = make_device_tensor(output_shape, real_dtype<T>());
    ws.noise = make_device_tensor(output_shape, real_dtype<T>());
    ws.mask = make_device_tensor(output_shape, kMaskDtype);
    ws.peak_mask = make_device_tensor(output_shape, kMaskDtype);
    // Both buffers are written flat as (batch, doppler, range), so the
    // declared shape is free — give whichever one carries the emitted
    // cfar_power the full-rank output_shape so the port matches
    // detection_score. When fold == 1 the CFAR surface is `power` itself
    // (out_batch == batch_in is enforced by run_cfar_stage); when fold > 1 it
    // is `integrated_power`. Emitting the buffer directly keeps the port
    // zero-copy either way.
    ws.power = make_device_tensor(
        plan.fold > 1
            ? std::vector<int64_t>{plan.batch_in, num_doppler, num_range}
            : output_shape,
        real_dtype<T>());
    ws.integrated_power = make_device_tensor(output_shape, real_dtype<T>());
    ws.batch_in = plan.batch_in;
    ws.out_batch = plan.out_batch;
    ws.num_doppler = num_doppler;
    ws.num_range = num_range;
  }
  auto& score = ws.score;
  auto& noise = ws.noise;
  auto& mask = ws.mask;
  auto& peak_mask = ws.peak_mask;
  auto& power = ws.power;
  auto& integrated_power = ws.integrated_power;

  const bool peak_only = peak_only_.get();
  const bool peak_full8 =
      peak_neighborhood_.get() == CfarPeakNeighborhood::kSquare8;

  const int guard_d = static_cast<int>(effective_guard_cells_[0]);
  const int guard_r = static_cast<int>(effective_guard_cells_[1]);
  const int train_d = static_cast<int>(effective_training_cells_[0]);
  const int train_r = static_cast<int>(effective_training_cells_[1]);

  const CfarVariant variant = cfar_variant_.get();
  if (variant == CfarVariant::kOS) {
    if (effective_os_rank_ < 0) {
      throw std::invalid_argument(
          "CFARDetectorOp: os_rank must be non-negative for OS variant");
    }
    const int64_t max_train = detail::max_training_cell_count(
        static_cast<int>(num_doppler), static_cast<int>(num_range), guard_d,
        guard_r, train_d, train_r);
    if (max_train > detail::kMaxOsTrainingCells) {
      throw std::invalid_argument(
          "CFARDetectorOp: OS window footprint exceeds the supported "
          "training-cell limit");
    }
    if (effective_os_rank_ >= max_train) {
      throw std::invalid_argument(
          "CFARDetectorOp: os_rank must be less than the maximum training-"
          "cell count for the configured window footprint");
    }
  }

  // CA and OS LUTs are indexed by training-cell count (N), which is
  // map-size-independent while the map is at least as large as the CFAR window.
  // Key those on a canonical window-sized geometry so every such map shares one
  // cache entry (and the start-time pre-warm). GO and SO are indexed by the
  // aliased go_lut_index and are genuinely map-dependent (a window-sized map
  // yields different — and, at low pfa, non-convergent — calibration), so they
  // keep the actual map dimensions and are not pre-warmed. A sub-window CA/OS
  // map needs a map-specific table and forces a runtime rebuild — warned below.
  const bool map_independent_variant =
      variant == CfarVariant::kCA || variant == CfarVariant::kOS;
  const int win_d = 2 * (guard_d + train_d) + 1;
  const int win_r = 2 * (guard_r + train_r) + 1;
  const bool use_canonical_dims =
      map_independent_variant && num_doppler >= win_d && num_range >= win_r;
  if (map_independent_variant && !use_canonical_dims &&
      !window_exceeds_map_warned_) {
    HOLOSCAN_LOG_WARN(
        "CFARDetectorOp: CFAR window ({}x{}) exceeds the RD-map extent "
        "({}x{}); building a map-specific alpha LUT with reduced-N edge "
        "thresholds (the start-time pre-warmed table does not apply).",
        win_d, win_r, num_doppler, num_range);
    window_exceeds_map_warned_ = true;
  }
  const int key_num_doppler =
      use_canonical_dims ? win_d : static_cast<int>(num_doppler);
  const int key_num_range =
      use_canonical_dims ? win_r : static_cast<int>(num_range);

  AlphaLutKey lut_key{variant,
                      effective_pfa_,
                      plan.fold,
                      effective_os_rank_,
                      guard_d,
                      guard_r,
                      train_d,
                      train_r,
                      key_num_doppler,
                      key_num_range};

  auto upload_alpha_lut = [&](const std::vector<double>& host_lut) {
    std::vector<T> lut_t(host_lut.size());
    for (size_t i = 0; i < host_lut.size(); ++i) {
      lut_t[i] = static_cast<T>(host_lut[i]);
    }
    alpha_lut_dev_ = make_device_tensor(
        {static_cast<int64_t>(lut_t.size())}, real_dtype<T>());
    if (cudaMemcpyAsync(alpha_lut_dev_->data(), lut_t.data(),
                        lut_t.size() * sizeof(T), cudaMemcpyHostToDevice,
                        stream) != cudaSuccess) {
      throw std::runtime_error("CFARDetectorOp: alpha LUT H2D copy failed");
    }
  };

  const bool had_external_alpha_lut = using_external_alpha_lut_;
  const bool has_external_alpha_lut = !effective_alpha_lut_.empty();

  if (had_external_alpha_lut && !has_external_alpha_lut) {
    if (!alpha_lut_cleared_pfa_active_warned_) {
      HOLOSCAN_LOG_WARN(
          "CFARDetectorOp: alpha_lut cleared; pfa-derived alpha calibration is "
          "now active and may use the slow Monte Carlo path for GO, SO, or OS "
          "with K > 1");
      alpha_lut_cleared_pfa_active_warned_ = true;
    }
    using_external_alpha_lut_ = false;
    // Drop the external table from the host cache so the non-blocking stale
    // path cannot keep applying it while pfa-derived calibration runs.
    alpha_lut_host_cache_.clear();
    alpha_lut_dev_.reset();
    alpha_lut_key_ = {};
    pending_lut_key_ = {};
    pending_lut_active_ = false;
    stale_lut_warned_ = false;
    external_alpha_lut_validated_ = false;
  }

  if (pfa_override_changed && has_external_alpha_lut &&
      !pfa_ignored_while_alpha_lut_warned_) {
    HOLOSCAN_LOG_WARN(
        "CFARDetectorOp: pfa override has no effect while alpha_lut is "
        "supplied");
    pfa_ignored_while_alpha_lut_warned_ = true;
  }

  if (has_external_alpha_lut) {
    // max_alpha_lut_index is closed-form for maps that fit the window (the
    // common case); only needed to validate an externally supplied LUT.
    // Re-validate only when the table or CFAR geometry/fold changes — not
    // every tick.
    const bool lut_changed = effective_alpha_lut_ != alpha_lut_host_cache_;
    const bool geometry_changed =
        !external_alpha_lut_validated_ ||
        validated_external_variant_ != variant ||
        validated_external_fold_ != plan.fold ||
        validated_external_num_doppler_ != num_doppler ||
        validated_external_num_range_ != num_range ||
        validated_external_guard_d_ != guard_d ||
        validated_external_guard_r_ != guard_r ||
        validated_external_train_d_ != train_d ||
        validated_external_train_r_ != train_r;
    if (lut_changed || geometry_changed) {
      const int64_t max_lut_index = max_alpha_lut_index(
          variant, static_cast<int>(num_doppler), static_cast<int>(num_range),
          guard_d, guard_r, train_d, train_r);
      validate_external_alpha_lut(variant, plan.fold, effective_alpha_lut_,
                                  max_lut_index);
      external_alpha_lut_validated_ = true;
      validated_external_variant_ = variant;
      validated_external_fold_ = plan.fold;
      validated_external_num_doppler_ = num_doppler;
      validated_external_num_range_ = num_range;
      validated_external_guard_d_ = guard_d;
      validated_external_guard_r_ = guard_r;
      validated_external_train_d_ = train_d;
      validated_external_train_r_ = train_r;
      if (lut_changed) {
        alpha_lut_host_cache_ = effective_alpha_lut_;
        upload_alpha_lut(alpha_lut_host_cache_);
      }
    }
    using_external_alpha_lut_ = true;
  } else {
    using_external_alpha_lut_ = false;
    const bool blocking_lut = blocking_alpha_lut_rebuild_.get();
    if (lut_key != alpha_lut_key_) {
      // Reset the warn-once flags only on a genuine key change — NOT on every
      // tick while the same key's background build is still pending (the
      // adopted key stays stale until the build completes, so this branch is
      // re-entered each tick; resetting here would re-fire the runtime-rebuild
      // warning every tick). Preserve the flags for an in-progress build; clear
      // them when switching to a different pending key.
      if (!(pending_lut_active_ && pending_lut_key_ == lut_key)) {
        runtime_build_warned_ = false;
        stale_lut_warned_ = false;
      }
      const auto adopt_ready_lut = [&](std::vector<double> lut) {
        alpha_lut_host_cache_ = std::move(lut);
        alpha_lut_key_ = lut_key;
        pending_lut_key_ = {};
        pending_lut_active_ = false;
        stale_lut_warned_ = false;
        upload_alpha_lut(alpha_lut_host_cache_);
      };
      const auto warn_slow_pfa_lut = [&]() {
        if (pfa_lut_slow_warned_ ||
            !uses_slow_pfa_lut_calibration(lut_key.variant, lut_key.fold)) {
          return;
        }
        HOLOSCAN_LOG_WARN(
            "CFARDetectorOp: deriving the alpha LUT from pfa for {} is the "
            "slow calibration path (Monte Carlo LUT build); for low-latency "
            "live updates supply a precomputed alpha_lut",
            to_string(lut_key.variant));
        pfa_lut_slow_warned_ = true;
      };
      // Fires whenever a build is forced during a tick (a runtime rebuild /
      // stall) rather than served from the start-time warm cache — the signal
      // the operator did not pre-calibrate this configuration.
      const auto warn_runtime_rebuild = [&]() {
        if (runtime_build_warned_) {
          return;
        }
        HOLOSCAN_LOG_WARN(
            "CFARDetectorOp: calibrating the alpha LUT at runtime (the "
            "configuration was not pre-warmed at start — e.g. a [live] "
            "override, an NCI fold not covered by nci_fold, or a sub-window "
            "map); this stalls the tick or serves a stale LUT while building.");
        runtime_build_warned_ = true;
      };
      // Poll first: a key pre-warmed at start() (or built on an earlier tick)
      // is served with no calibration and no runtime-rebuild warning.
      detail::AlphaLutSnapshot snap = detail::poll_cached_alpha_lut(lut_key);
      if (snap.ready) {
        adopt_ready_lut(std::move(snap.lut));
      } else if (blocking_lut || alpha_lut_host_cache_.empty()) {
        // Blocking mode, or the first materialization (no previous LUT to
        // serve): build now and stall this tick.
        warn_runtime_rebuild();
        warn_slow_pfa_lut();
        adopt_ready_lut(detail::get_cached_alpha_lut(lut_key));
      } else {
        // Non-blocking: serve the previous LUT this tick, build in background.
        warn_runtime_rebuild();
        warn_slow_pfa_lut();
        detail::ensure_alpha_lut_build(lut_key);
        pending_lut_key_ = lut_key;
        pending_lut_active_ = true;
        if (!stale_lut_warned_) {
          HOLOSCAN_LOG_WARN(
              "CFARDetectorOp: alpha LUT for the new configuration is still "
              "calibrating; using the previous LUT for this tick");
          stale_lut_warned_ = true;
        }
      }
    } else if (pending_lut_active_ && pending_lut_key_ == lut_key) {
      detail::AlphaLutSnapshot snap = detail::poll_cached_alpha_lut(lut_key);
      if (snap.ready) {
        alpha_lut_host_cache_ = std::move(snap.lut);
        alpha_lut_key_ = lut_key;
        pending_lut_key_ = {};
        pending_lut_active_ = false;
        stale_lut_warned_ = false;
        upload_alpha_lut(alpha_lut_host_cache_);
      }
    }
  }

  // Upload the NCI lookup table only when the integration layout changes;
  // otherwise reuse the cached device buffer (no per-tick alloc/free/upload).
  const int64_t* nci_table_dev_ptr = nullptr;
  if (plan.fold > 1) {
    if (plan.batch_in_table != nci_table_host_cache_) {
      nci_table_dev_ = make_device_tensor(
          {static_cast<int64_t>(plan.batch_in_table.size())}, kInt64Dtype);
      if (cudaMemcpyAsync(nci_table_dev_->data(), plan.batch_in_table.data(),
                          plan.batch_in_table.size() * sizeof(int64_t),
                          cudaMemcpyHostToDevice, stream) != cudaSuccess) {
        throw std::runtime_error("CFARDetectorOp: NCI table H2D copy failed");
      }
      nci_table_host_cache_ = plan.batch_in_table;
    }
    nci_table_dev_ptr = static_cast<const int64_t*>(nci_table_dev_->data());
  }

  detail::run_cfar_stage<T>(
      variant, input->data(), input_is_complex, plan.batch_in, plan.out_batch,
      num_doppler, num_range, guard_d, guard_r, train_d, train_r,
      effective_os_rank_, effective_threshold_scale_, plan.fold,
      nci_table_dev_ptr, alpha_lut_dev_->data(),
      static_cast<int64_t>(alpha_lut_host_cache_.size()), power->data(),
      integrated_power->data(), score->data(), noise->data(), mask->data(),
      peak_only, peak_full8, peak_only ? peak_mask->data() : nullptr, stream);

  // With peak_only, downstream (count / compact / detection_mask) consumes the
  // NMS-refined peak_mask; otherwise the raw candidate mask.
  const auto& det_mask_tensor = peak_only ? peak_mask : mask;
  const bool* det_mask = static_cast<const bool*>(det_mask_tensor->data());

  const int64_t plane = num_doppler * num_range;
  const int64_t out_elems = plan.out_batch * plane;
  const int64_t idx_cols = 2 + plan.n_preserved;

  if (!detection_idx_compact_) {
    detection_idx_compact_ =
        std::make_shared<detail::DetectionIdxCompactState>();
  }
  const int64_t total_detections = detail::count_mask_detections(
      det_mask, out_elems, detection_idx_compact_.get(), stream);

  const bool capped = effective_max_detections_ > 0;
  const int64_t num_detections =
      capped ? std::min(total_detections, effective_max_detections_)
             : total_detections;
  if (capped && total_detections > effective_max_detections_) {
    HOLOSCAN_LOG_WARN(
        "CFARDetectorOp: detections exceeded max_detections ({}); surplus "
        "dropped",
        effective_max_detections_);
  }

  auto detection_idx =
      make_device_tensor({num_detections, idx_cols}, kInt64Dtype);
  detail::compact_detection_idx(
      det_mask, out_elems, num_doppler, num_range, plan.preserved_dims,
      num_detections, static_cast<int64_t*>(detection_idx->data()),
      detection_idx_compact_.get(), stream);

  if (stream != cudaStreamDefault) {
    op_output.set_cuda_stream(stream, "detection_mask");
    op_output.set_cuda_stream(stream, "detection_score");
    op_output.set_cuda_stream(stream, "noise_estimate");
    op_output.set_cuda_stream(stream, "detection_idx");
    op_output.set_cuda_stream(stream, "cfar_power");
  }
  // The thresholded surface: integrated_power when the NCI fold ran, else the
  // raw square-law power map (see the workspace allocation above).
  emit_single(op_output, "cfar_power",
              plan.fold > 1 ? integrated_power : power);
  emit_single(op_output, "detection_mask", det_mask_tensor);
  emit_single(op_output, "detection_score", score);
  emit_single(op_output, "noise_estimate", noise);
  emit_single(op_output, "detection_idx", detection_idx);

  retained_input_ = std::move(input);
}

template class CFARDetectorOp<float>;
template class CFARDetectorOp<double>;

}  // namespace holoscan::radar
