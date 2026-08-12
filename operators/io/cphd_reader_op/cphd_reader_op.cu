// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "io/cphd_reader_op/cphd_reader_op.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <filesystem>  // NOLINT(build/c++17): repo builds with C++17
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

#include "io/cphd_reader_op/cphd_file.hpp"
#include "radar_common/descriptors/coordinate_frame.hpp"
#include "radar_common/descriptors/enums.hpp"
#include "radar_common/sdk/arg_setters.hpp"
#include "radar_common/sdk/tensor_io.hpp"
#include "radar_common/tensor/dlpack_buffer.hpp"
#include "radar_common/tensor/tensor_utils.hpp"

namespace holoscan::radar {

namespace {

template <typename T>
constexpr const char* precision_name() {
  return std::is_same_v<T, float> ? "float" : "double";
}

// The metadata key the boundary announcement rides on (D22). The reader is
// the only stage that knows where a collection ends, so it stamps the key on
// the messages of the batch that *is* the last one -- announcing on the
// following batch would be a boundary reported after the closing aperture
// had already been contaminated.
//
// The SAR tier is the interim value R10 prescribes: the namespace scope is a
// team decision that may not be settled in code, so the key sits under `sar`
// until it is answered, committing FMCW to nothing. Promotion to the module
// level is then a change to this one constant. See SAR/SIGPROC_PLAN.md.
constexpr const char* kStreamBoundaryKey = "holoscan.radar.sar.stream_boundary";
constexpr const char* kEndOfStream = "end_of_stream";
constexpr const char* kCollectionBoundary = "collection_boundary";

// Expand the `path` parameter into the ordered list of collections to read.
// A directory contributes its `.cphd` entries in lexicographic order, which
// is what the contract specifies; anything else is taken as a file and passed
// through, so an explicit list keeps the order it was written in even when
// that is not lexicographic.
std::vector<std::string> resolve_paths(const CollectionPaths& parameter,
                                       const std::string& who) {
  std::vector<std::string> files;
  for (const std::string& entry : parameter.entries) {
    std::error_code error;
    if (!std::filesystem::exists(entry, error)) {
      throw std::runtime_error(who + ": path '" + entry + "' does not exist");
    }
    if (!std::filesystem::is_directory(entry, error)) {
      files.push_back(entry);
      continue;
    }
    std::vector<std::string> found;
    for (const auto& item : std::filesystem::directory_iterator(entry)) {
      if (item.is_regular_file() && item.path().extension() == ".cphd") {
        found.push_back(item.path().string());
      }
    }
    if (found.empty()) {
      throw std::runtime_error(who + ": directory '" + entry +
                               "' contains no .cphd files");
    }
    // directory_iterator order is unspecified, so the contract's
    // lexicographic promise has to be imposed here rather than assumed.
    std::sort(found.begin(), found.end());
    files.insert(files.end(), found.begin(), found.end());
  }
  if (files.empty()) {
    throw std::runtime_error(who + ": path selects no files");
  }
  return files;
}

// Copy a host buffer to a freshly allocated device tensor of `shape`.
template <typename Element>
std::shared_ptr<holoscan::Tensor> upload(const std::vector<Element>& host,
                                         const std::vector<int64_t>& shape,
                                         DLDataType dtype, const char* who) {
  auto tensor = make_device_tensor(shape, dtype);
  const size_t bytes = sizeof(Element) * host.size();
  // The shape sizes the allocation while the vector sizes the copy, so a
  // disagreement between the two writes past the device buffer rather than
  // merely emitting a wrongly shaped tensor.
  if (bytes != static_cast<size_t>(tensor->nbytes())) {
    throw std::runtime_error(
        std::string(who) + ": H2D upload of " + std::to_string(bytes) +
        " host bytes into a " + std::to_string(tensor->nbytes()) +
        "-byte device tensor");
  }
  // Synchronous by design, not an oversight. The host source is one of this
  // tick's locals, so the copy has to complete before this returns; that also
  // means the data is visible to every stream by the time the tensor is
  // emitted, so no stream ordering against the consumer is needed.
  if (cudaMemcpy(tensor->data(), host.data(), bytes,
                 cudaMemcpyHostToDevice) != cudaSuccess) {
    throw std::runtime_error(std::string(who) + ": H2D upload failed");
  }
  return tensor;
}

// Narrow the file's doubles to the instantiated precision. PVP records are
// always float64 on disk; single-precision instances pay the cast here so
// the rest of the chain sees one dtype.
template <typename T>
std::vector<T> narrow(const std::vector<double>& values) {
  std::vector<T> narrowed(values.size());
  for (size_t i = 0; i < values.size(); ++i) {
    narrowed[i] = static_cast<T>(values[i]);
  }
  return narrowed;
}

}  // namespace

template <typename T>
CPHDReaderOp<T>::~CPHDReaderOp() = default;

template <typename T>
void CPHDReaderOp<T>::initialize() {
  ensure_radar_arg_setters();
  // `path` is the reader's own parameter type rather than a shared
  // descriptor, so it registers its setter here rather than in the common
  // list. Registration is idempotent.
  ArgumentSetter::ensure_type<CollectionPaths>();

  // The scheduling term this operator disables once the selected range is
  // exhausted. Created here rather than by the application so terminating is
  // the reader's own business, matching the SDK's source operators.
  stop_condition_ =
      fragment()->make_condition<holoscan::BooleanCondition>("stop_condition");
  add_arg(stop_condition_);

  Operator::initialize();
}

template <typename T>
void CPHDReaderOp<T>::setup(OperatorSpec& spec) {
  spec.output<holoscan::TensorMap>("phase_history");
  spec.output<holoscan::TensorMap>("positions");
  spec.output<holoscan::TensorMap>("reference_position");
  // Optional outputs: kNone so an unconnected port never gates scheduling.
  spec.output<holoscan::TensorMap>("sample_axis")
      .condition(ConditionType::kNone);
  spec.output<holoscan::TensorMap>("pulse_times")
      .condition(ConditionType::kNone);
  spec.output<holoscan::TensorMap>("velocities")
      .condition(ConditionType::kNone);
  spec.output<CoordinateFrame>("coordinate_frame")
      .condition(ConditionType::kNone);

  spec.param(path_, "path", "CPHD collection path",
             "A single CPHD file, a directory of them, or an explicit "
             "ordered list. A directory is read in lexicographic order; a "
             "list is read in the order given.");

  spec.param(coordinate_frame_, "coordinate_frame", "Coordinate frame",
             "Frame the emitted positions are expressed in. A frame "
             "disagreeing with the one the file declares is an error, not a "
             "silent reinterpretation.");

  spec.param(pulses_per_tick_, "pulses_per_tick", "Pulses per tick",
             "Batch size along the pulse axis. The final batch is short when "
             "the selected range is not a multiple of it.");

  spec.param(channel_, "channel", "CPHD channel index",
             "Which CPHD channel to read. One channel per instance is "
             "deliberate; see OPERATOR_API.md.",
             static_cast<int64_t>(0));

  spec.param(pulse_range_, "pulse_range", "Pulse range",
             "Half-open [first, last) pulse range to read. Default: the "
             "whole collection.",
             ParameterFlag::kOptional);
}

// Opens files_[file_index_] and resolves the pulse range within it. Called
// once per collection: at start(), and again at each file seam.
template <typename T>
void CPHDReaderOp<T>::open_current_file() {
  const std::string who =
      std::string("CPHDReaderOp<") + precision_name<T>() + ">";
  const std::string& path = files_[file_index_];

  file_ = std::make_unique<cphd::CphdFile>(path);
  const cphd::ChannelInfo& info = file_->channel(channel_.get());

  // CPHD carries positions in the earth-centred fixed frame, so anything
  // else the caller asks for would be a reinterpretation of the file's
  // numbers rather than a conversion of them. Checked per file: a list may
  // mix collections, and only the file being read can be validated against.
  const CoordinateFrame& frame = coordinate_frame_.get();
  if (frame.frame() != CoordinateFrameType::kEcef) {
    throw std::runtime_error(
        who + ": file '" + path +
        "' declares its positions in ECF, but coordinate_frame is " +
        std::string(to_string(frame.frame())) +
        ". CPHD positions are ECEF; conversion is a downstream concern.");
  }

  // Every field a tick reads unconditionally, checked here rather than left
  // to the first read: start() opens each file up front precisely so a
  // collection that cannot be read fails before any pulse is emitted.
  // TxVel/RcvVel and AmpSF are deliberately absent -- the operator reads
  // those only from a file that declares them.
  constexpr const char* kRequiredPvpFields[] = {"TxTime", "TxPos", "RcvPos",
                                                "SRPPos", "SC0",   "SCSS"};
  std::string missing;
  for (const char* field : kRequiredPvpFields) {
    if (!file_->has_pvp_field(field)) {
      if (!missing.empty()) {
        missing += ", ";
      }
      missing += field;
    }
  }
  if (!missing.empty()) {
    throw std::runtime_error(who + ": file '" + path +
                             "' does not declare the PVP field(s) " + missing +
                             ", which every batch reads");
  }

  first_pulse_ = 0;
  last_pulse_ = info.num_vectors;
  const auto range = pulse_range_.try_get();
  if (range.has_value()) {
    // Per file, per the contract: each file is its own collection, so the
    // same window is taken from each rather than the range running across
    // the concatenation.
    first_pulse_ = (*range)[0];
    last_pulse_ = (*range)[1];
    if (first_pulse_ < 0 || last_pulse_ > info.num_vectors ||
        first_pulse_ >= last_pulse_) {
      throw std::runtime_error(
          who + ": pulse_range [" + std::to_string(first_pulse_) + ", " +
          std::to_string(last_pulse_) + ") is out of range for '" + path +
          "'; its channel has " + std::to_string(info.num_vectors) +
          " pulses");
    }
  }
  next_pulse_ = first_pulse_;
  frame_emitted_ = false;

  HOLOSCAN_LOG_INFO(
      "{}: '{}' ({} of {}) channel {} ({}): {} pulses x {} samples, {}, "
      "reading [{}, {}) in batches of {}",
      who, path, file_index_ + 1, files_.size(), channel_.get(),
      info.identifier, info.num_vectors, info.num_samples,
      to_string(file_->sample_format()), first_pulse_, last_pulse_,
      pulses_per_tick_.get());
}

template <typename T>
void CPHDReaderOp<T>::start() {
  const std::string who =
      std::string("CPHDReaderOp<") + precision_name<T>() + ">";

  const CoordinateFrame& frame = coordinate_frame_.get();
  if (!frame.is_valid()) {
    throw std::runtime_error(who + ": coordinate_frame is not valid");
  }
  if (pulses_per_tick_.get() <= 0) {
    throw std::runtime_error(who + ": pulses_per_tick must be positive, got " +
                             std::to_string(pulses_per_tick_.get()));
  }
  const auto range = pulse_range_.try_get();
  if (range.has_value() && range->size() != 2) {
    throw std::runtime_error(who + ": pulse_range needs 2 entries, got " +
                             std::to_string(range->size()));
  }

  // The boundary announcement (D22) rides the metadata plane, so with metadata
  // disabled there is nowhere to put it and a consumer never learns where a
  // collection ends -- it would simply stop receiving pulses. Not fatal, since
  // a consumer that forms one image per run needs no seam, but not silent
  // either. Said here rather than at the seam because the setting cannot
  // change in flight, so warning once means warning before the first pulse
  // rather than after the last.
  if (!is_metadata_enabled()) {
    HOLOSCAN_LOG_WARN(
        "{}: metadata is disabled, so the end-of-collection announcement "
        "('{}') cannot be emitted; a consumer will not be told where a "
        "collection ends. Enable metadata on the fragment to receive it.",
        who, kStreamBoundaryKey);
  }

  files_ = resolve_paths(path_.get(), who);
  file_index_ = 0;

  // Every file is opened and checked before the first pulse is emitted.
  // Opening costs only the header and XML, whereas discovering an
  // unreadable third file after two collections have been formed wastes the
  // run and leaves a partial product.
  if (files_.size() > 1) {
    for (size_t index = 0; index < files_.size(); ++index) {
      file_index_ = index;
      open_current_file();
    }
    file_index_ = 0;
  }
  open_current_file();
}

template <typename T>
void CPHDReaderOp<T>::compute([[maybe_unused]] InputContext& op_input,
                              OutputContext& op_output,
                              [[maybe_unused]] ExecutionContext& context) {
  const std::string who =
      std::string("CPHDReaderOp<") + precision_name<T>() + ">";
  const cphd::ChannelInfo& info = file_->channel(channel_.get());
  const int64_t num_samples = info.num_samples;

  // The final batch is short whenever the range is not a multiple of the
  // batch size; downstream infers the count from the tensor, never from
  // pulses_per_tick.
  const int64_t count =
      std::min(pulses_per_tick_.get(), last_pulse_ - next_pulse_);
  const int64_t first = next_pulse_;
  // Final for this collection, which is not the same as final overall once
  // path names more than one file.
  const bool is_final = (first + count >= last_pulse_);
  const bool is_last_file = (file_index_ + 1 == files_.size());

  // 1. Per-pulse geometry. Every field this tick needs comes out of one read
  // of the parameter block: the records are interleaved, so reading eight
  // fields separately would be eight passes over the same bytes. Read first
  // because the signal decode needs the amplitude scale factors from it.
  //
  // Optional because the block borrows its layout and path from the file that
  // produced it, and the tail of this function may replace that file: the
  // borrow has to be given up before the reopen, not merely left unused.
  std::optional<cphd::PvpBlock> pvp =
      file_->read_pvp_block(channel_.get(), first, count);

  // 2. Phase history, read straight into the staging buffer at this
  // instance's precision. A single-precision instance reading CF8 makes no
  // copy beyond this one: CF8 is big-endian complex<float>, so the bytes land
  // here and are swapped in place.
  std::vector<std::complex<T>> history(
      static_cast<size_t>(count * num_samples));
  file_->read_signal_into(channel_.get(), first, count, history.data(),
                          &pvp.value());

  // The aperture phase centre is the midpoint of the transmit and receive
  // phase centres, which reduces to either one for the monostatic
  // collections this wave covers. Which of the three readings the contract
  // means is a convention still to be pinned against the mode 3 collections
  // -- see this operator's OPEN_ITEMS.md.
  const std::vector<double> tx_pos = pvp->field("TxPos", 3);
  const std::vector<double> rcv_pos = pvp->field("RcvPos", 3);
  std::vector<double> apc(tx_pos.size());
  for (size_t i = 0; i < apc.size(); ++i) {
    apc[i] = 0.5 * (tx_pos[i] + rcv_pos[i]);
  }
  const std::vector<double> srp = pvp->field("SRPPos", 3);

  // Public collections do carry non-finite ephemeris on a minority of
  // vectors -- one surveyed Umbra file has NaN Tx fields on 32 of 4844, with
  // a fully populated signal array. Passing them through keeps a collection
  // that is otherwise good, but silently: the NaN reaches the image as blank
  // pixels with nothing to attribute them to. So say it once, with the count,
  // and leave the policy decision (drop? refuse?) to whoever hits a file bad
  // enough to need one.
  if (!nonfinite_warned_) {
    int64_t bad = 0;
    for (int64_t pulse = 0; pulse < count; ++pulse) {
      const size_t base = static_cast<size_t>(pulse) * 3;
      if (!std::isfinite(apc[base]) || !std::isfinite(apc[base + 1]) ||
          !std::isfinite(apc[base + 2])) {
        ++bad;
      }
    }
    if (bad > 0) {
      nonfinite_warned_ = true;
      HOLOSCAN_LOG_WARN(
          "{}: {} of {} pulses in this batch have non-finite phase-centre "
          "positions (NaN TxPos/RcvPos in the file); they are passed through "
          "and will appear as blank image regions. Warned once per instance.",
          who, bad, count);
    }
  }

  // 3. The fast-time sample axis. SC0 and SCSS carry it in both domains --
  // Hz and Hz per sample for an FX collection, seconds and seconds per sample
  // for a TOA one -- so the arithmetic is the same and only the unit the
  // contract ascribes to the port differs.
  //
  // They are per-vector fields while the axis is one shared row, so their
  // constancy over the batch is a precondition rather than an assumption: a
  // collection that steps its start frequency would otherwise be read at the
  // wrong frequencies and produce a plausible, wrong image.
  //
  // Tolerantly, though: these are doubles decoded per vector from the file,
  // and a collection holding one start frequency across the aperture still
  // shows last-place jitter between records. A genuinely stepped axis moves by
  // parts in a million or more, which this still refuses.
  const std::vector<double> sc0 = pvp->field("SC0", 1);
  const std::vector<double> scss = pvp->field("SCSS", 1);
  constexpr double kConstantRelTol = 1e-12;
  const auto differs = [&](double value, double reference) {
    return std::abs(value - reference) >
           kConstantRelTol * std::max(std::abs(value), std::abs(reference));
  };
  for (int64_t pulse = 1; pulse < count; ++pulse) {
    if (differs(sc0[pulse], sc0[0]) || differs(scss[pulse], scss[0])) {
      throw std::runtime_error(
          who + ": SC0/SCSS vary across pulses by more than floating-point "
          "jitter (pulse " + std::to_string(first + pulse) +
          " differs from the batch's first by more than a relative 1e-12); a "
          "per-pulse sample axis is not supported");
    }
  }
  std::vector<T> axis(static_cast<size_t>(num_samples));
  for (int64_t sample = 0; sample < num_samples; ++sample) {
    axis[static_cast<size_t>(sample)] =
        static_cast<T>(sc0[0] + static_cast<double>(sample) * scss[0]);
  }

  // 4. Emit. The boundary key is stamped before the emits so it rides the
  // messages of this batch rather than the next one -- a seam reported one
  // batch late would already have let the next collection's pulses into the
  // closing aperture.
  if (is_final && is_metadata_enabled()) {
    metadata()->set(kStreamBoundaryKey, std::string(is_last_file
                                                        ? kEndOfStream
                                                        : kCollectionBoundary));
  }

  const std::vector<int64_t> pulse_shape{count, 3};
  emit_single(
      op_output, "phase_history",
      upload(history, {count, num_samples}, complex_dtype<T>(), who.c_str()));
  emit_single(
      op_output, "positions",
      upload(narrow<T>(apc), pulse_shape, real_dtype<T>(), who.c_str()));
  emit_single(
      op_output, "reference_position",
      upload(narrow<T>(srp), pulse_shape, real_dtype<T>(), who.c_str()));
  emit_single(op_output, "sample_axis",
              upload(axis, {num_samples}, real_dtype<T>(), who.c_str()));

  // Pulse times stay float64 in both instantiations: a collection-relative
  // timestamp does not survive single precision to useful accuracy.
  const std::vector<double> times = pvp->field("TxTime", 1);
  emit_single(op_output, "pulse_times",
              upload(times, {count}, real_dtype<double>(), who.c_str()));

  if (pvp->has("TxVel") && pvp->has("RcvVel")) {
    const std::vector<double> tx_vel = pvp->field("TxVel", 3);
    const std::vector<double> rcv_vel = pvp->field("RcvVel", 3);
    std::vector<double> velocity(tx_vel.size());
    for (size_t i = 0; i < velocity.size(); ++i) {
      velocity[i] = 0.5 * (tx_vel[i] + rcv_vel[i]);
    }
    emit_single(op_output, "velocities",
                upload(narrow<T>(velocity), pulse_shape, real_dtype<T>(),
                       who.c_str()));
  }

  // The descriptor edge is emitted once, at the first tick: a consumer
  // wiring it into a [live] override needs the value once, overrides being
  // sticky.
  if (!frame_emitted_) {
    op_output.emit(coordinate_frame_.get(), "coordinate_frame");
    frame_emitted_ = true;
  }

  next_pulse_ = first + count;
  if (!is_final) {
    return;
  }
  if (is_last_file) {
    HOLOSCAN_LOG_INFO("{}: end of stream after {} pulses of '{}'", who,
                      last_pulse_ - first_pulse_, files_[file_index_]);
    stop_condition_->disable_tick();
    return;
  }
  // Advance to the next collection. The seam has already been announced on
  // the batch just emitted, so the consumer's flush and reset happen before
  // any pulse of the next file reaches it.
  HOLOSCAN_LOG_INFO("{}: collection boundary after {} pulses of '{}'", who,
                    last_pulse_ - first_pulse_, files_[file_index_]);
  // Released before the file it points into is replaced: the block holds the
  // old file's PVP layout and path by pointer.
  pvp.reset();
  ++file_index_;
  open_current_file();
}

template class CPHDReaderOp<float>;
template class CPHDReaderOp<double>;

}  // namespace holoscan::radar
