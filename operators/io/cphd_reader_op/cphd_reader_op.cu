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
#include <type_traits>
#include <utility>
#include <vector>

#include "io/cphd_reader_op/cphd_file.hpp"
#include "radar_common/descriptors/coordinate_frame.hpp"
#include "radar_common/descriptors/enums.hpp"
#include "radar_common/sdk/arg_setters.hpp"
#include "radar_common/sdk/stream_boundary.hpp"
#include "radar_common/sdk/tensor_io.hpp"
#include "radar_common/tensor/dlpack_buffer.hpp"
#include "radar_common/tensor/tensor_utils.hpp"

namespace holoscan::radar {

struct CphdReaderResources {
  void* history{nullptr};
  size_t history_bytes{0};
  void* scratch{nullptr};
  size_t scratch_bytes{0};

  CphdReaderResources() = default;
  CphdReaderResources(const CphdReaderResources&) = delete;
  CphdReaderResources& operator=(const CphdReaderResources&) = delete;

  ~CphdReaderResources() {
    if (history != nullptr) {
      cudaFreeHost(history);
    }
    if (scratch != nullptr) {
      cudaFreeHost(scratch);
    }
  }

  void ensure_history(size_t bytes) {
    if (bytes <= history_bytes) {
      return;
    }
    if (history != nullptr) {
      cudaFreeHost(history);
      history = nullptr;
      history_bytes = 0;
    }
    if (bytes == 0) {
      return;
    }
    if (cudaMallocHost(&history, bytes) != cudaSuccess) {
      throw std::runtime_error(
          "CPHDReaderOp: pinned history allocation failed");
    }
    history_bytes = bytes;
  }

  void ensure_scratch(size_t bytes) {
    if (bytes <= scratch_bytes) {
      return;
    }
    if (scratch != nullptr) {
      cudaFreeHost(scratch);
      scratch = nullptr;
      scratch_bytes = 0;
    }
    if (bytes == 0) {
      return;
    }
    if (cudaMallocHost(&scratch, bytes) != cudaSuccess) {
      throw std::runtime_error(
          "CPHDReaderOp: pinned decode-scratch allocation failed");
    }
    scratch_bytes = bytes;
  }
};

namespace {

template <typename T>
constexpr const char* precision_name() {
  return std::is_same_v<T, float> ? "float" : "double";
}

// Whether `path` ends in the CPHD extension, compared without ASCII case:
// a directory holding SCENE.CPHD would otherwise contribute nothing, and the
// extension is the only thing that selects a collection.
bool has_cphd_extension(const std::filesystem::path& path) {
  const std::string extension = path.extension().string();
  constexpr char kExpected[] = ".cphd";
  if (extension.size() != sizeof(kExpected) - 1) {
    return false;
  }
  for (size_t index = 0; index < extension.size(); ++index) {
    const char character = extension[index];
    const char lowered = (character >= 'A' && character <= 'Z')
                             ? static_cast<char>(character - 'A' + 'a')
                             : character;
    if (lowered != kExpected[index]) {
      return false;
    }
  }
  return true;
}

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
      if (item.is_regular_file() && has_cphd_extension(item.path())) {
        found.push_back(item.path().string());
      }
    }
    if (found.empty()) {
      throw std::runtime_error(who + ": directory '" + entry +
                               "' contains no .cphd files");
    }
    // directory_iterator order is unspecified, so the contract's
    // lexicographic promise has to be imposed here rather than assumed.
    //
    // Lexicographic by the whole path string, which is the defined collection
    // order and is not numeric: `pulse_10.cphd` precedes `pulse_2.cphd`. A
    // directory whose names carry an index needs them zero-padded to a fixed
    // width; an explicit list is the way to read any other order.
    std::sort(found.begin(), found.end());
    files.insert(files.end(), found.begin(), found.end());
  }
  if (files.empty()) {
    throw std::runtime_error(who + ": path selects no files");
  }
  return files;
}

// Copy a host buffer to a freshly allocated device tensor of `shape`.
// On `stream` (non-default) the copy is async; the caller must keep
// `host` alive until that stream catches up, and should stamp the
// stream on the emitted port so the consumer waits for it.
template <typename Element>
std::shared_ptr<holoscan::Tensor> upload(const Element* host, size_t count,
                                         const std::vector<int64_t>& shape,
                                         DLDataType dtype, cudaStream_t stream,
                                         const char* who) {
  auto tensor = make_device_tensor(shape, dtype);
  const size_t bytes = sizeof(Element) * count;
  if (bytes != static_cast<size_t>(tensor->nbytes())) {
    throw std::runtime_error(
        std::string(who) + ": H2D upload of " + std::to_string(bytes) +
        " host bytes into a " + std::to_string(tensor->nbytes()) +
        "-byte device tensor");
  }
  if (bytes == 0) {
    return tensor;
  }
  const cudaError_t copied =
      (stream == cudaStreamDefault)
          ? cudaMemcpy(tensor->data(), host, bytes, cudaMemcpyHostToDevice)
          : cudaMemcpyAsync(tensor->data(), host, bytes,
                            cudaMemcpyHostToDevice, stream);
  if (copied != cudaSuccess) {
    throw std::runtime_error(std::string(who) + ": H2D upload failed");
  }
  return tensor;
}

template <typename Element>
std::shared_ptr<holoscan::Tensor> upload(const std::vector<Element>& host,
                                         const std::vector<int64_t>& shape,
                                         DLDataType dtype, cudaStream_t stream,
                                         const char* who) {
  return upload(host.data(), host.size(), shape, dtype, stream, who);
}

// Stamp the GXF-managed H2D stream on the port before emit: set_cuda_stream
// must precede emit, and only accepts streams from allocate_cuda_stream.
void emit_with_stream(holoscan::OutputContext& op_output, const char* port,
                      std::shared_ptr<holoscan::Tensor> tensor,
                      cudaStream_t stream) {
  op_output.set_cuda_stream(stream, port);
  emit_single(op_output, port, std::move(tensor));
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

  // H2D needs a GXF-managed stream so set_cuda_stream can attach it to
  // output ports. A homemade cudaStreamCreate stream is not in GXF's map
  // and logs an error on every port every tick. Non-blocking so the copy
  // does not wait for unrelated kernels on the default stream.
  add_arg(fragment()->make_resource<holoscan::CudaStreamPool>(
      "cuda_stream_pool", 0,
      static_cast<uint32_t>(cudaStreamNonBlocking), 0, 1, 0));

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

  // Pinned staging lives for the operator's lifetime so a tick does not
  // value-initialize 60 MiB of phase history. The H2D stream is allocated
  // in compute() from the GXF pool (set_cuda_stream requires that).
  if (!resources_) {
    resources_ = std::make_shared<CphdReaderResources>();
  }
}

template <typename T>
void CPHDReaderOp<T>::compute([[maybe_unused]] InputContext& op_input,
                              OutputContext& op_output,
                              ExecutionContext& context) {
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

  // 2. Phase history, read into reused pinned staging at this instance's
  // precision. A single-precision instance reading CF8 makes no copy
  // beyond this one: CF8 is big-endian complex<float>, so the bytes land
  // here and are swapped in place. The buffer is not zeroed; decode
  // overwrites every element.
  if (resources_ == nullptr) {
    throw std::runtime_error(who + ": H2D staging was not created");
  }
  auto maybe_stream = context.allocate_cuda_stream("h2d");
  if (!maybe_stream) {
    throw std::runtime_error(who + ": failed to allocate H2D stream: " +
                             maybe_stream.error().what());
  }
  const cudaStream_t stream = maybe_stream.value();
  const size_t nsamp = static_cast<size_t>(count * num_samples);
  resources_->ensure_history(nsamp * sizeof(std::complex<T>));
  auto* history = static_cast<std::complex<T>*>(resources_->history);
  resources_->ensure_scratch(nsamp * 8);
  file_->read_signal_into(channel_.get(), first, count, history,
                          &pvp.value(),
                          static_cast<char*>(resources_->scratch),
                          resources_->scratch_bytes);

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
    metadata()->set(
        kStreamBoundaryKey,
        std::string(is_last_file ? kStreamBoundaryEndOfStream
                                 : kStreamBoundaryCollection));
  }

  const std::vector<int64_t> pulse_shape{count, 3};
  // Named host buffers: async H2D from a temporary would dangle when
  // the statement ends, before the stream sync below.
  std::vector<T> pos_host(apc.size());
  std::vector<T> srp_host(srp.size());
  for (size_t i = 0; i < apc.size(); ++i) {
    pos_host[i] = static_cast<T>(apc[i]);
  }
  for (size_t i = 0; i < srp.size(); ++i) {
    srp_host[i] = static_cast<T>(srp[i]);
  }
  emit_with_stream(
      op_output, "phase_history",
      upload(history, nsamp, {count, num_samples}, complex_dtype<T>(),
             stream, who.c_str()),
      stream);
  emit_with_stream(
      op_output, "positions",
      upload(pos_host, pulse_shape, real_dtype<T>(), stream, who.c_str()),
      stream);
  emit_with_stream(
      op_output, "reference_position",
      upload(srp_host, pulse_shape, real_dtype<T>(), stream, who.c_str()),
      stream);
  emit_with_stream(op_output, "sample_axis",
                   upload(axis, {num_samples}, real_dtype<T>(), stream,
                          who.c_str()),
                   stream);

  // Pulse times stay float64 in both instantiations: a collection-relative
  // timestamp does not survive single precision to useful accuracy.
  const std::vector<double> times = pvp->field("TxTime", 1);
  emit_with_stream(op_output, "pulse_times",
                   upload(times, {count}, real_dtype<double>(), stream,
                          who.c_str()),
                   stream);

  std::vector<T> vel_host;
  if (pvp->has("TxVel") && pvp->has("RcvVel")) {
    const std::vector<double> tx_vel = pvp->field("TxVel", 3);
    const std::vector<double> rcv_vel = pvp->field("RcvVel", 3);
    vel_host.resize(tx_vel.size());
    for (size_t i = 0; i < tx_vel.size(); ++i) {
      vel_host[i] = static_cast<T>(0.5 * (tx_vel[i] + rcv_vel[i]));
    }
    emit_with_stream(op_output, "velocities",
                     upload(vel_host, pulse_shape, real_dtype<T>(), stream,
                            who.c_str()),
                     stream);
  }

  // Locals (positions, axis, times) back the small copies; wait so they
  // stay alive for the DMA. History is pinned and reused, so this wait
  // is also what keeps the next tick from overwriting it while this
  // tick's H2D is still in flight.
  if (cudaStreamSynchronize(stream) != cudaSuccess) {
    throw std::runtime_error(who + ": H2D stream sync failed");
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
    HOLOSCAN_LOG_INFO(
        "{}: end of stream: {} pulses read from the final collection '{}' "
        "({} of {})",
        who, last_pulse_ - first_pulse_, files_[file_index_], file_index_ + 1,
        files_.size());
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
