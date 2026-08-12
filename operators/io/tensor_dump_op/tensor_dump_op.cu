// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "io/tensor_dump_op/tensor_dump_op.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>  // NOLINT(build/c++17): repo builds with C++17
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <holoscan/holoscan.hpp>

#include "io/tensor_dump_op/npy_writer.hpp"
#include "radar_common/sdk/arg_setters.hpp"
#include "radar_common/sdk/override_param.hpp"
#include "radar_common/sdk/tensor_io.hpp"
#include "radar_common/sdk/yaml_converters.hpp"
#include "radar_common/tensor/tensor_utils.hpp"

// YAML::convert for the operator-local TensorDumpFormat enum, mirroring the
// shared converters in radar_common/sdk/yaml_converters.hpp (reusing
// detail::parse_enum) so ArgumentSetter::ensure_type<TensorDumpFormat>()
// supports from_config(). Kept in this TU because the enum is dump-local;
// common must not depend on it.
namespace YAML {
template <>
struct convert<holoscan::radar::TensorDumpFormat> {
  static Node encode(const holoscan::radar::TensorDumpFormat& value) {
    return Node(std::string(holoscan::radar::to_string(value)));
  }
  static bool decode(const Node& node,
                     holoscan::radar::TensorDumpFormat& value) {
    if (!node.IsScalar()) {
      return false;
    }
    static constexpr holoscan::radar::TensorDumpFormat kAll[] = {
        holoscan::radar::TensorDumpFormat::kNpy,
        holoscan::radar::TensorDumpFormat::kRaw};
    value = holoscan::radar::detail::parse_enum(node.as<std::string>(),
                                                "TensorDumpFormat", kAll);
    return true;
  }
};
}  // namespace YAML

namespace holoscan::radar {

namespace {

constexpr const char* kOpName = "TensorDumpOp";
constexpr const char* kIndexToken = "{index}";

// Substitute every occurrence of `{index}`. A pattern without the token is
// legal (start() warns when more than one file can be written) so dumping a
// single tensor needs no placeholder.
std::string substitute_index(const std::string& pattern, int64_t index) {
  const std::string replacement = std::to_string(index);
  std::string out;
  out.reserve(pattern.size() + replacement.size());
  size_t pos = 0;
  for (;;) {
    const size_t hit = pattern.find(kIndexToken, pos);
    if (hit == std::string::npos) {
      out.append(pattern, pos, std::string::npos);
      return out;
    }
    out.append(pattern, pos, hit - pos);
    out.append(replacement);
    pos = hit + std::char_traits<char>::length(kIndexToken);
  }
}

// Reject the value the contract refuses to guess at: zero reads as "write
// nothing", which is what max_files = 0 already means.
void require_positive_every_n(int64_t every_n, const char* origin) {
  if (every_n < 1) {
    std::ostringstream msg;
    msg << kOpName << ": every_n must be >= 1, got " << every_n << " (from "
        << origin
        << "). Zero is not coerced to 1: to write nothing, set max_files: 0.";
    throw std::invalid_argument(msg.str());
  }
}

}  // namespace

void TensorDumpOp::initialize() {
  ensure_radar_arg_setters();
  // `format` is the sink's own enum rather than a shared descriptor, so it
  // registers its setter here rather than in the common list. Idempotent.
  ArgumentSetter::ensure_type<TensorDumpFormat>();
  Operator::initialize();
}

void TensorDumpOp::setup(OperatorSpec& spec) {
  spec.input<holoscan::TensorMap>("tensor");

  spec.param(path_, "path", "Output path pattern",
             "Where each dump is written. '{index}' is replaced by the tick "
             "counter, so a pattern without it overwrites one file.");

  spec.param(format_, "format", "On-disk format",
             "npy carries shape and dtype and reads back directly from "
             "NumPy; raw is element bytes with no header.",
             TensorDumpFormat::kNpy);

  spec.param(every_n_, "every_n", "Write every nth message",
             "Throttle for long runs. Must be >= 1; zero and negative are "
             "rejected rather than coerced.",
             static_cast<int64_t>(1));
  add_override_port<int64_t>(spec, "every_n");

  spec.param(max_files_, "max_files", "File limit",
             "Stop writing after this many files, warning once. Unset means "
             "unlimited.",
             ParameterFlag::kOptional);
}

void TensorDumpOp::start() {
  require_positive_every_n(every_n_.get(), "the every_n parameter");
  effective_every_n_ = every_n_.get();

  if (max_files_.has_value() && max_files_.get() < 0) {
    std::ostringstream msg;
    msg << kOpName << ": max_files must be >= 0, got " << max_files_.get()
        << " (0 writes nothing; leave it unset for unlimited)";
    throw std::invalid_argument(msg.str());
  }

  const bool one_file_only = max_files_.has_value() && max_files_.get() <= 1;
  if (path_.get().find(kIndexToken) == std::string::npos && !one_file_only) {
    HOLOSCAN_LOG_WARN(
        "{}: path '{}' has no '{}' placeholder, so every dump overwrites the "
        "previous one. Add '{}' to keep them, or set max_files: 1 to say one "
        "file is what you meant.",
        kOpName, path_.get(), kIndexToken, kIndexToken);
  }
}

void TensorDumpOp::compute(InputContext& op_input,
                           [[maybe_unused]] OutputContext& op_output,
                           [[maybe_unused]] ExecutionContext& context) {
  const int64_t index = tick_++;

  bool changed = false;
  const int64_t every_n = resolve_override_param<int64_t>(
      op_input, "every_n", effective_every_n_, &changed);
  require_positive_every_n(every_n, "the every_n override port");
  if (changed) {
    HOLOSCAN_LOG_INFO("{}: every_n overridden to {} at tick {}", kOpName,
                      every_n, index);
  }
  effective_every_n_ = every_n;

  // Received unconditionally: the message is consumed whether or not this
  // tick is written, so throttling never backs up the upstream queue.
  std::shared_ptr<holoscan::Tensor> tensor =
      receive_single(op_input, "tensor", kOpName);

  if (index % every_n != 0) {
    return;
  }
  if (max_files_.has_value() && files_written_ >= max_files_.get()) {
    if (!limit_warned_) {
      limit_warned_ = true;
      HOLOSCAN_LOG_WARN(
          "{}: max_files ({}) reached at tick {}; later messages are consumed "
          "and discarded.",
          kOpName, max_files_.get(), index);
    }
    return;
  }

  require_cuda_accessible(*tensor, "tensor", kOpName);
  // The dump is one flat write, so a strided view would land bytes that do
  // not correspond to the shape recorded beside them.
  require_contiguous(*tensor, "tensor", kOpName);

  const std::string out_path = substitute_index(path_.get(), index);
  const std::filesystem::path parent =
      std::filesystem::path(out_path).parent_path();
  if (!parent.empty() && parent != last_parent_) {
    std::error_code ec;
    std::filesystem::create_directories(parent, ec);
    if (ec) {
      throw std::runtime_error(std::string(kOpName) +
                               ": cannot create output directory '" +
                               parent.string() + "': " + ec.message());
    }
    last_parent_ = parent.string();
  }

  const size_t nbytes = tensor->nbytes();
  host_.resize(nbytes);
  if (nbytes > 0) {
    cudaStream_t stream = op_input.receive_cuda_stream("tensor");
    // Stream-ordered, then synchronized, and both are load-bearing: the copy
    // orders itself behind the producer's kernels on this stream, which a
    // plain cudaMemcpy would race for device memory, and the synchronize is
    // what makes host_ safe to read -- the async copy has only been queued
    // by the time it returns.
    const cudaError_t copied =
        cudaMemcpyAsync(host_.data(), tensor->data(), nbytes,
                        cudaMemcpyDefault, stream);
    if (copied != cudaSuccess) {
      throw std::runtime_error(std::string(kOpName) +
                               ": device-to-host copy failed: " +
                               cudaGetErrorString(copied));
    }
    const cudaError_t synced = cudaStreamSynchronize(stream);
    if (synced != cudaSuccess) {
      throw std::runtime_error(std::string(kOpName) +
                               ": stream synchronize failed: " +
                               cudaGetErrorString(synced));
    }
  }

  if (format_.get() == TensorDumpFormat::kNpy) {
    npy::write(out_path, tensor->dtype(), tensor->shape(), host_.data(),
               nbytes);
  } else {
    npy::write_raw(out_path, host_.data(), nbytes);
  }
  ++files_written_;

  HOLOSCAN_LOG_DEBUG("{}: wrote {} ({} bytes, {})", kOpName, out_path, nbytes,
                     to_string(format_.get()));
}

void TensorDumpOp::stop() {
  HOLOSCAN_LOG_INFO("{}: wrote {} file(s) over {} message(s) to '{}'", kOpName,
                    files_written_, tick_, path_.get());
}

}  // namespace holoscan::radar
