// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "io/tensor_dump_op/tensor_dump_op.hpp"

#include <cuda_runtime_api.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>  // NOLINT(build/c++17)
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include <holoscan/core/expected_macros.hpp>
#include <holoscan/logger/logger.hpp>

#include "io/tensor_dump_op/npy_writer.hpp"
#include "radar_common/sdk/override_param.hpp"
#include "radar_common/sdk/tensor_io.hpp"
#include "radar_common/tensor/tensor_utils.hpp"

namespace holoscan::radar {
namespace {
constexpr const char* kOpName = "TensorDumpOp";
constexpr const char* kIndexToken = "{index}";

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
}  // namespace

void TensorDumpOp::setup(OperatorSpec& spec) {
  constexpr TensorRepresentation device{
      .memory_kind = MemoryKind::kCudaDevice, .dtype = std::nullopt, .rank = std::nullopt};
  spec.input(tensor).queue_depth(2U).expects_tensor(tensor_input(device));
  spec.input(every_n_in).optional().queue_depth(2U);
}

Contract TensorDumpOp::contract() const {
  Contract result;
  result.trigger(OnEach{tensor});
  add_live_override_to_contract(result, every_n_in);
  return result;
}

void TensorDumpOp::start() {
  const auto& p = params();
  if (p.path.empty() || p.every_n < 1 || (p.max_files.has_value() && *p.max_files < 0)) {
    throw std::invalid_argument(
        "TensorDumpOp: path must be nonempty, every_n >= 1, and max_files >= 0");
  }
  const bool one_file_only = p.max_files.has_value() && *p.max_files <= 1;
  if (p.path.find(kIndexToken) == std::string::npos && !one_file_only) {
    HOLOSCAN_LOG_WARN("{}: path '{}' has no '{}' placeholder; later dumps overwrite it", kOpName,
                      p.path, kIndexToken);
  }
}

expected<void, Error> TensorDumpOp::compute(ExecutionContext& context) {
  const int64_t index = tick_++;
  const auto& p = context.params<TensorDumpParams>();
  HOLOSCAN_TRY(auto every_n, resolve_live_override(every_n_in, p.every_n));
  if (every_n < 1) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument, "TensorDumpOp: every_n must be >= 1"});
  }
  HOLOSCAN_TRY(auto input, receive_tensor(tensor, kOpName));
  if (index % every_n != 0) {
    return {};
  }
  if (p.max_files.has_value() && files_written_ >= *p.max_files) {
    if (!limit_warned_) {
      limit_warned_ = true;
      HOLOSCAN_LOG_WARN("{}: max_files ({}) reached at tick {}", kOpName, *p.max_files, index);
    }
    return {};
  }
  try {
    require_cuda_accessible(input, "tensor", kOpName);
    require_contiguous(input, "tensor", kOpName);
  } catch (const std::invalid_argument& e) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument, e.what()});
  }
  const std::string out_path = substitute_index(p.path, index);
  const std::filesystem::path parent = std::filesystem::path(out_path).parent_path();
  if (!parent.empty() && parent.string() != last_parent_) {
    std::error_code ec;
    std::filesystem::create_directories(parent, ec);
    if (ec) {
      return make_unexpected(
          Error{ErrorCode::kFailure, std::string(kOpName) + ": cannot create output directory '" +
                                         parent.string() + "': " + ec.message()});
    }
    last_parent_ = parent.string();
  }
  const size_t nbytes = input.nbytes();
  const auto t0 = std::chrono::steady_clock::now();
  host_.resize(nbytes);
  if (nbytes > 0) {
    cudaStream_t stream = context.cuda_stream().get();
    if (stream == nullptr) {
      return make_unexpected(
          Error{ErrorCode::kNotSupported, "TensorDumpOp: no CUDA compute stream"});
    }
    const cudaError_t copied =
        cudaMemcpyAsync(host_.data(), input.data(), nbytes, cudaMemcpyDefault, stream);
    if (copied != cudaSuccess) {
      return make_unexpected(Error{
          ErrorCode::kFailure,
          std::string("TensorDumpOp: device-to-host copy failed: ") + cudaGetErrorString(copied)});
    }
    const cudaError_t synced = cudaStreamSynchronize(stream);
    if (synced != cudaSuccess) {
      return make_unexpected(
          Error{ErrorCode::kFailure,
                std::string("TensorDumpOp: stream sync failed: ") + cudaGetErrorString(synced)});
    }
  }
  const std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
  try {
    if (p.format == TensorDumpFormat::kNpy) {
      npy::write(out_path, input.dtype(), shape, host_.data(), nbytes);
    } else {
      npy::write_raw(out_path, host_.data(), nbytes);
    }
  } catch (const std::exception& e) {
    return make_unexpected(Error{ErrorCode::kFailure, e.what()});
  }
  ++files_written_;
  const double elapsed =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  HOLOSCAN_LOG_INFO("{}: wrote {} ({:.2f} GiB) in {:.3f} s", kOpName, out_path,
                    static_cast<double>(nbytes) / (1024.0 * 1024.0 * 1024.0), elapsed);
  return {};
}

void TensorDumpOp::stop() {
  HOLOSCAN_LOG_INFO("{}: wrote {} file(s) over {} message(s) to '{}'", kOpName, files_written_,
                    tick_, params().path);
}

}  // namespace holoscan::radar
