// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/parameter_schema.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>

#include "radar_common/sdk/config_codecs.hpp"

namespace holoscan::radar {

enum class TensorDumpFormat { kNpy, kRaw };

constexpr std::string_view to_string(TensorDumpFormat format) {
  switch (format) {
    case TensorDumpFormat::kNpy:
      return "npy";
    case TensorDumpFormat::kRaw:
      return "raw";
  }
  return "unknown";
}

namespace config_detail {
inline constexpr TensorDumpFormat kTensorDumpFormats[] = {
    TensorDumpFormat::kNpy, TensorDumpFormat::kRaw};  // NOLINT(whitespace/indent_namespace)
using TensorDumpFormatCodec = CanonicalEnumCodec<TensorDumpFormat, kTensorDumpFormats>;
using TensorDumpFormatStorage = PortableParamStorageTraits<TensorDumpFormat>;
}  // namespace config_detail
}  // namespace holoscan::radar

namespace holoscan::config {
template <>
struct ConfigCodec<radar::TensorDumpFormat> : radar::config_detail::TensorDumpFormatCodec {};
}  // namespace holoscan::config

namespace holoscan {
template <>
struct ParamStorageTraits<radar::TensorDumpFormat> : radar::config_detail::TensorDumpFormatStorage {
};
}  // namespace holoscan

namespace holoscan::radar {

struct TensorDumpParams {
  std::string path{};
  TensorDumpFormat format{TensorDumpFormat::kNpy};
  int64_t every_n{1};
  std::optional<int64_t> max_files{};
};

HOLOSCAN_PARAM_SCHEMA(
    TensorDumpParams, holoscan::param_schema("com.nvidia.holoscan.radar.TensorDumpParams", 1),
    holoscan::field<&TensorDumpParams::path>("path").description(
        "Output path pattern; {index} is replaced by the zero-based input tick."),
    holoscan::field<&TensorDumpParams::format>("format").description("npy or raw."),
    holoscan::field<&TensorDumpParams::every_n>("every_n").description(
        "Write every nth input tick; must be positive."),
    holoscan::field<&TensorDumpParams::max_files>("max_files")
        .description("Optional file count limit; zero writes no files."));

// A deliberately dtype-agnostic debugging sink. No Tensor outputs.
class TensorDumpOp : public holoscan::Operator<TensorDumpParams> {
 public:
  void setup(OperatorSpec& spec) override;
  [[nodiscard]] Contract contract() const override;
  void start() override;
  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override;
  void stop() override;

  Input<Tensor> tensor{"tensor"};
  Input<int64_t> every_n_in{"every_n"};

 private:
  int64_t tick_{0};
  int64_t files_written_{0};
  bool limit_warned_{false};
  std::string last_parent_;
  std::vector<uint8_t> host_;
};

}  // namespace holoscan::radar
