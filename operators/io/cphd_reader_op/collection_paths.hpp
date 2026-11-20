// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <utility>
#include <vector>

#include <holoscan/config/codec.hpp>
#include <holoscan/core/parameter.hpp>

#include "radar_common/sdk/config_codecs.hpp"

// The `path` parameter of the standard-format readers, which the operator README
// defines as "a single CPHD file, a directory of them, or an explicit ordered
// list". Two of those three are written as a YAML scalar and the third as a
// sequence, and no built-in parameter type accepts both: the codec for
// std::vector<std::string> refuses a scalar, so a plain vector parameter
// would force every single-file application to write `path: [one.cphd]`.
//
// Hence a small type of our own, whose ConfigCodec takes either form.
// Operator parameters decode through holoscan::config, so the installed
// header needs no yaml-cpp. It lives beside the reader rather than in
// radar_common because it is a parameter spelling rather than a shared
// descriptor -- invariant 2 bounds that set deliberately. `CRSDReaderOp` takes
// the same parameter and should share this when it lands, at which point the
// io group grows a common header.
//
// An empty sequence (`path: []`) decodes successfully, to no entries. It is
// the reader that refuses it, at `start()`, naming the parameter -- deciding
// here instead would cost that message: the codec would report a generic
// decode error rather than naming an empty list.
namespace holoscan::radar {

struct CollectionPaths {
  std::vector<std::string> entries;

  CollectionPaths() = default;
  explicit CollectionPaths(std::vector<std::string> values) : entries(std::move(values)) {}
  explicit CollectionPaths(std::string value) : entries{std::move(value)} {}

  bool empty() const { return entries.empty(); }
};

}  // namespace holoscan::radar

namespace holoscan::config {
template <>
struct ConfigCodec<holoscan::radar::CollectionPaths> {
  static ConfigResult<holoscan::radar::CollectionPaths> decode(ConfigView view,
                                                               const DecodeOptions& options,
                                                               DecodeContext& context) {
    if (view.kind() == ConfigValueKind::kString) {
      auto one = ConfigCodec<std::string>::decode(std::move(view), options, context);
      if (!one)
        return tl::make_unexpected(one.error());
      return holoscan::radar::CollectionPaths(std::move(*one));
    }
    auto many = ConfigCodec<std::vector<std::string>>::decode(std::move(view), options, context);
    if (!many)
      return tl::make_unexpected(many.error());
    return holoscan::radar::CollectionPaths(std::move(*many));
  }
  static ConfigResult<ConfigValue> encode(const holoscan::radar::CollectionPaths& value,
                                          const EncodeOptions& options, EncodeContext& context) {
    return ConfigCodec<std::vector<std::string>>::encode(value.entries, options, context);
  }
};
}  // namespace holoscan::config

namespace holoscan {
template <>
struct ParamStorageTraits<radar::CollectionPaths>
: radar::config_detail::PortableParamStorageTraits<radar::CollectionPaths> {};
}  // namespace holoscan
