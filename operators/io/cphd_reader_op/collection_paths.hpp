// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <yaml-cpp/yaml.h>

#include <string>
#include <utility>
#include <vector>

#include <holoscan/config/codec.hpp>
#include <holoscan/core/parameter.hpp>

#include "radar_common/sdk/config_codecs.hpp"

// The `path` parameter of the standard-format readers, which OPERATOR_API.md
// defines as "a single CPHD file, a directory of them, or an explicit ordered
// list". Two of those three are written as a YAML scalar and the third as a
// sequence, and no built-in parameter type accepts both: yaml-cpp refuses a
// scalar where it wants std::vector<std::string>, so a plain vector parameter
// would force every single-file application to write `path: [one.cphd]`.
//
// Hence a small type of our own, whose converter takes either form. It lives
// beside the reader rather than in radar_common because it is a parameter
// spelling rather than a shared descriptor -- invariant 2 bounds that set
// deliberately. `CRSDReaderOp` takes the same parameter and should share this
// when it lands, at which point the io group grows a common header.
//
// An empty sequence (`path: []`) decodes successfully, to no entries. It is
// the reader that refuses it, at `start()`, naming the parameter -- deciding
// here instead would cost that message: a converter can only report failure
// as a YAML conversion error, which surfaces as an unset required parameter
// rather than as an empty list.
namespace holoscan::radar {

struct CollectionPaths {
  std::vector<std::string> entries;

  CollectionPaths() = default;
  explicit CollectionPaths(std::vector<std::string> values) : entries(std::move(values)) {}
  explicit CollectionPaths(std::string value) : entries{std::move(value)} {}

  bool empty() const { return entries.empty(); }
};

}  // namespace holoscan::radar

namespace YAML {

template <>
struct convert<holoscan::radar::CollectionPaths> {
  static Node encode(const holoscan::radar::CollectionPaths& value) {
    Node node(NodeType::Sequence);
    for (const std::string& entry : value.entries) {
      node.push_back(entry);
    }
    return node;
  }

  static bool decode(const Node& node, holoscan::radar::CollectionPaths& value) {
    if (node.IsScalar()) {
      value.entries = {node.as<std::string>()};
      return true;
    }
    if (node.IsSequence()) {
      value.entries.clear();
      for (const Node& child : node) {
        if (!child.IsScalar()) {
          return false;
        }
        value.entries.push_back(child.as<std::string>());
      }
      return true;
    }
    return false;
  }
};

}  // namespace YAML

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
