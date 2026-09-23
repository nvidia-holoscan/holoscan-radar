// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "radar_common/dsp/virtual_array.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace holoscan::radar::detail {
namespace {

constexpr double kPositionTol = 1e-9;

bool positions_equal(const double* a, const double* b) {
  for (int axis = 0; axis < 3; ++axis) {
    if (std::abs(a[axis] - b[axis]) > kPositionTol) {
      return false;
    }
  }
  return true;
}

void validate_positions(const std::vector<double>& positions, int64_t count, const char* field,
                        const char* caller_name) {
  if (count <= 0) {
    throw std::invalid_argument(std::string(caller_name) + ": " + field +
                                " element count must be positive");
  }
  if (positions.size() != static_cast<std::size_t>(count * 3)) {
    throw std::invalid_argument(std::string(caller_name) + ": " + field + " length must be num_" +
                                (field[0] == 't' ? "tx" : "rx") + " * 3");
  }
  for (double coordinate : positions) {
    if (!std::isfinite(coordinate)) {
      throw std::invalid_argument(std::string(caller_name) + ": " + field +
                                  " entries must be finite");
    }
  }
}

struct Group {
  double position[3];
  int64_t rep_channel{0};
  std::vector<int64_t> members;
};

}  // namespace

VirtualAssemblyPlan build_virtual_plan(const std::vector<double>& tx_positions,
                                       const std::vector<double>& rx_positions, int64_t num_tx,
                                       int64_t num_rx, OverlapPolicy policy,
                                       const char* caller_name) {
  if (caller_name == nullptr || caller_name[0] == '\0') {
    throw std::invalid_argument("virtual_array: caller_name must be non-empty");
  }
  validate_positions(tx_positions, num_tx, "tx_positions", caller_name);
  validate_positions(rx_positions, num_rx, "rx_positions", caller_name);

  VirtualAssemblyPlan plan;
  plan.policy = policy;
  plan.num_tx = num_tx;
  plan.num_rx = num_rx;

  const int64_t num_channels = num_tx * num_rx;
  if (policy == OverlapPolicy::kKeepAll) {
    plan.num_virtual = num_channels;
    plan.virtual_positions.resize(static_cast<std::size_t>(num_channels * 3));
    plan.src_channel.resize(static_cast<std::size_t>(num_channels));
    for (int64_t tx = 0; tx < num_tx; ++tx) {
      for (int64_t rx = 0; rx < num_rx; ++rx) {
        const int64_t channel = tx * num_rx + rx;
        const std::size_t out = static_cast<std::size_t>(channel * 3);
        for (int axis = 0; axis < 3; ++axis) {
          plan.virtual_positions[out + static_cast<std::size_t>(axis)] =
              tx_positions[static_cast<std::size_t>(tx * 3 + axis)] +
              rx_positions[static_cast<std::size_t>(rx * 3 + axis)];
        }
        plan.src_channel[static_cast<std::size_t>(channel)] = channel;
      }
    }
    return plan;
  }

  std::vector<Group> groups;
  groups.reserve(static_cast<std::size_t>(num_channels));
  for (int64_t tx = 0; tx < num_tx; ++tx) {
    for (int64_t rx = 0; rx < num_rx; ++rx) {
      const int64_t channel = tx * num_rx + rx;
      double candidate[3];
      for (int axis = 0; axis < 3; ++axis) {
        candidate[axis] = tx_positions[static_cast<std::size_t>(tx * 3 + axis)] +
                          rx_positions[static_cast<std::size_t>(rx * 3 + axis)];
      }
      int64_t group_index = -1;
      for (std::size_t group = 0; group < groups.size(); ++group) {
        if (positions_equal(groups[group].position, candidate)) {
          group_index = static_cast<int64_t>(group);
          break;
        }
      }
      if (group_index < 0) {
        Group group;
        for (int axis = 0; axis < 3; ++axis) {
          group.position[axis] = candidate[axis];
        }
        group.rep_channel = channel;
        group.members.push_back(channel);
        groups.push_back(std::move(group));
      } else {
        groups[static_cast<std::size_t>(group_index)].members.push_back(channel);
      }
    }
  }

  plan.num_virtual = static_cast<int64_t>(groups.size());
  plan.virtual_positions.resize(static_cast<std::size_t>(plan.num_virtual * 3));
  plan.member_offsets.reserve(static_cast<std::size_t>(plan.num_virtual + 1));
  plan.member_offsets.push_back(0);
  for (int64_t group_index = 0; group_index < plan.num_virtual; ++group_index) {
    const Group& group = groups[static_cast<std::size_t>(group_index)];
    for (int axis = 0; axis < 3; ++axis) {
      plan.virtual_positions[static_cast<std::size_t>(group_index * 3 + axis)] =
          group.position[axis];
    }
    if (policy == OverlapPolicy::kDedup) {
      plan.src_channel.push_back(group.rep_channel);
    } else {
      for (int64_t channel : group.members) {
        plan.member_indices.push_back(channel);
      }
      plan.member_offsets.push_back(static_cast<int64_t>(plan.member_indices.size()));
    }
  }
  return plan;
}

ArrayGeometry make_array_geometry(const VirtualAssemblyPlan& plan, double wavelength) {
  ArrayGeometry geometry;
  geometry.element_positions = plan.virtual_positions;
  geometry.wavelength = wavelength;
  geometry.validate();
  return geometry;
}

}  // namespace holoscan::radar::detail
