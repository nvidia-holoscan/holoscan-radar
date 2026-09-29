// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <exception>
#include <filesystem>  // NOLINT(build/c++17)
#include <iostream>
#include <string>

#include "io/cphd_fixture.hpp"  // NOLINT(build/include_subdir)

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: generate_cphd_fixture <path.cphd>\n";
    return 2;
  }
  try {
    const auto path = holoscan::radar::testing::cphd_fixture::make_fixture(argv[1]);
    holoscan::radar::testing::cphd_fixture::write_truth(path);
    const auto shifted = path.parent_path() / (path.stem().string() + "_shifted.cphd");
    holoscan::radar::testing::cphd_fixture::make_fixture(shifted, 0.0, 0.0, 6378100.0);
    const auto long_stream = path.parent_path() / (path.stem().string() + "_long.cphd");
    holoscan::radar::testing::cphd_fixture::make_fixture(long_stream, 0.0, 0.0, 6378000.0, false,
                                                         20);
    const auto wide = path.parent_path() / (path.stem().string() + "_wide.cphd");
    holoscan::radar::testing::cphd_fixture::make_fixture(wide, 0.0, 0.0, 6378200.0, true);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
