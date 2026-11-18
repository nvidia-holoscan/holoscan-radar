// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>  // NOLINT(build/c++17)
#include <fstream>
#include <optional>
#include <string>

#include "app/config_fixture.hpp"
#include "radar_app/sar/config.hpp"

namespace {

class SarConfigPaths : public ::testing::Test {
 protected:
  void SetUp() override {
    if (const char* value = std::getenv("HOLOHUB_DATA_DIR"))
      previous_root_ = value;
  }

  void TearDown() override {
    if (previous_root_)
      ::setenv("HOLOHUB_DATA_DIR", previous_root_->c_str(), 1);
    else
      ::unsetenv("HOLOHUB_DATA_DIR");
  }

  holoscan::radar::testing::ConfigDirectory directory_;
  std::optional<std::string> previous_root_;
};

TEST_F(SarConfigPaths, RuntimeRootResolvesRelocatedPresetData) {
  const auto input = directory_.write("runtime/sar/relocated.cphd", "fixture");
  const auto root = input.parent_path().parent_path();
  ASSERT_EQ(::setenv("HOLOHUB_DATA_DIR", root.c_str(), 1), 0);
  EXPECT_EQ(resolve_existing("sar/relocated.cphd", root / "missing_yaml_dir"), input);
  // Relative runtime roots are resolved from the caller's working directory.
  const auto relative_root = std::filesystem::relative(root);
  ASSERT_EQ(::setenv("HOLOHUB_DATA_DIR", relative_root.c_str(), 1), 0);
  EXPECT_TRUE(std::filesystem::equivalent(
      resolve_existing("sar/relocated.cphd", root / "missing_yaml_dir"), input));
}

TEST_F(SarConfigPaths, ExistingCallerAndYamlInputsRetainPrecedence) {
  const auto input = directory_.write("caller/sar/explicit.cphd", "caller");
  const auto yaml_input = directory_.write("yaml/sar/explicit.cphd", "yaml");
  const auto runtime_input = directory_.write("runtime/sar/explicit.cphd", "runtime");
  const auto root = runtime_input.parent_path().parent_path();
  ASSERT_EQ(::setenv("HOLOHUB_DATA_DIR", root.c_str(), 1), 0);
  EXPECT_EQ(resolve_existing(input, root), input);
  EXPECT_TRUE(
      std::filesystem::equivalent(resolve_existing(std::filesystem::relative(input), root), input));
  EXPECT_EQ(resolve_existing("sar/explicit.cphd", yaml_input.parent_path().parent_path()),
            yaml_input);
  const auto missing_absolute = root / "missing.cphd";
  EXPECT_EQ(resolve_existing(missing_absolute, root), missing_absolute);
}

TEST_F(SarConfigPaths, EmptyUnsetAndMissingRuntimeRootsRetainCompiledFallback) {
  const auto relative = std::filesystem::path("sar") / "compiled_fallback.cphd";
  const auto compiled = std::filesystem::path(HOLOSCAN_RADAR_DATA_DIR) / relative;
  std::filesystem::create_directories(compiled.parent_path());
  std::ofstream(compiled) << "compiled";
  const auto yaml_dir = directory_.write("yaml/config.yaml", "{}").parent_path();
  const auto runtime = directory_.write("runtime/" + relative.string(), "runtime");
  const auto runtime_root = runtime.parent_path().parent_path();
  ASSERT_EQ(::setenv("HOLOHUB_DATA_DIR", runtime_root.c_str(), 1), 0);
  EXPECT_EQ(resolve_existing(relative, yaml_dir), runtime);
  ASSERT_EQ(::unsetenv("HOLOHUB_DATA_DIR"), 0);
  EXPECT_EQ(resolve_existing(relative, yaml_dir), compiled);
  ASSERT_EQ(::setenv("HOLOHUB_DATA_DIR", "", 1), 0);
  EXPECT_EQ(resolve_existing(relative, yaml_dir), compiled);
  ASSERT_EQ(::setenv("HOLOHUB_DATA_DIR", (yaml_dir / "missing").c_str(), 1), 0);
  EXPECT_EQ(resolve_existing(relative, yaml_dir), compiled);
  std::filesystem::remove(compiled);
  EXPECT_EQ(resolve_existing(relative, yaml_dir), relative);
}

}  // namespace
