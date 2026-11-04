// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <filesystem>  // NOLINT(build/c++17)
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "app/config_fixture.hpp"
#include "radar_app/fmcw/config_loader.hpp"

namespace {
namespace config = holoscan::radar::app::config;
using holoscan::radar::testing::ConfigDirectory;

TEST(FmcwConfigLoader, MergesNestedValuesAndReplacesArraysScalarsAndNull) {
  ConfigDirectory files;
  files.write("base.yaml",
              "app: {run: {num_frames: 250, frame_period: 0.1}, targets: [1, 2], range_loss: 3}\n");
  const auto child = files.write(
      "child.yaml",
      "extends: base.yaml\napp: {run: {num_frames: 2}, targets: [4], range_loss: null}\n");
  const auto document = config::load_inherited_document(child);
  EXPECT_EQ(document.get<int>("/app/run/num_frames"), 2);
  EXPECT_DOUBLE_EQ(document.get<double>("/app/run/frame_period"), 0.1);
  EXPECT_EQ(document.get<std::vector<int>>("/app/targets"), std::vector<int>{4});
  EXPECT_FALSE(document.get<std::optional<int>>("/app/range_loss"));
  EXPECT_FALSE(document.find("/extends"));
  // The existing SAR/single-document entry point does not interpret inheritance.
  EXPECT_TRUE(config::load_document(child).find("/extends"));
}

TEST(FmcwConfigLoader, ResolvesChainedPathsRelativeToEachContainingFile) {
  ConfigDirectory files;
  files.write("base.yaml", "value: 1\nuntouched: 4\n");
  files.write("sub/middle.yaml", "extends: ../base.yaml\nvalue: 2\n");
  const auto child = files.write("sub/deep/child.yaml", "extends: ../middle.yaml\nvalue: 3\n");
  ASSERT_NE(std::filesystem::current_path(), child.parent_path());
  const auto document = config::load_inherited_document(child);
  EXPECT_EQ(document.get<int>("/value"), 3);
  EXPECT_EQ(document.get<int>("/untouched"), 4);
}

TEST(FmcwConfigLoader, ResolvesAnchorsWithinFilesBeforeMerging) {
  ConfigDirectory files;
  files.write("base.yaml", "first: &settings {value: 1, retained: 2}\nsecond: *settings\n");
  const auto child = files.write("child.yaml", "extends: base.yaml\nfirst: {value: 3}\n");
  const auto document = config::load_inherited_document(child);
  EXPECT_EQ(document.get<int>("/first/value"), 3);
  EXPECT_EQ(document.get<int>("/first/retained"), 2);
  EXPECT_EQ(document.get<int>("/second/value"), 1);
  const auto invalid = files.write("invalid.yaml", "extends: base.yaml\nthird: *settings\n");
  EXPECT_THROW(config::load_inherited_document(invalid), std::runtime_error);
}

TEST(FmcwConfigLoader, RejectsMalformedDeclarationsAndNonMappingRoots) {
  ConfigDirectory files;
  for (const auto* text : {"extends: ''", "extends: null", "extends: [base.yaml]", "extends: 1",
                           "extends: {}", "[]", "null", "42"}) {
    SCOPED_TRACE(text);
    const auto path = files.write("invalid.yaml", text);
    EXPECT_THROW(config::load_inherited_document(path), std::runtime_error);
  }
}

TEST(FmcwConfigLoader, ReportsMissingFilesAndCyclesWithFileContext) {
  ConfigDirectory files;
  const auto child = files.write("child.yaml", "extends: missing.yaml\n");
  try {
    static_cast<void>(config::load_inherited_document(child));
    FAIL() << "Missing base was accepted";
  } catch (const std::runtime_error& error) {
    EXPECT_NE(std::string(error.what()).find("child.yaml"), std::string::npos);
    EXPECT_NE(std::string(error.what()).find("missing.yaml"), std::string::npos);
  }
  files.write("child.yaml", "extends: other.yaml\n");
  files.write("other.yaml", "extends: child.yaml\n");
  try {
    static_cast<void>(config::load_inherited_document(child));
    FAIL() << "Cycle was accepted";
  } catch (const std::runtime_error& error) {
    EXPECT_NE(std::string(error.what()).find("cycle"), std::string::npos);
    EXPECT_NE(std::string(error.what()).find("child.yaml"), std::string::npos);
    EXPECT_NE(std::string(error.what()).find("other.yaml"), std::string::npos);
  }
}

TEST(FmcwConfigLoader, Accepts32FilesAndRejects33) {
  ConfigDirectory files;
  for (int i = 0; i < 32; ++i) {
    files.write("layer" + std::to_string(i) + ".yaml",
                i == 31 ? "value: 7\n" : "extends: layer" + std::to_string(i + 1) + ".yaml\n");
  }
  const auto first = files.write("entry.yaml", "extends: layer0.yaml\n");
  EXPECT_EQ(config::load_inherited_document(first.parent_path() / "layer0.yaml").get<int>("/value"),
            7);
  EXPECT_THROW(config::load_inherited_document(first), std::runtime_error);
}
}  // namespace
