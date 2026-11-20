// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <unistd.h>

#include <chrono>
#include <cstdint>
#include <ctime>
#include <exception>
#include <filesystem>  // NOLINT(build/c++17)
#include <fstream>
#include <iomanip>
#include <istream>
#include <optional>
#include <ostream>
#include <stdexcept>
#include <string>
#include <system_error>

#include <holoscan/config/io.hpp>
#include <holoscan/config/codec.hpp>

namespace holoscan::radar::app::detail {

inline constexpr std::int64_t kPresentationWeekSeconds = 7 * 24 * 60 * 60;

inline std::int64_t presentation_now_seconds() {
  return std::chrono::duration_cast<std::chrono::seconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

struct PresentationChoice {
  bool window;
  bool x11_available;
  bool interactive;
  std::optional<bool> configured;
  std::filesystem::path cache_file;
  std::string setup_key;
  bool choose_again{false};
  std::int64_t now_seconds{presentation_now_seconds()};
  std::string config_path;
};

struct RememberedPresentation {
  bool x11;
  std::optional<std::int64_t> expires_at;
};

inline std::optional<RememberedPresentation> read_presentation(const PresentationChoice& choice) {
  if (choice.cache_file.empty())
    return std::nullopt;
  auto document =
      holoscan::config_io::try_load(choice.cache_file, holoscan::config_io::Format::kJson,
                                    {.max_source_bytes = 4096U, .max_depth = 4U, .max_nodes = 16U});
  if (!document)
    return std::nullopt;
  try {
    if (!document->find("/expires_at"))
      return std::nullopt;
    const auto presenter = document->get<std::string>("/presenter");
    const auto saved_at = document->get<std::int64_t>("/saved_at");
    const auto expires_at = document->get<std::optional<std::int64_t>>("/expires_at");
    if (document->get<int>("/version") != 2 ||
        document->get<std::string>("/setup_key") != choice.setup_key ||
        (presenter != "x11" && presenter != "native") || saved_at <= 0 ||
        choice.now_seconds < saved_at ||
        (expires_at && (*expires_at <= saved_at || *expires_at <= choice.now_seconds ||
                        *expires_at - saved_at != kPresentationWeekSeconds)))
      return std::nullopt;
    return RememberedPresentation{.x11 = presenter == "x11", .expires_at = expires_at};
  } catch (const std::exception&) {
    // Missing fields, wrong types, and future/legacy formats are cache misses.
    return std::nullopt;
  }
}

// Replace atomically so interrupted writes cannot become valid preferences.
// Timestamps are UTC Unix seconds; null expires_at means until changed.
inline void remember_presentation(const PresentationChoice& choice, bool x11, bool forever,
                                  std::ostream& output) {
  const auto& path = choice.cache_file;
  if (path.empty()) {
    output << "Cannot remember this choice: no cache directory is available.\n";
    return;
  }
  std::error_code error;
  std::filesystem::create_directories(path.parent_path(), error);
  auto temporary = path;
  temporary += "." + std::to_string(::getpid()) + ".tmp";
  if (!error) {
    std::ofstream file(temporary);
    // setup_key is generated as decimal digits by the startup resolver.
    file << "{\"version\":2,\"presenter\":\"" << (x11 ? "x11" : "native") << "\",\"setup_key\":\""
         << choice.setup_key << "\",\"saved_at\":" << choice.now_seconds << ",\"expires_at\":"
         << (forever ? "null" : std::to_string(choice.now_seconds + kPresentationWeekSeconds))
         << "}\n";
    file.close();
    if (file) {
      std::filesystem::rename(temporary, path, error);
      if (!error)
        return;
    }
  }
  std::filesystem::remove(temporary, error);
  output << "Could not remember this choice; continuing with the selected presenter.\n";
}

inline std::string presentation_answer(std::string answer) {
  const auto begin = answer.find_first_not_of(" \t\r");
  return begin == std::string::npos
             ? ""
             : answer.substr(begin, answer.find_last_not_of(" \t\r") - begin + 1U);
}

inline bool choose_x11_presentation(const PresentationChoice& choice, std::istream& input,
                                    std::ostream& output) {
  const auto config_location =
      choice.config_path.empty() ? "your YAML" : "\"" + choice.config_path + "\"";
  if (choice.configured || !choice.window || !choice.x11_available) {
    if (choice.configured && choice.choose_again)
      output << "--choose-presenter: explicit app.visualization.x11_presentation takes priority; "
                "omit it in "
             << config_location << " to choose interactively.\n";
    return choice.configured.value_or(true);
  }

  if (choice.choose_again) {
    if (!choice.interactive)
      throw std::runtime_error(
          "--choose-presenter requires an interactive terminal; set "
          "app.visualization.x11_presentation explicitly for unattended runs");
    std::error_code error;
    if (!choice.cache_file.empty())
      std::filesystem::remove(choice.cache_file, error);
    if (error)
      throw std::runtime_error("Cannot clear the remembered presenter: " + error.message());
  } else if (const auto saved = read_presentation(choice)) {
    output << "Using remembered " << (saved->x11 ? "X11" : "native") << " presentation (";
    if (saved->expires_at) {
      const std::time_t expiry = *saved->expires_at;
      std::tm utc{};
      ::gmtime_r(&expiry, &utc);
      output << "expires " << std::put_time(&utc, "%Y-%m-%d %H:%M UTC");
    } else {
      output << "until changed";
    }
    output << "). To change it, rerun with --choose-presenter.\n";
    return saved->x11;
  }

  if (!choice.interactive) {
    output << "No interactive terminal; defaulting to X11 presentation. "
              "Set app.visualization.x11_presentation to true or false in "
           << config_location << " to select explicitly.\n";
    return true;
  }

  output << "\nChoose how to display the radar application:\n"
            "  x: X11 image transport (recommended for SSH/X11 forwarding)\n"
            "  n: Native HoloViz window (for a local desktop with Vulkan presentation)\n"
            "To bypass this question and any remembered choice, set\n"
            "app.visualization.x11_presentation: true (X11) or false (native) in "
         << config_location << ".\n";
  bool x11 = true;
  for (;;) {
    output << "Presenter [x/n, Enter = x]: " << std::flush;
    std::string answer;
    if (!std::getline(input, answer)) {
      output << "\nNo answer available; using X11 for this run.\n";
      return true;
    }
    answer = presentation_answer(answer);
    if (answer.empty() || answer == "x" || answer == "X" || answer == "x11")
      break;
    if (answer == "n" || answer == "N" || answer == "native") {
      x11 = false;
      break;
    }
    output << "Please choose x for X11 or n for native HoloViz.\n";
  }

  output << "Remember this choice?\n"
            "  1: This run only (default)\n"
            "  2: Remember for 7 days\n"
            "  3: Remember until changed\n";
  for (;;) {
    output << "Remember [1/2/3, Enter = 1]: " << std::flush;
    std::string answer;
    if (!std::getline(input, answer)) {
      output << "\nUsing the selected presenter for this run only.\n";
      return x11;
    }
    answer = presentation_answer(answer);
    if (answer.empty() || answer == "1")
      return x11;
    if (answer == "2" || answer == "3") {
      remember_presentation(choice, x11, answer == "3", output);
      return x11;
    }
    output << "Please choose 1, 2, or 3.\n";
  }
}

}  // namespace holoscan::radar::app::detail
