// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <array>
#include <cstdlib>
#include <cstring>
#include <string>
#include <tuple>

#include "applications/common/radar_app/sar/window_place.hpp"

namespace {
namespace xp = x11_place_detail;

// Model Xlib's native-long storage for format 32 and its smaller buffers for
// formats 8/16. Rejected buffers intentionally have no long-sized padding.
class WindowPropertyTest : public ::testing::Test {
 protected:
  void SetUp() override {
    current_ = this;
    api_.XInternAtom = [](xp::XDisplay*, const char*, int) -> xp::XAtom {
      return current_->missing_ ? 0 : 100;
    };
    api_.XGetWindowProperty = [](xp::XDisplay*, xp::XWindow win, xp::XAtom, xp::X11Long,
                                 xp::X11Long, int, xp::XAtom requested, xp::XAtom* actual,
                                 int* format, xp::X11ULong* count, xp::X11ULong* after,
                                 unsigned char** data) {
      auto& state = *current_;
      if (requested != xp::kXaCardinal && requested != xp::kXaWindow) {
        ++state.title_reads_;
        state.last_window_ = win;
        *actual = requested;
        *format = 8;
        *count = 6;
        *after = 0;
        *data = static_cast<unsigned char*>(std::malloc(7));
        std::memcpy(*data, win == state.match_window_ ? "target" : "other!", 7);
        return xp::kXSuccess;
      }
      *actual = state.wrong_type_ ? 999 : requested;
      *format = state.format_;
      *count = state.count_;
      *after = state.after_;
      *data = nullptr;
      if (!state.null_data_) {
        const auto bytes = state.format_ == 32 ? sizeof(xp::X11ULong) : state.format_ / 8;
        *data = static_cast<unsigned char*>(std::calloc(state.count_ + 1, bytes));
        if (state.format_ == 32 && state.count_ >= 4) {
          std::memcpy(*data, state.values_.data(), sizeof(state.values_));
        }
      }
      return state.result_;
    };
    api_.XFree = [](void* data) {
      ++current_->frees_;
      std::free(data);
      return 0;
    };
  }
  inline static WindowPropertyTest* current_ = nullptr;
  xp::X11Api api_{};
  std::array<xp::X11ULong, 4> values_{12, 34, 1920, 1080};
  int format_ = 32;
  xp::X11ULong count_ = 4;
  xp::X11ULong after_ = 0;
  int result_ = xp::kXSuccess;
  bool wrong_type_ = false;
  bool missing_ = false;
  bool null_data_ = false;
  int frees_ = 0;
  int title_reads_ = 0;
  xp::XWindow last_window_ = 0;
  xp::XWindow match_window_ = 0;
};

using PropertyCases = ::testing::WithParamInterface<int>;
class InvalidWindowPropertyTest : public WindowPropertyTest, public PropertyCases {};

TEST_P(InvalidWindowPropertyTest, RejectsMalformedPropertiesAndFreesBuffers) {
  switch (GetParam()) {
    case 0:
      format_ = 8;
      break;
    case 1:
      format_ = 16;
      break;
    case 2:
      wrong_type_ = true;
      break;
    case 3:
      count_ = 0;
      break;
    case 4:
      result_ = 1;
      break;
    case 5:
      null_data_ = true;
      break;
    case 6:
      missing_ = true;
      break;
    default:
      FAIL() << "Unknown property case";
  }
  int x = 9;
  int y = 8;
  int width = 640;
  int height = 480;
  EXPECT_FALSE(xp::x11_work_area(api_, nullptr, 1, &x, &y, &width, &height));
  EXPECT_EQ(std::make_tuple(x, y, width, height), std::make_tuple(9, 8, 640, 480));
  int top = -1;
  int bottom = -1;
  xp::x11_frame_top_bottom(api_, nullptr, 1, &top, &bottom);
  EXPECT_EQ(top, 0);
  EXPECT_EQ(bottom, 0);
  EXPECT_EQ(xp::x11_find_window(api_, nullptr, 1, "target"), 0U);
  EXPECT_EQ(title_reads_, 0);
  EXPECT_EQ(frees_, null_data_ || missing_ ? 0 : 3);
}
INSTANTIATE_TEST_SUITE_P(Metadata, InvalidWindowPropertyTest, ::testing::Range(0, 7));

TEST_F(WindowPropertyTest, RejectsShortGeometryAndOversizedClientList) {
  count_ = 3;
  int x = 0;
  int y = 0;
  int width = 640;
  int height = 480;
  EXPECT_FALSE(xp::x11_work_area(api_, nullptr, 1, &x, &y, &width, &height));
  int top = -1;
  int bottom = -1;
  xp::x11_frame_top_bottom(api_, nullptr, 1, &top, &bottom);
  EXPECT_EQ(top, 0);
  EXPECT_EQ(bottom, 0);
  count_ = 1025;
  EXPECT_EQ(xp::x11_find_window(api_, nullptr, 1, "target"), 0U);
  EXPECT_EQ(title_reads_, 0);
  EXPECT_EQ(frees_, 3);
}

TEST_F(WindowPropertyTest, ReadsNativeLongGeometryAndFirstDesktopOfMultipleDesktops) {
  after_ = 16;  // More desktop rectangles may follow the four requested items.
  int x = 0;
  int y = 0;
  int width = 0;
  int height = 0;
  ASSERT_TRUE(xp::x11_work_area(api_, nullptr, 1, &x, &y, &width, &height));
  EXPECT_EQ(std::make_tuple(x, y, width, height), std::make_tuple(12, 34, 1920, 1080));
  int top = 0;
  int bottom = 0;
  xp::x11_frame_top_bottom(api_, nullptr, 1, &top, &bottom);
  EXPECT_EQ(top, 1920);
  EXPECT_EQ(bottom, 1080);
  EXPECT_EQ(frees_, 2);
}

TEST_F(WindowPropertyTest, FindsNativeLongWindowIdAndFreesListAndTitle) {
  // Match the second native-long slot: packed 32-bit iteration has the wrong
  // stride on LP64. The high bit also catches accidental signed-ID narrowing.
  values_[1] = match_window_ = 0xf1234567UL;
  EXPECT_EQ(xp::x11_find_window(api_, nullptr, 1, "target"), values_[1]);
  EXPECT_EQ(last_window_, values_[1]);
  EXPECT_EQ(title_reads_, 2);
  EXPECT_EQ(frees_, 3);
}
}  // namespace
