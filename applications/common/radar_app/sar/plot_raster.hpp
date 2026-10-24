// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace holoscan::radar::app::sar {

inline void plot_put_px(std::vector<uint8_t>* img, int64_t w, int64_t h, int64_t x, int64_t y,
                        const std::array<uint8_t, 3>& c) {
  if (x < 0 || x >= w || y < 0 || y >= h)
    return;
  const size_t o = static_cast<std::size_t>((y * w + x) * 4);
  (*img)[o + 0] = c[0];
  (*img)[o + 1] = c[1];
  (*img)[o + 2] = c[2];
  (*img)[o + 3] = 255;
}

constexpr int kFont5x7Cols = 5;
constexpr int kFont5x7Rows = 7;
constexpr int kFont5x7Advance = 6;

inline std::array<uint8_t, 7> plot_glyph_5x7(char c) {
  if (c >= 'a' && c <= 'z')
    c = static_cast<char>('A' + (c - 'a'));
  switch (c) {
    case '0':
      return {14, 17, 17, 17, 17, 17, 14};
    case '1':
      return {4, 12, 4, 4, 4, 4, 14};
    case '2':
      return {14, 17, 1, 2, 4, 8, 31};
    case '3':
      return {14, 17, 1, 6, 1, 17, 14};
    case '4':
      return {2, 6, 10, 18, 31, 2, 2};
    case '5':
      return {31, 16, 30, 1, 1, 17, 14};
    case '6':
      return {14, 17, 16, 30, 17, 17, 14};
    case '7':
      return {31, 1, 2, 4, 4, 4, 4};
    case '8':
      return {14, 17, 17, 14, 17, 17, 14};
    case '9':
      return {14, 17, 17, 15, 1, 17, 14};
    case 'A':
      return {4, 10, 17, 17, 31, 17, 17};
    case 'B':
      return {30, 17, 17, 30, 17, 17, 30};
    case 'C':
      return {14, 17, 16, 16, 16, 17, 14};
    case 'D':
      return {30, 17, 17, 17, 17, 17, 30};
    case 'E':
      return {31, 16, 16, 30, 16, 16, 31};
    case 'F':
      return {31, 16, 16, 30, 16, 16, 16};
    case 'G':
      return {14, 17, 16, 19, 17, 17, 14};
    case 'H':
      return {17, 17, 17, 31, 17, 17, 17};
    case 'I':
      return {14, 4, 4, 4, 4, 4, 14};
    case 'J':
      return {1, 1, 1, 1, 1, 17, 14};
    case 'K':
      return {17, 18, 20, 24, 20, 18, 17};
    case 'L':
      return {16, 16, 16, 16, 16, 16, 31};
    case 'M':
      return {17, 27, 21, 21, 17, 17, 17};
    case 'N':
      return {17, 25, 21, 19, 17, 17, 17};
    case 'O':
      return {14, 17, 17, 17, 17, 17, 14};
    case 'P':
      return {30, 17, 17, 30, 16, 16, 16};
    case 'Q':
      return {14, 17, 17, 17, 21, 18, 13};
    case 'R':
      return {30, 17, 17, 30, 20, 18, 17};
    case 'S':
      return {14, 17, 16, 14, 1, 17, 14};
    case 'T':
      return {31, 4, 4, 4, 4, 4, 4};
    case 'U':
      return {17, 17, 17, 17, 17, 17, 14};
    case 'V':
      return {17, 17, 17, 17, 17, 10, 4};
    case 'W':
      return {17, 17, 17, 21, 21, 21, 10};
    case 'X':
      return {17, 17, 10, 4, 10, 17, 17};
    case 'Y':
      return {17, 17, 10, 4, 4, 4, 4};
    case 'Z':
      return {31, 1, 2, 4, 8, 16, 31};
    case '+':
      return {0, 4, 4, 31, 4, 4, 0};
    case '-':
      return {0, 0, 0, 14, 0, 0, 0};
    case '.':
      return {0, 0, 0, 0, 0, 4, 4};
    case ':':
      return {0, 4, 4, 0, 4, 4, 0};
    case '/':
      return {16, 8, 4, 2, 1, 0, 0};
    case '*':
      return {4, 21, 14, 4, 14, 21, 4};
    case ' ':
      return {0, 0, 0, 0, 0, 0, 0};
    default:
      return {0, 0, 0, 0, 0, 0, 0};
  }
}

inline int64_t plot_text_width_5x7(const std::string& s, int scale) {
  return static_cast<int64_t>(s.size()) * kFont5x7Advance * scale;
}

inline void plot_draw_text_5x7(std::vector<uint8_t>* img, int64_t w, int64_t h, int64_t x,
                               int64_t y, const std::string& s, const std::array<uint8_t, 3>& col,
                               int scale) {
  int64_t cx = x;
  for (char ch : s) {
    const auto rows = plot_glyph_5x7(ch);
    for (int r = 0; r < kFont5x7Rows; ++r) {
      for (int ci = 0; ci < kFont5x7Cols; ++ci) {
        const int bit = kFont5x7Cols - 1 - ci;
        if ((rows[static_cast<size_t>(r)] >> bit) & 1) {
          for (int sy = 0; sy < scale; ++sy) {
            for (int sx = 0; sx < scale; ++sx)
              plot_put_px(img, w, h, cx + ci * scale + sx, y + r * scale + sy, col);
          }
        }
      }
    }
    cx += kFont5x7Advance * scale;
  }
}

}  // namespace holoscan::radar::app::sar
