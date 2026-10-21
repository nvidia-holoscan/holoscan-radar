// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Measurement-only factory interposer; never preload this into production runs.
// Requires the staging HSDK 5 internal renderer ABI.
#include <cuda.h>

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#if defined(RADAR_VIZ_DEGREE_PREVIEW)
#include <dlfcn.h>
#include <string>
#include <string_view>
#endif

#include <holoviz/device.hpp>
#include <holoviz/text/monospace_text_renderer.hpp>
#include <holoviz/text/text_renderer.hpp>

namespace holoviz::text {
expected<TextRenderer> create_monospace_text_renderer(Device& device,
                                                      const MonospaceTextRendererDesc&) {
  auto context = device.cuda_context();
  if (!context || cuCtxPushCurrent(*context) != CUDA_SUCCESS)
    throw std::runtime_error("UTF-8 prototype: cannot activate renderer context");
  // Measurement-only stream: retained until this process/context exits.
  CUstream stream{};
  if (cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING) != CUDA_SUCCESS)
    throw std::runtime_error("UTF-8 prototype: cannot create upload stream");
  TextRendererDesc desc;
  desc.stream = stream;
  desc.frames_in_flight = 4;
  const bool ttf = std::getenv("UTF8_TTF") != nullptr;
  if (ttf)
    desc.ttf_path = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf";
  desc.base_font_size_px = 32.0F;
  desc.debug_name = "Radar UTF-8 measurement prototype";
  auto result = create_text_renderer(device, desc);
  CUcontext popped{};
  cuCtxPopCurrent(&popped);
  std::fprintf(stderr, "SDK UTF-8 renderer prototype (%s)\n",
               ttf ? "DejaVu Sans Mono" : "default font");
  return result;
}
#if defined(RADAR_VIZ_DEGREE_PREVIEW)
// Visual-only probe: benchmark builds omit this text-rewriting interposer.
expected<void> TextRenderer::add_text(const TextDesc& desc, std::string_view text) {
  using Add = expected<void> (*)(TextRenderer*, const TextDesc&, std::string_view);
  static auto add =
      reinterpret_cast<Add>(dlsym(dlopen("/usr/lib/libholoviz.so.5", RTLD_NOW | RTLD_LOCAL),
                                  "_ZN7holoviz4text12TextRenderer8add_textERKNS0_"
                                  "8TextDescESt17basic_string_viewIcSt11char_traitsIcEE"));
  if (!add)
    throw std::runtime_error("degree preview: cannot resolve SDK add_text");
  if (text.ends_with(" deg")) {
    std::string converted{text.substr(0, text.size() - 4)};
    converted += "\xC2\xB0";
    return add(this, desc, converted);
  }
  return add(this, desc, text);
}
#endif
}  // namespace holoviz::text
