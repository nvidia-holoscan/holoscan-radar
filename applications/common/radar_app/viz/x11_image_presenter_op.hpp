// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <X11/Xlib.h>
#include <X11/Xatom.h>
#include <X11/Xutil.h>
#include <X11/extensions/Xrender.h>
#include <X11/keysym.h>
#include <dlpack/dlpack.h>

#include <algorithm>
#include <chrono>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include "radar_app/app/gui_lifecycle.hpp"
#include "radar_app/viz/x11_docked_window.hpp"

#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/memory_kind.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/core/tensor_contract.hpp>
#include <holoscan/sensor_io/schemas/image_schema_traits.hpp>

namespace holoscan::radar::app::viz {

using X11PropertyWord = unsigned long;  // NOLINT(runtime/int): Xlib format-32 property ABI

// X11 transports an already-composed GPU frame. It does not draw radar plots or
// text, and has no Vulkan surface or presentation-GPU requirement. XRender scales
// the complete canvas on the display server, including over an SSH connection.
class X11ImagePresenterOp final : public Operator<> {
 public:
  X11ImagePresenterOp(std::int32_t width, std::int32_t height, std::string title, bool fit = false,
                      std::shared_ptr<FrameProgress> progress = authoring_frame_progress(),
                      std::string position = {}, std::shared_ptr<X11DockedWindow> dock = {},
                      bool footer = false)
  : width_(width),
    height_(height),
    title_(std::move(title)),
    fit_(fit),
    progress_(std::move(progress)),
    position_(std::move(position)),
    dock_(std::move(dock)),
    footer_(footer) {
    if (width_ <= 0 || height_ <= 0)
      throw std::invalid_argument("X11 image dimensions must be positive");
  }

  void setup(OperatorSpec& spec) override {
    spec.input(image)
        .tensor_field(schema::ImageTensorFields::data)
        .expects_tensor(TensorInputSpec{
            .representation = {.memory_kind = MemoryKind::kHost,
                               .dtype = DLDataType{kDLUInt, 8U, 1U},
                               .rank = 3U,
                               .strides = TensorStridePolicy::kContiguousOnly},
            .bounds =
                TensorBounds{.max_rank = 3U,
                             .max_byte_span = static_cast<std::uint64_t>(width_) * height_ * 4U}});
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = std::chrono::milliseconds{16}}).input(image, DrainAll{});
    return result;
  }

  void start() override {
    state_ = std::make_shared<State>();
    auto& state = *state_;
    state.display = XOpenDisplay(nullptr);
    if (!state.display)
      throw std::runtime_error("X11 image presentation: cannot open DISPLAY");
    const int screen = DefaultScreen(state.display);
    auto* visual = DefaultVisual(state.display, screen);
    const int depth = DefaultDepth(state.display, screen);
    int event_base = 0;
    int error_base = 0;
    auto* format = XRenderFindVisualFormat(state.display, visual);
    if (!XRenderQueryExtension(state.display, &event_base, &error_base) || !format ||
        visual->c_class != TrueColor)
      throw std::runtime_error("X11 image presentation requires TrueColor and XRender");

    const int y = position_ == "bottom_left"
                      ? std::max(0, DisplayHeight(state.display, screen) - height_ - 48)
                      : 0;
    state.window = XCreateSimpleWindow(state.display,
                                       dock_ ? dock_->parent() : RootWindow(state.display, screen),
                                       0, y, width_, height_, 0, 0, 0);
    XStoreName(state.display, state.window, title_.c_str());
    if (dock_)
      state.counters = XInternAtom(state.display, "RADAR_PRESENTATION_COUNTERS", False);
    state.close = XInternAtom(state.display, "WM_DELETE_WINDOW", False);
    XSetWMProtocols(state.display, state.window, &state.close, 1);
    XSelectInput(state.display, state.window, ExposureMask | StructureNotifyMask | KeyPressMask);
    state.pixmap = XCreatePixmap(state.display, state.window, width_, height_, depth);
    state.gc = XCreateGC(state.display, state.pixmap, 0, nullptr);
    state.source = XRenderCreatePicture(state.display, state.pixmap, format, 0, nullptr);
    state.target = XRenderCreatePicture(state.display, state.window, format, 0, nullptr);
    XRenderSetPictureFilter(state.display, state.source, FilterBilinear, nullptr, 0);
    state.pixels =
        XCreateImage(state.display, visual, depth, ZPixmap, 0, nullptr, width_, height_, 32, 0);
    if (!state.pixels)
      throw std::runtime_error("X11 image presentation: cannot allocate image descriptor");
    state.pixels->data = static_cast<char*>(
        std::calloc(static_cast<std::size_t>(state.pixels->bytes_per_line), height_));
    if (!state.pixels->data)
      throw std::runtime_error("X11 image presentation: cannot allocate image pixels");
    state.width = width_;
    state.height = height_;
    XMapWindow(state.display, state.window);
    XSync(state.display, False);
    if (dock_) {
      state.dock = dock_;
      state.footer = footer_;
      dock_->attach(footer_, state.window);
    }
  }

  void stop() override {
    state_.reset();
    have_frame_ = false;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    auto& state = *state_;
    if (dock_ && dock_->poll()) {
      context.request_stop();
      return {};
    }
    bool redraw = false;
    while (XPending(state.display)) {
      XEvent event;
      XNextEvent(state.display, &event);
      if ((event.type == ClientMessage &&
           static_cast<Atom>(event.xclient.data.l[0]) == state.close) ||
          (event.type == KeyPress && XLookupKeysym(&event.xkey, 0) == XK_Escape)) {
        context.request_stop();
        return {};
      }
      if (event.type == Expose)
        redraw = true;
      if (event.type == ConfigureNotify) {
        redraw = true;
        state.width = event.xconfigure.width;
        state.height = event.xconfigure.height;
      }
    }

    HOLOSCAN_TRY(auto frames, image.selected_batch());
    if (!frames.empty()) {
      const auto& sample = frames.back();
      const auto& frame = sample.data;
      if (frame.width != width_ || frame.height != height_ || !frame.data || !frame.data->data() ||
          frame.encoding != schema::ImageEncoding_RGBA8 || frame.data->ndim() != 3 ||
          frame.data->shape()[0] != height_ || frame.data->shape()[1] != width_ ||
          frame.data->shape()[2] != 4)
        return make_unexpected(Error{ErrorCode::kInvalidArgument, "X11 RGBA frame shape mismatch"});

      // Pack for the server's visual, including its channel masks and byte order.
      // XPutPixel is presentation format conversion, not a signal-processing path.
      const auto* rgba = static_cast<const std::uint8_t*>(frame.data->data());
      const auto channel = [](std::uint8_t value, std::uint64_t mask) {
        if (mask == 0)
          return std::uint64_t{0};
        unsigned int shift = 0;
        while ((mask & 1UL) == 0) {
          mask >>= 1U;
          ++shift;
        }
        return ((static_cast<std::uint64_t>(value) * mask + 127UL) / 255UL) << shift;
      };
      std::array<std::uint64_t, 256> red{};
      std::array<std::uint64_t, 256> green{};
      std::array<std::uint64_t, 256> blue{};
      for (std::size_t i = 0; i < red.size(); ++i) {
        red[i] = channel(static_cast<std::uint8_t>(i), state.pixels->red_mask);
        green[i] = channel(static_cast<std::uint8_t>(i), state.pixels->green_mask);
        blue[i] = channel(static_cast<std::uint8_t>(i), state.pixels->blue_mask);
      }
      const auto opaque =
          ~(state.pixels->red_mask | state.pixels->green_mask | state.pixels->blue_mask);
      for (std::int32_t y = 0; y < height_; ++y) {
        for (std::int32_t x = 0; x < width_; ++x) {
          const auto* pixel = rgba + (static_cast<std::size_t>(y) * width_ + x) * 4U;
          const std::uint64_t packed = red[pixel[0]] | green[pixel[1]] | blue[pixel[2]] | opaque;
          XPutPixel(state.pixels, x, y, packed);
        }
      }
      XPutImage(state.display, state.pixmap, state.gc, state.pixels, 0, 0, 0, 0, width_, height_);
      ++state.uploads;
      if (dock_)
        dock_->uploaded(footer_, static_cast<std::uint64_t>(width_) * height_ * 4U);
      redraw = true;
      have_frame_ = true;
      if (progress_)
        progress_->capture_ns.store(sample.metadata.capture_time().timestamp_ns);
    }
    if (redraw && have_frame_ && state.width > 0 && state.height > 0) {
      int target_width = state.width;
      int target_height = state.height;
      if (fit_) {
        const double scale = std::min(static_cast<double>(state.width) / width_,
                                      static_cast<double>(state.height) / height_);
        target_width = std::max(1, static_cast<int>(width_ * scale));
        target_height = std::max(1, static_cast<int>(height_ * scale));
        XSetForeground(state.display, state.gc, 0);
        XFillRectangle(state.display, state.window, state.gc, 0, 0, state.width, state.height);
      }
      XTransform transform{{{XDoubleToFixed(static_cast<double>(width_) / target_width), 0, 0},
                            {0, XDoubleToFixed(static_cast<double>(height_) / target_height), 0},
                            {0, 0, XDoubleToFixed(1.0)}}};
      XRenderSetPictureTransform(state.display, state.source, &transform);
      XRenderComposite(state.display, PictOpSrc, state.source, 0, state.target, 0, 0, 0, 0,
                       (state.width - target_width) / 2, (state.height - target_height) / 2,
                       target_width, target_height);
      ++state.repaints;
      if (dock_) {
        // Xlib format-32 properties use native unsigned long array elements.
        const std::array<X11PropertyWord, 2> counters{state.uploads, state.repaints};
        XChangeProperty(state.display, state.window, state.counters, XA_CARDINAL, 32,
                        PropModeReplace, reinterpret_cast<const unsigned char*>(counters.data()),
                        counters.size());
      }
      XFlush(state.display);
    }
    return {};
  }

  Input<schema::ImageT> image{"image"};

 private:
  struct State {
    Display* display{};
    Window window{};
    Pixmap pixmap{};
    GC gc{};
    Picture source{};
    Picture target{};
    XImage* pixels{};
    Atom close{}, counters{};
    X11PropertyWord uploads{0}, repaints{0};
    std::shared_ptr<X11DockedWindow> dock;
    bool footer{false};
    int width{};
    int height{};

    ~State() {
      if (pixels)
        XDestroyImage(pixels);
      if (!display)
        return;
      if (source)
        XRenderFreePicture(display, source);
      if (target)
        XRenderFreePicture(display, target);
      if (gc)
        XFreeGC(display, gc);
      if (pixmap)
        XFreePixmap(display, pixmap);
      if (window) {
        if (dock)
          dock->release(footer, display, window);
        else
          XDestroyWindow(display, window);
      }
      XCloseDisplay(display);
    }
  };

  std::int32_t width_;
  std::int32_t height_;
  std::string title_;
  bool fit_;
  std::shared_ptr<FrameProgress> progress_;
  bool have_frame_{false};
  std::string position_;
  std::shared_ptr<X11DockedWindow> dock_;
  bool footer_;
  std::shared_ptr<State> state_;
};

}  // namespace holoscan::radar::app::viz
