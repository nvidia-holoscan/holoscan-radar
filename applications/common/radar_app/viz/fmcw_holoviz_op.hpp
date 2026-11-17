// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "radar_app/viz/frame_composer_op.hpp"
#include "radar_app/viz/native_image_presenter_op.hpp"

namespace holoscan::radar::app::viz {
using FmcwFrameComposerOp = FrameComposerOp;
using FmcwHolovizOp = NativeImagePresenterOp;
}  // namespace holoscan::radar::app::viz
