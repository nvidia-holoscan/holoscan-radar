# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# App-specific selection policy for fmcw_aoa_estimators (Python).
#
# Unlike the C++ app, the Python graph does NOT insert a pass-through router or
# a K=1 slice operator on the data plane. Holoscan's Python emit path cannot
# produce an empty (0-detection) device tensor that downstream C++ operators
# accept as CUDA-accessible (cupy reports a null pointer in
# __cuda_array_interface__ for any 0-size array), and bool is_valid tensors are
# not receivable in a Python operator at all. So the snapshot/is_valid data
# plane stays entirely in C++ (SnapshotGatherOp -> estimator), and estimator
# selection is expressed purely through dynamic-flow callbacks that touch no
# tensors. Beamscan is single-snapshot, so it is fed by its own K=1 gather
# rather than a slice of the multi-look gather the covariance/gridless
# estimators share.
#
# The EstimatorSelector below is the shared policy: with no external signal it
# rotates on a fixed dwell (headless, deterministic); a control file/FIFO
# latches a live override, else falls back to the rotation. It is queried by
# two dynamic-flow callbacks (the multi-look gather's and the single-look
# gather's) which must agree on the active estimator each frame regardless of
# the order Holoscan invokes them; the two-phase latch guarantees that.
#
# Imported bare (``from custom_ops import ...``) from fmcw_aoa_estimators.py.

from __future__ import annotations

import logging
import os

import cupy as cp

from holoscan.core import ConditionType, Operator, OperatorSpec
from holoscan.operators import HolovizOp

# Pixel target for the RD-map title, mirroring viz.hpp's kRdTitlePx, and the
# glyph advance ratio its push_abs_text uses for centring (kGlyphAdvanceRatio).
# Holoviz bakes its font atlas at a fixed 25 px and scales those glyphs, so text
# reads crispest at or just under that size.
K_RD_TITLE_PX = 25.0
K_GLYPH_ADVANCE_RATIO = 0.52

logger = logging.getLogger("fmcw_aoa_estimators")


def normalize_name(text: str) -> str:
    return text.replace("-", "").replace("_", "").lower()


class FrameCaptureOp(Operator):
    """Diagnostic: save the Holoviz render_buffer_output framebuffer to PNGs.

    Enabled only when the app is wired for capture (AOA_CAPTURE_DIR set). Writes
    frame_<n>.png into out_dir for the first max_frames frames. Robust to how the
    VideoBuffer arrives in Python (tensor map, bare tensor, or CUDA-array-like)."""

    def __init__(self, fragment, *args, out_dir: str, max_frames: int = 24, **kwargs):
        self._out_dir = out_dir
        self._max = max_frames
        self._n = 0
        self._logged = False
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec) -> None:
        spec.input("render")

    def compute(self, op_input, op_output, context) -> None:
        msg = op_input.receive("render")
        if self._n < self._max:
            arr = self._to_numpy(msg)
            if arr is not None:
                from PIL import Image

                if arr.ndim == 3 and arr.shape[2] == 4:
                    img = Image.fromarray(arr, "RGBA")
                elif arr.ndim == 3 and arr.shape[2] == 3:
                    img = Image.fromarray(arr, "RGB")
                else:
                    img = Image.fromarray(arr)
                img.save(os.path.join(self._out_dir, f"frame_{self._n:03d}.png"))
        self._n += 1

    def _to_numpy(self, msg):
        import numpy as np

        candidate = msg
        # TensorMap-like: take the single tensor.
        try:
            if hasattr(msg, "items"):
                vals = list(msg.values())
                candidate = vals[0] if vals else msg
            elif isinstance(msg, dict):
                candidate = next(iter(msg.values()))
        except (StopIteration, TypeError):
            candidate = msg
        if not self._logged:
            logger.info(
                "FrameCaptureOp: received %s; candidate %s cai=%s",
                type(msg),
                type(candidate),
                hasattr(candidate, "__cuda_array_interface__"),
            )
            self._logged = True
        try:
            arr = cp.asarray(candidate)
            return cp.asnumpy(arr).astype(np.uint8)
        except (ValueError, RuntimeError, TypeError, BufferError):
            pass
        try:
            return np.asarray(candidate).astype(np.uint8)
        except (ValueError, RuntimeError, TypeError):
            logger.warning("FrameCaptureOp: could not convert framebuffer")
            return None


class EstimatorLabelOp(Operator):
    """Live RD-map title for the Holoviz overlay. Ticks once per frame off
    rd_maps and re-emits a single TEXT InputSpec (on input_specs) reading
    ``Range-Doppler - <ESTIMATOR>``, plus that layer's (1, 3) text-position
    tensor (on label_pos -> receivers). HolovizOp appends dynamic input_specs to
    the compose-time layers and matches the position tensor to the spec by name,
    so the plot's own heading names the estimator running this frame. The app
    blanks ClusterOverlayOp.rd_title so this is the only title drawn.

    HolovizOp has a single input_specs port, so this op is the TAIL of the shared
    spec chain: the RD-map chrome (ClusterOverlayOp's scenario clock + axes) and
    the primitive side panels arrive on specs_in, and this op appends its caption
    before emitting the combined list.
    """

    def __init__(
        self,
        fragment,
        *args,
        selector: EstimatorSelector,
        display_names: list[str],
        view_px_height: float = 600.0,
        window_aspect: float = 1.0,
        title_center_x: float = 0.5,
        title_top_y: float = 0.0,
        base_title: str = "Range-Doppler",
        **kwargs,
    ):
        self._selector = selector
        self._display_names = display_names
        # On-screen pixel height of the (full-window) view, for pixel-locked
        # text sizing against Holoviz's fixed font atlas.
        self._view_px_height = view_px_height
        # Window aspect + the title anchor emit_rd_axes would have used: the
        # centre of the inset image box, one title height above its top edge.
        self._window_aspect = window_aspect
        self._title_center_x = title_center_x
        self._title_top_y = title_top_y
        self._base_title = base_title
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec) -> None:
        spec.input("rd_maps")
        # Upstream chrome/panel specs (overlay -> rolling plot -> boresight ->
        # here). Optional so the op still works with no panels in the graph.
        spec.input("specs_in").condition(ConditionType.NONE)
        spec.output("input_specs")
        spec.output("label_pos")

    def compute(self, op_input, op_output, context) -> None:
        op_input.receive("rd_maps")  # frame tick only
        idx = self._selector.current
        name = self._display_names[idx] if 0 <= idx < len(self._display_names) else "?"
        caption = name.replace("_", "-").upper()

        text = f"{self._base_title} - {caption}"
        spec = HolovizOp.InputSpec("estimator_label", HolovizOp.InputType.TEXT)
        spec.text = [text]
        spec.color = [1.0, 1.0, 1.0, 1.0]  # styled like the chrome it replaces
        spec.priority = 8
        # No View: render in full-window [0, 1] space so glyph advance isn't
        # squished by an anisotropic sub-view (which clipped the estimator name).
        # Append to the upstream chrome/panel specs so the whole chain reaches
        # HolovizOp's single input_specs port.
        specs = list(op_input.receive("specs_in") or [])
        specs.append(spec)
        op_output.emit(specs, "input_specs")

        # Text (x, y, size) in full-window [0, 1] space; size is the glyph
        # height as a fraction of the window height, pixel-locked to the title
        # target. Holoviz anchors TEXT at its left edge, so centre it by hand
        # the way viz.hpp's push_abs_text does.
        size = K_RD_TITLE_PX / max(self._view_px_height, 1.0)
        char_w = K_GLYPH_ADVANCE_RATIO * size / max(self._window_aspect, 1e-3)
        x = self._title_center_x - 0.5 * char_w * len(text)
        pos = cp.asarray([[x, self._title_top_y, size]], dtype=cp.float32)
        op_output.emit({"estimator_label": pos}, "label_pos")


class EstimatorSelector:
    """Frame-synchronized estimator selection policy shared by the two gather
    dynamic-flow callbacks.

    Selection sources, in resolution order:
      hold   - a fixed estimator (hold_index), ignores external input.
      rotate - fixed dwell: index = (frame // rotate_period) % N.
      file   - poll control_path each frame; a recognized name latches (sticky)
               until changed; an absent/empty/unrecognized file falls back to
               the rotation.
    """

    def __init__(
        self,
        *,
        mode: str,
        rotate_period: int,
        hold_index: int,
        control_path: str,
        log_every: int,
        display_names: list[str],
    ):
        self.mode = mode
        self.rotate_period = rotate_period if rotate_period > 0 else 1
        self.hold_index = hold_index
        self.control_path = control_path
        self.log_every = log_every
        self.display_names = display_names
        self.num_participants = 0  # callbacks registered per frame
        self._frame = 0
        self._current = 0
        self._latched = -1
        self._seen = 0
        self._last_logged = -1

    @property
    def current(self) -> int:
        """The most recently computed selection, without advancing the frame
        latch. Read by the viz caption (a one-frame lag is immaterial there)."""
        return self._current

    def select(self) -> int:
        """Return this frame's active estimator index. Advances the frame once
        all registered callbacks have queried it (order-independent)."""
        if self._seen == 0:
            self._current = self._compute()
            self._log()
        self._seen += 1
        if self._seen >= max(1, self.num_participants):
            self._frame += 1
            self._seen = 0
        return self._current

    def _compute(self) -> int:
        n = len(self.display_names)
        rotated = (self._frame // self.rotate_period) % n
        if self.mode == "hold":
            return max(0, min(self.hold_index, n - 1))
        if self.mode == "rotate":
            return rotated
        # A recognized value sticks until the next recognized value; an
        # absent/empty/unrecognized poll leaves the prior latch in place (so a
        # one-shot ``echo music > path`` persists). Rotation is the fallback
        # only until the first override latches.
        requested = self._poll_control_file()
        if requested >= 0:
            self._latched = requested
        return self._latched if self._latched >= 0 else rotated

    def _log(self) -> None:
        changed = self._current != self._last_logged
        periodic = self.log_every > 0 and (self._frame % self.log_every == 0)
        if changed or periodic:
            logger.info(
                "EstimatorSelector: frame %d -> %s estimator",
                self._frame,
                self.display_names[self._current],
            )
            self._last_logged = self._current

    def _poll_control_file(self) -> int:
        if not self.control_path or not os.path.isfile(self.control_path):
            return -1
        try:
            with open(self.control_path, encoding="utf-8") as handle:
                token = handle.read().split()
        except OSError:
            return -1
        if not token:
            return -1
        norm = normalize_name(token[0])
        for i, name in enumerate(self.display_names):
            if norm == normalize_name(name):
                return i
        if len(norm) == 1 and "1" <= norm <= "9":
            idx = int(norm) - 1
            if idx < len(self.display_names):
                return idx
        return -1
