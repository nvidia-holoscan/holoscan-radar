# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
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

from holoscan.core import Operator, OperatorSpec
from holoscan.operators import HolovizOp

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
    """Live estimator caption for the Holoviz overlay. Ticks once per frame off
    rd_maps and re-emits a single TEXT InputSpec (on input_specs) whose string is
    the selector's current estimator, plus that layer's (1, 3) text-position
    tensor (on label_pos -> receivers). HolovizOp appends dynamic input_specs to
    the compose-time layers and matches the position tensor to the spec by name,
    so this draws a top-left caption that updates as the rotation advances.
    """

    def __init__(
        self,
        fragment,
        *args,
        selector: EstimatorSelector,
        display_names: list[str],
        **kwargs,
    ):
        self._selector = selector
        self._display_names = display_names
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec) -> None:
        spec.input("rd_maps")
        spec.output("input_specs")
        spec.output("label_pos")

    def compute(self, op_input, op_output, context) -> None:
        op_input.receive("rd_maps")  # frame tick only
        idx = self._selector.current
        name = self._display_names[idx] if 0 <= idx < len(self._display_names) else "?"
        caption = name.replace("_", "-").upper()

        spec = HolovizOp.InputSpec("estimator_label", HolovizOp.InputType.TEXT)
        spec.text = [f"Estimator: {caption}"]
        spec.color = [1.0, 0.95, 0.2, 1.0]  # amber, reads over the turbo map
        spec.priority = 10  # draw above the RD image
        # No View: render in full-window [0, 1] space so glyph advance isn't
        # squished by an anisotropic sub-view (which clipped the estimator name).
        op_output.emit([spec], "input_specs")

        # Text (x, y, size) in full-window [0, 1] space; size is the glyph
        # height as a fraction of the window height.
        pos = cp.asarray([[0.012, 0.03, 0.03]], dtype=cp.float32)
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
        requested = self._poll_control_file()
        self._latched = requested if requested >= 0 else -1
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
