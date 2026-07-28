# Holoviz open issues (upstream Holoscan SDK)

Tracks known robustness / behavior gaps in `holoscan::ops::HolovizOp` (and the
underlying `libholoscan_viz`) that affect the FMCW demo apps in this repo but
whose real fix lives in the Holoscan SDK, not here. Each entry records the
symptom, the traced root cause (with SDK source references), what we already do
to mitigate it at the app level, and the proposed upstream fix.

- **SDK version observed:** Holoscan 4.3.0 (`nvcr.io/nvidia/clara-holoscan/holoscan:v4.3.0-cuda13`)
- **Component:** `modules/holoviz` (`libholoscan_viz.so.4.3.0`) + `src/operators/holoviz`
- **Affected apps:** all four demos (`fmcw_siso`, `fmcw_simo`, `fmcw_mimo`,
  `fmcw_dense_simo`), C++ and Python, whenever a live Holoviz window is shown
  (`visualization.enabled: true`).

> SDK source paths below are relative to `/workspace/holoscan-sdk` inside the
> `holoscan-radar-*` container image (the SDK ships its Holoviz sources there).

---

## Issue 1 — Window resize crashes the app: fatal throw on an out-of-date swapchain

### Symptom

Dragging / resizing the live Holoviz window crashes the whole application:

```text
[info]  [framebuffer_sequence.cpp:385] Using present mode 'Immediate'
[error] [gxf_wrapper.cpp:158] Exception occurred for operator: 'holoviz' - Failed to acquire next swap chain image.
[error] [entity_executor.cpp:770] Failed to tick codelet holoviz in entity: holoviz code: GXF_FAILURE
[warning] [greedy_scheduler.cpp:252] Error while executing entity ... named 'holoviz': GXF_FAILURE
[error] [driver.hpp:93] fmcw_mimo: Failed to acquire next swap chain image.
```

Process exits non-zero. Frequency scales with render/tick rate: at a normal
(high) frame rate an active drag crashes almost immediately; at a low data
cadence (e.g. 1 Hz) it only crashes if you happen to be mid-drag exactly when a
frame is rendered (see Issue 2 for why the two are linked).

### Root cause

Every rendered frame acquires the next swapchain image:

- `vulkan_app.cpp:1054` — `Vulkan::Impl::prepare_frame()` calls
  `fb_sequence_->acquire()`.
- `framebuffer_sequence.cpp:473-476` — `FramebufferSequence::acquire()`:

  ```cpp
  if (swap_chain_) {
    if (!swap_chain_->acquire()) {
      throw std::runtime_error("Failed to acquire next swap chain image.");
    }
  }
  ```

`nvvk::SwapChain::acquire()` wraps `vkAcquireNextImageKHR`, which returns
`VK_ERROR_OUT_OF_DATE_KHR` when the window surface no longer matches the
swapchain — exactly what happens during a resize. On that path `acquire()`
returns `false` and Holoviz **throws** instead of recreating the swapchain and
retrying. The exception unwinds out of `HolovizOp::compute()`, is caught at
`gxf_wrapper.cpp:158`, and is reported to the scheduler as `GXF_FAILURE`, which
deactivates the graph and terminates the app.

There *is* a resize handler that rebuilds the swapchain —
`Vulkan::Impl::on_framebuffer_size()` (`vulkan_app.cpp:1407`) does
`device_.waitIdle()` then `fb_sequence_->update(...)`, and it is registered as
GLFW's framebuffer-size callback (`vulkan_app.cpp:758`). That is why you see
`Using present mode 'Immediate'` re-logged on each resize: the swapchain *is*
being recreated. But there is an unavoidable race during a **continuous** drag:

1. The framebuffer-size callback rebuilds the swapchain for size `N`.
2. Before/while the frame is acquired and presented, the ongoing drag moves the
   surface to size `N+1`.
3. `acquire()` (or the subsequent present) sees the now-stale swapchain, gets
   `VK_ERROR_OUT_OF_DATE_KHR`, and throws.

Holoviz never loops "recreate until the acquire succeeds," so a live drag can
always slip a stale surface between rebuild and acquire.

### Present mode makes it worse (but is not the cause)

`HolovizOp`'s `vsync` parameter defaults to **false**, which resolves to the
`IMMEDIATE` Vulkan present mode (`holoviz.cpp:1661` selects FIFO only when
`vsync` is true; otherwise present mode `AUTO` picks MAILBOX → IMMEDIATE →
FIFO). IMMEDIATE has no internal presentation queue and does not gate on
vertical blank, so frames — and therefore swapchain acquires — are issued as
fast as the pipeline allows. More acquires per second means more chances for one
to land on an out-of-date surface during a drag. This only changes the
*probability* of hitting the throw; the throw itself is unconditional.

### App-level mitigation (in this repo)

Commit `ae33d1b` added a `visualization.vsync` toggle, **defaulting to `true`**,
threaded through the shared `VisualizationConfig` (C++ `config.hpp` + Python
`demo_config.py`), `HolovizLayout` / `make_rd_holoviz_op`, all four C++ and
Python apps, and the six YAMLs. With `vsync: true` the apps use a FIFO present
mode (frame-rate capped, internally queued), which issues far fewer acquires per
resize and makes the crash substantially rarer. Set `vsync: false` for uncapped
/ low-latency rendering at the cost of resize robustness (and possible tearing).

This is a probability reduction, **not a fix** — the fatal throw remains
reachable during an active drag.

### Proposed upstream fix

Make `FramebufferSequence::acquire()` handle a lost/out-of-date swapchain the
standard Vulkan way instead of throwing: on `VK_ERROR_OUT_OF_DATE_KHR` (and
ideally `VK_SUBOPTIMAL_KHR`), recreate the swapchain at the current surface
extent and either retry the acquire or skip the frame, returning control to the
render loop rather than raising. This is the conventional resize idiom and would
remove the crash regardless of present mode or tick rate.

---

## Issue 2 — Window responsiveness is coupled to the data/tick rate (sluggish at low frame rates)

### Symptom

When the pipeline's data cadence is low (e.g. `run.frame_period` set for ~1 Hz),
the live window feels broken: dragging is jerky, repaint lags, resize/move
events are serviced in ~1 s steps. The signal-processing pipeline itself is
fine; only the window interaction is sluggish. (This low-rate mode is also what
makes Issue 1 crash only "while resizing during rendering.")

### Root cause

GLFW window events are only pumped from inside a rendered Holoviz frame:

- `holoviz.cpp:187` — `viz::Begin()` → `Context::begin()` at the top of each
  frame.
- `glfw_window.cpp:700-704` — `GLFWWindow::begin()` calls `glfwPollEvents()`.

`viz::Begin()` runs once per `HolovizOp::compute()` call, and `HolovizOp` ticks
once per message from its upstream `receivers`. So `glfwPollEvents()` — which is
what dispatches resize, move, repaint, and input events, including the
framebuffer-size callback from Issue 1 — runs **exactly as often as upstream
produces a frame**.

At ~1 Hz that means the window processes events once per second; in between it is
effectively frozen. This is inherent to HolovizOp polling the window only during
`compute()`; it is not a leak or a stall in the app graph.

### Interaction with Issue 1

Because the resize callback (which rebuilds the swapchain) *and* the
`acquire()`/present both happen inside the same once-per-tick frame, a low tick
rate simply samples the Issue 1 race far less often. High tick rate → responsive
window but frequent crashes on resize; low tick rate → crash-resistant but
sluggish window. The two symptoms are the same coupling viewed from opposite
ends.

### App-level mitigation

None clean. There is no app-level knob to make HolovizOp service the window
independently of upstream cadence. A workaround would be to drive HolovizOp at a
fixed display rate (e.g. 30 Hz) decoupled from the data rate via a periodic
re-render, but that does not fit Holoscan's condition model cleanly (an operator
ticks when *all* its conditions are satisfied, so adding a periodic condition
reduces rather than increases tick rate) and is treated as an unresolved spike,
not a shipped mitigation.

### Proposed upstream improvement

Decouple window event servicing (and optionally presentation) from the operator
tick rate — e.g. pump `glfwPollEvents()` / repaint on an internal display-rate
timer or a dedicated thread — so the window stays responsive even when the data
graph produces frames slowly. Fixing Issue 1 (graceful out-of-date handling)
independently removes the crash regardless of this coupling.

---

## Quick reference

| | Issue 1 (resize crash) | Issue 2 (low-rate sluggishness) |
| --- | --- | --- |
| Trigger | Resizing / dragging the window | Low `run.frame_period` (low data cadence) |
| Mechanism | `acquire()` throws on out-of-date swapchain | Event polling tied to `compute()` tick rate |
| Key SDK refs | `framebuffer_sequence.cpp:474`, `vulkan_app.cpp:1054,1407`, `gxf_wrapper.cpp:158` | `holoviz.cpp:187`, `glfw_window.cpp:703` |
| Our mitigation | `visualization.vsync: true` (FIFO) — reduces, not eliminates | none clean |
| Real fix | Upstream: recreate+retry on `VK_ERROR_OUT_OF_DATE_KHR` | Upstream: decouple event pump from tick rate |
