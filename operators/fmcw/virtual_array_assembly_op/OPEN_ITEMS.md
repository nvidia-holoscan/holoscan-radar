# VirtualArrayAssemblyOp — Open Items

Deferred work and **team-visible imposed decisions** for this operator.
Remove items when they land (note the commit in the PR).

---

## A. Imposed decisions (contract was underspecified)

### A.1 Virtual position formula

**Pinned:** `p_virtual = p_tx + p_rx` (vector sum in the array-local frame, meters).
`tx_positions` / `rx_positions` are static parameters, row-major `(num, 3)`.
Lengths must match input `M_tx` / `N_rx` trailing axes.

### A.2 Channel flatten order

**Underspecified:** index formula for `(tx, rx)` → virtual channel.

**Pinned:** linear index `c = tx · N_rx + rx` (TX outer, RX inner). Input RD
layout `(…, M_tx, N_rx, num_doppler, num_range)` uses the same `(tx, rx)`
ordering before flatten.

### A.3 Overlap policies

| Policy | Behavior |
| --- | --- |
| `dedup` (default) | Coincident `p_virtual` (see §A.4) collapse to one virtual channel; **first** `(tx, rx)` slot in scan order is the representative. |
| `average` | Coincident positions are complex-averaged across member channels. |
| `keep_all` | No collapse; `num_virtual = M_tx · N_rx`. |

`num_virtual ≤ M·N`; equality holds for `keep_all` or when no positions collide.

### A.4 Coincidence tolerance

Positions equal when all three components agree within `1e-9` m (`kPositionTol`).
Not scaled by wavelength. Near-duplicates closer than this are merged; farther
apart are distinct virtual elements.

### A.5 Emitted `array_geometry`

Runtime output on port `array_geometry` (descriptor exception — not a tensor).
Contains `element_positions` for the **post-policy** virtual set and `wavelength`
from the effective `[live]` `wavelength` parameter. `geometry_type` is left
unset (informational field only). Downstream angle ops should consume this port
as the `[live]` override when wiring MIMO graphs.

### A.6 Doppler / range axes unchanged

Assembly is a channel gather/average only; `num_doppler` and `num_range` pass
through unchanged. Expects compensated, separated `(tx, rx)` input from
`MIMOCoherentCompensationOp` (DDM may have reduced `num_doppler` per TX).

---

## B. Deferred follow-on work

### B.1 Sub-wavelength overlap rules

Current dedup is exact coordinate match within `1e-9` m. A future option might
quantize to λ/2 grids or apply wavelength-relative tolerance for synthetic
arrays with numeric noise.

### B.2 Per-tick output allocation (module-wide)

Allocate-per-tick via `make_device_tensor`; xl profiling does not show the same
malloc-dominated profile as `MIMODeinterleaveOp`. See
[`operators/OPEN_ITEMS.md`](../../OPEN_ITEMS.md) §1.

### B.3 Graph mid-stream sticky `[live] wavelength`

**Blocks checklist:** Tests → “[live] overrides: mid-stream + sticky”.

**Class:** HSDK 5 EA limitation. See
[`HSDK5_PORT.md`](../../../docs/planning/HSDK5_PORT.md) §1.

**Interim.** Operator still declares sticky `Input<double> wavelength` and
resolves via `resolve_sticky_override`. The port is omitted from `contract()`
so unwired graphs compile. Graph sticky unit tests remain `GTEST_SKIP`’d.
`perf_virtual_array_assembly` measures an inline `wavelength` transition with
the supported single-clock / exact-contract / single-partition shape.

**Status:** open for the sticky unit test; perf override timing is restored.
