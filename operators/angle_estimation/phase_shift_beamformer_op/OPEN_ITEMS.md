# PhaseShiftBeamformerOp open items

Follow-up for `PhaseShiftBeamformerOp` that blocks full
[`NEW_OPERATOR_CHECKLIST.md`](../../../docs/planning/NEW_OPERATOR_CHECKLIST.md)
completion. Cross-cut SDK / migration detail lives in
[`docs/planning/HSDK5_PORT.md`](../../../docs/planning/HSDK5_PORT.md).

When an item closes, summarize in [`STATUS.md`](../../../docs/planning/STATUS.md)
if it affects the module, then remove or archive the section here.

---

## 1. Inline live-override performance timing

The C++ Graph tests cover all four live payload families and their one-tick
revert behavior. The performance harness still reports only cold and steady
Params-driven runs; it does not co-emit a host-Tensor or wire-schema override
mid-session and populate `override_ms` / `override_delta_ms`.

**Status:** open — ordinary performance-harness work, no SDK blocker.
