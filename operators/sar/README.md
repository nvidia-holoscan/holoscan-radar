# SAR formation operators

The HSDK 5 C++/CUDA port implements five SAR library operators. The CPHD reader
and tensor dump are in [operators/io](../io/README.md). Floating-point operators support single and double samples, with separate geometry precision on the manager and backprojector; data ports use plain tensors
with named axes. Fat outputs capture `MaxTensorExtents` when added to the graph.

| Operator | Main data flow | Role |
| --- | --- | --- |
| `SARRangeCompressionOp<T>` | `phase_history` → `range_profiles` | Window, shift, inverse FFT and center the differential-range axis |
| `ApertureManagerOp<T>` | pulse batch + typed boundary → managed batch + `control_out` | Retain, evict, weight, and flush pulses at aperture/collection boundaries |
| `BackprojectionOp<T>` | managed profiles + positions + control → `image` | Accumulate complex images on a grid or sample points, with reset or scroll geometry |
| `PhaseCorrectionOp<T>` | profiles + per-pulse phase → `range_profiles_out` | Apply a declared add/subtract phase convention |
| [`AutofocusOp<T>`](autofocus_op/README.md) | conjugate-grid `image` → `image_out` + pulse-ordered `phase_correction` | Estimate the scene-common phase error by phase gradient autofocus |

The manager joins `range_profiles`, `positions`, `reference_position`, and
`stream_boundary` at the same capture time. Other `[live]` inputs use HSDK 5
`Exact{}` with `SafeDefault{}` per tick. Optional outputs avoid computation
when unwired. The range stage emits `range_window_out` and backprojection emits
`sample_geometry_out` as descriptor outputs with distinct names from their
inputs. `PhaseCorrectionOp` likewise uses `range_profiles_out`.

Public headers/schemas define declarations and defaults; operator READMEs define behavior. [Shared architecture](../../docs/ARCHITECTURE.md) covers cross-cutting rules. The
[SAR application index](../../applications/sar/README.md) shows
five runnable formation modes plus the autofocus workflow. Direct-sampled and
deramped branches and standard-format product writers remain planned, not
implemented in this port.
The copied NVRadar kernels carry the OSRB obligation documented in the [backprojection guide](backprojection_op/README.md#remaining-qualification-and-contract-questions). Synthetic C++ tests do not complete that review or real-data qualification.
