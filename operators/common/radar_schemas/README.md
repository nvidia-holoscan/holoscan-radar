# Radar Schema Payloads

This directory owns the committed FlatBuffers sources for shared HSDK 5 wire
payloads. Generated headers and traits are build outputs and must not be
committed.

Schemas here provide physical transport for existing logical radar
descriptors; they do not create additional shared descriptor types. One
schema target owns each canonical payload identity, and all operator
consumers use that target rather than copying `.fbs` files.

The generated schema roots and their logical descriptors are:

| Logical descriptor | Physical wire payload | Schema | File identifier |
| --- | --- | --- | --- |
| `ArrayGeometry` | `WireArrayGeometry` | [array_geometry.fbs](array_geometry.fbs) | `RAGY` |
| `WindowSpec` | `WireWindowSpec` | [window_spec.fbs](window_spec.fbs) | `RWSP` |
| `FMCWWaveformDescriptor` | `wire::FMCWWaveformT` | [fmcw_waveform.fbs](fmcw_waveform.fbs) | `RWFV` |
| `CoordinateFrame` | `WireCoordinateFrame` | [coordinate_frame.fbs](coordinate_frame.fbs) | `RCFR` |
| `SampleGeometry` | `WireSampleGeometry` | [sample_geometry.fbs](sample_geometry.fbs) | `RSGE` |

Each root table's four-byte `file_identifier` is required by HSDK 5 traits
generation. `WireArrayGeometry`, `WireWindowSpec`, `WireCoordinateFrame`, and
`WireSampleGeometry` are aliases for their generated owning types.

`LFMWaveformDescriptor` uses `WireLFMWaveform`, an alias for the same generated
`wire::FMCWWaveformT` payload. The [LFM bridge](lfm_waveform_bridge.hpp) converts
its fields and validates the distinct logical LFM descriptor; there is no
separate LFM schema root. Any shared waveform schema change must review both
waveform descriptors and bridges.

The remaining logical descriptors (`AngleConvention`,
`ShiftInvariantSubarrayPair`, `DDMCodeDescriptor`, and `TXSchedule`) are static
parameters in the current operator contracts and have no generated wire
schema here. A scalar/POD live override is a separate physical representation;
it does not create another shared descriptor.

C++ and Python graphs can connect registered native descriptor ports directly.
Python callers configure logical mappings through Params or use native live
producers; arbitrary generated wire-object construction is not exposed.

See [schema payload policy](../../../docs/ARCHITECTURE.md) for the
generator checklist, compatibility rules, Tensor bounds, zero-length optional
array convention, test obligations, and C++/Python support boundary.
