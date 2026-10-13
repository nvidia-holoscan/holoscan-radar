# Radar Schema Payloads

This directory owns the committed FlatBuffers sources for shared HSDK 5 wire
payloads. Generated headers and traits are build outputs and must not be
committed.

Schemas here provide physical transport for existing logical radar
descriptors; they do not create additional shared descriptor types. One
schema target owns each canonical payload identity, and all operator
consumers use that target rather than copying `.fbs` files.

Each root table declares a four-byte `file_identifier` required by current
HSDK 5 traits generation: `RAGY` (`ArrayGeometry`), `RWSP` (`WindowSpec`),
`RWFV` (`FMCWWaveform`).

See [`SCHEMA_PAYLOADS.md`](../../../docs/planning/SCHEMA_PAYLOADS.md) for the
generator checklist, compatibility rules, Tensor bounds, zero-length optional
array convention, test obligations, and C++/Python support boundary.
