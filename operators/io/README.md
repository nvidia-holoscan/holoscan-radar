# SAR I/O operators

The HSDK 5 C++/CUDA SAR port includes `CPHDReaderOp<T>` and `TensorDumpOp`.
The remaining planned CPHD/CRSD/SICD/SIDD readers and writers in the SAR
planning documents are deferred. Each operator has an independent `OP_<name>`
CMake gate; `CPHDReaderOp` needs tinyxml2. Python bindings and performance
harnesses are outside this port.

| Operator | Ports | Role |
| --- | --- | --- |
| `CPHDReaderOp<T>` | `phase_history`, `positions`, `reference_position`, optional diagnostic tensors and `coordinate_frame`, plus `stream_boundary` | Decode a selected CPHD channel into device tensors and stamp every batch with a typed boundary |
| `TensorDumpOp` | `tensor` input, optional `[live]` `every_n` input | Write raw or NumPy `.npy` tensor bytes for inspection |

`CPHDReaderOp.stream_boundary` is a `uint8` scalar: `0` means none, `1`
means collection boundary, `2` means end of stream. It shares the pulse batch's
capture time. A splitter must place the boundary on the final slice of that
batch. The reader accepts CF8, CI4, and CI2 samples and applies per-vector
amplitude scale factors. The generated C++ CPHD fixture and reader tests cover
format decoding and boundary ordering; public collections remain an optional
validation step.

`TensorDumpOp` accepts any tensor dtype and rank. It copies to host and writes
synchronously, so callers should expect a device synchronization and file I/O
cost whenever a tick is dumped. `every_n` is an optional same-tick override;
`max_files` is static. A Python interpreter is not needed for the C++ tests.

The public contract is in
[SAR OPERATOR_API.md](../../docs/planning/SAR/OPERATOR_API.md).
Unresolved implementation and validation questions are collected in the
[SAR open-items roll-up](../../docs/planning/SAR/OPEN_ITEMS.md) and the historical per-operator notes.
