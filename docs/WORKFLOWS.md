# Processing workflows

Use operator README contracts for port semantics and application READMEs for exact graphs, presets, and runnable commands. All paths require a CUDA-capable NVIDIA GPU. Samples default to single precision; SAR applications keep geometry double precision.

## FMCW

| Pattern | Choose it for | Runnable application |
| --- | --- | --- |
| SISO range–Doppler | Baseline one-TX/one-RX processing; split or fused FFT stages | [SISO](../applications/fmcw/siso/README.md) |
| Detect then beamform | Sparse detections before angle estimation | [SIMO](../applications/fmcw/simo/README.md) |
| Beamform then detect | Dense angle/range/Doppler maps | [Dense beamforming](../applications/fmcw/dense_beamforming/README.md) |
| MIMO TDM | Deinterleave TX slots, Doppler compensation, virtual-array assembly | [MIMO TDM](../applications/fmcw/mimo_tdm/README.md) |
| MIMO DDM | Doppler folding/demultiplexing with linear or nonlinear codebooks | [MIMO DDM](../applications/fmcw/mimo_ddm/README.md) |
| Adaptive AoA | Multi-source screening and sparse MUSIC/MVDR/ESPRIT estimation | [Adaptive AoA](../applications/fmcw/adaptive_aoa/README.md) |
| Alternating coherent profiles | Per-frame waveform changes with matching Exact configuration | [Multi-waveform](../applications/fmcw/multi_waveform/README.md) |

Range compression applies optional dechirp and fast-time window/FFT. Optional clutter filtering precedes the slow-time Doppler FFT. Detection emits separate index, score, and noise tensors; explicit gather/assembly/reduction operators bridge those into angle estimation and clustering. Dense processing can beamform before detection; sparse processing gathers only selected snapshots. DDM preserves raw coded slow time until the selected demultiplexing path. TDM compensation uses each TX's timing before assembling geometry. See the [FMCW operator index](../operators/fmcw/README.md).

## SAR

The implemented FX path is CPHD reader → range compression → aperture manager → backprojection → optional dump/display. Range profiles and geometry align by capture time; stream boundaries flush partial apertures. The application supplies the emit schedule, geometry, and source configuration.

| Pattern | Retention/emission | Runnable application |
| --- | --- | --- |
| Spotlight | One fixed lattice, complete aperture, one product | [Spotlight](../applications/sar/spotlight/README.md) |
| Video SAR | Sliding aperture with signed eviction, recurring products | [Video SAR](../applications/sar/video_sar/README.md) |
| Tiled strip | Successive independent tiles over a longer strip | [Tiled strip](../applications/sar/tiled_strip/README.md) |
| Continuous strip | Scrolling lattice retaining overlapping rows | [Continuous strip](../applications/sar/continuous_strip/README.md) |
| Independent images | Separate collections and lattices with boundary resets | [Independent images](../applications/sar/independent_images/README.md) |
| Autofocus | Estimate on a slow-time-conjugate auxiliary image, resample phase, correct replayed profiles, form product | [Autofocus](../applications/sar/autofocus/README.md) |

A focused spaceborne scene may need no autofocus. Direct in-image PGA correction is meaningful only on a slow-time-conjugate grid; a ground/DEM product uses estimation followed by profile correction and reformation. A sharpness increase is not evidence of accurate phase recovery. The app-owned phase resampler bridges estimation extent to pulse count; whether it should become library API remains open.

Public collections are optional and prepared by [checkout tools](../applications/sar/tools/README.md). Generated fixtures establish analytic correctness, not operational qualification of every collection, backend, seam, or radiometric convention. Deramped/direct-sampled branches, CRSD ingress, multilook/geocoding, product writers, wide-swath/circular/volumetric modes remain deferred. No Pulse-Doppler/PVA, CDM/BPM, FDM, or PMCW workflow is supplied.
