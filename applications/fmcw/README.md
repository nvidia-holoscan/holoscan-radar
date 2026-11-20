# FMCW application presets and modes

All seven FMCW applications support the same single-file configuration interface
in C++ and Python. A preset can inherit one base file and override only selected
settings:

```yaml
extends: adaptive_aoa.yaml
app:
    aoa_algorithm: esprit
```

`extends` is resolved relative to the file containing it, independently of the
working directory. Bases can themselves extend another file, up to 32 files in
one chain. Missing files, cycles, invalid declarations, and non-mapping roots
produce errors naming the files involved. Standalone YAMLs remain supported.

Mappings merge recursively using the SDK configuration API. Scalars and arrays
replace the complete base value: overriding `app.targets`, a velocity vector, or
`app.profiles` requires the whole desired array. `null` replaces a value too;
smoke presets use `app.range_loss: null` to retain their constant-amplitude
simulation. It is not a general key-deletion operation. YAML anchors are local
to each file and resolve before merging. Changing an anchored value in an
override does not change other resolved copies of that anchor; waveform presets
must override each affected waveform block explicitly.

The GUI shows the selected mode description below its main title, with frame
status on a separate line. Normal mode descriptions match across C++ and Python;
the build generates the shared GUI copy from the C++ manifests. Algorithm/coding
settings select the description, including for custom inherited presets.

## Launch modes

Modes are the optional second positional argument to the host CLI. Python is
selected when the language is omitted; pass `--language cpp` for C++.

```bash
./holoscan-radar run mimo_tdm smoke
./holoscan-radar run mimo_tdm smoke --language cpp
./holoscan-radar run adaptive_aoa esprit
./holoscan-radar run mimo_ddm nonlinear_ddm
```

| Application | Modes |
| --- | --- |
| All FMCW applications | `default`, `smoke` |
| MIMO DDM additionally | `nonlinear_ddm`, `nonlinear_ddm_smoke` |
| Adaptive AoA additionally | `root_music`, `esprit`, `mvdr` |

`default` retains each application's existing normal preset: linear DDM and
MUSIC for those applications. `smoke` is the existing two-frame headless preset;
Adaptive AoA has one MUSIC smoke mode. Alternate algorithms and DDM coding remain
structural choices made before graph construction, rather than live settings.
The DDM normal presets share `mimo_ddm_base.yaml`, which is not a launch preset.

For a custom preset, use the default mode and pass its filename or path:

```bash
./holoscan-radar run adaptive_aoa --run-args=/absolute/path/my_aoa.yaml
```

The CLI appends `--run-args` to the selected mode's command. Named modes already
supply a preset, so use `default` when selecting a different file. Existing
`--run-args=app_smoke.yaml` commands continue to work. The Python
`build_graph(path)` API accepts the same inherited or standalone files. All
shipped presets and bases are staged together beside the C++ executable, so
relative inheritance also works from the build tree.

## MIMO source memory allowance

The MIMO TDM, DDM, adaptive AoA and multi-waveform examples use a shared
source preflight in both C++ and Python. Set a positive byte allowance in YAML:

```yaml
app:
    max_source_working_bytes: 536870912  # Default 512 MiB; raise or lower as needed
```

Oversized configurations fail before geometry construction or simulation.
Every multi-waveform profile is checked independently, including later profiles.

The conservative logical payload estimate for one synthesis is
`24*R*L*S + 80*R*L + 16*T*L + 64*S + 24*T*R + 48*(T+R)` bytes,
plus `8*L` for a TDM transmit pattern. Here `T` is the TX count,
`R = az_elements * el_elements`, `S` is samples per chirp, and
`L = T * num_chirps` for TDM (`num_chirps` is per TX), or
`L = num_chirps` for DDM. Profile configs use `chirps_per_tx` for TDM.
The terms cover the float-complex output and double-complex accumulator,
host/device synthesis scratch and upload conversion, excitation weights,
sample/reference arrays, virtual positions, and scene/flattened antenna positions.
Scratch terms are conservatively summed even when lifetimes do not overlap.

For the shipped 2-TX, 16-RX, 128-slot, 256-sample configuration the estimate
is about 12.2 MiB; the 512-sample profile is about 24.2 MiB. These are
source payload estimates, not measured process peaks. The allowance excludes
allocator reservations, retained/queued frames, generated or authored DDM code
descriptors and their copies, target/config storage, graph Tensor pools,
FFT/library caches, downstream operators and visualization. Frame count affects
run duration and does not multiply this single-synthesis estimate.
It does not limit general library simulator or operator APIs.

YAML can raise its own allowance, so this guard catches accidental oversized
workloads rather than enforcing a boundary against hostile configuration files.
The separate DDM generated-table allowance is documented in
[the DDM example](mimo_ddm/README.md#configurations).
