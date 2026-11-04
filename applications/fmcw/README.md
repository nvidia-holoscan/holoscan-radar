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
