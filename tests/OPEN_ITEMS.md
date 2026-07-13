# Test Suite — Open Items

Deferred, agreed-upon work for the test tree. Each item records the intent
and the reason it was deferred, so it can be picked up without re-litigating
the design. Remove items when they land (note the commit in the PR).

## 1. Mirror the source tree under `tests/cpp/`

**Status:** agreed 2026-07-10, deferred until `deustice/sigproc-ops` merges.

Reorganize the flat `tests/cpp/` file list to mirror `operators/`, so the
"where is the test for X" answer stays mechanical as operator families
multiply (beamforming, DDM/MIMO, estimation are planned — see
`docs/planning/OPERATOR_API.md`):

```text
tests/cpp/
├── utils/                          # shared harness, fixtures, bin_model (unchanged)
├── common/
│   └── test_common.cpp             # see item 2
├── fmcw/
│   ├── test_fmcw_range_processor.cpp
│   ├── test_fmcw_doppler_processor.cpp
│   ├── test_fmcw_range_doppler_processor.cpp
│   └── test_clutter_filter.cpp
└── test_pipeline.cpp               # cross-family integration stays at the root
```

Conventions locked in with the layout:

- Each new operator family `operators/<family>/` gets a matching
  `tests/cpp/<family>/`.
- Cross-family integration tests live at the `tests/cpp/` root (as
  `test_pipeline.cpp` does today) — their position signals scope.
- Keep the single `tests/cpp/CMakeLists.txt`; `radar_add_test()` takes
  relative paths, no per-directory build files at this scale.

**Why deferred:** pure `git mv` churn on five files mid-review of an
~8k-line branch would turn reviewers' per-file diffs into rename+edit
noise. It is a mechanical, self-contained follow-up commit once the branch
lands.

## 2. Split `test_common.cpp` by concern

**Status:** agreed 2026-07-10, do together with item 1.

`test_common.cpp` covers window DSP, descriptors, the config cache, the
point-target simulator, and the YAML converters (~500 lines and growing).
When `tests/cpp/common/` exists, split it along the `radar_common`
subdirectory layers, e.g.:

- `test_window.cpp` (generation + cache keys)
- `test_descriptors.cpp` (`FMCWWaveformDescriptor`, future descriptor types)
- `test_sdk.cpp` (config cache, YAML converters, arg setters)
- `test_sim.cpp` (simulator self-checks)
