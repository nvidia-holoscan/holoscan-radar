# Team-Blocked Items

Contract / design questions that need a **team decision**. Individuals must
not silently resolve these in code.

Per-operator detail lives in each operator’s
[`OPEN_ITEMS.md`](../../operators/). This file is the roll-up. When a decision
lands, record it in [`STATUS.md`](STATUS.md) → Settled decisions, then remove
or archive the operator section *and* the matching row here.

Items fall into **two classes**. Keep that distinction sharp when adding rows:

| Class | Meaning | Development impact |
| --- | --- | --- |
| **Stops development** | Work cannot finish (or cannot start the next operator) until the team picks an answer | Checklist incomplete, param intentionally undeclared, or next-wave op blocked from landing |
| **Needs team look** | An interim answer is already in code (or the feature is out of current scope); the team should ratify, revise, or reject | Does **not** block landing or day-to-day implementation; do not treat as a stop-ship |

**Not in this file:** perf follow-ups, encoding notes pending bindings, deferred
algorithm slices that are simply out of scope, and SDK caveats that are only
documented (unless they need a product policy — then they belong in one of the
two classes below).

---

## How to use

| Action | Do this |
| --- | --- |
| Discover a new team question | Write it in `operators/<group>/<op>/OPEN_ITEMS.md`, then add a row in the **correct class** below |
| Land code ahead of the contract | Record the conflict in `OPEN_ITEMS.md`; put the row under **Needs team look** (unless it truly stops checklist completion) |
| Resolve | Team decision → `STATUS.md` Settled decisions → delete/archive both places |

Agents: [`AGENTS.md`](../../AGENTS.md) (“Do not silently resolve documented open
items”) and [`OPERATOR_DEVELOPMENT.md`](OPERATOR_DEVELOPMENT.md) § Open items.

---

## Stops development

These are the only items that should pause or refuse checklist completion /
starting a dependent operator.

**None currently.** (S1 spatial-smoothing, S2 source-count, S3 subarray-selection,
and S4 dense-path angle features are all settled — see STATUS.md → Settled
decisions; residual ratification forks are under **Needs team look** below.)

---

## Needs team look

Interim stance already chosen (or the capability is explicitly out of current
scope). Development and checklist completion proceed; these need ratification
or a deliberate contract change when the team has bandwidth.

### CFAR / detection

| # | Decision needed | Interim stance | Detail |
| --- | --- | --- | --- |
| L1 | **Bool mask dtype vs HSDK 4.3 DLPack** — keep `bool` (target HSDK 5), add `uint8` compat, or document C++-only until 5 | Emit/accept `kDLBool`; C++ wiring works with a logged GXF error; **Python bindings drain bool masks via `NullSinkOp` / harness `drain_ports` (no NumPy/`PyTensor` conversion on 4.3)** | [CFAR §1](../../operators/detection/cfar_detector_op/OPEN_ITEMS.md); SnapshotGather / Beamscan / MVDR |
| L6 | **Exact-sized `detection_idx`** — keep (accept sync) or add fixed-capacity / valid-length | Matches locked exact-size contract today | [CFAR §8](../../operators/detection/cfar_detector_op/OPEN_ITEMS.md) |

### Clustering (`ClusterDBSCANOp`)

| # | Decision needed | Interim stance | Detail |
| --- | --- | --- | --- |
| L7 | **`max_clusters` surplus policy** | Drop highest dense ids; surplus → noise (`-1`) | [ClusterDBSCAN §1](../../operators/clustering/cluster_dbscan_op/OPEN_ITEMS.md) |
| L8 | **`epsilon` scalar-vs-vector encoding** | `std::vector<double>` only (`epsilon: [0.5]` for scalar) | [ClusterDBSCAN §2](../../operators/clustering/cluster_dbscan_op/OPEN_ITEMS.md) |
| L9 | **Border-point multi-cluster tie-break** | Min-root neighbor | [ClusterDBSCAN §5](../../operators/clustering/cluster_dbscan_op/OPEN_ITEMS.md) |
| L10 | **Cluster-id numbering order** | Ascending component-root | [ClusterDBSCAN §6](../../operators/clustering/cluster_dbscan_op/OPEN_ITEMS.md) |

### Angle conventions & MIMO scope

| # | Decision needed | Interim stance | Detail |
| --- | --- | --- | --- |
| L11 | **`AngleConvention::kCustom` boundary mapping** | Presets work; **`custom` throws** — operators are checklist-complete without it | [Beamscan §1](../../operators/angle_estimation/beamscan_estimator_op/OPEN_ITEMS.md); [PhaseShift §1](../../operators/angle_estimation/phase_shift_beamformer_op/OPEN_ITEMS.md) |
| L12 | **Non-uniform / staggered TDM schedules** | **Rejected** today; uniform repeating TDM only — ops checklist-complete for that scope | [MIMODeinterleave §B.1](../../operators/fmcw/mimo_deinterleave_op/OPEN_ITEMS.md) |
| L13 | **S1 spatial-smoothing forks** — (a) integer means sub-array **count** vs length; (b) holey arrays route to S3 vs. in-operator explicit partitions | **Implemented:** count semantics; holey → error → S3 / upstream custom smoother | [MVDR OPEN_ITEMS](../../operators/angle_estimation/mvdr_estimator_op/OPEN_ITEMS.md) |
| L14 | **S3 subarray-selection forks** — (a) auto vs explicit; (b) objective: max element-count vs aperture; (c) contiguous ULA vs sum-set holey smoothing; (d) standalone op vs `VirtualArrayAssemblyOp` mode | **Implemented (`SubarraySelectionOp`):** auto largest filled ULA/URA + explicit; max element-count, azimuth tie-break; contiguous only; standalone op. Root-MUSIC and ESPRIT now consume the resulting uniform geometry | [SubarraySelection OPEN_ITEMS](../../operators/angle_estimation/subarray_selection_op/OPEN_ITEMS.md) |
