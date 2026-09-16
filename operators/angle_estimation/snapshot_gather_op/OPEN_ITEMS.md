# SnapshotGatherOp — open items

## Dense-CFAR index projection for sparse AoA refinement

`CFARDetectorOp` on an RDA cube emits rows shaped
`(az_bin, el_bin, doppler_bin, range_bin)`. Gathering a refinement snapshot
from the original SIMO channel cube `(num_channels, num_doppler, num_range)`
should use only the trailing Doppler/range coordinates. The leading angle-bin
columns are candidate annotations, not axes of that input cube.

`SnapshotGatherOp` currently interprets every column before Doppler/range as a
preserved-batch coordinate and requires one matching leading `rd_maps` axis
per column. It therefore rejects this otherwise valid dense-detect followed by
sparse-AoA refinement path.

Add an explicit static index-projection or batch-column mapping that lets the
operator distinguish columns which address real `rd_maps` leading axes from
annotation columns ignored during gathering. Keep the current strict
all-preserved-columns behavior as the default; do not infer or silently discard
columns from tensor rank alone. Output rows must remain aligned with the
original full `detection_idx` rows so coarse `az_bin`/`el_bin` values stay
available to reporting, feature assembly, and future local-grid refinement.

The implementation must cover:

- rank-3 SIMO `rd_maps` with dense four-column RDA detections;
- existing batched `rd_maps` behavior without projection;
- invalid, duplicate, and out-of-range column mappings;
- empty detections and all three snapshot edge modes;
- graph integration through a sparse AoA estimator using a shared refinement
  grid.

This is ordinary deferred feature work, not a team-blocked contract decision.
