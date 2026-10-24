# Workspace Safety

Use `performance/optimization/workspace.py` for these operations. They preserve
accepted changes and do not create commits or modify agent configuration.

## Skills and Campaign Branches

Commit authorized optimization-skill updates on the designated skills branch.
Keep operator changes and campaign reports on the campaign branch. After a
skills update, rebase the campaign branch onto the designated skills branch
before continuing candidate work so it uses the committed workflow. Record
personal branch names and local artifact paths in session artifacts.

After rebasing, keep sealed manifests and reports immutable for their recorded
`source_id` and `git_commit`. If `source_id` changes, start a new confirmed
session, rebuild both manifests, and capture a new baseline before candidate
work. If only `git_commit` changes, preserve the provenance of sealed artifacts
and record the new commit on new artifacts.

Seal active measurement sessions before changing skills or rebasing; preserve
their source/build identities and evidence. Preserve uncommitted work when
switching branches. Do not infer permission to commit an operator candidate
merely to make a branch switch possible. When the user authorizes both, commit
the candidate first, commit the workflow change on the skills branch, then
rebase the campaign and verify retained file contents. This branch convention
does not authorize pushes or new measurement budget.

## Scratch Profiling

Before adding temporary instrumentation, register every file and every target
the scratch build will touch, using the session's current-best source ID:

```bash
python3 performance/optimization/workspace.py scratch-begin \
  --root CHECKOUT --source-id CURRENT_BEST --files RELATIVE_FILES \
  --targets ABSOLUTE_EXECUTABLES_AND_LIBRARIES --out /absolute/scratch.json
```

After profiling, `radar_perf.py seal` records the instrumented file state. Then:

```bash
python3 performance/optimization/workspace.py scratch-restore \
  --checkpoint /absolute/scratch.json --state /absolute/sealed.json \
  --out /absolute/restore-receipt.json
```

Restore refuses intervening file edits and checks the full current-best
fingerprint. It does not reset the checkout to a committed baseline. Rebuild
every declared target without instrumentation, refresh the hashed manifests,
then verify source IDs, artifact hashes and post-restore build timestamps:

```bash
python3 performance/optimization/workspace.py scratch-check \
  --receipt /absolute/restore-receipt.json --manifests ABSOLUTE_MANIFESTS \
  --out /absolute/rebuild-check.json
```

Keep these receipts as test/coverage evidence. Do not proceed to acceptance
timing until the check passes. This does not discover unlisted build consumers;
shared-header impact review remains mandatory.

## Relocated Archives

Do not move an active session. For historical sessions already moved, create a
separate JSON map from original absolute directory prefixes to their new roots.
Prefer the longest applicable prefix; do not change the original ledger, paths,
hashes or confirmed policy. Audit references recursively through gate JSON,
pair indexes and raw reports:

```bash
python3 performance/optimization/workspace.py audit \
  --session /absolute/session.json --relocations /absolute/relocations.json \
  --out /absolute/relocation-audit.json
```

The output preserves original paths and lists resolved paths, checked hashes
and failures. Missing/changed evidence fails the audit; do not invent artifacts
or rehash them to pass. This is read-only archive replay, not acceptance under
a new measurement policy. Runtime dependencies and archived source/build
execution require their own provenance checks. Record any relocation-map path
in the corresponding session artifacts.
