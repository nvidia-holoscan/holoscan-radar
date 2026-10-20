# Measurement design and capability checks

Use the approved [screening and confirmation workflow](confirmation.md).
Qualification means establishing that the measurement design and tools can
support the declared comparison. It does not require an unchanged operator to
pass a universal repeatability limit or a standalone A/A campaign before edits.

1. Inventory cases, precision, native verification and affected consumers.
   Record deployment preflight, immutable image/SDK identity, runtime libraries,
   source hashes and timing boundaries. Check that the executable toolchain
   supports the approved protocol; legacy version-3 admission is not equivalent.
2. Require a positive caller-specified measurement budget before timing.
   Forecast screening, four-run confirmation blocks, builds and storage. Keep
   independent unresolved scopes visible; do not spend a closed allocation's
   nominal unused balance or automatically request larger experiments.
3. Use existing captures, source review and inexpensive profiles first. Explore
   sampling only to resolve a concrete question. More frames address within-run
   distribution estimation; more independent blocks address between-run
   uncertainty. The old 50/500/2,000-frame and 3/7-repeat grid is not a mandatory
   exploration sequence or universal benchmark default.
4. Keep five warmups and existing process-warmup semantics by default. Record
   chronological raw frames and boundaries. Declare sampling/horizons explicitly;
   never silently discard early frames, raise warmup or translate legacy
   repeat counts into four-run confirmation blocks. Inspect chronological modes,
   dependence and tail estimation before lengthening runs.
5. Screen a correct bounded candidate with paired A/B. Freeze the candidate,
   matrix, targets, estimator, sampling, randomized balanced schedule, error
   budget and stopping design before fresh held-out confirmation. Confirmation
   contains two A and two B runs per block, with A/A and B/B controls alongside
   the A/B distribution effect. Screening data do not enter acceptance.
6. Protect early/full p50/p90/p99, setup, cold and applicable live transitions
   across the confirmed cases and consumers. Treat complete blocks as inference
   units and account for shared-control dependence. Include operator/candidate/
   endpoint/interim-look multiplicity. Derive block count from resolution and
   expected sensitivity; a default pair count is not a power guarantee.
7. Report invalid evidence, unresolved confounding, limited sensitivity,
   unsupported intervals, ordinary variability and demonstrated slowdowns
   separately. Show absolute and relative A/B and A/A–B/B diagnostics, uncertainty,
   maxima/stalls and chronology. High scatter alone is not a measurement failure
   or an operator bug. Non-significant controls do not establish stability.

Standalone A/A or B/B may answer a specific tooling or deployment diagnostic
question, with an explicit bounded purpose. They are optional diagnostics, not
admission gates. Existing repeats can provide exploratory within-build
comparisons, but shared observations remain dependent and cannot replace fresh
held-out confirmation after candidate selection.

No fixed percentage or microsecond consistency threshold applies to new-policy
controls. Do not subtract observed A/A noise or use it as a permitted regression
margin. Confidence width and +/-1% remain diagnostics, not universal requirements.
Preserve exact correctness and the declared practical objective and protection
semantics. See [distribution inference](distributions.md) for decisions and
[the measurement review](measurement-redesign.md) for rationale and sampling limits.

Use version 4 for this workflow; see the
[command guide](../../../performance/optimization/CONFIRMATION.md). Its campaign
admission verifies native behavior and does not invoke legacy A/A qualification.
The old `qualification.qualify()` API and version-2/version-3 commands retain
steady 5%/10%, setup 15%/25%, cold 20%/30% and override 25%/40% P90-delta/IQR
rules for historical replay. Those are not new-workflow defaults. Never rewrite
old results or silently promote them to version-4 evidence.

Keep hash-verified units, source/build/runtime identities, randomization and
policy identity across resume. Per-device advisory locks serialize cooperative
measurements. Clean up only owned processes/containers; external GPU activity
is a context observation, not permission to stop other work. Unknown clocks,
power, affinity or profiling capabilities stay unknown. Never change clock,
power, affinity or privileged profiling settings implicitly.
