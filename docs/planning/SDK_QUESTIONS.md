# Questions for the Holoscan SDK Team

Context: we have a SAR pipeline that accumulates pulses across ticks and needs
to know when a source is done (end of file, end of aperture, geometry change).
We're deciding between an explicit `control` port and metadata on the data
stream. Preferring an existing SDK idiom over inventing our own.

Checked against Holoscan 4.3.0 headers. Tracking: **R5** / **R7** in
[SAR/TEAM_BLOCKED.md](SAR/TEAM_BLOCKED.md).

---

1. **End-of-stream / flush.** How should a finite source tell a downstream
   stateful operator "flush now"? `Operator::stop()` can't emit, and stopping
   ticking looks the same as being slow. Is there an intended pattern?

2. **Metadata as control.** Is the metadata dictionary meant for control
   signals (reset, flush), or only descriptive provenance? If control is OK,
   what delivery/ordering guarantees does it have under greedy and multithread
   schedulers?

3. **`kRaise` + multi-port emit.** Default policy is `kRaise`, but `emit()`
   attaches the whole dictionary to every output. A consumer that `receive()`s
   multiple ports from the same upstream then throws on the second receive.
   Intentional? Or should identical-value duplicates be allowed / metadata be
   per-port?

4. **Key namespacing.** Is there a convention for library-owned metadata keys
   (prefix, reverse-DNS, registry)? Without one, two libraries collide and the
   app that composed them gets the throw.

5. **Fragment boundaries.** Does metadata cross fragments in a distributed
   graph? If not, we can't use it for control in partitioned deployments.

6. **Per-key policy.** Policy is per-operator, not per-key. A stage that
   legitimately rewrites one key has to choose `kUpdate` and lose collision
   checks on everything else. Is `erase()`-then-set the intended workaround,
   or is per-key / owned-key policy on the roadmap?

---

**Default if unanswered:** ship the explicit `control` port. It works today and
doesn't depend on any of the above.
