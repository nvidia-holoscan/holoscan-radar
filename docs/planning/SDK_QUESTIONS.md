# Questions for the Holoscan SDK Team

Context: we have a SAR pipeline that accumulates pulses across ticks and needs
to know when a source is done (end of file, end of collection). We have now
**decided** how to do that — the source stamps a metadata key and whoever owns
the accumulator's control port translates it (`D22` in
[SAR/TEAM_BLOCKED.md](SAR/TEAM_BLOCKED.md)) — so this file is no longer a
decision gate. What remains is one design question we would rather not answer
alone, and a short record of what we concluded on the rest, in case any of it is
wrong.

Checked against Holoscan 4.3.0 headers.

---

## The live question

**A value-aware alternative to `kRaise`.** `MetadataPolicy` is per operator, and
`kRaise` fires on a key being *present* rather than on two values *disagreeing*.
That makes it unusable for us, for a reason we think is structural rather than
particular to our graph: `emit()` attaches one dictionary to every output port,
so a consumer taking several ports from the same producer merges identical
content several times and throws on the second `receive()`. Ours takes five or
six. The same thing happens in any diamond, where two paths from one ancestor
reconverge.

So we run `kUpdate`, and the check we actually wanted is gone: two stages
disagreeing about a key is silently last-writer-wins, which is the failure
`kRaise` looks like it exists to catch. Three sub-questions, in the order we
would find them useful:

1. Is a **value-comparing** policy plausible — accept a duplicate whose value is
   equal, raise where it differs? That would make `kRaise` usable for our shape
   without changing anything else, and we cannot see what it would break.
2. Is per-key or owned-key policy on the roadmap? A stage that legitimately
   rewrites one key currently has to take `kUpdate` for all of them.
3. If neither: is `erase()`-then-set the intended way to express "I own this key
   and I am replacing it", and would you expect a library to lean on it?

Our fallback is a documented ownership rule — one writer per key, and an
operator overwrites keys it defines but never keys it merely carries — enforced
by review rather than by the SDK. That works, but it is a convention we would
rather inherit than invent.

---

## Answered by our own design, recorded in case we got one wrong

- **End-of-stream / flush.** We concluded there is no SDK-provided pattern:
  `Operator::stop()` cannot emit, and a source that stops ticking is
  indistinguishable from a slow one, so the announcement has to be data-plane or
  metadata. We chose metadata, because a control *port* cannot stay aligned with
  the data it describes across a coalescing stage — non-blocking, the flag races
  ahead of a batch still in flight; blocking, FIFO alignment holds only while
  every intervening operator is strictly 1:1, and ours is not. Correct us if
  there is an intended mechanism we missed.
- **Metadata as control.** We are using it for exactly one transient
  announcement rather than for reset or flush commands, on the reading that
  "descriptive" covers *this batch is the last one* — an observation about the
  data, not an instruction. Ordering is not a concern for us because the key
  rides the message it describes, so no cross-port alignment is involved. If
  metadata is meant to be strictly provenance, we would like to know before we
  land it.
- **Key namespacing.** No convention exists that we could find, so we set our
  own: `holoscan.radar.<key>`, mirroring the C++ namespace and Python package,
  with modality tiers reserved beneath it. Flagged internally as R10 — whether
  the key belongs module-wide or under a modality — which is ours to decide. A
  convention or registry from your side would supersede ours happily.
- **Fragment boundaries.** Unverified, and it is on our own list to check before
  the convention is written as unconditional. If metadata does not cross
  fragments, our boundary announcement does not work in a partitioned
  deployment and we need a second mechanism there — worth knowing early.
