# Questions for the Holoscan SDK Team — Metadata and Stream Control

## Why we are asking

`holoscan-radar` is adding a SAR modality whose formation stage **accumulates
across ticks**: pulses arrive indefinitely, an image is integrated over many of
them, and something has to say when an aperture ends and when a finite source
has run out. We have two candidate designs — a parallel `control` input port,
and boundary facts attached to the data stream as metadata — and which one is
correct depends on what the SDK intends rather than on what we prefer.

We would rather use an existing idiom than invent a convention that a later SDK
feature contradicts. The questions below are ordered by how much each answer
would change our contract.

Everything stated as observed was checked against the **Holoscan SDK 4.3.0**
headers in `/opt/nvidia/holoscan/include/holoscan/core/`, with file and line
references given so they are easy to correct if we have misread them.

## The problem, concretely

A CPHD file reader is a **finite** source. When it exhausts its input, the
partial aperture held by the downstream accumulator has to be flushed, or the
tail of the collection is silently discarded — a product that looks correct and
is missing its last image.

The obvious implementation is not available: `Operator::stop()` takes no
arguments and has no `OutputContext` (`operator.hpp:485`), so nothing can be
emitted from it. A source that simply ceases to tick is also indistinguishable
from one that is slow, so the downstream operator cannot infer termination
either. Termination therefore has to be **announced**, and the only operator
that knows is the reader.

The same announcement is needed for two further cases: a file boundary when
reading a directory of collections, and an application-initiated change of
imaging geometry.

## What we found in 4.3, and why we are unsure

Three mechanisms are adjacent to this problem and none of them appears to be
an end-of-stream or flush signal:

- **The metadata plane.** A `MetadataDictionary` travels with messages, is
  cleared at the start of every `compute()`, is populated by
  `InputContext::receive()` and attached to every `OutputContext::emit()`
  (`operator.hpp:563-577`). Attractive here because a boundary is genuinely a
  *property of a batch*, and because attaching it to the data makes co-arrival
  structural rather than a protocol rule we have to police.
- **Execution ports.** `kInputExecPortName` / `kOutputExecPortName`
  (`operator.hpp:169-172`) provide dataless edges that sequence execution.
  They express "run after this" but carry no payload, so they cannot
  distinguish the several boundary kinds we need.
- **Dynamic flows.** `add_dynamic_flow()` (`operator.hpp:787-823`) lets an
  operator choose at runtime which downstream operators its output reaches. A
  source could in principle route its final tick differently, but expressing
  "this is the last batch" as a routing change seems like a misuse.

That the SDK has all three, and that none covers this case, is what makes us
suspect we are either missing an idiom or looking at a real gap.

## Questions

### 1. What is the intended way for a finite source to flush downstream stateful operators?

**Why it matters.** This is not specific to radar. Any accumulating operator
fed by a file reader, a dataset replay, or a bounded queue has the same
problem — video decode, batch inference and dataset-driven pipelines included.
Our workaround is a `control` input port on the accumulating operator that some
upstream operator must drive, which works but makes every application wire an
extra edge whose omission is silent.

**What would change.** If there is an intended idiom, we will use it instead of
defining our own vocabulary. If there is not, we would like to know whether
this is considered a gap, because a first-class end-of-stream signal would let
a large class of operators stop reinventing one.

### 2. Is the metadata dictionary intended as a control channel, or as descriptive provenance only?

**Why it matters.** Reset and flush semantics are load-bearing: if a boundary
flag is dropped, reordered, or delivered twice, an image is silently wrong
rather than obviously broken. Provenance metadata can tolerate weaker
guarantees than control signalling can.

**What would change.** If metadata is descriptive by design, we will not build
reset semantics on it and will keep the explicit port. If it is a legitimate
control channel, we would like to know which delivery and ordering properties
are actually promised — particularly under the greedy and multithread
schedulers, and with queue depths greater than one.

### 3. Is `kRaise` the intended default, given that `emit()` broadcasts the whole dictionary to every output port?

**Observed.** `kDefaultMetadataPolicy = MetadataPolicy::kRaise` and
`kDefaultMetadataEnabled = true` (`fragment.hpp:83-84`). On a key collision the
dictionary throws (`metadata.hpp:177-198`), and the throw happens whenever the
key already exists — there is no exemption for a duplicate carrying an
identical value. Meanwhile the operator's whole dictionary is attached to
*every* `emit()`.

**Why it matters.** Those two defaults interact badly in an ordinary topology.
One of our operators emits four related tensors — samples, positions, a
reference point, and a per-sample weight — on four ports, and the downstream
operator receives all four. A single key stamped upstream is therefore
broadcast to all four outputs and collides on the second `receive()`, so the
consumer throws unless it sets `kUpdate` or `kInplaceUpdate`. Stamping the key
narrowly does not help, because the broadcast reintroduces it after one hop.

The consequence is that **any library using metadata forces a policy change on
every downstream operator that consumes more than one of its ports**, including
third-party operators whose authors never opted into metadata at all.

**What would change.** If this is deliberate, we will document the policy
requirement as a condition of composing with our operators. If it is not, the
two fixes we can see are exempting duplicate-identical values from the raise,
or allowing `emit()` to attach metadata per port rather than to all outputs.

### 4. Is there a namespacing convention for library-defined metadata keys?

**Why it matters.** If several operator libraries stamp keys into a shared
dictionary, collisions across libraries are hit by the same throwing default in
question 3, and the failure appears in an application that composed two
libraries neither of whose authors did anything wrong.

**What would change.** We would adopt whatever convention exists — a reserved
prefix, a reverse-DNS scheme, a registry. Absent one we would namespace our own
keys and say so, which only helps for collisions involving us.

### 5. Does metadata survive a fragment boundary in a distributed graph?

**Why it matters.** If metadata does not cross fragments, it cannot be a
control channel for distributed deployments, and a design that works in a
single-fragment application would fail on partitioning — a failure that appears
at deployment time, far from the change that caused it.

**What would change.** A negative answer rules metadata out for control in our
contract regardless of the other answers, and we would keep the explicit port.

### 6. Can a policy be expressed per key rather than per operator?

**Observed.** `policy_` is a member of the `MetadataDictionary`, and the
collision switch consults that single value for every key
(`metadata.hpp:177-198`). An operator therefore has one policy covering every
key it receives, not one per key.

**Why it matters.** Our intended use is not a fixed set of facts carried
end to end. It is a plane whose entries are **transformed stage by stage** —
that is the premise of the metadata-plane item in our own planning docs, and we
have already hit it twice in the port namespace. The reader emits the file's
fast-time sample axis; range compression replaces it with a range axis, and we
deliberately gave the two different port names so they could not be confused.
A geocoding stage legitimately *changes* the coordinate frame, so that key must
be rewritten by exactly one stage and by no other.

An operator that redefines one key while merely carrying a dozen others has to
choose a single policy for all of them. `kRaise` forbids the redefinition the
stage exists to perform. `kUpdate` permits it and simultaneously silences
accidental collisions on every other key — including the silent disagreement
between a processing stage and a downstream standard-format writer, which is
the failure a metadata plane would be adopted to prevent. The only usable
policy defeats the purpose of the mechanism.

**What would change.** A per-key policy, or a way for an operator to declare
which keys it owns and may overwrite, would let us keep strict checking on
carried keys while allowing rewrite where a stage genuinely redefines meaning.
Absent that, the workaround we can see is for a transforming stage to `erase()`
a key before writing it, and we would like to know whether that is the intended
idiom or an abuse of it.

## What we will do absent answers

Ship the explicit `control` port. It works today with no SDK change, it is
visible in the graph so a missing connection can be rejected at initialization,
and it does not depend on any of the semantics questioned above. The metadata
design is better ergonomics — no extra edge for the common case, and co-arrival
guaranteed by construction rather than by a rule — and we would prefer it if
the answers support it.

This question is tracked as **R5** in
[SAR/TEAM_BLOCKED.md](SAR/TEAM_BLOCKED.md), where the metadata dictionary is
already named as the candidate mechanism for a metadata plane, and as **R7**,
which is the end-of-stream announcement specifically.
