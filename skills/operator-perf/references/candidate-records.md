# Candidate Records

Every attempted optimization needs a readable record of what changed, why it
was kept, rejected or blocked, and its measured performance impact. Gate JSON
and timing tables support that explanation; they do not replace it.

## Register and timing

Keep a versioned `candidates.md` in the campaign report directory, linked from
`overview.md` and `final.md`. For a standalone operator session, use its report
directory. Give every attempt a stable key containing operator, session and
candidate ID; `candidate-1` alone is not unique across sessions.

The register includes one row per attempted candidate:

| Candidate | Operator / session | Change | Outcome and reason | Measured impact | Explanation |
| --- | --- | --- | --- | --- | --- |
| Stable attempt key | Operator and session ID | Concrete mechanism | Accepted (kept), rejected, or validation_blocked, with decisive evidence | Cases/precisions, absolute before/after and effect; or why unavailable | Link to this attempt's document |

Use the sealed engineering decision record or maintained formal session's
outcome; label acceptance basis `engineering` or `formal-v4` and distinguish
rejected candidates from blocked validation. A candidate that fails screening still needs a record.
Unattempted source leads, preparatory fixture failures and harnessless operators
are not rejected candidates. Report attempted/accepted/rejected/blocked counts
explicitly, including zeros, and reconcile them with session records. Preserve
earlier attempts when a later candidate supersedes them; historical acceptance
and current retained state are separate facts.

Prepare the explanation in ignored session notes while source identities are
frozen. Publish the register and attempt documents between sealed sessions,
including interrupted/blocked attempts when possible. Never change active
source fingerprints merely to update a report. Preserve incomplete evidence
and state what remains pending; publication does not authorize more runs.

## Per-attempt explanation

Use a stable path such as
`candidates/<operator>-<session>-<candidate-id>.md`. An existing report may serve
as the attempt document if it identifies exactly one attempt, contains the
information below, and remains available when later attempts are reported.
Keep current operator summaries linked to these historical records.

Write for a reviewer who has not read the conversation. Include:

- **Identity and outcome:** operator/session/candidate IDs, parent/current-best
  and candidate source/build identities, decision, and whether the change is
  currently retained or has been restored/superseded. Link the registered plan,
  session/candidate record and gate artifacts.
- **Problem and actual change:** explain the previous execution path and its
  attributed bottleneck, then the new execution path. Name changed files and
  functions, how work/allocation/synchronization changed, relevant thresholds,
  and tradeoffs. A small example, pseudocode or diagram can clarify the change.
  A title such as "parallel planner" or a list of modified files is insufficient.
- **Behavior preservation:** explain the relevant ordering, math, precision,
  error, ownership and temporal constraints and how the candidate preserves
  them. Describe any candidate-introduced failure without hiding it behind an
  aggregate gate result.
- **Decision rationale:** name the decisive passing or failing gates and the
  evidence supporting interpretation. For rejection, identify the correctness
  failure, missed objective or demonstrated protected regression; for blocked
  validation, identify missing/invalid evidence or unresolved confounding.
  Explain restoration and link its receipt where applicable. Never infer
  acceptance from a screen or a favorable control review alone.
- **Performance impact:** identify target/protected cases, recipe dimensions,
  both precisions and affected consumers. Show absolute before/after latency,
  percentage effect and descriptive variability or declared formal uncertainty
  for the decision-relevant metrics;
  link the measured set and explain omitted variants/consumer timing. Separate
  expected gains, exploratory screens,
  confirmation and direct original/final results. Preserve unfavorable results,
  inconclusive protection and coverage gaps. When timing was not reached or is
  invalid, say so and make no measured-gain claim. Never multiply intermediate
  gains. A fresh first-candidate check comparing original to final retained
  bytes already supplies final evidence; reference it rather than duplicate it.
- **Limits and next action:** include attribution limits, unresolved findings
  and the disposition's consequence for the remaining queue/budget.

Check that register links resolve and reported outcomes/counts agree with the
sealed session. Keep raw artifacts in ignored paths and refresh publication
provenance separately from immutable measurement identities. These records add
reporting requirements, not new acceptance rules, measurement budget or
commit/push authorization.
