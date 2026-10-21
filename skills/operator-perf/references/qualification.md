# Measurement design and capability checks

Use [proportionate acceptance](acceptance.md) by default. Qualification means
checking that tools, provenance and timing boundaries support the intended
comparison; it is not a mandatory repeatability study or full-matrix A/A sweep.

1. Identify the question: bottleneck attribution, an engineering keep decision,
   or a formal statistical claim. Choose the cheapest evidence that can answer it.
2. Record deployment/image, build/runtime/source/recipe identities and timing
   boundaries. Validate the selected executable and list actual recipe cases.
   Do not infer device capability from host orchestration tests.
3. Require a positive caller-specified measurement budget. Forecast actual
   launches, verification, setup and storage as well as GPU work. Separate build
   preparation from measurement-job accounting. Large overhead is a reason to
   revise the experiment, not automatically request more hours.
4. Choose representative performance cases by changed execution path and risk.
   Maintain broad relevant correctness separately. Downstream functional
   coverage does not imply a downstream timing Cartesian product.
5. Keep existing recipe dimensions and five warmups/50 measured frames as a
   cheap starting point. Record raw chronology and setup/cold/live observations;
   only make distribution/tail claims supported by the horizon and replication.
   More frames and more independent processes address different uncertainties.
6. Predeclare paired order, gain/regression criteria, the fresh check and a
   modest stopping ceiling. Ambiguity at the ceiling is an inconclusive outcome,
   not a reason to keep sampling until acceptance. Never discard genuine losses.
7. Investigate invalid timing/provenance or confounding separately from ordinary
   variability. Report absolute effects, variability, missing scope and the
   actual basis of acceptance. A non-significant control is not proof of stability.

Standalone A/A, four-run controls, position probes and longer horizons can answer
specific diagnostic questions; they are not universal admission gates. Optional
[formal version-4 confirmation](confirmation.md) retains its declared block,
multiplicity and replay requirements. Do not reinterpret small engineering
checks as formal confidence evidence or migrate historical policies in place.
