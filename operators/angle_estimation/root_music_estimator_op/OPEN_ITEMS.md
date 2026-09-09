# RootMUSICEstimatorOp — open items

## Batched polynomial root solver

Serial per-item `cusolverDnXgeev` dominates latency. A batched or
device-resident replacement remains postponed by team decision.
