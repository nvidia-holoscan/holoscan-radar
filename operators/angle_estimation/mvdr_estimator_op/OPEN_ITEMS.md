# MVDREstimatorOp — open items

## Automatic diagonal-loading calibration

Auto diagonal loading uses
`1e-2 * trace(covariance) / num_channels`; the factor remains a first cut.

## Capon reduction performance

Capon reduction remains latency-bound; warp-cooperative or GEMM paths are
deferred.
