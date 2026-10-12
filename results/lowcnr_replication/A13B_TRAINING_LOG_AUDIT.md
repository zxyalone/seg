# A13b training log audit

The original logs contain epoch durations, total training loss, validation loss and explicit non-finite-gradient/loss update skips. They do not contain separately recorded epoch consistency loss; no historical component values are synthesized.

| Run | Epochs | Mean time (s) | Median time (s) | Gradient skips | Loss skips |
|---|---:|---:|---:|---:|---:|
| A13b_s0 | 200 | 93.19 | 93.31 | 18 | 0 |
| A13b_s1 | 200 | 93.37 | 93.87 | 17 | 0 |
| A13b_s2 | 200 | 93.17 | 93.55 | 18 | 0 |

The preflight established 41 positive consistency-loss calls in 50 synthetic AMP steps. That is preflight evidence, not the consistency-loss trajectory during the completed real-data training.
