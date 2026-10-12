# MLX inference benchmark

Started 2026-10-10T06:55:35.856491+00:00; model revision: a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3.

| Path | Mode | Clients | Completed/attempted | First text p50 ms | Latency p50 / p95 ms | Output tokens/s | Success req/s | Fail / timeout |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| coordinator | streaming | 1 | 30/30 | 30.38 | 507.02 / 508.88 | 247.17 | 2.58 | 0.0% / 0.0% |
| coordinator | nonstreaming | 1 | 30/30 | n/a | 507.47 / 513.32 | 246.54 | 2.58 | 0.0% / 0.0% |
| provider | streaming | 1 | 30/30 | 29.31 | 505.33 / 509.06 | 247.93 | 2.59 | 0.0% / 0.0% |
| provider | nonstreaming | 1 | 30/30 | n/a | 505.80 / 509.23 | 247.19 | 2.58 | 0.0% / 0.0% |
| direct | streaming | 1 | 30/30 | 28.28 | 504.97 / 506.99 | 248.63 | 2.60 | 0.0% / 0.0% |
| direct | nonstreaming | 1 | 30/30 | n/a | 504.69 / 507.06 | 248.68 | 2.60 | 0.0% / 0.0% |

Output equivalence: True.
Validation problems: [].

Tokens/s uses actual completion tokens divided by measured workload wall time. Success latency excludes failures; throughput includes their elapsed time. First text is observable only for streaming. Warmups and startup are excluded.

Percentiles use linear interpolation. These are descriptive local estimates; even 30 successes provide few p95 tail observations. Concurrency >1 measures admission/rejection with the existing single MLX slot, not GPU scaling.

See results.json for every request, startup and first-request timings, boundary RSS samples, environment, workload, and cleanup evidence. RSS is not GPU allocation or peak memory. See docs/benchmarking.md for definitions.
