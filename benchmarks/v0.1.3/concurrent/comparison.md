# MLX inference benchmark

Started 2026-10-10T06:57:06.572864+00:00; model revision: a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3.

| Path | Mode | Clients | Completed/attempted | First text p50 ms | Latency p50 / p95 ms | Output tokens/s | Success req/s | Fail / timeout |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| direct | streaming | 2 | 30/60 | 28.63 | 504.90 / 510.84 | 247.81 | 2.59 | 50.0% / 0.0% |
| direct | nonstreaming | 2 | 30/60 | n/a | 504.85 / 508.87 | 248.28 | 2.60 | 50.0% / 0.0% |
| provider | streaming | 2 | 30/60 | 30.12 | 507.34 / 509.67 | 247.04 | 2.58 | 50.0% / 0.0% |
| provider | nonstreaming | 2 | 30/60 | n/a | 509.62 / 514.56 | 244.96 | 2.56 | 50.0% / 0.0% |
| coordinator | streaming | 2 | 30/60 | 31.73 | 508.79 / 516.11 | 244.25 | 2.55 | 50.0% / 0.0% |
| coordinator | nonstreaming | 2 | 30/60 | n/a | 515.68 / 524.90 | 241.62 | 2.53 | 50.0% / 0.0% |

Output equivalence: True.
Validation problems: [].

Tokens/s uses actual completion tokens divided by measured workload wall time. Success latency excludes failures; throughput includes their elapsed time. First text is observable only for streaming. Warmups and startup are excluded.

Percentiles use linear interpolation. These are descriptive local estimates; even 30 successes provide few p95 tail observations. Concurrency >1 measures admission/rejection with the existing single MLX slot, not GPU scaling.

See results.json for every request, startup and first-request timings, boundary RSS samples, environment, workload, and cleanup evidence. RSS is not GPU allocation or peak memory. See docs/benchmarking.md for definitions.
