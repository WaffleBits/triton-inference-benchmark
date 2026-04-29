# Design Notes

## Benchmark Model

The benchmark separates the harness from the inference client:

- `MockInferenceClient` provides a deterministic, dependency-free workload for CI.
- `TritonHttpInferenceClient` calls a live NVIDIA Triton Inference Server over HTTP.
- `run_benchmark` owns concurrency, retries, timing, and result collection.
- `summarize_results` owns percentile, throughput, and success-rate calculation.

This keeps the core logic testable without requiring a GPU, a running model server, or a CUDA runtime.

## Metrics

The tool reports:

- Success and failure counts.
- Success rate.
- End-to-end duration.
- Throughput in requests per second.
- Average, p50, p95, p99, min, and max latency.

## Why Mock Mode Exists

AI infrastructure repos often fail basic review because they cannot run without specialized hardware. Mock mode makes the benchmark harness reviewable anywhere while live Triton mode remains available for real server testing.

## Production Extensions

- Add warmup windows and separate cold-start metrics.
- Add request payload profiles by model family.
- Add Prometheus export for benchmark runs.
- Add comparison mode for baseline versus candidate model versions.
- Add GPU telemetry capture through DCGM.
- Add distributed load generation across multiple clients.

