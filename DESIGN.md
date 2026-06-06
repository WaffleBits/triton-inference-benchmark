# Design Notes

## Benchmark Model

The benchmark separates the harness from the inference client:

- `MockInferenceClient` provides a deterministic, dependency-free workload for CI.
- `TritonHttpInferenceClient` calls a live Triton-compatible inference server over HTTP.
- `run_benchmark` owns concurrency, retries, timing, and result collection.
- `summarize_results` owns percentile, throughput, and success-rate calculation.

This keeps the core logic testable without requiring a GPU, a running model server, or a CUDA runtime.
Live HTTP workers use thread-local Triton clients because the upstream Python HTTP client is not thread-safe.

## Metrics

The tool reports:

- Success and failure counts.
- Success rate.
- End-to-end duration.
- Throughput in requests per second.
- Average, p50, p95, p99, min, and max latency.

When `--prometheus` is enabled, the same core measurements are written as Prometheus text-format gauges and counters beside the JSON result. This keeps the harness dependency-free while making the output easy to archive in CI, push to a metrics gateway, or ingest into a dashboarding workflow.

When `--telemetry-prometheus` is provided, the benchmark attaches a correlated summary from a Triton/DCGM Prometheus text snapshot. The summary keeps server-side GPU utilization, memory use, queue duration, request duration, and inference-duration counters beside the client-side benchmark result. The parser is dependency-free so CI can validate the behavior with synthetic fixtures while live runs can still consume real scrape artifacts.

## Batch-Invariance Probe

`--batch-invariance-probes` checks whether infrastructure scheduling changes model
outputs. Each fixed synthetic input is sent once in isolation and once in a
concurrent workload mixed with unrelated noise requests. The harness fingerprints
all Triton outputs, compares them exactly, records mismatched sample IDs, and can
fail CI with `--fail-on-batch-variance`.

Exact equality is intentionally strict. It is useful for deterministic serving,
prefix-cache validation, replayable rollouts, and debugging numerical changes
caused by dynamic batching or reduction order. Models that intentionally permit
small floating-point drift will eventually need a model-aware tolerance policy;
the current probe reports that drift as a mismatch instead of hiding it.

## Regression Comparison

The `--baseline` option compares a candidate benchmark run with a saved JSON result. The comparison report focuses on release-relevant signals:

- p95 latency percentage change.
- success-rate delta.
- throughput percentage change.
- explicit regression reasons when thresholds are exceeded.

The default gates mark a run as a regression when p95 latency rises by more than 10% or success rate drops by more than 0.01. `--fail-on-regression` makes that comparison exit non-zero for CI. Those thresholds are intentionally CLI-configurable because production latency budgets differ by model, accelerator, queueing policy, and product surface.

## Why Mock Mode Exists

AI infrastructure repos often fail basic review because they cannot run without specialized hardware. Mock mode makes the benchmark harness reviewable anywhere while live Triton mode remains available for real server testing.

## Production Extensions

- Add warmup windows and separate cold-start metrics.
- Add request payload profiles by model family.
- Add threshold checks for correlated GPU telemetry and queue depth.
- Add distributed load generation across multiple clients.
- Add model-aware numeric tolerance policies for batch-invariance probes.
