# Portfolio Review Notes

This project is intentionally designed as a public-safe AI infrastructure benchmarking artifact.

## What To Review

- `benchmark.py`: CLI, benchmark execution, concurrency, retries, and summary metrics.
- `tests/`: validation for metrics, success accounting, and benchmark behavior.
- `DESIGN.md`: benchmark model, tradeoffs, and production extension plan.
- `docs/OPERATIONS.md`: regression triage, Prometheus artifact usage, and SLO-oriented review notes.
- `sample_results/mock_telemetry_before.prom` and `mock_telemetry.prom`:
  synthetic paired Triton/DCGM snapshots for counter-window review.
- `deploy/kubernetes/benchmark-job.yaml`: cluster-local benchmark execution shape.

## What This Demonstrates

- Repeatable model-serving benchmarks instead of one-off timing claims.
- Latency percentile reporting for p50, p95, and p99.
- Throughput and success-rate accounting under configurable concurrency.
- Single-process open-loop constant-rate request pacing with measured submission
  lag kept separate from successful-completion throughput.
- A CI-friendly mock mode that keeps the repo reviewable without GPU hardware.
- Phase-separated warmup evidence that is excluded from measured latency,
  throughput, regression, streaming-token, and cost results.
- A live Triton HTTP path and an OpenAI-compatible streaming path for authorized model-serving infrastructure.
- Measured streaming TTFT, inter-chunk latency, output bytes, and server-reported token throughput with explicit usage coverage.
- Opt-in W3C trace-context propagation on both live HTTP clients, with a real
  local SSE fixture proving header wiring and identifier-free artifacts.
- Prometheus-compatible benchmark artifacts for dashboard and CI ingestion.
- Baseline-versus-candidate comparison with explicit regression reasons.
- Correlated server telemetry for GPU utilization, memory pressure, queue time, and Triton counters.
- Fail-closed server failure-rate and queue-fraction gates derived from paired
  cumulative counters, with aggregate-decrease, missing-family, and hashed
  series-membership validation.
- Opt-in HTTP(S) telemetry capture bracketed around the measured request phase,
  with explicit authentication and artifact privacy boundaries.
- Sampled GPU gauge windows that reject target churn while persisting only a
  series-membership hash and count rather than raw Prometheus labels.
- Batch-invariance testing under concurrent noise traffic, with exact comparison
  by default and privacy-safe run-scoped numeric tolerance gates.
- Token-throughput, GPU-capacity, energy, and normalized cost-to-serve estimates with explicit assumptions.
- Kubernetes job posture with non-root runtime settings.

## Technical Scope

- AI infrastructure: model-serving reliability, benchmark methodology, latency analysis, and regression tracking.
- Platform engineering: CLI ergonomics, JSON outputs, testable boundaries, and live-service extension points.
- Performance engineering: concurrency sweeps, percentile metrics, retry behavior, request trace continuity, token throughput, cost modeling, and reproducible reports.
- Infrastructure/SRE: operational runbooks, release regression thresholds, deterministic serving checks, Prometheus output, telemetry correlation, and Kubernetes execution shape.

## Gaps Worth Closing Next

- Add controlled server-lifecycle hooks for defensible cold-start measurements.
- Add coordinated distributed load generation across multiple clients.
- Add saved benchmark reports with trend comparisons over time.
