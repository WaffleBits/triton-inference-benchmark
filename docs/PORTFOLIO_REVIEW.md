# Portfolio Review Notes

This project is intentionally designed as a public-safe AI infrastructure benchmarking artifact.

## What To Review

- `benchmark.py`: CLI, benchmark execution, concurrency, retries, and summary metrics.
- `tests/`: validation for metrics, success accounting, and benchmark behavior.
- `DESIGN.md`: benchmark model, tradeoffs, and production extension plan.
- `docs/OPERATIONS.md`: regression triage, Prometheus artifact usage, and SLO-oriented review notes.
- `deploy/kubernetes/benchmark-job.yaml`: cluster-local benchmark execution shape.

## What This Demonstrates

- Repeatable model-serving benchmarks instead of one-off timing claims.
- Latency percentile reporting for p50, p95, and p99.
- Throughput and success-rate accounting under configurable concurrency.
- A CI-friendly mock mode that keeps the repo reviewable without GPU hardware.
- A live HTTP path that can be connected to real model-serving infrastructure.
- Prometheus-compatible benchmark artifacts for dashboard and CI ingestion.
- Baseline-versus-candidate comparison with explicit regression reasons.
- Kubernetes job posture with non-root runtime settings.

## Technical Scope

- AI infrastructure: model-serving reliability, benchmark methodology, latency analysis, and regression tracking.
- Platform engineering: CLI ergonomics, JSON outputs, testable boundaries, and live-service extension points.
- Performance engineering: concurrency sweeps, percentile metrics, retry behavior, and reproducible reports.
- Infrastructure/SRE: operational runbooks, release regression thresholds, Prometheus output, and Kubernetes execution shape.

## Gaps Worth Closing Next

- Add warmup, cold-start, and steady-state separation.
- Add distributed load generation across multiple clients.
- Add saved benchmark reports with trend comparisons over time.
- Add server-side telemetry correlation for GPU utilization, queue depth, and batching behavior.
