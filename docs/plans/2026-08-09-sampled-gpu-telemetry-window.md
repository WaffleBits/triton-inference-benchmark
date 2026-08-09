# Sample GPU gauges across the measured request window

## Market signal

Reviewed on 2026-08-09 from active official job-board feeds:

- Anthropic, Performance Engineer, Inference Systems: cross-layer throughput,
  latency, reliability, correctness, cost, observability, telemetry, and GPU
  performance analysis.
- Anthropic, Performance Engineer, GPU: GPU utilization, profiling, memory
  bandwidth, distributed systems, and production serving performance.
- OpenAI, Software Engineer, Model Inference: visibility into bottlenecks and
  instability across high-volume, low-latency distributed inference.
- OpenAI, Software Engineer, GPU Infrastructure - HPC: Prometheus/Grafana,
  noisy-data analysis, fleet health, performance, and automated remediation.

The existing benchmark already brackets cumulative Triton counters around the
measured request phase. Its DCGM values are only post-run point samples, so they
cannot describe accelerator behavior across that phase.

## Decision

Extend this repository rather than create another project. It already owns the
load phase, opt-in Prometheus client, DCGM parser, artifacts, and release gates.
The addition should preserve the existing before/after counter semantics while
adding an explicitly opt-in sampled gauge window.

## Implementation

1. Add `--telemetry-sample-interval-seconds`, valid only with
   `--telemetry-url` and disabled by default.
2. Start a bounded sampler after measured requests have been submitted, collect
   Prometheus snapshots at the configured interval while requests are in
   flight, stop it when the measured phase completes, and propagate any scrape
   failure rather than publishing partial qualification evidence.
3. Aggregate known DCGM GPU utilization, memory-copy utilization, and memory-used
   gauges across the pre-boundary, in-window, and post-boundary scrapes. Record
   scrape/value coverage plus sample average, p50, p95, min, and max.
4. Label the result as sampled, not time-weighted. Preserve the limitation that
   a shared endpoint can include unrelated traffic and changing scrape targets.
5. Export the sampled window to Prometheus without serializing endpoint URLs,
   bearer tokens, authorization headers, environment-variable names, or raw
   scrape text.
6. Exercise the real CLI against the deterministic local HTTP fixture and keep
   the existing counter gates passing.

## Verification gates

- Unit tests for phase sampling, statistics, failed-scrape propagation, CLI
  validation, Prometheus export, and artifact privacy.
- Full `unittest` discovery and Python bytecode compilation.
- Real CLI run against `tests/telemetry_fixture_server.py` with multiple
  in-window scrapes and passing counter gates.
- Container build/run using the repository Dockerfile, dependency audit, and
  `git diff --check`.
- Pull-request CI and public main-branch CI must pass before profile or resume
  wording changes.

## Claim boundary

The output is a sample distribution over bounded scrapes, not a time-weighted
integral and not hardware-counter profiling. It does not prove stable target
membership, isolate unrelated traffic, or turn the deterministic fixture into a
GPU/model/fleet measurement.
