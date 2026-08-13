# Add a measured-phase open-loop arrival schedule

## Market signal

Reviewed on 2026-08-13 from active official company job-board pages and feeds:

- Anthropic, [Performance Engineer, Inference Systems](https://job-boards.greenhouse.io/anthropic/jobs/5224564008), official board updated 2026-08-03: **$350,000–$850,000 USD annual salary**; San Francisco, New York City, or Seattle. The role asks for cross-layer throughput/latency/reliability analysis, profiling, telemetry, correctness gates, and latency-cost modeling around batch sizing and utilization.
- Anthropic, [Performance Engineer, GPU](https://job-boards.greenhouse.io/anthropic/jobs/4926227008), official board updated 2026-08-03: **$280,000–$850,000 USD annual salary**; San Francisco, New York City, or Seattle. It names GPU optimization, Nsight profiling, distributed communication, NVLink, custom kernels, and production inference performance.
- Anthropic, [Staff+ Site Reliability Engineer, Safeguards ML Infra](https://job-boards.greenhouse.io/anthropic/jobs/5230394008), official board updated 2026-08-10: **$405,000–$485,000 USD annual salary**; remote-friendly with travel or based in San Francisco, Seattle, or New York City. It asks for verified change management, canaries, rollback authority, cloud operation, incident response, automation, Python, and LLM inference familiarity.
- Anthropic, [Software Engineer, Infrastructure, Interpretability](https://job-boards.greenhouse.io/anthropic/jobs/5388612008), official board updated 2026-08-12: **$320,000–$485,000 USD annual salary**; San Francisco or New York City. It asks for Python plus secure, scalable cloud/distributed infrastructure, Kubernetes, infrastructure as code, schedulers, accelerator fleets, and observability.
- Anthropic, [Staff+ Software Engineer, Infrastructure (Distributed Systems)](https://job-boards.greenhouse.io/anthropic/jobs/4970314008), official board updated 2026-08-07: **$320,000–$485,000 USD annual salary**; San Francisco, New York City, or Seattle. It asks for production distributed systems, Kubernetes/cloud infrastructure, operational processes, incident response, Python/Rust/Go/Java, and ML accelerator/networking experience.

The recurring signal is performance and reliability evidence under an explicit workload, with the offered request pattern kept distinct from completion throughput.

## Evidence map

### Already demonstrated

- Concurrent measured and warmup phases with latency percentiles, completion throughput, retries, and success accounting.
- Live Triton and OpenAI-compatible clients plus deterministic real-CLI fixtures.
- Bracketed server telemetry, sampled GPU gauges, regression/correctness gates, and privacy-safe trace-continuation evidence.
- Explicit workload-shape and cost assumptions that do not masquerade as measurements.

### Present but buried

- The phase runner already centralizes request submission and caps active workers with `ThreadPoolExecutor`.
- The artifact already distinguishes warmup from measured work and can carry an additional bounded schedule record without storing prompts, responses, credentials, or trace identifiers.

### Missing

- Requests are currently submitted as fast as the executor accepts them. Concurrency is configurable, but offered request rate is not.
- A reviewer cannot qualify latency at a stated arrival rate or distinguish an open-loop arrival target from achieved completion throughput.

## Decision

Extend this repository rather than create another project. Arrival scheduling belongs beside its existing concurrency, phase, artifact, telemetry, and regression semantics.

Implement exactly one coherent capability: **an optional open-loop constant-rate submission schedule for the measured request phase**.

## Implementation

1. Add `--request-rate-rps`; zero preserves the existing immediate-submission behavior, while a positive value schedules measured requests at monotonic constant-rate deadlines.
2. Keep warmup immediate and separately reported. The configured rate applies only to measured submissions.
3. Preserve `--concurrency` as the maximum number of active client workers. If service time exceeds capacity, executor work can queue; submissions remain independent of completions.
4. Report configured rate, scheduled/observed submission and request-start spans, achieved submission rate, submission lag, executor queue delay, and request-start lag separately from successful-completion throughput.
5. Use monotonic timing and validate the option as non-negative.
6. Add deterministic schedule/unit tests, a real-time phase integration test, CLI parsing validation, artifact/Prometheus tests, and exercise the real OpenAI-compatible fixture CLI with pacing enabled.

## Verification gates

- Full `unittest` discovery and Python bytecode compilation.
- Real paced OpenAI-compatible CLI run against the deterministic HTTP/SSE fixture.
- Existing bracketed-telemetry CLI fixture and fail-closed gates.
- Docker image build and containerized paced mock execution.
- Supported Python container dependency audit.
- Artifact privacy inspection and `git diff --check`.
- Pull-request CI and post-merge main CI.

## Claim boundary

This feature controls client-side submission deadlines within one process. The artifact reports scheduler lag and observed submission rate; it does not claim exact server arrival times, synchronized clocks, distributed load generation, queue isolation, production scale, or a stable service capacity. Successful-completion throughput remains a separate measurement.
