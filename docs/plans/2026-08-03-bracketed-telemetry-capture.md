# Harness-bracketed telemetry capture plan

Date: 2026-08-03

## Primary-source role evidence

The official Greenhouse feeds were read at 2026-08-03T11:04:56Z. All four
records were active in their company feeds. Compensation and feed timestamps
below are copied from those records rather than inferred.

| Company | Role | Published / updated | Published compensation | Location / work model | Relevant requirements |
|---|---|---|---|---|---|
| Anthropic | [Performance Engineer, Inference Systems](https://job-boards.greenhouse.io/anthropic/jobs/5224564008) | Published 2026-05-20; updated 2026-07-14 | Annual Salary: `$350,000—$850,000 USD` | San Francisco, New York City, or Seattle; hybrid, at least 25% in office | Cross-layer inference investigation; throughput, latency, reliability, correctness, cost, telemetry, dashboards, Python, and release criteria |
| Anthropic | [Performance Engineer, GPU](https://job-boards.greenhouse.io/anthropic/jobs/4926227008) | Published 2025-09-22; updated 2026-07-14 | Annual Salary: `$280,000—$850,000 USD` | San Francisco, New York City, or Seattle; hybrid, at least 25% in office | GPU optimization, CUDA/Triton/CUTLASS, distributed communication, profiling, and production ML performance |
| Anthropic | [Staff+ Software Engineer, Safeguards ML Infrastructure](https://job-boards.greenhouse.io/anthropic/jobs/4778843008) | Published 2025-06-24; updated 2026-07-14 | Annual Salary: `$320,000—$485,000 USD` | San Francisco; hybrid, at least 25% in office | Production distributed systems, SLOs, observability, alerting, incident response, Python, and optional Rust |
| SpaceXAI | [Software Engineer - Training/Inference (C++)](https://job-boards.greenhouse.io/xai/jobs/4533894007) | Published 2024-10-04; updated 2026-07-28 | `$180,000 - $440,000 USD` | Palo Alto, California; remote status not exposed | C++/Rust, distributed model serving, load balancing, autoscaling, continuous batching, GPU kernels, benchmarking, testing, and reliability |

The recurring portfolio signal is not another isolated mock latency number. It
is a reproducible qualification window that connects client load to authorized
server telemetry and can support a fail-closed release decision.

## Evidence map

### Already demonstrated

- Concurrent Triton and OpenAI-compatible request paths with phase-separated
  warmup and measured requests.
- Client latency, throughput, success, streaming, cost, and correctness gates.
- Parsing of Triton/DCGM Prometheus text plus model-filtered before/after counter
  deltas and fail-closed failure-rate and queue-fraction gates.
- Public-safe artifacts that omit prompts, raw scrapes, credentials, and source
  paths.

### Present but buried

- The benchmark runner already owns the exact warmup-to-measurement transition,
  but telemetry files are attached only after the run. An external operator has
to line up both snapshots with the measured phase.

### Missing before this change

- No opt-in HTTP scrape path captures server counters immediately before and
  after the measured request phase.
- File-based windows must remain labeled `operator_supplied_unverified`, even
  when an operator intended to bracket the benchmark.
- There is no test proving that warmup completes before the first scrape, or
  that endpoint URLs, bearer tokens, and raw scrape text stay out of artifacts.

## Existing-repository decision

Extend `triton-inference-benchmark`; do not create a repository. The gap belongs
inside the runner that controls request phase ordering and already owns the
Prometheus parser, telemetry summaries, release gate, CLI, deterministic
fixtures, and CI workflow.

## Implementation

1. Add an HTTP(S)-only Prometheus snapshot client with a response-size limit,
   timeout, and optional bearer token read only from the environment variable
   explicitly named by the operator.
2. Add mutually exclusive CLI support for `--telemetry-url` versus existing
   snapshot files. Do not use ambient API keys and do not serialize the URL,
   environment-variable name, authorization header, or raw response.
3. Capture one scrape after warmup and immediately before measured requests,
   then one immediately after measured requests complete.
4. Reuse the existing model-filtered counter-window and gate logic, but label
   this alignment `harness_bracketed_measured_phase`. State that unrelated
   server traffic and scrape-target membership are still external controls.
5. Preserve file-based telemetry behavior and its unverified alignment label.

## Verification gates

- Tests fail before implementation and pass after it.
- Unit tests cover phase order, missing/explicit authentication, URL validation,
  response bounds, artifact privacy, and legacy file behavior.
- A real CLI run scrapes a deterministic local HTTP fixture twice and produces
  a passing JSON/Prometheus telemetry gate with the new alignment label.
- A threshold-failing CLI run returns exit status 4.
- Canonical unit tests, Python compilation, Docker Python 3.12 build/run,
  dependency audit, artifact inspection, and `git diff --check` pass.

## Claim boundary

Harness bracketing proves only the order of the two HTTP scrape calls relative
to this process's warmup and measured request phases. It does not isolate the
server from unrelated traffic, prove stable per-replica series membership, turn
post-run DCGM gauges into window averages, or represent the synthetic local
fixture as a model or GPU measurement.
