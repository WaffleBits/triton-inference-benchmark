# Coordinated multi-client load qualification

## Primary-source role evidence

Checked against the official job feeds on 2026-09-08:

- Anthropic, [Performance Engineer, Inference
  Systems](https://job-boards.greenhouse.io/anthropic/jobs/5224564008):
  feed update `2026-08-21T12:50:15-04:00`; on-site in San Francisco,
  New York City, or Seattle; published annual salary `$350,000 — $850,000
  USD`. The role names fleet-wide throughput, latency, reliability,
  correctness, distributed routing, autoscaling, performance modeling, and
  cross-layer investigations.
- Anthropic, [Staff + Senior Software Engineer, Inference
  Infrastructure](https://job-boards.greenhouse.io/anthropic/jobs/5245851008):
  feed update `2026-09-03T09:29:10-04:00`; on-site in San Francisco,
  New York City, or Seattle; published annual salary `$320,000 — $485,000
  USD`. It asks for distributed serving, orchestration, routing, Kubernetes,
  observability, and performance tuning.
- Together AI, [AI Infrastructure Systems
  Engineer](https://job-boards.greenhouse.io/togetherai/jobs/5138540007):
  feed update `2026-08-04T14:43:48-04:00`; San Francisco; published US base
  salary `$190,000 - $270,000 + equity + benefits`. It emphasizes fleet
  automation, Kubernetes, distributed systems, performance, and reliability.
- Baseten, [Software Engineer - Baseten Inference
  Stack](https://jobs.ashbyhq.com/baseten/c8701794-bdc1-4932-bffa-a444ce57ed73):
  published `2026-06-02T00:16:06.843+00:00`; hybrid in San Francisco;
  compensation is not exposed in the official Ashby feed. It calls for
  distributed inference orchestration, routing, autoscaling, scheduling,
  observability, benchmarking, release automation, and operational ownership.

The recurring requirements relevant to this repository are distributed load
and orchestration, request routing, inspectable observability, performance
analysis, correctness, and fail-closed operational checks. Salary values above
are copied from the feeds rather than inferred.

## Evidence map

- **Already demonstrated:** this repository has real CLI paths for Triton and
  OpenAI-compatible endpoints, retries, open-loop pacing, W3C trace propagation,
  privacy-safe telemetry, request-path accounting, and regression gates. Other
  public WaffleBits repositories separately expose tested GPU kernels, a Rust
  inference scheduler, and an RBAC/observability gateway.
- **Present but buried:** the supervised local process-lifecycle fixture already
  coordinates a router, backend supervisor, and backend, but it qualifies one
  benchmark client and does not reconcile independent load generators.
- **Missing before this change:** multiple independently executing benchmark
  clients with one measured window, complete-client validation, truthful
  aggregate throughput, and explicit rejection of non-mergeable percentile and
  overlapping server-counter claims. Authenticated multi-node agents and
  cross-host clock-quality evidence remain out of scope.

## Evidence-backed gap

Current inference-infrastructure listings repeatedly ask for distributed systems,
request routing, observability, performance analysis, and reliable orchestration.
This repository already measures one client process well, but its roadmap correctly
states that coordinated multi-client load is missing. Adding another repository
would duplicate the benchmark's request, retry, streaming, privacy, and gate logic.

## One selected change

Add a local coordinator for independent benchmark CLI processes and a fail-closed
aggregate artifact. Each child receives a common future start time and writes a
privacy-safe coordination record. The coordinator validates a complete set of
unique client indexes, common run/configuration fingerprints, common planned start,
and bounded actual start skew before reporting aggregate request/retry counts and
successful-completion throughput over the union of the client windows.

## Claim boundary

The checked fixture will be a deterministic, single-host, multi-process run against
a local OpenAI-compatible server. It demonstrates process coordination, artifact
reconciliation, and same-host start-skew gating. It does **not** demonstrate
multi-node load, synchronized clocks across hosts, network fault handling, an
isolated production service, a model, a GPU, or a mergeable global latency
percentile. Per-client latency percentiles will not be averaged or relabeled as a
global percentile.

## TDD and verification

1. Add failing tests for coordination metadata, complete-client validation,
   configuration mismatch rejection, count aggregation, start-skew failure, and
   artifact privacy.
2. Implement child timing metadata plus the coordinator and aggregate Prometheus
   export.
3. Exercise two real benchmark CLI processes against the deterministic local SSE
   fixture and verify trace uniqueness, aggregate counts, start skew, and redaction.
4. Run the complete unit suite, the existing supervised lifecycle fixture, Python
   compilation, Docker build/smoke, dependency audit, and `git diff --check`.

## Local verification recorded on 2026-09-08

- Python 3.9 and Python 3.13 container discovery: 113 tests passed on each.
- Deterministic two-client SSE fixture: eight of eight logical requests
  succeeded, eight distinct valid trace contexts reached the fixture, measured
  start skew was `0.003 ms`, the windows overlapped for `0.15225 s`, and the
  coordination gate passed. These timing values are fixture observations, not
  production or multi-node measurements.
- Existing supervised lifecycle fixture: one controlled backend crash, one
  completed restart, six ingress attempts, five backend receipts, four
  successes, and all request-path/retry/trace/lifecycle gates passed.
- Python compilation and `git diff --check`: passed.
- Docker image `sha256:3463562339434e032ef8caf91ba97f0f0562b5926ac320013a9a77ada897df01`
  built from `python:3.12-slim`; the standard mock CLI and packaged coordinator
  both ran as UID 10001. A three-client mock run also reconciled 18 requests with
  overlapping windows and a passing coordination gate.
- `pip-audit 2.9.0 --strict` against `requirements.txt` in
  `python:3.13-bookworm`: no known vulnerabilities found.
