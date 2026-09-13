# Idempotent agent-result recovery

## Primary-source role evidence

Checked live through the companies' official Greenhouse job-board APIs on
2026-09-13. `first_published` and `updated_at` dates below come from each per-job
API response. Compensation is copied from the posting; equity or total
compensation is not inferred.

| Role | Canonical listing | First published | Feed updated | Published compensation | Location / constraints | Relevant requirements |
|---|---|---:|---:|---|---|---|
| Anthropic, Software Engineer, Research Infrastructure | https://job-boards.greenhouse.io/anthropic/jobs/5283063008 | 2026-07-06 | 2026-09-09 | `$405,000—$625,000 USD` annual salary | San Francisco or New York City; at least 25% office time | Build and operate distributed research infrastructure; reliability, scalability, cloud infrastructure, infrastructure-as-code |
| Anthropic, Staff + Senior Software Engineer, Inference Infrastructure | https://job-boards.greenhouse.io/anthropic/jobs/5245851008 | 2026-06-08 | 2026-09-03 | `$320,000—$485,000 USD` annual salary | San Francisco, New York City, or Seattle; at least 25% office time | Distributed serving, routing, orchestration, deployment, Kubernetes/cloud, observability, Python or Rust |
| Anthropic, Performance Engineer, Inference Systems | https://job-boards.greenhouse.io/anthropic/jobs/5224564008 | 2026-05-20 | 2026-08-21 | `$350,000—$850,000 USD` annual salary | San Francisco, New York City, or Seattle; at least 25% office time | Cross-layer throughput, latency, reliability and correctness; profiling, telemetry, routing, autoscaling, tail latency |
| Together AI, AI Infrastructure Systems Engineer | https://job-boards.greenhouse.io/togetherai/jobs/5138540007 | 2026-05-14 | 2026-09-09 | `$190,000 - $270,000 + equity + benefits` US base salary | San Francisco; no remote option exposed | Fleet automation, diagnosis and remediation, distributed systems, Python/Go/Rust, Linux, Kubernetes, Terraform or Ansible |

All four per-job APIs returned active listings. None exposed an application
deadline. These are senior or staff-scale production responsibilities; they are
market evidence, not a claim that this repository demonstrates their level or
fleet scale.

Recurring requirements are distributed infrastructure, reliability under partial
failure, measurable performance/correctness, operational telemetry, and Python
systems work. Three listings explicitly join performance or scale with
reliability; two explicitly require infrastructure automation or orchestration.

## Live public evidence inventory

GitHub's live API returned 21 public repositories and six pinned repositories on
2026-09-13. The pins were `market-microstructure-engine`,
`readiness-control-tower`, `secure-gpu-inference-gateway`, `triton-kernel-lab`,
`deterministic-inference-scheduler`, and `heterocore-compiler`.

Strongest relevant public default-branch evidence inspected:

- `triton-inference-benchmark` at `d1895be`: authenticated agents, sampled clock
  bounds, open-loop load, trace/retry/path/lifecycle gates, privacy-safe
  aggregates, and passing default-branch CI.
- `secure-gpu-inference-gateway` at `dd3ff63`: authenticated policy and budget
  controls, Prometheus/OpenTelemetry, deployment posture checks, SBOM and
  vulnerability gates, with both default-branch workflows passing.
- `triton-kernel-lab` at `fbb930c`: correctness-gated Triton kernels and raw RTX
  5070 Ti measurements, with default-branch CI passing.
- `deterministic-inference-scheduler` at `ca65b7b`: Rust continuous batching,
  paged KV-cache accounting, deterministic replay and release gates, with
  default-branch CI passing.
- `market-microstructure-engine` at `951d124`: C++20/Python matching engines,
  parity tests and bounded local measurements, with default-branch CI passing.

## Evidence map

### Already demonstrated

- Authenticated coordinator-to-agent execution with opt-in credentials and a
  minimal child environment.
- NTP-style clock sampling and conservative skew, overlap and throughput bounds.
- Replay rejection by hashed run/client identity.
- Real CLI fixtures spanning agent services, benchmark child processes and a
  synthetic SSE target.

### Present but buried

- A run/client identity already acts as a natural idempotency key.
- The agent validates a complete bounded request before child launch.
- The aggregate builder already needs only request counts, retry data, load
  scheduling and coordination windows. The agent can project to those fields
  before returning or caching a child result, rather than retaining child
  configuration, URLs, prompts, outputs or filesystem paths.

### Missing before this change

- If an agent completes a child but its HTTP response is lost, the coordinator
  cannot distinguish failure from completion. Retrying receives a replay conflict,
  so the completed shard is unavailable and the coordinated run fails.
- The public fixture does not inject an ambiguous response loss or prove that a
  retry retrieves the first execution without running the child twice.

## Exactly one selected gap

**Recover a completed agent result after an ambiguous response loss without
re-executing the benchmark child.**

This belongs in `triton-inference-benchmark`: the existing agent owns the
run/client identity, child execution and artifact boundary, while the coordinator
owns transport and aggregate semantics. A new repository would duplicate the
protocol and would not exercise the real benchmark CLI.

## Intended implementation

1. Replace replay-only bookkeeping with a bounded in-memory run record keyed by
   the existing hashed run/client identity. Store only a canonical request
   fingerprint and a coordinator-only completed-artifact projection; do not
   retain raw run IDs, benchmark arguments, target URLs, prompts, outputs,
   configuration or credentials.
2. Return the cached artifact for an identical completed request. Reject the same
   identity with a different request fingerprint, an unfinished prior request, or
   a completed result whose bounded cache entry expired. Never re-execute those
   cases.
3. Add an explicit bounded coordinator option for transport recovery attempts.
   Retry only ambiguous transport/response failures, not explicit HTTP
   rejections. Validate the agent identity on every response.
4. Record aggregate counts for transport retries, cached results and results
   recovered after a transport failure. Export only scalar Prometheus counters;
   keep agent URLs, request fingerprints and authorization data out of artifacts.
5. Extend the real two-agent CLI fixture with a proxy that forwards one run to an
   agent, drops its successful response, then allows the retry. Assert eight total
   target requests, not twelve, proving the affected child was not run twice.

## TDD and verification

1. Add failing unit tests for completed-result replay, payload-conflict rejection,
   bounded/expired records, transport retry metadata and aggregate redaction.
2. Implement the bounded server record/cache and coordinator retry semantics.
3. Run the complete unit suite and the fault-injected two-agent fixture, then rerun
   the same-host coordinator and supervised lifecycle fixtures.
4. Run Python compilation, the clean Python 3.13 container suite, Docker image
   build and packaged smoke tests, strict dependency audit and `git diff --check`.
5. Publish by pull request only after local checks pass. Require pull-request and
   post-merge `main` CI, then read the implementation and wording from public
   `main` before changing profile or resume claims.

## Claim boundary

The deterministic fixture will prove one bounded case: an authenticated agent
completed a child, an intermediary discarded the successful HTTP response, and a
coordinator retry retrieved the in-memory cached artifact without a second child
execution. It will not prove recovery from agent or coordinator process loss,
persistent deduplication, arbitrary packet loss, multi-host behavior, a production
network, a real model/GPU, traffic isolation or fleet scale.
