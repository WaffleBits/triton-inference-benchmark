# Authenticated remote-agent benchmark coordination

## Primary-source role evidence

Checked live through the companies' official job-board feeds on 2026-09-12.
Greenhouse exposes both `first_published` and `updated_at`; compensation below is
copied exactly from each posting and does not include inferred equity value.

| Role | Canonical listing | First published | Feed updated | Published compensation | Location / constraints | Relevant requirements |
|---|---|---:|---:|---|---|---|
| Anthropic, Staff + Senior Software Engineer, Inference Infrastructure | https://job-boards.greenhouse.io/anthropic/jobs/5245851008 | 2026-06-08 | 2026-09-03 | `$320,000—$485,000 USD` annual salary | San Francisco, New York City, or Seattle; on-site metadata and at least 25% office time | Distributed serving, routing, load balancing, orchestration, deployment, Kubernetes/cloud, observability, Python or Rust |
| Anthropic, Performance Engineer, Inference Systems | https://job-boards.greenhouse.io/anthropic/jobs/5224564008 | 2026-05-20 | 2026-08-21 | `$350,000—$850,000 USD` annual salary | San Francisco, New York City, or Seattle; on-site metadata and at least 25% office time | Cross-layer throughput/latency/reliability/correctness work, profiling, Python, telemetry analysis, routing, autoscaling, tail latency |
| Anthropic, Software Engineer, Research Infrastructure | https://job-boards.greenhouse.io/anthropic/jobs/5283063008 | 2026-07-06 | 2026-09-09 | `$405,000—$625,000 USD` annual salary | San Francisco or New York City; on-site metadata and at least 25% office time | Design and operation of large-scale distributed infrastructure, ML research workflows, reliability, cloud infrastructure, infrastructure-as-code |
| Together AI, AI Infrastructure Systems Engineer | https://job-boards.greenhouse.io/togetherai/jobs/5138540007 | 2026-05-14 | 2026-09-09 | `$190,000 - $270,000 + equity + benefits` US base salary | San Francisco; no remote option exposed in the feed | Fleet automation, diagnosis and remediation, distributed systems, Python/Go/Rust, Linux, Kubernetes, Terraform or Ansible, performance and reliability |

The listings were active when their per-job official APIs returned them. The
Anthropic postings expose no deadline and say applications are reviewed on a
rolling basis. Together AI exposes no deadline.

## Live public evidence inventory

GitHub's live APIs returned 21 public repositories and six pins on 2026-09-12.
The pins were `market-microstructure-engine`, `readiness-control-tower`,
`secure-gpu-inference-gateway`, `triton-kernel-lab`,
`deterministic-inference-scheduler`, and `heterocore-compiler`.

Strongest relevant evidence inspected from public default branches:

- `triton-inference-benchmark`: real Triton/OpenAI client paths, open-loop pacing,
  retry/path/lifecycle/trace gates, privacy-safe artifacts, and same-host
  multi-process coordination; public `main` CI passed at `86f5326`.
- `secure-gpu-inference-gateway`: authenticated policy and budget controls,
  Prometheus/OpenTelemetry, deployment posture checks, SBOM, and vulnerability
  gates; both public default-branch workflows passed at `dd3ff63`.
- `triton-kernel-lab`: correctness-gated Triton kernels and raw RTX 5070 Ti
  measurements with explicit hardware-counter limits; public CI passed at
  `fbb930c`.
- `deterministic-inference-scheduler`: Rust continuous batching, paged KV-cache
  accounting, deterministic replay, and promote/hold/rollback gates; public CI
  passed at `ca65b7b`.
- `market-microstructure-engine`: C++20/Python price-time matching with parity
  checks and bounded local measurements; public CI passed at `951d124`.

## Evidence map

### Already demonstrated

- Inspectable model-serving benchmarks, streaming timing, retries, trace
  continuity, request-path accounting, lifecycle qualification, and regression
  gates.
- Independent benchmark processes coordinated against one planned start on one
  host, with complete-shard/configuration checks and privacy-safe aggregation.
- Separate public GPU-kernel correctness/measurement, Rust scheduler, and
  model-access control-plane evidence.

### Present but buried

- The coordinator already has the shard reconciliation and fail-closed window
  semantics needed after remote execution.
- Child artifacts already hash the run/configuration identity and omit raw run
  IDs, prompts, outputs, trace IDs, and authorization headers.

### Missing before this change

- An authenticated coordinator-to-agent protocol.
- Explicit cross-clock offset and uncertainty sampling before a coordinated run.
- Conservative start-skew, overlap, and throughput bounds that account for clock
  uncertainty.
- A real CLI fixture that exercises separate agent services. Actual multi-host,
  production-network, model/GPU, and fleet evidence remain unavailable.

## Exactly one selected gap

**Add authenticated remote-agent coordination with explicit clock-quality gates.**

This is one coherent extension of the existing coordinator, not a new system or
repository. The benchmark already owns request execution, shard semantics,
privacy controls, and aggregation; creating another repository would duplicate
those controls and weaken the evidence chain.

## Intended implementation

1. Add a dependency-free benchmark-agent HTTP service. It accepts only
   authenticated clock probes and benchmark jobs, invokes the existing CLI, uses
   a bounded request body, rejects replayed run/client identities, gives children
   a minimal explicitly allowed environment, and never returns its raw agent ID.
2. Require coordinator callers to opt into a named API-key environment variable.
   Permit cleartext HTTP only for loopback fixtures; require HTTPS for non-loopback
   agents, and refuse redirects so bearer credentials remain pinned to the selected
   origin. Never serialize keys, raw challenges, agent URLs, raw agent IDs, child
   paths, endpoints, prompts, outputs, or trace IDs into aggregates.
3. Estimate agent-minus-coordinator clock offset with multiple NTP-style
   request/response exchanges, select the minimum-network-delay sample, and retain
   only offset, delay, uncertainty, sample count, and hashed agent identity.
4. Convert each agent window into the coordinator clock domain. Gate on clock
   uncertainty plus conservative start-skew and overlap bounds. Report observed
   normalized throughput separately from a conservative lower bound; do not
   relabel either as a synchronized hardware measurement.
5. Exercise two independently running loopback agents and two real benchmark CLI
   children against the deterministic SSE fixture. Keep the claim limited to
   authenticated protocol wiring and cross-clock accounting over loopback.

## TDD and verification

1. Add failing tests for URL/auth transport policy, clock-sample selection,
   remote-shard normalization, conservative bounds, unique agent identities, and
   artifact redaction.
2. Implement the service/client protocol and remote coordinator mode.
3. Run the complete unit suite and real two-agent fixture, then rerun the existing
   same-host and supervised lifecycle fixtures.
4. Run Python compilation, a clean Python 3.13 container suite, Docker image
   build plus packaged remote-agent smoke, strict dependency audit, and
   `git diff --check`.
5. Publish by PR only after local checks pass; require feature-branch and
   post-merge default-branch CI, then read the implementation and claims back from
   public `main` before changing profile or resume wording.

## Claim boundary

The deterministic acceptance fixture runs two authenticated HTTP agent services,
two benchmark child processes, and one synthetic SSE target on a single CI host.
It can prove protocol authentication, opt-in credential selection, replay
rejection, clock-sample handling, conservative time bounds, shard reconciliation,
and aggregate redaction. It cannot prove separate machines, Internet or datacenter
network behavior, production trust, synchronized hardware clocks, a real model or
GPU, fleet orchestration, or production scale.
