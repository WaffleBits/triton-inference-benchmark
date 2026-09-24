# Controlled lifecycle qualification — 2026-09-24

## Primary-source role evidence

The following official Ashby listings were read on 2026-09-24. The pages were
listed as active when read; `datePosted` is the page's structured publication
field. Compensation is copied exactly from the listing and is not a total-
compensation estimate.

| Role | Canonical listing | Published | Compensation | Location / work mode | Repeated requirements relevant here |
|---|---|---:|---|---|---|
| Baseten, Software Engineer - Baseten Inference Stack | https://jobs.ashbyhq.com/baseten/c8701794-bdc1-4932-bffa-a444ce57ed73 | 2026-06-02 | `$180K – $360K • Offers Equity` | San Francisco; Hybrid | Distributed inference orchestration; routing, autoscaling, scheduling, observability; reliability and benchmarking; deployment and monitoring; Kubernetes/GPU workloads |
| OpenAI, Software Engineer, GPT Infrastructure | https://jobs.ashbyhq.com/openai/f3ddd41c-541f-485e-90d6-86c26e018e9f | 2026-04-27 | `$293K – $385K • Offers Equity` | San Francisco; Hybrid; Seattle listed as a secondary location | Long-running workload qualification; retries, checkpointing, budgets, observability; secure execution; correctness/performance evaluation; provenance and regression workflows |
| OpenAI, Software Engineer, Model Runtime | https://jobs.ashbyhq.com/openai/ec317080-e2d2-4a73-93e6-e0a9ae6fdf96 | 2026-08-24 | `$266K – $445K • Offers Equity` | San Francisco; Hybrid | Scheduling, continuous batching, memory/KV-cache management; latency/throughput/utilization; profiling, observability, benchmarking; correctness, reliability, and graceful behavior |
| Perplexity, Member of Technical Staff (AI Inference Engineer) | https://jobs.ashbyhq.com/perplexity/8a976851-9bef-4b07-8d36-567fa9540aef | 2026-04-13 | `$220K – $485K • Offers Equity` | San Francisco; Palo Alto and New York City secondary locations | Rust/Python/CUDA serving; ingress-to-kernel profiling; reliability, dashboards, alerts, automated remediation; Kubernetes/GPU scheduling/autoscaling |

## Evidence map

### Already demonstrated

- Repeatable mock, Triton-style, and OpenAI-compatible serving benchmarks.
- Measured latency, throughput, retry amplification, request-path and service-
  lifecycle counters, trace propagation, privacy-safe JSON/Prometheus artifacts,
  and saved adjacent-run regression reports.
- Authenticated agent coordination with explicit TLS trust, completed-result
  recovery, coordinator restart reconciliation, and conservative clock bounds.
- Public Rust scheduler, Triton kernel correctness/performance work, and
  security/observability controls in the gateway repository.

### Present but buried

- The benchmark has a warmup phase and a service-restart counter gate, but the
  README explicitly says warmup is not a cold-start measurement.
- The design and portfolio notes already identify server-lifecycle hooks as a
  production extension; no CLI currently owns the boundary from controlled
  process launch to readiness, then to a measured benchmark phase.

### Missing

- A reproducible, local-only lifecycle qualification that starts an explicitly
  supplied service command without a shell, waits for an explicit health
  endpoint, runs the existing benchmark, and emits a privacy-safe startup-to-
  readiness measurement with a clear scope label.
- This does not claim model cold-start time, remote orchestration, production
  scale, or server-side readiness beyond the selected health endpoint.

## Exactly one selected gap

**Add controlled local service-lifecycle qualification to the existing benchmark.**

This is one lifecycle wrapper around the existing runner, not a new repository or
another performance metric. It closes the recurring role requirements for
qualification workflows, operational readiness, benchmarking, and actionable
failure boundaries while preserving the repository's existing privacy semantics.

## Implementation plan

1. Add `lifecycle_qualification.py`, a standard-library CLI that launches an
   operator-supplied command with `shell=False`, substitutes one ephemeral loopback
   port, polls an explicit loopback health URL, runs the existing benchmark with
   the same port substitution, and terminates the service deterministically.
2. Persist only a bounded JSON/Markdown projection: measured process-launch to
   HTTP-200 readiness latency, probe count, lifecycle termination status, and the
   existing benchmark's safe headline projection. Hash command and health URL;
   never serialize them, benchmark paths, prompts, credentials, or raw output.
3. Add unit tests first for loopback URL validation, placeholder handling,
   threshold validation, projection privacy, and termination behavior.
4. Add a real CLI fixture with a delayed-readiness local HTTP service and an
   OpenAI-compatible benchmark request. Assert the service receives the expected
   requests, startup latency is measured, and the emitted artifacts contain no
   command, URL, temporary path, prompt, or authorization data.
5. Document the command, operational boundary, and non-claim in `README.md`,
   `docs/OPERATIONS.md`, `DESIGN.md`, and the portfolio review notes. Add the
   fixture to CI.
6. Run the full Python unit suite and all existing CLI qualification fixtures,
   then inspect the diff and run the new CLI fixture again before publication.

## Verification boundary

The implementation proves a real local subprocess launch, health transition, and
benchmark invocation with deterministic fixture behavior. `startup_latency_ms`
is process-launch to the selected HTTP health response; it is not model-weight
load time, accelerator initialization time, service MTTR, a remote-host timing,
or a production SLO. The operator must choose a health endpoint whose semantics
match the lifecycle question and keep workload/model/configuration comparable.
