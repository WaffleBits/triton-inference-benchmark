# Durable agent-result recovery across process restart

## Primary-source role evidence

Checked through Anthropic's official Greenhouse per-job API on 2026-09-14.
`first_published` and `updated_at` are feed fields. Compensation is copied from
each posting; equity and total compensation are not inferred.

| Role | Canonical listing | First published | Feed updated | Published compensation | Location / constraints | Relevant requirements |
|---|---|---:|---:|---|---|---|
| Anthropic, Performance Engineer, Inference Engine | https://job-boards.greenhouse.io/anthropic/jobs/5418323008 | 2026-09-09 | 2026-09-09 | `$350,000-$850,000 USD` annual salary | San Francisco or New York City; at least 25% office time | Improve inference throughput, cost, reliability and latency; reuse cached state instead of recomputing; measure before changing; strong tested systems code |
| Anthropic, Software Engineer, Research Infrastructure | https://job-boards.greenhouse.io/anthropic/jobs/5283063008 | 2026-07-06 | 2026-09-09 | `$405,000-$625,000 USD` annual salary | San Francisco or New York City; at least 25% office time | Design, build and operate distributed research infrastructure; own reliability and scalability; cloud and infrastructure-as-code |
| Anthropic, Staff + Senior Software Engineer, Inference Infrastructure | https://job-boards.greenhouse.io/anthropic/jobs/5245851008 | 2026-06-08 | 2026-09-03 | `$320,000-$485,000 USD` annual salary | San Francisco, New York City or Seattle; at least 25% office time | Distributed serving, request routing, load balancing, orchestration, deployment pipelines and high-performance systems |
| Anthropic, Staff+ Site Reliability Engineer, Safeguards ML Infra | https://job-boards.greenhouse.io/anthropic/jobs/5416709008 | 2026-09-04 | 2026-09-04 | `$320,000-$485,000 USD` annual salary | Remote-friendly with travel, or San Francisco, Seattle or New York City; company policy states at least 25% office time | Safe change management, continuous validation, configuration provenance, incident response, Python and operational automation |

All four records were active in the official feed. None exposed an application
deadline. These roles include senior and staff production responsibilities; they
are market evidence, not a claim that this repository demonstrates their level
or operating scale.

The recurring requirements are reliable distributed infrastructure, explicit
recovery and operational behavior, measurable correctness/performance, tested
systems code, and repeatable automation. The new Inference Engine listing is
especially direct about reusing cached state rather than recomputing work.

## Live public evidence inventory

GitHub's live API returned 21 public repositories and six non-fork,
non-archived pins on 2026-09-14. The strongest relevant default-branch evidence
inspected was:

- `triton-inference-benchmark` at `3730c3c`: authenticated agents, sampled clock
  bounds, bounded in-memory idempotent result recovery, open-loop load,
  trace/retry/path/lifecycle gates, privacy-safe aggregates and passing CI.
- `secure-gpu-inference-gateway`: authenticated policy and budget controls,
  Prometheus/OpenTelemetry, deployment posture checks, SBOM and vulnerability
  gates.
- `triton-kernel-lab`: correctness-gated Triton kernels and raw RTX 5070 Ti
  measurements.
- `deterministic-inference-scheduler`: Rust continuous batching, paged KV-cache
  accounting, deterministic replay and release gates.
- `market-microstructure-engine`: C++20/Python matching engines, parity tests and
  bounded local measurements.

The benchmark remains the right repository. Its agent already owns the
idempotency identity, execution boundary and privacy-safe result projection. A
new repository would duplicate that protocol without testing its real CLI.

## Evidence map

### Already demonstrated

- Explicit bearer authentication, redirect refusal and a minimal child
  environment.
- Hashed run/client identities, canonical request fingerprints and conflict
  rejection.
- Bounded in-memory completed-result recovery after an ambiguous response loss.
- Real two-agent CLI execution against a synthetic SSE target with clock,
  overlap, trace and aggregate privacy checks.

### Present but buried

- The coordinator-only result projection is already small enough to persist
  without target URLs, prompts, outputs, child paths, trace identifiers or
  credentials.
- The agent commits a completed result before writing its HTTP response, which is
  the correct durability boundary for recovery after response loss.
- An accepted-but-incomplete identity already fails closed instead of risking a
  second child execution.

### Missing before this change

- The completed-result cache is process-local. Restarting the agent after a child
  commits but before the coordinator receives the response loses the recoverable
  result.
- There is no explicit opt-in durable state path, transactional restart state,
  durable-versus-memory result-source accounting, or real process-restart fixture.

## Exactly one selected gap

**Recover a committed authenticated-agent result after the agent process restarts,
without re-executing the benchmark child.**

This is one durability extension to the existing idempotency mechanism. It does
not add general workflow orchestration, coordinator persistence or multi-host
claims.

## Intended implementation

1. Add an explicit `--state-db` option. With no option, preserve the bounded
   in-memory behavior. With the option, use a standard-library SQLite store with
   full synchronous commits and owner-only file permissions.
2. Persist only hashed run/client identity, canonical request fingerprint,
   hashed agent identity binding, state, bounded ordering metadata and the
   existing coordinator-only result projection. Never persist the raw run ID,
   benchmark arguments, target URL, prompt/output data, child path, trace
   identifiers, agent key or raw agent ID.
3. Commit `accepted` before launching the child. A restart after acceptance but
   before completion remains fail-closed. Commit `completed` and any bounded
   result eviction before sending success, so an identical retry can return the
   durable projection after restart.
4. Preserve the existing 1,024-identity, eight-result and 64 MiB bounds. Mark
   durable retrieval separately from in-process cache retrieval in JSON and
   Prometheus aggregate accounting.
5. Add unit tests for restart loading, conflict/incomplete/expiry behavior,
   permissions and persisted-field privacy. Add a real CLI fixture that drops a
   completed response, stops the agent, starts a new process with the same state
   database and identity, and verifies recovery with no extra target request.

## TDD and verification

1. Add the durable-store unit and CLI-fixture expectations first and confirm they
   fail against the current implementation.
2. Implement the transactional state store and source accounting.
3. Run all unit tests plus the remote-agent, restart, coordinated-client and
   supervised lifecycle fixtures.
4. Run Python compilation, a clean Python 3.13 container suite, the production
   Docker image and packaged smokes, strict dependency audit and
   `git diff --check`.
5. Publish through a pull request only after local checks pass. Require pull-
   request and post-merge `main` CI and read the code/docs from public `main`
   before publishing profile or resume wording.

## Claim boundary

The deterministic fixture will prove one local protocol case: a child completed,
an intermediary discarded the success response, the agent process stopped, and
a new process using the same SQLite file returned the committed projection for
an identical request without another target call. It will not prove coordinator
restart recovery, recovery of an interrupted child, multi-host storage,
production-network behavior, a real model/GPU, traffic isolation, fleet scale or
production operational experience.
