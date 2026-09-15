# Coordinator restart reconciliation for authenticated agent runs

## Primary-source role evidence

Checked through the companies' official job-board feeds on 2026-09-15.
`first_published` and `updated_at` are Greenhouse feed fields. Compensation is
copied from each posting; equity and total compensation are not inferred.

| Role | Canonical listing | First published | Feed updated | Published compensation | Location / constraints | Relevant requirements |
|---|---|---:|---:|---|---|---|
| Anthropic, Performance Engineer, Inference Engine | https://job-boards.greenhouse.io/anthropic/jobs/5418323008 | 2026-09-09 | 2026-09-09 | `$350,000-$850,000 USD` annual salary | San Francisco or New York City; at least 25% office time | Improve inference throughput, cost, reliability, and latency; reuse cached state instead of recomputing; measure before changing; tested systems code |
| Anthropic, Software Engineer, Research Infrastructure | https://job-boards.greenhouse.io/anthropic/jobs/5283063008 | 2026-07-06 | 2026-09-09 | `$405,000-$625,000 USD` annual salary | San Francisco or New York City; at least 25% office time | Design, build, and operate distributed research infrastructure; own reliability and scalability; cloud infrastructure and infrastructure-as-code |
| Anthropic, Staff + Senior Software Engineer, Inference Infrastructure | https://job-boards.greenhouse.io/anthropic/jobs/5245851008 | 2026-06-08 | 2026-09-03 | `$320,000-$485,000 USD` annual salary | San Francisco, New York City, or Seattle; at least 25% office time | Resilient distributed serving, request routing, load balancing, orchestration, deployment pipelines, observability, Python or Rust |
| Together AI, Staff Software Engineer, Inference / Compute Infrastructure Engineering | https://job-boards.greenhouse.io/togetherai/jobs/5186628007 | 2026-07-16 | 2026-09-09 | `$240,000 - $280,000 + equity + benefits` US base salary | San Francisco; the feed did not expose a remote option | Durable workflow orchestration that survives failures and resumes mid-execution; explicit state transitions, reconciliation loops, idempotency, retries, rollback, drift detection, Python/Go/Rust, and tested CI/CD |

The Anthropic Inference Infrastructure posting states that applications are
reviewed on a rolling basis with no deadline. The other three feed records expose
no deadline. These are senior and staff production roles. They are market
signals, not claims that this repository demonstrates their operating scale or
professional level.

Recurring requirements are reliable distributed infrastructure, explicit and
recoverable orchestration state, idempotent work, measurable correctness and
performance, observability, and tested automation. The Together AI listing is
the most direct evidence for restart-safe workflow reconciliation; the Anthropic
roles connect that behavior to inference reliability and avoiding recomputation.

## Live public evidence inventory

GitHub's live API returned 21 public repositories and six non-fork,
non-archived pins on 2026-09-15. The strongest relevant default-branch evidence
inspected was:

- `triton-inference-benchmark` at `e3bb060`: authenticated agents, sampled clock
  bounds, idempotent response-loss recovery, opt-in SQLite result recovery across
  an agent restart, open-loop load, trace/retry/path/lifecycle gates,
  privacy-safe aggregates, and passing CI.
- `secure-gpu-inference-gateway`: authenticated policy and budget controls,
  Prometheus/OpenTelemetry, deployment posture checks, SBOM, and vulnerability
  gates.
- `triton-kernel-lab`: correctness-gated Triton kernels and raw RTX 5070 Ti
  measurements.
- `deterministic-inference-scheduler`: Rust continuous batching, paged KV-cache
  accounting, deterministic replay, and release gates.
- `market-microstructure-engine`: C++20/Python matching engines, parity tests,
  and bounded local measurements.

The benchmark remains the right repository. Its coordinator already owns the
workflow fan-out, clock plan, reconciliation, and final artifacts, while its
agents already enforce idempotent execution. A new repository would duplicate
those real boundaries instead of testing the existing CLI.

## Evidence map

### Already demonstrated

- Explicit bearer authentication, redirect refusal, bounded requests, and a
  minimal benchmark-child environment.
- Hashed run/client identities, request fingerprints, conflict rejection, and
  exact-result retrieval after ambiguous response loss.
- Optional agent-side SQLite state that recovers a completed result after the
  agent process restarts.
- Real two-agent CLI execution against a synthetic SSE target with clock,
  overlap, trace, duplicate-execution, and artifact-privacy checks.

### Present but buried

- The coordinator creates one run identity and one per-agent planned start before
  concurrent dispatch.
- Each agent can return the same completed projection without executing another
  child, so a restarted coordinator can safely reconcile already-completed work
  if it can reconstruct the exact request.
- Clock profiles and result projections already omit raw challenges, agent IDs,
  agent URLs, credentials, prompts, outputs, target URLs, paths, and trace IDs.

### Missing before this change

- The coordinator's run identity, original clock plan, and per-agent request
  parameters exist only in process memory. A coordinator process loss therefore
  cannot reconstruct the same agent requests.
- There is no authenticated status query that lets a resumed coordinator verify
  all shards completed before retrieval. Blindly redispatching after a stale
  planned start could launch missing work too late and invalidate the coordinated
  window.
- No real CLI fixture kills one coordinator after both agents finish, starts a
  second coordinator process, and proves aggregate recovery without additional
  target requests.

## Exactly one selected gap

**Reconcile an authenticated multi-agent run after the coordinator process
restarts, without repeating completed target work or launching missing shards
against a stale clock plan.**

This is one bounded post-execution recovery path for the existing coordinated
benchmark. It is not a general workflow engine, partial-workflow continuation,
or in-flight child recovery.

## Intended implementation

1. Add opt-in coordinator state and an explicitly selected resume-token
   environment variable. Derive the agent run ID from the token plus a persisted
   random nonce; persist neither the token nor derived run ID.
2. Atomically write an owner-only, bounded manifest before workload dispatch. It
   contains the original privacy-safe clock profiles, hashed agent identities,
   per-agent planned starts, and a configuration fingerprint, but no benchmark
   arguments, endpoint or agent URLs, prompts, outputs, credentials, environment
   names, paths, challenges, trace IDs, or result bodies.
3. Add an authenticated agent status endpoint using the exact run-request
   fingerprint. It reports only missing, accepted, completed, or expired state
   and whether the bounded result remains available.
4. On coordinator resume, probe and match agent identities, verify the manifest
   token/configuration binding, and require every exact request to be completed
   and retrievable before issuing result-retrieval calls. Refuse missing,
   incomplete, expired, or conflicting state rather than launching work after a
   stale planned start.
5. Record whether coordinator recovery was enabled and resumed in aggregate JSON
   and Prometheus output without exposing private state.
6. Add unit tests for derivation, state integrity/permissions/privacy, status
   semantics, and fail-closed validation. Add a real fixture that blocks both
   completed responses, kills the first coordinator, resumes with a second
   process, and verifies eight target requests remain eight.

## TDD and verification

1. Add the state, status, and restart-fixture expectations first and confirm they
   fail against the current implementation.
2. Implement the bounded manifest, status protocol, resume preflight, and
   aggregate accounting.
3. Run all unit tests plus the remote-agent, agent-restart,
   coordinator-restart, coordinated-client, and supervised-lifecycle fixtures.
4. Run Python compilation, a clean Python 3.13 container suite and fixtures, the
   production Docker image and packaged smokes, strict dependency audit, and
   `git diff --check`.
5. Publish through a pull request only after local checks pass. Require pull-
   request and post-merge `main` CI and read the implementation from public
   `main` before publishing profile or resume wording.

## Claim boundary

The deterministic fixture will prove one local synthetic protocol case: two
agents complete eight target requests, two intermediaries withhold the successful
responses, the coordinator is killed, and a new coordinator uses an owner-only
manifest plus an explicit resume token to verify and retrieve both stored agent
results. It will not prove recovery when a shard is missing or incomplete,
continuation of an interrupted benchmark child, agent loss without retained
state, arbitrary network faults, multi-host operation, production networking,
hardware clock synchronization, a real model/GPU, traffic isolation, Temporal or
Cadence experience, or fleet scale.
