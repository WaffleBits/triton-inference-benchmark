# Plan: cancel distributed benchmark work on coordinator interruption

Date: 2026-10-12 (America/New_York)
Repository: `WaffleBits/triton-inference-benchmark`
Branch: `feat/remote-agent-cancellation-20261012`

## Current official role evidence (checked 2026-10-12)

The OpenAI Ashby public job-board API and Anthropic Greenhouse public job-board API were queried directly. OpenAI exposes `publishedAt` and `isListed`; Anthropic exposes `updated_at` but not original publication dates. All three entries were present in their official feeds when checked.

| Role | Official listing | Published / updated | Compensation exactly as listed | Location / work mode | Repeated technical requirements |
|---|---|---|---|---|---|
| OpenAI, Software Engineer, Inference - Performance Optimization | https://jobs.ashbyhq.com/openai/85fceac9-fb8a-4d71-a524-a8e5f1e9b01b | Published `2026-04-25T02:24:38.573+00:00`; `isListed: true` | `$266K – $500K • Offers Equity` | San Francisco; full-time; workplace type / remote policy not exposed in the feed | Cross-layer inference profiling and benchmarking; performance/cost modeling; latency, throughput, capacity, and accelerator efficiency. |
| Anthropic, Performance Engineer, Inference Engine | https://job-boards.greenhouse.io/anthropic/jobs/5418323008 | Updated `2026-09-09T19:09:53-04:00`; original publication date not exposed | `Annual Salary: $350,000 — $850,000 USD` | San Francisco, CA or New York City, NY; Greenhouse metadata `On-Site` | Rust/C++ systems programming; profile, model, and measure before changes; GPU/accelerator performance and memory/interconnect; correctness and tests. |
| Anthropic, Staff + Senior Software Engineer, Inference Infrastructure | https://job-boards.greenhouse.io/anthropic/jobs/5245851008 | Updated `2026-09-03T09:29:10-04:00`; original publication date not exposed | `Annual Salary: $320,000 — $485,000 USD` | San Francisco, CA; New York City, NY; or Seattle, WA; Greenhouse metadata `On-Site` | Distributed inference systems; request routing/load balancing; fleet orchestration/autoscaling; Python or Rust; production deployment and observability. |

Across this small, role-relevant sample, recurring needs are measured performance work, systems-level inference, reliability of distributed coordination, and tests that make failure behavior inspectable. Compensation is published salary, not total compensation, except OpenAI's listing separately notes equity.

## Public evidence inventory and map

The public profile bio is “AI infrastructure, GPU performance, secure distributed systems, and mission software”; its README highlights the gateway, kernel lab, scheduler, and benchmark. Six pinned repositories include the security gateway, Triton kernel lab, and deterministic scheduler, but not the benchmark. GitHub's user endpoint reported 21 public repositories. The directly relevant unarchived, non-fork Python repository [`triton-inference-benchmark`](https://github.com/WaffleBits/triton-inference-benchmark) was recently updated and its public description/topics cover authenticated Triton/OpenAI serving benchmarks, inference, observability, performance engineering, and Triton. Other strong evidence includes the GPU-kernel lab (CUDA/Triton correctness and measured comparisons), deterministic Rust inference scheduler (replay and release gates), and secure inference gateway (authorization and telemetry). The benchmark's `README.md`, `DESIGN.md`, tests, and GitHub Actions workflow provide the strongest home for a distributed cancellation improvement. Its public `main` was `6db3a0da465e0151b8eb86cd289187fbe4284339`; the prior post-merge CI run [38134745823](https://github.com/WaffleBits/triton-inference-benchmark/actions/runs/38134745823) passed.

**Already demonstrated:** coordinated benchmarks can reconcile completed agent results after a coordinator process restart; the local lifecycle wrapper safely cleans up its own benchmark child and service process on SIGINT/SIGTERM.

**Present but incomplete:** `coordinated_benchmark.py` waits for remote `/v1/run` calls, but coordinator interruption did not send cancellation to agents. The synchronous single-threaded `HTTPServer` could not accept simultaneous control-plane requests during an active run. This could leave remote work running after the coordinator exited.

**Missing and not claimed:** durable continuation of interrupted shards, replacement-agent recovery, fleet-scale operation, production network or serving behavior, and production SLOs.

## Exactly one selected gap

**Propagate coordinator SIGINT/SIGTERM to active authenticated remote agents, terminate and reap their benchmark child, and prevent the cancelled run from becoming a success artifact.** Keep the agent single-job-at-a-time; make its authenticated cancellation endpoint responsive during execution; preserve exact request identity checks and fail-closed completed-result recovery. This is a bounded distributed cancellation feature, not durable partial-work recovery.

The profile README already surfaces local process cleanup and agent restart reconciliation, but neither proves cancellation propagation across the authenticated coordinator-agent boundary. The added source fixture is the selected evidence; profile publication remains gated on public CI and the source landing in the repository's default branch.

## Implementation and verification

1. The agent now serves bounded concurrent control requests (maximum eight handlers) while preserving one active benchmark per agent. Authenticated `/v1/run-cancel` accepts only the exact hashed run identity and request fingerprint; cancellation terminates/reaps the child, prevents artifact completion, and retains fail-closed replay semantics.
2. The coordinator captures SIGINT/SIGTERM in remote mode, cancels the exact request on each configured agent, reports confirmed acknowledgements, writes no success aggregate, and exits with `128 + signal`.
3. Added unit coverage for cancellation acknowledgement validation, cancelled status, idempotency, conflicting request fingerprints, and bounded cancellation state. The TLS fixture rejects an invalid bearer credential on the cancellation endpoint. The real-CLI fixture holds target requests open and verifies two agent children exit for both signals.
4. Updated README, DESIGN, and OPERATIONS evidence boundaries and added the two signal fixtures to the Python 3.13 GitHub Actions workflow.
5. Local verification in clean `python:3.13-bookworm` (Python 3.13.15): 154 unittest cases passed; every direct Python fixture step in `ci.yml` passed, including both lifecycle-signal fixtures, remote-agent cancellation SIGINT/SIGTERM, agent restart, and coordinator-restart reconciliation; `py_compile` passed; `pip-audit 2.9.0 -r requirements.txt` reported no known vulnerabilities. `git diff --check` is run before commit. One existing fixture race was made tolerant of partially written state files, and coordinator-restart testing now uses SIGKILL to model a crash rather than the newly handled SIGTERM.
6. Publication is pending: fetch and verify the base, commit/push without force, open the PR, and require the GitHub Actions workflow to pass. Merge only after checks are green and remote head is current. Then update the Astro profile from its clean checkout, reproducing its CI in `node:24-bookworm` and verifying the live deployment read-back.

## Evidence boundary

The fixture is a deterministic loopback protocol test and validates process cleanup under operator cancellation. It does not establish cancellation delivery through arbitrary production networks, durable recovery/continuation of interrupted work, multi-host fleet scale, production inference performance, or production SLOs. Cancellation confirmation is not guaranteed for an unreachable or unresponsive agent; its benchmark may continue until that agent's configured timeout or operator intervention.
