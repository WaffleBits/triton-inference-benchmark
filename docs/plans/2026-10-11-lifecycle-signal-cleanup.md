# Plan: clean lifecycle subprocesses on operator interruption

Date: 2026-10-11 (America/New_York)
Repository: `WaffleBits/triton-inference-benchmark`
Branch: `fix/lifecycle-signal-cleanup-20261011`

## Current role evidence (official listings checked 2026-10-11)

| Role | Official listing | Published / updated | Compensation as published | Location / work mode | Repeated requirements relevant here |
|---|---|---|---|---|---|
| OpenAI, Software Engineer, Inference - Performance Optimization | https://jobs.ashbyhq.com/openai/85fceac9-fb8a-4d71-a524-a8e5f1e9b01b | Ashby feed published 2026-04-25 02:24:38 UTC; still listed when rechecked today | `$266K – $500K • Offers Equity` | San Francisco; full-time; no remote designation exposed | Cross-layer performance measurement, benchmarking and models; bottleneck analysis across latency, throughput, capacity, utilization and cost. |
| Anthropic, Performance Engineer, Inference Systems | https://job-boards.greenhouse.io/anthropic/jobs/5224564008 | Greenhouse updated 2026-08-21 12:50:15 -04:00; original publication date not exposed | `Annual Salary: $350,000 — $850,000 USD` | San Francisco, New York City or Seattle; listing's location type is On-Site; company-wide policy says at least 25% office time | Python, performance investigation, observability, reliability, and correctness/regression checks across serving configurations. |
| Anthropic, Performance Engineer, Inference Engine | https://job-boards.greenhouse.io/anthropic/jobs/5418323008 | Greenhouse updated 2026-09-09 19:09:53 -04:00; original publication date not exposed | `Annual Salary: $350,000 — $850,000 USD` | San Francisco or New York City; listing's location type is On-Site; company-wide policy says at least 25% office time | Rust/C++/systems programming, profiling before changes, throughput/latency/cost/reliability, and reproducibility through determinism, replay and tests. |

The common requirements support the existing benchmark's measurement and regression-gate work. The selected addition addresses one narrowly observable reliability gap in its controlled local lifecycle CLI; it does not imply production or distributed operation.

## Live public evidence inventory

The authenticated GitHub API confirms the `WaffleBits` account has 21 public repositories. The profile repository is the public TypeScript/Astro site, and the strongest directly relevant existing code repository is `triton-inference-benchmark` (Python; not a fork or archived; one public star). Its current public `main` is `af8375d6a610317019000aeaaf523763472f7ee8`; its latest CI run, 38047497635, passed. The repository already has unit tests and real CLI fixtures for lifecycle readiness and process cleanup, plus independent benchmark/agent restart recovery. Its lifecycle claim is explicitly local process-launch-to-health readiness.

## Evidence map

**Already demonstrated:** controlled loopback HTTP-200 readiness, a real benchmark CLI run, safe report projection, shutdown after normal completion, bounded graceful-then-forced process-group termination, and deterministic local fixtures.

**Present but incomplete:** the lifecycle wrapper starts the service in a separate process session, then runs the benchmark as a child, but the CLI does not handle SIGTERM itself. A SIGTERM to the wrapper bypasses Python `finally` cleanup and can leave the service process group and benchmark child running. The existing fixture covers normal completion, not operator cancellation.

**Missing and not claimed:** service recovery, distributed cancellation/leases, production SLOs, remote hosts, and production inference behavior.

## Exactly one selected gap

**Make SIGINT/SIGTERM cancellation of `lifecycle_qualification.py` terminate and reap the benchmark subprocess group and the controlled service process group, then return the conventional `128 + signal` status.**

This is a bounded failure-safety improvement in the existing benchmark repository. It will not add restart recovery, signal propagation across remote agents, new metrics, or production claims.

## Implementation and acceptance plan

1. Add a deterministic CLI fixture that interrupts the wrapper while a benchmark request is held open; first verify that it exposes the current orphan-process behavior.
2. Record cancellation signals without raising asynchronously through process creation, poll for cancellation during readiness and benchmark execution, terminate/reap the benchmark process group, and always run existing service-group cleanup.
3. Assert the CLI returns `143` for SIGTERM, the fixture observes service termination, no benchmark child remains active, and no successful lifecycle report is emitted for the cancelled run. Preserve normal-success behavior.
4. Add the fixture to the canonical GitHub Actions workflow and document only the tested cancellation behavior and its local scope.
5. Run Python 3.13 unit tests, all checked-in CLI fixtures, `pip-audit -r requirements.txt`, `git diff --check`, and the GitHub Actions workflow. Commit/push without force only after local checks pass; require CI before merging.
6. Only after the source change is public and CI passes, surface one restrained proof link in the Astro profile, run its `node:24-bookworm` clean-copy checks/audit/build/render validation, publish through its Pages workflow, and read the live page back.

## Local verification (2026-10-11)

The new fixture first reproduced the previous behavior: sending SIGTERM caused
`lifecycle_qualification.py` to exit with `-15`, bypassing its normal cleanup and
status handling. With the change applied:

- The SIGINT and SIGTERM fixtures each interrupted a held-open local benchmark
  request, observed service-group shutdown and benchmark-child exit, confirmed
  status `130`/`143`, and verified that no success report was written.
- Python 3.13.14 ran all 151 unit tests and all 12 checked-in Python CLI fixture
  invocations successfully; the normal lifecycle fixture still passed.
- `pip-audit -r requirements.txt` reported no known vulnerabilities.
- `py_compile` passed for the lifecycle module and changed fixture scripts.
- `git diff --cached --check` passed. The scoped change contains only the plan,
  lifecycle implementation, fixture, CI assertion, and README documentation.

The GitHub Actions, PR, merge, and profile publication read-backs are recorded in
the daily market-alignment report after those steps complete.

## Remaining evidence boundary

The fixture is a deterministic, single-host subprocess test. It does not establish remote cancellation, durable in-flight recovery, cross-host process supervision, a model/GPU lifecycle, production availability, or production SLOs.
