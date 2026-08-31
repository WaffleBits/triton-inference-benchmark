# Service-lifecycle qualification — 2026-08-31

## Market-backed gap

Current AI infrastructure listings repeatedly require failure diagnosis, explicit
retry and recovery models, incident response, and automated remediation. The
benchmark now proves retry and request-path accounting across separate local
router and backend processes, but every failure is an HTTP response from a
still-running fixture. It does not observe a process exit, a completed restart,
or a retry that crosses that lifecycle boundary.

## Existing-repository decision

Implement this in `triton-inference-benchmark`. It already owns the measured
request window, client retry semantics, bracketed multi-source Prometheus
capture, privacy-safe counter handling, and the deterministic multi-process
qualification. A new repository would duplicate those controls and produce a
weaker evidence chain.

## One coherent implementation

Add an opt-in, privacy-safe service-lifecycle qualification:

1. Accept one operator-selected cumulative service-restart counter and minimum /
   maximum restart bounds. Derive its delta from the same paired Prometheus
   snapshots that bracket measured load.
2. Fail closed on missing, non-finite, negative, non-integral, duplicated,
   reset, or membership-changing counter evidence. Persist a metric-name hash,
   counter delta, bounds, and gate result—not the metric name, labels, endpoint
   URLs, or raw scrapes.
3. Add a non-negative fixed retry delay so a controlled supervisor has a bounded
   interval in which to restore a child process before the next client attempt.
   Record the configured delay; do not call it measured remediation time.
4. Extend the deterministic path fixture with a separate supervisor process.
   Its backend child exits during one measured request, the supervisor observes
   the exit and starts a replacement on the same local port, and the client
   retries after the configured delay.
5. Require the real CLI artifact to show one completed restart, six client and
   router attempts, five backend receipts, four backend successes, two recovered
   logical requests, and passing lifecycle, request-path, retry, and trace gates.
6. Export only aggregate lifecycle evidence to JSON and Prometheus. Explicitly
   state that simultaneous aggregate deltas do not prove the restart caused a
   particular retry to recover and are not service MTTR.

## TDD and verification

- Start with failing unit tests for lifecycle counter validity, privacy, bounds,
  Prometheus export, retry delay, and CLI exit behavior.
- Run the complete unit suite and the actual supervised multi-process CLI fixture
  on the host and in the repository's Python container.
- Build and smoke-test the packaged image, compile Python sources, run strict
  requirement and installed-environment audits, and run `git diff --check`.
- Require pull-request CI and post-merge `main` CI to pass before updating any
  profile or resume wording.

## Claim boundaries

The fixture proves a local child process exited, a local supervisor started a
replacement, the replacement became ready, the aggregate restart counter
increased once inside the bracketed request window, and client retries recovered
all four synthetic logical requests. It does not demonstrate Kubernetes,
production orchestration, a real model or GPU, multi-node clocks, production
incident response, or service MTTR. Aggregate lifecycle, path, and retry counters
do not establish per-request causality.

## Local verification record

Completed on 2026-08-31 before publication:

- `python3 -m unittest discover -s tests`: 105 tests passed.
- `python3 tests/run_multi_process_path_fixture.py`: one controlled backend crash,
  one completed supervisor restart, six ingress attempts, five backend receipts,
  four successes, two retry-recovered logical requests, and all four gates passed;
  the same fixture also passed in `python:3.12-slim`.
- Paired operator-supplied snapshot CLI exercise: exact one-restart gate passed
  with `operator_supplied_unverified` alignment.
- Docker image build plus an eight-request mock run: eight successful requests.
- `pip-audit -r requirements.txt` in `python:3.13-bookworm`: no known
  vulnerabilities found.
- `python3 -m compileall -q benchmark.py tests` and `git diff --check`: passed.

These are deterministic local/CI fixture results. They are not observations from
a production model server, orchestrator, GPU, or incident.
