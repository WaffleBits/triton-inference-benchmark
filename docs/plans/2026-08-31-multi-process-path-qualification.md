# Multi-process request-path qualification — 2026-08-31

## Market-backed gap

Current AI infrastructure listings repeatedly ask for cross-layer inference
diagnosis, observable distributed systems, explicit failure/retry behavior, and
automation that makes recovery verifiable. The benchmark can reconcile three
counters, but its deterministic request-path proof currently obtains every
counter from one fixture process. That does not exercise independent router and
backend telemetry sources, and the retry record counts recovered requests without
reporting their end-to-end latency.

## Existing-repository decision

Implement this in `triton-inference-benchmark`. It already owns the measured
request window, streaming client, retry accounting, paired Prometheus capture,
privacy-safe artifacts, and exact path gate. A new repository would duplicate
those boundaries instead of extending the strongest relevant evidence chain.

## One coherent implementation

Add a reproducible multi-process request-path qualification:

1. Make `--telemetry-url` repeatable and scrape all configured HTTP(S) endpoints
   concurrently at each benchmark boundary.
2. Fail closed if any source fails, cap both per-source and combined response
   size, reject duplicate endpoints, and retain only the source count—not URLs,
   credentials, headers, or raw scrapes—in artifacts.
3. Preserve the existing isolated-counter invariants over the combined snapshot.
4. Add separate deterministic router and backend fixture processes. Inject one
   router-local 503 and one backend 503 on different logical requests, then prove
   retry recovery through independent ingress/backend/success counters.
5. Report the measured end-to-end latency distribution of logical requests that
   succeeded after retry. Label it client-observed recovery latency, not service
   MTTR or proof of process recovery.
6. Exercise the real CLI with six client attempts, six router receipts, five
   backend receipts, four successful completions, and two recovered logical
   requests for four measured requests.

## Verification and claim boundaries

- Start with failing unit tests for concurrent source capture, source-count
  privacy, repeated CLI options, duplicate rejection, and recovered-request
  latency export.
- Run the complete unit suite and the actual multi-process CLI fixture on the
  host and in the repository's Python container.
- Build and smoke-test the packaged image, compile Python sources, run strict
  dependency audits, and run `git diff --check`.
- Require pull-request CI and post-merge `main` CI to pass before publishing any
  profile or resume claim.

The fixture uses separate local OS processes and independent HTTP metric sources,
but one host and synthetic failures. It does not demonstrate Kubernetes, a real
model or GPU, production traffic, distributed clocks, autoscaling, deployed
router software, or service MTTR.
