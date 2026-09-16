# TLS transport qualification for authenticated benchmark agents

## Primary-source role evidence

Official company job feeds were read on 2026-09-16. Dates below are feed fields;
compensation is copied exactly as exposed. Equity and total compensation are not
estimated.

| Role | Canonical listing | First published | Feed updated | Published compensation | Location / constraints | Repeated requirements relevant here |
|---|---|---:|---:|---|---|---|
| Anthropic, Performance Engineer, Inference Systems | https://job-boards.greenhouse.io/anthropic/jobs/5224564008 | 2026-05-20 | 2026-08-21 | `$350,000-$850,000 USD` annual salary | San Francisco, New York City, or Seattle; at least 25% office time | Cross-layer throughput/latency/reliability/correctness, observability, regression investigation, Python, serving infrastructure |
| Anthropic, Staff + Senior Software Engineer, Inference Infrastructure | https://job-boards.greenhouse.io/anthropic/jobs/5245851008 | 2026-06-08 | 2026-09-03 | `$320,000-$485,000 USD` annual salary | San Francisco, New York City, or Seattle; at least 25% office time | Distributed serving, request routing, load balancing, orchestration, deployment pipelines, observability, Python or Rust |
| OpenAI, Software Engineer, Compute Infrastructure | https://jobs.ashbyhq.com/openai/ca300a6d-a2a7-4580-aad7-323fbdfee7b1 | 2026-08-14 | Not exposed | `$230K – $405K • Offers Equity` | Ashby feed: San Francisco; detail page exposes telecommute eligibility for the United States and United Kingdom plus New York City, Seattle, and London | Durable compute infrastructure, distributed systems, Kubernetes/scheduling, observability, reliability, benchmarking, safe operation across infrastructure layers |
| Together AI, AI Infrastructure Systems Engineer | https://job-boards.greenhouse.io/togetherai/jobs/5138540007 | 2026-05-14 | 2026-09-09 | `$190,000 - $270,000 + equity + benefits` US base salary | San Francisco; no remote option exposed in the feed | Fleet automation, automated validation, diagnosis/remediation, network and hardware health, GPU availability/reliability, Python and distributed systems |

The postings are production-role signals, not claims that this repository
operates at their scale or demonstrates their level.

## Live evidence inventory

GitHub's public API returned 21 repositories for `WaffleBits` on 2026-09-16. The
six pinned non-fork, non-archived repositories are `market-microstructure-engine`,
`readiness-control-tower`, `secure-gpu-inference-gateway`, `triton-kernel-lab`,
`deterministic-inference-scheduler`, and `heterocore-compiler`. The profile README
and portfolio already surface `triton-inference-benchmark` as the strongest
reliability-focused repository. Its public `main` was `9ba6180` after successful
CI on 2026-09-15; the portfolio site's `main` was `7aca7e3` and its Pages workflow
was green.

The benchmark already demonstrates authenticated agent requests, redirect
refusal, clock-quality bounds, retry/reconciliation, durable agent state,
coordinator restart recovery, artifact redaction, and Kubernetes deployment
shape. Its public claim boundary correctly says that existing fixtures use
separate processes on one host and do not prove physical multi-host behavior.

## Evidence map

### Already demonstrated

- Explicit bearer-key selection, no ambient child credential inheritance, and
  refusal to follow redirects.
- HTTPS required for non-loopback agent URLs at URL-validation time.
- Authenticated clock probes, conservative clock uncertainty, bounded request
  payloads, and restart-safe result reconciliation.
- Real CLI fixtures with separate agent processes, synthetic streaming target,
  response-loss injection, SQLite persistence, and privacy assertions.

### Present but buried

- `remote_agent.py` already has one transport boundary (`_post_json`) shared by
  clock, status, and run requests, so certificate trust can be made explicit in
  one place without duplicating protocol code.
- The agent server is a standard-library `HTTPServer`, so an opt-in TLS socket
  wrapper can preserve the existing bounded protocol and child lifecycle.

### Missing

- The agent executable has no operator-facing certificate/key configuration.
- The coordinator has no explicit custom-CA trust-anchor option for private
  authorized deployments.
- CI has no real HTTPS agent fixture proving certificate verification, bearer
  authentication, CLI wiring, and artifact privacy together.

## Exactly one selected gap

**Qualify the existing authenticated remote-agent protocol over explicitly
configured TLS, with certificate verification and no cleartext fallback.**

This is a transport/security extension of the existing agent protocol. It does
not claim physical multi-host execution, mutual TLS, production certificates,
network-fleet scale, or a real model/GPU.

## Implementation plan

1. Add `--tls-cert-file` and `--tls-key-file` to the agent. Require the pair,
   load them through `ssl.SSLContext`, require TLS 1.2 or newer, and wrap the
   listening socket before serving. Keep plain HTTP available only for the
   existing explicitly local development/fixture path.
2. Add `--agent-ca-file` to the coordinator. Pass an explicit CA file only to
   HTTPS requests; otherwise use the platform verifier. Reject missing,
   directory, or symbolic-link trust files and never serialize their paths,
   certificate material, or authorization headers.
3. Add unit tests for TLS option validation, custom-context wiring, and the
   invariant that an HTTP redirect or certificate failure is terminal rather
   than silently downgraded.
4. Add `tests/run_tls_remote_agent_fixture.py`: generate an ephemeral
   localhost certificate with OpenSSL, start two checked-in TLS agent
   processes, run the real coordinated CLI against the synthetic SSE target,
   assert both TLS and bearer checks, exercise a wrong-CA failure, and inspect
   JSON/Prometheus for certificate-policy evidence without private values.
5. Add the fixture to CI and update `README.md` and `DESIGN.md` with the command,
   operator trust model, and explicit single-host synthetic claim boundary.

## Verification gates

- Unit suite and the TLS fixture pass on the host.
- The clean `python:3.13-bookworm` copy passes the suite and fixtures.
- Production Docker image build and packaged mock/agent smokes pass.
- `pip-audit 2.9.0 --strict -r requirements.txt` passes.
- `python -m py_compile` and `git diff --check` pass.
- Pull-request and post-merge `main` CI pass before any profile or resume claim
  is updated.

Only after the implementation is public and post-merge CI is green will the
portfolio and resume mention the verified TLS path. Wording will state that the
fixture is localhost/synthetic and will not imply multi-host or production
network operation.
