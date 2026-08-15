# Client-attempt amplification gate — 2026-08-15

## Market-backed gap

Current high-compensation inference and infrastructure listings repeatedly ask for reliability, tail-latency investigation, load balancing/routing, observability, and measurable release controls. This benchmark already retries failed logical requests, but its artifacts expose only the configured retry limit and final request outcome. A successful run can therefore hide extra client attempts and the load they may add during a degraded interval.

## Existing-repository decision

Implement this in `triton-inference-benchmark`. The request executor, JSON summary, Prometheus exporter, CLI fail gates, deterministic OpenAI-compatible fixture, and operational documentation already live here. A separate project would duplicate the load path and would not create a stronger evidence chain.

## One coherent implementation

Add measured-phase client-attempt accounting and an optional amplification gate:

1. Record the number of `InferenceClient.infer` calls used by each logical request.
2. Summarize total client attempts, retry attempts, retried logical requests, recovered requests, exhausted requests, and client-attempt amplification.
3. Keep warmup attempt accounting separate from the measured phase.
4. Export the aggregate evidence to JSON and Prometheus.
5. Add `--max-client-attempt-amplification` and `--fail-on-retry-gate`; fail closed when the measured factor exceeds the explicit run-scoped budget.
6. Exercise one transient HTTP 503 followed by recovery through the real CLI and deterministic local SSE fixture.

## Claim and privacy boundaries

- “Client attempt” means one harness call to the inference client. It does not prove that an endpoint, router, model server, or accelerator received the request.
- The amplification factor is measured client attempts divided by measured logical requests; configured retries are only a ceiling.
- Warmup attempts do not affect the measured gate.
- Artifacts contain aggregate counts and thresholds only. Prompts, streamed output, authorization headers, trace identifiers, endpoint query parameters, and exception text remain absent.

## Verification

- Add tests before implementation for retry recovery/exhaustion accounting, pass/fail gate behavior, CLI validation, Prometheus output, and privacy boundaries.
- Run the full unit/integration suite.
- Run the actual CLI against the local OpenAI-compatible fixture with one deterministic transient failure and inspect JSON/Prometheus artifacts.
- Build and run the Docker image in mock mode.
- Run dependency audit and `git diff --check`.
- Require pull-request CI and post-merge `main` CI to pass before profile/resume publication.
