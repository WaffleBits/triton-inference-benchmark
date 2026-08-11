# Propagate privacy-safe W3C trace context from live benchmark requests

## Market signal

Reviewed on 2026-08-11 from active official company job-board pages and feeds:

- Anthropic, [Performance Engineer, Inference Systems](https://job-boards.greenhouse.io/anthropic/jobs/5224564008), first published 2026-05-20 and updated 2026-08-03: **$350,000–$850,000 USD annual salary**; San Francisco, New York City, or Seattle with Anthropic's location-based hybrid policy. The role calls for tracing tail-latency regressions across routing, batching, and kernels; cross-layer root-cause analysis; correctness; and distributed-system observability.
- OpenAI, [Software Engineer, Inference - Performance Optimization](https://jobs.ashbyhq.com/openai/85fceac9-fb8a-4d71-a524-a8e5f1e9b01b), published 2026-04-25: **$295K–$555K plus equity**; San Francisco. The role calls for end-to-end analysis across application, model, and fleet layers, performance profiling, benchmarking, cost models, and latency/utilization/capacity tradeoffs.
- OpenAI, [ChatGPT Performance Engineer](https://jobs.ashbyhq.com/openai/38ddaa2c-a490-427a-8457-0e92bf00138c), published 2026-04-15: **$325K–$405K plus equity**; San Francisco, New York City, Seattle, or US remote. The role emphasizes root-cause analysis, instrumentation, tracing, observability, reliability, and performance testing across application and infrastructure layers.
- Etched, [Software Engineer – Performance Profiling](https://jobs.ashbyhq.com/etched/610c3836-9798-46ea-931a-02bb95b29467), published 2026-01-18: **$150K–$275K plus significant equity**; on-site in San Jose. The role asks for host-side tracing, hardware counters, and correlated events across CPUs, drivers, PCIe, accelerators, and hosts.

The repeated requirement is request-level attribution across layers. Bracketed counters explain a benchmark window, but they cannot connect one client request to its downstream server, scheduler, and kernel spans.

## Evidence map

### Already demonstrated

- The harness brackets Triton counters around measured requests and can sample DCGM gauges during that window.
- It emits client-side latency, streaming TTFT/inter-chunk distributions, server counter gates, and privacy-preserving series-membership evidence.
- `secure-gpu-inference-gateway` already demonstrates sanitized trace export inside a gateway, but the load harness cannot currently establish trace continuity at the request boundary.

### Present but buried

- Both live clients already own the outbound HTTP request boundary.
- The Triton Python HTTP client accepts per-request headers, and the OpenAI-compatible client constructs its own request headers.
- Existing local HTTP fixtures and artifact-privacy tests can verify propagation without a model, GPU, collector, or production trace.

### Missing

- Live benchmark requests carry no W3C `traceparent`, so a trace-enabled vLLM or Triton server cannot continue a benchmark-originated request context.
- The CLI cannot opt into trace-context propagation.
- There is no artifact contract proving that trace IDs and parent IDs stay out of shareable JSON and Prometheus output.

## Decision

Extend this repository rather than create another project. It already owns both live request clients, retry behavior, the CLI, JSON/Prometheus artifacts, fixtures, and privacy tests.

Implement one coherent capability: **opt-in W3C Trace Context propagation for live Triton and OpenAI-compatible requests**. Each physical HTTP attempt receives a fresh standards-shaped `traceparent`; no trace ID, parent ID, header value, prompt, or authorization value is persisted.

## Implementation

1. Add a dependency-free W3C `traceparent` generator using non-zero 128-bit trace IDs, non-zero 64-bit parent IDs, version `00`, and the sampled flag.
2. Add `--propagate-trace-context`, valid only for live `triton` and `openai` modes; default behavior remains unchanged.
3. Inject a fresh header into every physical request attempt in both clients, including retries, without adding `tracestate` or reading ambient tracing configuration.
4. Persist only a configured propagation summary and a numeric Prometheus enabled indicator. State explicitly that header injection does not prove server acceptance, export, or clock synchronization.
5. Add unit tests for header shape, uniqueness, default-off behavior, Triton/OpenAI injection, CLI validation, and artifact privacy.
6. Exercise the real OpenAI-compatible CLI against a deterministic local SSE fixture, verify every request receives a unique valid header, and verify none of those identifiers appears in the JSON or Prometheus artifact.

## Verification gates

- Full `unittest` discovery and Python bytecode compilation.
- Real CLI run against the deterministic OpenAI-compatible HTTP fixture.
- Docker image build and containerized mock execution.
- Dependency audit, `git diff --check`, pull-request CI, and main-branch CI.

## Claim boundary

This feature proves that the benchmark generated and placed a valid W3C `traceparent` on each tested outbound HTTP request. It does not prove that a remote server accepted the context, created or exported spans, sampled according to the flag, synchronized clocks, or correlated accelerator work. The local fixture verifies HTTP wiring and artifact privacy; it is not a production trace or multi-host performance measurement.
