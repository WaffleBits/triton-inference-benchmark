# Gate on privacy-safe HTTP response trace continuation

## Market signal

Reviewed on 2026-08-12 from active official company job-board pages and feeds:

- Anthropic, [Performance Engineer, Inference Systems](https://job-boards.greenhouse.io/anthropic/jobs/5224564008), official board updated 2026-08-03: **$350,000–$850,000 USD annual salary**; San Francisco, New York City, or Seattle. The role asks for cross-layer tracing, observability, regression detection, correctness, and root-cause analysis across routing, batching, and kernels.
- Anthropic, [Performance Engineer, GPU](https://job-boards.greenhouse.io/anthropic/jobs/4926227008), official board updated 2026-08-03: **$280,000–$850,000 USD annual salary**; San Francisco, New York City, or Seattle. The role names Triton/CUDA, Nsight profiling, distributed communication, NVLink, kernel optimization, and production inference performance.
- OpenAI, [Software Engineer, Inference - Performance Optimization](https://jobs.ashbyhq.com/openai/85fceac9-fb8a-4d71-a524-a8e5f1e9b01b), published 2026-04-25: **$295K–$555K plus equity**; San Francisco. The role asks for end-to-end inference analysis, profiling, benchmarking, cost models, and reasoning across applications, kernels, accelerators, networking, and fleet scheduling.
- OpenAI, [ChatGPT Performance Engineer](https://jobs.ashbyhq.com/openai/38ddaa2c-a490-427a-8457-0e92bf00138c), published 2026-04-15: **$325K–$405K plus equity**; San Francisco, New York City, Seattle, or US remote. The role emphasizes root-cause analysis, instrumentation, tracing, observability, reliability, and performance tests with latency/throughput SLOs.
- Etched, [Software Engineer – Performance Profiling](https://jobs.ashbyhq.com/etched/610c3836-9798-46ea-931a-02bb95b29467), published 2026-01-18: **$150K–$275K plus significant equity**; in-person in San Jose. The role asks for host/API tracing, hardware counters, and correlated events across CPUs, drivers, PCIe, accelerators, and hosts.

The recurring requirement is not merely emitting instrumentation. It is producing reliable, reviewable evidence that context and performance events correlate across system boundaries.

## Evidence map

### Already demonstrated

- The benchmark can add a fresh sampled W3C `traceparent` to each live HTTP attempt without persisting identifiers.
- The OpenAI-compatible client exercises real HTTP/SSE serialization and reports streaming TTFT, inter-chunk latency, output bytes, and server-reported token usage.
- The benchmark already has fail-closed regression, correctness, and telemetry gates.
- `secure-gpu-inference-gateway` accepts W3C context, creates a child server span, returns a child `traceparent`, and can export sanitized spans.

### Present but buried

- A trace-aware HTTP server can return a response `traceparent` that retains the request trace ID while using its own child span ID.
- The benchmark has the outbound request header in scope while reading the response, so it can validate same-trace continuation without retaining either identifier.
- The deterministic SSE fixture can exercise this contract through the real CLI.

### Missing

- The benchmark currently records only that propagation was configured and says server acceptance is not verified.
- It does not inspect response context, distinguish missing/invalid/mismatched response headers, report coverage, or fail a qualification when continuation is absent.
- Public artifacts therefore cannot establish even the bounded HTTP contract that a trace-aware server returned context from the same trace.

## Decision

Extend this repository rather than create another project. It already owns the live client, artifact schema, CLI, deterministic fixture, privacy tests, and release-gate behavior.

Implement one coherent capability: **a privacy-safe response trace-continuation observation and optional fail-closed gate for OpenAI-compatible streaming runs**.

## Implementation

1. Classify a successful HTTP response as `matched`, `missing`, `invalid`, or `mismatched` by validating a version-`00` W3C `traceparent` and comparing its trace ID with the outbound request in memory.
2. Retain only the classification. Never retain the request or response trace ID, span ID, or full header in an inference result or shareable artifact.
3. Aggregate measured successful responses into counts, match coverage, and an explicit scope note. Do not call a matching response proof of span export, collector delivery, sampling, clock synchronization, or accelerator attribution.
4. Add `--fail-on-trace-context-gap`, valid only with OpenAI-compatible mode and `--propagate-trace-context`. Exit non-zero when any measured request fails or any successful response is missing, invalid, or on a different trace.
5. Exercise the real CLI against the deterministic SSE fixture. The fixture returns child response context on the same trace, and the artifact check verifies complete continuation coverage and identifier privacy.
6. Add unit tests for every classification, default-off behavior, aggregation, CLI validation, and Prometheus output.

## Verification gates

- Full `unittest` discovery and Python bytecode compilation.
- Real OpenAI-compatible CLI run against the deterministic local HTTP/SSE fixture with the continuation gate enabled.
- Bracketed telemetry fixture and existing fail-closed gates.
- Docker image build and containerized mock execution.
- Supported Python container dependency audit.
- `git diff --check`, pull-request CI, and post-merge main CI.

## Claim boundary

A matched result means the tested HTTP response carried a syntactically valid version-`00` `traceparent` with the same trace ID as the outbound request. This is bounded evidence of response-level trace continuation for that successful request. It does not prove that the server created a span, exported it, preserved baggage or tracestate, delivered it to a collector, synchronized clocks, attributed scheduler/GPU work, or operated at production scale. The deterministic fixture is local protocol evidence, not a production trace.
