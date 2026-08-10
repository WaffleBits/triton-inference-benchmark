# Design Notes

## Benchmark Model

The benchmark separates the harness from the inference client:

- `MockInferenceClient` provides a deterministic, dependency-free workload for CI.
- `TritonHttpInferenceClient` calls a live Triton-compatible inference server over HTTP.
- `OpenAICompatibleStreamingClient` calls an authorized OpenAI-compatible completion endpoint and measures streamed text events.
- `run_benchmark` owns phase ordering, concurrency, retries, timing, and result collection.
- `summarize_results` owns percentile, throughput, success-rate, and streaming aggregation.

This keeps the core logic testable without requiring a GPU, a running model server, or a CUDA runtime.
Live HTTP workers use thread-local Triton clients because the upstream Python HTTP client is not thread-safe.

## Measurement Phases

`--warmup-requests` executes a complete pre-measurement phase through the same
client, worker pool width, and retry policy as the measured phase. The JSON and
Prometheus artifacts preserve warmup outcomes, duration, throughput, and latency
separately. Existing top-level metrics, streaming aggregation, baseline gates,
and cost models use only measured requests.

The phase is deliberately named warmup rather than cold start. The harness does
not restart a model server, reload weights, flush accelerator state, or prove
that a remote endpoint was cold. A future controlled cold-start feature needs
explicit server-lifecycle hooks instead of inferring state from request order.

## Metrics

The tool reports:

- Success and failure counts.
- Success rate.
- End-to-end duration.
- Throughput in requests per second.
- Average, p50, p95, p99, min, and max latency.

When `--prometheus` is enabled, the same core measurements are written as Prometheus text-format gauges and counters beside the JSON result. This keeps the harness dependency-free while making the output easy to archive in CI, push to a metrics gateway, or ingest into a dashboarding workflow.

When `--telemetry-prometheus` is provided, the benchmark attaches a correlated
summary from a Triton/DCGM Prometheus text snapshot. The summary keeps
server-side GPU utilization, memory use, queue duration, request duration, and
inference-duration counters beside the client-side benchmark result. The parser
is dependency-free so CI can validate the behavior with synthetic fixtures while
live runs can still consume real scrape artifacts.

An optional `--telemetry-baseline-prometheus` snapshot turns cumulative Triton
counters into an observed window. The tool subtracts matching, model-filtered
before values from after values, detects aggregate counter decreases and missing
metric families, and derives failure rate plus queue-duration fraction.
Configured telemetry gates fail closed when a derived value is unavailable. The
JSON and Prometheus outputs contain summaries and deltas, not raw scrape text or
operator filesystem paths.

File snapshot alignment is explicitly operator-supplied and unverified. With
`--telemetry-url`, the harness instead fetches one bounded HTTP(S) snapshot after
warmup and immediately before measured requests, then another after the measured
phase completes. This alignment is labeled
`harness_bracketed_measured_phase`; scrape time is excluded from the request
phase duration. The URL, optional bearer token, authorization header, and raw
responses are not serialized.

Harness bracketing does not isolate the server. Counter deltas can include
unrelated traffic. The harness canonicalizes logical metric names and sorted
labels in memory, persists only a SHA-256 series-membership fingerprint plus a
count, and invalidates paired counter evidence when the selected series set
changes. By default, DCGM values are gauges from the post snapshot. A positive
`--telemetry-sample-interval-seconds` starts a separate sampler after measured
requests are submitted. The sampler stops when the request phase completes and
any failed scrape aborts the qualification. Its DCGM values are combined with
the boundary scrapes into sample average, p50, p95, min, and max records with
explicit scrape/value coverage. These are sample statistics, not a time-weighted
integral. Any known GPU-series membership change across those scrapes aborts the
window. Raw label keys and values are not persisted. Matching fingerprints do not
prove physical target identity, health, isolation, or time synchronization.
Authentication is opt-in through an explicitly named environment variable;
ambient API keys are not sent.

## Cost-To-Serve Model

Token and cost inputs attach an estimated cost model to the benchmark result.
The model records input/output tokens per successful request, token throughput,
requests per GPU-hour, GPU-time cost, optional electricity cost, and normalized
cost per million requests or tokens.

GPU capacity is charged for the measured request-phase wall-clock duration,
including time spent on failed measured requests. A configured warmup phase is
reported separately and excluded. Token totals count successful measured
requests only. This keeps measured failures visible as consumed capacity without
crediting them as delivered work. GPU hourly price and electricity are separate
inputs because cloud prices usually bundle facility power while owned-capacity
models may not.

The report excludes CPU, network, storage, idle fleet headroom, and engineering
costs. It is a transparent scenario model for comparing like-for-like runs, not
an accounting claim.

## LLM Decode Metrics

OpenAI-compatible streaming mode measures time to the first non-empty text
event, the gap between subsequent non-empty events, output bytes, and transport
chunk count. It uses `usage.completion_tokens` when the server supplies it and
only reports output-token throughput when every successful request includes
usage. Chunks remain a separate transport metric because one SSE event is not
necessarily one token.

The optional logical LLM record remains available for values that cannot be
inferred from a generic endpoint: context size, decode batch, KV-cache footprint,
bytes read per output token, estimated joules per output token, and same-evaluation
quality delta. Triton and mock runs may also accept caller-supplied TTFT and
inter-token latency when an external measurement exists. Those values remain
labeled as caller-provided rather than measured by the generic Triton client.

Token throughput and requests per GPU-hour reuse the cost-model token counts and
benchmark wall time. Energy uses configured average GPU board power multiplied
by wall time; it does not subtract idle power. Memory traffic is caller-supplied
logical traffic unless a separate hardware-counter artifact is attached.

## Batch-Invariance Probe

`--batch-invariance-probes` checks whether infrastructure scheduling changes model
outputs. Each fixed synthetic input is sent once in isolation and once in a
concurrent workload mixed with unrelated noise requests. The harness fingerprints
all Triton outputs, compares them exactly, records mismatched sample IDs, and can
fail CI with `--fail-on-batch-variance`.

Exact equality is intentionally strict. It is useful for deterministic serving,
prefix-cache validation, replayable rollouts, and debugging numerical changes
caused by dynamic batching or reduction order. Models that intentionally permit
small floating-point drift will eventually need a model-aware tolerance policy;
the current probe reports that drift as a mismatch instead of hiding it.

## Regression Comparison

The `--baseline` option compares a candidate benchmark run with a saved JSON result. The comparison report focuses on release-relevant signals:

- p95 latency percentage change.
- success-rate delta.
- throughput percentage change.
- explicit regression reasons when thresholds are exceeded.

The default gates mark a run as a regression when p95 latency rises by more than 10% or success rate drops by more than 0.01. `--fail-on-regression` makes that comparison exit non-zero for CI. Those thresholds are intentionally CLI-configurable because production latency budgets differ by model, accelerator, queueing policy, and product surface.

## Why Mock Mode Exists

AI infrastructure repos often fail basic review because they cannot run without specialized hardware. Mock mode makes the benchmark harness reviewable anywhere while live Triton mode remains available for real server testing.

## Production Extensions

- Add server-lifecycle hooks for controlled cold-start measurements.
- Add request payload profiles by model family.
- Add distributed load generation across multiple clients.
- Add model-aware numeric tolerance policies for batch-invariance probes.
