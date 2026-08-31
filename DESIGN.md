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

## Arrival Schedule

By default, each phase submits work immediately and concurrency limits active
workers. A positive `--request-rate-rps` changes only the measured phase to an
open-loop constant-rate schedule. Deadlines are computed from a client-monotonic
clock and are independent of prior request completions. `--concurrency` remains
the worker cap, so executor work can queue when service demand exceeds available
client workers.

The artifact reports the configured rate, scheduled/observed submission and
request-start spans, observed submission rate, submission lag, executor queue
delay, and request-start lag. These are client observations; successful-
completion throughput remains a separate metric. The scheduler does not
establish exact server arrival times, queue isolation, synchronized clocks,
distributed traffic, or service capacity.

## Metrics

The tool reports:

- Success and failure counts.
- Success rate.
- End-to-end duration.
- Throughput in requests per second.
- Configured and observed client submission rate plus submission lag when paced.
- Average, p50, p95, p99, min, and max latency.

## Retry Accounting

Every logical request records how many times the harness called
`InferenceClient.infer`. The measured summary reports total client attempts,
attempts after the first call, retried requests, successful recovery after a
retry, exhausted requests, and client-attempt amplification. The optional gate
compares that measured factor with an explicit run-scoped maximum.

For logical requests that succeed after retry, the summary also reports their
client-observed end-to-end latency distribution. That duration includes failed
attempts and retry work; it is not service MTTR or evidence that a process was
restarted or healed.

Warmup uses the same retry policy but keeps its attempt summary separate and
does not contribute to the measured gate. End-to-end successful latency already
includes time spent in failed attempts. Attempt counts do not establish server
receipt: DNS, connection, TLS, client serialization, or transport failures can
occur before an endpoint observes a request. The artifact therefore does not
label client-attempt amplification as server load amplification.

## Request-Path Counter Accounting

Three optional operator-selected Prometheus counter families can bridge that
client-only boundary: ingress receipts, backend receipts, and successful backend
completions. The feature reuses the paired telemetry snapshots around the
measured phase. It requires integer cumulative values, stable label membership,
no duplicate series, and no per-series reset. Multiple stable series are summed
only after each one passes those checks.

The artifact retains fixed stage names, aggregate deltas and ratios, plus
SHA-256 fingerprints of the selected metric names and stage/label membership.
Raw metric names, labels, and scrape text are excluded. The fixed accounting
gate checks exact ingress-to-client-attempt equality, non-increasing stage
counts, and exact success-to-successful-logical-request equality.

That gate is intentionally opt-in because exact reconciliation requires the
selected series to be isolated to the benchmark. `--telemetry-url` is repeatable;
configured sources are scraped concurrently at each boundary, every scrape must
succeed, and the combined response remains bounded. Artifacts retain the source
count but not URLs or raw responses. Harness-bracketed aggregate counters still
do not establish per-request causality, correct instrumentation, process
identity, traffic isolation, or synchronized clocks. Operator-supplied files
have an additional unverified-timing boundary.

The deterministic multi-process qualification runs separate local router and
backend OS processes, injects one router-local and one backend failure on
different logical requests, and validates independent ingress/backend/success
counters. It is synthetic single-host evidence, not a deployed router/model
server, orchestrated recovery, production traffic, or service MTTR.

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

File snapshot alignment is explicitly operator-supplied and unverified. With one
or more `--telemetry-url` options, the harness instead fetches bounded HTTP(S)
snapshots concurrently after warmup and immediately before measured requests,
then fetches them again after the measured phase completes. This alignment is labeled
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

## Live Request Trace Context

`--propagate-trace-context` is an explicit opt-in for the Triton and
OpenAI-compatible HTTP clients. Every physical request attempt receives a new
sampled W3C `traceparent` containing a non-zero random 128-bit trace ID and
non-zero random 64-bit parent ID. A retry receives a new context rather than
reusing the failed attempt's identifiers. The benchmark does not read ambient
OpenTelemetry configuration or add `tracestate`.

Trace IDs, parent IDs, and full header values are not retained in request
results, JSON, or Prometheus text. In OpenAI-compatible mode, the client validates
a version-`00` response `traceparent`, requires a different response span ID, and
compares the response trace ID with the request trace ID in memory. The retained
observation is only `matched`, `missing`, `invalid`, or `mismatched`.

The aggregate records the four counts, match coverage, and completeness.
`--fail-on-trace-context-gap` makes any failed request or non-matching successful
response exit with status 5. The option requires OpenAI-compatible mode plus
propagation; Triton's generic client does not expose response headers through the
same interface. A match is bounded response-level continuation evidence. It does
not prove span creation/export, collector delivery, sampling behavior,
request-to-GPU attribution, or clock synchronization. The deterministic local
SSE fixture verifies HTTP wiring, continuation classification, and artifact
privacy, not production tracing.

## Batch-Invariance Probe

`--batch-invariance-probes` checks whether infrastructure scheduling changes model
outputs. Each fixed synthetic input is sent once in isolation and once in a
concurrent workload mixed with unrelated noise requests. The harness captures an
exact fingerprint plus numeric Triton tensors in process memory and can fail CI
with `--fail-on-batch-variance`.

Exact equality remains the zero-tolerance default. `--batch-output-atol` and
`--batch-output-rtol` opt one run into element-wise numeric comparison using
`absolute_error <= atol + rtol * abs(isolated_value)`. Output names, dtypes,
shapes, and element counts remain exact invariants. Non-numeric, structurally
different, and non-finite output differences fail closed. Reports separate exact
matches from tolerance matches and publish finite aggregate errors, but never
tensor values, bytes, or fingerprints.

The policy is run-scoped rather than automatically model-safe. An operator must
select tolerances for the model, backend, dtype, accelerator, and quality target.
Passing does not prove semantic equivalence, traffic isolation, deterministic
kernels, or cross-accelerator correctness.

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
- Exercise multi-source request-path accounting in an orchestrated router and
  model-server deployment rather than only the committed single-host fixture.
