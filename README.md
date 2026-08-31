# Triton Inference Benchmark

[![CI](https://github.com/WaffleBits/triton-inference-benchmark/actions/workflows/ci.yml/badge.svg)](https://github.com/WaffleBits/triton-inference-benchmark/actions/workflows/ci.yml)

Load-generation harness for Triton-style model serving. It drives concurrent
requests, records latency percentiles, accounts for retries and failures,
exports Prometheus text, and gates candidate runs against a saved baseline. A
dependency-free mock backend runs in CI; an optional HTTP mode drives a real
inference endpoint.

## Features

- Concurrent load generation with configurable request and worker counts.
- Optional open-loop constant-rate pacing for measured submissions, with
  client-side submission-lag statistics kept separate from completion throughput.
- Optional phase-separated warmup requests with their own outcomes, latency,
  throughput, JSON, and Prometheus records; headline and cost metrics remain
  scoped to the measured phase.
- Retry-aware request execution with measured client-attempt amplification,
  recovery/exhaustion accounting, client-observed recovered-request latency,
  an optional fixed between-attempt delay, and a fail gate.
- Opt-in ingress/backend/success counter accounting that can scrape independent
  telemetry endpoints concurrently and reconcile measured client attempts with
  privacy-safe aggregate serving-path evidence.
- Opt-in service-restart counter accounting with minimum/maximum gates, stable
  series validation, and no raw metric identities or endpoint URLs in artifacts.
- Latency metrics: average, p50, p95, p99, min, max, plus throughput and success rate.
- JSON output and Prometheus text export for trend tracking.
- Baseline-versus-candidate comparison with p95 and success-rate gates.
- Paired before/after Triton counter windows with fail-closed server failure-rate
  and queue-fraction gates plus hashed series-membership validation; raw scrapes,
  labels, and operator paths stay out of artifacts.
- Opt-in HTTP(S) Prometheus capture that scrapes after warmup and immediately
  around the measured request phase without sending ambient credentials.
- Optional repeated DCGM gauge sampling across that bracketed window, reporting
  scrape/value coverage plus sample average, p50, p95, min, and max without
  calling the result a time-weighted measurement; target churn rejects the window.
- Named workload profiles for interactive, long-context, and throughput traffic;
  each records context, output, batch, TTFT, decode, and KV-cache assumptions.
- Dependency-free mock backend for CI, an optional Triton HTTP mode, and an
  OpenAI-compatible streaming mode for vLLM/SGLang-style endpoints.
- Measured streaming TTFT, inter-chunk latency, output bytes, and server-reported
  output-token throughput without treating transport chunks as tokens.
- Opt-in W3C `traceparent` propagation for live Triton and OpenAI-compatible
  requests, with fresh identifiers per HTTP attempt and no identifiers in artifacts.
- Isolated-versus-concurrent output gates with exact fingerprints by default and
  opt-in run-scoped numeric tolerances; output values stay in process memory.

## Quick Start

Run a mock benchmark without GPU dependencies:

```bash
python benchmark.py --mode mock --num-requests 100 --concurrency 8
```

Write JSON plus Prometheus text-format artifacts:

```bash
python benchmark.py --mode mock --num-requests 500 --concurrency 32 --prometheus
```

Run a workload-shaped qualification with explicit LLM serving assumptions:

```bash
python benchmark.py \
  --mode mock \
  --workload-profile long-context \
  --num-requests 200 \
  --concurrency 16 \
  --prometheus
```

The named profiles are `interactive`, `long-context`, and `throughput`. They
are transparent starting points for repeatable comparisons, not universal SLOs
or measurements of a particular model. Explicit token and latency flags can
override profile values when an operator has measured workload data.

Precondition the serving path before the measured request window:

```bash
python benchmark.py \
  --mode mock \
  --warmup-requests 20 \
  --num-requests 200 \
  --concurrency 16 \
  --prometheus
```

Warmup requests use the same client, concurrency, and retry policy, but their
outcomes and latency distribution are reported under a separate `warmup`
record. They do not contribute to headline latency, throughput, streaming-token,
regression, or cost calculations. This phase preconditions a serving path; it
does not prove a process, model, or accelerator cold start.

Pace only the measured phase at an explicit offered request rate:

```bash
python benchmark.py \
  --mode mock \
  --warmup-requests 20 \
  --num-requests 200 \
  --concurrency 32 \
  --request-rate-rps 50 \
  --prometheus
```

`--request-rate-rps` uses client-monotonic constant-rate deadlines. The JSON and
Prometheus artifacts keep configured/observed submission rate, submission lag,
executor queue delay, and request-start lag separate from successful-completion
throughput. Concurrency remains the worker cap; requests can queue in the client
executor if service time exceeds that capacity. This is single-process
scheduling evidence, not proof of exact server arrival times, isolated server
queues, synchronized clocks, or distributed load.

Gate the extra client work used to recover failed logical requests:

```bash
python benchmark.py \
  --mode openai \
  --server-url http://localhost:8000/v1 \
  --model-name local-model \
  --num-requests 200 \
  --concurrency 16 \
  --retries 2 \
  --retry-backoff-seconds 0.25 \
  --max-client-attempt-amplification 1.05 \
  --fail-on-retry-gate \
  --prometheus
```

The factor is the number of measured calls to `InferenceClient.infer` divided
by measured logical requests. JSON and Prometheus also separate retried,
recovered, and exhausted requests plus the end-to-end latency distribution for
logical requests recovered by retry. Warmup has its own attempt accounting and is
excluded from the measured gate. A client call can fail before endpoint receipt,
so this is harness-attempt evidence rather than a server request count or proof
of retry traffic reaching a router, model server, or accelerator. Recovered-
request latency is client-observed and includes failed attempts; it is not
service MTTR or proof that a process recovered.

`--retry-backoff-seconds` adds the configured fixed delay only after a failed
attempt when another attempt remains. It is a client policy input recorded with
the run, not measured recovery time, adaptive backoff, or proof that the service
was ready before the next attempt.

Reconcile client attempts with counters from an isolated serving path:

```bash
python benchmark.py \
  --mode openai \
  --server-url http://localhost:8000/v1 \
  --model-name local-model \
  --num-requests 200 \
  --concurrency 16 \
  --retries 2 \
  --telemetry-url http://gateway.local:8000/metrics \
  --telemetry-url http://model-server.local:8001/metrics \
  --request-path-ingress-metric gateway_requests_received_total \
  --request-path-backend-metric model_server_requests_received_total \
  --request-path-success-metric model_server_requests_succeeded_total \
  --fail-on-request-path-gap \
  --prometheus
```

Repeat `--telemetry-url` to capture separate metric sources concurrently at each
boundary; if any source fails, the qualification aborts. All three selected
metric families must be cumulative counters with integer values and stable series
membership across the paired scrapes. The report keeps the endpoint count,
ingress/backend/success deltas, and adjacent-stage ratios, but stores neither the
URLs nor raw scrapes and uses only SHA-256 fingerprints for metric names and label
membership. The opt-in exact gate requires ingress receipts to equal measured
client attempts, stage counts to be non-increasing, and success receipts to equal
successful logical requests.

Use that gate only when the selected series are isolated to this run. Aggregate
counter agreement does not establish per-request causality, correct metric
instrumentation, process identity, traffic isolation, or synchronized clocks.
The same accounting can consume explicitly paired snapshot files, but their
timing is labeled operator-supplied and unverified.

Gate a lifecycle counter alongside the request path when a controller or
orchestrator exposes completed restarts:

```bash
python benchmark.py \
  --mode openai \
  --server-url http://localhost:8000/v1 \
  --model-name local-model \
  --num-requests 200 \
  --concurrency 16 \
  --retries 2 \
  --retry-backoff-seconds 0.25 \
  --telemetry-url http://gateway.local:8000/metrics \
  --telemetry-url http://model-server.local:8001/metrics \
  --telemetry-url http://supervisor.local:8002/metrics \
  --request-path-ingress-metric gateway_requests_received_total \
  --request-path-backend-metric model_server_requests_received_total \
  --request-path-success-metric model_server_requests_succeeded_total \
  --fail-on-request-path-gap \
  --service-restart-metric model_server_completed_restarts_total \
  --min-service-restarts 1 \
  --max-service-restarts 1 \
  --fail-on-service-lifecycle-gap \
  --prometheus
```

Use a maximum of zero for a no-restart reliability gate, or equal minimum and
maximum values for a controlled fault-injection run. The selected family must be
a cumulative integer counter with stable series membership; a missing counter,
reset, duplicate series, or label churn invalidates the gate. The report stores
only the aggregate delta and SHA-256 identity/membership fingerprints. Counter
agreement does not prove process identity, health readiness, per-request
causality, autonomous remediation, traffic isolation, or restart duration.

Compare a candidate run against a saved baseline and fail on regression:

```bash
python benchmark.py \
  --mode mock \
  --num-requests 500 \
  --concurrency 32 \
  --baseline sample_results/mock_run.json \
  --max-p95-regression-pct 10 \
  --max-success-rate-drop 0.01 \
  --fail-on-regression
```

Gate on server counters from two operator-supplied Prometheus snapshots:

```bash
python benchmark.py \
  --mode mock \
  --num-requests 20 \
  --concurrency 4 \
  --telemetry-baseline-prometheus sample_results/mock_telemetry_before.prom \
  --telemetry-prometheus sample_results/mock_telemetry.prom \
  --max-server-failure-rate 0.02 \
  --max-server-queue-fraction 0.10 \
  --fail-on-telemetry-gate \
  --prometheus
```

The committed snapshots are deterministic synthetic fixtures, not observations
from the mock benchmark, a model, or a GPU. In an authorized environment, an
operator or sidecar must capture the first snapshot before and the second after
the intended observation window. The harness computes Triton counter deltas but
cannot prove that supplied files bracket its own invocation. It fails a
configured check when a required counter is missing, resets, or has a zero
denominator, and fails configured counter gates if the selected series membership
changes. DCGM utilization and memory remain post-snapshot gauges rather than
window averages.

Capture the same counter window directly from an authorized Prometheus endpoint:

```bash
export TELEMETRY_TOKEN="..."
python benchmark.py \
  --mode triton \
  --server-url localhost:8000 \
  --model-name resnet50_trt_fp16 \
  --warmup-requests 20 \
  --num-requests 200 \
  --concurrency 16 \
  --telemetry-url http://prometheus.monitoring.svc:9090/federate \
  --telemetry-api-key-env TELEMETRY_TOKEN \
  --telemetry-sample-interval-seconds 1 \
  --max-server-failure-rate 0.02 \
  --max-server-queue-fraction 0.10 \
  --fail-on-telemetry-gate \
  --prometheus
```

The first HTTP scrape runs after warmup and immediately before measured work is
submitted. The second runs immediately after all measured requests finish.
Authentication is opt-in: omit `--telemetry-api-key-env` for unauthenticated
endpoints, and no ambient API key is sent. The bearer token, endpoint URLs, raw
scrapes, and authorization headers are not written to JSON or Prometheus
artifacts. Endpoints must use absolute HTTP(S) URLs without embedded credentials;
the combined snapshot has a shared 10 MiB response budget. When multiple
URLs are configured, one explicitly selected bearer token is sent to every listed
endpoint, so combine only endpoints authorized to receive that credential.

`harness_bracketed_measured_phase` describes process ordering, not server
isolation. Counter deltas can still include unrelated traffic. The harness hashes
logical metric names and sorted labels, persists only the SHA-256 fingerprint and
series count, and invalidates a paired counter window if membership changes. With
no sampling interval, DCGM values remain post-run point gauges. With an explicit
positive interval, a sampler starts after measured requests are submitted and
adds GPU utilization, memory-copy utilization, and memory-use samples until
measured work completes. The artifact combines those values with the two boundary
scrapes and labels the distribution as sampled rather than time-weighted. A
membership change rejects that sampled window. Matching hashes do not prove
physical target identity, target health, isolation from unrelated activity, or
clock synchronization.

## Gate batch-dependent numeric drift

Compare fixed synthetic Triton inputs in isolation and while mixed with
concurrent noise traffic:

```bash
python benchmark.py \
  --mode triton \
  --server-url localhost:8000 \
  --model-name resnet50_trt_fp16 \
  --num-requests 200 \
  --concurrency 8 \
  --batch-invariance-probes 16 \
  --batch-output-atol 0.0001 \
  --batch-output-rtol 0.001 \
  --fail-on-batch-variance \
  --prometheus
```

Zero tolerances preserve exact SHA-256 fingerprint comparison. With either
tolerance set, a numerically different element passes only when
`absolute_error <= atol + rtol * abs(isolated_value)`. Output names, dtypes,
shapes, and element counts must still match exactly; non-numeric and non-finite
differences fail closed. JSON and Prometheus report the run-scoped policy,
exact and tolerance match counts, mismatch classes, and finite aggregate worst
errors. Tensor values, bytes, and fingerprints are not serialized.

Tolerance is not a model-quality claim. The operator must choose it for the
specific model, backend, dtype, and acceptance boundary. This single-client
probe does not prove semantic equivalence, production correctness, traffic
isolation, deterministic kernels, or cross-accelerator parity.

## Sample output (mock backend)

The mock backend generates synthetic latencies so the harness can run in CI
without a server. The numbers below are illustrative mock output, not a
measurement of any real model or hardware. The committed fixture is
[sample_results/mock_run.json](sample_results/mock_run.json).

```json
{
  "mode": "mock",
  "num_requests": 100,
  "concurrency": 8,
  "successful_requests": 100,
  "failed_requests": 0,
  "success_rate": 1.0,
  "throughput_rps": 305.42,
  "latency_ms": {
    "avg": 21.38,
    "p50": 21.74,
    "p95": 33.11,
    "p99": 34.67
  }
}
```

## Run against an OpenAI-compatible LLM endpoint

The streaming mode works with authorized vLLM, SGLang, or other compatible
`/v1/completions` servers. Pass a server root, `/v1` URL, or full completions
URL; the client normalizes all three forms. Bearer authentication is opt-in:
the benchmark sends no `Authorization` header unless `--openai-api-key-env`
explicitly names an environment variable.

```bash
export OPENAI_API_KEY="..."
python benchmark.py \
  --mode openai \
  --server-url http://localhost:8000/v1 \
  --model-name local-model \
  --workload-profile interactive \
  --num-requests 100 \
  --concurrency 8 \
  --openai-max-tokens 128 \
  --openai-api-key-env OPENAI_API_KEY \
  --propagate-trace-context \
  --prometheus
```

For an unauthenticated local server, omit both the export and
`--openai-api-key-env`.

The JSON and Prometheus outputs keep observed transport chunks separate from
server-reported output tokens. They report end-to-end latency, measured TTFT,
measured inter-chunk latency, output bytes, and token throughput only when every
successful streamed response supplies usage data. With complete usage coverage,
server-reported output tokens also replace configured estimates in the cost and
logical LLM summaries. This prevents profile assumptions from being presented as
measured throughput.

The default prompt is synthetic; do not put production prompts, outputs,
endpoint credentials, or private URLs in committed artifacts.

`--propagate-trace-context` adds a fresh sampled W3C `traceparent` to every
physical HTTP attempt, including retries, in `openai` and `triton` modes. The
flag is off by default and is rejected in mock mode. The generated trace ID,
parent ID, and full header are never written to JSON or Prometheus output. This
allows a trace-enabled server to continue request context.

In OpenAI-compatible mode, the benchmark also classifies each successful
response `traceparent` as matched, missing, invalid, or mismatched. A match means
the response header is valid version-`00` context with the request trace ID and a
different span ID. Artifacts retain only counts and match coverage; identifiers
remain in memory. Add `--fail-on-trace-context-gap` to exit with status 5 unless
every measured response matches. This gate does not prove server span creation
or export, collector delivery, sampling, clock synchronization, or accelerator
attribution.

```bash
python benchmark.py \
  --mode openai \
  --server-url http://localhost:8000 \
  --model-name my-model \
  --propagate-trace-context \
  --fail-on-trace-context-gap \
  --prometheus
```

## Run against a real endpoint

Point the harness at a live Triton HTTP server to measure real latency:

```bash
pip install -r requirements.txt
python benchmark.py \
  --mode triton \
  --server-url localhost:8000 \
  --model-name resnet50_trt_fp16 \
  --input-name input \
  --input-shape 1,3,224,224 \
  --num-requests 500 \
  --concurrency 32 \
  --prometheus
```

In this mode every latency, throughput, and success-rate value comes from the
server under test rather than the mock generator.

## Test

```bash
python -m unittest discover -s tests
```

## More

- `DESIGN.md` covers the benchmark model and production extensions.
- `docs/OPERATIONS.md` covers regression triage and Prometheus export usage.
- `deploy/kubernetes/benchmark-job.yaml` shows a cluster-run shape.

## Roadmap

- Server-lifecycle hooks for controlled cold-start measurements.
- Coordinated distributed load generation for multi-client benchmarking.
- Exercise the multi-source path gate in a real orchestrated router/model-server
  deployment; the committed qualification remains a synthetic single-host fixture.
