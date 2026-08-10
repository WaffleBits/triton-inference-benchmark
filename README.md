# Triton Inference Benchmark

[![CI](https://github.com/WaffleBits/triton-inference-benchmark/actions/workflows/ci.yml/badge.svg)](https://github.com/WaffleBits/triton-inference-benchmark/actions/workflows/ci.yml)

Load-generation harness for Triton-style model serving. It drives concurrent
requests, records latency percentiles, accounts for retries and failures,
exports Prometheus text, and gates candidate runs against a saved baseline. A
dependency-free mock backend runs in CI; an optional HTTP mode drives a real
inference endpoint.

## Features

- Concurrent load generation with configurable request and worker counts.
- Optional phase-separated warmup requests with their own outcomes, latency,
  throughput, JSON, and Prometheus records; headline and cost metrics remain
  scoped to the measured phase.
- Retry-aware request execution with failure accounting.
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
Authentication is opt-in: omit `--telemetry-api-key-env` for an unauthenticated
endpoint, and no ambient API key is sent. The bearer token, endpoint URL, raw
scrapes, and authorization header are not written to JSON or Prometheus
artifacts. The endpoint must be an absolute HTTP(S) URL without embedded
credentials, and each response is bounded to 10 MiB.

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
- Distributed load generation for multi-client benchmarking.
