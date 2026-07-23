# Triton Inference Benchmark

[![CI](https://github.com/WaffleBits/triton-inference-benchmark/actions/workflows/ci.yml/badge.svg)](https://github.com/WaffleBits/triton-inference-benchmark/actions/workflows/ci.yml)

Load-generation harness for Triton-style model serving. It drives concurrent
requests, records latency percentiles, accounts for retries and failures,
exports Prometheus text, and gates candidate runs against a saved baseline. A
dependency-free mock backend runs in CI; an optional HTTP mode drives a real
inference endpoint.

## Features

- Concurrent load generation with configurable request and worker counts.
- Retry-aware request execution with failure accounting.
- Latency metrics: average, p50, p95, p99, min, max, plus throughput and success rate.
- JSON output and Prometheus text export for trend tracking.
- Baseline-versus-candidate comparison with p95 and success-rate gates.
- Named workload profiles for interactive, long-context, and throughput traffic;
  each records context, output, batch, TTFT, decode, and KV-cache assumptions.
- Dependency-free mock backend for CI, and an optional Triton HTTP mode.

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

- Warmup windows and separate cold-start metrics.
- Distributed load generation for multi-client benchmarking.
- Threshold checks for GPU utilization, queue depth, and server-side errors.
