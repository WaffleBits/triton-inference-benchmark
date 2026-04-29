# Triton Inference Benchmark

[![CI](https://github.com/WaffleBits/triton-inference-benchmark/actions/workflows/ci.yml/badge.svg)](https://github.com/WaffleBits/triton-inference-benchmark/actions/workflows/ci.yml)

Benchmark harness for Triton-style model serving. The tool supports a dependency-free mock mode for CI and an optional live HTTP mode for testing a real inference server endpoint.

## Why This Exists

Inference infrastructure work is not just "run a model." Strong systems need repeatable benchmarks, clear latency percentiles, failure accounting, configurable concurrency, and results that can be compared over time. This repo is a small but reviewable artifact around that workflow.

## Features

- Concurrent load generation with configurable request count and worker count.
- Retry-aware request execution.
- Latency metrics: average, p50, p95, p99, min, max.
- Throughput and success-rate reporting.
- Dependency-free mock mode for CI and reviewer demos.
- Optional Triton HTTP mode for live model-serving benchmarks.
- JSON output for trend tracking and regression analysis.

## Engineering Scope

This repo focuses on repeatable load generation, concurrency controls, retry accounting, percentile latency, throughput, and machine-readable output that can feed regression tracking.

Relevant areas:

- AI infrastructure: model-serving reliability, latency analysis, failure accounting, and benchmark methodology.
- Platform engineering: CLI design, JSON artifacts, CI-friendly mock mode, and extension points for live services.
- Performance engineering: percentile metrics, concurrency sweeps, throughput measurement, and reproducible comparison paths.

## Reviewer Fast Path

- Start with `benchmark.py` for benchmark orchestration and CLI behavior.
- Review `tests/` for metric and execution coverage.
- Read `DESIGN.md` for benchmark tradeoffs and production extensions.
- Read `docs/PORTFOLIO_REVIEW.md` for the technical review guide.

## Quick Start

Run a local mock benchmark without GPU dependencies:

```bash
python benchmark.py --mode mock --num-requests 100 --concurrency 8
```

Run against a live Triton endpoint:

```bash
pip install -r requirements.txt
python benchmark.py \
  --mode triton \
  --server-url localhost:8000 \
  --model-name resnet50_trt_fp16 \
  --input-name input \
  --input-shape 1,3,224,224 \
  --num-requests 500 \
  --concurrency 32
```

## Example Output

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

## Test

```bash
python -m unittest discover -s tests
```

## Design Notes

See `DESIGN.md` for the benchmark model, tradeoffs, and production extensions.

## Engineering Notes

This project covers benchmarking discipline, model-serving concepts, latency percentiles, failure accounting, and a clean path from local mock testing to live inference measurement.

## Gaps Worth Closing Next

- Add baseline-versus-candidate comparison mode.
- Add Prometheus export or a dashboard-ready report format.
- Add warmup windows and separate cold-start metrics.
- Add payload profiles for chat, embeddings, vision, and long-context workloads.
- Add distributed load generation for multi-client benchmarking.
