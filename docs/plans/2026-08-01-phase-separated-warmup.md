# Phase-separated warmup implementation plan

**Goal:** Add an explicit pre-measurement warmup phase so baseline and candidate
runs can precondition a serving path without mixing those requests into the
headline latency, throughput, token, or cost results.

**Evidence:** Current inference-performance roles repeatedly ask for controlled
benchmarking, workload experiments, tail-latency analysis, and root-cause
evidence. The harness already records steady-run percentiles and regression
gates, but cannot currently distinguish setup effects from the measured window.

**Boundary:** A warmup request is not proof of a process, model, or accelerator
cold start. The tool will call this a warmup phase, report it separately, and
exclude it from headline and cost calculations.

## Tasks

1. Add `warmup_requests` to `BenchmarkConfig` and `--warmup-requests` to the CLI.
2. Execute warmup requests before the measured phase with the same client,
   concurrency, and retry policy.
3. Persist a separate warmup summary with request outcomes, wall time,
   throughput, and latency distribution.
4. Keep existing top-level results and regression comparisons scoped to measured
   requests only.
5. Export the warmup summary under separate Prometheus metric names and a
   `phase="warmup"` label.
6. Add tests for call counts, phase isolation, CLI wiring, and Prometheus output.
7. Exercise the real CLI in mock mode and inspect its JSON and Prometheus
   artifacts.
8. Update the README, design notes, operations guide, portfolio review, CI mock
   run, and Kubernetes example without claiming real fleet or GPU evidence.

## Verification

```bash
python -m unittest discover -s tests
python benchmark.py \
  --mode mock \
  --warmup-requests 4 \
  --num-requests 20 \
  --concurrency 4 \
  --prometheus \
  --output-dir /tmp/triton-inference-warmup-proof
python -m compileall -q benchmark.py tests
docker build -t triton-inference-benchmark:phase-warmup .
docker run --rm triton-inference-benchmark:phase-warmup \
  --mode mock --warmup-requests 2 --num-requests 4 --concurrency 2 \
  --output-dir /tmp/results
git diff --check
```