# Counter-window telemetry gate implementation plan

**Goal:** Turn existing Triton Prometheus correlation into a truthful release gate
by deriving server failure rate and queue share from paired before/after counter
snapshots instead of treating cumulative counters as observed-window values.

**Evidence:** Current high-compensation inference roles repeatedly require
cross-layer telemetry, tail-latency and throughput analysis, correctness/release
validation, and measurable SLOs. This harness already attaches one Triton/DCGM
snapshot, but it cannot attribute cumulative server counters to the benchmark
window or fail a qualification run on server-side evidence.

**Boundary:** DCGM utilization and memory values remain point-in-time snapshot
gauges. Only Triton counters present in both snapshots become observed-window
deltas. Snapshot bracketing is operator-supplied and cannot be proven by the
harness. A counter reset or missing denominator makes the affected gate
unevaluable and therefore failed; no production or GPU measurement is claimed
from synthetic fixtures.

## Tasks

1. Add tests for paired counter deltas, model filtering, counter resets, missing
   denominators, threshold pass/fail behavior, CLI validation, Prometheus export,
   and absence of source paths/raw snapshots in shareable artifacts.
2. Add a pre-run telemetry snapshot option while keeping the existing post-run
   snapshot option compatible.
3. Derive success/failure/request-duration/queue-duration/compute-duration deltas,
   server failure rate, and queue-duration fraction for the observed window.
4. Add configurable maximum failure-rate and queue-fraction checks plus an
   opt-in non-zero exit when a telemetry gate fails or cannot be evaluated.
5. Export window values and gate status to JSON and Prometheus without serializing
   operator paths or raw scrape text.
6. Add deterministic synthetic before/after fixtures and exercise the real CLI.
7. Update the README, design notes, operations guide, portfolio review, and CI
   mock run with the semantic boundaries above.

## Verification

```bash
python -m unittest discover -s tests
python benchmark.py \
  --mode mock \
  --num-requests 20 \
  --concurrency 4 \
  --telemetry-baseline-prometheus sample_results/mock_telemetry_before.prom \
  --telemetry-prometheus sample_results/mock_telemetry.prom \
  --max-server-failure-rate 0.02 \
  --max-server-queue-fraction 0.10 \
  --fail-on-telemetry-gate \
  --prometheus \
  --output-dir /tmp/triton-inference-telemetry-gate-proof
python -m compileall -q benchmark.py tests
docker build -t triton-inference-benchmark:telemetry-gate .
docker run --rm \
  -v "$PWD/sample_results:/fixtures:ro" \
  triton-inference-benchmark:telemetry-gate \
  --mode mock --num-requests 4 --concurrency 2 \
  --telemetry-baseline-prometheus /fixtures/mock_telemetry_before.prom \
  --telemetry-prometheus /fixtures/mock_telemetry.prom \
  --max-server-failure-rate 0.02 \
  --max-server-queue-fraction 0.10 \
  --fail-on-telemetry-gate \
  --output-dir /tmp/results
git diff --check
```

## Verification recorded on 2026-08-02

- `python3 -m unittest discover -s tests`: 45 tests passed.
- Passing mock CLI: 100 successful and 1 failed counter deltas, derived failure
  rate `0.009901`, queue fraction `0.055556`, and gate status `passed: true`.
- Failing mock CLI at a `0.005` maximum failure rate: exit status 4.
- JSON privacy inspection: no fixture paths or raw Prometheus metric names.
- Docker Python 3.12 build and real CLI: passed as UID 10001; synchronous Triton
  import passed with `urllib3 2.7.0` and without the unused `aiohttp` extra.
- `pip-audit -r requirements.txt` in Python 3.12: no known vulnerabilities.
- `python3 -m compileall -q benchmark.py tests` and `git diff --check`: passed.
