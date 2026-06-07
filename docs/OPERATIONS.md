# Operations Notes

This benchmark is intended to produce repeatable evidence for model-serving changes, not one-off timing screenshots.

## Run Pattern

1. Run a baseline benchmark against the current model or serving image.
2. Save the JSON result as the comparison baseline.
3. Run the candidate benchmark with the same request count, concurrency, payload shape, and retry settings.
4. If available, scrape a Triton/DCGM Prometheus snapshot close to the run.
5. Export JSON and Prometheus text artifacts.
6. Review p95 latency, success rate, throughput, failure count, queue time, and GPU utilization before promoting the candidate.
7. When capacity inputs are known, compare token throughput and normalized cost under identical workload assumptions.

Example:

```bash
python benchmark.py --mode mock --num-requests 500 --concurrency 32 --prometheus
python benchmark.py \
  --mode mock \
  --num-requests 500 \
  --concurrency 32 \
  --batch-invariance-probes 16 \
  --fail-on-batch-variance \
  --baseline sample_results/mock_run.json \
  --telemetry-prometheus sample_results/mock_telemetry.prom \
  --max-p95-regression-pct 10 \
  --max-success-rate-drop 0.01 \
  --input-tokens-per-request 1024 \
  --output-tokens-per-request 256 \
  --gpu-count 2 \
  --gpu-hourly-cost-usd 4.50 \
  --fail-on-regression \
  --prometheus
```

## SLO-Oriented Checks

For a production-style inference service, the benchmark output should be reviewed against service goals such as:

- Success rate stays at or above the service target for the tested workload.
- p95 and p99 latency do not regress beyond the accepted release threshold.
- Throughput remains stable under the expected concurrency level.
- Retry behavior and failure count are visible in the report, not hidden by averages.
- GPU utilization and queue duration explain whether a latency change is client-side load, accelerator pressure, or server-side scheduling.
- Fixed probe inputs retain exact output fingerprints when mixed with concurrent traffic.

This repo does not claim a universal SLO because real targets depend on model size, accelerator type, batch policy, and product latency budget.

## Prometheus Artifact

Use `--prometheus` to write a `.prom` file next to the JSON result. The text-format output includes:

- request totals by outcome
- success rate
- benchmark duration
- throughput
- latency gauges for average, min, max, p50, p95, and p99
- configured concurrency and retry count

The artifact can be pushed to a metrics gateway, archived by CI, or scraped from a shared results volume.

## Telemetry Correlation

Use `--telemetry-prometheus <path>` to attach a Prometheus text snapshot from Triton and DCGM exporter. The benchmark does not need scrape permissions itself; it consumes a file captured by CI, a sidecar, or an operator command.

The JSON result includes:

- GPU utilization average and max.
- GPU memory-copy utilization average and max.
- GPU memory used average and max.
- Triton success, failure, request-duration, queue-duration, and compute-infer counters for the configured model.

The Prometheus export mirrors the correlated values with `triton_benchmark_gpu_*` and `triton_benchmark_server_*` metrics so a benchmark artifact can be compared with server-side behavior in the same dashboard.

## Cost-To-Serve Review

Use token and capacity inputs to attach a scenario estimate to each result. Keep
the same token profile, GPU count, price basis, and power assumptions across
baseline and candidate runs.

Review:

- output and total token throughput
- successful requests per GPU-hour
- accelerator and optional electricity cost for the run
- cost per million successful requests
- cost per million input, output, and total tokens

Treat the numbers as comparison inputs, not invoices. Cloud accelerator prices
usually include electricity, while owned-hardware models may add it separately.
The estimate excludes CPU, storage, network, fleet headroom, and engineering cost.

## Batch-Invariance Triage

Use `--batch-invariance-probes <count>` to compare fixed requests in two layouts:
isolated execution and concurrent execution mixed with unrelated requests. Add
`--fail-on-batch-variance` in CI when exact repeatability is required.

When a mismatch appears:

- Re-run with the same seed and model version.
- Confirm sampling is disabled and all model inputs are deterministic.
- Compare server batching, precision, kernel, and accelerator settings.
- Check whether reduction order or dynamic batching changes floating-point results.
- Treat prefix-cache reuse and resumable rollout replay as unsafe until the mismatch is understood.

## Incident / Regression Triage

When a candidate run is marked as a regression:

- Compare p95 and p99 before looking at averages.
- Check whether failures increased or retries masked transient errors.
- Re-run with the same seed in mock mode to validate harness behavior.
- Re-run live mode with a fixed model version and payload shape.
- Inspect server logs, accelerator telemetry, queue depth, and request batching before changing the benchmark threshold.
- Compare correlated queue duration and GPU utilization before changing concurrency or batching policy.

## Public-Safe Boundaries

Do not commit production prompts, customer payloads, model weights, secrets, traces, or logs. Keep benchmark samples synthetic or generated.
