# Operations Notes

This benchmark is intended to produce repeatable evidence for model-serving changes, not one-off timing screenshots.

## Run Pattern

1. Run a baseline benchmark against the current model or serving image.
2. Save the JSON result as the comparison baseline.
3. Run the candidate benchmark with the same request count, concurrency, payload shape, and retry settings.
4. Export JSON and Prometheus text artifacts.
5. Review p95 latency, success rate, throughput, and failure count before promoting the candidate.

Example:

```bash
python benchmark.py --mode mock --num-requests 500 --concurrency 32 --prometheus
python benchmark.py \
  --mode mock \
  --num-requests 500 \
  --concurrency 32 \
  --baseline sample_results/mock_run.json \
  --max-p95-regression-pct 10 \
  --max-success-rate-drop 0.01 \
  --fail-on-regression \
  --prometheus
```

## SLO-Oriented Checks

For a production-style inference service, the benchmark output should be reviewed against service goals such as:

- Success rate stays at or above the service target for the tested workload.
- p95 and p99 latency do not regress beyond the accepted release threshold.
- Throughput remains stable under the expected concurrency level.
- Retry behavior and failure count are visible in the report, not hidden by averages.

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

## Incident / Regression Triage

When a candidate run is marked as a regression:

- Compare p95 and p99 before looking at averages.
- Check whether failures increased or retries masked transient errors.
- Re-run with the same seed in mock mode to validate harness behavior.
- Re-run live mode with a fixed model version and payload shape.
- Inspect server logs, accelerator telemetry, queue depth, and request batching before changing the benchmark threshold.

## Public-Safe Boundaries

Do not commit production prompts, customer payloads, model weights, secrets, traces, or logs. Keep benchmark samples synthetic or generated.
