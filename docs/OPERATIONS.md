# Operations Notes

This benchmark is intended to produce repeatable evidence for model-serving changes, not one-off timing screenshots.

## Run Pattern

1. Run a baseline benchmark against the current model or serving image.
2. Save the JSON result as the comparison baseline.
3. Run the candidate benchmark with the same warmup count, measured request count, concurrency, payload shape, and retry settings.
4. If available, scrape a Triton/DCGM Prometheus snapshot close to the run.
5. Export JSON and Prometheus text artifacts.
6. Review p95 latency, success rate, throughput, failure count, queue time, and GPU utilization before promoting the candidate.
7. When capacity inputs are known, compare token throughput and normalized cost under identical workload assumptions.

## Workload Profiles

Use `--workload-profile` when a run needs an explicit, repeatable serving shape:

```bash
python benchmark.py \
  --mode triton \
  --workload-profile interactive \
  --server-url localhost:8000 \
  --model-name my-model \
  --num-requests 500 \
  --concurrency 32 \
  --prometheus
```

The built-in profiles are `interactive`, `long-context`, and `throughput`.
They attach context length, generated-token count, logical decode batch, TTFT,
inter-token latency, KV-cache, and byte-read assumptions to the result. They
make baseline and candidate comparisons comparable across serving shapes; they
do not replace measurements from the model or platform under test. Supply an
explicit token or latency flag when measured values are available.

Example:

```bash
python benchmark.py --mode mock --num-requests 500 --concurrency 32 --prometheus
python benchmark.py \
  --mode mock \
  --warmup-requests 32 \
  --num-requests 500 \
  --concurrency 32 \
  --batch-invariance-probes 16 \
  --fail-on-batch-variance \
  --baseline sample_results/mock_run.json \
  --telemetry-baseline-prometheus sample_results/mock_telemetry_before.prom \
  --telemetry-prometheus sample_results/mock_telemetry.prom \
  --max-server-failure-rate 0.02 \
  --max-server-queue-fraction 0.10 \
  --fail-on-telemetry-gate \
  --max-p95-regression-pct 10 \
  --max-success-rate-drop 0.01 \
  --input-tokens-per-request 1024 \
  --output-tokens-per-request 256 \
  --gpu-count 2 \
  --gpu-hourly-cost-usd 4.50 \
  --fail-on-regression \
  --prometheus
```

## Warmup Phase

Use `--warmup-requests` to precondition the same request path before headline
measurement begins. The warmup phase completes before measured work is
submitted. Its success rate, throughput, and latency distribution are saved in
the JSON `warmup` object and in `triton_benchmark_warmup_*` Prometheus metrics.

Warmup requests are excluded from:

- top-level latency, success rate, and throughput
- streaming token and TTFT aggregation
- cost and energy estimates
- baseline regression comparisons

Keep the warmup count identical between a baseline and candidate. Investigate
warmup failures even though they do not contaminate the measured result. Do not
call the first warmup request a cold-start measurement unless an external
lifecycle controller actually restarts the server, reloads the model, and
records that boundary.

## OpenAI-Compatible Streaming Runs

Use `--mode openai` for an authorized vLLM, SGLang, or compatible completion
endpoint. Authentication is opt-in: the API key is read only from the environment
variable explicitly named by `--openai-api-key-env`; it is never accepted as a CLI
value or written to the result. Omitting the flag sends no `Authorization` header.

```bash
export OPENAI_API_KEY="..."
python benchmark.py \
  --mode openai \
  --server-url http://localhost:8000/v1 \
  --model-name local-model \
  --workload-profile interactive \
  --num-requests 100 \
  --concurrency 8 \
  --openai-timeout-seconds 60 \
  --prometheus
```

Review TTFT and inter-chunk p95/p99 alongside end-to-end latency and success
rate. Output-token throughput is valid only when all successful requests return
server usage. `observed_output_chunks` is diagnostic transport evidence and
must not be interpreted as a token count.

Do not run batch-invariance probes in this mode. The streaming client does not
yet fingerprint deterministic outputs, and the CLI rejects that combination.

## SLO-Oriented Checks

For a production-style inference service, the benchmark output should be reviewed against service goals such as:

- Success rate stays at or above the service target for the tested workload.
- p95 and p99 latency do not regress beyond the accepted release threshold.
- Throughput remains stable under the expected concurrency level.
- Retry behavior and failure count are visible in the report, not hidden by averages.
- GPU utilization and queue duration explain whether a latency change is client-side load, accelerator pressure, or server-side scheduling.
- Paired Triton counters stay within the workload's accepted failure-rate and
  queue-fraction thresholds.
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
- separate warmup outcomes, duration, throughput, and latency when configured

The artifact can be pushed to a metrics gateway, archived by CI, or scraped from a shared results volume.

## Telemetry Correlation

Use `--telemetry-prometheus <path>` to attach a Prometheus text snapshot from
Triton and DCGM exporter. The benchmark does not need scrape permissions itself;
it consumes a file captured by CI, a sidecar, or an operator command.

The JSON result includes:

- GPU utilization average and max.
- GPU memory-copy utilization average and max.
- GPU memory used average and max.
- Triton success, failure, request-duration, queue-duration, and compute-infer counters for the configured model.

The Prometheus export mirrors the correlated values with `triton_benchmark_gpu_*` and `triton_benchmark_server_*` metrics so a benchmark artifact can be compared with server-side behavior in the same dashboard.

### Paired counter window and release gate

Triton request and duration metrics are cumulative counters. To evaluate an
observation window, supply a before snapshot with
`--telemetry-baseline-prometheus` and an after snapshot with
`--telemetry-prometheus`. The tool emits counter deltas and derives:

- server failure rate: failed-request delta divided by total-request delta
- server queue fraction: queue-duration delta divided by request-duration delta

Use `--max-server-failure-rate` and `--max-server-queue-fraction` to record
explicit thresholds. Add `--fail-on-telemetry-gate` for exit status 4 when a
threshold is exceeded or cannot be evaluated. Missing counters, counter resets,
and zero denominators fail closed instead of being interpreted as zero.

The files are operator-supplied. The harness cannot prove that they bracket its
own request phase, so the artifact labels alignment as unverified. Capture them
around the intended window with an authorized sidecar or operator workflow. The
post-snapshot DCGM gauges are not window averages. Raw scrapes and source paths
are not serialized into the shareable artifacts. The gate compares aggregate
model counters, not per-replica series membership, so keep the scrape target set
stable across the two snapshots.

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
