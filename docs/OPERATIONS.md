# Operations Notes

This benchmark is intended to produce repeatable evidence for model-serving changes, not one-off timing screenshots.

## Run Pattern

1. Run a baseline benchmark against the current model or serving image.
2. Save the JSON result as the comparison baseline.
3. Run the candidate benchmark with the same warmup count, measured request count, concurrency, offered request rate, payload shape, and retry settings.
4. If available, scrape Triton/DCGM Prometheus counters around the measured
   phase with `--telemetry-url`, or supply operator-captured files.
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
  --batch-output-atol 0.0001 \
  --batch-output-rtol 0.001 \
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
  --propagate-trace-context \
  --prometheus
```

Review TTFT and inter-chunk p95/p99 alongside end-to-end latency and success
rate. Output-token throughput is valid only when all successful requests return
server usage. `observed_output_chunks` is diagnostic transport evidence and
must not be interpreted as a token count.

Do not run batch-invariance probes in this mode. The streaming client does not
yet fingerprint deterministic outputs, and the CLI rejects that combination.

### Open-loop request pacing

Use `--request-rate-rps` to offer measured requests at a client-side constant
rate instead of submitting the full phase immediately:

```bash
python benchmark.py \
  --mode openai \
  --server-url http://localhost:8000/v1 \
  --model-name local-model \
  --warmup-requests 20 \
  --num-requests 200 \
  --concurrency 32 \
  --request-rate-rps 50 \
  --prometheus
```

Warmup remains immediate and is excluded from the schedule record. Compare the
configured rate with achieved submission rate, submission lag, executor queue
delay, and request-start lag before interpreting service latency. Submission lag
shows whether the scheduler met its executor deadline; executor queue delay shows
whether a worker was available. Concurrency still caps active workers, so a high
offered rate can queue requests before they reach the server. Do not report
configured request rate as achieved server throughput.

### Trace correlation

Use `--propagate-trace-context` only when the authorized live endpoint is
configured to consume W3C Trace Context. The flag adds a fresh sampled
`traceparent` to each physical OpenAI-compatible or Triton HTTP attempt. It is
off by default, does not read ambient tracing configuration, and does not add
`tracestate`.

The JSON and Prometheus artifacts record that propagation was configured but do
not retain trace IDs, parent IDs, or header values. Correlate downstream spans
inside the authorized collector rather than copying traces into public benchmark
artifacts. A configured flag is not proof that the server accepted the context,
exported spans, respected sampling, or synchronized clocks.

## SLO-Oriented Checks

For a production-style inference service, the benchmark output should be reviewed against service goals such as:

- Success rate stays at or above the service target for the tested workload.
- p95 and p99 latency do not regress beyond the accepted release threshold.
- Throughput remains stable under the expected concurrency level.
- Retry behavior and failure count are visible in the report, not hidden by averages.
- GPU utilization and queue duration explain whether a latency change is client-side load, accelerator pressure, or server-side scheduling.
- Paired Triton counters stay within the workload's accepted failure-rate and
  queue-fraction thresholds.
- Fixed probe inputs remain exact or within an explicitly reviewed run-scoped
  numeric tolerance when mixed with concurrent traffic.

This repo does not claim a universal SLO because real targets depend on model size, accelerator type, batch policy, and product latency budget.

## Prometheus Artifact

Use `--prometheus` to write a `.prom` file next to the JSON result. The text-format output includes:

- request totals by outcome
- success rate
- benchmark duration
- throughput
- latency gauges for average, min, max, p50, p95, and p99
- configured concurrency and retry count
- configured/achieved submission rate, submission lag, executor queue delay, and
  request-start lag when pacing is enabled
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
zero denominators, and changed selected-series membership fail closed instead of
being interpreted as zero.

The files are operator-supplied. The harness cannot prove that they bracket its
own request phase, so the artifact labels alignment as unverified. Capture them
around the intended window with an authorized sidecar or operator workflow. The
post-snapshot DCGM gauges are not window averages. Raw scrapes and source paths
are not serialized into the shareable artifacts. The harness fingerprints the
logical counter metric names and sorted labels, then persists only the digest and
series count. A changed fingerprint invalidates the window. A matching fingerprint
does not prove that labels map honestly to physical targets or exclude unrelated
traffic.

### Harness-bracketed HTTP capture

Use `--telemetry-url <http-or-https-url>` instead of the two file options when
the benchmark process is authorized to read the Prometheus endpoint. The
harness fetches the baseline after warmup and immediately before measured work,
then fetches the candidate immediately after the measured futures complete.

```bash
export TELEMETRY_TOKEN="..."
python benchmark.py \
  --mode triton \
  --server-url localhost:8000 \
  --model-name my-model \
  --warmup-requests 32 \
  --num-requests 500 \
  --concurrency 32 \
  --telemetry-url http://prometheus.monitoring.svc:9090/federate \
  --telemetry-timeout-seconds 10 \
  --telemetry-sample-interval-seconds 1 \
  --telemetry-api-key-env TELEMETRY_TOKEN \
  --max-server-failure-rate 0.02 \
  --max-server-queue-fraction 0.10 \
  --fail-on-telemetry-gate \
  --prometheus
```

Authentication is disabled unless `--telemetry-api-key-env` explicitly names a
non-empty environment variable. The client does not inspect ambient API-key
variables. URLs containing user information are rejected. Each UTF-8 response
is capped at 10 MiB, and a failed scrape aborts the qualification instead of
silently producing an unbracketed artifact.

Artifacts contain only parsed summaries, deltas, and gates. They exclude the
endpoint URL, environment-variable name, bearer token, authorization header,
and raw response. The alignment label proves only this process's phase order;
unrelated server traffic remains an external control, and the post-run DCGM
values are still point-in-time gauges unless sampling is explicitly enabled.
Selected counter-series churn is detected by a label-derived SHA-256 fingerprint,
without publishing the labels.

### Response trace-continuation gate

For an OpenAI-compatible service that returns W3C context, use the opt-in
response gate to require a valid continuation on every measured successful
response:

```bash
python benchmark.py \
  --mode openai \
  --server-url http://inference.example.internal \
  --model-name my-model \
  --propagate-trace-context \
  --fail-on-trace-context-gap \
  --prometheus
```

The client compares the request and response trace IDs only in memory and
requires the response span ID to differ. JSON and Prometheus output contain
classification counts and coverage, never the headers or identifiers. Exit
status 5 means at least one measured request failed or a successful response was
missing context, syntactically invalid, or continued another trace.

Treat a passing gate as proof only of the observed HTTP header contract. It does
not establish that spans were exported or delivered, that sampling occurred,
that clocks align, or that server spans correlate with scheduler, kernel, or GPU
events. Verify those properties in the authorized tracing backend.

With `--telemetry-sample-interval-seconds`, the harness also samples known DCGM
gauges after measured requests have been submitted and until the measured phase
completes. The JSON and Prometheus artifacts report the number of boundary and
in-window scrapes, matched-value coverage, and sample average, p50, p95, min,
and max for GPU utilization, memory-copy utilization, and memory used. The two
boundary scrapes are included in the distribution. These values are not
time-weighted, and a shared scrape can include unrelated activity. The harness
normalizes supported exporter aliases and verifies the known GPU-series
fingerprint across every scrape. A changed fingerprint or failed in-window scrape
aborts the qualification rather than publishing a mixed-membership or partial
sampled window. Only the stable fingerprint and series count are persisted; raw
labels are not.

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
`--fail-on-batch-variance` in CI to enforce the configured policy. Comparison is
exact by default. `--batch-output-atol` and `--batch-output-rtol` opt a run into
numeric comparison while output metadata remains exact. The artifact reports
exact matches, tolerance matches, incompatibility counts, and aggregate worst
finite errors; it does not retain tensor values, bytes, or fingerprints.

Treat tolerances as model/backend/dtype-specific release inputs, not universal
defaults. Increase one only after a quality owner has established an acceptable
numeric boundary. Non-numeric, non-finite, shape, dtype, and output-name changes
fail closed when fingerprints differ.

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

Do not commit production prompts, customer payloads, model weights, secrets,
trace identifiers, traces, or logs. Keep benchmark samples synthetic or generated.
