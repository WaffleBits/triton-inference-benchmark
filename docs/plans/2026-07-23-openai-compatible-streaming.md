# OpenAI-compatible streaming benchmark implementation plan

**Goal:** Add a dependency-free live LLM streaming mode that records TTFT, inter-chunk latency, output volume, end-to-end latency, and token throughput from OpenAI-compatible endpoints.

**Architecture:** Keep the current benchmark harness and result model. Add an `openai` client that posts a synthetic completion request with streaming enabled, parses server-sent events, and returns a structured observation to the existing retry/concurrency layer. Existing mock and Triton behavior must not change.

**Tech stack:** Python standard library, `unittest`, local `ThreadingHTTPServer`, JSON, SSE.

---

### Task 1: Define streaming observations

**Files:**
- Modify: `benchmark.py`
- Test: `tests/test_benchmark.py`

1. Add a frozen observation dataclass containing TTFT, inter-chunk latency, observed output chunks, reported output tokens, and generated text bytes.
2. Extend `InferenceResult` with optional streaming fields.
3. Update `execute_with_retries` so existing clients returning `None` behave exactly as before while streaming observations are preserved.
4. Add tests for successful observation capture and retry behavior.

### Task 2: Implement the OpenAI-compatible client

**Files:**
- Modify: `benchmark.py`
- Create: `tests/test_openai_streaming.py`

1. Add URL normalization for a server root, `/v1`, or full `/v1/completions` URL.
2. Build a bounded non-chat completion request using synthetic prompt text, `stream=true`, deterministic sampling, and optional API key from a named environment variable.
3. Parse `data:` SSE events until `[DONE]`.
4. Measure first non-empty text event and gaps between subsequent non-empty events.
5. Prefer server-reported `usage.completion_tokens`; otherwise retain an explicit observed-chunk count rather than pretending chunks are tokens.
6. Test against a local streaming HTTP server, including malformed JSON and missing-output failure cases.

### Task 3: Aggregate and export measured streaming metrics

**Files:**
- Modify: `benchmark.py`
- Test: `tests/test_benchmark.py`

1. Add streaming request count, TTFT p50/p95/p99, inter-chunk p50/p95/p99, output tokens, observed chunks, and output bytes to the JSON summary.
2. Calculate measured output-token throughput only when reported token counts exist.
3. Export the same fields in Prometheus text format.
4. Label fallback chunk counts separately from token counts.

### Task 4: Add CLI configuration

**Files:**
- Modify: `benchmark.py`
- Test: `tests/test_benchmark.py`

1. Add `openai` to `--mode`.
2. Add bounded flags for prompt text, maximum output tokens, request timeout, and API-key environment variable name.
3. Reuse workload-profile output-token defaults when an explicit maximum is absent.
4. Reject batch-invariance mode for the streaming client until deterministic output fingerprinting is implemented.

### Task 5: Update public documentation

**Files:**
- Modify: `README.md`
- Modify: `DESIGN.md`
- Modify: `docs/OPERATIONS.md`
- Modify: `docs/PORTFOLIO_REVIEW.md`

Document an authorized vLLM/SGLang-style endpoint run, metric semantics, security boundaries, and the difference between observed chunks and reported tokens. Do not commit prompts, outputs, endpoint URLs, or credentials.

### Task 6: Verify and publish

Run:

```bash
python -m unittest discover -s tests
python benchmark.py --mode mock --num-requests 20 --concurrency 4
python -m compileall -q benchmark.py tests
```

Expected: all tests pass, mock behavior remains intact, and no generated secret or endpoint data is committed.

Then rebuild the portfolio site and résumé, inspect the git diff, commit each repository, and push if authenticated GitHub write access is available.
