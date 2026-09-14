# Runtime Contract

- Status: **Canonical / Current**
- Contract version: `runtime-foundation.contract.v1`
- Foundation version: `0.1.0`

Every top-level response includes `contract_version`. Contract payloads are JSON-safe and may be stored by a consumer as raw execution evidence. The Foundation does not persist a Benchmark profile.

## First-class contracts

| Contract | Role |
| --- | --- |
| `HostProfile` | Point-in-time host observation. |
| `EngineIdentity` | Engine name, version, and build. |
| `EngineCapability` | What an installed/available adapter can execute or resolve. |
| `ModelArtifactBinding` | Consumer-supplied artifact id/path/format/hash/revision. |
| `RuntimeOptions` | Requested engine-neutral context/KV/prefill/cache/acceleration settings. |
| `RuntimeSettingsResolution` | Requested settings, effective settings, option status, warnings. |
| `GenerationRequest` | Messages, generation controls, artifact id, consumer lease, and runtime options. |
| `GenerationResult` | Output, token usage, raw metrics, execution id, and trace. |
| `StreamEvent` | `started`, `delta`, `completed`, or `error` NDJSON event. |
| `TokenUsage` | Prompt/completion/total token observations. |
| `RuntimeMetrics` | Raw timing, memory, pressure, swap, failure, cancellation, and cleanup observations. |
| `LoadResult` / `UnloadResult` | Lifecycle operation results and lease information. |
| `HealthResult` | Foundation/service health and lifecycle, without eligibility. |
| `RuntimeErrorRecord` | Stable machine-readable error shape. |
| `ExecutionTrace` | Execution id, engine, binding, requested/effective options, host, metrics, finish/error. |

## Runtime options

```json
{
  "contract_version": "runtime-foundation.contract.v1",
  "schema_version": "runtime-foundation.runtime-options.v1",
  "context": {"context_length": 32768, "sliding_window": null},
  "kv_cache": {
    "mode": "quantized",
    "precision": "int8",
    "bits": 8,
    "group_size": 64,
    "quantization_start": 0,
    "max_size_tokens": null,
    "cache_limit_bytes": null
  },
  "prefill": {"chunk_size": 512, "batch_size": 1},
  "prompt_cache": {"enabled": false, "max_entries": null, "max_size_tokens": null},
  "acceleration": {"backend": "auto", "device": null, "threads": null},
  "engine_options": {}
}
```

`requested_runtime_settings` and `effective_runtime_settings` are separate fields. An adapter may resolve `acceleration.backend=auto` to `metal` or `cpu`; the resolution is recorded. Unsupported or unavailable options produce an explicit error or capability status. Silent fallback is not allowed.

Foundation can execute quantized KV settings when an adapter supports them. Whether that setting is acceptable for a consumer's Max Quality policy is outside this contract.

## Raw metrics

The metrics payload can include load/unload duration, cold/warm TTFT, prefill tokens and duration, prefill/generation throughput, completion tokens, process/peak memory, memory pressure, swap before/after/delta, context length, engine failure classification, Metal allocation failure, context failure, timeout, cancellation, and cleanup status. Missing observations are `null` or an explicit `unavailable` provenance; they are not inferred.

## Error shape

```json
{
  "contract_version": "runtime-foundation.contract.v1",
  "error": {
    "code": "unsupported_runtime_option",
    "message": "MLX does not support prompt cache",
    "retryable": false,
    "details": {"path": "prompt_cache.enabled", "execution_id": "execution_..."}
  }
}
```

Important error codes include `artifact_not_found`, `engine_unavailable`, `engine_runtime_error`, `model_not_loaded`, `runtime_busy`, `load_conflict`, `unload_conflict`, `unsupported_runtime_option`, `invalid_runtime_option`, `unsupported_generation_setting`, `cancelled`, and `runtime_timeout`.

## Stream

`POST /generate/stream` returns newline-delimited JSON. The first event is `started`; zero or more `delta` events follow; the terminal event is `completed` with a `GenerationResult` payload or `error` with a `RuntimeError` payload. The service does not convert stream events into a Benchmark score.
