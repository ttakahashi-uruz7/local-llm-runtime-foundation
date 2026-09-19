# Runtime Contract

- Status: **Canonical / Current**
- Contract version: `runtime-foundation.contract.v1`
- Foundation version: `0.2.0`

Every top-level response includes `contract_version`. Contract payloads are JSON-safe and may be stored by a consumer as raw execution evidence. The Foundation does not persist a Benchmark profile.

## Contract v2 compatibility

`runtime-foundation.contract.v2` is additive to `runtime-foundation.contract.v1`.
Existing v1 request/result/trace payloads remain supported, and historical v1
evidence is not inferred-upgraded. `GET /health` exposes both values through
`supported_contract_versions` (and the compatibility alias
`supported_contracts`). See [RAH1_CONTRACT_V2.md](RAH1_CONTRACT_V2.md) for the
Execution Binding, Artifact Binding, Engine Binding, Foundation Binding,
effective settings fingerprint, Thinking v2, and Execution Guard foundations.
See [RAH2_SAFETY_COMPATIBILITY_GATE.md](RAH2_SAFETY_COMPATIBILITY_GATE.md) for
the active pre-generation enforcement semantics.

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
| `GenerationResult` | Output, token usage, raw metrics, execution id, runtime settings resolution, and trace. |
| `StreamEvent` | `started`, `delta`, `completed`, or `error` NDJSON event. |
| `TokenUsage` | Prompt/completion/total token observations. |
| `RuntimeMetrics` | Raw timing, memory, pressure, swap, failure, cancellation, and cleanup observations. |
| `LoadResult` / `UnloadResult` | Lifecycle operation results and lease information. |
| `HealthResult` | Foundation/service health and lifecycle, without eligibility. |
| `RuntimeErrorRecord` | Stable machine-readable error shape. |
| `ExecutionTrace` | Execution id, engine, binding, requested/effective options, raw runtime settings resolution, host, metrics, finish/error. |

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

Every successful `GenerationResult` and terminal streaming `completed` result includes the first-class `runtime_settings_resolution` object. The same object is present in `ExecutionTrace` and contains `contract_version`, `requested_runtime_settings`, `effective_runtime_settings`, `option_status`, and `warnings`. This is consumer-visible raw evidence from the adapter resolution that produced the execution. Consumers such as Benchmark must persist it and must not reconstruct, infer, or replace `option_status` or `warnings`.

For v2 requests with `execution_guard`, Foundation resolves these effective
settings before comparing the requested `expected_runtime_settings_fingerprint`
and before invoking adapter generation. An effective-settings mismatch is a
hard pre-generation block. A supplied
`expected_execution_binding_fingerprint` is checked against the actual
complete Execution Binding, including artifact, engine, Foundation, adapter,
and effective settings identity. FAST, LEGACY, or unknown identities are not
strict certified matches. Guard PASS/FAIL evidence is preserved in trace v2;
guard failure does not invoke `generate` or `stream` and does not emit a
stream `started` event.

If runtime option resolution fails, Foundation returns the existing machine-readable option error and does not manufacture a successful resolution or an effective-settings payload. The trace retains the requested settings and error context where available.

`context.context_length` is an execution budget, not a silent model reconfiguration. Foundation counts prompt tokens using the selected adapter and enforces `prompt_tokens + max_tokens <= context_length` before and during generation. A violation returns `context_length_exceeded`; Foundation never truncates the prompt or silently reduces the generation budget. MLX advertises this option as a Foundation preflight budget because upstream `mlx-lm` does not expose a generic context-length keyword on `stream_generate`.

`GenerationRequest.timeout_ms` is a cooperative timeout. Foundation starts a deadline, signals the adapter cancellation hook when it expires, and returns `runtime_timeout` with `timeout_semantics=cooperative`. An adapter must check the cancellation/deadline between engine output steps. A blocking engine call may only return after the engine yields control; v1 does not claim hard thread termination.

Foundation can execute quantized KV settings when an adapter supports them. Whether that setting is acceptable for a consumer's Max Quality policy is outside this contract.

## Raw metrics

The metrics payload can include load/unload duration, cold/warm TTFT, prefill tokens and duration, prefill/generation throughput, completion tokens, current process RSS (`process_memory_bytes`), observed engine peak memory (`peak_memory_bytes`), memory pressure, swap before/after/delta, context length, engine failure classification, Metal allocation failure, context failure, timeout, cancellation, and cleanup status. `process_memory_bytes` is never populated from a peak-only counter such as Unix `ru_maxrss`; if current RSS is unavailable it is `null`. Missing observations are `null` or an explicit `unavailable` provenance; they are not inferred.

`UnloadResult.raw.cleanup_status` is the adapter's raw cleanup observation when unload executes. `clean` means the adapter reported cleanup success; `cleanup_error` means it reported a cleanup failure. A missing or unknown value is unresolved evidence, not a success signal. Consumers must preserve the raw payload and apply their own policy mapping.

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

Important error codes include `artifact_not_found`, `engine_not_found`, `engine_unavailable`, `engine_runtime_error`, `context_length_exceeded`, `model_not_loaded`, `runtime_busy`, `load_conflict`, `unload_conflict`, `unsupported_runtime_option`, `invalid_runtime_option`, `unsupported_generation_setting`, `cancelled`, and `runtime_timeout`.

The Foundation Python client raises `RemoteRuntimeError` for remote error responses and preserves the wire `code`, `message`, strict-boolean `retryable`, `details`, and HTTP `status_code`. Consumers such as Benchmark must use this Foundation-provided retryability and must not infer or recreate it from `code`.

Guard errors are mechanical execution verification errors. `invalid_request`
with `mismatch_category=invalid_expectation` identifies malformed expected
fingerprints; `execution_guard_mismatch` identifies a resolved difference;
and `execution_binding_unresolvable` identifies an actual state that cannot be
safely certified. Foundation does not convert these outcomes into deployment
eligibility or quality policy.

## Stream

`POST /generate/stream` returns newline-delimited JSON. When the pre-generation
gate passes, the first event is `started`; zero or more `delta` events follow;
the terminal event is `completed` with a `GenerationResult` payload. A guard
failure emits only a terminal `error` event and never emits `started` or a
token delta. The service does not convert stream events into a Benchmark score.
