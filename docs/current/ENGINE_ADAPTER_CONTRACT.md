# Engine Adapter Contract

- Status: **Canonical / Current**
- Contract version: `runtime-foundation.contract.v1`

## Required adapter operations

Each adapter implements:

1. `discover_capability()`
2. `load(ModelArtifactBinding)`
3. `unload(artifact_id)`
4. `resolve_runtime_options(RuntimeOptions)`
5. `generate(GenerationRequest, cancel_event)`
6. `stream(GenerationRequest, cancel_event)`
7. `cancel(request_id)`
8. `health()`
9. `runtime_metrics()`

The Core never calls an engine library directly. Engine-specific arguments are created only inside the selected adapter.

## MLX primary adapter

The MLX adapter is lazy and import-safe on Windows. On Apple Silicon it observes `mlx`/`mlx-lm`, the default device, package versions, and the callable signatures needed for capability reporting. It passes consumer-provided chat messages through `tokenizer.apply_chat_template`, including an explicit `thinking_enabled` flag when requested. If the tokenizer cannot accept that flag, the adapter returns `unsupported_generation_setting`; it does not silently remove the request.

The reference mapping is:

| Contract setting | MLX/mx-lm mapping | Failure if unavailable |
| --- | --- | --- |
| `context.context_length` | Foundation preflight budget: prompt tokens + `max_tokens` | `context_length_exceeded` when the request exceeds the budget |
| `prefill.chunk_size` | `prefill_step_size` | `unsupported_runtime_option` |
| `kv_cache.bits` | `kv_bits` | `unsupported_runtime_option` |
| `kv_cache.group_size` | `kv_group_size` | `unsupported_runtime_option` |
| `kv_cache.quantization_start` | `quantized_kv_start` | `unsupported_runtime_option` |
| `kv_cache.max_size_tokens` | `max_kv_size` | `unsupported_runtime_option` |
| `temperature` / `top_p` | MLX sampler construction | `unsupported_generation_setting` |
| `acceleration.backend=auto` | resolved to `metal` | explicit option error if not supported |

The current upstream `mlx-lm` API exposes `load`, `stream_generate`, `GenerationResponse`, chat templates, and the KV/prefill arguments used above. The Foundation implementation still marks real execution as **Mac validation pending** because this Windows host cannot validate API behavior, Metal allocation, cache cleanup, or observed throughput.

The adapter loads only a consumer-supplied local path. Foundation v1 does not download a model or mutate a registry.

`GenerationRequest.timeout_ms` is enforced cooperatively by the Core deadline and the adapter's cancellation/deadline checks. The adapter must not silently ignore it. MLX peak memory is recorded only when the engine reports a peak value; current process RSS comes from the host observation helper and is not substituted for peak memory.

## Mock adapter

Mock is deterministic and explicitly labels `measurement_kind=mock` and `measurement_provenance=simulated`. It exercises contract/lifecycle/stream/cancel behavior without fabricating Apple memory, Metal, swap, or production throughput observations.

## llama.cpp boundary

`llama.cpp` / GGUF has a capability and error boundary in v1. Production GGUF execution is not required for this milestone.

## Reference sources

- MLX-LM Python examples: <https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/examples/generate_response.py>
- MLX-LM streaming API and response fields: <https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/generate.py>
