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

The optional additive execution-input boundary is `validate_execution_input()`
followed by `load_execution_input()`. Existing `load(ModelArtifactBinding)`
remains the path for all single-artifact requests. An adapter that cannot
execute a supplied composition must return an explicit unsupported error.

The Core never calls an engine library directly. Engine-specific arguments are created only inside the selected adapter.

## MLX primary adapter

The MLX adapter is lazy and import-safe on Windows. On Apple Silicon it observes `mlx`/`mlx-lm`, the default device, package versions, and the callable signatures needed for capability reporting. Its structured Build Identity includes both distribution versions; if either version is unavailable, the build identity remains unknown. It passes consumer-provided chat messages through `tokenizer.apply_chat_template`, including an explicit `thinking_enabled` flag when requested. If the tokenizer cannot accept that flag, the adapter returns `unsupported_generation_setting`; it does not silently remove the request.

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

The current upstream `mlx-lm` API exposes `load`, `stream_generate`, `GenerationResponse`, chat templates, and the KV/prefill arguments used above. The Foundation execution path has now been validated on a real Apple Silicon Mac for API behavior, Metal allocation, cache cleanup, and observed throughput; see `MAC_PRODUCTION_VALIDATION_RUNBOOK.md`. Windows remains the contract/Mock environment. Build Identity is compatibility provenance, not a performance, model-approval, or Production Eligibility result.

The adapter loads only consumer-supplied local paths. For a one-Adapter
`ExecutionInputV1`, MLX validates adapter config and safetensors metadata plus
the Base module dimensions, then calls `mlx_lm.load(base_path,
adapter_path=adapter_path)`. It does not fuse, save, export, or materialize a
model. A zero-Adapter execution input and every existing single-artifact input
use the existing single-path `load` operation. Foundation does not own a Model
Store, download a model, or mutate a consumer registry. See
[DIRECT_EXECUTION_INPUT_V1.md](DIRECT_EXECUTION_INPUT_V1.md).

`GenerationRequest.timeout_ms` is enforced cooperatively by the Core deadline and the adapter's cancellation/deadline checks. The adapter must not silently ignore it. MLX peak memory is recorded only when the engine reports a peak value; current process RSS comes from the host observation helper and is not substituted for peak memory.

## Mock adapter

Mock is deterministic and explicitly labels `measurement_kind=mock` and `measurement_provenance=simulated`. It exercises contract/lifecycle/stream/cancel behavior without fabricating Apple memory, Metal, swap, or production throughput observations.

## llama.cpp formal adapter

`llama.cpp` is a first-class Execution Engine and `gguf` is its Artifact
Format; the two names are not interchangeable. `llama-cpp-python` and its
bundled native library are an optional dependency selected by the formal design
in [LLAMA_CPP_RUNTIME_DESIGN.md](LLAMA_CPP_RUNTIME_DESIGN.md). The adapter
validates GGUF metadata and content identity, uses only a metadata-selected
chat template, and keeps native arguments inside the adapter boundary.

V3 capabilities report load-scoped context/batch/thread/KV/offload controls
only where they are supported and observed. Requested, resolved, and effective
values remain distinct. Python-visible errors are normalized. The adapter
runs in the Foundation process, so native crashes are not isolated or promised
to be caught. Streaming, cancellation, and timeout are cooperative between
native iterator steps. See
[LLAMA_CPP_RUNTIME_DESIGN.md](LLAMA_CPP_RUNTIME_DESIGN.md) and
[DUAL_RUNTIME_CONTRACT_V3.md](DUAL_RUNTIME_CONTRACT_V3.md).

## Reference sources

- MLX-LM Python examples: <https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/examples/generate_response.py>
- MLX-LM streaming API and response fields: <https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/generate.py>
