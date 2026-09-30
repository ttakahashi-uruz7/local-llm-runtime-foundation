# Dual Runtime Contract v3

- Status: **Canonical / Current**
- Contract version: `runtime-foundation.contract.v3`
- Added schemas: LoadOptions v1, GenerationOptions v1, ExecutionConstraints v1, Execution Binding v3, Execution Evidence v3, Execution Trace v3

## Compatibility boundary

Contract v3 is additive. The v1 and v2 payload shapes, serialization, and
fingerprint meanings remain unchanged. Historical payloads are not silently
upgraded. Explicit v1/v2-to-v3 conversion is available only for fields whose
meaning has an exact v3 counterpart; ambiguous or engine-specific values fail
with an explicit unsupported-option error. The baseline compatibility tests
pin representative v1 and v2 request shapes and fingerprints.

## Engine and artifact are separate dimensions

The internal enum may use `EngineFamily.LLAMA_CPP`; the canonical wire engine
identifier remains `llama.cpp`. At input boundaries, `llama.cpp`, `llama_cpp`,
and `llama-cpp` normalize to that wire identifier. Existing MLX aliases remain
accepted. Artifact format selects a default engine but does not establish
compatibility:

| Artifact format | Default engine | Compatibility rule |
| --- | --- | --- |
| `mlx`, `safetensors` | `mlx` | The selected adapter must declare support for the format. |
| `gguf` | `llama.cpp` | The selected adapter must validate the observed GGUF and declare support. |
| `ggml` | none | Not part of the new GGUF execution contract. |

An explicit engine selection is checked against adapter capability and the
artifact format. Foundation rejects incompatibility before load; it does not
use the default-engine table as the compatibility check. Mock remains an
explicit development choice.

## Three independent option scopes

`LoadOptions` affect model/context state and therefore require explicit unload
before a different effective load is created. Its fields are
`model_context_size`, `batch`, `ubatch`, `threads`, `kv_cache`, and
`acceleration` (backend, device, and GPU offload layers). For GPU offload,
`null` means the adapter default, a non-negative integer selects that many
layers, and `"all"` selects all model layers when supported. A field is
supported only when the selected adapter controls or observes it; unsupported
fields fail closed.

`ExecutionConstraints.max_context_tokens` is a Foundation-enforced budget for
prompt tokens plus requested completion tokens. It is not the model's allocated
context size. A request that cannot be token-counted exactly is rejected before
generation when this budget is set. A budget larger than the loaded context is
explicitly resolved down to that context and recorded.

`GenerationOptions` own per-request values: `max_tokens`, `temperature`,
`top_p`, `top_k`, repetition controls, `stop`, `seed`, and `thinking_intent`.
These fields are not duplicated in the v3 request envelope. Legacy v1/v2
conversion owns the fields already present in those request versions. A
conflicting v2 `thinking_enabled` and `thinking_intent` is an error.
For an unset optional generation field, the selected adapter must resolve a
declared engine default and record the concrete effective value when the
runtime exposes it; otherwise it must report the value as unavailable or
reject the request. Engine-specific unsupported controls are capability
`unsupported` and fail closed when explicitly requested.

Thinking `AUTO` requires the Studio to supply an explicit `ON` or `OFF`
resolution. The adapter must show how the chosen GGUF template/runtime
represents each requested intent. Unsupported effort or token budget is
rejected; it is never dropped silently.

### MLX adapter boundary

The MLX v3 adapter selects the MLX default Metal device at load and records the
device observation; it does not claim independent proof that a particular
kernel executed on Metal. MLX does not expose load-time controls for context
allocation, batch/ubatch, thread count, KV key/value types, or GPU-layer
offload, so those explicit `LoadOptions` fail closed. `max_context_tokens` is a
Foundation preflight budget: Foundation renders the selected tokenizer chat
template, counts with mlx-lm's BOS handling, and passes those same token IDs to
`stream_generate`.

MLX maps `temperature` and `top_p`/`top_k` through `make_sampler`, and maps
repetition penalty/window through `make_logits_processors` when the installed
mlx-lm build exposes those APIs. `top_p: null` resolves to the sampler's
inspectable native default. A null repetition window with an explicit penalty
resolves to the native processor default. Since mlx-lm greedy sampling ignores
`top_p` and `top_k` at temperature zero, explicitly requesting a non-neutral
filter with zero temperature is rejected. A `top_k` value of zero means
disabled where the Engine supports it; MLX resolves `top_k: null` from the
inspected sampler default (zero in the current mlx-lm build), while llama.cpp
requires a positive `top_k`. Foundation trims configured stop strings across
stream chunks and closes the iterator cooperatively; this is not a hard native
termination guarantee. Per-request seed is unsupported because the current
MLX random state is not isolated by this adapter. Thinking intent is supported
only when the loaded tokenizer template can be inspected and exposes
`enable_thinking`; effort and token budgets remain unsupported.

These mappings follow the installed runtime's inspected signatures. The
upstream API references are [mlx-lm generation](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/generate.py)
and [mlx-lm sampler helpers](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/sample_utils.py).

## Binding, evidence, and fingerprints

Execution Binding v3 and Execution Evidence v3 are separate from the pinned v2
Execution Binding. The v3 fingerprint includes the existing Artifact identity,
Engine identity, Engine Build identity, Foundation identity, and effective
LoadOptions, GenerationOptions, and ExecutionConstraints. The load identity
fingerprint includes Artifact identity, Engine/Build identity, and effective
LoadOptions.

Evidence retains requested, resolved, and effective values separately for all
three scopes. Any change between those values requires an explicit resolution
record with the reason. An adapter may reject an unsupported value; it may not
silently replace it.

## Lifecycle and reuse

A loaded model is reusable only when Artifact identity, selected Engine and
Build identity, and effective LoadOptions all match. Any difference returns a
load conflict. Foundation never recreates a context behind a live lease or
generation request. The caller must release leases, unload, and then load the
new identity. Existing wire lifecycle values remain unchanged.

## GGUF observed identity

The existing Artifact Binding v2 remains the single artifact identity system.
For GGUF, Foundation independently reads container magic/version, architecture,
file type/quantization, architecture context metadata, chat-template metadata,
tensor descriptors, and complete-file SHA-256 (`complete-file-v1`). Caller
metadata is treated as a claim and checked against observed values. File
extension is not evidence of GGUF format.

Core opens and validates the file before engine selection, then passes the
observation and descriptor to the llama.cpp adapter. The adapter duplicates
that descriptor and revalidates it immediately before native load. On POSIX it
loads through the descriptor path so an atomic rename cannot substitute another
file. After load it rechecks the pinned descriptor identity. Platform
implementations without a stable descriptor-path load revalidate the file
identity after native load and report this limitation in evidence. The observed
artifact identity is recorded with the execution and load identity.

## Capability and host evidence

V3 option capability uses `supported`, `unsupported`, `unavailable`, or
`unknown`; each record identifies its scope, reload requirement, allowed
values/range/unit, evidence/reason, and applicability to artifact, engine,
build, host, or current load. Engine unavailability is not represented as an
unsupported option.

The v3 host observation separately reports Apple Silicon, host Metal, MLX
availability and MLX default-device Metal, llama.cpp availability and the
selected native build's Metal capability, plus the actual selected execution
acceleration. The legacy `HostProfile.metal_available` keeps its existing MLX
default-device meaning. A Metal-capable Mac does not imply a Metal-enabled
llama.cpp build or a Metal execution.

## API

The versioned service surface is additive: `/v3/host`,
`/v3/engines/{engine}/capability`, `/v3/models/load`, `/v3/generate`,
`/v3/generate/stream`, and `/v3/executions/{id}`. Existing routes and clients
continue to serve v1/v2 contracts.
