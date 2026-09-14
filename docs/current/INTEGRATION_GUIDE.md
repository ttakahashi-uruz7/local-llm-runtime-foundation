# Integration Guide

- Status: **Canonical / Current**
- Contract version: `runtime-foundation.contract.v1`

## Common flow

```text
Consumer
  │ artifact binding + requested runtime settings + messages
  ▼
Foundation Client → Local Runtime Service → Foundation Core → Adapter
  ▲                                                           │
  └──────── raw GenerationResult / StreamEvent / Trace ──────┘
```

Start one local Foundation service per local runtime authority. Consumers identify themselves with a stable `consumer_id`, retain the returned `lease_id`, and include those values in later generation/unload calls.

When `engine` is omitted on load, Foundation resolves only the engine mapped to the artifact format (`mlx`/`safetensors` → MLX, `gguf` → llama.cpp). An unavailable mapped engine returns `engine_unavailable`; unknown formats return `engine_not_found`. Mock is available only when the consumer explicitly requests `engine="mock"` and is always marked as simulated.

The unauthenticated v1 service binds to loopback only: `127.0.0.1`, `::1`, or `localhost`. `RUNTIME_FOUNDATION_HOST` rejects `0.0.0.0` and LAN/external addresses. `LocalRuntimeClient` applies the same loopback-only rule to `base_url` at construction time, so remote service access is a future authenticated/TLS milestone rather than a v1 client option.

Foundation does not know a consumer's registry record, quality policy, or production profile. A consumer should store the returned execution trace in its own evidence system if it needs durable lineage.

### Runtime resolution and cleanup evidence

On a successful generate, and on the terminal `completed` event from streaming, read `runtime_settings_resolution` from the result. The same payload is available in the `ExecutionTrace`; it is the adapter's raw resolution evidence and includes requested/effective settings, per-option status, and warnings. Consumers must store these values as received and must not recompute them from capability data.

On unload, preserve `UnloadResult.raw.cleanup_status`. `clean` is a successful cleanup observation, `cleanup_error` is a failure observation, and a missing/unknown value is unresolved. Policy-owning consumers must not treat unresolved cleanup as a pass. `GenerationRequest.timeout_ms` remains a Foundation Core cooperative runtime deadline; an HTTP client timeout must not replace its `runtime_timeout` authority.

### Remote error propagation

The Python `LocalRuntimeClient` raises `RemoteRuntimeError` for Foundation error responses and preserves the wire `code`, `message`, strict-boolean `retryable`, `details`, and HTTP `status_code`. Consumers must use Foundation's retryability as raw authority and must not reconstruct it from an error code.

## Benchmark Studio

Benchmark keeps:

- Model Registry meaning, history, hashes, validation state, and artifact promotion.
- Runtime Optimizer, Runtime Profile, candidate generation/scoring, Safety Gate, Quality Benchmark, Production Runtime Validation, Deployment, and Evidence authority.

Benchmark should remove direct MLX/Metal/host telemetry/generate/load/unload implementation and call the Foundation client instead. Benchmark converts raw observations into its own profile/evidence/policy outcomes; Foundation does not do so.

Suggested request:

```python
from runtime_foundation import GenerationRequest, ModelArtifactBinding
from runtime_foundation.client import LocalRuntimeClient

artifact = ModelArtifactBinding(
    artifact_id="registry-artifact-id",
    local_path="/local/path/supplied-by-registry",
    format="mlx",
    quantization="4bit",
    artifact_hash="sha256:...",
    revision="registry-revision",
)

with LocalRuntimeClient() as runtime:
    lease = runtime.load(artifact, engine="mlx", consumer_id="benchmark")
    request = GenerationRequest(
        model_artifact_id=artifact.artifact_id,
        consumer_id="benchmark",
        lease_id=lease["lease_id"],
        messages=[{"role": "user", "content": "validation prompt"}],
    )
    raw_result = runtime.generate(request)
```

The Benchmark Production Runtime Gate remains the authority for whether a run is valid. It must require observed provenance and its own policy inputs; a Foundation `status=completed` only means the engine completed an execution.

## Novel Studio

Novel uses the same client/service boundary for generation. Novel owns manuscript, role assignment, Production Config revision, switch, rollback, and publish decisions. Foundation receives a request and never writes Novel configuration.

## Learning Studio

Learning may use the Foundation client to check inference behavior of a learned artifact. Learning owns dataset/training/adapter promotion/model profile semantics. Foundation receives only the artifact binding and runtime request.

## Service API

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Foundation/service lifecycle health. |
| `GET /host` | Host observation. |
| `GET /engines` | All engine capabilities. |
| `GET /engines/{engine}/capability` | One capability payload. |
| `POST /models/load` | Load artifact and create/reuse a consumer lease. |
| `POST /models/unload` | Release lease and unload when last lease is released. |
| `POST /generate` | Complete generation. |
| `POST /generate/stream` | NDJSON streaming generation. |
| `POST /requests/{id}/cancel` | Cancellation request. |
| `GET /runtime/metrics` | Raw runtime/service metrics. |
| `GET /executions/{id}` | In-memory execution trace for diagnostics. |

## No automatic acquisition

The consumer must resolve Model Registry/acquisition outside Foundation and pass a local path. Foundation may verify existence and may receive a hash, but it does not download, cache, promote, or register the artifact.
