# Local LLM Runtime Foundation

Local LLM Runtime Foundation is the shared, policy-free execution authority for Novel Studio, Benchmark Studio, and Learning Studio.

It owns how a local model is observed and executed: host observation, engine capability discovery, model load/unload, generation, streaming, cancellation, effective runtime settings, health, and raw runtime metrics. It does not decide whether a runtime is good, production-ready, publishable, or suitable for a Studio workflow.

Current status: **FOUNDATION DEVELOPMENT READY / REAL MAC PRODUCTION VALIDATION PENDING**.

## Architecture

```text
Novel / Benchmark / Learning
             │  Python client / loopback HTTP
             ▼
Local Runtime Service
             ▼
Foundation Core → MLX Adapter (primary) / Mock Adapter / llama.cpp skeleton
```

The service owns one loaded model process. Consumer leases prevent one consumer from unloading or replacing another consumer's loaded artifact. The Core has no dependency on Benchmark Registry, Evidence, Optimizer, Quality, Deployment, Novel, or Learning modules.

## Development

```powershell
python -m pip install -e ".[test]"
python -m pytest --basetemp=.pytest-temp
python -m runtime_foundation
```

The Windows development path uses `MockAdapter` only when the consumer explicitly requests `engine="mock"`. An MLX/GGUF artifact never silently falls back to Mock. MLX and Metal are imported lazily and are expected to be unavailable on Windows. Foundation does not download models, commit weights, call cloud inference, or modify consumer configuration.

`RuntimeSettingsResolution` is consumer-visible raw evidence. Successful generate results, streaming `completed` results, and `ExecutionTrace` expose the same requested/effective settings, `option_status`, and `warnings`; consumers must preserve these values rather than infer them. Unload preserves the adapter's raw `cleanup_status` (`clean` or `cleanup_error`) in `UnloadResult.raw`. `GenerationRequest.timeout_ms` remains a Foundation Core cooperative runtime timeout.

The Python client preserves Foundation remote errors without policy reinterpretation: `RemoteRuntimeError` exposes the wire `code`, `message`, strict-boolean `retryable`, `details`, and HTTP `status_code`. Consumers must use the Foundation-provided retryability and must not reconstruct it from an error code.

## Contract and service

- Contract version: `runtime-foundation.contract.v1`
- Supported contracts: `runtime-foundation.contract.v1`, `runtime-foundation.contract.v2`
- Foundation version: `0.2.0`
- Runtime package / contract baseline commit: `c00328e0356e706b0ab6504ef8fba2219a65778c`
- Repository canonicalization merge: `753bbd935b444542a269bb93d8bb2e42172b5225` (repository operating documentation only; runtime baseline remains `c00328e...`)
- Default local service: `http://127.0.0.1:8765`
- Service bind is loopback-only (`127.0.0.1`, `::1`, or `localhost`); unauthenticated LAN/remote binding is rejected.
- `GET /health`, `GET /host`, `GET /engines`
- `GET /engines/{engine}/capability`
- `POST /models/load`, `POST /models/unload`
- `POST /generate`, `POST /generate/stream`
- `POST /requests/{request_id}/cancel`
- `GET /runtime/metrics`, `GET /executions/{execution_id}`

See [RUNTIME_CONTRACT.md](docs/current/RUNTIME_CONTRACT.md) and [INTEGRATION_GUIDE.md](docs/current/INTEGRATION_GUIDE.md).

RAH-1 v2 details are in [RAH1_CONTRACT_V2.md](docs/current/RAH1_CONTRACT_V2.md), and active RAH-2 guard enforcement is documented in [RAH2_SAFETY_COMPATIBILITY_GATE.md](docs/current/RAH2_SAFETY_COMPATIBILITY_GATE.md). v2 separates registry artifact identity, complete/fast content identity, and locator; separates MLX engine family from `mlx-lm` implementation; binds Foundation and effective runtime settings; exposes deterministic execution fingerprints; and hard-blocks guarded execution before adapter generation when expected state does not match or cannot be certified. The Foundation remains policy-free. A second MLX runtime, real llama.cpp execution, LM Studio integration, and production performance conclusions are deferred.

## Repository history reference

The initial extraction was designed from Benchmark Studio PR #10 (`bc81701aefaf04aba37d38d7272e255e10db8eed`) against the then-stated Benchmark main (`a18baef9e32644c813ea715855cfe103fd606f88`). Benchmark PR #10 was later merged into Benchmark main as `618dc037694665dd7ccbdbe995cc52c8e2515fb6` on 2026-09-14. These SHAs are historical extraction/integration anchors, not Current moving-main pointers; Benchmark main has advanced since that merge.
