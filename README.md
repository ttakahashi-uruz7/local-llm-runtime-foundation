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

The Windows development path uses `MockAdapter` only when the consumer explicitly requests `engine="mock"`. An MLX/GGUF artifact never silently falls back to Mock. MLX and Metal are imported lazily and are expected to be unavailable on Windows. Foundation v1 does not download models, commit weights, call cloud inference, or modify consumer configuration.

`RuntimeSettingsResolution` is consumer-visible raw evidence. Successful generate results, streaming `completed` results, and `ExecutionTrace` expose the same requested/effective settings, `option_status`, and `warnings`; consumers must preserve these values rather than infer them. Unload preserves the adapter's raw `cleanup_status` (`clean` or `cleanup_error`) in `UnloadResult.raw`. `GenerationRequest.timeout_ms` remains a Foundation Core cooperative runtime timeout.

The Python client preserves Foundation remote errors without policy reinterpretation: `RemoteRuntimeError` exposes the wire `code`, `message`, strict-boolean `retryable`, `details`, and HTTP `status_code`. Consumers must use the Foundation-provided retryability and must not reconstruct it from an error code.

## Contract and service

- Contract version: `runtime-foundation.contract.v1`
- Foundation version: `0.1.1`
- Default local service: `http://127.0.0.1:8765`
- Service bind is loopback-only (`127.0.0.1`, `::1`, or `localhost`); unauthenticated LAN/remote binding is rejected.
- `GET /health`, `GET /host`, `GET /engines`
- `GET /engines/{engine}/capability`
- `POST /models/load`, `POST /models/unload`
- `POST /generate`, `POST /generate/stream`
- `POST /requests/{request_id}/cancel`
- `GET /runtime/metrics`, `GET /executions/{execution_id}`

See [RUNTIME_CONTRACT.md](docs/current/RUNTIME_CONTRACT.md) and [INTEGRATION_GUIDE.md](docs/current/INTEGRATION_GUIDE.md).

## Repository history reference

The initial extraction was designed from Benchmark Studio PR #10 (`bc81701aefaf04aba37d38d7272e255e10db8eed`) against the stated Benchmark main (`a18baef9e32644c813ea715855cfe103fd606f88`). PR #10 remains open and is not closed, merged, or repurposed by this repository.
