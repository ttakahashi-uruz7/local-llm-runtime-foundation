# Local LLM Runtime Foundation Specification

- Document ID: `LLRF-SPEC-001`
- Version: `1.0`
- Status: **Canonical / Current**
- Foundation version: `0.1.0`
- Contract version: `runtime-foundation.contract.v1`
- Development host: Windows
- Production validation host: Apple Silicon Mac

## 1. Purpose

The Foundation is the common low-level local execution authority shared by Novel Studio, Benchmark Studio, and Learning Studio. It answers **how a supplied local artifact was observed and executed**. It does not answer whether the artifact, settings, output, or deployment is good.

The Foundation is not a fourth Studio and is not a workbench.

## 2. Selected architecture

Foundation v1 uses three layers:

```text
Consumer Studio
    │
    └── Python client
            │ loopback HTTP / JSON / NDJSON
            ▼
    Local Runtime Service (one model-memory owner)
            │
            ▼
    Foundation Core
        ├── Host observation
        ├── Lifecycle + consumer leases
        └── Engine Adapter interface
                ├── Mock Adapter
                ├── MLX Adapter
                └── llama.cpp / GGUF skeleton
```

The selected boundary is `package + optional service facade`, with the service as the normal multi-consumer path. A shared Python package alone cannot safely own one model allocation across three independent processes. A distributed architecture is unnecessary for v1.

| Option | Decision | Reason |
| --- | --- | --- |
| Shared Python package only | Not the default | No single process owns model memory or unload conflicts across consumers. |
| Local HTTP service | Included | Gives process ownership, crash isolation, streaming, cancellation, and a stable wire version. |
| Package + optional service facade | **Selected** | Core is directly testable and embeddable; the service is the shared local authority. |

## 3. Foundation owns

- Engine identity, versions, build observations, capability discovery.
- Host observation: OS/platform/architecture/hardware/CPU/memory/Metal/MLX/memory pressure/swap.
- Consumer-supplied artifact binding validation and local path existence.
- Model load/unload, generation, streaming, cancellation, chat-template invocation, and engine-specific option mapping.
- Requested versus effective runtime settings.
- Raw runtime metrics and execution traces.
- Health and lifecycle state.

## 4. Foundation does not own

- Model Registry authority, artifact validation history, `UNVALIDATED`, production status, or promotion.
- Runtime Optimizer, candidate generation/scoring, successive halving, Safety Gate, Quality Benchmark, Blind Review, or Deployment eligibility.
- Novel production configuration, role assignment, publish/rollback, manuscript, or UI.
- Learning data/training/LoRA/QLoRA/preference learning/learned asset promotion.
- Model download, cloud inference, OS tuning, cluster scheduling, or remote execution.

## 5. Lifecycle and consumer ownership

The service holds at most one physical artifact/engine pair in v1 and one active generation request. A consumer receives a lease when it loads an artifact. Loading the same artifact from another consumer reuses the physical model and creates a second lease. A consumer cannot unload a lease owned by another consumer. A different artifact cannot replace a loaded artifact until all leases are released. Unload while a request is active returns a conflict.

This is a safety boundary, not a scheduling policy.

## 6. Policy-free rule

Foundation may report `memory_pressure=critical`, `swap_delta_bytes`, or an engine failure. It never turns those observations into `VALID`, `INVALID`, `production_eligible`, quality scores, or deployment decisions.

## 7. Windows status

Windows supports contract serialization, Core lifecycle, Mock execution, HTTP/client integration, error semantics, and deterministic tests. Real MLX/Metal/Unified Memory/swap/throughput remains pending on a real Apple Silicon Mac.
