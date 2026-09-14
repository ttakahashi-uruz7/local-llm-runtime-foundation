# Benchmark Studio PR #10 Migration Classification

- Status: **Canonical / Current Foundation migration map**
- Benchmark repository: `ttakahashi-uruz7/llm-benchmark-studio`
- Benchmark base used for this classification: `a18baef9e32644c813ea715855cfe103fd606f88`
- PR #10 HEAD: `bc81701aefaf04aba37d38d7272e255e10db8eed`
- PR #10 URL: <https://github.com/ttakahashi-uruz7/llm-benchmark-studio/pull/10>
- PR #10 handling: **open / not merged / not closed / not repurposed**

The local Benchmark checkout also contained a newer local `main` ref (`278a544`). It was not used for this classification because the user-specified PR base is `a18baef`. The table below covers all 18 files changed by PR #10 against that base.

## 1. Three-way classification

| Category | PR #10 changed files | Decision |
| --- | --- | --- |
| **A — Move to Foundation** | `app/runtime/adapters.py` (MLX/engine execution subset); `app/runtime/host.py` (host/Metal/MLX/swap observation subset) | Execution and observation move to Foundation-owned adapters/host modules. The old direct implementation is decommissioned from Benchmark after client integration. |
| **B — Keep in Benchmark** | `app/api/m2.py`; `app/api/m3.py`; `app/optimizer/candidates.py`; `app/optimizer/measurement.py`; `app/quality/scoring.py`; `app/quality/service.py`; `app/quality/session.py`; `app/registry/models.py`; `app/validation/__init__.py`; `app/validation/production_runtime.py`; `docs/current/implementation/FIRST_PRODUCTION_VALIDATION_PLAN.md`; `docs/current/implementation/PRODUCTION_VALIDATION_RUNBOOK.md`; `scripts/production_validation.py`; `tests/test_production_validation_preparation.py` | These files own Registry meaning, optimization, quality, production validation, operator flow, and Benchmark tests. Their runtime calls will be redirected to the Foundation client where required. |
| **C — Shared contract / integration boundary** | `app/api/m1.py` (mixed runtime/registry API); `app/runtime/errors.py` (runtime error subset plus Benchmark-specific errors); the unchanged reference `app/runtime/contracts.py` and `app/runtime/manager.py` are also boundary sources to replace, even though they are not changed in this PR diff | Foundation owns the versioned wire contract. Benchmark consumes it and maps raw results/errors into Benchmark evidence and policy. Mixed files are not copied wholesale. |

## 2. Responsibility-level reading of the mixed files

### `app/runtime/adapters.py`

- **MOVE:** MLX lazy import, `mlx_lm.load`, chat-template execution, streaming, cancellation signal, runtime option mapping, raw MLX metrics, unload cleanup.
- **KEEP/REPLACE:** Benchmark Mock measurement profiles or optimizer-specific simulation metadata must not become Foundation production metrics. Benchmark may keep its test fixture behavior behind the client.
- **DELETE after integration:** Benchmark's direct engine implementation, once the client path is proven.

### `app/runtime/host.py`

- **MOVE:** platform/architecture/OS/hardware/memory/Metal/MLX/swap observation.
- **DELETE after integration:** Benchmark's direct telemetry implementation.
- **KEEP in Benchmark:** only consumer-side interpretation and evidence/policy rules.

### `app/api/m1.py`

- **C/REPLACE:** runtime health/capability/load/unload/generate/stream/cancel endpoints become calls to the Foundation service/client.
- **KEEP:** any Benchmark Registry endpoints and Registry authority remain Benchmark-owned.
- **Do not copy:** SQLite Evidence, Registry scan, or Benchmark-specific routes into Foundation.

### `app/runtime/errors.py`

- **C:** stable wire error codes (`engine_unavailable`, `unsupported_runtime_option`, `cancelled`, lifecycle conflicts, etc.) are Foundation contracts.
- **KEEP:** optimizer and production-validation exceptions remain Benchmark-owned and are not exported by Foundation.

### `app/validation/production_runtime.py`

This is **Benchmark Authority**. It stays in Benchmark. Its direct `InferenceManager` calls are a planned **REPLACE** with `LocalRuntimeClient` calls. The decision to validate/promote a profile remains Benchmark policy.

## 3. KEEP / MOVE / REPLACE / DELETE plan

| PR #10 area | Action after Foundation integration | Owner after migration |
| --- | --- | --- |
| `app/runtime/adapters.py` MLX/engine execution | **MOVE**, then delete direct Benchmark implementation | Foundation Adapter |
| `app/runtime/host.py` host observation | **MOVE**, then delete direct Benchmark implementation | Foundation HostProfile |
| `app/runtime/contracts.py` request/result/settings semantics | **REPLACE** with `runtime-foundation.contract.v1` client/wire contracts | Foundation contract; Benchmark consumer mapping |
| `app/runtime/errors.py` runtime subset | **REPLACE** with Foundation wire errors; retain Benchmark policy exceptions | Foundation + Benchmark mapping |
| `app/runtime/manager.py` Registry/Evidence-coupled manager | **REPLACE** with Foundation client; no wholesale move | Foundation service/client |
| `app/api/m1.py` runtime facade | **REPLACE** runtime implementation; **KEEP** Benchmark-only Registry routes | Split boundary |
| `app/api/m2.py`, `app/api/m3.py` | **KEEP**, update runtime call sites if they bypass client | Benchmark |
| `app/validation/production_runtime.py` | **KEEP** authority; **REPLACE** direct runtime calls with client | Benchmark |
| Optimizer files | **KEEP** | Benchmark |
| Quality files | **KEEP** | Benchmark |
| Registry model | **KEEP** | Benchmark |
| Production validation docs/CLI/tests | **KEEP**, link to Foundation runbook and client contract | Benchmark |
| Direct old runtime portions after client migration | **DELETE** only after regression coverage and review | Benchmark decommission step |

No deletion or modification of Benchmark PR #10 is performed by this Foundation repository.

## 4. Required Benchmark follow-up

1. Add a Foundation client dependency and service lifecycle configuration.
2. Replace direct imports of MLX, Metal detection, HostProfile telemetry, engine load/unload, and raw generation.
3. Pass Registry-owned `ModelArtifactBinding` values to Foundation.
4. Store returned `ExecutionTrace`/raw metrics in Benchmark Evidence.
5. Keep Production Runtime Validation, Quality, Optimizer, Registry, Deployment, and Novel export decisions in Benchmark.
6. Add a compatibility fingerprint component for Foundation version/contract version.
7. Run Benchmark regression tests, then decommission only the proven direct runtime code.
