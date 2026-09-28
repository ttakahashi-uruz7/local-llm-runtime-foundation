# Mac Production Validation Runbook

- Status: **Canonical / Current**
- Current state: **REAL MAC PRODUCTION VALIDATION COMPLETED**

This runbook is intentionally separate from Windows/Mock development. Passing the Windows test suite is not MLX/Metal production validation.

## Preconditions

1. Use a real Apple Silicon Mac.
2. Record macOS version/build, hardware model, architecture, physical/Unified Memory, MLX version, mlx-lm version, and Foundation version.
3. Install MLX/MLX-LM according to the consumer's controlled environment. Do not download model weights from Foundation validation code.
4. Use a consumer-supplied local model artifact binding with a verified hash/revision.
5. Start the Foundation service on loopback and confirm `GET /health`, `GET /host`, and `GET /engines/mlx/capability`.

## Required checks

- Host observation: `platform=darwin`, `architecture=arm64`, Metal and MLX availability, Unified Memory, memory pressure, swap.
- Engine identity/version/build discovery.
- Tokenizer chat-template rendering with the exact requested messages.
- `thinking_enabled=true` and `false` pass-through where the tokenizer supports the flag; explicit unsupported error otherwise.
- Load duration, load failure behavior, and loaded artifact identity.
- Cold and warm TTFT where prompt-cache semantics are actually implemented and observable.
- Prefill tokens/duration/throughput and generation tokens/duration/throughput.
- Context lengths required by the consumer; record context failure separately from policy.
- Process footprint, peak MLX-reported memory, memory pressure, swap before/after/delta.
- Streaming event order, cancellation, timeout behavior, and cleanup.
- Unload and `mlx.core.clear_cache` behavior; capture cleanup status and any residual footprint.
- Service restart and stale lifecycle recovery.

## Evidence

Store the raw Foundation `HostProfile`, `EngineCapability`, `LoadResult`, `GenerationResult`/`StreamEvent` sequence, `ExecutionTrace`, and error payload in the consumer's evidence store. Foundation does not create `Production Runtime Profile` or `Deployment Eligibility`.

## Validation record: 2026-09-22

The Foundation real execution path was validated on an Apple Silicon Mac. This record describes Foundation observations only; it does not approve the artifact or create a Benchmark Production Eligibility result.

- Host: Mac Studio `Mac17,14`, Apple M5 Max, 64 GiB Unified Memory, macOS `27.0` (build `26A428`), `arm64`.
- Engine: Metal GPU execution PASS; `mlx 0.32.2`, `mlx-lm 0.31.3`.
- Artifact: Qwen3.8-27B MLX 8-bit text-only; provider `Hugging Face / lukaskremla` with the source URI `https://huggingface.co/lukaskremla/Qwen3.8-27B-8bit-MLX-TextOnly`, revision `c8fb201897784269fc6433f0dafd0528e7275b3a`.
- Foundation complete Content Identity: SHA-256 `2e66eda92f10f7041bb1b43e62983384c0dee0ac8895fb2b9f6bf0d060932e32`.
- PASS: load, generation, streaming, cancellation, unload, reload, `mlx.core.clear_cache` cleanup, tokenizer chat template, thinking `true`/`false`, context enforcement, quantized KV (`int8`, 8-bit, group size 64), restart/stale lifecycle recovery, repeated execution, traces, and real-model error semantics.
- Observations: memory pressure `normal`; swap delta `0`; representative cold TTFT approximately 2.6–3.7 seconds and warm TTFT approximately 0.35 seconds; observed generation throughput approximately 18–25 tokens/second.
- Prompt cache: explicitly unsupported in Foundation v1 because prompt-cache object lineage is not implemented; no silent fallback or false PASS is reported.
- The stale active-request state after a preflight failure was fixed and regression-tested in PR #10.

The Qwen artifact is a consumer-supplied validation input. Foundation does not own Model Registry authority, Benchmark scoring, Production Eligibility, deployment status, or model approval.

## Direct Base + Adapter API gate: 2026-09-28

This focused integration gate exercised the newly versioned execution input
through `LocalRuntimeClient` → the Foundation loopback HTTP API → Core → MLX
Adapter → generation → v2 Execution Binding/Trace → unload. The service API
was hosted in-process by FastAPI's test transport on the same Apple Silicon
Mac; model loading and inference used the installed MLX/Metal runtime. This
records direct execution mechanics and does not replace or extend a consumer's
Production Eligibility decision.

- Host/engine: Mac Studio `Mac17,14`, `arm64`, macOS `27.0` build `26A428`,
  64 GiB Unified Memory; `mlx 0.32.2`, `mlx-lm 0.31.3`, Metal GPU.
- Base: consumer-supplied Qwen3.8-27B 8-bit MLX artifact, revision
  `c8fb201897784269fc6433f0dafd0528e7275b3a`, complete SHA-256
  `2e66eda92f10f7041bb1b43e62983384c0dee0ac8895fb2b9f6bf0d060932e32`.
- Adapter: the Learning Studio Mac Gate test Adapter, rank 8, targets
  `self_attn.q_proj` and `self_attn.v_proj`, complete SHA-256
  `bb26d0445be4c1600fb957d73ba26b6607e57d76db52047afcfae5b8dfebf299`.
- PASS: complete Base and Adapter content identity, target Base revision and
  identity, physical Base quantization config (`8` bits, group size `64`,
  affine mode), config rank/modules, and actual Base/Adapter safetensors shapes
  validated before MLX Direct Load. MLX received the Base path and
  `adapter_path`; generation returned “4. This is the sum of two and two.”
- `execution_input_fingerprint`:
  `sha256:d92d2f8d2cf519f1a4692118f5f61ab4e9798b954b09b49c567e918e6e2a0c1f`.
  The successful trace recorded Execution Binding fingerprint
  `sha256:a048dd5a361105bb498e49172b835451bd7d0f55d2d7d03714cc7bb016555b80`.
- Runtime observations: load `3401 ms`; cold TTFT `2437 ms`; prefill 41
  tokens at `41.40 tokens/s`; generation 12 tokens at `20.36 tokens/s`;
  MLX-reported peak memory `28,865,260,734 bytes`; memory pressure `normal`;
  swap was `687,467,397 bytes` before and after (`0` delta).
- PASS: unload returned `cleanup_status=clean`. Complete Base and Adapter
  identities were unchanged after execution. The Foundation repository had
  zero files at least 1 GB before and after; direct fused/materialized artifact
  count was `0`. No Production access occurred.
- A first API-path attempt exposed MLX's per-thread stream affinity when load
  and generation ran on different service worker threads. The MLX Adapter now
  runs load, stream generator steps, generator close, and cache cleanup on one
  dedicated thread; this focused Mac gate passed after that fix.

This run did not measure warm TTFT, cancellation, restart recovery, or a
consumer's full production workload for the direct Adapter composition.

## Stop conditions

If MLX is unavailable, Metal allocation fails, a context run fails, memory pressure/swap is observed, or cleanup is incomplete, record the raw observation and failure. Do not convert it inside Foundation into a Benchmark verdict. The consumer applies its own current policy.
