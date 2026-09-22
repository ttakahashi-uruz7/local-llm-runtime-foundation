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

## Stop conditions

If MLX is unavailable, Metal allocation fails, a context run fails, memory pressure/swap is observed, or cleanup is incomplete, record the raw observation and failure. Do not convert it inside Foundation into a Benchmark verdict. The consumer applies its own current policy.
