# Mac Production Validation Runbook

- Status: **Canonical / Current**
- Current state: **REAL MAC PRODUCTION VALIDATION PENDING**

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

## Stop conditions

If MLX is unavailable, Metal allocation fails, a context run fails, memory pressure/swap is observed, or cleanup is incomplete, record the raw observation and failure. Do not convert it inside Foundation into a Benchmark verdict. The consumer applies its own current policy.
