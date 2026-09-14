# AGENTS.md

## Repository purpose

This repository is the independent Local LLM Runtime Foundation shared by Novel Studio, Benchmark Studio, and Learning Studio. It is infrastructure, not a fourth Studio and not a user-facing workbench.

Canonical current specifications live under `docs/current/`. Historical Benchmark documents and PR #10 code are references only; they do not override the Foundation contracts.

## Authority boundary

Foundation owns:

- Engine Adapter interface and capability discovery.
- MLX / `mlx-lm` execution, Mock execution, and the future llama.cpp boundary.
- Host observation: platform, architecture, OS, hardware, memory, Metal/MLX availability, memory pressure, and swap observations.
- Artifact binding validation, model load/unload, generation, streaming, cancellation, lifecycle, health, requested/effective settings, raw metrics, and execution traces.
- The versioned local service and Python client boundary.

Foundation does not own:

- Benchmark Model Registry meaning/history/status, Runtime Optimizer, Quick/Full Optimize, scoring, Quality Benchmark, Blind Review, Safety Gate, Production Runtime Validation, Deployment, or Novel Export.
- Novel Production Config, role assignment, production switch, rollback, manuscript, or UI.
- Learning data, training, SFT, LoRA/QLoRA, preference learning, learned asset promotion, or learned model profiles.

Consumers decide whether Foundation observations satisfy their own policies.

## Safety and data rules

- Do not commit model weights, caches, huge artifacts, secrets, or user-specific absolute model paths.
- Do not make automatic model download part of validation. Model acquisition is a separate workflow.
- Do not add cloud/API inference or automatic cloud fallback.
- Do not write Novel Production settings, Benchmark scores, Benchmark Registry status, Deployment status, or Learning authority from this repository.
- Do not hard-code Qwen or any other model family. A Qwen integration is only an integration test target.
- Do not represent `runtime_mode=production-mac`, `production_eligible`, pass/fail thresholds, or quality judgments in Foundation contracts.

## Windows and Mac

Windows is the development and Mock/HTTP contract environment. Real MLX, Metal, Apple Unified Memory, swap behavior, memory pressure, and throughput require an Apple Silicon Mac. Until that validation is complete, report **REAL MAC PRODUCTION VALIDATION PENDING**.

The MLX adapter is lazy and must remain import-safe on Windows. It must not turn the absence of MLX into a cloud fallback or a fake production result.

## Git

Use Sourcetree Embedded Git only:

`C:\Users\user\AppData\Local\Atlassian\Sourcetree\git_local\cmd\git.exe`

Never switch silently to PATH/system Git. Do not use force push, reset --hard, git clean, history rewrite, or destructive changes to sibling repositories. Do not commit directly to `main`; main merge always requires explicit user approval.

The Benchmark reference PR #10 remains open. This repository must not close, merge, or repurpose that PR.

## Codex protocol

All multi-step work is reported as `Step X/N`. If scope expands, update N. Investigate repository state and current specifications before asking routine design questions. Fix ordinary test failures, type errors, missing fixtures, and small contract decisions autonomously. Stop only for repository identity/permission/corruption, inseparable user changes, an unresolvable canonical contradiction, credential failure that prevents the requested Git operation, or a destructive-only solution.

Before commit:

1. Inspect diff, staged paths, secrets, and existing changes.
2. Run focused/full tests and compile/type checks available in the repository.
3. Run `git diff --check`.
4. Record Base SHA, final SHA, remote SHA, and clean status.

## Mac validation gate

Before calling the Foundation MLX path production validated, run the current Mac runbook, record engine versions/builds, artifact binding, chat-template behavior, load/unload cleanup, cold/warm TTFT, prefill/generation metrics, process footprint, peak memory, memory pressure, swap, cancellation, and failure evidence. No Windows/Mock result can substitute for those observations.
