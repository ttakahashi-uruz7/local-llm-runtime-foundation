# RAH-1: Foundation Contract v2 / Execution Binding

RAH-1 adds `runtime-foundation.contract.v2` alongside the existing
`runtime-foundation.contract.v1`. v1 payloads and historical v1 evidence are
not rewritten or inferred-upgraded. Health discovery reports both supported
contract versions.

## Artifact Binding v2

Artifact identity is split into three independent parts:

- Registry identity: `artifact_id`.
- Content identity: `algorithm`, `digest`, `scope`, `scheme`, and the
  versioned `canonicalization_scheme`.
- Locator: `type` and `value` (for example, a filesystem path).

The locator is recorded in Artifact Binding and Trace evidence, but is excluded
from the certified Execution Binding fingerprint. A path-only relocation does
not change that fingerprint. `scheme=complete` requires a versioned
canonicalization scheme and a complete SHA-256 digest. The current schemes are
`complete-file-v1` and `complete-directory-manifest-v1`; the directory manifest
sorts UTF-8 relative paths and frames each path, entry type, content length, and
file digest. `scheme=fast` uses versioned `fast-file-v1` or
`fast-directory-manifest-v1` discovery identities and can never be promoted to
`complete`. RAH-1 does not implement automatic relocation or rebinding.

Historical v1 `artifact_hash` values, including 64-hex values, are preserved
as `scheme=legacy` and are never inferred-upgraded to production-complete
identity. The current execution boundary supports filesystem locators only;
non-filesystem locator types fail explicitly and remain available for future
extension.

## Engine and Foundation Binding

Engine Binding v2 separates:

- engine family (`mlx`);
- implementation (`mlx-lm`) and its version;
- build identity, which remains `null` when it cannot be observed; and
- adapter identity.

Foundation Binding includes contract version, Foundation package version,
optional build identity, and adapter identity. MLX family and `mlx-lm`
implementation are intentionally distinct. `mlx-lm` APIs do not leak through
the Studio-facing boundary.

## Execution Binding and settings fingerprint

Execution Binding combines Artifact, Engine, Foundation, and Runtime Settings
Binding. Its canonical artifact payload contains registry identity, content
identity, source revision, format, and quantization, but not the locator. It is
serialized with sorted object keys and compact JSON, then hashed with SHA-256
as `execution_binding_fingerprint`.

Runtime Settings Binding records the runtime-options schema version and the
fingerprint of the exact adapter-effective settings, not merely requested
values. Any artifact content, engine implementation/build, Foundation binding,
or effective settings change changes the execution binding fingerprint.

## Generation, Thinking, Trace, and Guard foundations

Generation Request v2 preserves v1 `thinking_enabled` and adds
`thinking_intent`: `OFF`, `AUTO`, or `ON`, with optional `LOW`, `MEDIUM`, or
`HIGH` effort and token budget. Studio/task policy is the authority for AUTO:
AUTO requires `studio_resolved_thinking` with effective `ON` or `OFF` before
Foundation proceeds. Foundation never resolves AUTO itself. Trace v2 keeps
requested, Studio-resolved, and Foundation/Adapter-effective thinking
distinct. If an adapter cannot represent an explicit effort or budget,
Foundation returns an explicit unsupported resolution rather than silently
downgrading it.

Execution Guard v1 carries:

- `expected_execution_binding_fingerprint`;
- `expected_runtime_settings_fingerprint`.

RAH-1 established the serialization boundary. RAH-2 activates the
pre-generation Safety / Compatibility Gate described in
[RAH2_SAFETY_COMPATIBILITY_GATE.md](RAH2_SAFETY_COMPATIBILITY_GATE.md).

## Policy boundary and deferred work

Foundation remains policy-free: it observes and executes supplied local
artifacts, resolves engine-neutral runtime settings, and records raw execution
evidence. It does not own Benchmark/Novel/Learning policy, quality judgments,
deployment eligibility, or task semantics.

This release does not implement a second MLX runtime, a real llama.cpp runtime,
LM Studio integration, generic artifact relocation, production deployment, or
production performance conclusions. Windows and Mock results are development
contract evidence, not production performance evidence. Real MLX production
validation still requires the Mac runbook.

Package version: `0.2.0`.
