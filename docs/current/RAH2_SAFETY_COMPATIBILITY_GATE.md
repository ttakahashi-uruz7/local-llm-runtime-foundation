# RAH-2: Foundation Safety / Compatibility Gate

RAH-2 activates the `ExecutionGuardV1` fields introduced by RAH-1. It remains
a mechanical Foundation verification step, not a deployment or quality
policy. A consumer may provide an expected certified execution state; the
Foundation resolves the actual state and executes only when the requested
expectations match.

## Pre-generation gate

For a guarded v2 request, the Core performs this sequence before invoking an
adapter's `generate` or `stream` method:

1. Authorize the loaded artifact and consumer lease.
2. Select the currently loaded adapter.
3. Resolve requested runtime options into the adapter's effective settings.
4. Build the actual v2 Execution Binding from the loaded artifact, selected
   engine, Foundation/adapter binding, and effective Runtime Settings Binding.
5. Verify the supplied Execution Guard.
6. Resolve Thinking intent and map it to the adapter.
7. Invoke the adapter.

The same internal resolution and guard path is used by `generate()` and
`stream()`. A stream does not emit `started` before the gate passes.

## Expected and actual fingerprints

`expected_execution_binding_fingerprint`, when supplied, is compared with the
actual `execution_binding_fingerprint`. The actual binding includes:

- complete Artifact Binding content identity;
- engine family, implementation id/version, build identity, and adapter id;
- Foundation contract/version, build identity, and adapter id; and
- the exact effective Runtime Settings Binding.

`expected_runtime_settings_fingerprint`, when supplied, is compared with
`RuntimeSettingsBindingV2.exact_settings_fingerprint` from the adapter's
effective settings. Matching requested values alone is insufficient: adapter
normalization or defaults are part of the certified effective state.

Expected values must use the canonical `sha256:<64 lowercase hex characters>`
format. A malformed value is an `invalid_request` with an
`INVALID_EXPECTATION` trace decision.

## Guard outcomes and safety

Guard evidence distinguishes:

- `MATCH`: all supplied expectations match and generation may proceed;
- `MISMATCH`: the actual state is resolved but differs, so generation is
  blocked with `execution_guard_mismatch`;
- `UNRESOLVABLE`: the actual certified state cannot be established, so
  generation is blocked with `execution_binding_unresolvable`; and
- `INVALID_EXPECTATION`: the request expectation is malformed.

Strict execution-binding verification requires `scheme=complete` artifact
identity and known engine and Foundation build identities. `scheme=fast`,
`scheme=legacy`, missing content identity, unknown build identity, and missing
binding components are never treated as certified matches. This is a
mechanical verification rule; Foundation does not decide deployment
eligibility or whether a consumer should use an artifact.

No token generation begins on a guard failure. Adapter generation methods are
not invoked. Errors include expected and actual fingerprints, mismatch
category, execution id, and `generation_started=false` without including model
data or secrets.

## Trace evidence

Trace v2 includes the actual Execution Binding, supplied guard, guard
verification status, expected and actual fingerprints, mismatch category,
structured error when applicable, and `generation_started`. Guard PASS evidence
is retained on successful generate and terminal stream results. Guard FAIL
evidence is retained even though no generation output exists.

## Compatibility and policy boundary

Ungarded v2 development requests continue to use the existing load, generate,
stream, cancel, and unload flows. v1 requests do not receive forced v2 Guard
enforcement, and historical v1 evidence is not rewritten or upgraded.

RAH-1 Thinking semantics remain unchanged: `OFF`, `AUTO`, and `ON`, optional
`LOW`/`MEDIUM`/`HIGH` effort, Studio-authoritative AUTO resolution, and
explicit failure for unsupported effort or budget mechanisms. Thinking failures
remain Thinking errors rather than Guard mismatches.

Foundation remains policy-free. Consumers own deployment, quality, and
production eligibility decisions; Foundation only verifies an explicitly
supplied expected state mechanically.
