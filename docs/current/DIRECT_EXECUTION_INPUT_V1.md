# Direct Execution Input v1

- Status: **Canonical / Current**
- Contract version: `runtime-foundation.contract.v2`
- Execution input schema: `runtime-foundation.execution-input.v1`
- Adapter lineage schema: `runtime-foundation.adapter-lineage.v1`

## Supported execution modes

Foundation accepts both of these load forms:

1. **Single Artifact**: the existing v1 `ModelArtifactBinding` or v2
   `ArtifactBindingV2`. This remains the path for Base-only models and existing
   materialized/fused artifacts.
2. **Base + Adapter Direct**: a versioned execution input with one Base and
   zero or one Adapter. MLX loads the Base and the Adapter directly without
   fusing or saving a new model artifact.

The single-artifact API and its saved bindings remain valid. Consumers do not
need to rewrite old requests or traces into an execution input.

## Execution input

`POST /models/load` accepts the following additive top-level property. Each
Base and Adapter reuses `ArtifactBindingV2` rather than defining a second
artifact identity schema.

```json
{
  "execution_input": {
    "contract_version": "runtime-foundation.contract.v2",
    "schema_version": "runtime-foundation.execution-input.v1",
    "kind": "base_plus_adapters",
    "base": {
      "contract_version": "runtime-foundation.contract.v2",
      "artifact_id": "consumer-base-id",
      "content_identity": {
        "algorithm": "sha256",
        "digest": "<64 lowercase hex characters>",
        "scope": "artifact",
        "scheme": "complete",
        "canonicalization_scheme": "complete-directory-manifest-v1"
      },
      "locator": {"type": "filesystem", "value": "/consumer/path/base"},
      "format": "mlx",
      "quantization": "mlx:bits=8;group_size=64;mode=affine",
      "revision": "consumer-base-revision",
      "metadata": {
        "model_identity": "consumer-model-identity",
        "tokenizer_identity": "consumer-tokenizer-identity",
        "target_module_shapes": {
          "self_attn.q_proj": {"in_features": 5120, "out_features": 12288}
        }
      }
    },
    "adapters": []
  }
}
```

For one Adapter, `adapters` contains its `ArtifactBindingV2`. The Base
metadata's `target_module_shapes` is required when at least one Adapter is
present. The Adapter binding metadata contains
`runtime_foundation_adapter_lineage` using
`runtime-foundation.adapter-lineage.v1`: the exact target Base artifact ID,
complete content identity, revision, format, quantization, model identity,
tokenizer identity, positive LoRA rank, target module names, and the `lora_a`
and `lora_b` tensor shapes. An optional target Base locator is provenance
only. `lora_a` uses `[in_features, rank]` and `lora_b` uses `[rank,
out_features]`.

An empty adapter array is valid and identifies a Base-only composition through
the new execution-input contract. One Adapter is valid. Two or more Adapters
are rejected with `unsupported_execution_input`; they are not silently
truncated or applied sequentially.

The consumer supplies artifact IDs, revision, content identity, model and
tokenizer identity, module shape manifest, adapter lineage, and locators.
Foundation does not own a Model Store, resolve consumer registry semantics, or
download artifacts. On direct load it verifies local Base and Adapter paths
and recomputes their complete content identities before engine load. Identity
verification reads the supplied artifact content; the MLX compatibility
preflight then inspects JSON and safetensors headers without loading tensors.

## Compatibility gate

Foundation rejects an input before engine model load when any of these checks
fail:

- complete SHA-256 content identity, Base/Adapter revision, Base format or
  quantization, or consumer-supplied model/tokenizer identity is missing;
- the Adapter's target Base identity differs from the requested Base;
- a target module is absent from the Base shape manifest;
- the declared LoRA rank or tensor shapes disagree with the Base dimensions;
- local content does not match the supplied complete content identity;
- MLX Base quantization config does not match the canonical binding value
  `mlx:bits=<bits>;group_size=<size>;mode=<mode>` (or
  `mlx:unquantized` for an unquantized Base);
- MLX adapter config or safetensors metadata disagrees with the declared
  lineage, or the adapter files are missing.

These failures use `artifact_incompatible` or `artifact_not_found`. The MLX
loader's permissive options are not treated as a compatibility guarantee.

## Identity and trace

`execution_input_fingerprint` uses canonical JSON and SHA-256. It includes the
execution-input schema and kind, Base identity and model/tokenizer/module
manifest, and the ordered adapter identities and lineage (role `lora_adapter`,
order `0`). It excludes all filesystem locators, including an optional target
Base locator in adapter lineage. Repointing an otherwise identical binding
does not change content identity. A Base, Adapter, lineage, engine/build,
Foundation/build, or effective Runtime Settings change changes the complete
Execution Binding fingerprint recorded in v2 trace.

The load response, loaded state, lease reuse decision, generation validation,
Execution Binding, and trace identify the full composition. Generation Request
v2 carries the same `execution_input`; Foundation compares its fingerprint to
the loaded composition before generation. Missing or different input returns
`execution_input_mismatch` before an engine generation call. A lease remains
owned by its consumer and unloading still releases the one loaded runtime.

To replace an Adapter, release the existing leases and unload first, then load
the Base with the new Adapter. Retaining a Base allocation while hot-swapping
Adapters is not guaranteed by this version.

## MLX execution

For an Adapter-bearing input, the MLX Engine Adapter validates local adapter
configuration, rank, target modules, Base safetensors dimensions, and Adapter
tensor shapes before calling `mlx_lm.load(base_path,
adapter_path=adapter_path)`. It does not call fuse, save, export, or
materialization APIs. With zero Adapters, the existing single-path MLX load is
used. Existing single-artifact Base and materialized/fused loads also retain
their prior single-path behavior.

## Ownership and non-goals

Foundation executes only consumer-supplied local artifacts. It does not own a
Model Store, download or move models, modify Learning Studio, or create
materialized artifacts. Learning Studio remains authoritative for training,
datasets, adapter promotion, and learned-model policy. Foundation records
execution mechanics and raw trace evidence only.
